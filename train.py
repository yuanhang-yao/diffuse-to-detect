import os
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import shutil
import math

from args import parse_args
from models.alcl_model import ALCLNet
from models.dna_net import DNANet, Res_CBAM_block
from utils.dataset import create_dataset
from utils.metrics import SigmoidMetric, SamplewiseSigmoidMetric, PD_FA_2
from utils.diffusion_annotation import Diffusion_Annotation, post_process


MODEL_LABELS = {
    "alclnet": "ALCLNet",
    "dnanet": "DNANet",
}


def create_model(model_name):
    if model_name == "alclnet":
        return ALCLNet()
    if model_name == "dnanet":
        return DNANet(
            num_classes=1,
            input_channels=3,
            block=Res_CBAM_block,
            num_blocks=[2, 2, 2, 2],
            nb_filter=[16, 32, 64, 128, 256],
            deep_supervision=True,
        )
    raise ValueError(f"Unsupported model: {model_name}")


class WeighterMLP(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class TrainingPipeline:
    def __init__(self, args, model, train_loader, val_loader, test_loader, pde_run_tag):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.test_start_epoch = args.test_start_epoch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for training.")
        self.device = torch.device(f"cuda:{int(str(args.gpu_id).split(',')[0])}")
        torch.cuda.set_device(self.device)
        self.pde_run_tag = pde_run_tag

        self.model = self.model.to(self.device)

        self.dataset_dir = self.train_loader.dataset.root_dir
        self.pde_root_dir = os.path.join(self.dataset_dir, 'single_point', self.pde_run_tag)
        for split in ['train', 'val']:
            split_dir = os.path.join(self.pde_root_dir, split)
            if os.path.exists(split_dir):
                shutil.rmtree(split_dir)
            os.makedirs(split_dir, exist_ok=True)

        self.theta_D0_u = torch.nn.Parameter(torch.zeros([], device=self.device))
        self.theta_beta_u = torch.nn.Parameter(torch.zeros([], device=self.device))
        self.theta_rho_u = torch.nn.Parameter(torch.zeros([], device=self.device))
        self.theta_alpha_u = torch.nn.Parameter(torch.zeros([], device=self.device))
        self.pde_optimizer = torch.optim.AdamW(
            [self.theta_D0_u, self.theta_beta_u, self.theta_rho_u, self.theta_alpha_u],
            lr=1e-2,
            weight_decay=0.0,
        )

        self._dump_split_pde_masks_to_disk(
            self.train_loader.dataset.img_ids,
            os.path.join(self.pde_root_dir, 'train'),
        )
        self._dump_split_pde_masks_to_disk(
            self.val_loader.dataset.img_ids,
            os.path.join(self.pde_root_dir, 'val'),
        )

        self.weight_mlp = WeighterMLP(in_dim=16, hidden_dim=32).to(self.device)
        self.weight_mlp_opt = torch.optim.AdamW(self.weight_mlp.parameters(), lr=1e-3, weight_decay=0.0)
        self._ema_loss = {}
        self.prior_difficulty = {}
        with open(os.path.join(self.dataset_dir, "mode", "prior_difficulty.txt"), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.prior_difficulty[parts[0]] = float(parts[1])
        prior_vals = torch.tensor(list(self.prior_difficulty.values()), dtype=torch.float32)
        self.prior_mu = float(prior_vals.mean().item())
        self.prior_std = float(prior_vals.std(unbiased=False).clamp_min(1e-6).item())
        self._bce_logits = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)

    def _pde_surrogate_from_pred(self, probs_detached):
        P = probs_detached
        D0, beta, rho, alpha = self._current_pde_phys()
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],
                               device=P.device, dtype=P.dtype).view(1,1,3,3)
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],
                               device=P.device, dtype=P.dtype).view(1,1,3,3)
        gx = F.conv2d(P, sobel_x, padding=1)
        gy = F.conv2d(P, sobel_y, padding=1)
        grad_mag = torch.sqrt(gx*gx + gy*gy + 1e-12)

        win = 7
        pad = win // 2
        P_mean = F.avg_pool2d(P, win, stride=1, padding=pad)
        P_mean = P_mean.clamp(1e-4, 1-1e-4)
        entropy_local = -(P_mean * P_mean.log() + (1-P_mean)*(1-P_mean).log())
        entropy_local = entropy_local / math.log(2.0)

        complexity = alpha * grad_mag + (1 - alpha) * entropy_local
        with torch.no_grad():
            c_mu = complexity.mean(dim=(1,2,3), keepdim=True)
            c_std= complexity.std(dim=(1,2,3), keepdim=True).clamp_min(1e-4)
        complexity_n = (complexity - c_mu) / c_std
        diff_coef = D0 * torch.exp(-beta * complexity_n).clamp(min=1e-4)

        lap_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],
                                  device=P.device, dtype=P.dtype).view(1,1,3,3)
        U = P
        dt = 0.15
        for _ in range(3):
            lap = F.conv2d(U, lap_kernel, padding=1)
            U = U + dt * diff_coef * lap
        U = torch.clamp(U, 0.0, 1.0)

        Bsz, _, H, W = P.shape
        flat = P.view(Bsz, -1)
        k = max(int(0.005 * flat.shape[1]), 1)
        topk_vals, _ = torch.topk(flat, k, dim=1)
        kth = topk_vals[:, -1].view(Bsz,1,1,1)
        source_map_raw = torch.relu(P - kth)
        source_map = source_map_raw / (source_map_raw.amax(dim=(1,2,3), keepdim=True) + 1e-6)
        k3 = torch.ones(1,1,3,3, device=P.device, dtype=P.dtype) / 9.0
        source_map = F.conv2d(source_map, k3, padding=1).clamp(0,1)

        fused = rho * U + (1 - rho) * source_map
        edge_penalty = 0.15 * grad_mag
        logits_like = fused - edge_penalty
        return torch.sigmoid(logits_like)

    def _dump_split_pde_masks_to_disk(self, img_ids, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.model.eval()
        for img_id_str in img_ids:
            mask_np = self._generate_single_pde_mask_numpy(img_id_str)
            out_path = os.path.join(out_dir, f'{img_id_str}.png')
            cv2.imwrite(out_path, mask_np)
        self.model.train()

    def _current_pde_phys(self):
        D0 = 3.2 * (1.0 + 0.3 * torch.tanh(self.theta_D0_u))
        beta = 6.0 * (1.0 + 0.3 * torch.tanh(self.theta_beta_u))
        rho = 0.84 + 0.05 * torch.tanh(self.theta_rho_u)
        alpha = 0.6 + 0.05 * torch.tanh(self.theta_alpha_u)
        return D0, beta, rho, alpha

    def _generate_single_pde_mask_numpy(self, img_id_str):

        img_path = os.path.join(self.dataset_dir, 'images', f'{img_id_str}.png')
        info_path = os.path.join(self.dataset_dir, 'image_info_centroid', f'{img_id_str}.npy')

        img = np.array(Image.open(img_path).convert('RGB'))
        mask_info = np.load(info_path, allow_pickle=True)
        with torch.no_grad():
            D0_phys, beta_phys, rho_phys, alpha_phys = self._current_pde_phys()

        base_mask = Diffusion_Annotation(
            img=img, mask_info=mask_info,
            D0=float(D0_phys.item()),
            beta=float(beta_phys.item()),
            rho=float(rho_phys.item()),
            alpha=float(alpha_phys.item()),
            dt=0.2, id_num=img_id_str,
            point_steps=3,
            spot_particles=260,
            spot_steps=40,
            point_dt_scale=0.9,
            point_center_prior_w=0.25,
            point_center_prior_sigma=0.25,
            source_weighted=0,
        )
        return post_process(
            mask_info, img, base_mask, img_id_str,
            crf_iter=5,
            crf_sxy=8,
            crf_srgb=10,
            crf_compat=110,
        ).astype(np.uint8)

    def _batch_pde_masks_from_disk(self, img_ids, split):
        base_dir = os.path.join(self.pde_root_dir, split)
        masks = []
        for img_id_str in img_ids:
            path = os.path.join(base_dir, f'{img_id_str}.png')
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (self.args.crop_size, self.args.crop_size), interpolation=cv2.INTER_NEAREST)

            masks.append(m)
        masks_np = np.stack(masks, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(masks_np).to(device=self.device, dtype=torch.float32).unsqueeze(1)

    def _update_ema_history(self, img_ids, loss_per_detached):
        for i, iid in enumerate(img_ids):
            l = float(loss_per_detached[i].item())
            if iid not in self._ema_loss:
                self._ema_loss[iid] = l
            else:
                self._ema_loss[iid] = 0.9 * self._ema_loss[iid] + 0.1 * l

    def _build_weight_features16(self, logits_t, probs_t, Mt, img_ids,
                                 logits_v=None, Mv=None):
        with torch.no_grad():
            P = probs_t.detach()
            Z = logits_t.detach()
            M_t = Mt.detach().clamp(0, 1)

            Bsz = P.shape[0]
            eps = 1e-6
            H, W = P.shape[-2], P.shape[-1]
            Npix = H * W
            m_prob = P.mean(dim=(1,2,3))
            m_logit = Z.abs().mean(dim=(1,2,3))
            P_clamp = P.clamp(1e-6, 1 - 1e-6)
            entropy = (-(P_clamp * P_clamp.log() + (1 - P_clamp) * (1 - P_clamp).log())).mean(dim=(1,2,3))

            R_t = (P - M_t)
            L1 = R_t.abs().mean(dim=(1,2,3))
            L2 = (R_t.pow(2)).mean(dim=(1,2,3))

            k = max(1, int(0.1 * Npix))
            absR = R_t.abs().reshape(Bsz, -1)
            topk_vals, _ = torch.topk(absR, k, dim=1)
            topk_L1 = topk_vals.mean(dim=1)

            if logits_v is not None and Mv is not None:
                Z_v = logits_v.detach()
                P_v = torch.sigmoid(Z_v).detach()
                M_v = Mv.detach().clamp(0, 1)
                R_v = (P_v - M_v)
                R_v_mean = R_v.mean(dim=0, keepdim=True)
                cc_dot = (R_t * R_v_mean).mean(dim=(1,2,3))
                a = R_t.reshape(Bsz, -1)
                b_ref = R_v_mean.reshape(1, -1)
                ab = (a * b_ref).sum(dim=1)
                na = (a.pow(2).sum(dim=1) + eps).sqrt()
                nb = (b_ref.pow(2).sum(dim=1) + eps).sqrt()
                cc_cos = ab / (na * nb + eps)
                sign_frac = (torch.sign(R_t).eq(torch.sign(R_v_mean))).float().mean(dim=(1,2,3))
            else:
                cc_dot = torch.zeros(Bsz, device=self.device)
                cc_cos = torch.zeros(Bsz, device=self.device)
                sign_frac = torch.zeros(Bsz, device=self.device)

            P_bin = (P > 0.5).float()
            Mt_bin = (M_t > 0.5).float()
            dil = F.max_pool2d(Mt_bin, kernel_size=3, stride=1, padding=1)
            ero = 1.0 - F.max_pool2d(1.0 - Mt_bin, kernel_size=3, stride=1, padding=1)
            band = (dil - ero).clamp(min=0.0)
            band = (band > 0).float()
            band_L1 = (R_t.abs() * band).sum(dim=(1,2,3)) / (band.sum(dim=(1,2,3)) + eps)
            dice_pm = 2.0 * (P_bin * Mt_bin).sum(dim=(1,2,3)) / (
                P_bin.sum(dim=(1,2,3)) + Mt_bin.sum(dim=(1,2,3)) + eps
            )
            fg_ratio_bin = P_bin.mean(dim=(1,2,3))

            M_phi = self._pde_surrogate_from_pred(P).detach()
            pde_L1 = (M_phi - M_t).abs().mean(dim=(1,2,3))
            Mphi_bin = (M_phi > 0.5).float()
            dice_pde = 2.0 * (Mphi_bin * Mt_bin).sum(dim=(1,2,3)) / (
                Mphi_bin.sum(dim=(1,2,3)) + Mt_bin.sum(dim=(1,2,3)) + eps
            )
            ema_loss_vec = torch.tensor(
                [self._ema_loss.get(iid, 0.0) for iid in img_ids],
                dtype=torch.float32, device=self.device
            )
            prior_z = torch.tensor(
                [(self.prior_difficulty.get(iid, 0.0) - self.prior_mu) / (self.prior_std + 1e-6) for iid in img_ids],
                dtype=torch.float32, device=self.device
            )
            return torch.stack([
                m_prob, m_logit, entropy, L1, L2, topk_L1,
                cc_dot, cc_cos, sign_frac,
                band_L1, dice_pm, fg_ratio_bin,
                pde_L1, dice_pde,
                ema_loss_vec, prior_z,
            ], dim=1)

    def _outer_update_weight_mlp(self, Xt, Yt, idt, Xv, Yv):
        Xt = Xt.to(self.device)
        Xv = Xv.to(self.device)
        Yt = Yt.to(self.device)
        Yv = Yv.to(self.device)
        self.weight_mlp.train()

        with torch.no_grad():
            logits_t = self.model(Xt, uni_output=False)
            logits_t_tensor = logits_t[0] if isinstance(logits_t, (list, tuple)) else logits_t
            probs_t  = torch.sigmoid(logits_t_tensor)

            logits_v = self.model(Xv, uni_output=False)
            logits_v_tensor = logits_v[0] if isinstance(logits_v, (list, tuple)) else logits_v
            Lval = self._calculate_loss(logits_v, Yv, reduction='mean')
            s_i = torch.relu(
                ((torch.sigmoid(logits_t_tensor) - Yt) * (torch.sigmoid(logits_v_tensor) - Yv).mean(dim=0, keepdim=True))
                .mean(dim=(1,2,3))
            )
        s_i = (s_i - s_i.mean()) / (s_i.std() + 1e-6)

        with torch.no_grad():
            h_t = self._build_weight_features16(
                logits_t_tensor, probs_t, Yt, idt,
                logits_v=logits_v_tensor, Mv=Yv
            )
            mu, std = h_t.mean(0, keepdim=True), h_t.std(0, keepdim=True).clamp_min(1e-6)
            h_t_n = (h_t - mu) / std

        logits_alpha = self.weight_mlp(h_t_n).squeeze(-1)
        alpha_raw = torch.sigmoid(logits_alpha / 0.25)
        alpha = alpha_raw / (alpha_raw.detach().mean() + 1e-6)

        self.weight_mlp_opt.zero_grad()
        f_psi = -0.25 * torch.mean(alpha * s_i)
        f_psi.backward()
        self.weight_mlp_opt.step()

        return float(Lval.detach().cpu().item())


    def _outer_update_pde(self, Xv, val_masks):
        device = self.device
        Xv = Xv.to(device)
        Yv = val_masks.to(device)

        self.model.eval()
        with torch.no_grad():
            logits_v = self.model(Xv, uni_output=False)
            probs_v = torch.sigmoid(logits_v[0] if isinstance(logits_v, (list, tuple)) else logits_v)
        self.model.train()

        with torch.no_grad():
            Pv_det = probs_v.detach().clamp(1e-4, 1-1e-4)
            Yv_det = Yv.detach().clamp(0,1)
        Mv_phi = self._pde_surrogate_from_pred(Pv_det)
        f_phi = F.binary_cross_entropy(Mv_phi.clamp(1e-4,1-1e-4), Yv_det)
        f_phi = f_phi + 0.5 * F.binary_cross_entropy(Mv_phi.clamp(1e-4,1-1e-4), Pv_det)

        self.pde_optimizer.zero_grad(set_to_none=True)
        f_phi.backward()
        for g in [self.theta_D0_u, self.theta_beta_u, self.theta_rho_u, self.theta_alpha_u]:
            if g.grad is not None:
                g.grad.data = torch.nan_to_num(g.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
        self.pde_optimizer.step()


    def bilevel_training(self, epoch):
        self.model.train()
        losses = []
        val_losses = []
        max_steps = getattr(self.args, "max_train_steps", None)
        val_iterator = iter(self.val_loader)
        for i, (img, mask, img_id, h, w) in enumerate(self.train_loader):
            if max_steps is not None and i >= max_steps:
                break
            img = img.to(self.device)
            mask = mask.to(self.device)

            logits = self.model(img, uni_output=False)
            logits_tensor = logits[0] if isinstance(logits, (list, tuple)) else logits
            probs = torch.sigmoid(logits_tensor)

            with torch.no_grad():
                ht = self._build_weight_features16(
                    logits_tensor, probs, mask, img_id,
                    logits_v=None, Mv=None
                )
                mu = ht.mean(dim=0, keepdim=True)
                std = ht.std(dim=0, keepdim=True).clamp_min(1e-6)
                ht = (ht - mu) / std

                if epoch >= 80:
                    alpha = self.weight_mlp(ht).squeeze(-1)
                    alpha = torch.sigmoid(alpha / 0.25)
                    alpha = alpha / (alpha.detach().mean() + 1e-6)
                else:
                    alpha = torch.ones(ht.size(0), device=ht.device)

            loss_per = self._calculate_loss(logits, mask, reduction='none')
            Ltr = torch.mean(alpha.detach() * loss_per)

            try:
                val_img, val_mask, val_img_id, _, _ = next(val_iterator)
            except StopIteration:
                val_iterator = iter(self.val_loader)
                val_img, val_mask, val_img_id, _, _ = next(val_iterator)
            val_img = val_img.to(self.device)
            val_mask = val_mask.to(self.device)

            with torch.no_grad():
                Mv_cons = self._batch_pde_masks_from_disk(val_img_id, split='val')

            if epoch >= 80 and ((i + 1) % 4 == 0):
                logits_v = self.model(val_img, uni_output=False)
                Lval = self._calculate_loss(logits_v, val_mask, reduction='mean')
                Lcons = self._bce_logits(logits_v[0] if isinstance(logits_v, (list, tuple)) else logits_v, Mv_cons)
                y_loss = Ltr + (1.0 / 1.2) * (Lval + 0.1 * Lcons)
            else:
                y_loss = Ltr

            self.optimizer.zero_grad(set_to_none=True)
            y_loss.backward()
            self.optimizer.step()

            self._update_ema_history(img_id, loss_per.detach())
            losses.append(y_loss.item())

            if epoch >= 80 and ((i + 1) % 4 == 0):
                val_losses.append(self._outer_update_weight_mlp(img, mask, img_id, val_img, val_mask))

        if epoch >= 80 and epoch % 20 == 0:
            try:
                val_img, val_mask, val_img_id, _, _ = next(val_iterator)
            except StopIteration:
                val_iterator = iter(self.val_loader)
                val_img, val_mask, val_img_id, _, _ = next(val_iterator)
            val_img = val_img.to(self.device)
            val_mask = val_mask.to(self.device)

            self._outer_update_pde(val_img, val_mask)
            self._dump_split_pde_masks_to_disk(
                self.train_loader.dataset.img_ids,
                os.path.join(self.pde_root_dir, 'train'),
            )
            self._dump_split_pde_masks_to_disk(
                self.val_loader.dataset.img_ids,
                os.path.join(self.pde_root_dir, 'val'),
            )

            self.train_loader = create_dataset(self.args, split='train', pde_run_tag=self.pde_run_tag)
            self.val_loader = create_dataset(self.args, split='val', pde_run_tag=self.pde_run_tag)

        return np.mean(losses), np.mean(val_losses) if len(val_losses) > 0 else None


    def _calculate_loss(self, outputs, targets, reduction='mean'):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        logits_list = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
        loss_list = []
        for logits in logits_list:
            if logits.dim() == 3:
                logits = logits.unsqueeze(1)
            pred = torch.sigmoid(logits)
            inter = torch.sum(pred * targets, dim=(1,2,3))
            pred_sum = torch.sum(pred, dim=(1,2,3))
            target_sum = torch.sum(targets, dim=(1,2,3))
            soft_iou_per = 1 - (inter + 1.0) / (pred_sum + target_sum - inter + 1.0)

            if reduction == 'none':
                loss_list.append(soft_iou_per)
            elif reduction == 'mean':
                loss_list.append(soft_iou_per.mean())
            elif reduction == 'sum':
                loss_list.append(soft_iou_per.sum())
            else:
                raise ValueError("reduction must be 'mean', 'none' or 'sum'")

        return torch.stack(loss_list, dim=0).mean(dim=0)

    def testing(self, epoch):
        if epoch < self.test_start_epoch:
            return {'IoU': 0, 'nIoU': 0, 'PD': 0, 'FA': 0}

        self.model.eval()
        metric_iou = SigmoidMetric(score_thresh=0.5)
        metric_niou = SamplewiseSigmoidMetric(nclass=1, score_thresh=0.5)
        metric_pd_fa = PD_FA_2(nclass=1)

        with torch.no_grad():
            for img, mask, _, _, _ in self.test_loader:
                img = img.to(self.device)
                mask = mask.to(self.device)
                outputs = torch.sigmoid(self.patches_test(img))
                for b_idx in range(outputs.shape[0]):
                    metric_iou.update(outputs[b_idx:b_idx+1], mask[b_idx:b_idx+1])
                    metric_niou.update(outputs[b_idx:b_idx+1], mask[b_idx:b_idx+1])
                    metric_pd_fa.update(outputs[b_idx:b_idx+1], mask[b_idx:b_idx+1])

        _, IoU = metric_iou.get()
        _, nIoU = metric_niou.get()
        FA, PD = metric_pd_fa.get()
        return {'IoU': IoU, 'nIoU': nIoU, 'PD': PD, 'FA': FA}

    def patches_test(self, img):
        b, c, h, w = img.shape

        batch_size = self.args.batch_size
        patch_size = self.args.crop_size
        stride = self.args.crop_size

        if h > patch_size and w > patch_size:
            img_unfold = F.unfold(img, kernel_size=patch_size, stride=stride)
            img_unfold = img_unfold.reshape(b, c, patch_size, patch_size, -1).permute(0, 4, 1, 2, 3)
            patch_num = img_unfold.size(1)
            preds_list = []
            for i in range(0, patch_num, batch_size):
                end = min(i + batch_size, patch_num)
                batch_patches = img_unfold[:, i:end, :, :, :].reshape(-1, c, patch_size, patch_size)
                preds_list.append(self.model(batch_patches.float()))

            preds_unfold = torch.cat(preds_list, dim=0).permute(1, 2, 3, 0)
            preds_unfold = preds_unfold.reshape(b, -1, patch_num)
            preds = F.fold(preds_unfold, kernel_size=patch_size, stride=stride, output_size=(h, w))
        else:
            preds = self.model(img)

        return preds

def train():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_label = MODEL_LABELS[args.model]
    pde_run_tag = f"{date_str}_{args.model}_seed{args.seed}"

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        model_save_dir = os.path.join(args.save_dir, f"{model_label}_SIRST3")
        os.makedirs(model_save_dir, exist_ok=True)

    model = create_model(args.model)

    best_metrics = {'IoU': 0, 'nIoU': 0, 'PD': 0, 'FA': 0}
    best_iou = 0

    train_loader = create_dataset(args, split='train', pde_run_tag=pde_run_tag)
    val_loader = create_dataset(args, split='val', pde_run_tag=pde_run_tag)
    test_loader = create_dataset(args, split='test', pde_run_tag=pde_run_tag)
    pipeline = TrainingPipeline(args, model, train_loader, val_loader, test_loader, pde_run_tag=pde_run_tag)

    print(f"\n{'='*60}")
    print(f"Training {model_label} on SIRST3 dataset")

    for epoch in range(args.epochs):
        print("=" * 60)

        train_loss, val_loss = pipeline.bilevel_training(epoch)

        test_metrics = pipeline.testing(epoch)
        current_iou = test_metrics.get('IoU', 0)

        if epoch >= args.test_start_epoch and current_iou > best_iou:
            best_metrics = test_metrics
            best_iou = current_iou
            if args.save_dir is not None:
                save_path = os.path.join(model_save_dir, 'best_model.pth')
                print(f"Saving checkpoint at {save_path}, metric {best_iou}")
                torch.save({
                    'epoch': epoch,
                    'model': model_label,
                    'model_state_dict': pipeline.model.state_dict(),
                    'metrics': best_metrics,
                }, save_path)

        if epoch >= args.test_start_epoch:
            if val_loss is not None:
                print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}")
            print(f"Metrics - IoU: {test_metrics['IoU']:.4f}, nIoU: {test_metrics['nIoU']:.4f}, PD: {test_metrics['PD']:.4f}, FA: {test_metrics['FA']:.8f}")
            print(f"Best Metrics - IoU: {best_metrics['IoU']:.4f}, nIoU: {best_metrics['nIoU']:.4f}, PD: {best_metrics['PD']:.4f}, FA: {best_metrics['FA']:.8f}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Testing not started yet")

    print(
        "Training completed. Best model saved with metrics:\n"
        f"IoU: {best_metrics['IoU']:.4f}\n"
        f"nIoU: {best_metrics['nIoU']:.4f}\n"
        f"PD: {best_metrics['PD']:.4f}\n"
        f"FA: {best_metrics['FA']:.8f}"
    )


if __name__ == "__main__":
    train()
