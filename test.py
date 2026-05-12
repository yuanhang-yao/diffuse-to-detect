import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

from models.alcl_model import ALCLNet
from models.dna_net import DNANet, Res_CBAM_block
from utils.dataset import image_mean_std
from utils.metrics import PD_FA_2, SamplewiseSigmoidMetric, SigmoidMetric


DEFAULT_DATASET_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Ours", "dataset")
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alclnet", choices=["alclnet", "dnanet"])
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="predictions")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
    return parser.parse_args()


def prepare_dataset(args):
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "SIRST3"))
    src_root = os.path.abspath(os.path.join(args.dataset_root, "SIRST3"))
    os.makedirs(os.path.join(dataset_dir, "mode"), exist_ok=True)

    for src_name, dst_name in [("images", "images"), ("full-supervised", "labels")]:
        dst = os.path.join(dataset_dir, dst_name)
        if not os.path.lexists(dst) and os.path.exists(os.path.join(src_root, src_name)):
            os.symlink(os.path.join(src_root, src_name), dst)

    for name in ["train", "val", "test"]:
        src = os.path.join(src_root, "mode_bilevel", f"{name}.txt")
        dst = os.path.join(dataset_dir, "mode", f"{name}.txt")
        if os.path.exists(src):
            tmp = f"{dst}.{os.getpid()}.tmp"
            with open(src, "r", encoding="utf-8") as f_src, open(tmp, "w", encoding="utf-8") as f_dst:
                f_dst.write(f_src.read())
            os.replace(tmp, dst)
    return dataset_dir


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


def load_state_dict(model, weights_path):
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "net", "network"]:
            if isinstance(checkpoint.get(key), dict):
                state_dict = checkpoint[key]
                break
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {weights_path}")

    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)


def predict_batch(model, img, batch_size, crop_size):
    b, c, h, w = img.shape
    if h <= crop_size or w <= crop_size:
        return model(img)

    patches = torch.nn.functional.unfold(img, kernel_size=crop_size, stride=crop_size)
    patches = patches.reshape(b, c, crop_size, crop_size, -1).permute(0, 4, 1, 2, 3)
    outputs = []
    for i in range(0, patches.size(1), batch_size):
        batch_patches = patches[:, i:i + batch_size].reshape(-1, c, crop_size, crop_size)
        outputs.append(model(batch_patches.float()))
    outputs = torch.cat(outputs, dim=0).permute(1, 2, 3, 0).reshape(b, -1, patches.size(1))
    return torch.nn.functional.fold(outputs, kernel_size=crop_size, stride=crop_size, output_size=(h, w))


def load_image(path, crop_size, mean, std):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    img = img.resize((crop_size, crop_size), Image.BILINEAR)
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = (img - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1)), h, w


def load_mask(path, crop_size):
    mask = Image.open(path).convert("L")
    mask = mask.resize((crop_size, crop_size), Image.NEAREST)
    return torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0).unsqueeze(0)


def test():
    args = parse_args()
    device = torch.device(f"cuda:{int(str(args.gpu_id).split(',')[0])}")
    torch.cuda.set_device(device)

    dataset_dir = prepare_dataset(args)
    img_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")
    with open(os.path.join(dataset_dir, "mode", "test.txt"), "r", encoding="utf-8") as f:
        img_ids = [line.strip() for line in f if line.strip()]
    stat_ids = []
    for name in ["train", "val"]:
        with open(os.path.join(dataset_dir, "mode", f"{name}.txt"), "r", encoding="utf-8") as f:
            stat_ids.extend(line.strip() for line in f if line.strip())

    mean, std = image_mean_std(img_dir, stat_ids)
    model = create_model(args.model).to(device)
    load_state_dict(model, args.weights)
    model.eval()
    metric_iou = SigmoidMetric(score_thresh=0.5)
    metric_niou = SamplewiseSigmoidMetric(nclass=1, score_thresh=0.5)
    metric_pd_fa = PD_FA_2(nclass=1)

    os.makedirs(args.output_dir, exist_ok=True)
    with torch.no_grad():
        for i in range(0, len(img_ids), args.batch_size):
            batch_ids = img_ids[i:i + args.batch_size]
            batch_imgs, batch_masks, sizes = [], [], []
            for img_id in batch_ids:
                img, h, w = load_image(os.path.join(img_dir, f"{img_id}.png"), args.crop_size, mean, std)
                batch_imgs.append(img)
                batch_masks.append(load_mask(os.path.join(label_dir, f"{img_id}.png"), args.crop_size))
                sizes.append((h, w))

            probs = torch.sigmoid(predict_batch(
                model,
                torch.stack(batch_imgs, dim=0).to(device),
                args.batch_size,
                args.crop_size,
            ))
            masks = torch.stack(batch_masks, dim=0).to(device)
            for b_idx in range(probs.shape[0]):
                metric_iou.update(probs[b_idx:b_idx+1], masks[b_idx:b_idx+1])
                metric_niou.update(probs[b_idx:b_idx+1], masks[b_idx:b_idx+1])
                metric_pd_fa.update(probs[b_idx:b_idx+1], masks[b_idx:b_idx+1])

            preds = (probs > 0.5).to(torch.uint8).cpu().numpy() * 255
            for img_id, pred, (h, w) in zip(batch_ids, preds[:, 0], sizes):
                if pred.shape != (h, w):
                    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(args.output_dir, f"{img_id}.png"), pred)

    print(f"Saved {len(img_ids)} predictions to {os.path.abspath(args.output_dir)}")
    _, iou = metric_iou.get()
    _, niou = metric_niou.get()
    fa, pd = metric_pd_fa.get()
    print(f"IoU: {iou:.4f}, nIoU: {niou:.4f}, PD: {pd:.4f}, FA: {fa:.8f}")


if __name__ == "__main__":
    test()
