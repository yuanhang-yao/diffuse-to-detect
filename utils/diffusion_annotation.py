import numpy as np
import cv2
from skimage import filters
from skimage.segmentation import slic
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


def Diffusion_Annotation(
    img,
    mask_info,
    D0,
    beta,
    rho,
    alpha,
    dt,
    id_num,
    point_steps=1,
    spot_particles=200,
    spot_steps=30,
    point_dt_scale=0.8,
    point_center_prior_w=0.0,
    point_center_prior_sigma=0.25,
    source_weighted=0,
):
    D0, beta, rho, alpha = float(D0), float(beta), float(rho), float(alpha)
    point_center_prior_w = float(np.clip(point_center_prior_w, 0.0, 1.0))
    point_center_prior_sigma = max(float(point_center_prior_sigma), 1e-3)
    info = mask_info.item()
    out = np.zeros_like(img[:, :, 0], dtype=np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
    gray = clahe.apply((gray * 255).astype(np.uint8)).astype(np.float32) / 255.0

    for target_num in range(len(info.get("Ymax_f"))):
        y0 = info.get("Ymin_f")[target_num]
        y1 = info.get("Ymax_f")[target_num]
        x0 = info.get("Xmin_f")[target_num]
        x1 = info.get("Xmax_f")[target_num]
        cy = info.get("centroid_label_y")[target_num]
        cx = info.get("centroid_label_x")[target_num]
        target_type = info.get("target_type")[target_num]

        py0 = max(0, y0 - min(4, y0))
        py1 = min(gray.shape[0], y1 + min(4, gray.shape[0] - y1))
        px0 = max(0, x0 - min(4, x0))
        px1 = min(gray.shape[1], x1 + min(4, gray.shape[1] - x1))
        crop = gray[py0:py1, px0:px1]

        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            print(f"Image: {id_num} - Target: {target_num}. 'crop' is empty or too small, skipping")
            continue

        gy, gx = np.gradient(crop)
        entropy = filters.rank.entropy((crop * 255).astype(np.uint8), footprint=np.ones((5, 5))) / 8.0
        coef = D0 * np.exp(-beta * (alpha * np.sqrt(gx**2 + gy**2) + (1 - alpha) * entropy))

        dt_i = dt
        if target_type == "Point":
            particles = 20
            steps = point_steps
            dt_i = dt * point_dt_scale
        elif target_type == "Spot":
            particles = spot_particles
            steps = spot_steps
        elif target_type == "Extended":
            particles = 400
            steps = 120
            dt_i = dt * 2.0
        else:
            particles = spot_particles
            steps = spot_steps

        seg = slic(
            cv2.resize(crop[..., np.newaxis], (0, 0), fx=1.0, fy=1.0),
            n_segments=16,
            compactness=0.32,
            channel_axis=None,
        )
        ry, rx = cy - py0, cx - px0

        if 0 <= ry < crop.shape[0] and 0 <= rx < crop.shape[1]:
            cluster = (seg == seg[int(ry), int(rx)]).astype(np.float32)
        else:
            cluster = (crop > filters.threshold_otsu(crop)).astype(np.float32)

        points = []
        if 0 <= ry < crop.shape[0] and 0 <= rx < crop.shape[1]:
            points.append((ry, rx))

        local_max = filters.rank.maximum((crop * 255).astype(np.uint8), footprint=np.ones((7, 7)))
        peaks = np.argwhere(
            (local_max == (crop * 255).astype(np.uint8))
            & (crop > np.mean(crop) + 1.5 * np.std(crop))
        )
        extra = []
        for y, x in peaks:
            if cluster[y, x] > 0:
                extra.append((y, x))
                if len(extra) >= 5:
                    break
        points.extend(extra)

        if not points:
            points.append((crop.shape[0] // 2, crop.shape[1] // 2))

        heat = np.zeros_like(crop)
        if source_weighted and len(points) > 1:
            strength = np.array([float(crop[int(y), int(x)]) + 1e-6 for y, x in points], dtype=np.float32)
            weight = strength / np.maximum(strength.sum(), 1e-6)
            src_particles = np.maximum(1, np.floor(weight * particles).astype(np.int32))
            deficit = int(particles - int(src_particles.sum()))
            if deficit > 0:
                order = np.argsort(-weight)
                for idx in order[:deficit]:
                    src_particles[idx] += 1
            elif deficit < 0:
                order = np.argsort(weight)
                for idx in order:
                    if deficit == 0:
                        break
                    take = min(src_particles[idx] - 1, -deficit)
                    if take > 0:
                        src_particles[idx] -= take
                        deficit += take
            src_particles = src_particles.tolist()
        else:
            src_particles = [particles] * len(points)

        for (sy, sx), n_src in zip(points, src_particles):
            ys = np.ones(n_src) * sy
            xs = np.ones(n_src) * sx
            for _ in range(steps):
                iy = np.clip(ys.astype(int), 0, crop.shape[0] - 1)
                ix = np.clip(xs.astype(int), 0, crop.shape[1] - 1)
                noise_y = np.random.normal(0, 1, n_src)
                noise_x = np.random.normal(0, 1, n_src)
                ys = ys + gy[iy, ix] * dt_i + np.sqrt(2 * coef[iy, ix] * dt_i) * noise_y
                xs = xs + gx[iy, ix] * dt_i + np.sqrt(2 * coef[iy, ix] * dt_i) * noise_x

            iy = np.clip(ys.astype(int), 0, crop.shape[0] - 1)
            ix = np.clip(xs.astype(int), 0, crop.shape[1] - 1)
            for i in range(n_src):
                heat[iy[i], ix[i]] += 1

        heat = cv2.GaussianBlur(heat, (3, 3), 0)
        if heat.max() > 0:
            heat = heat / heat.max()

        fused = rho * heat + (1 - rho) * cluster
        if (
            target_type == "Point"
            and point_center_prior_w > 0.0
            and 0 <= ry < crop.shape[0]
            and 0 <= rx < crop.shape[1]
        ):
            hh, ww = crop.shape
            yy, xx = np.ogrid[:hh, :ww]
            sigma = max(1.0, point_center_prior_sigma * float(max(hh, ww)))
            d2 = (yy - float(ry)) ** 2 + (xx - float(rx)) ** 2
            fused = fused * (
                1.0
                - point_center_prior_w
                + point_center_prior_w * np.exp(-0.5 * d2 / (sigma * sigma)).astype(np.float32)
            )
        fused = np.clip(fused, 0, 1)
        out[py0:py1, px0:px1] = np.maximum(out[py0:py1, px0:px1], fused)

    return out


def post_process(
    mask_info,
    img,
    repeated_mask,
    id_num=-1,
    otsu_factor=1.0,
    crf_iter=5,
    crf_sxy=16,
    crf_srgb=10,
    crf_compat=60,
    disable_crf=False,
    target_thresh=-1.0,
    post_close_k=0,
    post_open_k=0,
    post_min_area=0,
):
    info = mask_info.item()
    otsu_factor = max(0.0, otsu_factor)

    if np.max(repeated_mask) > 0:
        raw = (repeated_mask * 255).astype(np.uint8)
        if np.sum(raw > 0) > 10:
            th, _ = cv2.threshold(raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = (raw > min(int(th * otsu_factor), 255)).astype(np.uint8) * 255
        else:
            binary = (raw > 127).astype(np.uint8) * 255
    else:
        binary = np.zeros_like(repeated_mask, dtype=np.uint8)

    final = np.zeros_like(binary)
    close_kernel = np.ones((post_close_k, post_close_k), np.uint8) if post_close_k > 1 else None
    open_kernel = np.ones((post_open_k, post_open_k), np.uint8) if post_open_k > 1 else None

    for target_num in range(len(info.get("Ymax_f"))):
        y0 = info.get("Ymin_f")[target_num]
        y1 = info.get("Ymax_f")[target_num]
        x0 = info.get("Xmin_f")[target_num]
        x1 = info.get("Xmax_f")[target_num]

        crop = binary[y0:y1, x0:x1]
        if crop.max() == 0:
            fg = np.expand_dims(np.zeros_like(crop), axis=0)
        else:
            fg = np.expand_dims(crop / crop.max(), axis=0)
        bin_thresh = target_thresh if target_thresh >= 0.0 else 0.5
        if disable_crf or crf_iter <= 0:
            pred = (fg.squeeze(0) > bin_thresh).astype(np.uint8)
        else:
            probs = np.concatenate([1 - fg, fg], axis=0).astype(np.float32)
            _, h, w = probs.shape
            d = dcrf.DenseCRF2D(w, h, 2)
            d.setUnaryEnergy(np.ascontiguousarray(unary_from_softmax(probs)))
            d.addPairwiseBilateral(
                sxy=crf_sxy,
                srgb=crf_srgb,
                rgbim=img[y0:y1, x0:x1].copy(order="C"),
                compat=crf_compat,
            )
            prob_seg = np.array(d.inference(crf_iter), dtype=np.float32).reshape((2, h, w))
            if target_thresh >= 0.0:
                pred = (prob_seg[1] > target_thresh).astype(np.uint8)
            else:
                pred = np.argmax(prob_seg, axis=0).astype(np.uint8)

        if close_kernel is not None:
            pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, close_kernel)
        if open_kernel is not None:
            pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, open_kernel)
        if post_min_area > 0 and pred.any():
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
            filtered = np.zeros_like(pred, dtype=np.uint8)
            for lab in range(1, n_labels):
                if stats[lab, cv2.CC_STAT_AREA] >= post_min_area:
                    filtered[labels == lab] = 1
            pred = filtered

        final[y0:y1, x0:x1] = pred
        final[final > 0] = 255

    return final
