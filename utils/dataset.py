import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2


def image_mean_std(img_dir, img_ids):
    names = [f"{img_id}.png" for img_id in img_ids if img_id]
    if not names:
        raise FileNotFoundError("No images found.")

    means, stds = [], []
    for name in names:
        img = np.asarray(Image.open(os.path.join(img_dir, name)).convert("RGB"), dtype=np.float32) / 255.0
        pixels = img.reshape(-1, 3)
        means.append(pixels.mean(axis=0))
        stds.append(pixels.std(axis=0))

    mean = [float(v) for v in np.mean(means, axis=0)]
    std = [float(v) for v in np.maximum(np.mean(stds, axis=0), 1e-6)]
    print(f"SIRST3 mean={mean}, std={std}")
    return mean, std


class Augmentation:
    def __init__(self, crop_size, is_train=True):
        self.crop_size = crop_size
        self.is_train = is_train

    def __call__(self, img, mask):
        if isinstance(img, Image.Image):
            img = np.array(img)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        if img.dtype in (np.float32, np.float64):
            img = (img * 255).astype(np.uint8)
        if mask.dtype in (np.float32, np.float64):
            mask = (mask * 255).astype(np.uint8)

        h, w, _ = img.shape
        assert (h, w) == mask.shape
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        if self.is_train:
            w, h = img.size
            long_size = random.randint(int(self.crop_size * 0.5), int(self.crop_size * 2.0))
            if h > w:
                new_h = long_size
                new_w = int(1.0 * w * long_size / h + 0.5)
                short_size = new_w
            else:
                new_w = long_size
                new_h = int(1.0 * h * long_size / w + 0.5)
                short_size = new_h

            img = F.resize(img, [new_h, new_w], interpolation=Image.BILINEAR)
            mask = F.resize(mask, [new_h, new_w], interpolation=Image.NEAREST)
            if short_size < self.crop_size:
                img = F.pad(img, [0, 0, max(0, self.crop_size - new_w), max(0, self.crop_size - new_h)], fill=0, padding_mode="reflect")
                mask = F.pad(mask, [0, 0, max(0, self.crop_size - new_w), max(0, self.crop_size - new_h)], fill=0, padding_mode="reflect")

            current_w, current_h = img.size
            if current_w >= self.crop_size and current_h >= self.crop_size:
                i, j, h_crop, w_crop = transforms.RandomCrop.get_params(img, (self.crop_size, self.crop_size))
                img = F.crop(img, i, j, h_crop, w_crop)
                mask = F.crop(mask, i, j, h_crop, w_crop)
        else:
            img = F.resize(img, [self.crop_size, self.crop_size], interpolation=Image.BILINEAR)
            mask = F.resize(mask, [self.crop_size, self.crop_size], interpolation=Image.NEAREST)

        return np.array(img), np.array(mask)


class UnifiedDataset(Dataset):
    def __init__(self, args, split, pde_run_tag=None):
        self.split = split
        self.is_train = split == "train"
        self.crop_size = args.crop_size
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "SIRST3"))
        src_root = os.path.abspath(os.path.join(args.dataset_root, "SIRST3"))

        for name in ["mode", "single_point", "image_info_centroid"]:
            os.makedirs(os.path.join(self.root_dir, name), exist_ok=True)
        for src_name, dst_name in [("images", "images"), ("full-supervised", "labels")]:
            dst = os.path.join(self.root_dir, dst_name)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(src_root, src_name), dst)
        for name in ["train", "val", "test", "prior_difficulty"]:
            src = os.path.join(src_root, "mode_bilevel", f"{name}.txt")
            dst = os.path.join(self.root_dir, "mode", f"{name}.txt")
            tmp = f"{dst}.{os.getpid()}.tmp"
            with open(src, "r", encoding="utf-8") as f_src, open(tmp, "w", encoding="utf-8") as f_dst:
                f_dst.write(f_src.read())
            os.replace(tmp, dst)

        self.img_dir = os.path.join(self.root_dir, "images")
        if split == "test":
            self.mask_dir = os.path.join(self.root_dir, "labels")
        else:
            self.mask_dir = os.path.join(self.root_dir, "single_point", pde_run_tag, split)
        with open(os.path.join(self.root_dir, "mode", f"{split}.txt"), "r", encoding="utf-8") as f:
            self.img_ids = [line.strip() for line in f if line.strip()]

        stat_ids = []
        for name in ["train", "val"]:
            with open(os.path.join(self.root_dir, "mode", f"{name}.txt"), "r", encoding="utf-8") as f:
                stat_ids.extend(line.strip() for line in f if line.strip())
        mean, std = image_mean_std(self.img_dir, stat_ids)
        self.transform = Augmentation(self.crop_size, self.is_train)
        self.train_augmenter = A.Compose([
            A.SomeOf([
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0, p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.3, p=0.2),
                A.Rotate(limit=45, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, p=0.5),
                A.GaussNoise(p=0.2),
                A.NoOp(),
                A.NoOp(),
            ], 3, p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ])
        self.valtest_augmenter = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = Image.open(os.path.join(self.img_dir, img_id + ".png")).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, img_id + ".png")).convert("L")
        w, h = img.size
        img, mask = self.transform(img, mask)
        transformed = (self.train_augmenter if self.is_train else self.valtest_augmenter)(image=img, mask=mask)
        return transformed["image"], transformed["mask"].unsqueeze(0).float() / 255.0, img_id, h, w


def create_dataset(args, split, pde_run_tag=None):
    return DataLoader(
        dataset=UnifiedDataset(args, split=split, pde_run_tag=pde_run_tag),
        batch_size=args.batch_size,
        shuffle=split != "test",
        num_workers=args.workers,
        drop_last=split == "train",
    )
