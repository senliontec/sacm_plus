"""Unified data loading.

- SegmentationDataset: train/val splits (root/{split}/images + masks),
  optional joint augmentation (verbatim from the original train_sam.py).
- TestDataset: a single images/ + masks/ root, returning items at the
  original resolution for evaluation (verbatim from test_sam.py).
"""

import logging
import os

from PIL import Image
from torch.utils.data import Dataset

# DIS5K 含超大原图(1 亿+ 像素),关闭 PIL 的"解压炸弹"保护以避免警告刷屏;
# 数据为本项目自管数据集,无恶意输入风险。
Image.MAX_IMAGE_PIXELS = None

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def _find_pairs(image_dir, mask_dir):
    pairs = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            base_name = os.path.splitext(filename)[0]
            for ext in SUPPORTED_EXTENSIONS:
                if os.path.exists(os.path.join(mask_dir, base_name + ext)):
                    pairs.append(filename)
                    break
    return pairs


class SegmentationDataset(Dataset):
    """Train/val split dataset (root/<split>/images + root/<split>/masks)."""

    def __init__(self, root_dir, split='train', transform=None, augment=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.augment = augment
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'masks')

        self.image_files = _find_pairs(self.image_dir, self.mask_dir)
        if not self.image_files:
            raise ValueError(f"No valid image-mask pairs found in {self.image_dir}")
        logging.info(f"Found {len(self.image_files)} image-mask pairs in {split} set")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        base_name = os.path.splitext(img_name)[0]
        mask_path = None
        for ext in SUPPORTED_EXTENSIONS:
            potential = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(potential):
                mask_path = potential
                break
        if mask_path is None:
            raise ValueError(f"No corresponding mask found for image {img_name}")

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Joint augmentation (train only): identical geometric transform
        # applied to image and mask so the annotation stays aligned.
        if self.augment is not None:
            image, mask = self.augment(image, mask)

        return image, mask


class TestDataset(Dataset):
    """Evaluation dataset (root/images + root/masks): the model input is
    resized to 1024x1024, the GT stays at the ORIGINAL resolution for
    metric computation."""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')

        self.image_files = _find_pairs(self.image_dir, self.mask_dir)
        if not self.image_files:
            raise ValueError(f"No valid image-mask pairs found in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        import torch
        from torchvision import transforms

        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        base_name = os.path.splitext(img_name)[0]
        mask_path = None
        for ext in SUPPORTED_EXTENSIONS:
            potential = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(potential):
                mask_path = potential
                break
        if mask_path is None:
            raise ValueError(f"No corresponding mask found for image {img_name}")

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        original_size = image.size

        model_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
        mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        model_input = model_transform(image)
        original_mask = mask_transform(mask)

        return model_input, original_mask, original_size, img_name
