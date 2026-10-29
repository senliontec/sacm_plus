"""K-Prism preprocessing and augmentation.

Training reproduces the upstream albumentations 1.3.1 pipeline from
``kprism/data_mapper/train_data_utils.py`` verbatim.  Evaluation reproduces
``SegmentationPreprocessor``: resize the longest side to 512, centre-pad to
512, then normalize with the upstream mean/std.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations import (
    Compose,
    GaussianBlur,
    GaussNoise,
    GridDistortion,
    HorizontalFlip,
    LongestMaxSize,
    Normalize,
    PadIfNeeded,
    RandomBrightnessContrast,
    ShiftScaleRotate,
)
from albumentations.pytorch import ToTensorV2

MEAN = (0.26, 0.26, 0.26)
STD = (0.30, 0.30, 0.30)


def build_train_transform(image_size: int = 512) -> Compose:
    """Upstream ``build_image_transform`` (same transforms and probabilities)."""
    return Compose([
        LongestMaxSize(max_size=image_size),
        PadIfNeeded(
            min_height=image_size, min_width=image_size,
            border_mode=0, value=0, mask_value=0),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.05, rotate_limit=45, p=0.3,
            border_mode=0, value=0, mask_value=0),
        RandomBrightnessContrast(p=0.4),
        GaussianBlur(p=0.3),
        GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        GridDistortion(p=0.3, border_mode=0, value=0, mask_value=0),
        Normalize(mean=MEAN, std=STD, max_pixel_value=255.0),
        ToTensorV2(transpose_mask=True),
    ])


def read_rgb(path) -> np.ndarray:
    """Read an RGB uint8 image (upstream test loader uses cv2 BGR->RGB)."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f'cv2 could not read {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_binary_mask(path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f'cv2 could not read {path}')
    return (mask > 127).astype(np.uint8)


def resize_pad_normalize(
        image_rgb: np.ndarray, image_size: int = 512,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int, int, int]]:
    """Official eval preprocessing, returning CHW normalized float32.

    Returns ``(image, (height, width), (pad_top, pad_bottom, pad_left,
    pad_right))``.  The original size is the pre-resize image size, used to
    map the prediction back to the GT grid.
    """
    tensor = torch.from_numpy(
        np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).float()
    _, height, width = tensor.shape
    scale = image_size / max(height, width)
    new_h, new_w = int(height * scale), int(width * scale)
    resized = F.interpolate(
        tensor.unsqueeze(0), size=(new_h, new_w),
        mode='bilinear', align_corners=False)
    pad_h = image_size - new_h
    pad_w = image_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = F.pad(
        resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    mean = torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1)
    normalized = (padded / 255.0 - mean) / std
    return (
        normalized[0],
        (height, width),
        (pad_top, pad_bottom, pad_left, pad_right),
    )
