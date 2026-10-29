"""Offline 18x augmentation for BCU-Net (paper Section 4.1).

The paper says each retinal dataset was augmented 18 times with rotation,
flipping, noise, brightness, blurring, and combinations, but gives no
probabilities or amplitudes.  This module fixes one deterministic variant
per (image, index) pair; the recipe below is a documented reconstruction.
Images stay in the official BGR [0,255] space and masks are thresholded
before augmentation, matching the released data loader.
"""

from __future__ import annotations

import zlib

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# (rotation degrees, hflip, vflip, brightness, blur sigma, noise sigma)
_VARIANTS = (
    (0.0, False, False, 1.0, 0.0, 0.0),
    (15.0, False, False, 1.0, 0.0, 0.0),
    (-15.0, False, False, 1.0, 0.0, 0.0),
    (30.0, False, False, 1.0, 0.0, 0.0),
    (-30.0, False, False, 1.0, 0.0, 0.0),
    (0.0, True, False, 1.0, 0.0, 0.0),
    (0.0, False, True, 1.0, 0.0, 0.0),
    (0.0, True, True, 1.0, 0.0, 0.0),
    (0.0, False, False, 1.2, 0.0, 0.0),
    (0.0, False, False, 0.8, 0.0, 0.0),
    (0.0, False, False, 1.0, 0.0, 10.0),
    (0.0, False, False, 1.0, 0.0, 20.0),
    (0.0, False, False, 1.0, 1.0, 0.0),
    (0.0, False, False, 1.0, 2.0, 0.0),
    (15.0, True, False, 1.1, 0.0, 0.0),
    (-15.0, False, True, 1.0, 0.0, 8.0),
    (0.0, True, False, 0.9, 1.0, 0.0),
    (0.0, False, True, 1.0, 1.5, 5.0),
)

N_VARIANTS = len(_VARIANTS)

# Paper Section 4.1 scopes the 18x augmentation to retinal vessel datasets.
RETINAL_VESSEL_DATASETS = frozenset({
    'CHASEDB1', 'DRIVE', 'STARE', 'DRHAGIS', 'HRF', 'LES-AV',
})


def _read_pair(sample):
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f'cv2 could not read {sample.image_path}')
    if mask is None:
        raise SystemExit(f'cv2 could not read {sample.mask_path}')
    mask = (mask >= 128).astype(np.uint8)
    return image, mask


def _stable_seed(key: str, variant: int, seed: int) -> int:
    return zlib.crc32(f'{seed}:{key}:{variant}'.encode('utf-8'))


def apply_variant(image, mask, variant: int, seed: int = 0, key: str = ''):
    """Apply one fixed augmentation recipe to a BGR image / binary mask."""
    angle, hflip, vflip, brightness, blur, noise = _VARIANTS[variant]
    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)
    if angle:
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D(
            (width / 2.0, height / 2.0), angle, 1.0)
        image = cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask = cv2.warpAffine(
            mask, matrix, (width, height), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if hflip:
        image = np.flip(image, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()
    if vflip:
        image = np.flip(image, axis=0).copy()
        mask = np.flip(mask, axis=0).copy()
    if brightness != 1.0:
        image = np.clip(
            image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    if blur > 0:
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=blur)
    if noise > 0:
        rng = np.random.default_rng(_stable_seed(key, variant, seed))
        image = np.clip(
            image.astype(np.float32)
            + rng.normal(0.0, noise, image.shape), 0, 255).astype(np.uint8)
    return image, mask


def _resize_pair(image, mask, size: int):
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(
        np.transpose(image, (2, 0, 1)).copy()).float()
    # The release casts the INTER_AREA label to int64 (truncation).
    mask = torch.from_numpy(mask.astype(np.int64).copy())
    return image, mask


class BCUNetTrainDataset(Dataset):
    """Deterministic 18-variant expansion of the BCU-Net training split.

    ``augment_scope='retinal'`` applies the paper's transforms only to
    retinal vessel datasets; other samples are repeated with the identity
    variant so the pooled 18/6 domain balance is unchanged.
    """

    def __init__(self, samples, input_size: int = 512, variants: int = 18,
                 seed: int = 0, augment_scope: str = 'retinal'):
        if variants < 1:
            raise ValueError('variants must be at least 1')
        if variants > N_VARIANTS:
            raise ValueError(
                f'variants={variants} exceeds the {N_VARIANTS} fixed recipes')
        if augment_scope not in ('retinal', 'all'):
            raise ValueError(
                f'augment_scope must be retinal|all, got {augment_scope!r}')
        self.samples = list(samples)
        self.input_size = int(input_size)
        self.variants = int(variants)
        self.seed = int(seed)
        self.augment_scope = augment_scope

    def __len__(self):
        return len(self.samples) * self.variants

    def __getitem__(self, index):
        sample_index, variant = divmod(index, self.variants)
        sample = self.samples[sample_index]
        if (self.augment_scope == 'retinal'
                and sample.dataset not in RETINAL_VESSEL_DATASETS):
            variant = 0
        image, mask = _read_pair(sample)
        image, mask = apply_variant(
            image, mask, variant, seed=self.seed, key=sample.stem)
        return _resize_pair(image, mask, self.input_size)
