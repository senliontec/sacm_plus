"""DSCNet 2D baseline: dynamic snake convolution for curvilinear structures.

Official reference: ``3rd/DSCNet/DSCNet_2D_opensource`` (MIT).  The port
vendors the model, DSConv and soft-BCE loss; the NIfTI/MONAI training and
patch-prediction plumbing is replaced by the shared runtime and a
dependency-free PIL implementation.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import rotate as nd_rotate
from torch import nn

from baseline_zoo.data import load_mask, load_split_samples
from baseline_zoo.models.dscnet import build_dscnet, cross_loss
from baseline_zoo.runtime import BaselineAdapter


class DSCNetAdapter(BaselineAdapter):
    id = 'dscnet'
    display_name = 'DSCNet'
    official_source = 'YaoleiQi/DSCNet (self-contained 2D copy)'
    official_entrypoint = 'https://github.com/YaoleiQi/DSCNet'
    input_size = 224
    official_lr = 1e-4
    official_weight_decay = 0.01
    train_batch_size = 1
    selection = 'dice'
    deviations = (
        'The official DRIVE pipeline uses NIfTI I/O, MONAI augmentation and '
        'SimpleITK inference; this port uses PIL grayscale I/O, simple '
        'flips/90-degree rotations/+/-20-degree rotations/noise and the '
        'same patch inference, with no MONAI or SimpleITK dependency.',
        'The official global scalar mean/std is computed over all training '
        'images; this port computes it over the shared 18-image train split '
        'for each seed and stores it in the checkpoint.',
        'Official DataLoader workers and RNG are not preserved: this port '
        'uses workers=0 and the shared seed. The checkpoint-selection rule '
        'also differs (see next item).',
        'The official checkpoint selection predicts the test set from epoch '
        '200 onward and selects by test Dice; this port selects on the '
        'shared 6-image validation Dice.',
        'The official batch-1, 224-crop, AdamW 1e-4 with betas (0.9,0.95) '
        'recipe and the soft BCE loss are preserved; the official 400-epoch '
        'schedule is extended to the shared 800-epoch budget. The '
        'built-but-never-stepped ReduceLROnPlateau is omitted (constant LR). '
        'A soft Dice term is added to the official BCE for the 3-shot '
        'class-imbalance protocol; --dsc-dice-weight=0 restores BCE-only.',
        'The official patch-blending kernel has a singular 1/0 center; the '
        'port uses a finite 1/(1+d^4) quartic window and explicit last-patch '
        'coverage. No official weights are present; training starts from '
        'scratch.',
        'Predictions are exported at original GT resolution; the upstream '
        'pipeline saves probability NIfTI volumes.',
        'No test-time augmentation.',
    )

    def __init__(self):
        super().__init__()
        self._mean: float | None = None
        self._std: float | None = None
        self._criterion = cross_loss()

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=224)
        parser.add_argument('--dsc-basic-layer', type=int, default=16)
        parser.add_argument('--dsc-kernel-size', type=int, default=9)
        parser.add_argument('--dsc-stride', type=int, default=112)
        parser.add_argument(
            '--dsc-dice-weight', type=float, default=1.0,
            help='soft Dice weight added to the official soft BCE '
                 '(0 restores the official BCE-only loss)')

    @staticmethod
    def _compute_stats(data_root: Path) -> tuple[float, float]:
        samples = load_split_samples(Path(data_root), 'train')
        total = 0.0
        sumsq = 0.0
        count = 0
        for sample in samples:
            with Image.open(sample.image_path) as image:
                values = np.asarray(image.convert('L'), dtype=np.float64)
            total += float(values.sum())
            sumsq += float((values ** 2).sum())
            count += int(values.size)
        if count == 0:
            return 0.0, 1.0
        mean = total / count
        std = math.sqrt(max(sumsq / count - mean ** 2, 1e-6))
        return mean, std

    def build_model(self, args) -> nn.Module:
        self.input_size = int(args.input_size)
        if self._mean is None or self._std is None:
            self._mean, self._std = self._compute_stats(Path(args.data_root))
        return build_dscnet(
            n_channels=1,
            n_classes=1,
            basic_layer=int(args.dsc_basic_layer),
            kernel_size=int(args.dsc_kernel_size),
        )

    def configure_trainable(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.9, 0.95))

    def _load(self, sample):
        with Image.open(sample.image_path) as image:
            values = np.asarray(image.convert('L'), dtype=np.float32)
        mask = load_mask(sample.mask_path).astype(np.float32)
        return values, mask

    def _normalize(self, values: np.ndarray) -> torch.Tensor:
        mean = self._mean if self._mean is not None else 0.0
        std = self._std if self._std is not None else 1.0
        values = (values - mean) / std
        return torch.from_numpy(
            values.astype(np.float32).copy()).unsqueeze(0)

    def _preprocess(self, sample, rng: random.Random, train: bool):
        values, mask = self._load(sample)
        if train:
            if rng.random() < 0.5:
                values = np.flip(values, axis=1)
                mask = np.flip(mask, axis=1)
            if rng.random() < 0.5:
                values = np.flip(values, axis=0)
                mask = np.flip(mask, axis=0)
            if rng.random() < 0.5:
                k = rng.randrange(4)
                values = np.rot90(values, k)
                mask = np.rot90(mask, k)
            elif rng.random() < 0.3:
                angle = rng.uniform(-20, 20)
                values = nd_rotate(values, angle, order=1, reshape=False)
                mask = nd_rotate(
                    mask.astype(np.float32), angle, order=0,
                    reshape=False)

            h, w = values.shape
            if h < self.input_size or w < self.input_size:
                values = np.asarray(Image.fromarray(
                    np.clip(values, 0, 255).astype(np.uint8)).resize(
                        (self.input_size, self.input_size), Image.BILINEAR),
                    dtype=np.float32)
                mask = np.asarray(Image.fromarray(
                    mask.astype(np.uint8)).resize(
                        (self.input_size, self.input_size), Image.NEAREST),
                    dtype=np.float32)
            else:
                top = rng.randrange(0, h - self.input_size + 1)
                left = rng.randrange(0, w - self.input_size + 1)
                values = values[top:top + self.input_size,
                                left:left + self.input_size]
                mask = mask[top:top + self.input_size,
                            left:left + self.input_size]

        return self._normalize(values), torch.from_numpy(
            (mask > 0.5).astype(np.float32).copy()).unsqueeze(0)

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image, mask = self._preprocess(sample, rng, train=True)
            images.append(image)
            masks.append(mask)
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    @staticmethod
    def _soft_dice(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dims = (1, 2, 3)
        intersection = (pred * target).sum(dims)
        denominator = pred.sum(dims) + target.sum(dims)
        return (1.0 - (2.0 * intersection + 1e-5)
                / (denominator + 1e-5)).mean()

    def train_step(self, model, batch):
        images, masks = batch
        pred = model(images)
        loss = self._criterion(masks, pred)
        weight = float(getattr(self.args, 'dsc_dice_weight', 1.0))
        if weight:
            loss = loss + weight * self._soft_dice(pred, masks)
        return loss

    def prepare_eval_sample(self, sample, device):
        values, _ = self._load(sample)
        height, width = values.shape
        image = self._normalize(values).unsqueeze(0).to(device)
        return {'image': image, 'original_size': (height, width)}

    @staticmethod
    def _starts(length: int, patch: int, stride: int) -> list[int]:
        starts = list(range(0, max(length - patch, 0) + 1, stride))
        last = length - patch
        if last > 0 and (not starts or starts[-1] != last):
            starts.append(last)
        return starts or [0]

    def predict_sample(self, model, prepared, sample):
        image = prepared['image']
        height, width = prepared['original_size']
        patch = self.input_size
        stride = max(1, int(getattr(self.args, 'dsc_stride', patch // 2)))
        if stride > patch:
            raise SystemExit(
                f'--dsc-stride ({stride}) must be <= --input-size ({patch}); '
                'larger strides leave uncovered image stripes')
        side = max(height, width)
        side_p = max(patch, (side // patch + 1) * patch)
        pad_h = side_p - height
        pad_w = side_p - width
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h))

        coords = torch.arange(patch, device=image.device,
                              dtype=image.dtype) - patch // 2
        window = 1.0 / (1.0 + coords[:, None] ** 4 + coords[None, :] ** 4)
        window = window / window.max()
        window = window.view(1, 1, patch, patch)

        prob = torch.zeros_like(image)
        norm = torch.zeros_like(image)
        for top in self._starts(side_p, patch, stride):
            for left in self._starts(side_p, patch, stride):
                tile = image[..., top:top + patch, left:left + patch]
                pred = model(tile)
                prob[..., top:top + patch, left:left + patch] += (
                    pred * window)
                norm[..., top:top + patch, left:left + patch] += window
        prob = prob / norm.clamp_min(1e-12)
        prob = prob[..., :height, :width]
        return prob[0, 0].detach().cpu().numpy() > 0.5

    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'dsc_basic_layer': int(getattr(args, 'dsc_basic_layer', 16)),
            'dsc_kernel_size': int(getattr(args, 'dsc_kernel_size', 9)),
            'dsc_stride': int(getattr(args, 'dsc_stride', 112)),
            'dsc_dice_weight': float(getattr(args, 'dsc_dice_weight', 1.0)),
            'mean': float(self._mean if self._mean is not None else 0.0),
            'std': float(self._std if self._std is not None else 1.0),
        }

    def restore_config(self, args, config):
        for key in ('input_size', 'dsc_basic_layer', 'dsc_kernel_size',
                    'dsc_stride', 'dsc_dice_weight'):
            if key in config:
                setattr(args, key, config[key])
        if 'mean' in config:
            self._mean = float(config['mean'])
        if 'std' in config:
            self._std = float(config['std'])
        self.input_size = int(getattr(args, 'input_size', 224))
