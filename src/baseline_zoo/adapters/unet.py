"""U-Net baseline: vendored milesial/PyTorch-UNet architecture.

The model, BCE+Dice objective, per-image scale preprocessing and
logit-space bilinear upsampling are preserved from milesial/PyTorch-UNet.
Training on the 18-image 3-shot split uses a task-adapted recipe modelled
on the standard 2D supervised U-Net trainer in Mamba-UNet/SSL4MIS:

* fixed-size 512x512 random patches.  Mamba-UNet resizes whole slices to
  256x256 and trains with batch 24; native patches keep the thin
  CREMI/CrackTree structures instead of downsampling them away,
* half of the patches are centred on a foreground pixel, because protocol
  foreground is only 0.25%-9% of the image,
* batch 8 with eight patches per training image per epoch (~14.4k
  optimizer steps over 800 epochs, the same order as Mamba-UNet's 10k
  iterations),
* joint flips / 90-degree rotations plus mild photometric jitter,
* AdamW 1e-3 / wd 1e-4 with cosine over 800 epochs, gradient clipping 1.0.

Validation and prediction stay at native resolution with the official
preprocessing and logit-space bilinear upsampling.  Label the row
"3-shot U-Net (official architecture; adapted recipe)".
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from baseline_zoo.models.unet import UNet
from baseline_zoo.runtime import BaselineAdapter


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Official milesial Dice loss (global over batch, eps=1e-6).

    Matches ``dice_coeff(..., reduce_batch_first=True)``: the empty-set
    branch reuses ``inter`` so two empty masks give Dice 1, and the
    epsilon is added to numerator and denominator.
    """
    prob = torch.sigmoid(logits)
    # Upstream calls dice_coeff(..., reduce_batch_first=True): one Dice
    # value over the whole batch/spatial extent, not a per-image mean.
    inter = 2.0 * (prob * target).sum()
    sets_sum = prob.sum() + target.sum()
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (inter + 1e-6) / (sets_sum + 1e-6)
    return 1.0 - dice


class _RepeatedSampleDataset(Dataset):
    """Repeat each training image so one epoch covers more random patches."""

    def __init__(self, samples, repeats: int):
        self.samples = list(samples)
        self.repeats = int(repeats)
        if self.repeats < 1:
            raise ValueError('repeats must be at least 1')

    def __len__(self) -> int:
        return len(self.samples) * self.repeats

    def __getitem__(self, index: int):
        return self.samples[index % len(self.samples)]


class UNetAdapter(BaselineAdapter):
    id = 'unet'
    display_name = 'U-Net'
    official_source = 'milesial/PyTorch-UNet (self-contained copy)'
    official_entrypoint = 'https://github.com/milesial/Pytorch-UNet'
    # Official preprocessing is a per-image scale factor, so the model input
    # is variable-size; 0 marks that this adapter does not use a fixed size.
    input_size = 0
    scale = 1.0
    # Task-adapted 3-shot recipe; the official RMSprop recipe is 1e-5/1e-8.
    official_lr = 1e-3
    official_weight_decay = 1e-4
    patch_size = 512
    crops_per_image = 8
    fg_crop_prob = 0.5
    photometric = True
    train_batch_size = 8
    default_grad_clip = 1.0
    selection = 'dice'
    # The shared runtime steps the scheduler after validation; the override
    # below handles cosine and any future metric-based scheduler.
    scheduler_after_validation = True
    deviations = (
        'Official milesial architecture, BCE+Dice loss, per-image scale '
        'preprocessing and logit-space bilinear upsampling are preserved.',
        'The literal official 5-epoch RMSprop lr=1e-5/ReduceLROnPlateau '
        'recipe collapsed the LR to 1e-8 by epoch 19 on the 18-image split '
        '(three groups of five non-improving epochs; PyTorch eps=1e-8 then '
        'prevents further reductions).  The 3-shot recipe follows the '
        'standard 2D supervised U-Net recipe used by Mamba-UNet/SSL4MIS: '
        'fixed-size inputs, large batch, a long schedule, flips/rotation '
        'and Dice+CE.  This port uses native 512x512 patches, batch 8, '
        'eight patches per training image per epoch, AdamW lr=1e-3/'
        'wd=1e-4, cosine over 800 epochs and gradient clipping 1.0.  Label '
        'this row "3-shot U-Net (official architecture; adapted recipe)", '
        'not an official U-Net reproduction.',
        'Half of the training patches are centred on a foreground pixel '
        '(protocol foreground is 0.25%-9%); patches receive joint flips, '
        '0/90/180/270-degree rotations and mild photometric jitter.  '
        'Mamba-UNet uses whole-image resize plus batch 24; native patches '
        'keep the thin CREMI/CrackTree structures that downsampling to 256 '
        'or 512 would destroy.',
        'A one-class (BCE + foreground Dice) head is used; upstream '
        'train.py defaults to two classes (CE + multiclass Dice), but its '
        '1-class branch is the official binary mode.',
        'Grayscale protocol images are converted to RGB because the '
        'vendored model expects three input channels; upstream keeps 2-D '
        'images single-channel and would assert.',
        'Validation and prediction use the official native-resolution '
        'preprocessing (the model is fully convolutional); checkpoint '
        'selection is the 6-image val Dice.  Upstream saves every epoch '
        'and evaluates a random 10% split.',
        'Official DataLoader workers=os.cpu_count(), shuffle=True and a '
        'global RNG; this port uses workers=0 and a per-epoch seeded RNG, '
        'so data order and floating-point accumulation are not bitwise '
        'identical.',
    )

    def __init__(self):
        super().__init__()
        # The 18 training images are decoded once instead of once per crop.
        self._pair_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add_args(self, parser):
        parser.add_argument(
            '--scale', type=float, default=self.scale,
            help='Per-image resampling factor before patch cropping '
                 '(default 1.0 = native)')
        parser.add_argument(
            '--unet-patch-size', type=int, default=self.patch_size,
            help='Training patch size (native-resolution 512x512)')
        parser.add_argument(
            '--unet-crops-per-image', type=int, default=self.crops_per_image,
            help='Random patches per training image per epoch')
        parser.add_argument(
            '--unet-fg-crop-prob', type=float, default=self.fg_crop_prob,
            help='Probability that a training patch is centred on a '
                 'foreground pixel')
        parser.add_argument(
            '--unet-photometric', type=int, choices=[0, 1],
            default=int(self.photometric),
            help='Apply mild brightness/contrast/gamma jitter to patches')

    def build_model(self, args) -> nn.Module:
        self.scale = float(getattr(args, 'scale', self.scale))
        self.patch_size = max(
            64, int(getattr(args, 'unet_patch_size', self.patch_size)))
        self.crops_per_image = max(
            1, int(getattr(args, 'unet_crops_per_image',
                           self.crops_per_image)))
        self.fg_crop_prob = min(1.0, max(
            0.0, float(getattr(args, 'unet_fg_crop_prob',
                               self.fg_crop_prob))))
        self.photometric = bool(
            int(getattr(args, 'unet_photometric', int(self.photometric))))
        return UNet(n_channels=3, n_classes=1)

    def make_train_dataset(self, samples, args):
        repeats = max(1, int(getattr(
            args, 'unet_crops_per_image', self.crops_per_image)))
        if repeats == 1:
            return None
        return _RepeatedSampleDataset(samples, repeats)

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    def make_scheduler(self, optimizer, args):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(getattr(args, 'epochs', 1)), 1))

    def step_scheduler(self, scheduler, metrics: dict[str, float]) -> None:
        if scheduler is None:
            return
        if isinstance(scheduler,
                      torch.optim.lr_scheduler.ReduceLROnPlateau):
            key = (self.args.selection if self.args is not None
                   else self.selection)
            scheduler.step(metrics[key])
        else:
            # Cosine / other epoch-based schedules.
            scheduler.step()

    # ------------------------------------------------------------------
    # Official preprocessing (validation / prediction)
    # ------------------------------------------------------------------
    def _load_image(self, sample):
        """Official BasicDataset.preprocess for the image."""
        with Image.open(sample.image_path) as image_file:
            image = image_file.convert('RGB')
            width, height = image.size
            new_w = int(self.scale * width)
            new_h = int(self.scale * height)
            if new_w <= 0 or new_h <= 0:
                raise RuntimeError(
                    f'{self.id}: scale={self.scale} produces an empty '
                    f'image for {sample.image_path} ({width}x{height})')
            image = image.resize((new_w, new_h), Image.BICUBIC)
            image_np = np.asarray(image, dtype=np.float32).copy()
        if (image_np > 1).any():  # official uint8 -> [0, 1]
            image_np /= 255.0
        return image_np, (width, height)

    # ------------------------------------------------------------------
    # Task-adapted patch training
    # ------------------------------------------------------------------
    def _read_pair(self, sample):
        """Decode an image/mask pair once and cache the uint8 arrays."""
        key = str(sample.image_path)
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached
        with Image.open(sample.image_path) as handle:
            image = handle.convert('RGB')
            image_size = image.size
            image_np = np.asarray(image).copy()
        with Image.open(sample.mask_path) as handle:
            mask = handle.convert('L')
            mask_size = mask.size
            mask_np = np.asarray(mask).copy()
        if image_size != mask_size:
            raise RuntimeError(
                f'{self.id}: image/mask size mismatch for '
                f'{sample.image_path}: {image_size} vs {mask_size}')
        entry = (image_np, mask_np)
        # 18 training images: at most ~100 MB of uint8 pixels for the
        # largest protocol image, and decoding then happens once.
        if len(self._pair_cache) < 64:
            self._pair_cache[key] = entry
        return entry

    def _pad_to_patch(self, image: np.ndarray, mask: np.ndarray):
        """Reflect-pad (edge fallback) so both spatial sides reach patch."""
        patch = self.patch_size
        height, width = mask.shape
        pad_h = max(0, patch - height)
        pad_w = max(0, patch - width)
        if not pad_h and not pad_w:
            return image, mask
        mode = 'reflect' if (pad_h < height and pad_w < width) else 'edge'
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode=mode)
        return image, mask

    def _crop_origin(self, mask: np.ndarray,
                     rng: random.Random) -> tuple[int, int]:
        """Random patch origin, foreground-centred with ``fg_crop_prob``."""
        patch = self.patch_size
        height, width = mask.shape
        if height <= patch and width <= patch:
            return 0, 0
        if rng.random() < self.fg_crop_prob:
            ys, xs = np.nonzero(mask)
            if len(ys):
                # Place one foreground pixel at a uniformly random position
                # inside the patch; clipping keeps the patch in bounds.
                index = rng.randrange(len(ys))
                cy = int(ys[index])
                cx = int(xs[index])
                top = min(max(0, cy - rng.randrange(patch)),
                          height - patch)
                left = min(max(0, cx - rng.randrange(patch)),
                           width - patch)
                return top, left
        top = rng.randint(0, height - patch)
        left = rng.randint(0, width - patch)
        return top, left

    def _train_pair(self, sample, rng: random.Random):
        image_np, mask_np = self._read_pair(sample)
        if self.scale != 1.0:
            height, width = mask_np.shape
            new_w = max(1, int(self.scale * width))
            new_h = max(1, int(self.scale * height))
            image_np = np.asarray(Image.fromarray(image_np).resize(
                (new_w, new_h), Image.BICUBIC))
            mask_np = np.asarray(Image.fromarray(mask_np).resize(
                (new_w, new_h), Image.NEAREST))
        mask = mask_np > 127
        image, mask = self._pad_to_patch(image_np, mask)
        top, left = self._crop_origin(mask, rng)
        patch = self.patch_size
        image = image[top:top + patch, left:left + patch]
        mask = mask[top:top + patch, left:left + patch]

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32)
        if rng.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if rng.random() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
        k = rng.randrange(4)
        if k:
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        if self.photometric and rng.random() < 0.5:
            gain = rng.uniform(0.85, 1.15)
            bias = rng.uniform(-0.06, 0.06)
            image = np.clip(image * gain + bias, 0.0, 1.0)
        if self.photometric and rng.random() < 0.25:
            image = np.clip(image, 0.0, 1.0) ** rng.uniform(0.8, 1.25)

        image_t = torch.from_numpy(image).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return image_t, mask_t

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image, mask = self._train_pair(sample, rng)
            images.append(image)
            masks.append(mask)
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    def train_step(self, model, batch):
        images, masks = batch
        logits = model(images)
        return (nn.BCEWithLogitsLoss()(logits, masks) +
                dice_loss(logits, masks))

    def prepare_eval_sample(self, sample, device):
        # predict.py preprocesses the image only; the GT mask is opened
        # later for the output size.
        image_np, _ = self._load_image(sample)
        tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        return tensor.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        # Upstream moves the logits to CPU, upsamples them, and only then
        # applies sigmoid; interpolating probabilities changes boundary
        # decisions, and CPU bilinear matches predict.py exactly.
        logits = model(prepared).float().cpu()
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        logits = F.interpolate(logits, size=(height, width),
                               mode='bilinear', align_corners=False)
        return (torch.sigmoid(logits)[0, 0].numpy() > 0.5)

    def save_config(self, args):
        return {
            'scale': float(self.scale),
            'patch_size': int(self.patch_size),
            'crops_per_image': int(self.crops_per_image),
            'fg_crop_prob': float(self.fg_crop_prob),
            'photometric': bool(self.photometric),
        }

    def restore_config(self, args, config):
        if 'scale' in config:
            self.scale = float(config['scale'])
            if hasattr(args, 'scale'):
                args.scale = self.scale
        if 'patch_size' in config:
            self.patch_size = max(64, int(config['patch_size']))
            if hasattr(args, 'unet_patch_size'):
                args.unet_patch_size = self.patch_size
        if 'crops_per_image' in config:
            self.crops_per_image = max(1, int(config['crops_per_image']))
            if hasattr(args, 'unet_crops_per_image'):
                args.unet_crops_per_image = self.crops_per_image
        if 'fg_crop_prob' in config:
            self.fg_crop_prob = float(config['fg_crop_prob'])
            if hasattr(args, 'unet_fg_crop_prob'):
                args.unet_fg_crop_prob = self.fg_crop_prob
        if 'photometric' in config:
            self.photometric = bool(config['photometric'])
            if hasattr(args, 'unet_photometric'):
                args.unet_photometric = int(self.photometric)
