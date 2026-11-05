"""U-Net baseline: vendored milesial/PyTorch-UNet architecture.

The model graph and BCE+Dice objective are preserved from
milesial/PyTorch-UNet.  The deployed baseline is a task adaptation for the
pooled 18-image / 6-domain 3-shot protocol:

* full native-resolution images, batch 1, so training and validation see
  the same spatial distribution and thin structures are not downsampled;
* InstanceNorm instead of BatchNorm.  With pooled multi-domain training and
  full-image evaluation, BatchNorm running statistics are a cross-domain
  average that does not represent any individual image; local diagnostics
  showed eval-mode predictions collapsing to all-background while batch-stat
  predictions were sensible.  InstanceNorm normalizes each sample and
  removes that train/eval gap;
* joint flips / 90-degree rotations plus mild photometric jitter,
* AdamW lr 3e-4 / wd 1e-4, warmup + cosine, gradient clipping 1.0.

`--unet-norm batch --unet-train-mode patch --batch-size 8 --lr 1e-3
--epochs 800 --warmup-epochs 0` restores the earlier official-BN patch
recipe.  Label the deployed row "3-shot U-Net (official architecture;
task-adapted InstanceNorm/full-image recipe)", not an official U-Net
reproduction.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from baseline_zoo.data import load_mask
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


def _replace_batchnorm(model: nn.Module, kind: str) -> nn.Module:
    """Swap the vendored BatchNorm layers for task-adapted normalization.

    The official model file under ``models/unet`` stays byte-identical; the
    adapter performs the replacement at construction time.  InstanceNorm is
    per-sample and therefore has no cross-domain running statistics.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            if kind == 'instance':
                replacement = nn.InstanceNorm2d(
                    child.num_features, affine=True,
                    track_running_stats=False)
            elif kind == 'group':
                replacement = nn.GroupNorm(
                    min(8, child.num_features), child.num_features)
            else:
                raise ValueError(f'unknown norm kind: {kind!r}')
            setattr(model, name, replacement)
        else:
            _replace_batchnorm(child, kind)
    return model


class UNetAdapter(BaselineAdapter):
    id = 'unet'
    display_name = 'U-Net'
    official_source = 'milesial/PyTorch-UNet (self-contained copy)'
    official_entrypoint = 'https://github.com/milesial/Pytorch-UNet'
    # Official preprocessing is a per-image scale factor, so the model input
    # is variable-size; 0 marks that this adapter does not use a fixed size.
    input_size = 0
    scale = 1.0
    # Task-adapted 3-shot recipe; the official RMSprop recipe is 1e-5.
    official_lr = 3e-4
    official_weight_decay = 1e-4
    train_mode = 'full'
    norm = 'instance'
    warmup_epochs = 20
    patch_size = 512
    crops_per_image = 8
    fg_crop_prob = 0.5
    photometric = True
    train_batch_size = 1
    default_grad_clip = 1.0
    selection = 'dice'
    # The shared runtime steps the scheduler after validation; the override
    # below handles cosine and any future metric-based scheduler.
    scheduler_after_validation = True
    deviations = (
        'Official milesial architecture and BCE+Dice loss are preserved. '
        'The deployed task adaptation trains full native-resolution images '
        'with InstanceNorm, batch 1, AdamW lr=3e-4/wd=1e-4, a 20-epoch '
        'warmup followed by cosine over the configured budget, and '
        'flips/rot90/mild photometric jitter.',
        'BatchNorm is not used in the deployed row.  Under pooled '
        'six-domain 3-shot training, its running statistics are a '
        'cross-domain average: local diagnostics showed eval-mode '
        'predictions collapsing to all-background while batch-stat '
        'predictions on the same weights were sensible.  InstanceNorm '
        'normalizes each sample and removes the train/eval gap.  This is '
        'an architectural adaptation, not the official BN model.',
        'The official full-image recipe uses scale=0.5, batch 1, RMSprop '
        '1e-5 and 5 epochs; the literal ReduceLROnPlateau recipe collapsed '
        'the LR to 1e-8 by epoch 19 on the 18-image split.  This port uses '
        'scale=1.0 to preserve thin curvilinear structures and a warmup + '
        'cosine schedule over 300 epochs.  Label the row "3-shot U-Net '
        '(official architecture; task-adapted InstanceNorm/full-image '
        'recipe)", not an official U-Net reproduction.',
        'Restore the earlier official-BN patch ablation with '
        '`--unet-norm batch --unet-train-mode patch --batch-size 8 '
        '--lr 1e-3 --epochs 800 --warmup-epochs 0`.  That mode retains '
        'native 512x512 patches, half foreground-centred, eight patches per '
        'training image per epoch and AdamW lr=1e-3; its validation curve is '
        'expected to be less stable because of the BN running-statistics '
        'issue above.',
        'A one-class (BCE + foreground Dice) head is used; upstream '
        'train.py defaults to two classes (CE + multiclass Dice), but its '
        '1-class branch is the official binary mode.',
        'Grayscale protocol images are converted to RGB because the '
        'vendored model expects three input channels; upstream keeps 2-D '
        'images single-channel and would assert.',
        'Validation and prediction use scale=1.0 (native resolution) so '
        'thin structures are preserved; upstream train.py/predict.py '
        'default to scale=0.5.  Checkpoint selection is the 6-image val '
        'Dice; upstream evaluates a random 10% split.',
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
            '--unet-train-mode', choices=['full', 'patch'],
            default=self.train_mode,
            help='full: one native-resolution image per batch; patch: '
                 'random 512x512 crops with repeated samples')
        parser.add_argument(
            '--unet-norm', choices=['instance', 'batch', 'group'],
            default=self.norm,
            help='normalization: instance (task default), batch (official '
                 'BN ablation) or group')
        parser.add_argument(
            '--scale', type=float, default=self.scale,
            help='Per-image resampling factor before training '
                 '(default 1.0 = native)')
        parser.add_argument(
            '--unet-patch-size', type=int, default=self.patch_size,
            help='Training patch size (patch mode only)')
        parser.add_argument(
            '--unet-crops-per-image', type=int, default=self.crops_per_image,
            help='Random patches per training image per epoch (patch mode)')
        parser.add_argument(
            '--unet-fg-crop-prob', type=float, default=self.fg_crop_prob,
            help='Probability that a patch is centred on foreground '
                 '(patch mode)')
        parser.add_argument(
            '--unet-photometric', type=int, choices=[0, 1],
            default=int(self.photometric),
            help='Apply mild brightness/contrast/gamma jitter')

    def build_model(self, args) -> nn.Module:
        self.scale = float(getattr(args, 'scale', self.scale))
        if self.scale <= 0:
            raise ValueError(
                f'{self.id}: --scale must be > 0, got {self.scale}')
        self.train_mode = str(getattr(
            args, 'unet_train_mode', self.train_mode)).lower()
        if self.train_mode not in ('full', 'patch'):
            raise ValueError(
                f'{self.id}: unknown --unet-train-mode '
                f'{self.train_mode!r}')
        self.norm = str(getattr(args, 'unet_norm', self.norm)).lower()
        if self.norm not in ('instance', 'batch', 'group'):
            raise ValueError(
                f'{self.id}: unknown --unet-norm {self.norm!r}')
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
        model = UNet(n_channels=3, n_classes=1)
        if self.norm != 'batch':
            _replace_batchnorm(model, self.norm)
        return model

    def make_train_dataset(self, samples, args):
        if self.train_mode == 'full':
            # One sample per optimizer step; full images have different
            # sizes and are collated as a batch of one.
            return None
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
        epochs = max(int(getattr(args, 'epochs', 1)), 1)
        warmup_raw = getattr(args, 'warmup_epochs', None)
        warmup = (self.warmup_epochs if warmup_raw is None
                  else int(warmup_raw))
        warmup = max(0, min(warmup, epochs - 1))

        def lr_lambda(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                return float(epoch + 1) / float(warmup)
            progress = (epoch - warmup) / max(1, epochs - warmup)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)

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
    # Task-adapted training
    # ------------------------------------------------------------------
    def _read_pair(self, sample):
        """Decode an image/mask pair once and cache the arrays.

        The mask goes through ``load_mask`` rather than a local ``> 127``
        test: some source masks are literal 0/1 PNGs, for which ``> 127``
        would silently produce an all-background target.
        """
        key = str(sample.image_path)
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached
        with Image.open(sample.image_path) as handle:
            image = handle.convert('RGB')
            image_size = image.size
            image_np = np.asarray(image).copy()
        mask = load_mask(sample.mask_path)
        mask_size = (mask.shape[1], mask.shape[0])
        if image_size != mask_size:
            raise RuntimeError(
                f'{self.id}: image/mask size mismatch for '
                f'{sample.image_path}: {image_size} vs {mask_size}')
        entry = (image_np, mask)
        # 18 training images: at most ~100 MB of uint8 pixels for the
        # largest protocol image, and decoding then happens once.
        if len(self._pair_cache) < 64:
            self._pair_cache[key] = entry
        return entry

    def _pad_to_patch(self, image: np.ndarray, mask: np.ndarray):
        """Pad small images with a crop margin, then reflect/edge-pad.

        Padding only up to ``patch`` makes ``_crop_origin`` return (0, 0)
        for every crop when both sides are <= patch (DCA1 300x300 and CORN
        384x384), so the documented 50% foreground-centred random-patch
        recipe never runs for those domains.  Keep one extra margin on a
        side that is smaller than the patch so a real random crop exists.
        """
        patch = self.patch_size
        height, width = mask.shape
        # 32 px keeps reflect-padding valid for a 300x300 image
        # (pad=276 < 300) while giving 64 px of crop freedom.
        margin = max(16, patch // 16)
        target_h = patch + 2 * margin if height < patch else height
        target_w = patch + 2 * margin if width < patch else width
        pad_h = target_h - height
        pad_w = target_w - width
        if not pad_h and not pad_w:
            return image, mask
        mode = 'reflect' if (pad_h < height and pad_w < width) else 'edge'
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        image = np.pad(
            image, ((top, bottom), (left, right), (0, 0)), mode=mode)
        mask = np.pad(mask, ((top, bottom), (left, right)), mode=mode)
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

    def _augment(self, image: np.ndarray, mask: np.ndarray,
                 rng: random.Random):
        """Joint flips/90-degree rotations and mild photometric jitter."""
        image = image.astype(np.float32, copy=False)
        mask = mask.astype(np.float32, copy=False)
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
        return image, mask

    def _train_pair(self, sample, rng: random.Random):
        """Random native patch for the legacy patch-mode ablation."""
        image_np, mask = self._read_pair(sample)
        if self.scale != 1.0:
            height, width = mask.shape
            new_w = max(1, int(self.scale * width))
            new_h = max(1, int(self.scale * height))
            image_np = np.asarray(Image.fromarray(image_np).resize(
                (new_w, new_h), Image.BICUBIC))
            mask = np.asarray(Image.fromarray(
                mask.astype(np.uint8) * 255).resize(
                    (new_w, new_h), Image.NEAREST)) > 127
        image, mask = self._pad_to_patch(image_np, mask)
        top, left = self._crop_origin(mask, rng)
        patch = self.patch_size
        image = image[top:top + patch, left:left + patch]
        mask = mask[top:top + patch, left:left + patch]

        image = image.astype(np.float32) / 255.0
        image, mask = self._augment(image, mask, rng)
        image_t = torch.from_numpy(image).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return image_t, mask_t

    def _train_full_pair(self, sample, rng: random.Random):
        """One full native-resolution image and its mask."""
        image_np, _ = self._load_image(sample)
        mask = load_mask(sample.mask_path)
        target_shape = image_np.shape[:2]
        if mask.shape != target_shape:
            # Covers --scale != 1 and any source image/mask size mismatch.
            mask = np.asarray(Image.fromarray(
                mask.astype(np.uint8) * 255).resize(
                    (target_shape[1], target_shape[0]),
                    Image.NEAREST)) > 127
        image, mask = self._augment(image_np, mask, rng)
        image_t = torch.from_numpy(image).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return image_t, mask_t

    def prepare_train_batch(self, samples, rng: random.Random):
        if self.train_mode == 'full':
            # Native images have different sizes; batch 1 keeps collation
            # and activation memory bounded.  The CLI default is 1.
            if len(samples) != 1:
                raise ValueError(
                    f'{self.id}: full-image training requires '
                    f'--batch-size 1, got a batch of {len(samples)}')
            image, mask = self._train_full_pair(samples[0], rng)
            return (image.unsqueeze(0).to(self.device),
                    mask.unsqueeze(0).to(self.device))
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
            'train_mode': str(self.train_mode),
            'norm': str(self.norm),
            'scale': float(self.scale),
            'patch_size': int(self.patch_size),
            'crops_per_image': int(self.crops_per_image),
            'fg_crop_prob': float(self.fg_crop_prob),
            'photometric': bool(self.photometric),
        }

    def restore_config(self, args, config):
        # Checkpoints written before this adaptation have no train_mode/norm
        # keys; they were trained with the BatchNorm patch recipe, so infer
        # that architecture when loading them for evaluation or resume.
        if 'train_mode' not in config and 'norm' not in config:
            self.train_mode = 'patch'
            self.norm = 'batch'
            if hasattr(args, 'unet_train_mode'):
                args.unet_train_mode = self.train_mode
            if hasattr(args, 'unet_norm'):
                args.unet_norm = self.norm
        if 'train_mode' in config:
            mode = str(config['train_mode']).lower()
            if mode not in ('full', 'patch'):
                raise ValueError(
                    f'{self.id}: unknown checkpoint train_mode {mode!r}')
            self.train_mode = mode
            if hasattr(args, 'unet_train_mode'):
                args.unet_train_mode = mode
        if 'norm' in config:
            kind = str(config['norm']).lower()
            if kind not in ('instance', 'batch', 'group'):
                raise ValueError(
                    f'{self.id}: unknown checkpoint norm {kind!r}')
            self.norm = kind
            if hasattr(args, 'unet_norm'):
                args.unet_norm = kind
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
