"""U-Net baseline: vendored milesial/PyTorch-UNet architecture.

The official milesial architecture, BCE+Dice loss, per-image scale
preprocessing and logit-space bilinear upsampling are preserved.  The
literal official training recipe (RMSprop lr=1e-5, ReduceLROnPlateau,
scale=0.5, no augmentation) is not usable on the 18-image 3-shot split:
its plateau collapses the LR to 1e-8 by epoch 19.  This port therefore
uses a documented 3-shot recipe: scale=1.0 (native resolution), batch 1,
AdamW lr=1e-3/wd=1e-4, cosine over 800 epochs, flips/90-degree
rotations, gradient clipping 1.0.  Label the row "3-shot U-Net (official
architecture; adapted recipe)".
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

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


class UNetAdapter(BaselineAdapter):
    id = 'unet'
    display_name = 'U-Net'
    official_source = 'milesial/PyTorch-UNet (self-contained copy)'
    official_entrypoint = 'https://github.com/milesial/Pytorch-UNet'
    # Official preprocessing is a per-image scale factor, so the model input
    # is variable-size; 0 marks that this adapter does not use a fixed size.
    input_size = 0
    scale = 1.0
    # 3-shot adapted defaults; the official RMSprop recipe is 1e-5/1e-8.
    official_lr = 1e-3
    official_weight_decay = 1e-4
    train_batch_size = 1
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
        'prevents further reductions).  The 3-shot recipe therefore uses '
        'scale=1.0 native resolution, batch 1, AdamW lr=1e-3/wd=1e-4, '
        'cosine over 800 epochs and flips/90-degree rotations.  Label this '
        'row "3-shot U-Net (official architecture; adapted recipe)", not '
        'an official U-Net reproduction.',
        'A one-class (BCE + foreground Dice) head is used; upstream '
        'train.py defaults to two classes (CE + multiclass Dice), but its '
        '1-class branch is the official binary mode.',
        'Grayscale protocol images are converted to RGB because the '
        'vendored model expects three input channels; upstream keeps 2-D '
        'images single-channel and would assert.',
        'Checkpoint selected on the 6-image val split by Dice; upstream '
        'saves every epoch and evaluates a random 10% split.',
        'Validation Dice is computed at GT resolution by the shared '
        'evaluator; upstream evaluates at the scaled input resolution.',
        'Official DataLoader workers=os.cpu_count(), shuffle=True and a '
        'global RNG; this port uses workers=0 and a per-epoch seeded RNG, '
        'so data order and floating-point accumulation are not bitwise '
        'identical.',
    )

    def add_args(self, parser):
        parser.add_argument(
            '--scale', type=float, default=self.scale,
            help='Per-image resampling factor (default 1.0 = native; use '
                 '0.5 if a 24 GiB card cannot fit the largest image)')

    def build_model(self, args) -> nn.Module:
        self.scale = float(getattr(args, 'scale', self.scale))
        return UNet(n_channels=3, n_classes=1)

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

    def _load_mask(self, sample, image_size):
        """Official BasicDataset.preprocess for the mask (training only)."""
        with Image.open(sample.mask_path) as mask_file:
            mask = mask_file.convert('L')
            if mask.size != image_size:
                raise RuntimeError(
                    f'{self.id}: image/mask size mismatch for '
                    f'{sample.image_path}: {image_size} vs {mask.size}')
            new_w = int(self.scale * mask.size[0])
            new_h = int(self.scale * mask.size[1])
            mask = mask.resize((new_w, new_h), Image.NEAREST)
        # Every protocol mask is {0, 255}; this is equivalent to the
        # official sorted mask_values mapping and keeps the binary head.
        mask_np = (np.asarray(mask, dtype=np.float32).copy() > 127.0)
        return mask_np.astype(np.float32)

    def _resize_pair(self, sample):
        """Official BasicDataset.preprocess at ``scale`` (image/mask pair)."""
        image_np, image_size = self._load_image(sample)
        mask_np = self._load_mask(sample, image_size)
        return image_np, mask_np

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image_np, mask_np = self._resize_pair(sample)
            image = torch.from_numpy(image_np).permute(2, 0, 1)
            mask = torch.from_numpy(mask_np).unsqueeze(0)
            # Joint 3-shot augmentation (the official trainer has none).
            if rng.random() < 0.5:
                image = torch.flip(image, [-1])
                mask = torch.flip(mask, [-1])
            if rng.random() < 0.5:
                image = torch.flip(image, [-2])
                mask = torch.flip(mask, [-2])
            k = rng.randrange(4)
            if k:
                image = torch.rot90(image, k, dims=[-2, -1])
                mask = torch.rot90(mask, k, dims=[-2, -1])
            images.append(image)
            masks.append(mask)
        shapes = {tuple(tensor.shape) for tensor in images}
        if len(shapes) > 1:
            raise RuntimeError(
                f'{self.id}: scale={self.scale} produces variable input '
                f'sizes ({sorted(shapes)}); use --batch-size 1')
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
        return {'scale': float(self.scale)}

    def restore_config(self, args, config):
        if 'scale' in config:
            self.scale = float(config['scale'])
            if hasattr(args, 'scale'):
                args.scale = self.scale
