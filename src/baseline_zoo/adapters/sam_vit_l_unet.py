"""SAM ViT-L + U-Net decoder (project extension baseline)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from baseline_zoo.data import load_rgb
from baseline_zoo.models.sam_vit_l_unet import SAMViTLUNet
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    denom = prob.sum() + target.sum()
    return 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)


class SAMViTLUNetAdapter(BaselineAdapter):
    id = 'sam_vit_l_unet'
    display_name = 'SAM ViT-L + U-Net'
    official_source = 'project extension (not a paper baseline)'
    official_entrypoint = 'previous project adapter, self-contained now'
    input_size = 1024
    official_epochs = 200
    official_lr = 1e-3
    official_weight_decay = 1e-4
    train_batch_size = 4
    selection = 'dice'
    deviations = (
        'This is the project SAM+U-Net probe, not one of the paper ten '
        'baselines; the decoder is unchanged from the previous adapter.',
        'Frozen ViT features are recomputed per batch instead of cached once '
        'per split; the math is identical but training is slower.',
        'Checkpoint selection uses the unified val Dice; the previous '
        'adapter also selected by validation Dice.',
        'Predictions are restored to the original GT resolution.',
    )

    def add_args(self, parser):
        parser.add_argument(
            '--sam-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_l_0b3195.pth')
        parser.add_argument('--input-size', type=int, default=1024)

    def build_model(self, args):
        if not args.sam_checkpoint.is_file():
            raise SystemExit(
                f'missing SAM ViT-L checkpoint: {args.sam_checkpoint}')
        self.input_size = args.input_size
        return SAMViTLUNet(str(args.sam_checkpoint))

    def configure_trainable(self, model):
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in model.decoder.parameters():
            parameter.requires_grad_(True)

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.decoder.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    def make_scheduler(self, optimizer, args):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=1e-6)

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image = load_rgb(sample.image_path).resize(
                (self.input_size, self.input_size), Image.BICUBIC)
            mask = Image.open(sample.mask_path).convert('L').resize(
                (self.input_size, self.input_size), Image.NEAREST)
            image_t = torch.from_numpy(
                np.asarray(image, dtype=np.float32).copy()
            ).permute(2, 0, 1) / 255.0
            mask_t = torch.from_numpy(
                (np.asarray(mask) > 127).astype(np.float32).copy()
            ).unsqueeze(0)
            if rng.random() < 0.5:
                image_t = image_t.flip(-1)
                mask_t = mask_t.flip(-1)
            if rng.random() < 0.5:
                image_t = image_t.flip(-2)
                mask_t = mask_t.flip(-2)
            k = rng.randrange(4)
            if k:
                image_t = torch.rot90(image_t, k, [-2, -1])
                mask_t = torch.rot90(mask_t, k, [-2, -1])
            images.append(image_t.contiguous())
            masks.append(mask_t.contiguous())
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    def train_step(self, model, batch):
        images, masks = batch
        logits = model(images)
        return (nn.BCEWithLogitsLoss()(logits, masks) +
                dice_loss(logits, masks))

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size), Image.BICUBIC)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        prob = torch.sigmoid(model(prepared))
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        prob = F.interpolate(prob, size=(height, width),
                             mode='bilinear', align_corners=False)
        return (prob[0, 0].cpu().numpy() > 0.5)
