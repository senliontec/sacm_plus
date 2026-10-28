"""Conv-LoRA baseline adapter (ICLR 2024).

Official reference: AutoGluon's semantic-segmentation SAM model with
``optim.peft=conv_lora``.  This port is a binary, prompt-free adaptation:
the image is resized to SAM's 1024 square, the frozen prompt encoder emits
the constant no-mask embedding, the mask decoder is fine-tuned, and
Conv-LoRA is injected into the image-encoder qkv projections.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baseline_zoo.data import load_mask, load_rgb
from baseline_zoo.models.convlora import StructureLoss, build_convlora
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256


class ConvLoRAAdapter(BaselineAdapter):
    id = 'convlora'
    display_name = 'Conv-LoRA'
    official_source = 'Conv-LoRA / AutoGluon (self-contained port)'
    official_entrypoint = (
        'https://github.com/autogluon/autogluon/tree/master/'
        'examples/automm/Conv-LoRA')
    input_size = 1024
    official_lr = 1e-4
    official_weight_decay = 1e-4
    train_batch_size = 1
    accumulation_steps = 4
    default_grad_clip = 1.0
    selection = 'dice'
    deviations = (
        'The official AutoGluon recipe trains on full benchmark datasets '
        'with effective batch 4 and 30 epochs (100 epochs in its low-data '
        'study). This port uses the shared 18-image 3-shot split, batch 1 '
        'with 4-step gradient accumulation, and the project epoch budget.',
        'The paper uses SAM ViT-H with Conv-LoRA on q/k/v self-attention '
        'and a one-token binary mask decoder. This port keeps that model '
        'and prompt-free constant no-mask prompt. The paper says the mask '
        'decoder is fully fine-tuned and reports 4.02M trainable params '
        '(3.51M decoder + 0.51M Conv-LoRA); the released AutoGluon PEFT '
        'optimizer filter would train only .*lora_* parameters, so this '
        'port follows the paper and trains the decoder, keeping only the '
        'IoU head frozen as declared in the official SAM config.',
        'Images and GT masks are resized to 1024x1024 square with bicubic/'
        'nearest interpolation; the only training augmentation is random '
        'horizontal flip. The official AutoGluon config normalizes with '
        'ImageNet statistics (image_norm: imagenet), so Meta SAM pixel '
        'mean/std are deliberately not used. Predictions are bilinearly '
        'resized from 1024 to the original GT resolution.',
        'The MoE load-balancing loss (official loss coefficient 1e-2 per '
        'layer, summed with weight 1.0 for binary segmentation) is kept. '
        'Structure loss is the official weighted BCE + weighted IoU loss.',
        'The paper states Adam, lr 1e-4 and weight decay 1e-4; this port '
        'uses those values (AutoGluon defaults are AdamW, wd 1e-3). The '
        'official cosine schedule with 10% warmup and gradient clipping '
        '1.0 is applied per epoch because the shared runtime steps the '
        'scheduler once per epoch; AutoGluon steps it per iteration. '
        'AutoGluon layerwise LR decay (0.9) and greedy-soup checkpoint '
        'averaging are not used; checkpoint selection is validation Dice.',
        'No official Conv-LoRA weights are present. Training starts from '
        'the official SAM ViT-H checkpoint; the local file is required so '
        'the offline server never downloads weights.',
    )

    def __init__(self):
        super().__init__()
        self._criterion = StructureLoss()
        self._sam_sha: str | None = None
        self._encoder_weights: Path | None = None

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=1024,
                            help='SAM input size (fixed at 1024)')
        parser.add_argument(
            '--convlora-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_h_4b8939.pth',
            help='local Meta SAM checkpoint (ViT-H by default)')
        parser.add_argument(
            '--convlora-model-type', default='vit_h',
            choices=['vit_h', 'vit_l', 'vit_b'],
            help='SAM backbone matching the checkpoint')
        parser.add_argument('--convlora-rank', type=int, default=3)
        parser.add_argument('--convlora-alpha', type=int, default=8)
        parser.add_argument('--convlora-experts', type=int, default=8)
        parser.add_argument('--convlora-moe-weight', type=float, default=1.0)

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', 1024))
        if self.input_size != 1024:
            raise SystemExit('Conv-LoRA requires --input-size 1024')
        checkpoint = Path(getattr(
            args, 'convlora_checkpoint',
            REPO_ROOT / 'checkpoints' / 'sam_vit_h_4b8939.pth'))
        if not checkpoint.is_file():
            raise SystemExit(
                f'missing SAM checkpoint: {checkpoint}\n'
                'Upload the official SAM ViT-H file to checkpoints/.')
        self._sam_sha = file_sha256(checkpoint)
        self._encoder_weights = checkpoint
        return build_convlora(
            checkpoint=checkpoint,
            model_type=str(getattr(args, 'convlora_model_type', 'vit_h')),
            image_size=self.input_size,
            rank=int(getattr(args, 'convlora_rank', 3)),
            alpha=int(getattr(args, 'convlora_alpha', 8)),
            experts=int(getattr(args, 'convlora_experts', 8)),
            moe_weight=float(getattr(args, 'convlora_moe_weight', 1.0)),
        )

    def load_pretrained(self, model, args) -> None:
        print('loaded SAM base weights and injected Conv-LoRA')

    def configure_trainable(self, model) -> None:
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        if trainable == 0:
            raise RuntimeError('Conv-LoRA has no trainable parameters')
        if not any('lora_A' in name for name, p in model.named_parameters()
                   if p.requires_grad):
            raise RuntimeError('Conv-LoRA layers are not trainable')

    def make_optimizer(self, model, args):
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def make_scheduler(self, optimizer, args):
        # AutoGluon's defaults (used by the released Conv-LoRA example) are
        # cosine decay with ``warmup_steps=0.1`` (a fraction of all steps).
        # The shared runtime steps once per epoch, so the fraction is
        # applied to epochs; this is a coarse version of the official
        # per-iteration Lightning schedule.
        epochs = max(1, int(args.epochs))
        warmup = int(0.1 * epochs)

        def factor(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    def _pair(self, sample, rng: random.Random):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size), Image.BICUBIC)
        mask = load_mask(sample.mask_path)
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (self.input_size, self.input_size), Image.NEAREST))
        mask = (mask > 127).astype(np.float32)
        image_np = np.asarray(image, dtype=np.float32).copy()
        if rng.random() < 0.5:
            image_np = np.flip(image_np, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        image_t = torch.from_numpy(image_np).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask)[None]
        return image_t, mask_t

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image, mask = self._pair(sample, rng)
            images.append(image)
            masks.append(mask)
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    def train_step(self, model, batch):
        images, masks = batch
        logits, moe_loss = model(images)
        loss = self._criterion(logits, masks)
        weight = float(getattr(self.args, 'convlora_moe_weight', 1.0))
        return loss + weight * moe_loss

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size), Image.BICUBIC)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1).unsqueeze(0)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return tensor.to(device), (int(width), int(height))

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        image, (width, height) = prepared
        logits, _ = model(image)
        logits = F.interpolate(
            logits, size=(height, width), mode='bilinear',
            align_corners=False)
        return torch.sigmoid(logits[0, 0]).cpu().numpy() > 0.5

    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'model_type': str(getattr(args, 'convlora_model_type', 'vit_h')),
            'rank': int(getattr(args, 'convlora_rank', 3)),
            'alpha': int(getattr(args, 'convlora_alpha', 8)),
            'experts': int(getattr(args, 'convlora_experts', 8)),
            'moe_weight': float(
                getattr(args, 'convlora_moe_weight', 1.0)),
            'convlora_checkpoint': str(
                self._encoder_weights
                or getattr(args, 'convlora_checkpoint', '')),
            'sam_checkpoint_sha256': self._sam_sha,
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
            args.input_size = self.input_size
        if 'model_type' in config:
            args.convlora_model_type = str(config['model_type'])
        if 'rank' in config:
            args.convlora_rank = int(config['rank'])
        if 'alpha' in config:
            args.convlora_alpha = int(config['alpha'])
        if 'experts' in config:
            args.convlora_experts = int(config['experts'])
        if 'moe_weight' in config:
            args.convlora_moe_weight = float(config['moe_weight'])
        if 'convlora_checkpoint' in config:
            args.convlora_checkpoint = Path(config['convlora_checkpoint'])
