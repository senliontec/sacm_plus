"""CurvField adapter: Conv-LoRA plus a shared structure field.

Project extension, not an official Conv-LoRA/SACM baseline.  The Conv-LoRA
base is built and used unchanged; CurvField adds a structure-field head on
the frozen patch embedding and zero-initialized structure-conditioned
adapters on a few transformer blocks.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baseline_zoo.data import load_mask, load_rgb
from baseline_zoo.models.convlora import StructureLoss, build_convlora
from baseline_zoo.models.curvfield import (
    CurvFieldSAM,
    field_loss,
    structure_targets,
)
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256


class CurvFieldAdapter(BaselineAdapter):
    id = 'curvfield'
    display_name = 'CurvField'
    family = 'sam'
    official_source = (
        'project extension: Conv-LoRA + shared structure field; not an '
        'official Conv-LoRA or SACM baseline')
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
        'CurvField is a project extension. It builds the existing Conv-LoRA '
        'port unchanged and adds a 128x128 structure-field head and '
        'zero-initialized field-conditioned FiLM adapters; no official '
        'CurvField weights exist.',
        'The structure field is supervised only from training GT masks: '
        'skeleton/centerline, distance-transform radius, skeleton tangent '
        'orientation, branch points and orientation coherence. No validation '
        'or test masks enter training.',
        'The field does not replace the Conv-LoRA qkv experts or the official '
        'one-token decoder; those remain the B0 baseline. The v2 adapter keeps '
        'the local depthwise branch and conditions it with a field-driven '
        'FiLM scale/shift; the v1 strip/scalar gate was removed after an '
        'oracle-field diagnosis showed it was field-insensitive. A '
        'zero-initialized field-conditioned decoder logit bias is trained '
        'directly, so the field can affect decisions from early training. '
        'The adapter is residual with gamma initialized to zero, so the '
        'initial model is numerically the Conv-LoRA baseline.',
        'The default local profile is SAM ViT-B. ViT-L/H can be selected for '
        'the 24 GiB server profile. The project 800-epoch budget, 18-image '
        '3-shot split, validation-Dice checkpoint selection, GT-resolution '
        'export and shared evaluator are the project protocol.',
    )

    def __init__(self):
        super().__init__()
        self._criterion = StructureLoss()
        self._sam_sha: str | None = None
        self._encoder_weights: Path | None = None
        self._model_type = 'vit_b'
        self._rank = 3
        self._alpha = 32
        self._experts = 8
        self._moe_weight = 1.0
        self._field_hidden = 128
        self._field_size = 128
        self._adapter_hidden = 64
        self._adapter_fractions = (0.25, 0.5, 0.75)
        self._film_scale = 0.5
        self._decoder_bias = True
        self._field_weight = 0.2
        self._new_lr_mult = 10.0
        self._static_field = False

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=1024)
        parser.add_argument(
            '--convlora-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_b_01ec64.pth')
        parser.add_argument(
            '--convlora-model-type', default='vit_b',
            choices=['vit_h', 'vit_l', 'vit_b'])
        parser.add_argument('--convlora-rank', type=int, default=3)
        parser.add_argument('--convlora-alpha', type=int, default=32)
        parser.add_argument('--convlora-experts', type=int, default=8)
        parser.add_argument('--convlora-moe-weight', type=float, default=1.0)
        parser.add_argument('--curvfield-field-hidden', type=int, default=128)
        parser.add_argument(
            '--curvfield-field-size', type=int, default=128,
            choices=[64, 128], help='structure-field output resolution')
        parser.add_argument('--curvfield-adapter-hidden', type=int, default=64)
        parser.add_argument(
            '--curvfield-adapter-fractions', nargs='+', type=float,
            default=[0.25, 0.5, 0.75])
        parser.add_argument(
            '--curvfield-film-scale', type=float, default=0.5,
            help='maximum field-driven FiLM scale/shift')
        parser.add_argument(
            '--curvfield-decoder-bias',
            type=lambda value: str(value).lower() in ('1', 'true', 'yes', 'on'),
            default=True, help='add field-conditioned decoder logit bias')
        parser.add_argument(
            '--curvfield-field-weight', type=float, default=0.2)
        parser.add_argument(
            '--curvfield-new-lr-mult', type=float, default=10.0,
            help='learning-rate multiplier for field head and adapters')
        parser.add_argument(
            '--curvfield-static-field', action='store_true',
            help='ablation: use a constant field instead of the prediction')

    def _apply_arch(self, args):
        self.input_size = int(getattr(args, 'input_size', 1024))
        if self.input_size != 1024:
            raise SystemExit('CurvField requires SAM input 1024')
        self._model_type = str(
            getattr(args, 'convlora_model_type', 'vit_b'))
        if self._model_type not in ('vit_h', 'vit_l', 'vit_b'):
            raise SystemExit('unsupported Conv-LoRA model type')
        self._rank = int(getattr(args, 'convlora_rank', 3))
        self._alpha = int(getattr(args, 'convlora_alpha', 32))
        self._experts = int(getattr(args, 'convlora_experts', 8))
        self._moe_weight = float(
            getattr(args, 'convlora_moe_weight', 1.0))
        self._field_hidden = int(
            getattr(args, 'curvfield_field_hidden', 128))
        self._field_size = int(
            getattr(args, 'curvfield_field_size', 128))
        self._adapter_hidden = int(
            getattr(args, 'curvfield_adapter_hidden', 64))
        fractions = tuple(
            float(value) for value in getattr(
                args, 'curvfield_adapter_fractions', [0.25, 0.5, 0.75]))
        self._adapter_fractions = fractions
        self._film_scale = float(
            getattr(args, 'curvfield_film_scale', 0.5))
        self._decoder_bias = bool(
            getattr(args, 'curvfield_decoder_bias', True))
        self._field_weight = float(
            getattr(args, 'curvfield_field_weight', 0.2))
        self._new_lr_mult = float(
            getattr(args, 'curvfield_new_lr_mult', 10.0))
        self._static_field = bool(
            getattr(args, 'curvfield_static_field', False))
        if self._field_hidden < 8 or self._adapter_hidden < 8:
            raise SystemExit('CurvField hidden dimensions must be >= 8')
        if self._field_size not in (64, 128):
            raise SystemExit('--curvfield-field-size must be 64 or 128')
        if self._film_scale <= 0:
            raise SystemExit('--curvfield-film-scale must be positive')
        if not self._adapter_fractions:
            raise SystemExit('CurvField needs at least one adapter fraction')
        if any(not 0.0 <= value <= 1.0
               for value in self._adapter_fractions):
            raise SystemExit('CurvField adapter fractions must be in [0, 1]')
        if self._field_weight < 0 or self._new_lr_mult <= 0:
            raise SystemExit('CurvField weights must be non-negative')
        self._encoder_weights = Path(getattr(
            args, 'convlora_checkpoint',
            REPO_ROOT / 'checkpoints' / 'sam_vit_b_01ec64.pth'))

    def build_model(self, args):
        self._apply_arch(args)
        assert self._encoder_weights is not None
        if not self._encoder_weights.is_file():
            raise SystemExit(
                f'missing Conv-LoRA SAM checkpoint: {self._encoder_weights}')
        self._sam_sha = file_sha256(self._encoder_weights)
        base = build_convlora(
            checkpoint=self._encoder_weights,
            model_type=self._model_type,
            image_size=self.input_size,
            rank=self._rank,
            alpha=self._alpha,
            experts=self._experts,
            moe_weight=self._moe_weight,
        )
        return CurvFieldSAM(
            base,
            field_hidden=self._field_hidden,
            field_size=self._field_size,
            adapter_hidden=self._adapter_hidden,
            adapter_fractions=self._adapter_fractions,
            film_scale=self._film_scale,
            decoder_bias=self._decoder_bias,
            static_field=self._static_field,
        )

    def configure_trainable(self, model):
        for parameter in model.field_head.parameters():
            parameter.requires_grad_(True)
        for parameter in model.adapters.parameters():
            parameter.requires_grad_(True)
        if model.decoder_bias:
            for parameter in model.field_decoder.parameters():
                parameter.requires_grad_(True)
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        if trainable == 0:
            raise RuntimeError('CurvField has no trainable parameters')
        print(f'CurvField parameters: {trainable:,} trainable')

    def make_optimizer(self, model, args):
        base_params, new_params = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith(('field_head.', 'adapters.', 'field_decoder.')):
                new_params.append(parameter)
            else:
                base_params.append(parameter)
        groups = []
        if base_params:
            groups.append({'params': base_params, 'lr': float(args.lr)})
        if new_params:
            groups.append({
                'params': new_params,
                'lr': float(args.lr) * self._new_lr_mult,
            })
        return torch.optim.Adam(
            groups, weight_decay=float(args.weight_decay))

    def make_scheduler(self, optimizer, args):
        epochs = max(1, int(args.epochs))

        def factor(epoch: int) -> float:
            return max(0.0, 1.0 - epoch / epochs)

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
        logits, moe_loss, field = model(images)
        loss = self._criterion(logits, masks)
        if field is not None and self._field_weight > 0:
            targets, foreground = structure_targets(
                masks, size=field.shape[-1])
            epoch = int(getattr(self, '_epoch', 1))
            epochs = max(1, int(getattr(self.args, 'epochs', 800)))
            ramp = min(1.0, epoch / max(1, int(round(0.1 * epochs))))
            loss = loss + self._field_weight * ramp * field_loss(
                field, targets, foreground)
        return loss + self._moe_weight * moe_loss

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size), Image.BICUBIC)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return tensor.unsqueeze(0).to(device), (int(width), int(height))

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        image, (width, height) = prepared
        logits, _, _ = model(image)
        logits = F.interpolate(
            logits, size=(height, width), mode='bilinear',
            align_corners=False)
        return torch.sigmoid(logits[0, 0]).cpu().numpy() > 0.5

    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'model_type': self._model_type,
            'rank': self._rank,
            'alpha': self._alpha,
            'experts': self._experts,
            'moe_weight': self._moe_weight,
            'convlora_checkpoint': str(self._encoder_weights or ''),
            'sam_checkpoint_sha256': self._sam_sha,
            'field_hidden': self._field_hidden,
            'field_size': self._field_size,
            'adapter_hidden': self._adapter_hidden,
            'adapter_fractions': list(self._adapter_fractions),
            'film_scale': self._film_scale,
            'decoder_bias': self._decoder_bias,
            'field_weight': self._field_weight,
            'new_lr_mult': self._new_lr_mult,
            'static_field': self._static_field,
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
        mapping = {
            'field_hidden': 'curvfield_field_hidden',
            'field_size': 'curvfield_field_size',
            'adapter_hidden': 'curvfield_adapter_hidden',
            'adapter_fractions': 'curvfield_adapter_fractions',
            'film_scale': 'curvfield_film_scale',
            'decoder_bias': 'curvfield_decoder_bias',
            'field_weight': 'curvfield_field_weight',
            'new_lr_mult': 'curvfield_new_lr_mult',
            'static_field': 'curvfield_static_field',
        }
        for key, attr in mapping.items():
            if key in config:
                setattr(args, attr, config[key])
        self._apply_arch(args)
