"""NativeToken: Conv-LoRA with native-resolution residual tokenization.

This is a new project model, not an official baseline.  The Conv-LoRA
backbone, prompt-free decoder, structure loss and MoE loss are unchanged.
The only addition is a tokenizer that samples fine patches from the
original-resolution image and adds them to the coarse SAM tokens.  This
moves SACM's coarse-to-fine idea from the decoder to the tokenizer without
adding any SACM adapter, style conditioning or fusion module.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baseline_zoo.adapters.convlora import ConvLoRAAdapter
from baseline_zoo.data import load_mask, load_rgb

from .model import NativeTokenSAM


class NativeTokenAdapter(ConvLoRAAdapter):
    id = 'nativetok'
    display_name = 'NativeToken'
    official_source = (
        'project extension: Conv-LoRA + native-resolution residual '
        'tokenization; not an official baseline')
    official_entrypoint = (
        'project extension; official Conv-LoRA baseline is convlora')
    family = 'sam'
    default_epochs = 800
    deviations = ConvLoRAAdapter.deviations + (
        'NativeToken is a new project model, not an official Conv-LoRA '
        'or SACM baseline. It keeps the Conv-LoRA qkv LoRA, multi-scale '
        'conv MoE, prompt-free decoder, structure loss and MoE loss.',
        'The only new path is a tokenizer on SAM patch_embed: coarse tokens '
        'predict a per-token distribution over {no-fine, 2x fine, 4x fine}; '
        'fine patches are sampled from the original-resolution image at '
        'the token centres and embedded with a frozen copy of the SAM '
        'patch-embedding convolution. The fine embedding is added with a '
        'zero-initialized scalar residual, so step 0 is numerically '
        'Conv-LoRA.',
        'No extra image-encoder forward pass, extra token, decoder, loss, '
        'style conditioning, SACM adapter or fusion module is used. The '
        'original-resolution image is only used to sample a small patch '
        'around each token; it is never passed through the ViT again.',
        'The default screen uses SAM ViT-B and 100 epochs on the shared '
        '18-image 3-shot split. The official Conv-LoRA baseline remains '
        'unmodified for the main comparison.',
    )

    def __init__(self):
        super().__init__()
        self._scales = (8, 4)
        self._hidden = 32
        self._alpha_init = 0.0
        self._new_lr_mult = 10.0
        self._native_max_side = 0

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument(
            '--nativetok-scales', nargs='+', type=int, default=[8, 4],
            help='fine patch sides in 1024-image pixels; 8=2x, 4=4x')
        parser.add_argument(
            '--nativetok-hidden', type=int, default=32)
        parser.add_argument(
            '--nativetok-alpha-init', type=float, default=0.0)
        parser.add_argument(
            '--nativetok-new-lr-mult', type=float, default=10.0)
        parser.add_argument(
            '--nativetok-native-max-side', type=int, default=0,
            help='optional cap on the native image long side; 0 keeps '
                 'the original resolution')

    def _apply_nativetok(self, args):
        scales = tuple(int(value) for value in getattr(
            args, 'nativetok_scales', [8, 4]))
        self._scales = scales
        self._hidden = int(getattr(args, 'nativetok_hidden', 32))
        self._alpha_init = float(
            getattr(args, 'nativetok_alpha_init', 0.0))
        self._new_lr_mult = float(
            getattr(args, 'nativetok_new_lr_mult', 10.0))
        self._native_max_side = int(
            getattr(args, 'nativetok_native_max_side', 0))
        if not scales:
            raise SystemExit('NativeToken needs at least one fine scale')
        if any(value < 2 or value >= 16 for value in scales):
            raise SystemExit('NativeToken fine scales must be in [2, 15]')
        if len(set(scales)) != len(scales):
            raise SystemExit('NativeToken fine scales must be unique')
        if self._hidden < 8:
            raise SystemExit('NativeToken gate hidden width is too small')
        if self._new_lr_mult <= 0:
            raise SystemExit('NativeToken new LR multiplier must be positive')
        if self._native_max_side < 0:
            raise SystemExit('NativeToken native max side must be >= 0')

    def build_model(self, args):
        self._apply_nativetok(args)
        base = super().build_model(args)
        return NativeTokenSAM(
            base,
            scales=self._scales,
            hidden=self._hidden,
            alpha_init=self._alpha_init,
        )

    def configure_trainable(self, model):
        super().configure_trainable(model)
        for parameter in model.tokenizer.parameters():
            parameter.requires_grad_(True)
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        print(f'NativeToken parameters: {trainable:,} trainable')

    def make_optimizer(self, model, args):
        base_params, new_params = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith('tokenizer.'):
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

    def _native_image(self, sample):
        image = load_rgb(sample.image_path)
        if (self._native_max_side > 0
                and max(image.size) > self._native_max_side):
            scale = self._native_max_side / float(max(image.size))
            size = (max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)))
            image = image.resize(size, Image.BICUBIC)
        return image

    def _pair(self, sample, rng: random.Random):
        native_image = self._native_image(sample)
        global_image = native_image.resize(
            (self.input_size, self.input_size), Image.BICUBIC)
        mask = load_mask(sample.mask_path)
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (self.input_size, self.input_size), Image.NEAREST))
        mask = (mask > 127).astype(np.float32)
        image_np = np.asarray(global_image, dtype=np.float32).copy()
        native_np = np.asarray(native_image, dtype=np.float32).copy()
        if rng.random() < 0.5:
            image_np = np.flip(image_np, axis=1).copy()
            native_np = np.flip(native_np, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        image_t = torch.from_numpy(image_np).permute(2, 0, 1)
        native_t = torch.from_numpy(native_np).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask)[None]
        return image_t, native_t, mask_t

    def prepare_train_batch(self, samples, rng: random.Random):
        images, native_images, masks = [], [], []
        for sample in samples:
            image, native, mask = self._pair(sample, rng)
            images.append(image)
            native_images.append(native.to(self.device))
            masks.append(mask)
        return (torch.stack(images).to(self.device),
                native_images,
                torch.stack(masks).to(self.device))

    def train_step(self, model, batch):
        images, native_images, masks = batch
        logits, moe_loss = model(images, native_images)
        loss = self._criterion(logits, masks)
        weight = float(getattr(self.args, 'convlora_moe_weight', 1.0))
        return loss + weight * moe_loss

    def prepare_eval_sample(self, sample, device):
        native_image = self._native_image(sample)
        image = native_image.resize(
            (self.input_size, self.input_size), Image.BICUBIC)
        image_t = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1).unsqueeze(0)
        native_t = torch.from_numpy(
            np.asarray(native_image, dtype=np.float32).copy()
        ).permute(2, 0, 1)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return (image_t.to(device), native_t.to(device),
                (int(width), int(height)))

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        image, native, (width, height) = prepared
        logits, _ = model(image, [native])
        logits = F.interpolate(
            logits, size=(height, width), mode='bilinear',
            align_corners=False)
        return torch.sigmoid(logits[0, 0]).cpu().numpy() > 0.5

    def save_config(self, args):
        config = super().save_config(args)
        config.update({
            'nativetok_scales': list(self._scales),
            'nativetok_hidden': self._hidden,
            'nativetok_alpha_init': self._alpha_init,
            'nativetok_new_lr_mult': self._new_lr_mult,
            'nativetok_native_max_side': self._native_max_side,
        })
        return config

    def restore_config(self, args, config):
        super().restore_config(args, config)
        mapping = {
            'nativetok_scales': 'nativetok_scales',
            'nativetok_hidden': 'nativetok_hidden',
            'nativetok_alpha_init': 'nativetok_alpha_init',
            'nativetok_new_lr_mult': 'nativetok_new_lr_mult',
            'nativetok_native_max_side': 'nativetok_native_max_side',
        }
        for key, attr in mapping.items():
            if key in config:
                setattr(args, attr, config[key])
        self._apply_nativetok(args)


__all__ = ['NativeTokenAdapter']
