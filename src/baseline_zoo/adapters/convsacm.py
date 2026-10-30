"""Conv-SACM: Conv-LoRA + SACM-inspired global/high-res decisions.

This is a new project model, not the official SACM baseline.  The official
``sacm_official`` adapter remains unmodified.  Conv-SACM keeps Conv-LoRA's
qkv LoRA + multi-scale conv MoE and prompt-free decoder, adds a lightweight
external-context residual over a few encoder blocks, and refines the decoder
output from 256x256 to 512x512.  Both new paths are zero-initialized.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from baseline_zoo.adapters.convlora import ConvLoRAAdapter
from baseline_zoo.models.convsacm import ConvSACMSAM
from core.losses import SoftclDiceLoss


class ConvSACMAdapter(ConvLoRAAdapter):
    id = 'convsacm'
    display_name = 'Conv-SACM'
    official_source = (
        'project extension: Conv-LoRA local MoE + SACM-inspired external '
        'context/high-res decision; not an official baseline')
    official_entrypoint = (
        'project extension; official SACM baseline is sacm_official '
        '(unmodified)')
    family = 'sam'
    deviations = ConvLoRAAdapter.deviations + (
        'Conv-SACM is a new project model, not an official Conv-LoRA/SACM '
        'baseline. It keeps the Conv-LoRA qkv LoRA, multi-scale conv MoE, '
        'prompt-free decoder and training recipe, and adds a zero-initialized '
        'external-context residual plus a 512x512 second-stage refiner.',
        'The external-context path aggregates a few encoder block features; '
        'the high-resolution refiner uses a lightweight image stem. Both are '
        'zero-initialized, so step 0 is numerically Conv-LoRA.',
        'Unlike the official SACM implementation, this model explicitly '
        'supervises both the coarse and high-resolution outputs and never '
        'uses the untrained Stage-1/gating/4-head-only-1-loss mechanism. '
        'The official SACM baseline stays untouched for the main comparison.',
        'A soft clDice loss is applied to the high-resolution output with a '
        'short warmup; it is an auxiliary term, not a replacement for the '
        'Conv-LoRA structure loss.',
    )

    def __init__(self):
        super().__init__()
        self._external_hidden = 64
        self._external_fractions = (0.25, 0.5, 0.75)
        self._refiner_hidden = 16
        self._refiner_size = 512
        self._topo_weight = 0.2
        self._topo_warmup = 10
        self._new_lr_mult = 10.0
        self._topo = SoftclDiceLoss(num_iter=10)

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument(
            '--convsacm-external-hidden', type=int, default=64)
        parser.add_argument(
            '--convsacm-external-fractions', nargs='+', type=float,
            default=[0.25, 0.5, 0.75])
        parser.add_argument(
            '--convsacm-refiner-hidden', type=int, default=16)
        parser.add_argument(
            '--convsacm-refiner-size', type=int, default=512,
            choices=[512])
        parser.add_argument(
            '--convsacm-topo-weight', type=float, default=0.2)
        parser.add_argument(
            '--convsacm-topo-warmup', type=int, default=10)
        parser.add_argument(
            '--convsacm-new-lr-mult', type=float, default=10.0)

    def _apply_convsacm(self, args):
        self._external_hidden = int(
            getattr(args, 'convsacm_external_hidden', 64))
        fractions = tuple(
            float(value) for value in getattr(
                args, 'convsacm_external_fractions', [0.25, 0.5, 0.75]))
        self._external_fractions = fractions
        self._refiner_hidden = int(
            getattr(args, 'convsacm_refiner_hidden', 16))
        self._refiner_size = int(
            getattr(args, 'convsacm_refiner_size', 512))
        self._topo_weight = float(
            getattr(args, 'convsacm_topo_weight', 0.2))
        self._topo_warmup = int(
            getattr(args, 'convsacm_topo_warmup', 10))
        self._new_lr_mult = float(
            getattr(args, 'convsacm_new_lr_mult', 10.0))
        if not fractions or any(not 0.0 <= value <= 1.0
                                for value in fractions):
            raise SystemExit('Conv-SACM external fractions must be in [0,1]')
        if self._external_hidden < 8 or self._refiner_hidden < 4:
            raise SystemExit('Conv-SACM hidden dimensions are too small')
        if self._refiner_size != 512:
            raise SystemExit('Conv-SACM v1 supports refiner size 512')
        if self._topo_weight < 0 or self._topo_warmup < 0:
            raise SystemExit('Conv-SACM topo weights must be non-negative')
        if self._new_lr_mult <= 0:
            raise SystemExit('Conv-SACM new LR multiplier must be positive')

    def build_model(self, args):
        self._apply_convsacm(args)
        base = super().build_model(args)
        return ConvSACMSAM(
            base,
            external_fractions=self._external_fractions,
            external_hidden=self._external_hidden,
            refiner_hidden=self._refiner_hidden,
            refiner_size=self._refiner_size,
        )

    def configure_trainable(self, model):
        super().configure_trainable(model)
        for parameter in model.external.parameters():
            parameter.requires_grad_(True)
        for parameter in model.refiner.parameters():
            parameter.requires_grad_(True)
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        print(f'Conv-SACM parameters: {trainable:,} trainable')

    def make_optimizer(self, model, args):
        base_params, new_params = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith(('external.', 'refiner.')):
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

    def train_step(self, model, batch):
        images, masks = batch
        coarse, high_res, moe_loss = model(images)
        target_coarse = F.interpolate(
            masks, size=coarse.shape[-2:], mode='bilinear',
            align_corners=False)
        target_high = F.interpolate(
            masks, size=high_res.shape[-2:], mode='bilinear',
            align_corners=False)
        loss = 0.5 * self._criterion(coarse, target_coarse)
        loss = loss + self._criterion(high_res, target_high)
        if self._topo_weight > 0:
            epoch = int(getattr(self, '_epoch', 1))
            ramp = min(1.0, epoch / max(1, self._topo_warmup))
            loss = loss + self._topo_weight * ramp * self._topo(
                high_res, target_high)
        weight = float(getattr(self.args, 'convlora_moe_weight', 1.0))
        return loss + weight * moe_loss

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        image, (width, height) = prepared
        _, high_res, _ = model(image)
        high_res = F.interpolate(
            high_res, size=(height, width), mode='bilinear',
            align_corners=False)
        return torch.sigmoid(high_res[0, 0]).cpu().numpy() > 0.5

    def save_config(self, args):
        config = super().save_config(args)
        config.update({
            'convsacm_external_hidden': self._external_hidden,
            'convsacm_external_fractions': list(self._external_fractions),
            'convsacm_refiner_hidden': self._refiner_hidden,
            'convsacm_refiner_size': self._refiner_size,
            'convsacm_topo_weight': self._topo_weight,
            'convsacm_topo_warmup': self._topo_warmup,
            'convsacm_new_lr_mult': self._new_lr_mult,
        })
        return config

    def restore_config(self, args, config):
        super().restore_config(args, config)
        mapping = {
            'convsacm_external_hidden': 'convsacm_external_hidden',
            'convsacm_external_fractions': 'convsacm_external_fractions',
            'convsacm_refiner_hidden': 'convsacm_refiner_hidden',
            'convsacm_refiner_size': 'convsacm_refiner_size',
            'convsacm_topo_weight': 'convsacm_topo_weight',
            'convsacm_topo_warmup': 'convsacm_topo_warmup',
            'convsacm_new_lr_mult': 'convsacm_new_lr_mult',
        }
        for key, attr in mapping.items():
            if key in config:
                setattr(args, attr, config[key])
        self._apply_convsacm(args)


__all__ = ['ConvSACMAdapter']
