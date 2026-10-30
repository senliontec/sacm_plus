"""Conv-LoRA + SDSConv project extension.

The Conv-LoRA base is built unchanged.  A zero-initialized sparse-control
dynamic snake convolution (SDSConv) residual is applied to the SAM image
embedding before the mask decoder.  The residual is a bottleneck (64 channels
by default) so the local 11 GiB profile remains usable; the initial model is
numerically Conv-LoRA.
"""

from __future__ import annotations

from pathlib import Path

import torch

from baseline_zoo.adapters.convlora import ConvLoRAAdapter
from baseline_zoo.models.sdsconv import SDSConvSAM
from baseline_zoo.runtime import REPO_ROOT


class SDSConvAdapter(ConvLoRAAdapter):
    id = 'sdsconv'
    display_name = 'Conv-LoRA + SDSConv'
    official_source = (
        'project extension: Conv-LoRA + SDSConv (CSWinUNETR, MICCAI 2026); '
        'not an official Conv-LoRA/SACM baseline')
    official_entrypoint = (
        'https://github.com/labhai/CSWinUNETR')
    family = 'sam'
    deviations = ConvLoRAAdapter.deviations + (
        'SDSConv is a project extension, not an official Conv-LoRA baseline. '
        'It predicts sparse control points, RBF-interpolates a dense '
        'curvilinear sampling trajectory and applies a depthwise aggregation '
        'on a 64-channel bottleneck of the SAM image embedding.',
        'The SDSConv residual is zero-initialized (adapter gamma and final '
        'convolutions), so step 0 is numerically the Conv-LoRA baseline. '
        'No official SDSConv or Conv-LoRA weights exist for this combination.',
        'The default local screen uses SAM ViT-B, 100 epochs, the shared '
        '3-shot split and validation-Dice selection; ViT-H/L can be selected '
        'for the 24 GiB server profile.',
    )

    def __init__(self):
        super().__init__()
        self._sds_kernel = 7
        self._sds_hidden = 64
        self._sds_new_lr_mult = 10.0

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument(
            '--sdsconv-kernel-size', type=int, default=7,
            help='odd SDSConv trajectory length (>=5)')
        parser.add_argument(
            '--sdsconv-hidden', type=int, default=64,
            help='bottleneck channels for the SDSConv residual')
        parser.add_argument(
            '--sdsconv-new-lr-mult', type=float, default=10.0,
            help='learning-rate multiplier for SDSConv parameters')

    def _apply_sds(self, args):
        self._sds_kernel = int(
            getattr(args, 'sdsconv_kernel_size', 7))
        self._sds_hidden = int(getattr(args, 'sdsconv_hidden', 64))
        self._sds_new_lr_mult = float(
            getattr(args, 'sdsconv_new_lr_mult', 10.0))
        if self._sds_kernel < 5 or self._sds_kernel % 2 == 0:
            raise SystemExit('--sdsconv-kernel-size must be odd and >= 5')
        if self._sds_hidden < 8:
            raise SystemExit('--sdsconv-hidden must be >= 8')
        if self._sds_new_lr_mult <= 0:
            raise SystemExit('--sdsconv-new-lr-mult must be positive')

    def build_model(self, args):
        self._apply_sds(args)
        base = super().build_model(args)
        return SDSConvSAM(
            base,
            hidden=self._sds_hidden,
            kernel_size=self._sds_kernel,
        )

    def configure_trainable(self, model):
        super().configure_trainable(model)
        for parameter in model.sds_adapter.parameters():
            parameter.requires_grad_(True)
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        print(f'SDSConv parameters: {trainable:,} trainable')

    def make_optimizer(self, model, args):
        base_params, sds_params = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith('sds_adapter.'):
                sds_params.append(parameter)
            else:
                base_params.append(parameter)
        groups = []
        if base_params:
            groups.append({'params': base_params, 'lr': float(args.lr)})
        if sds_params:
            groups.append({
                'params': sds_params,
                'lr': float(args.lr) * self._sds_new_lr_mult,
            })
        return torch.optim.Adam(
            groups, weight_decay=float(args.weight_decay))

    def save_config(self, args):
        config = super().save_config(args)
        config.update({
            'sdsconv_kernel_size': self._sds_kernel,
            'sdsconv_hidden': self._sds_hidden,
            'sdsconv_new_lr_mult': self._sds_new_lr_mult,
        })
        return config

    def restore_config(self, args, config):
        super().restore_config(args, config)
        if 'sdsconv_kernel_size' in config:
            args.sdsconv_kernel_size = int(config['sdsconv_kernel_size'])
        if 'sdsconv_hidden' in config:
            args.sdsconv_hidden = int(config['sdsconv_hidden'])
        if 'sdsconv_new_lr_mult' in config:
            args.sdsconv_new_lr_mult = float(
                config['sdsconv_new_lr_mult'])
        self._apply_sds(args)


__all__ = ['SDSConvAdapter']
