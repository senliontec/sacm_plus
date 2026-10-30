"""Lightweight structure-field head for CurvField."""

from __future__ import annotations

import torch
from torch import nn


class StructureFieldHead(nn.Module):
    """Predict centerline, radius, orientation, branch and coherence.

    Input is the frozen SAM patch-embedding map ``[B,H,W,C]`` (or
    ``[B,C,H,W]``); output is ``[B,6,S,S]`` in this channel order:
    centerline, radius, cos(2theta), sin(2theta), branch, coherence.

    ``out_size=128`` upsamples the 64x64 patch grid with a lightweight
    transposed convolution.  The 64x64 field was too coarse for thin
    structures (CREMI/LEAF/TYRE), where the target skeleton and radius are
    merged by nearest downsampling.  The head remains deliberately small;
    it is a geometric representation, not a second segmentation network.
    """

    def __init__(self, in_dim: int, hidden: int = 128,
                 out_size: int = 128):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_size = int(out_size)
        if self.out_size not in (64, 128):
            raise ValueError('out_size must be 64 or 128')
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        if self.out_size == 128:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    hidden, hidden, 4, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
            )
        else:
            self.up = nn.Identity()
        self.out = nn.Conv2d(hidden, 6, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                f'expected [B,H,W,C] or [B,C,H,W], got '
                f'{tuple(features.shape)}')
        # SAM patch embedding returns NHWC; prefer NHWC when both the last
        # axis and the channel axis match the configured dimension.
        if features.shape[-1] == self.in_dim:
            features = features.permute(0, 3, 1, 2)
        elif features.shape[1] != self.in_dim:
            raise ValueError(
                f'feature channel mismatch: expected {self.in_dim}, got '
                f'{tuple(features.shape)}')
        field = self.up(self.stem(features))
        if field.shape[-2:] != (self.out_size, self.out_size):
            raise RuntimeError(
                f'field head produced {tuple(field.shape[-2:])}, expected '
                f'{(self.out_size, self.out_size)}')
        return self.out(field)


class FieldDecoderBias(nn.Module):
    """Small field -> logit-bias head for the segmentation decoder.

    The last convolution is zero-initialized, so the initial model is exactly
    the Conv-LoRA baseline.  Unlike the v1 gamma-gated adapter, this path is
    directly supervised by the segmentation loss: the last layer starts
    receiving gradient at step 1, which makes the oracle-field test meaningful
    early in training.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return self.net(field)


__all__ = ['FieldDecoderBias', 'StructureFieldHead']
