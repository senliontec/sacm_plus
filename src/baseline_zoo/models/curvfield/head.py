"""Lightweight structure-field head for CurvField."""

from __future__ import annotations

import torch
from torch import nn


class StructureFieldHead(nn.Module):
    """Predict centerline, radius, orientation, branch and coherence.

    Input is the frozen SAM patch-embedding map ``[B,H,W,C]`` (or
    ``[B,C,H,W]``); output is ``[B,6,H,W]`` in this channel order:
    centerline, radius, cos(2theta), sin(2theta), branch, coherence.
    The head is deliberately small: the field is a low-dimensional
    geometric representation, not a second segmentation network.
    """

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.in_dim = int(in_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
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
        return self.out(self.stem(features))


__all__ = ['StructureFieldHead']
