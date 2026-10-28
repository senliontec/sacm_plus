"""Lightweight coarse U-Net used by the reconstructed A2SAM SPG.

The official release does not include the paper's U-Net.  This module is the
paper-level reconstruction: a standard encoder/decoder trained with BCE only,
producing a likelihood map ``p`` and coarse logits.  GroupNorm is used because
the shared 3-shot protocol trains with batch 1.  The default widths give
approximately the paper's 4.32M U-Net parameter budget.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels: int) -> nn.Module:
    groups = 4
    if channels % groups != 0:
        groups = 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int,
                 out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                          align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class CoarseUNet(nn.Module):
    """BCE-only U-Net; returns logits at the input resolution."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        widths: tuple[int, int, int, int, int] = (36, 72, 144, 288, 288),
    ):
        super().__init__()
        c1, c2, c3, c4, cb = widths
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.bottleneck = ConvBlock(c4, cb)
        self.pool = nn.MaxPool2d(2)
        self.dec4 = UpBlock(cb, c4, c3)
        self.dec3 = UpBlock(c3, c3, c2)
        self.dec2 = UpBlock(c2, c2, c1)
        self.dec1 = UpBlock(c1, c1, c1)
        self.out = nn.Conv2d(c1, out_channels, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out(d1)

    def probability(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(image))

    def coarse_mask(self, image: torch.Tensor) -> torch.Tensor:
        return self.probability(image) > 0.5
