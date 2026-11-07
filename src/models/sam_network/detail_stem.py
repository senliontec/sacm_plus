"""Lightweight high-resolution detail stem for the SAM network.

The frozen ViT-L operates on 16x patches, so thin structures lose spatial
detail before the decoder.  This stem runs in parallel on the 1024x1024
input and produces 1/4 (256x256) and 1/2 (512x512) detail features with an
explicit high-pass branch.  It is intentionally small (depthwise separable
convolutions) and is fused into the decoder through zero-initialized
residuals, so enabling it starts from the existing decoder behaviour.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .common import LayerNorm2d


class _DepthwiseBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False)
        self.norm = LayerNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.dw(x)))
        return self.act(self.pw(x))


class DetailStem(nn.Module):
    """Produce 1/4 and 1/2 resolution high-frequency detail features.

    Args:
        in_channels: input image channels (3).
        channels: width of the 1/2-resolution branch.
        out_channels: channels of both returned feature maps.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 32,
        out_channels: int = 32,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, stride=2, padding=1),
            LayerNorm2d(channels),
            nn.GELU(),
            _DepthwiseBlock(channels),
        )
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            LayerNorm2d(channels * 2),
            nn.GELU(),
            _DepthwiseBlock(channels * 2),
        )
        self.out_512 = nn.Conv2d(channels * 2, out_channels, 1)
        self.out_256 = nn.Conv2d(channels * 4, out_channels, 1)

    @staticmethod
    def _high_pass(x: torch.Tensor) -> torch.Tensor:
        return x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(detail_256, detail_512)``."""
        x512 = self.stem(image)
        hp512 = self._high_pass(x512)
        detail_512 = self.out_512(torch.cat([x512, hp512], dim=1))

        x256 = self.down(x512)
        hp256 = self._high_pass(x256)
        detail_256 = self.out_256(torch.cat([x256, hp256], dim=1))
        return detail_256, detail_512
