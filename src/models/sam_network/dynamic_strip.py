"""Dynamic strip convolutions for the geometric adapters.

The fixed strip branch in ``GeoAdapterModule`` / ``AdapterEGlobal`` uses
axis-aligned ``(k, 1)`` and ``(1, k)`` depthwise convolutions.  Curvilinear
structures (vessels, cracks, roads) bend, so a fixed vertical/horizontal
strip often misses the local direction.  This module adds a small learned
offset field and applies torchvision's deformable convolution, allowing the
strip to follow the local tangent while keeping the depthwise parameter
cost small.

The offsets and branch scale are zero-initialized, so a dynamic variant
starts from exactly the same behaviour as the fixed strip branch and then
learns deformation during training.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class DynamicStripConv2d(nn.Module):
    """Depthwise ``(k, 1)`` or ``(1, k)`` strip with learned offsets."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 9,
        orientation: str = 'v',
        offset_groups: int = 1,
    ) -> None:
        super().__init__()
        if orientation not in ('v', 'h'):
            raise ValueError("orientation must be 'v' or 'h'")
        if kernel_size < 1:
            raise ValueError('kernel_size must be positive')
        if channels % int(offset_groups) != 0:
            raise ValueError('offset_groups must divide channels')

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.orientation = orientation
        self.offset_groups = int(offset_groups)

        if orientation == 'v':
            kernel = (self.kernel_size, 1)
            self.padding = (self.kernel_size // 2, 0)
        else:
            kernel = (1, self.kernel_size)
            self.padding = (0, self.kernel_size // 2)

        # Depthwise strip weights: one filter per channel.
        self.weight = nn.Parameter(torch.empty(channels, 1, *kernel))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(channels))

        # Offset field: 2 * offset_groups * k values per spatial position.
        offset_channels = 2 * self.offset_groups * kernel[0] * kernel[1]
        self.offset_conv = nn.Conv2d(
            channels, offset_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            from torchvision.ops import deform_conv2d
        except ImportError as exc:  # pragma: no cover - torchvision is required
            raise RuntimeError(
                'dynamic strip requires torchvision.ops.deform_conv2d') from exc

        offset = self.offset_conv(x)
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=1,
            padding=self.padding,
            dilation=1,
            groups=self.channels,
            offset_groups=self.offset_groups,
        )


class DynamicStripBranch(nn.Module):
    """Sequential vertical + horizontal dynamic strip branch."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 9,
        offset_groups: int = 1,
    ) -> None:
        super().__init__()
        self.strip_v = DynamicStripConv2d(
            channels, kernel_size, orientation='v',
            offset_groups=offset_groups)
        self.strip_h = DynamicStripConv2d(
            channels, kernel_size, orientation='h',
            offset_groups=offset_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.strip_h(self.strip_v(x))


__all__ = ['DynamicStripBranch', 'DynamicStripConv2d']
