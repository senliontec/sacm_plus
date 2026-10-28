"""TPDecoder building blocks: SEM, HLFF and LCCM.

Vendored from the TPSegformer release (``mmseg/models/utils/{sem,hlff,lccm}.py``)
with the external ``hilbertcurve``/``hbtrans``/``thop`` imports replaced by
the local Hilbert implementation.  The layer math, dimensions and the
``x * sigmoid(attention)`` gate are unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hilbert import expand_hilbert, flatten_hilbert, hilbert_permutation


class ECALayer(nn.Module):
    """ECANet channel attention plus the release's spatial 3x3 branch."""

    def __init__(self, channel: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size,
            padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.spatial = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        descriptor = self.avg_pool(x)
        spatial = self.spatial(x)
        attention = self.conv(
            descriptor.squeeze(-1).transpose(-1, -2)
        ).transpose(-1, -2).unsqueeze(-1)
        return spatial * self.sigmoid(attention).expand_as(spatial)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, padding: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class HighLowFeatureFusionModule(nn.Module):
    """Upsample X4, concatenate with X3, compress with a separable conv."""

    def __init__(self, C1: int, C2: int):
        super().__init__()
        self.depthwise_separable = DepthwiseSeparableConv2d(
            C1 + C2, C1, kernel_size=3, padding=1)

    def forward(self, low_res_features: torch.Tensor,
                high_res_features: torch.Tensor) -> torch.Tensor:
        height, width = high_res_features.shape[-2:]
        upsampled = F.interpolate(
            low_res_features, size=(height, width), mode='bilinear',
            align_corners=False)
        fused = torch.cat((high_res_features, upsampled), dim=1)
        return self.depthwise_separable(fused)


def _z_permutation(size: int) -> tuple[np.ndarray, np.ndarray]:
    def z_order(x: int, y: int) -> int:
        value = 0
        bit = 0
        while (1 << bit) < size:
            value |= ((x & (1 << bit)) << bit)
            value |= ((y & (1 << bit)) << (bit + 1))
            bit += 1
        return value

    values = np.array(
        [[z_order(row, column) for column in range(size)]
         for row in range(size)], dtype=np.int64).reshape(-1)
    indices = np.argsort(values, kind='stable')
    return indices, np.argsort(indices, kind='stable')


class LayerCorrelationCalculation(nn.Module):
    """Classwise spatial attention flattened with a space-filling curve."""

    def __init__(self, input_size: int = 32, mode: str = 'hilbert'):
        super().__init__()
        if mode not in ('hilbert', 'row-first', 'z-curve'):
            raise ValueError(f'unsupported flattening mode: {mode}')
        self.input_size = int(input_size)
        # The release takes the channel-wise max first, so QKV always
        # receives a single channel.
        self.qkv = nn.Conv2d(1, 3, kernel_size=1)
        self.mode = mode
        self.sigmoid = nn.Sigmoid()

    def _orders(self, height: int, width: int):
        if self.mode == 'row-first':
            indices = np.arange(height * width)
            return indices, indices
        if self.mode == 'z-curve':
            return _z_permutation(height)
        if height != width:
            raise ValueError('Hilbert mode requires a square feature map')
        return hilbert_permutation(height)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        if height != self.input_size or width != self.input_size:
            raise ValueError(
                f'expected {self.input_size}x{self.input_size}, got '
                f'{height}x{width}')
        x_max = torch.max(x, dim=1, keepdim=True).values
        qkv = self.qkv(x_max)
        query, key, value = torch.split(qkv, 1, dim=1)
        indices, inverse = self._orders(height, width)
        query_flat = flatten_hilbert(query, indices)
        key_flat = flatten_hilbert(key, indices)
        value_flat = flatten_hilbert(value, indices)
        attention = torch.matmul(
            query_flat.transpose(-1, -2), key_flat)
        attention = F.softmax(attention, dim=-2)
        output = torch.matmul(value_flat, attention)
        output = expand_hilbert(output, inverse, height)
        return x * self.sigmoid(output)
