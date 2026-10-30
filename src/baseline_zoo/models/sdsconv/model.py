"""Sparse-control dynamic snake convolution (SDSConv) residual for SAM.

Project extension based on CSWinUNETR (Moon et al., MICCAI 2026).  The
operator predicts a small number of control points, RBF-interpolates a dense
curvilinear sampling trajectory, anchors it at the centre and aggregates the
samples along both image axes.  Here it is used only as a zero-initialized
bottleneck residual on the SAM image embedding, so the initial model is
exactly Conv-LoRA.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from baseline_zoo.models.convlora import ConvLoRASAM


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = next(
        group for group in range(min(32, channels), 0, -1)
        if channels % group == 0)
    return nn.GroupNorm(groups, channels)


class SDSConv2d(nn.Module):
    """2D sparse-control dynamic snake convolution."""

    def __init__(self, channels: int, kernel_size: int = 7,
                 sigma: float = 0.35):
        super().__init__()
        if kernel_size < 5 or kernel_size % 2 == 0:
            raise ValueError('SDSConv kernel_size must be odd and >= 5')
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.control_points = self.kernel_size // 2
        self.sigma = float(sigma)

        offset_channels = 3 * self.control_points
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, offset_channels, 1),
        )
        self.offset_norm = _group_norm(offset_channels)
        self.register_buffer(
            'axis',
            torch.arange(self.kernel_size, dtype=torch.float32)
            - self.kernel_size // 2)
        self.register_buffer(
            'control_base',
            torch.linspace(-1.0, 1.0, self.control_points))
        self.log_scope_base = nn.Parameter(torch.zeros(2))
        self.log_scope_amp = nn.Parameter(torch.full((2,), -3.0))

        self.aggregators = nn.ModuleList()
        for axis in range(2):
            kernel = (
                (self.kernel_size, 1) if axis == 0 else (1, self.kernel_size))
            stride = kernel
            self.aggregators.append(nn.Sequential(
                nn.Conv2d(
                    channels, channels, kernel, stride=stride,
                    groups=channels),
                nn.Conv2d(channels, channels, 1),
                _group_norm(channels),
                nn.ReLU(),
            ))

        hidden = min(channels, max(channels // 8, 16))
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1, bias=False),
            _group_norm(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden,
                      bias=False),
            _group_norm(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.output_norm = _group_norm(channels)
        # Straight trajectories and zero residual at initialization.
        for layer in (self.offset_conv[-1], self.fuse[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def dense_offsets(self, controls: torch.Tensor,
                      positions: torch.Tensor) -> torch.Tensor:
        """RBF interpolation with centre anchoring.

        ``controls`` is ``[B,2,M,H,W]``; ``positions`` is ``[B,M,H,W]``.
        Returns dense offsets ``[B,2,P,H,W]``.
        """
        base = self.control_base.view(1, self.control_points, 1, 1)
        mu = (base + positions / (self.control_points - 1)).clamp(-1.0, 1.0)
        xi = (self.axis / (self.kernel_size // 2)).view(
            1, 1, self.kernel_size, 1, 1)
        weights = (
            -(xi - mu.unsqueeze(2)).square() / (2.0 * self.sigma ** 2)
        ).softmax(1)
        dense = (controls.unsqueeze(3) * weights.unsqueeze(1)).sum(2)
        center = self.kernel_size // 2
        return dense - dense[:, :, center:center + 1]

    def scope(self) -> torch.Tensor:
        distance = self.axis.abs() / (self.kernel_size // 2)
        log_scale = (
            self.log_scope_base[:, None]
            + F.softplus(self.log_scope_amp)[:, None] * distance)
        return (log_scale.tanh() * math.log(4.0)).exp()

    def _sample(self, x: torch.Tensor, offsets: torch.Tensor,
                axis: int) -> torch.Tensor:
        batch, _, height, width = x.shape
        points = self.kernel_size
        spatial = (height, width)
        support = self.axis.view(1, points, 1, 1)
        scale = self.scope()[axis].view(1, points, 1, 1)
        order = [0, 2, 3]
        order.insert(axis + 2, 1)
        output_shape = [height, width]
        output_shape[axis] *= points
        coordinates = []
        for dimension, size in enumerate(spatial):
            shape = [1, 1, 1, 1]
            shape[dimension + 2] = size
            base = torch.arange(
                size, device=x.device, dtype=x.dtype).view(shape)
            if dimension == axis:
                delta = support
            else:
                delta = offsets[:, dimension] * scale
            coordinate = (base + delta).expand(batch, points, height, width)
            coordinate = coordinate.permute(order).reshape(
                batch, *output_shape)
            coordinates.append(
                2.0 * coordinate / max(size - 1, 1) - 1.0)
        grid = torch.stack(coordinates[::-1], dim=-1)
        return F.grid_sample(
            x, grid, mode='bilinear', padding_mode='border',
            align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f'expected [B,C,H,W], got {tuple(x.shape)}')
        predicted = self.offset_norm(self.offset_conv(x)).tanh()
        with torch.autocast(device_type=x.device.type, enabled=False):
            *controls, positions = predicted.float().split(
                self.control_points, dim=1)
            offsets = self.dense_offsets(
                torch.stack(controls, dim=1), positions)
            samples = [
                self._sample(x.float(), offsets, axis) for axis in range(2)]
        responses = [
            layer(sample.to(x.dtype))
            for layer, sample in zip(self.aggregators, samples)]
        return self.output_norm(self.fuse(torch.cat(responses, dim=1)))


class SDSConvResidual(nn.Module):
    """Bottleneck SDSConv residual on an image-embedding map."""

    def __init__(self, channels: int, hidden: int = 64,
                 kernel_size: int = 7):
        super().__init__()
        if hidden < 8:
            raise ValueError('SDSConv hidden must be >= 8')
        self.down = nn.Conv2d(channels, hidden, 1, bias=False)
        self.sds = SDSConv2d(hidden, kernel_size=kernel_size)
        self.up = nn.Conv2d(hidden, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.up(self.sds(self.down(x)))


class SDSConvSAM(ConvLoRASAM):
    """Conv-LoRA with a zero-initialized SDSConv image-embedding residual."""

    def __init__(self, base: ConvLoRASAM, hidden: int = 64,
                 kernel_size: int = 7):
        super().__init__(base.sam, moe_weight=base.moe_weight)
        neck = getattr(self.sam.image_encoder, 'neck', None)
        if neck is None or not hasattr(neck[0], 'out_channels'):
            raise RuntimeError('cannot infer SAM image-embedding channels')
        channels = int(neck[0].out_channels)
        self.sds_adapter = SDSConvResidual(
            channels, hidden=hidden, kernel_size=kernel_size)
        self._sds_handle = self.sam.image_encoder.register_forward_hook(
            self._sds_hook)

    def _sds_hook(self, module, inputs, output):
        del module, inputs
        if output.ndim != 4:
            raise RuntimeError(
                f'unexpected image embedding shape: {tuple(output.shape)}')
        return self.sds_adapter(output)


__all__ = ['SDSConv2d', 'SDSConvResidual', 'SDSConvSAM']
