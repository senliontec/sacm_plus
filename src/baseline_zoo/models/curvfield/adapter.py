"""Structure-conditioned adapter for CurvField.

The adapter has two branches:

- a small isotropic depthwise convolution for local detail;
- a long anisotropic branch that samples K points along the predicted
  tangent, with a step proportional to the predicted local radius.

The field gates the two branches and the residual is zero-initialized, so
a freshly built CurvField is exactly the Conv-LoRA baseline.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class StructureConditionedAdapter(nn.Module):
    """Field-conditioned residual adapter."""

    def __init__(self, dim: int, hidden: int = 64, k: int = 7,
                 max_step: float = 4.0):
        super().__init__()
        if k < 3 or k % 2 == 0:
            raise ValueError('k must be an odd integer >= 3')
        if max_step < 1:
            raise ValueError('max_step must be >= 1')
        self.dim = int(dim)
        self.k = int(k)
        self.max_step = float(max_step)
        self.down = nn.Linear(dim, hidden, bias=False)
        self.local = nn.Conv2d(
            hidden, hidden, 5, padding=2, groups=hidden, bias=False)
        self.strip = nn.Conv3d(
            hidden, hidden, (1, 1, self.k),
            padding=(0, 0, self.k // 2), groups=hidden, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        self.up = nn.Linear(hidden, dim, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _to_map(self, tokens: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Return NCHW feature map and whether the input was NHWC."""
        if tokens.ndim == 4:
            if tokens.shape[-1] == self.dim:
                return tokens.permute(0, 3, 1, 2).contiguous(), True
            if tokens.shape[1] == self.dim:
                return tokens, False
            raise ValueError(
                f'token channel mismatch: expected {self.dim}, got '
                f'{tuple(tokens.shape)}')
        if tokens.ndim != 3:
            raise ValueError(
                f'expected tokens [B,H,W,C] or [B,N,C], got '
                f'{tuple(tokens.shape)}')
        batch, count, dim = tokens.shape
        side = int(round(math.sqrt(count)))
        if side * side != count:
            raise ValueError('non-square token grid is not supported')
        return tokens.reshape(batch, side, side, dim).permute(
            0, 3, 1, 2).contiguous(), True

    def forward(self, tokens: torch.Tensor,
                field: torch.Tensor | None) -> torch.Tensor:
        if field is None:
            return tokens
        feature, nhwc = self._to_map(tokens)
        batch, _, height, width = feature.shape
        if field.shape[0] != batch:
            raise ValueError('field/token batch mismatch')
        if field.shape[-2:] != (height, width):
            field = F.interpolate(
                field, size=(height, width), mode='bilinear',
                align_corners=False)

        hidden = self.down(feature.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2).contiguous()
        local = self.local(hidden)

        center = torch.sigmoid(field[:, 0:1])
        radius = torch.sigmoid(field[:, 1:2])
        cos2 = field[:, 2:3]
        sin2 = field[:, 3:4]
        coherence = torch.sigmoid(field[:, 5:6])
        theta = 0.5 * torch.atan2(sin2, cos2)
        direction_x = torch.cos(theta)
        direction_y = torch.sin(theta)
        step = 1.0 + (self.max_step - 1.0) * radius

        ys = torch.arange(height, device=tokens.device, dtype=tokens.dtype)
        xs = torch.arange(width, device=tokens.device, dtype=tokens.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        base_x = grid_x / max(width - 1, 1) * 2.0 - 1.0
        base_y = grid_y / max(height - 1, 1) * 2.0 - 1.0
        base_x = base_x.view(1, 1, height, width)
        base_y = base_y.view(1, 1, height, width)

        offsets = torch.arange(
            self.k, device=tokens.device, dtype=tokens.dtype)
        offsets = offsets - (self.k // 2)
        offsets = offsets.view(1, 1, 1, 1, self.k)
        dx = 2.0 * step * direction_x / max(width - 1, 1)
        dy = 2.0 * step * direction_y / max(height - 1, 1)
        sample_x = base_x.unsqueeze(-1) + offsets * dx.unsqueeze(-1)
        sample_y = base_y.unsqueeze(-1) + offsets * dy.unsqueeze(-1)
        grid = torch.stack([sample_x, sample_y], dim=-1)
        grid = grid.reshape(batch, height, width * self.k, 2)
        sampled = F.grid_sample(
            hidden, grid, mode='bilinear', padding_mode='border',
            align_corners=True)
        sampled = sampled.reshape(
            batch, hidden.shape[1], height, width, self.k)
        strip = self.strip(sampled).mean(dim=-1)

        gate_in = torch.cat([center, radius, coherence], dim=1)
        gate = torch.sigmoid(self.gate(gate_in))
        fused = gate * strip + (1.0 - gate) * local
        out = self.up(fused.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2).contiguous()
        out = self.gamma * out

        if nhwc:
            out = out.permute(0, 2, 3, 1)
        return tokens + out


__all__ = ['StructureConditionedAdapter']
