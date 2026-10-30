"""Field-conditioned FiLM adapter for CurvField.

The v1 anisotropic strip branch and scalar gate were removed after diagnosis:
the gate saturated near 0.49, an oracle GT field changed segmentation by less
than 0.3 pp, and the strip branch was worse than the local branch on LEAF/XCAD.
The v2 adapter keeps the local depthwise branch and conditions it directly with
a field-driven FiLM scale/shift.  The residual is still zero-initialized, so a
freshly built CurvField is exactly the Conv-LoRA baseline.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class StructureConditionedAdapter(nn.Module):
    """Field-conditioned residual adapter with a local depthwise branch."""

    def __init__(self, dim: int, hidden: int = 64,
                 film_scale: float = 0.5):
        super().__init__()
        if film_scale <= 0:
            raise ValueError('film_scale must be positive')
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.film_scale = float(film_scale)
        self.down = nn.Linear(dim, hidden, bias=False)
        self.local = nn.Conv2d(
            hidden, hidden, 5, padding=2, groups=hidden, bias=False)
        # Direct field -> per-channel scale/shift.  Small random init keeps the
        # field path alive while gamma=0 still gives exact baseline identity.
        self.film = nn.Conv2d(6, 2 * hidden, 1)
        nn.init.normal_(self.film.weight, std=0.01)
        nn.init.zeros_(self.film.bias)
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
        norm = torch.sqrt(cos2 * cos2 + sin2 * sin2).clamp_min(1e-6)
        cos2 = cos2 / norm
        sin2 = sin2 / norm
        branch = torch.sigmoid(field[:, 4:5])
        coherence = torch.sigmoid(field[:, 5:6])
        condition = torch.cat(
            [center, radius, cos2, sin2, branch, coherence], dim=1)
        film = self.film(condition)
        scale, shift = film.chunk(2, dim=1)
        scale = torch.tanh(scale) * self.film_scale
        shift = torch.tanh(shift) * self.film_scale
        fused = local * (1.0 + scale) + shift

        out = self.up(fused.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2).contiguous()
        out = self.gamma * out
        if nhwc:
            out = out.permute(0, 2, 3, 1)
        return tokens + out


__all__ = ['StructureConditionedAdapter']
