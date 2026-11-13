"""Zero-initialised linear fusion of the four native Stage-2 heads.

This is the only auxiliary module used by the canonical SSH recipe.  It
starts as the identity on head 0, so enabling it cannot change the deployed
head at initialisation.  The state-dict keys (``fuse.weight``/``fuse.bias``)
match the original linear fusion module.
"""

from __future__ import annotations

import torch
from torch import nn


class HeadFusion(nn.Module):
    """``out = z0 + conv([z1-z0, z2-z0, z3-z0])`` with a 1x1 conv."""

    def __init__(self) -> None:
        super().__init__()
        self.fuse = nn.Conv2d(3, 1, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, heads: torch.Tensor) -> torch.Tensor:
        if heads.ndim != 4 or heads.shape[1] != 4:
            raise ValueError(
                'head fusion expects four Stage-2 heads [B, 4, H, W], got '
                f'{tuple(heads.shape)}')
        base = heads[:, 0:1]
        residual = torch.cat(
            [heads[:, 1:2] - base, heads[:, 2:3] - base,
             heads[:, 3:4] - base], dim=1)
        return base + self.fuse(residual)
