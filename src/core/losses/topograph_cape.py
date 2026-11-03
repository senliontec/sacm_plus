"""Combined Topograph + CAPE-style topology objective.

Topograph enforces component/hole topology with strict guarantees; CAPE
adds dense path-level gradients along GT centerlines.  They target
different failure modes and can be used together with one small
``--topology_loss_weight``.
"""

from __future__ import annotations

import torch
from torch import nn

from .cape import CAPELoss
from .registry import register
from .topograph import TopographLossAdapter


class TopographCAPELoss(nn.Module):
    """Weighted sum of Topograph and CAPE-style path loss."""

    def __init__(
        self,
        resolution: int = 256,
        cape_weight: float = 0.25,
        cape_resolution: int | None = 128,
        **_: object,
    ) -> None:
        super().__init__()
        self.topograph = TopographLossAdapter(resolution=resolution)
        self.cape = CAPELoss(resolution=cape_resolution)
        self.cape_weight = float(cape_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.topograph(logits, target) + \
            self.cape_weight * self.cape(logits, target)


register('topograph_cape')(TopographCAPELoss)
