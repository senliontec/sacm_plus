"""Combined Topograph + Skeleton Recall topology objective.

Topograph handles component/hole errors; Skeleton Recall handles thin
branch connectivity.  The two are complementary and cheaper than the
persistent-homology family.
"""

from __future__ import annotations

import torch
from torch import nn

from .registry import register
from .skeleton_recall import SkeletonRecallLoss
from .topograph import TopographLossAdapter


class TopographSkeletonLoss(nn.Module):
    def __init__(
        self,
        resolution: int = 256,
        skeleton_weight: float = 0.5,
        skeleton_resolution: int | None = 256,
        **_: object,
    ) -> None:
        super().__init__()
        self.topograph = TopographLossAdapter(resolution=resolution)
        self.skeleton = SkeletonRecallLoss(resolution=skeleton_resolution)
        self.skeleton_weight = float(skeleton_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.topograph(logits, target) + \
            self.skeleton_weight * self.skeleton(logits, target)


register('topograph_skeleton')(TopographSkeletonLoss)
