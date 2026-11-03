"""Skeleton Recall Loss for thin curvilinear structures.

Lightweight, architecture-agnostic alternative/complement to clDice:
the ground-truth skeleton is computed once on CPU, dilated into a tube,
and the prediction is encouraged to recall foreground inside that tube.
No differentiable skeletonization is needed on the GPU.

This implements the loss from:
    Kirchhoff et al., "Skeleton Recall Loss for Connectivity Conserving
    and Resource Efficient Segmentation of Thin Tubular Structures",
    ECCV 2024.

It is registered as ``skeleton_recall`` so it can be selected with
``--topology_loss skeleton_recall``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from skimage.morphology import skeletonize
except ImportError:  # pragma: no cover - environment dependent
    skeletonize = None
try:
    from scipy.ndimage import binary_dilation
except ImportError:  # pragma: no cover
    binary_dilation = None

from .registry import register


class SkeletonRecallLoss(nn.Module):
    """Soft recall of the prediction inside a tubed GT skeleton.

    Args:
        resolution: work resolution for skeletonization; None keeps input.
        tube_radius: dilation radius (pixels) around the GT skeleton.
        smooth: numerical stabilizer.
    """

    def __init__(
        self,
        resolution: int | None = 256,
        tube_radius: int = 2,
        smooth: float = 1e-6,
        **_: object,
    ) -> None:
        super().__init__()
        if skeletonize is None or binary_dilation is None:
            raise ImportError(
                'SkeletonRecallLoss requires scikit-image and scipy')
        self.resolution = resolution
        self.tube_radius = int(tube_radius)
        self.smooth = float(smooth)

    def _tubes(self, target: torch.Tensor) -> torch.Tensor:
        masks = target.detach().cpu().numpy() > 0.5
        tubes = []
        for mask in masks:
            golden = skeletonize(mask[0])
            if golden.any():
                tube = binary_dilation(golden, iterations=self.tube_radius)
            else:
                tube = np.zeros_like(golden, dtype=bool)
            tubes.append(torch.from_numpy(tube.astype(np.float32)))
        return torch.stack(tubes, dim=0).unsqueeze(1).to(target.device)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or target.ndim != 4:
            raise ValueError('logits and target must be [B,1,H,W]')
        if self.resolution is not None and (
                logits.shape[-1] != self.resolution
                or logits.shape[-2] != self.resolution):
            logits = F.interpolate(
                logits, size=(self.resolution, self.resolution),
                mode='bilinear', align_corners=False)
            target = F.interpolate(
                target, size=(self.resolution, self.resolution),
                mode='bilinear', align_corners=False)
        tube = self._tubes(target)
        if tube.sum() <= 0:
            return logits.sum() * 0.0
        prob = torch.sigmoid(logits)
        recalled = (prob * tube).sum()
        return 1.0 - (recalled + self.smooth) / (tube.sum() + self.smooth)


register('skeleton_recall')(SkeletonRecallLoss)
