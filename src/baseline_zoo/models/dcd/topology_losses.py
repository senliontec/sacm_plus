"""Binary topology losses for DCD.

These are local, binary-logit implementations.  ``SoftclDiceLoss`` is part of
the typed recipe (the centerline head uses it with
``--typed-cl-dice-weight``); Tversky and SkeletonRecall remain optional and
are disabled by default.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from scipy.ndimage import binary_dilation
except ImportError:  # pragma: no cover - optional dependency
    binary_dilation = None
try:
    from skimage.morphology import skeletonize
except ImportError:  # pragma: no cover - optional dependency
    skeletonize = None


class SoftclDiceLoss(nn.Module):
    """Binary soft-clDice loss on logits and a binary target."""

    def __init__(self, num_iter: int = 10, smooth: float = 1.0):
        super().__init__()
        self.num_iter = max(int(num_iter), 1)
        self.smooth = float(smooth)

    @staticmethod
    def _soft_erode(img: torch.Tensor) -> torch.Tensor:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    @staticmethod
    def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

    def _soft_open(self, img: torch.Tensor) -> torch.Tensor:
        return self._soft_dilate(self._soft_erode(img))

    def _soft_skel(self, img: torch.Tensor) -> torch.Tensor:
        skel = F.relu(img - self._soft_open(img))
        for _ in range(self.num_iter):
            img = self._soft_erode(img)
            delta = F.relu(img - self._soft_open(img))
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        skel_pred = self._soft_skel(pred)
        skel_target = self._soft_skel(target)
        tprec = ((skel_pred * target).sum() + self.smooth) / (
            skel_pred.sum() + self.smooth)
        tsens = ((skel_target * pred).sum() + self.smooth) / (
            skel_target.sum() + self.smooth)
        cldice = 1.0 - 2.0 * tprec * tsens / (tprec + tsens)
        return cldice


class TverskyLoss(nn.Module):
    """Binary Tversky loss; beta > alpha favours recall."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        target = target.clamp(0.0, 1.0)
        dims = (1, 2, 3)
        tp = (prob * target).sum(dim=dims)
        fp = (prob * (1.0 - target)).sum(dim=dims)
        fn = ((1.0 - prob) * target).sum(dim=dims)
        score = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - score.mean()


class SkeletonRecallLoss(nn.Module):
    """Binary skeleton-recall loss (no differentiable skeleton on GPU)."""

    def __init__(self, tube_radius: int = 2, smooth: float = 1e-6):
        super().__init__()
        if skeletonize is None or binary_dilation is None:
            raise ImportError(
                'SkeletonRecallLoss requires scikit-image and scipy')
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
        tube = self._tubes(target)
        if tube.sum() <= 0:
            return logits.sum() * 0.0
        recalled = (torch.sigmoid(logits) * tube).sum()
        return 1.0 - (recalled + self.smooth) / (tube.sum() + self.smooth)


__all__ = ['SoftclDiceLoss', 'SkeletonRecallLoss', 'TverskyLoss']
