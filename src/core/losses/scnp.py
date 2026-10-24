"""Same-Class Neighbor Penalization (SCNP), CVPR 2026.

SCNP is a training-time logit transform, not a segmentation model. It
replaces each pixel's logit with the *hardest same-class* logit in its
local neighborhood before the ordinary segmentation loss is applied:

- foreground pixels receive the minimum foreground logit in the window
  (a low-confidence foreground pixel is penalized by its confident
  foreground neighbors);
- background pixels receive the maximum background logit in the window
  (a background pixel next to confident foreground is pushed down).

The transform is parameter-free, adds no inference cost and is
orthogonal to the segmentation loss (CE, Dice, clDice, topology, ...).

Vendored from the official implementation (MIT License):

    baselines/SCNP-SameClassNeighborPenalization/SCNP/scnp_example.py
    https://github.com/jmlipman/SCNP-SameClassNeighborPenalization

Copyright (c) 2026 J. Miguel Valverde.

Deviation from the paper's reference snippet: ``neighborhood_size`` is
validated (positive odd integer) and ``SCNPCriterion`` supplies a small
binary wrapper for this project's ``DiceBCELoss``. The transform itself
is byte-for-byte equivalent to the official ``SCNP.forward``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCNP(nn.Module):
    """Official Same-Class Neighbor Penalization transform.

    Args:
        dimensions: ``"2D"`` or ``"3D"``.
        neighborhood_size: odd pooling window (official default: 3).
    """

    def __init__(self, dimensions, neighborhood_size=3):
        super().__init__()
        if neighborhood_size < 1 or neighborhood_size % 2 == 0:
            raise ValueError(
                f"neighborhood_size must be a positive odd integer, "
                f"got {neighborhood_size}")

        if dimensions == '2D':
            self.mp = F.max_pool2d
            self.ns = (neighborhood_size, neighborhood_size)
            self.st = (1, 1)
            self.pad = (neighborhood_size // 2, neighborhood_size // 2)
        elif dimensions == '3D':
            self.mp = F.max_pool3d
            self.ns = (neighborhood_size,) * 3
            self.st = (1, 1, 1)
            self.pad = (neighborhood_size // 2,) * 3
        else:
            raise ValueError(
                f"`dimensions` must be '2D' or '3D', got {dimensions!r}")

    def forward(self, logits, target):
        # Official contract: target is one-hot with the same shape as
        # logits. For binary segmentation a single-channel 0/1 target is
        # already one-hot (N=1).
        assert logits.shape == target.shape, (
            "`target` should be one-hot encoded and have the same shape "
            "as `logits`")

        # Min-pooling in the foreground: the 9999 sentinel makes
        # background neighbors lose the min/max competition.
        t1 = -self.mp(-(logits * target + 9999 * (1 - target)),
                      self.ns, self.st, self.pad)
        # Max-pooling in the background.
        t2 = self.mp(logits * (1 - target) - 9999 * target,
                     self.ns, self.st, self.pad)
        return t1 * target + t2 * (1 - target)


class SCNPCriterion(nn.Module):
    """Apply SCNP to logits, then the wrapped segmentation criterion.

    This is the official ``SCNP(CEDice)`` variant: only the transformed
    logits are optimized. Setting ``original_weight > 0`` additionally
    keeps the ordinary criterion on the raw logits (the alternative
    objective shown as a comment in the official example).

    Args:
        criterion: logits-in criterion (e.g. ``DiceBCELoss``).
        neighborhood_size: odd pooling window (official default: 3).
        original_weight: weight of the raw-logit criterion; 0 disables it
            (official default).
        dimensions: ``"2D"`` or ``"3D"``.
    """

    def __init__(self, criterion, neighborhood_size=3,
                 original_weight=0.0, dimensions='2D'):
        super().__init__()
        if original_weight < 0:
            raise ValueError(
                f"original_weight must be >= 0, got {original_weight}")
        self.criterion = criterion
        self.scnp = SCNP(dimensions, neighborhood_size)
        self.original_weight = float(original_weight)

    def forward(self, logits, target):
        # SCNP's masking arithmetic requires an exact 0/1 target; the
        # project's masks are binary, but threshold defensively so a
        # soft/eval target cannot leak into the 9999 sentinel path.
        target = (target > 0.5).to(dtype=logits.dtype)
        loss = self.criterion(self.scnp(logits, target), target)
        if self.original_weight > 0:
            loss = loss + self.original_weight * self.criterion(logits, target)
        return loss
