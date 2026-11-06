"""Typed Stage-2 heads and structure composition for SACM-STC.

The four native Stage-2 heads are fixed to:
  0 -> dense mask
  1 -> centerline
  2 -> boundary
  3 -> normalized distance/width

A tiny composer fuses the four outputs.  It adds no new task heads; it only
turns the four task outputs into one deployment mask.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Bounded width parameterisation, in pixels at the typed resolution.  The
# upper bound matches the largest dilation radius used by the composer.
WIDTH_MIN_PX = 0.5
WIDTH_MAX_PX = 14.0


def width_to_pixels(logits: torch.Tensor) -> torch.Tensor:
    """Map unconstrained width logits to a bounded pixel width."""
    return WIDTH_MIN_PX + (WIDTH_MAX_PX - WIDTH_MIN_PX) * torch.sigmoid(logits)


def width_to_logits(width_px: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`width_to_pixels`.

    The width loss is applied in this logit space.  Supervising the logit
    (rather than the saturated pixel value) keeps a constant gradient even
    when the head starts far outside the useful range.
    """
    frac = ((width_px - WIDTH_MIN_PX) / (WIDTH_MAX_PX - WIDTH_MIN_PX)).clamp(
        1e-4, 1.0 - 1e-4)
    return torch.logit(frac)

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
except ImportError:  # pragma: no cover
    binary_erosion = None
    distance_transform_edt = None
try:
    from skimage.morphology import skeletonize
except ImportError:  # pragma: no cover
    skeletonize = None


def typed_targets(mask: torch.Tensor, resolution: int):
    """Return skeleton, boundary and normalized width targets."""
    if skeletonize is None or distance_transform_edt is None:
        raise ImportError('typed targets require scikit-image and scipy')
    m = F.interpolate(mask.float(), (resolution, resolution), mode='nearest')
    values = m.detach().cpu().numpy() > 0.5
    skeletons, boundaries, widths = [], [], []
    for item in values:
        arr = item[0]
        sk = skeletonize(arr) if arr.any() else np.zeros_like(arr, bool)
        boundary = (arr & ~binary_erosion(arr)
                    if arr.any() else np.zeros_like(arr, bool))
        dist = (distance_transform_edt(arr)
                if arr.any() else np.zeros_like(arr, np.float32))
        skeletons.append(torch.from_numpy(sk.astype(np.float32)))
        boundaries.append(torch.from_numpy(boundary.astype(np.float32)))
        widths.append(torch.from_numpy(
            dist.astype(np.float32) / float(resolution)))
    return (
        torch.stack(skeletons, 0).unsqueeze(1),
        torch.stack(boundaries, 0).unsqueeze(1),
        torch.stack(widths, 0).unsqueeze(1),
    )


class TypedComposer(nn.Module):
    """Fuse dense / centerline / boundary / width into one mask."""

    def __init__(self):
        super().__init__()
        # Small fusion only; no additional semantic task heads.
        # Inputs: dense prob, centerline prob, boundary prob, normalized
        # width, and explicit dense/structure disagreement.
        self.gate = nn.Conv2d(5, 1, 1)
        self.edge = nn.Conv2d(5, 1, 1)
        nn.init.zeros_(self.gate.weight)
        # Neutral start: the gate can open or close within the official LR
        # budget.  A closed (-8) start cannot reach the open regime because
        # Adam moves each parameter by at most ~lr per step.
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.edge.weight)
        nn.init.zeros_(self.edge.bias)

    def forward(self, dense, centerline, boundary, width, resolution):
        # width logits -> bounded pixel width (never softplus-saturated).
        width_px = width_to_pixels(width)
        kernels = (1, 3, 5, 7, 11, 15)
        radii = (0.0, 1.5, 3.5, 6.0, 10.0, 14.0)
        center = torch.sigmoid(centerline)
        candidates = [
            center,
            *[F.max_pool2d(center, k, 1, k // 2) for k in kernels[1:]],
        ]
        scores = []
        for radius in radii:
            scores.append(-((width_px - radius) ** 2) / (2.0 * 2.0 ** 2))
        weights = F.softmax(torch.cat(scores, dim=1), dim=1)
        structure = sum(
            weights[:, i:i + 1] * candidates[i]
            for i in range(len(candidates)))
        structure = structure.clamp(1e-4, 1.0 - 1e-4)
        structure_logits = torch.logit(structure)

        # Bounded features keep one saturated head from dominating the gate.
        # The explicit disagreement channel gives the 1x1 gate the quantity
        # the supervision target is actually about.
        dense_prob = torch.sigmoid(dense)
        boundary_prob = torch.sigmoid(boundary)
        features = torch.cat([
            dense_prob,
            center,
            boundary_prob,
            width_px / WIDTH_MAX_PX,
            (dense_prob - structure).abs(),
        ], dim=1)
        gate_logits = self.gate(features)
        gate = torch.sigmoid(gate_logits)
        edge = self.edge(features) * boundary_prob
        final = dense + gate * (structure_logits - dense) + edge
        return {
            'dense': dense,
            'centerline': centerline,
            'boundary': boundary,
            'width': width,
            'structure': structure,
            'gate': gate,
            'gate_logits': gate_logits,
            'final': final,
        }


__all__ = [
    'TypedComposer',
    'typed_targets',
    'WIDTH_MIN_PX',
    'WIDTH_MAX_PX',
    'width_to_pixels',
    'width_to_logits',
]
