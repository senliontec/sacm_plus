"""Structure-Task Composition (STC) v2 head for SACM-STC.

High-resolution SAM features drive three structural tasks (centerline,
boundary, width bins).  A separate dense branch consumes the SACM-STC anchor.
The final mask is a gated composition with a boundary-aware edge residual.

The dense branch fuses all four Stage-2 candidate logits.  The dense
residual, edge residual and structure gate are zero/closed at initialisation,
so the untrained module reproduces the four-head mean before learning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
except ImportError:  # pragma: no cover - optional dependency
    binary_erosion = None
    distance_transform_edt = None
try:
    from skimage.morphology import skeletonize
except ImportError:  # pragma: no cover - optional dependency
    skeletonize = None


def structure_targets(mask: torch.Tensor, resolution: int):
    """Build skeleton, boundary and width-bin targets from a binary mask."""
    if skeletonize is None or distance_transform_edt is None:
        raise ImportError('STC targets require scikit-image and scipy')
    m = F.interpolate(mask.float(), (resolution, resolution), mode='nearest')
    values = m.detach().cpu().numpy() > 0.5
    skeletons, boundaries, width_bins = [], [], []
    for item in values:
        arr = item[0]
        sk = skeletonize(arr) if arr.any() else np.zeros_like(arr, bool)
        boundary = (arr & ~binary_erosion(arr)
                    if arr.any() else np.zeros_like(arr, bool))
        dist = (distance_transform_edt(arr)
                if arr.any() else np.zeros_like(arr, np.float32))
        bins = np.zeros(arr.shape, dtype=np.int64)
        bins[arr & (dist > 1.5)] = 1
        bins[arr & (dist > 3.5)] = 2
        skeletons.append(torch.from_numpy(sk.astype(np.float32)))
        boundaries.append(torch.from_numpy(boundary.astype(np.float32)))
        width_bins.append(torch.from_numpy(bins))
    return (
        torch.stack(skeletons, 0).unsqueeze(1),
        torch.stack(boundaries, 0).unsqueeze(1),
        torch.stack(width_bins, 0),
    )


class STCHead(nn.Module):
    """SAM-feature structural tasks + boundary-aware gated composition."""

    def __init__(self, feature_ch: int, width: int = 32):
        super().__init__()
        self.sam_proj = nn.Conv2d(feature_ch, width, 1)
        self.image_proj = nn.Conv2d(3, width, 3, padding=1)
        self.structure_trunk = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.dense_fuse = nn.Conv2d(4, 1, 1)
        self.dense_trunk = nn.Sequential(
            nn.Conv2d(width + 4, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.dense = nn.Conv2d(width, 1, 1)
        self.centerline = nn.Conv2d(width, 1, 1)
        self.boundary = nn.Conv2d(width, 1, 1)
        self.width_bins = nn.Conv2d(width, 3, 1)
        self.gate = nn.Conv2d(width, 1, 1)
        self.edge = nn.Sequential(
            nn.Conv2d(width + 2, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv2d(width, 1, 1),
        )
        self.edge_gate = nn.Conv2d(width, 1, 1)

        # Identity start: no residual, closed structure/edge gates.
        nn.init.constant_(self.dense_fuse.weight, 0.25)
        nn.init.zeros_(self.dense_fuse.bias)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -6.0)
        nn.init.zeros_(self.edge[-1].weight)
        nn.init.zeros_(self.edge[-1].bias)
        nn.init.zeros_(self.edge_gate.weight)
        nn.init.constant_(self.edge_gate.bias, -6.0)

    def forward(self, anchor_logits: torch.Tensor, image: torch.Tensor,
                sam_features: torch.Tensor):
        size = anchor_logits.shape[-2:]
        if image.shape[-2:] != size:
            image = F.interpolate(
                image, size=size, mode='bilinear', align_corners=False)
        features = self.sam_proj(sam_features)
        if features.shape[-2:] != size:
            features = F.interpolate(
                features, size=size, mode='bilinear', align_corners=False)
        features = features + self.image_proj(image)

        # Structural tasks see SAM features only, never the dense anchor.
        structural = self.structure_trunk(features)
        center_logits = self.centerline(structural)
        boundary_logits = self.boundary(structural)
        width_logits = self.width_bins(structural)

        # Dense residual is allowed to use the anchor.
        dense_features = self.dense_trunk(
            torch.cat([features, anchor_logits], dim=1))
        refined = self.dense_fuse(anchor_logits) + self.dense(dense_features)

        center = torch.sigmoid(center_logits)
        width_weights = F.softmax(width_logits, dim=1)
        dilations = [
            center,
            F.max_pool2d(center, 3, 1, 1),
            F.max_pool2d(center, 5, 1, 2),
            F.max_pool2d(center, 7, 1, 3),
        ]
        structure = (
            width_weights[:, 0:1] * dilations[0]
            + width_weights[:, 1:2] * dilations[2]
            + width_weights[:, 2:3] * dilations[3]
        )
        structure = structure.clamp(1e-4, 1.0 - 1e-4)
        structure_logits = torch.logit(structure)

        gate_logits = self.gate(dense_features)
        gate = torch.sigmoid(gate_logits)
        boundary_prob = torch.sigmoid(boundary_logits)
        edge_residual = self.edge(
            torch.cat([features, structure, boundary_prob], dim=1))
        edge_gate = torch.sigmoid(self.edge_gate(dense_features))

        final_logits = (
            refined
            + gate * (structure_logits - refined)
            + edge_gate * edge_residual * boundary_prob
        )
        return {
            'refined': refined,
            'centerline': center_logits,
            'boundary': boundary_logits,
            'width_bins': width_logits,
            'structure': structure,
            'gate': gate,
            'gate_logits': gate_logits,
            'edge_residual': edge_residual,
            'final': final_logits,
        }


__all__ = ['STCHead', 'structure_targets']
