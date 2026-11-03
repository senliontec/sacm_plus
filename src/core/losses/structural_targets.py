"""Targets for the v3 structural auxiliary heads.

Given a binary/soft GT mask at the training resolution (1024), produce:

- centerline: 1-px skeleton of the foreground;
- boundary:   1-px inner boundary (foreground minus its 3x3 erosion);
- width:      half-width in pixels (2 * EDT), normalized to [0, 1].

The GT is downsampled with max-pooling / nearest-neighbour, never bilinear:
a 1-px membrane at 1024 would otherwise disappear at 512 (see
``topology_utils.py``).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
except ImportError:  # pragma: no cover - environment dependent
    binary_erosion = None
    distance_transform_edt = None

try:
    from skimage.morphology import skeletonize
except ImportError:  # pragma: no cover - environment dependent
    skeletonize = None

from .topology_utils import resize_target_for_topology


def build_structural_targets(
    gt: torch.Tensor,
    resolution: int,
    width_max: float = 16.0,
) -> Dict[str, torch.Tensor]:
    """Build centerline / boundary / width targets.

    Args:
        gt: ``[B, 1, H, W]`` mask in ``[0, 1]``.
        resolution: output spatial size.
        width_max: normalization constant for the width target.
    Returns:
        dict with ``centerline``, ``boundary`` and ``width`` tensors of
        shape ``[B, 1, resolution, resolution]`` on the GT device.
    """
    if skeletonize is None or binary_erosion is None:
        raise ImportError(
            'structural targets require scikit-image and scipy')
    if gt.ndim != 4:
        raise ValueError('gt must be [B, C, H, W]')

    # Max-pool/nearest keeps 1-px structures; threshold after resize.
    resized = resize_target_for_topology(gt, resolution)
    binary = (resized > 0.5).detach().cpu().numpy()

    centerlines = []
    boundaries = []
    widths = []
    for mask in binary[:, 0]:
        if mask.any():
            skel = skeletonize(mask)
            erode = binary_erosion(mask)
            boundary = mask & ~erode
            width = 2.0 * distance_transform_edt(mask)
            width = np.clip(width / max(width_max, 1e-6), 0.0, 1.0)
        else:
            skel = np.zeros_like(mask, dtype=bool)
            boundary = np.zeros_like(mask, dtype=bool)
            width = np.zeros_like(mask, dtype=np.float32)
        centerlines.append(torch.from_numpy(skel.astype(np.float32)))
        boundaries.append(torch.from_numpy(boundary.astype(np.float32)))
        widths.append(torch.from_numpy(width.astype(np.float32)))

    device = gt.device
    return {
        'centerline': torch.stack(centerlines).unsqueeze(1).to(device),
        'boundary': torch.stack(boundaries).unsqueeze(1).to(device),
        'width': torch.stack(widths).unsqueeze(1).to(device),
    }
