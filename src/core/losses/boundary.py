"""Boundary / HD95-oriented loss for thin curvilinear structures.

The existing Dice/clDice objective rewards overlap and connectivity but does
not directly penalise predictions far from the annotated boundary.  This
module adds a light Kervadec-style distance-map term:

    phi_G(x) = dist(x, foreground) - dist(x, background)
    L_b = sum(phi_G(x) * p(x)) / (sum(p(x)) + eps)

Inside the GT phi_G is negative and outside it is positive, so the loss is
minimised by placing probability mass inside the annotation.  The maps are
computed on a downsampled grid (default 512) and detached from autograd;
only ``p`` receives gradients.  Empty GT slices are skipped.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:  # pragma: no cover - scipy is a project dependency
    distance_transform_edt = None


class BoundaryLoss(nn.Module):
    """Distance-transform boundary loss on binary/soft targets."""

    def __init__(self, resolution: int = 512, eps: float = 1e-6):
        super().__init__()
        self.resolution = int(resolution)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if distance_transform_edt is None:
            return logits.sum() * 0.0
        if logits.shape != target.shape:
            raise ValueError(
                f"BoundaryLoss shape mismatch: {logits.shape} vs {target.shape}")
        if self.resolution and logits.shape[-1] > self.resolution:
            size = (self.resolution, self.resolution)
            logits = F.interpolate(
                logits, size=size, mode='bilinear', align_corners=False)
            target = F.interpolate(
                target, size=size, mode='bilinear', align_corners=False)

        prob = torch.sigmoid(logits)
        losses = []
        for b in range(prob.shape[0]):
            gt = (target[b, 0].detach().cpu().numpy() > 0.5)
            if not gt.any():
                continue
            outside = distance_transform_edt(~gt)
            inside = distance_transform_edt(gt)
            phi = outside - inside
            scale = max(float(np.abs(phi).max()), 1.0)
            phi = phi / scale
            phi_t = torch.from_numpy(phi).to(
                device=prob.device, dtype=prob.dtype)
            p = prob[b, 0]
            losses.append(
                (phi_t * p).sum() / (p.sum() + self.eps))
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()
