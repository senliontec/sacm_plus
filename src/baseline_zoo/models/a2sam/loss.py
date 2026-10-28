"""A2SAM losses: weighted IoU and the thickness-aware connectivity term.

Adapted from the released ``models/iou_ta_loss.py``.  The official function
uses Kornia's plus-kernel morphological opening and a CUDA-only kernel; this
port implements the same plus-shaped opening with local pooling and derives
the EDT/skeleton weights with SciPy/scikit-image.  The official divisor is
the number of non-zero pixels after the sigmoid (i.e. every pixel); this port
uses the foreground pixel count, which is the quantity the term is intended
to normalize by.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


class IOU(nn.Module):
    """Official IoU loss (1 - mean IoU over the batch)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        return (1 - inter / union).mean()


def _plus_erode(x: torch.Tensor) -> torch.Tensor:
    padded = F.pad(x, (1, 1, 1, 1))
    values = [
        padded[:, :, 1:-1, 1:-1],
        padded[:, :, :-2, 1:-1],
        padded[:, :, 2:, 1:-1],
        padded[:, :, 1:-1, :-2],
        padded[:, :, 1:-1, 2:],
    ]
    return torch.stack(values, dim=0).amin(dim=0)


def _plus_dilate(x: torch.Tensor) -> torch.Tensor:
    padded = F.pad(x, (1, 1, 1, 1))
    values = [
        padded[:, :, 1:-1, 1:-1],
        padded[:, :, :-2, 1:-1],
        padded[:, :, 2:, 1:-1],
        padded[:, :, 1:-1, :-2],
        padded[:, :, 1:-1, 2:],
    ]
    return torch.stack(values, dim=0).amax(dim=0)


def _plus_opening(x: torch.Tensor) -> torch.Tensor:
    return _plus_dilate(_plus_erode(x))


class ThicknessAwareLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def estimate_thin_vessel_weight(self, lab: torch.Tensor) -> torch.Tensor:
        weights = []
        for index in range(lab.shape[0]):
            mask = lab[index, 0].detach().cpu().numpy() > 0.5
            if not mask.any():
                weights.append(torch.zeros_like(lab[index:index + 1]))
                continue
            distance = distance_transform_edt(mask)
            skeleton = skeletonize(mask)
            thickness = distance * skeleton
            weight = 1.0 / (thickness + self.eps)
            weight = torch.from_numpy(weight).clamp(max=10.0)
            weights.append(weight.unsqueeze(0).unsqueeze(0).to(lab.device))
        return torch.cat(weights, dim=0)

    def smoothness_penalty(self, tensor: torch.Tensor,
                           weight: torch.Tensor | None = None):
        grad_x = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1])
        grad_y = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
        if weight is not None:
            weight_x = weight[:, :, :, 1:] * weight[:, :, :, :-1]
            weight_y = weight[:, :, 1:, :] * weight[:, :, :-1, :]
            return ((grad_x * weight_x).mean()
                    + (grad_y * weight_y).mean())
        return grad_x.mean() + grad_y.mean()

    def forward(self, img: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        img = _plus_opening(img)
        lab = _plus_opening(lab)
        img = torch.sigmoid(10 * (img - 0.5))
        lab_soft = torch.sigmoid(10 * (lab - 0.5))
        weight = self.estimate_thin_vessel_weight(lab_soft)
        smooth_img = self.smoothness_penalty(img, weight)
        smooth_lab = self.smoothness_penalty(lab_soft, weight)
        foreground = (lab > 0.5).sum().clamp_min(1)
        return torch.abs(smooth_img - smooth_lab) / foreground
