"""A2SAM losses.

``IOU`` matches the release (with a zero-union guard for robustness).

``ThicknessAwareLoss`` defaults to the paper's Eq. (24)-(30):

* ``W = log(1 + 1/(T + eps))``
* ``C = mean(grad_h * W + grad_v * W)`` with per-pixel weights
* denominator ``max(1, foreground count)``

``mode='release'`` reproduces the released implementation's reciprocal
clamped weight, adjacent-weight products and all-pixel denominator.  The
plus-shaped opening is kept from the release in both modes because the paper
does not specify its implementation and the release is the only executable
reference for that preprocessing detail.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


class IOU(nn.Module):
    """Official IoU loss with a zero-union guard."""

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        return (1 - inter / union.clamp_min(self.eps)).mean()


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
    def __init__(self, eps: float = 1e-6, mode: str = 'paper'):
        super().__init__()
        if mode not in {'paper', 'release'}:
            raise ValueError(f'unknown thickness-loss mode: {mode!r}')
        self.eps = float(eps)
        self.mode = mode

    def estimate_thin_vessel_weight(self, lab: torch.Tensor) -> torch.Tensor:
        weights = []
        for index in range(lab.shape[0]):
            mask = lab[index, 0].detach().cpu().numpy() > 0.5
            distance = distance_transform_edt(mask)
            skeleton = skeletonize(mask)
            thickness = distance * skeleton
            weight = torch.from_numpy(thickness)
            if self.mode == 'paper':
                weight = torch.log1p(1.0 / (weight + self.eps))
            else:
                weight = (1.0 / (weight + self.eps)).clamp(max=10.0)
            weights.append(weight.float().unsqueeze(0).unsqueeze(0)
                           .to(lab.device))
        return torch.cat(weights, dim=0)

    def smoothness_penalty(self, tensor: torch.Tensor,
                           weight: torch.Tensor,
                           mode: str) -> torch.Tensor:
        grad_x = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1])
        grad_y = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
        if mode == 'paper':
            # Eq. (28): each gradient is weighted by W at its source pixel.
            weight_x = weight[:, :, :, :-1]
            weight_y = weight[:, :, :-1, :]
        else:
            weight_x = weight[:, :, :, 1:] * weight[:, :, :, :-1]
            weight_y = weight[:, :, 1:, :] * weight[:, :, :-1, :]
        return (grad_x * weight_x).mean() + (grad_y * weight_y).mean()

    def forward(self, img: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        foreground = (lab > 0.5).sum()
        if foreground == 0:
            return img.sum() * 0.0
        opened_img = _plus_opening(img)
        opened_lab = _plus_opening(lab)
        img = torch.sigmoid(10 * (opened_img - 0.5))
        lab_soft = torch.sigmoid(10 * (opened_lab - 0.5))
        weight = self.estimate_thin_vessel_weight(lab_soft)
        smooth_img = self.smoothness_penalty(img, weight, self.mode)
        smooth_lab = self.smoothness_penalty(lab_soft, weight, self.mode)
        if self.mode == 'paper':
            denominator = foreground.clamp_min(1).to(smooth_img.dtype)
        else:
            denominator = torch.count_nonzero(lab_soft).to(smooth_img.dtype)
        return torch.abs(smooth_img - smooth_lab) / denominator
