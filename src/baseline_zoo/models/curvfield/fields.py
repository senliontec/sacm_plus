"""Structure-field targets for CurvField.

The field is computed from training masks only:

    c       centerline probability (GT skeleton)
    r       local radius normalized to [0, 1]
    cos2t   cos(2 * tangent angle)
    sin2t   sin(2 * tangent angle)
    b       branch-point probability
    kappa   orientation coherence

Targets are intentionally low-dimensional and geometry-only.  No image
appearance, dataset name, or test label enters this module.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter
from skimage.morphology import skeletonize


def _single_mask_fields(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return six field targets and a foreground mask for one 2D mask."""
    mask = mask.astype(bool)
    height, width = mask.shape
    fg = mask.astype(np.float32)

    empty = np.zeros((6, height, width), dtype=np.float32)
    if not mask.any():
        return empty, fg[None]

    skeleton = skeletonize(mask)
    centerline = skeleton.astype(np.float32)

    # 8-neighbour degree of each skeleton pixel; branch points have >2.
    # Exclude the centre pixel so a straight skeleton pixel has degree 2.
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    degree = convolve(skeleton.astype(np.uint8), kernel, mode='constant')
    branch = (skeleton & (degree > 2)).astype(np.float32)

    edt = distance_transform_edt(mask).astype(np.float32)
    radius = edt / max(1.0, 0.5 * min(height, width))
    radius = np.clip(radius, 0.0, 1.0)

    # Structure tensor of the binary mask gives a local tangent orientation.
    # The major eigenvector of the gradient tensor is normal to a thin line,
    # so the tangent is shifted by pi/2 (double angle gets a minus sign).
    gx, gy = np.gradient(fg)
    jxx = gaussian_filter(gx * gx, sigma=1.0)
    jyy = gaussian_filter(gy * gy, sigma=1.0)
    jxy = gaussian_filter(gx * gy, sigma=1.0)
    trace = jxx + jyy
    denom = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) + 1e-6
    cos2t = -(jxx - jyy) / denom
    sin2t = -(2.0 * jxy) / denom
    coherence = denom / (trace + 1e-6)
    coherence = np.clip(coherence, 0.0, 1.0).astype(np.float32)
    # Orientation is only meaningful near the structure.
    support = (centerline + branch > 0).astype(np.float32)
    cos2t = (cos2t * support).astype(np.float32)
    sin2t = (sin2t * support).astype(np.float32)

    fields = np.stack(
        [centerline, radius, cos2t, sin2t, branch, coherence], axis=0)
    return fields, fg[None]


def structure_targets(masks: torch.Tensor, size: int = 64):
    """Build field targets ``[B,6,S,S]`` and foreground ``[B,1,S,S]``.

    ``masks`` is a ``[B,1,H,W]`` float/bool tensor on any device.  The mask
    is resized to ``size`` with nearest interpolation before skeleton/EDT
    computation, which keeps this cheap enough to run inside the 18-image
    training loop.  No test mask is ever passed here.
    """
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(
            f'expected masks [B,1,H,W], got {tuple(masks.shape)}')
    size = int(size)
    if size < 8:
        raise ValueError('field target size must be >= 8')

    small = F.interpolate(
        masks.detach().float(), size=(size, size), mode='nearest')
    targets, foreground = [], []
    for index in range(small.shape[0]):
        fields, fg = _single_mask_fields(
            (small[index, 0] > 0.5).cpu().numpy())
        targets.append(torch.from_numpy(fields))
        foreground.append(torch.from_numpy(fg))
    return (torch.stack(targets).to(masks.device),
            torch.stack(foreground).to(masks.device))


def _soft_dice_loss(prob: torch.Tensor, target: torch.Tensor,
                    eps: float = 1.0) -> torch.Tensor:
    """Per-image soft Dice loss; stable for sparse centerline targets."""
    dims = tuple(range(1, prob.ndim))
    intersection = (prob * target).sum(dims)
    denominator = prob.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * intersection + eps) /
            (denominator + eps)).mean()


def _weighted_corr_loss(pred: torch.Tensor, target: torch.Tensor,
                        weight: torch.Tensor) -> torch.Tensor:
    """1 - weighted Pearson correlation; zero when variance is absent."""
    denominator = weight.sum().clamp_min(1.0)
    pred_mean = (pred * weight).sum() / denominator
    target_mean = (target * weight).sum() / denominator
    pred_centered = (pred - pred_mean) * weight
    target_centered = (target - target_mean) * weight
    pred_var = (pred_centered * pred_centered).sum() / denominator
    target_var = (target_centered * target_centered).sum() / denominator
    covariance = (pred_centered * target_centered).sum() / denominator
    correlation = covariance / (
        pred_var.clamp_min(1e-12).sqrt() *
        target_var.clamp_min(1e-12).sqrt())
    return 1.0 - correlation.clamp(-1.0, 1.0)


def field_loss(pred: torch.Tensor, target: torch.Tensor,
               foreground: torch.Tensor) -> torch.Tensor:
    """Supervised field loss for ``[B,6,S,S]`` predictions.

    Channel order: centerline, radius, cos2t, sin2t, branch, coherence.

    Centerline/branch add a soft-Dice term to the class-weighted BCE so the
    sparse channels are calibrated to cross 0.5 on structure.  Radius adds a
    foreground correlation term because plain L1 can collapse to a constant.
    Coherence is supervised on foreground pixels only; elsewhere its target is
    not meaningful.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f'field shape mismatch: {tuple(pred.shape)} vs '
            f'{tuple(target.shape)}')
    c_pred, r_pred, cos_pred, sin_pred, b_pred, k_pred = pred.unbind(1)
    c_t, r_t, cos_t, sin_t, b_t, k_t = target.unbind(1)
    fg = foreground[:, 0]

    pos_weight = torch.tensor(5.0, device=pred.device, dtype=pred.dtype)
    loss_c = (
        F.binary_cross_entropy_with_logits(c_pred, c_t, pos_weight=pos_weight)
        + _soft_dice_loss(torch.sigmoid(c_pred), c_t)
    )
    loss_b = 0.5 * (
        F.binary_cross_entropy_with_logits(b_pred, b_t, pos_weight=pos_weight)
        + _soft_dice_loss(torch.sigmoid(b_pred), b_t)
    )

    radius_pred = torch.sigmoid(r_pred)
    radius_weight = fg / fg.sum().clamp_min(1.0)
    loss_r = ((radius_pred - r_t).abs() * radius_weight).sum() / (
        radius_weight.sum().clamp_min(1.0))
    if fg.sum() >= 2:
        loss_r = loss_r + 0.1 * _weighted_corr_loss(radius_pred, r_t, fg)

    # Orientations are double-angle unit vectors.  The target is a unit
    # vector whose reliability is given by kappa; junctions have low kappa.
    pred_vec = torch.stack([cos_pred, sin_pred], dim=1)
    pred_vec = pred_vec / pred_vec.norm(dim=1, keepdim=True).clamp_min(1e-6)
    target_vec = torch.stack([cos_t, sin_t], dim=1)
    target_vec = target_vec / target_vec.norm(dim=1, keepdim=True).clamp_min(
        1e-6)
    dot = (pred_vec * target_vec).sum(1).clamp(-1.0, 1.0)
    orient_weight = k_t * (c_t + b_t).clamp(0.0, 1.0)
    loss_orient = ((1.0 - dot) * orient_weight).sum() / (
        orient_weight.sum().clamp_min(1.0))

    kappa_bce = F.binary_cross_entropy_with_logits(
        k_pred, k_t, reduction='none')
    loss_kappa = (kappa_bce * fg).sum() / fg.sum().clamp_min(1.0)
    return loss_c + loss_b + loss_r + loss_orient + 0.5 * loss_kappa


__all__ = ['field_loss', 'structure_targets']
