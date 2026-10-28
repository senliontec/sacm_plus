"""TPSegformer auxiliary losses: Dice and Sobel edge-aware L1.

The release's ``DiceLoss`` (squared denominator, eps 1e-3) and
``EdgeAwareLoss`` (Sobel gradient-magnitude L1, alpha 1.0) are reconstructed
here for the binary head without the MMSeg registry.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                   eps: float = 1e-3) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(2, 3))
    denominator = (probability.pow(2).sum(dim=(2, 3))
                   + target.pow(2).sum(dim=(2, 3)) + eps)
    return (1 - 2 * intersection / denominator).mean()


def _sobel(device, dtype):
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=dtype, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=dtype, device=device).view(1, 1, 3, 3)
    return sobel_x, sobel_y


def edge_aware_loss(logits: torch.Tensor, target: torch.Tensor,
                    alpha: float = 1.0) -> torch.Tensor:
    if logits.shape[1] != 1 or target.shape != logits.shape:
        raise ValueError(
            'edge_aware_loss expects matching [B,1,H,W] tensors, got '
            f'{tuple(logits.shape)} and {tuple(target.shape)}')
    probability = logits.sigmoid()
    sobel_x, sobel_y = _sobel(logits.device, probability.dtype)
    pred_x = F.conv2d(probability, sobel_x, padding=1)
    pred_y = F.conv2d(probability, sobel_y, padding=1)
    target_x = F.conv2d(target.float(), sobel_x, padding=1)
    target_y = F.conv2d(target.float(), sobel_y, padding=1)
    pred_edge = torch.sqrt(pred_x ** 2 + pred_y ** 2 + 1e-6)
    target_edge = torch.sqrt(target_x ** 2 + target_y ** 2 + 1e-6)
    return alpha * F.l1_loss(pred_edge, target_edge)


def joint_aux_loss(logits: torch.Tensor, target: torch.Tensor,
                   dice_weight: float = 0.5,
                   edge_weight: float = 1.0) -> torch.Tensor:
    return (dice_weight * soft_dice_loss(logits, target)
            + edge_weight * edge_aware_loss(logits, target))
