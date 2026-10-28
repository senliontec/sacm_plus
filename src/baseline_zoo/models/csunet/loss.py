"""Dice loss used by the CS-UNet notebook (segmentation_models style)."""

from __future__ import annotations

import torch


def dice_loss(prediction: torch.Tensor, target: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    prediction = prediction.float().reshape(prediction.shape[0], -1)
    target = target.float().reshape(target.shape[0], -1)
    intersection = (prediction * target).sum(dim=1)
    denominator = prediction.sum(dim=1) + target.sum(dim=1)
    return (1.0 - (2.0 * intersection + smooth)
            / (denominator + smooth)).mean()
