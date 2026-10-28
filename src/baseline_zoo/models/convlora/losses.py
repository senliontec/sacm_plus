"""Official Conv-LoRA structure loss (weighted IoU + weighted BCE)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureLoss(nn.Module):
    """Structure loss from PraNet, used by the Conv-LoRA official recipe."""

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        weight = 1 + 5 * torch.abs(
            F.avg_pool2d(target, kernel_size=31, stride=1, padding=15)
            - target)
        wbce = F.binary_cross_entropy_with_logits(
            logits, target, reduction='none')
        wbce = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

        probs = torch.sigmoid(logits)
        intersection = ((probs * target) * weight).sum(dim=(2, 3))
        union = ((probs + target) * weight).sum(dim=(2, 3))
        weighted_iou = 1 - (intersection + 1) / (union - intersection + 1)
        return (wbce + weighted_iou).mean()
