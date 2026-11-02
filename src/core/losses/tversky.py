"""Tversky loss for the recall-oriented typed head.

Tversky generalizes Dice with separate FP/FN weights:

    T = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

For a high-recall structural candidate set ``beta > alpha`` (the preset uses
0.3/0.7), so false negatives are penalized more than false positives.  This
head is deliberately paired with a base BCE/Dice term in the trainer; the
Tversky term only shapes its recall/precision trade-off.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(
                f"TverskyLoss shape mismatch: {logits.shape} vs {target.shape}")
        prob = torch.sigmoid(logits)
        target = target.clamp(0.0, 1.0)
        dims = (1, 2, 3)
        tp = (prob * target).sum(dim=dims)
        fp = (prob * (1.0 - target)).sum(dim=dims)
        fn = ((1.0 - prob) * target).sum(dim=dims)
        score = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - score.mean()
