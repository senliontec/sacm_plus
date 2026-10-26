"""Ported binary loss (synthetic calibration case)."""
import torch
import torch.nn.functional as F


def dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    inter = 2.0 * (prob * target).sum()
    sets_sum = prob.sum() + target.sum()
    return 1.0 - (inter + 1e-6) / (sets_sum + 1e-6)


def total_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return 1.0 * bce + 0.5 * dice_loss(logits, target)
