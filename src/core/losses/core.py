"""Core loss functions.

- DiceLoss / DiceBCELoss: kept identical to the original SACM training
  code so the reproduced baseline is directly comparable.
- SoftclDiceLoss: differentiable centerline-Dice loss, matching the
  OFFICIAL clDice repository implementation (Shit et al., CVPR 2021),
  verified line by line.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss that always applies its own sigmoid (logits in)."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.7, dice_weight=0.3):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)  # DiceLoss applies its own sigmoid
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def soft_dice(y_true, y_pred):
    """Dice loss on probabilities, official clDice repository form
    (smooth=1.0, added to numerator AND denominator)."""
    smooth = 1
    intersection = torch.sum((y_true * y_pred))
    coeff = (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return (1. - coeff)


class SoftSkeletonize(nn.Module):
    """Differentiable soft-skeletonization, matching the OFFICIAL clDice
    repository implementation (Shit et al., CVPR 2021,
    cldice_loss/pytorch/soft_skeleton.py), verified line by line.

    - soft_erode: min of a (3,1) and a (1,3) min-pooling (anisotropic
      erosion proxy)
    - soft_dilate: (3,3) max-pooling
    - soft_open: dilate(erode(x))
    - soft_skel: residual accumulation — skel = relu(x - open(x)), then
      num_iter times: erode x, delta = relu(x - open(x)),
      skel += relu(delta - skel * delta)

    Operates on probabilities in [0, 1]; input shape [B, C, H, W].
    """

    def __init__(self, num_iter=10):
        super().__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    def soft_dilate(self, img):
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, img):
        return self.soft_skel(img)


class SoftclDiceLoss(nn.Module):
    """Differentiable clDice loss, matching the official repository.

    API difference from the official module: this takes RAW LOGITS and
    applies sigmoid internally (the official soft_cldice expects
    probabilities). The tprec/tsens computation and the smoothing
    (smooth=1.0 added to numerator AND denominator) follow the official
    clDice.py exactly.

    Args:
        num_iter: soft-skeleton iterations (official default: 10).
        smooth: Tversky-style smoothing (official default: 1.0).
    """

    def __init__(self, num_iter=10, smooth=1.0):
        super().__init__()
        self.soft_skeletonize = SoftSkeletonize(num_iter=num_iter)
        self.smooth = smooth

    def forward(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        skel_pred = self.soft_skeletonize(pred)
        skel_target = self.soft_skeletonize(target)
        tprec = (torch.sum(skel_pred * target) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_target * pred) + self.smooth) / (torch.sum(skel_target) + self.smooth)
        cldice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cldice
