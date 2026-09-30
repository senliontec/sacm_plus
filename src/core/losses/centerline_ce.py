"""centerline-CE loss family (MICCAI 2024) — strict ports of the official
repository centerline_CE/nnUNet/losses/cldice_loss.py.

Classes: dice_clCE_loss (centerline_ce), dice_cldice_loss (dice_cldice),
CE_cldice_loss (ce_cldice), CE_clCE_loss (ce_clce).

Bridges (documented deviations):
- The official classes consume [B, 2, H, W] logits + long labels; ours
  consume [B, 1, H, W] logits + a float mask. cat([zeros, logits]) makes
  the softmax foreground channel exactly sigmoid(logits).
- The official CE variants use nnU-Net's RobustCrossEntropyLoss, which
  with default parameters is plain cross-entropy; F.cross_entropy is the
  equivalent (ignore_label support is not ported; the official
  conditional CE logic is preserved verbatim).
- CAVEAT: the pulled repo copy lacks soft_skeleton.py; its
  dice_cldice_loss baseline is formula-identical to the official clDice
  loss, so the VERIFIED official SoftSkeletonize is used. Re-verify if
  the original file becomes available.
"""

import torch
import torch.nn as nn

from .core import SoftSkeletonize
from .registry import register


def _soft_dice_clce(y_pred, y_true, smooth=1.0):
    """centerline_CE's soft_dice (cldice_loss.py): operates on the 2-channel
    tensors and slices the foreground channel [:, 1:]."""
    intersection = torch.sum((y_true * y_pred)[:, 1:, ...])
    coeff = (2.0 * intersection + smooth) / (torch.sum(y_true[:, 1:, ...]) + torch.sum(y_pred[:, 1:, ...]) + smooth)
    return 1.0 - coeff


def _soft_skel_clce(x, iterations=3):
    """Soft skeleton for the centerline-CE losses (see module docstring
    for the missing-file caveat)."""
    return SoftSkeletonize(num_iter=iterations)(x)


@register('centerline_ce')
class DiceCenterlineCELoss(nn.Module):
    """dice_clCE_loss — the paper's headline loss (centerline-CE)."""

    def __init__(self, iter_=3, smooth=1.0, weight_dice=1, weight_clCE=1):
        super().__init__()
        self.iter = iter_
        self.smooth = smooth
        self.weight_clCE = weight_clCE
        self.weight_dice = weight_dice

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_pred = torch.cat([torch.zeros_like(logits), logits], dim=1)  # [B, 2, H, W]
        y_true = (target > 0.5).long()

        y_true_oh = torch.zeros(y_pred.shape, device=y_pred.device)
        y_true_oh.scatter_(1, y_true, 1)  # nnUNet-training-loss-dice.py

        cross_ent = torch.nn.functional.cross_entropy(y_pred, y_true_oh, reduction="none")
        y_pred = y_pred.softmax(dim=1)

        dice = _soft_dice_clce(y_pred, y_true_oh, self.smooth)

        skel_pred = _soft_skel_clce(y_pred, self.iter)
        skel_true = _soft_skel_clce(y_true_oh, self.iter)
        tprec = torch.mul(cross_ent, skel_true[:, 1]).mean()
        tsens = torch.mul(cross_ent, skel_pred[:, 1]).mean()
        cl_ce = (tprec + tsens)
        result = self.weight_dice * dice + self.weight_clCE * cl_ce
        return result


@register('dice_cldice')
class DiceCLDiceLoss(nn.Module):
    """dice_cldice_loss — their clDice baseline variant (functionally
    overlaps with SoftclDiceLoss; registered for the official-variant
    comparison)."""

    def __init__(self, iter_=3, smooth=1.0, weight_dice=1, weight_cldice=1):
        super().__init__()
        self.iter_ = iter_
        self.smooth = smooth
        self.weight_dice = weight_dice
        self.weight_cldice = weight_cldice

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_pred = torch.cat([torch.zeros_like(logits), logits], dim=1)
        y_true = (target > 0.5).long()

        y_pred = y_pred.softmax(dim=1)

        y_true_oh = torch.zeros(y_pred.shape, device=y_pred.device)
        y_true_oh.scatter_(1, y_true, 1)  # nnUNet-training-loss-dice.py

        dice = _soft_dice_clce(y_true_oh, y_pred, self.smooth)

        skel_pred = _soft_skel_clce(y_pred, self.iter_)
        skel_true = _soft_skel_clce(y_true_oh, self.iter_)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true_oh)[:, 1:, ...]) + self.smooth) / (torch.sum(skel_pred[:, 1:, ...]) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:, 1:, ...]) + self.smooth) / (torch.sum(skel_true[:, 1:, ...]) + self.smooth)
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        result = self.weight_dice * dice + self.weight_cldice * cl_dice
        return result


@register('ce_cldice')
class CECLDiceLoss(nn.Module):
    """CE_cldice_loss — CE substitution documented in the module
    docstring; the official conditional CE computation is preserved."""

    def __init__(self, iter_=3, smooth=1.0, weight_ce=1, weight_cldice=1, ignore_label=None):
        super().__init__()
        self.iter_ = iter_
        self.smooth = smooth
        self.weight_cldice = weight_cldice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        net_output = torch.cat([torch.zeros_like(logits), logits], dim=1)
        y_true = (target > 0.5).long()

        ce_loss = torch.nn.functional.cross_entropy(net_output, y_true[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None) else 0

        target_oh = torch.zeros(net_output.shape, device=net_output.device)
        target_oh.scatter_(1, y_true, 1)  # nnUNet-training-loss-dice.py

        net_output = net_output.softmax(dim=1)
        skel_true = _soft_skel_clce(target_oh, self.iter_)
        skel_pred = _soft_skel_clce(net_output, self.iter_)
        tprec = (torch.sum(torch.multiply(skel_pred, target_oh)[:, 1:, ...]) + self.smooth) / (torch.sum(skel_pred[:, 1:, ...]) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, net_output)[:, 1:, ...]) + self.smooth) / (torch.sum(skel_true[:, 1:, ...]) + self.smooth)
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        result = self.weight_ce * ce_loss + self.weight_cldice * cl_dice
        return result


@register('ce_clce')
class CECLCELoss(nn.Module):
    """CE_clCE_loss — CE substitution documented in the module docstring."""

    def __init__(self, iter_=3, weight_ce=1, weight_clCE=1):
        super().__init__()
        self.iter = iter_
        self.weight_clCE = weight_clCE
        self.weight_ce = weight_ce

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_pred = torch.cat([torch.zeros_like(logits), logits], dim=1)
        y_true = (target > 0.5).long()

        y_true_oh = torch.zeros(y_pred.shape, device=y_pred.device)
        y_true_oh.scatter_(1, y_true, 1)  # nnUNet-training-loss-dice.py

        ce_loss = torch.nn.functional.cross_entropy(y_pred, y_true[:, 0])
        cross_ent = torch.nn.functional.cross_entropy(y_pred, y_true_oh, reduction="none")
        y_pred = y_pred.softmax(dim=1)
        skel_pred = _soft_skel_clce(y_pred, self.iter)
        skel_true = _soft_skel_clce(y_true_oh, self.iter)
        tprec = torch.mul(cross_ent, skel_true[:, 1]).mean()
        tsens = torch.mul(cross_ent, skel_pred[:, 1]).mean()
        cl_ce = (tprec + tsens)
        result = self.weight_ce * ce_loss + self.weight_clCE * cl_ce
        return result
