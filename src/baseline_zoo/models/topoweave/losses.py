"""Bounded topology-aware loss for TopoWeave.

MaskVSC's Connect_Loss is replaced by the verified differentiable soft-clDice
implementation in ``core.losses.core`` plus Dice and BCE terms.  All terms are
bounded and scale-comparable, so sparse curvilinear domains do not receive
the 10-100x weight explosion seen with MaskVSC's paper-mode L_C.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from core.losses.core import DiceLoss, SoftclDiceLoss


class TopoWeaveLoss(nn.Module):
    """BCE + Dice + soft-clDice on the final output, with coarse supervision.

    Args:
        dice_weight: weight of the soft Dice loss on the final logits.
        topo_weight: weight of soft-clDice; 0 disables soft skeletonization.
        coarse_weight: weight of the deep-supervision terms on coarse logits.
        bce_weight: weight of BCEWithLogits on both outputs.
        topo_iters: soft-skeleton iterations for soft-clDice.
    """

    def __init__(self, dice_weight: float = 1.0, topo_weight: float = 0.5,
                 coarse_weight: float = 0.4, bce_weight: float = 1.0,
                 topo_iters: int = 10):
        super().__init__()
        if dice_weight < 0 or topo_weight < 0 or coarse_weight < 0:
            raise ValueError('loss weights must be non-negative')
        self.dice_weight = float(dice_weight)
        self.topo_weight = float(topo_weight)
        self.coarse_weight = float(coarse_weight)
        self.bce_weight = float(bce_weight)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.topo = (SoftclDiceLoss(num_iter=topo_iters)
                     if self.topo_weight > 0 else None)

    def _single(self, logits: torch.Tensor, target: torch.Tensor,
                topology: bool) -> torch.Tensor:
        loss = self.bce_weight * self.bce(logits, target)
        loss = loss + self.dice_weight * self.dice(logits, target)
        if topology and self.topo is not None:
            value = self.topo(logits, target)
            loss = loss + self.topo_weight * value.clamp(0.0, 1.0)
        return loss

    def forward(self, outputs: dict[str, torch.Tensor],
                target: torch.Tensor) -> torch.Tensor:
        logits = outputs['logits']
        loss = self._single(logits, target, topology=True)
        coarse = outputs.get('coarse_logits')
        if self.coarse_weight > 0 and coarse is not None:
            loss = loss + self.coarse_weight * self._single(
                coarse, target, topology=False)
        return loss


def skeleton_weighted_bce(logits: torch.Tensor, target: torch.Tensor,
                          skeleton: torch.Tensor,
                          weight: float = 4.0) -> torch.Tensor:
    """Optional thin-structure emphasis; kept for ablations.

    The pixel weight is ``1 + weight * skeleton``.  Skeleton is expected to be
    a 0/1 tensor precomputed from training labels.
    """
    pixel_weight = 1.0 + float(weight) * skeleton
    return F.binary_cross_entropy_with_logits(
        logits, target, weight=pixel_weight)
