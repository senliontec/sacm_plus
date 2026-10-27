"""Official MedCAGD training objective.

Upstream uses ``BCEWithLogitsLoss`` for the main mask (binary config),
normalized deep-supervision weights ``[1.0, 0.4, 0.3, 0.2]`` and
``EdgeBCELoss`` over a distance-transform-like edge band with equal
weights.  This module keeps that objective without importing the upstream
global ``cfgs`` module.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

DEEP_SUPERVISION_WEIGHTS = (1.0, 0.4, 0.3, 0.2)
EDGE_SUPERVISION_WEIGHTS = (1.0, 1.0, 1.0)


def generate_edge_targets(masks: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """Reproduce upstream ``generate_edge_targets`` for binary masks.

    The edge band is ``dilate(mask) - erode(mask)`` with a square kernel of
    side ``2 * radius + 1``.  Masks may be ``[B,H,W]`` or ``[B,1,H,W]``.
    """
    if masks.dim() == 4:
        masks = masks.squeeze(1)
    masks = masks.float().unsqueeze(1)
    kernel = torch.ones(
        (1, 1, 2 * radius + 1, 2 * radius + 1),
        device=masks.device,
        dtype=masks.dtype,
    )
    dilated = F.conv2d(masks, kernel, padding=radius) > 0
    eroded = F.conv2d(masks, kernel, padding=radius) == kernel.numel()
    return (dilated & ~eroded).float()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                   smooth: float = 1.0) -> torch.Tensor:
    """Per-image soft Dice loss on sigmoid probabilities."""
    probs = torch.sigmoid(logits).flatten(1)
    target = target.float().flatten(1)
    intersection = (probs * target).sum(dim=1)
    denominator = probs.sum(dim=1) + target.sum(dim=1)
    return (1.0 - (2.0 * intersection + smooth)
            / (denominator + smooth)).mean()


class MedCAGDLoss(nn.Module):
    """BCE deep supervision + edge BCE + a foreground Dice term.

    The paper uses BCE only.  On the 3-shot protocol several domains have
    0.25-3% foreground, where BCE-only training collapses to an all-background
    solution; the soft Dice term on the main output is a task adaptation that
    prevents this.  Set ``dice_weight=0`` for the paper's BCE-only objective.
    """

    def __init__(self, edge_weight: float = 1.0, edge_radius: int = 3,
                 dice_weight: float = 1.0):
        super().__init__()
        self.edge_weight = float(edge_weight)
        self.edge_radius = int(edge_radius)
        self.dice_weight = float(dice_weight)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        main: torch.Tensor,
        auxiliaries: list[torch.Tensor],
        edges: list[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        weights = DEEP_SUPERVISION_WEIGHTS[: 1 + len(auxiliaries)]
        denom = sum(weights)
        seg_loss = self.bce(main, target) * weights[0]
        for aux, weight in zip(auxiliaries, weights[1:]):
            seg_loss = seg_loss + self.bce(aux, target) * weight
        seg_loss = seg_loss / denom

        edge_target = generate_edge_targets(target, radius=self.edge_radius)
        edge_loss = sum(
            self.bce(edge, edge_target) for edge in edges
        ) / max(len(edges), 1)
        dice_loss = soft_dice_loss(main, target)
        return (seg_loss + self.edge_weight * edge_loss
                + self.dice_weight * dice_loss)
