"""TopoQA: role-aware quality arbitration for SACM-STC.

This module is default-off.  When enabled, it reads the four post-gating
Stage-2 descriptors and the global SAM image embedding, predicts an IoU and a
soft-clDice quality score for every expert, and provides a topology-aware
stability score plus a Pareto arbitration rule.

The design follows SAM's native multimask mechanism: every candidate carries a
quality prediction, and inference selects a candidate instead of averaging
them.  The adaptation is that SACM's four candidates are *role experts*
(champion/dense, topology, boundary, recall), not ambiguity modes of the same
object, so the quality target is multi-metric and the decision is constrained.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from baseline_zoo.models.sacm_stc.topology_losses import SoftclDiceLoss


class RoleQualityHead(nn.Module):
    """Shared per-slot quality predictor.

    Parameters
    ----------
    embed_dim:
        Width of the Stage-2 mask descriptors (SAM decoder transformer dim).
    hidden:
        Hidden width of the shared MLP.
    role_dim:
        Width of the per-slot role embedding.  Slots are role-stable because
        the decoder applies its Stage-2 type embedding after gating sort.
    num_roles:
        Number of Stage-2 slots (four for the native SACM-STC decoder).
    num_metrics:
        Two outputs: (IoU, soft-clDice).
    """

    def __init__(self, embed_dim: int = 256, hidden: int = 64,
                 role_dim: int = 16, num_roles: int = 4,
                 num_metrics: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_roles = int(num_roles)
        self.num_metrics = int(num_metrics)
        self.role = nn.Embedding(self.num_roles, int(role_dim))
        input_dim = 2 * self.embed_dim + int(role_dim)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(hidden)),
            nn.GELU(),
        )
        self.score = nn.Linear(int(hidden), self.num_metrics)

    def forward(self, tokens: torch.Tensor,
                global_feature: torch.Tensor) -> torch.Tensor:
        """Predict per-slot quality.

        Parameters
        ----------
        tokens:
            ``[B, N, C]`` post-gating Stage-2 descriptors.
        global_feature:
            ``[B, C]`` mean-pooled SAM image embedding.

        Returns
        -------
        ``[B, N, 2]`` sigmoid scores for (IoU, soft-clDice).
        """
        if tokens.ndim != 3:
            raise ValueError(
                f'tokens must be [B, N, C], got {tuple(tokens.shape)}')
        if global_feature.ndim != 2:
            raise ValueError(
                'global_feature must be [B, C], got '
                f'{tuple(global_feature.shape)}')
        batch, slots, channels = tokens.shape
        if channels != self.embed_dim:
            raise ValueError(
                f'expected {self.embed_dim} token channels, got {channels}')
        if slots != self.num_roles:
            raise ValueError(
                f'expected {self.num_roles} role slots, got {slots}')
        if global_feature.shape != (batch, self.embed_dim):
            raise ValueError(
                f'global_feature must be {(batch, self.embed_dim)}, got '
                f'{tuple(global_feature.shape)}')
        global_feature = global_feature[:, None, :].expand(
            batch, slots, self.embed_dim)
        roles = self.role.weight.unsqueeze(0).expand(batch, slots, -1)
        features = torch.cat([tokens, global_feature, roles], dim=-1)
        return torch.sigmoid(self.score(self.trunk(features)))


@torch.no_grad()
def quality_metric_targets(candidate_logits: torch.Tensor,
                           target: torch.Tensor,
                           cl_loss: nn.Module | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute detached per-candidate (IoU, soft-clDice) targets.

    ``candidate_logits`` is ``[B, N, H, W]`` and ``target`` is
    ``[B, 1, H, W]`` or ``[B, H, W]``.  The IoU target follows the evaluator
    (two empty masks count as IoU 1).  The clDice target is the soft-clDice
    score ``1 - SoftclDiceLoss``; it is a cheap, differentiable proxy for the
    hard skeleton metric and is detached before the quality regression.
    """
    if candidate_logits.ndim != 4:
        raise ValueError(
            'candidate_logits must be [B, N, H, W], got '
            f'{tuple(candidate_logits.shape)}')
    if target.ndim == 4:
        if target.shape[1] != 1:
            raise ValueError(
                f'expected a single target channel, got {target.shape[1]}')
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError(
            f'target must be [B, H, W], got {tuple(target.shape)}')
    if target.shape[0] != candidate_logits.shape[0]:
        raise ValueError('target batch does not match candidate batch')
    if target.shape[-2:] != candidate_logits.shape[-2:]:
        target = F.interpolate(
            target[:, None].float(), size=candidate_logits.shape[-2:],
            mode='nearest')[:, 0]
    target = (target > 0.5).float()

    pred_bin = (candidate_logits > 0).float()
    inter = (pred_bin * target[:, None]).sum(dim=(2, 3))
    union = (pred_bin.sum(dim=(2, 3))
             + target[:, None].sum(dim=(2, 3)) - inter)
    iou = torch.where(
        union > 0, inter / union.clamp_min(1e-6),
        torch.ones_like(inter))

    if cl_loss is None:
        cl_loss = SoftclDiceLoss()
    smooth = float(getattr(cl_loss, 'smooth', 1.0))
    target = target[:, None]
    skel_target = cl_loss._soft_skel(target)
    cldice = []
    for index in range(candidate_logits.shape[1]):
        pred = torch.sigmoid(candidate_logits[:, index:index + 1])
        skel_pred = cl_loss._soft_skel(pred)
        tprec = ((skel_pred * target).sum(dim=(1, 2, 3)) + smooth) / (
            skel_pred.sum(dim=(1, 2, 3)) + smooth)
        tsens = ((skel_target * pred).sum(dim=(1, 2, 3)) + smooth) / (
            skel_target.sum(dim=(1, 2, 3)) + smooth)
        score = 2.0 * tprec * tsens / (tprec + tsens + 1e-3)
        both_empty = (
            ((pred > 0.5).sum(dim=(1, 2, 3)) <= 0)
            & (target.sum(dim=(1, 2, 3)) <= 0))
        score = torch.where(
            both_empty, torch.ones_like(score), score)
        cldice.append(score)
    cldice = torch.stack(cldice, dim=1).clamp(0.0, 1.0)
    return iou.clamp(0.0, 1.0), cldice


class TopologyStability(nn.Module):
    """Soft-skeleton clDice between masks at ``p = 0.5 +/- delta``.

    SAM2's stability score is the area IoU between two thresholded versions of
    the same logits.  Area is nearly unchanged when a one-pixel skeleton
    breaks, so it is a weak signal for curvilinear structures.  This score
    replaces the area with skeleton agreement, which is the topology analogue
    of SAM2 stability.  It is label-free and inference-only.
    """

    def __init__(self, num_iter: int = 5, delta: float = 0.1,
                 smooth: float = 1.0):
        super().__init__()
        self.num_iter = max(int(num_iter), 1)
        self.delta = float(delta)
        self.smooth = float(smooth)
        if not 0.0 <= self.delta < 0.5:
            raise ValueError(
                f'delta must be in [0, 0.5), got {self.delta}')
        self._skel = SoftclDiceLoss(
            num_iter=self.num_iter, smooth=self.smooth)

    @torch.no_grad()
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(
                'logits must be [B, N, H, W], got '
                f'{tuple(logits.shape)}')
        batch, slots, height, width = logits.shape
        prob = torch.sigmoid(logits)
        high = (prob > 0.5 + self.delta).float()
        low = (prob > 0.5 - self.delta).float()
        high = high.reshape(batch * slots, 1, height, width)
        low = low.reshape(batch * slots, 1, height, width)
        skel_high = self._skel._soft_skel(high)
        skel_low = self._skel._soft_skel(low)
        tprec = ((skel_high * low).sum(dim=(1, 2, 3)) + self.smooth) / (
            skel_high.sum(dim=(1, 2, 3)) + self.smooth)
        tsens = ((skel_low * high).sum(dim=(1, 2, 3)) + self.smooth) / (
            skel_low.sum(dim=(1, 2, 3)) + self.smooth)
        score = 2.0 * tprec * tsens / (tprec + tsens + 1e-6)
        both_empty = (
            (high.sum(dim=(1, 2, 3)) <= 0)
            & (low.sum(dim=(1, 2, 3)) <= 0))
        score = torch.where(
            both_empty, torch.ones_like(score), score)
        return score.reshape(batch, slots).clamp(0.0, 1.0)


def pareto_arbitrate(quality: torch.Tensor,
                     stability: torch.Tensor,
                     champion: int = 0,
                     margin_cldice: float = 0.01,
                     margin_iou: float = 0.005,
                     stability_threshold: float = 0.9) -> torch.Tensor:
    """Select one expert per image under a Pareto constraint.

    ``quality`` is ``[B, N, 2]`` with ``(IoU, clDice)`` and ``stability`` is
    ``[B, N]``.  Slot ``champion`` is the gating default.  A non-champion is
    eligible only when its predicted clDice improves by ``margin_cldice`` and
    its predicted IoU does not drop by more than ``margin_iou``.  If no
    candidate is eligible but the champion's topology stability is below
    ``stability_threshold``, fall back to the most stable IoU-eligible
    alternative.  Returns ``[B]`` slot indices.
    """
    if quality.ndim != 3 or quality.shape[-1] != 2:
        raise ValueError(
            'quality must be [B, N, 2], got '
            f'{tuple(quality.shape)}')
    if stability.shape != quality.shape[:2]:
        raise ValueError(
            f'stability must be {tuple(quality.shape[:2])}, got '
            f'{tuple(stability.shape)}')
    slots = quality.shape[1]
    if not 0 <= int(champion) < slots:
        raise ValueError(
            f'champion {champion} outside [0, {slots})')
    device = quality.device
    batch = quality.shape[0]
    champion = int(champion)
    if slots == 1:
        return torch.zeros(batch, dtype=torch.long, device=device)

    q_champion = quality[:, champion]
    iou = quality[..., 0]
    cldice = quality[..., 1]
    alternatives = [index for index in range(slots) if index != champion]
    alt_index = torch.tensor(alternatives, dtype=torch.long, device=device)

    alt_cldice = cldice[:, alt_index]
    alt_iou = iou[:, alt_index]
    alt_stability = stability[:, alt_index]

    eligible = (
        (alt_cldice >= q_champion[:, 1:2] + float(margin_cldice))
        & (alt_iou >= q_champion[:, 0:1] - float(margin_iou)))
    masked_cldice = alt_cldice.masked_fill(
        ~eligible, torch.finfo(alt_cldice.dtype).min)
    best_rel = masked_cldice.argmax(dim=1)
    has_eligible = eligible.any(dim=1)
    best = torch.where(
        has_eligible,
        alt_index[best_rel],
        torch.full_like(best_rel, champion))

    rescue = (
        (~has_eligible)
        & (stability[:, champion] < float(stability_threshold)))
    if bool(rescue.any()):
        rescue_iou_ok = (
            alt_iou >= q_champion[:, 0:1] - float(margin_iou))
        masked_stability = alt_stability.masked_fill(
            ~rescue_iou_ok, torch.finfo(alt_stability.dtype).min)
        rescue_rel = masked_stability.argmax(dim=1)
        rescue_has = rescue_iou_ok.any(dim=1)
        rescue_best = torch.where(
            rescue_has,
            alt_index[rescue_rel],
            torch.full_like(rescue_rel, champion))
        best = torch.where(rescue, rescue_best, best)
    return best


__all__ = [
    'RoleQualityHead',
    'TopologyStability',
    'pareto_arbitrate',
    'quality_metric_targets',
]
