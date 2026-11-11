"""TPD modules for SACM-STC.

These are optional, default-off pieces for the TPD experiment:

* :class:`ImageDensePrompt` conditions the prompt-free dense prompt on the
  image encoder output (MM-SAM style).  Its residual gate starts at 0.1 so
  the branch is active from the first step when the flag is enabled; the
  flag itself is default-off, so existing runs are unchanged.
* :class:`TopologyQualityHead` predicts a scalar quality for each native
  Stage-2 candidate.  Candidates remain proposals; only the deployed head
  receives the dense mask loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ImageDensePrompt(nn.Module):
    """Image-conditioned residual for SAM's prompt-free dense embedding."""

    def __init__(self, embed_dim: int = 256, prompt_dim: int | None = None,
                 init_scale: float = 0.1):
        super().__init__()
        self.embed_dim = int(embed_dim)
        prompt_dim = int(embed_dim if prompt_dim is None else prompt_dim)
        self.proj = nn.Sequential(
            nn.Conv2d(int(embed_dim), prompt_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(prompt_dim, prompt_dim, 1),
        )
        # LayerScale-style gate.  A zero initialisation leaves the residual
        # at exactly zero and, with the shared 3e-4 LR, the learned scale
        # only reached 2.3e-4 after 7 epochs -- the image prompt was an
        # effective no-op.  Start the gate at a small nonzero value so the
        # enabled branch actually contributes from the first step.
        self.scale = nn.Parameter(torch.full((1,), float(init_scale)))

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        if image_embeddings.ndim != 4:
            raise ValueError(
                'image_embeddings must be [B, C, H, W], got '
                f'{tuple(image_embeddings.shape)}')
        if image_embeddings.shape[1] != self.embed_dim:
            raise ValueError(
                f'expected {self.embed_dim} image-embedding channels, got '
                f'{image_embeddings.shape[1]}')
        return self.scale * self.proj(image_embeddings)


class TopologyQualityHead(nn.Module):
    """Shared per-candidate quality predictor.

    Input is ``[B, N, H, W]`` candidate logits.  Output is ``[B, N]`` in
    ``[0, 1]``.  The target can be any bounded topology-aware score; the
    adapter uses a Dice + soft-clDice target.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, int(hidden), 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden), int(hidden) * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.score = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(hidden) * 2, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, candidate_logits: torch.Tensor) -> torch.Tensor:
        if candidate_logits.ndim != 4:
            raise ValueError(
                'candidate_logits must be [B, N, H, W], got '
                f'{tuple(candidate_logits.shape)}')
        batch, heads, height, width = candidate_logits.shape
        flat = candidate_logits.reshape(batch * heads, 1, height, width)
        features = self.features(flat)
        scores = self.score(features).reshape(batch, heads)
        return torch.sigmoid(scores)

    @staticmethod
    def quality_target(
        candidate_logits: torch.Tensor,
        target: torch.Tensor,
        cl_loss: nn.Module | None = None,
        iou_weight: float = 0.0,
        dice_weight: float = 0.5,
        cldice_weight: float = 0.5,
    ) -> torch.Tensor:
        """Compute bounded per-candidate quality targets.

        ``candidate_logits`` is ``[B, N, H, W]``; ``target`` must already be
        resized to the same spatial size.  This helper runs under ``no_grad``
        at the call site; it is a scalar quality target, not a dense loss on
        the auxiliary candidates.  The IoU term follows SAMEO's quality
        prediction; the topology term is our addition for curvilinear masks.
        """
        if candidate_logits.ndim != 4:
            raise ValueError('candidate_logits must be [B, N, H, W]')
        if target.shape[0] != candidate_logits.shape[0]:
            raise ValueError('target batch does not match candidate batch')
        if target.shape[-2:] != candidate_logits.shape[-2:]:
            target = F.interpolate(
                target, size=candidate_logits.shape[-2:], mode='nearest')
        probs = torch.sigmoid(candidate_logits)
        # Keep the candidate dimension: [B, N, H, W] -> [B, N].
        intersection = (probs * target).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        iou = (intersection + 1e-6) / (
            union - intersection + 1e-6)

        scores = []
        for index in range(candidate_logits.shape[1]):
            score = (float(iou_weight) * iou[:, index]
                     + float(dice_weight) * dice[:, index])
            if cl_loss is not None and float(cldice_weight) > 0:
                cl = cl_loss(
                    candidate_logits[:, index:index + 1], target)
                score = score + float(cldice_weight) * (1.0 - cl)
            scores.append(score)
        return torch.stack(scores, dim=1).clamp(0.0, 1.0)


__all__ = ['ImageDensePrompt', 'TopologyQualityHead']
