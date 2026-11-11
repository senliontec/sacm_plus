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


class MorphPrompt(nn.Module):
    """Polarity-invariant ON/OFF morphological dense prompt.

    White top-hat (bright structures) and black top-hat (dark structures)
    swap under intensity inversion, so their sum is invariant to the
    foreground/background contrast sign.  Multi-scale sums are projected to
    the decoder's dense-prompt width and added residually.

    Offline validation (TopoMortar/DeepGlobe/LEAF/ROAD/WIRE/XCAD/DRIVE)
    showed the multi-scale top-hat sum is the strongest and exactly
    polarity-invariant structural feature (AUC 0.66-0.92, inversion
    correlation +1.000).
    """

    def __init__(self, embed_dim: int = 256,
                 scales: tuple[int, ...] = (3, 7, 15),
                 work_size: int = 512, gate_threshold: float = 0.1,
                 gate_bias: float = -4.0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.scales = tuple(int(s) for s in scales)
        self.work_size = int(work_size)
        self.proj = nn.Sequential(
            nn.Conv2d(len(self.scales), embed_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )
        # Per-image safety gate.  Its inputs are image intensity statistics
        # and the pooled ON/OFF response; the bias starts at -4 so the gate
        # is closed (and the prompt is exactly zero) until training opens it.
        self.gate = nn.Sequential(
            nn.Linear(6 + embed_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.constant_(self.gate[-1].bias, float(gate_bias))
        self.scale = nn.Parameter(torch.ones(1))
        self.gate_threshold = float(gate_threshold)
        self.last_gate = None

    @staticmethod
    def _erode(x: torch.Tensor, kernel: int) -> torch.Tensor:
        pad = kernel // 2
        return -F.max_pool2d(-x, kernel_size=kernel, stride=1, padding=pad)

    @staticmethod
    def _dilate(x: torch.Tensor, kernel: int) -> torch.Tensor:
        pad = kernel // 2
        return F.max_pool2d(x, kernel_size=kernel, stride=1, padding=pad)

    def morph_features(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                'morph prompt expects [B, 3, H, W] RGB images, got '
                f'{tuple(images.shape)}')
        gray = (0.299 * images[:, 0:1] + 0.587 * images[:, 1:2]
                + 0.114 * images[:, 2:3])
        if self.work_size and gray.shape[-1] != self.work_size:
            gray = F.interpolate(
                gray, size=(self.work_size, self.work_size),
                mode='bilinear', align_corners=False)
        features = []
        for kernel in self.scales:
            opening = self._dilate(self._erode(gray, kernel), kernel)
            closing = self._erode(self._dilate(gray, kernel), kernel)
            # White top-hat + black top-hat; the two terms swap under
            # inversion, so the sum is polarity-invariant.
            features.append((gray - opening) + (closing - gray))
        return self.proj(torch.cat(features, dim=1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.morph_features(images)
        stats = torch.cat(
            [images.mean(dim=(2, 3)), images.std(dim=(2, 3))], dim=1)
        pooled = features.mean(dim=(2, 3))
        gate = torch.sigmoid(
            self.gate(torch.cat([stats, pooled], dim=1)))
        # Training uses the soft gate so gradients reach the gate and the
        # projection.  Inference uses the hard safety gate: below the
        # threshold the prompt contributes exactly zero, so the downstream
        # model is bit-identical to the no-POMP path.
        if self.training:
            gated = gate
        else:
            gated = torch.where(
                gate >= self.gate_threshold, gate, torch.zeros_like(gate))
        self.last_gate = gate.detach()
        return self.scale * gated[:, :, None, None] * features


class HeadFusion(nn.Module):
    """Safe zero-initialised fusion of the four Stage-2 heads.

    Two modes are supported:

    * ``linear``: ``out = z0 + conv([z1-z0, z2-z0, z3-z0])`` with the 1x1
      conv zero-initialised.  This is the original global linear fusion and
      keeps the original state-dict keys (``fuse.weight``/``fuse.bias``).
    * ``image_gated``: an image-level gate predicts three weights from the
      per-head logit statistics (mean/std), so different images can trust
      different auxiliary heads.  The gate's last layer is zero-initialised,
      hence the fused output is exactly z0 at initialisation.

    The per-pixel residual inputs are shared by both modes.
    """

    MODES = ('linear', 'image_gated')

    def __init__(self, mode: str = 'linear'):
        super().__init__()
        self.mode = str(mode)
        if self.mode not in self.MODES:
            raise ValueError(
                f'unknown head-fusion mode {mode!r}; expected one of '
                f'{self.MODES}')
        if self.mode == 'linear':
            self.fuse = nn.Conv2d(3, 1, 1)
            nn.init.zeros_(self.fuse.weight)
            nn.init.zeros_(self.fuse.bias)
        else:
            # 4 head means + 4 head stds -> 3 residual weights in [-1, 1].
            self.gate = nn.Sequential(
                nn.LayerNorm(8),
                nn.Linear(8, 16),
                nn.GELU(),
                nn.Linear(16, 3),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def forward(self, heads: torch.Tensor) -> torch.Tensor:
        if heads.ndim != 4 or heads.shape[1] != 4:
            raise ValueError(
                'head fusion expects four Stage-2 heads [B, 4, H, W], got '
                f'{tuple(heads.shape)}')
        base = heads[:, 0:1]
        residual = torch.cat(
            [heads[:, 1:2] - base, heads[:, 2:3] - base,
             heads[:, 3:4] - base], dim=1)
        if self.mode == 'linear':
            return base + self.fuse(residual)
        stats = torch.cat(
            [heads.mean(dim=(2, 3)), heads.std(dim=(2, 3))], dim=1)
        weights = torch.tanh(self.gate(stats))
        return base + (weights[:, :, None, None] * residual).sum(
            dim=1, keepdim=True)


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


__all__ = [
    'HeadFusion', 'ImageDensePrompt', 'MorphPrompt', 'TopologyQualityHead',
]
