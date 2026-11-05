"""PEPA-style post-encoder adapters for curvilinear structure segmentation.

PEPA (ECCV 2026, "From Reconstruction to Decision") identifies two
post-encoder bottlenecks: passive upsampling blurs thin structures, and a
single global threshold cannot separate faint branches from noise.  This
module implements a lightweight, deterministic version of the two ideas:

* :class:`SnakeUpsample2d` -- target-conditioned snake upsampling (TCSU).
  It samples short X-major/Y-major chains from the low-resolution feature
  map, aggregates them with a learned target-conditioned length/weight,
  and adds a zero-initialized residual to the existing decoder upsampling.
* :class:`TargetAdaptiveThreshold` -- per-candidate adaptive threshold
  (TADT).  It predicts a threshold in logit space and returns logits whose
  zero level is that threshold, so the standard evaluator threshold (0.0)
  becomes target-adaptive without any test-time label access.

The implementation intentionally avoids ``torch.nn.functional.grid_sample``
and uses explicit bilinear gathers instead: ``grid_sample`` backward is not
deterministic on CUDA, while this project runs with
``deterministic_strict=True`` by default.

This is a research adapter, not a full reproduction of the paper.  The
paper's exact sub-pixel initialization and hyper-parameters are in its
supplementary material; the public paper dump used here does not include
that supplement.  The wrapper keeps the existing Skein path unchanged when
the flags are off, and the new preset is opt-in.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def _odd(value: int, minimum: int = 1) -> int:
    value = max(int(value), int(minimum))
    return value if value % 2 == 1 else value + 1


class SnakeUpsample2d(nn.Module):
    """Target-conditioned snake upsampling residual.

    Args:
        in_channels: channels of the decoder feature map.
        out_channels: channels of the returned residual.
        scale: spatial upsampling factor (4 maps 64x64 to 256x256).
        mid_channels: width of the snake-sampling branch.
        max_length: odd maximum chain length.
        target_dim: dimension of the target descriptor (mask token).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 4,
        mid_channels: int = 32,
        max_length: int = 5,
        target_dim: int = 256,
        target_hidden: int = 64,
        max_offset: float = 2.0,
    ) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError(f'upsampling scale must be >= 1, got {scale}')
        self.scale = int(scale)
        self.max_length = _odd(max_length)
        self.center = self.max_length // 2
        self.mid_channels = int(mid_channels)
        self.target_dim = int(target_dim)
        self.max_offset = float(max_offset)

        self.proj = nn.Conv2d(in_channels, self.mid_channels, 1)
        self.offset = nn.Conv2d(
            self.mid_channels, 2 * self.max_length, 3, padding=1)
        # Start with straight (zero-offset) chains; bends are learned after
        # the fusion residual has received its first gradient.
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.target_off = nn.Sequential(
            nn.LayerNorm(target_dim),
            nn.Linear(target_dim, target_hidden),
            nn.GELU(),
            nn.Linear(target_hidden, 2 * self.max_length),
        )
        nn.init.zeros_(self.target_off[-1].weight)
        nn.init.zeros_(self.target_off[-1].bias)
        self.target_gate = nn.Sequential(
            nn.LayerNorm(target_dim),
            nn.Linear(target_dim, target_hidden),
            nn.GELU(),
            nn.Linear(target_hidden, 1),
        )
        self.length = nn.Sequential(
            nn.LayerNorm(target_dim),
            nn.Linear(target_dim, target_hidden),
            nn.GELU(),
            nn.Linear(target_hidden, 1),
        )
        self.fuse = nn.Conv2d(3 * self.mid_channels, out_channels, 1)
        # Zero-initialized fusion: at step 0 the residual is exactly zero.
        # gamma stays at one so the fusion layer receives gradient immediately
        # (a second zero gate would block its first update).
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        self.gamma = nn.Parameter(torch.ones(1))

    def _sample_bilinear(
        self,
        feature: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic bilinear sampling.

        ``feature`` is [B, C, H, W]; ``x`` and ``y`` are [B, Ho, Wo] pixel
        coordinates in the low-resolution feature grid.  Positions outside
        the feature map are zeroed rather than clamped.
        """
        b, c, h, w = feature.shape
        flat = feature.reshape(b, c, h * w)
        x0 = torch.floor(x)
        y0 = torch.floor(y)
        x1 = x0 + 1.0
        y1 = y0 + 1.0
        wx = x - x0
        wy = y - y0
        inside = (
            (x >= 0.0) & (x <= float(w - 1)) &
            (y >= 0.0) & (y <= float(h - 1))
        )
        x0i = x0.clamp(0, w - 1).long()
        x1i = x1.clamp(0, w - 1).long()
        y0i = y0.clamp(0, h - 1).long()
        y1i = y1.clamp(0, h - 1).long()

        def gather(xi: torch.Tensor, yi: torch.Tensor) -> torch.Tensor:
            index = (yi * w + xi).reshape(b, 1, -1).expand(b, c, -1)
            return torch.gather(flat, 2, index).reshape(
                b, c, x.shape[1], x.shape[2])

        f00 = gather(x0i, y0i)
        f01 = gather(x1i, y0i)
        f10 = gather(x0i, y1i)
        f11 = gather(x1i, y1i)
        out = (
            f00 * (1.0 - wx) * (1.0 - wy)
            + f01 * wx * (1.0 - wy)
            + f10 * (1.0 - wx) * wy
            + f11 * wx * wy
        )
        return out * inside.unsqueeze(1).to(out.dtype)

    def _cumulative(self, increments: torch.Tensor) -> torch.Tensor:
        """Accumulate increments from the center outward.

        ``increments`` is [B, K, Ho, Wo].  Index ``center`` is exactly zero;
        positions to the left/right accumulate away from the center, matching
        the snake-chain construction in DSConv/TCSU.
        """
        c = self.center
        left = torch.flip(
            torch.cumsum(torch.flip(increments[:, :c + 1], dims=[1]), dim=1),
            dims=[1],
        )
        left = left - left[:, c:c + 1]
        right = torch.cumsum(increments[:, c:], dim=1)
        right = right - right[:, 0:1]
        return torch.cat([left[:, :c], right], dim=1)

    def forward(
        self,
        feature: torch.Tensor,
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return an upsampled residual with the same spatial size as
        ``output_upscaling(feature)``.

        Args:
            feature: [B, C, H, W] decoder feature.
            target: [B, D] target descriptor (mean mask token).  ``None`` is
                accepted for API convenience and replaced by zeros.
        """
        if target is None:
            target = feature.new_zeros(feature.shape[0], self.target_dim)
        b, _, h, w = feature.shape
        projected = self.proj(feature)
        _, mid, h, w = projected.shape
        out_h, out_w = h * self.scale, w * self.scale
        device, dtype = projected.device, projected.dtype

        # Coordinates of the high-resolution pixel centers measured in the
        # low-resolution feature grid.
        ys = (
            torch.arange(out_h, device=device, dtype=dtype) + 0.5
        ) / float(self.scale) - 0.5
        xs = (
            torch.arange(out_w, device=device, dtype=dtype) + 0.5
        ) / float(self.scale) - 0.5
        base_y, base_x = torch.meshgrid(ys, xs, indexing='ij')
        base_x = base_x.unsqueeze(0).expand(b, -1, -1)
        base_y = base_y.unsqueeze(0).expand(b, -1, -1)

        # Predict chain increments at low resolution, then lift them to the
        # output grid.  tanh bounds the per-step offset; the chain still
        # accumulates over at most ``max_length`` steps.
        offset = self.offset(projected)
        gate = torch.sigmoid(self.target_gate(target)).reshape(b, 1, 1, 1)
        offset = offset + gate * self.target_off(target).reshape(
            b, 2 * self.max_length, 1, 1)
        offset = torch.tanh(offset) * self.max_offset
        offset = F.interpolate(
            offset, size=(out_h, out_w), mode='bilinear',
            align_corners=False)

        k = self.max_length
        c = self.center
        dy = self._cumulative(offset[:, :k])
        dx = self._cumulative(offset[:, k:2 * k])

        # Target-conditioned soft chain length.  A sigmoid gate keeps this
        # differentiable; the paper's odd-length straight-through estimator
        # is unnecessary for a soft weighted aggregation.
        length = 1.0 + float(k - 1) * torch.sigmoid(
            self.length(target)).reshape(b, 1)
        positions = torch.arange(
            k, device=device, dtype=dtype) - float(c)
        gate_k = torch.sigmoid(
            (length - positions.abs().reshape(1, k)) / 0.5)
        gate_k = gate_k / gate_k.sum(dim=1, keepdim=True).clamp_min(1e-6)

        x_major = torch.zeros(
            b, mid, out_h, out_w, device=device, dtype=dtype)
        y_major = torch.zeros_like(x_major)
        for index in range(k):
            weight = gate_k[:, index].reshape(b, 1, 1, 1)
            x_major = x_major + weight * self._sample_bilinear(
                projected,
                base_x + positions[index],
                base_y + dy[:, index],
            )
            y_major = y_major + weight * self._sample_bilinear(
                projected,
                base_x + dx[:, index],
                base_y + positions[index],
            )

        bilinear = F.interpolate(
            projected, size=(out_h, out_w), mode='bilinear',
            align_corners=False)
        residual = self.fuse(torch.cat([bilinear, x_major, y_major], dim=1))
        return self.gamma * residual


class CandidateRanker(nn.Module):
    """PromptMoE-style candidate utility with pairwise interaction.

    Each candidate receives a utility score from its descriptor plus the
    mean pairwise interaction with the other candidates.  The score is a
    drop-in replacement for the IoU prediction used by ``selection=iou``;
    training uses a listwise loss against the per-candidate GT IoU.
    """

    def __init__(
        self,
        dim: int = 256,
        hidden: int = 128,
        pair_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.utility = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.pair = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, pair_hidden),
            nn.GELU(),
            nn.Linear(pair_hidden, 1),
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Return [B, K] candidate scores."""
        if descriptors.ndim != 3:
            raise ValueError(
                f'descriptors must be [B,K,D], got {tuple(descriptors.shape)}')
        scores = self.utility(descriptors).squeeze(-1)
        num = descriptors.shape[1]
        if num < 2:
            return scores
        interaction = torch.zeros_like(scores)
        for i in range(num):
            for j in range(i + 1, num):
                pair = self.pair(torch.cat(
                    [descriptors[:, i], descriptors[:, j]], dim=-1)).squeeze(-1)
                interaction[:, i] = interaction[:, i] + pair
                interaction[:, j] = interaction[:, j] + pair
        return scores + interaction / float(num - 1)


class TargetAdaptiveThreshold(nn.Module):
    """Per-candidate adaptive logit threshold.

    The module predicts a threshold ``t`` in ``[t_min, t_max]`` from each
    candidate descriptor, removes the spatial mean of the logits, and
    subtracts ``t``.  The returned logits therefore have their decision
    boundary at zero for the standard evaluator (``pred_threshold=0``).
    """

    def __init__(
        self,
        dim: int = 256,
        hidden: int = 128,
        t_min: float = -4.0,
        t_max: float = 4.0,
    ) -> None:
        super().__init__()
        if t_max <= t_min:
            raise ValueError(f't_max must exceed t_min, got {t_min}, {t_max}')
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Start at the midpoint (zero in the default symmetric range) so a
        # fresh checkpoint is a centered version of the original logits.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        descriptors: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return threshold-adjusted logits and the per-candidate threshold.

        Args:
            descriptors: [B, K, D] candidate descriptors.
            logits: [B, K, H, W] candidate logits.
        """
        if descriptors.ndim != 3 or logits.ndim != 4:
            raise ValueError('descriptors must be [B,K,D] and logits [B,K,H,W]')
        if descriptors.shape[:2] != logits.shape[:2]:
            raise ValueError(
                f'candidate mismatch: descriptors {tuple(descriptors.shape)} '
                f'vs logits {tuple(logits.shape)}')
        threshold = self.t_min + (self.t_max - self.t_min) * torch.sigmoid(
            self.mlp(descriptors)).squeeze(-1)
        centered = logits - logits.mean(dim=(2, 3), keepdim=True)
        adjusted = centered - threshold[:, :, None, None]
        return adjusted, threshold
