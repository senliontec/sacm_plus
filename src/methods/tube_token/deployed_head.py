"""Structure-aware deployed head for the TubeToken experiment.

The previous post-decode branch failed because it only touched candidate 3
through a tiny, uncertainty-gated residual.  This module instead makes the
deployed candidate 3 the output of a structure-aware renderer:

* coarse structure decoder at 512 resolution;
* dynamic tubular convolution (snake sampling) for curvilinear geometry;
* high-resolution rendering at 1024 from image structural evidence;
* no uncertainty gate, no zero-mean centering, no hand-tuned residual scale.

The head is trained by the unchanged DCD dense BCE+Dice and topology losses:
it is on the deployed path, so it cannot collapse into a no-op.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import StructuralEvidence


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = 8 if channels % 8 == 0 else 4
    return nn.GroupNorm(groups, channels)


class TubularDeformConv2D(nn.Module):
    """A lightweight dynamic snake convolution.

    Inspired by Dynamic Snake Convolution (ICCV 2023): the kernel samples
    along a learned, cumulatively deformed line instead of a fixed grid.
    Both horizontal and vertical snakes are used so arbitrary curve
    orientations are covered.
    """

    def __init__(self, channels: int, kernel_size: int = 5,
                 extend_scope: float = 0.5) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.extend_scope = float(extend_scope)
        k = self.kernel_size
        self.offset = nn.Conv2d(channels, 2 * k, 3, padding=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.reduce = nn.Conv2d(channels * k, channels, 1)
        self.norm = _group_norm(channels)

    def _sample_line(self, x: torch.Tensor, horizontal: bool
                     ) -> torch.Tensor:
        b, c, h, w = x.shape
        k = self.kernel_size
        offset = torch.tanh(self.offset(x)) * self.extend_scope
        offset = offset.view(b, 2, k, h, w)
        # Cumulative offsets make the kernel follow a snake rather than
        # independently jitter each sample.
        dx = offset[:, 0].cumsum(dim=1)
        dy = offset[:, 1].cumsum(dim=1)
        base = torch.arange(k, device=x.device, dtype=x.dtype)
        base = base - (k - 1) / 2.0
        ys = torch.arange(h, device=x.device, dtype=x.dtype).view(
            1, 1, h, 1).expand(b, k, h, w)
        xs = torch.arange(w, device=x.device, dtype=x.dtype).view(
            1, 1, 1, w).expand(b, k, h, w)
        if horizontal:
            sx = xs + base.view(1, k, 1, 1) + dx
            sy = ys + dy
        else:
            sx = xs + dx
            sy = ys + base.view(1, k, 1, 1) + dy
        gx = 2.0 * sx.clamp(0, w - 1) / max(w - 1, 1) - 1.0
        gy = 2.0 * sy.clamp(0, h - 1) / max(h - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1)
        samples = [
            F.grid_sample(x, grid[:, i], mode='bilinear',
                          padding_mode='border', align_corners=True)
            for i in range(k)
        ]
        out = self.reduce(torch.cat(samples, dim=1))
        return self.norm(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self._sample_line(x, True)
                      + self._sample_line(x, False))


class DeployedStructureHead(nn.Module):
    """High-resolution structure renderer for the deployed candidate."""

    def __init__(
        self,
        adapter_dim: int = 1024,
        feature_dim: int = 32,
        coarse_size: int = 512,
        render_size: int = 1024,
        snake_kernel: int = 5,
        render_channels: int = 16,
        scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.coarse_size = int(coarse_size)
        self.render_size = int(render_size)
        self.evidence = StructuralEvidence(out_size=self.render_size)
        self.adapter_proj = nn.Conv2d(4 * adapter_dim, feature_dim, 1)
        self.image_stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            _group_norm(16),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
        )
        self.evidence_proj = nn.Conv2d(7, 16, 3, padding=1)
        self.coarse_proj = nn.Conv2d(1, 8, 3, padding=1)
        fused_in = feature_dim + 16 + 16 + 8
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_in, feature_dim, 3, padding=1),
            _group_norm(feature_dim),
            nn.GELU(),
        )
        self.snake = TubularDeformConv2D(
            feature_dim, kernel_size=snake_kernel)
        self.snake_fuse = nn.Sequential(
            nn.Conv2d(2 * feature_dim, feature_dim, 3, padding=1),
            _group_norm(feature_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_dim, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.render = nn.Sequential(
            nn.Conv2d(1 + feature_dim + 7 + 3 + 1,
                      render_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(render_channels, 8, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 1, 1),
        )
        nn.init.zeros_(self.render[-1].weight)
        nn.init.zeros_(self.render[-1].bias)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(
        self,
        adapter_features: list[torch.Tensor] | None,
        images: torch.Tensor,
        stage2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if adapter_features is None or len(adapter_features) < 4:
            raise RuntimeError(
                'deployed structure head needs four adapter features')
        coarse = self.coarse_size
        render = self.render_size
        # ---- coarse 512 features --------------------------------------
        feats = []
        for feat in adapter_features[-4:]:
            feats.append(feat.permute(0, 3, 1, 2).contiguous())
        adapter = self.adapter_proj(torch.cat(feats, dim=1))
        adapter = F.interpolate(
            adapter, size=(coarse, coarse), mode='bilinear',
            align_corners=False)
        evidence = self.evidence(images)
        ev_coarse = F.interpolate(
            evidence, size=(coarse, coarse), mode='bilinear',
            align_corners=False)
        img_coarse = F.interpolate(
            images, size=(coarse, coarse), mode='bilinear',
            align_corners=False)
        p3 = F.interpolate(
            stage2[:, 3:4], size=(coarse, coarse), mode='bilinear',
            align_corners=False)
        fused = self.fuse(torch.cat([
            adapter,
            self.image_stem(img_coarse),
            self.evidence_proj(ev_coarse),
            self.coarse_proj(p3),
        ], dim=1))
        # Snake sampling at 256 keeps memory low; upsample back to 512.
        snake_in = F.interpolate(
            fused, size=(coarse // 2, coarse // 2), mode='bilinear',
            align_corners=False)
        snake = self.snake(snake_in)
        snake = F.interpolate(
            snake, size=(coarse, coarse), mode='bilinear',
            align_corners=False)
        fused = self.snake_fuse(torch.cat([fused, snake], dim=1))
        corr_coarse = self.decoder(fused)
        # ---- topology / role path at 256 -------------------------------
        corr_native = F.interpolate(
            corr_coarse, size=stage2.shape[-2:], mode='bilinear',
            align_corners=False)
        stage2_top = stage2.clone()
        stage2_top[:, 3:4] = (
            stage2_top[:, 3:4] + self.scale * corr_native)
        # ---- high-resolution rendering at 1024 -------------------------
        corr_high = F.interpolate(
            corr_coarse, size=(render, render), mode='bilinear',
            align_corners=False)
        fused_high = F.interpolate(
            fused, size=(render, render), mode='bilinear',
            align_corners=False)
        ev_high = evidence
        img_high = F.interpolate(
            images, size=(render, render), mode='bilinear',
            align_corners=False)
        p3_high = F.interpolate(
            p3, size=(render, render), mode='bilinear',
            align_corners=False)
        detail = self.render(torch.cat([
            corr_high, fused_high, ev_high, img_high, p3_high], dim=1))
        stage2_high = F.interpolate(
            stage2, size=(render, render), mode='bilinear',
            align_corners=False)
        base3 = stage2_high[:, 3:4]
        stage2_high[:, 3:4] = (
            base3 + self.scale * (corr_high + detail))
        cache = {
            'stage2_high': stage2_high,
            'corr_coarse': corr_coarse,
            'detail': detail,
            'scale': self.scale.detach(),
        }
        return stage2_top, stage2_high, cache


class StructuralPromptGenerator(nn.Module):
    """Image-derived dense mask prompt for the SAM decoder.

    The prompt is a learned soft mask built from deterministic ridge and
    gradient evidence.  Its scale starts small and is trained by the same
    DCD losses through the frozen/adapted SAM prompt encoder and decoder,
    so the prompt is a deployed architectural component rather than a
    detached post-hoc prior.
    """

    def __init__(self, evidence_size: int = 256, channels: int = 16,
                 scale_init: float = 0.1) -> None:
        super().__init__()
        self.evidence = StructuralEvidence(out_size=evidence_size)
        self.net = nn.Sequential(
            nn.Conv2d(7, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        evidence = self.evidence(images)
        prompt = torch.sigmoid(self.net(evidence))
        return prompt, self.scale


class SelfPromptGenerator(nn.Module):
    """Target-adaptive dense prompt from the model's own coarse prediction.

    The coarse candidate-3 logit, its uncertainty, the inter-head spread and
    image structural evidence are fused into a soft mask prompt.  A second
    SAM decoder pass consumes this prompt, so the refinement reuses the
    pretrained decoder instead of fitting a new high-resolution conv head.
    """

    def __init__(self, evidence_size: int = 256, channels: int = 16,
                 scale_init: float = 0.1) -> None:
        super().__init__()
        self.evidence = StructuralEvidence(out_size=evidence_size)
        self.net = nn.Sequential(
            nn.Conv2d(1 + 2 + 7, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, stage2: torch.Tensor,
                images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coarse = stage2.detach()
        p3 = torch.sigmoid(coarse[:, 3:4])
        uncertainty = (4.0 * p3 * (1.0 - p3)).clamp(0.0, 1.0)
        spread = coarse[:, :4].std(dim=1, keepdim=True)
        spread = torch.sigmoid(spread)
        evidence = self.evidence(images)
        if evidence.shape[-2:] != coarse.shape[-2:]:
            evidence = F.interpolate(
                evidence, size=coarse.shape[-2:], mode='bilinear',
                align_corners=False)
        prompt = torch.sigmoid(self.net(torch.cat(
            [coarse[:, 3:4], uncertainty, spread, evidence], dim=1)))
        return prompt, self.scale
