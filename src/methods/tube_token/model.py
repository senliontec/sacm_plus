"""Structure-token post-decode control for DCD.

The DCD candidate generator, Stage-1 gating, and fixed candidate-3
deployment are left untouched.  This module only produces an image-derived
structure field and uses it to control candidate 3 after decoding:

* a native-resolution residual added before DCD's topology losses;
* a high-resolution structure-guided residual used by the dense mask loss;
* an inference-time skeleton-calibrated threshold.

No new loss term is introduced.  The structure tokens are never fed to the
SAM decoder, so the physical identity of candidate 3 cannot drift through a
changed gating order.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _fixed_kernel(values: list[list[float]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).view(1, 1, 3, 3)


class StructuralEvidence(nn.Module):
    """Deterministic, label-free multi-scale structural maps.

    Channels: grayscale, gradient magnitude, Laplacian, local variance,
    and Hessian ridge responses at three scales.  Both bright and dark
    ridges are represented through the absolute Hessian response.
    """

    def __init__(self, out_size: int = 256) -> None:
        super().__init__()
        self.out_size = int(out_size)
        sobel_x = _fixed_kernel([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = _fixed_kernel([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        lap = _fixed_kernel([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        gauss = _fixed_kernel(
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('laplacian', lap)
        self.register_buffer('gaussian', gauss)

    @staticmethod
    def _gray(images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 1:
            return images[:, :1]
        return (images[:, 0:1] * 0.299 + images[:, 1:2] * 0.587
                + images[:, 2:3] * 0.114)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        gray = self._gray(images)
        if gray.shape[-1] != self.out_size:
            gray = F.interpolate(
                gray, size=(self.out_size, self.out_size),
                mode='bilinear', align_corners=False)
        blur = F.conv2d(gray, self.gaussian, padding=1)
        gx = F.conv2d(blur, self.sobel_x, padding=1)
        gy = F.conv2d(blur, self.sobel_y, padding=1)
        mag = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(blur, self.laplacian, padding=1).abs()
        mean = F.avg_pool2d(blur, 5, 1, 2)
        mean_sq = F.avg_pool2d(blur.square(), 5, 1, 2)
        var = (mean_sq - mean.square()).clamp_min(0.0).sqrt()

        ridges = []
        # Fixed multi-scale Hessian surrogate.  The absolute response keeps
        # both bright and dark curvilinear structures.
        for scale in (1, 2, 4):
            x = blur if scale == 1 else F.avg_pool2d(
                blur, scale, scale, ceil_mode=True)
            x = F.interpolate(
                x, size=blur.shape[-2:], mode='bilinear',
                align_corners=False)
            gxx = F.conv2d(F.conv2d(x, self.sobel_x, padding=1),
                           self.sobel_x, padding=1)
            gyy = F.conv2d(F.conv2d(x, self.sobel_y, padding=1),
                           self.sobel_y, padding=1)
            gxy = F.conv2d(F.conv2d(x, self.sobel_x, padding=1),
                           self.sobel_y, padding=1)
            disc = torch.sqrt((gxx - gyy).square() + 4.0 * gxy.square()
                              + 1e-6)
            lam1 = 0.5 * (gxx + gyy + disc)
            lam2 = 0.5 * (gxx + gyy - disc)
            ridges.append((lam1.abs() - lam2.abs()).abs())

        channels = [gray, mag, lap, var] + ridges
        out = torch.cat(channels, dim=1)
        mean = out.mean(dim=(2, 3), keepdim=True)
        std = out.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
        return ((out - mean) / std).clamp(-4.0, 4.0)


class StructureControl(nn.Module):
    """Image-derived tube tokens and the post-decode control path."""

    def __init__(
        self,
        adapter_dim: int = 1024,
        feature_dim: int = 64,
        grid_h: int = 8,
        grid_w: int = 16,
        render_size: int = 128,
        highres_size: int = 1024,
        residual_weight: float = 0.2,
        residual_prior: float = 0.5,
        decision_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.num_tokens = self.grid_h * self.grid_w
        self.render_size = int(render_size)
        self.highres_size = int(highres_size)
        self.residual_prior = float(residual_prior)
        self.decision_weight = float(decision_weight)

        self.evidence = StructuralEvidence(out_size=256)
        self.adapter_proj = nn.Conv2d(adapter_dim, 16, 1)
        self.evidence_proj = nn.Conv2d(7, 16, 3, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(64 + 16, feature_dim, 3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, feature_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.token_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=4, dim_feedforward=2 * feature_dim,
            dropout=0.0, activation='gelu', batch_first=True,
            norm_first=True)
        self.token_attn = nn.TransformerEncoder(layer, num_layers=2)
        self.geom_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 6),
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_weight)))
        self.high_scale = nn.Parameter(torch.tensor(0.05))

        self.image_stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
        )
        self.high_refiner = nn.Sequential(
            nn.Conv2d(4 + 16 + 7 + 1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        nn.init.normal_(self.high_refiner[-1].weight, std=1e-3)
        nn.init.zeros_(self.high_refiner[-1].bias)

    def _tokens(
        self,
        adapter_features: list[torch.Tensor] | None,
        images: torch.Tensor,
        evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if adapter_features is None or len(adapter_features) < 4:
            raise RuntimeError(
                'structure control requires at least four adapter features')
        feats = []
        for feat in adapter_features[-4:]:
            # ViT adapter outputs are [B, H, W, C].
            x = feat.permute(0, 3, 1, 2).contiguous()
            feats.append(self.adapter_proj(x))
        feat_map = torch.cat(feats, dim=1)
        ev = F.interpolate(
            evidence, size=feat_map.shape[-2:], mode='bilinear',
            align_corners=False)
        fused = self.fuse(torch.cat([feat_map, self.evidence_proj(ev)], 1))
        kh = fused.shape[-2] // self.grid_h
        kw = fused.shape[-1] // self.grid_w
        if (kh * self.grid_h == fused.shape[-2]
                and kw * self.grid_w == fused.shape[-1]):
            pooled = F.avg_pool2d(fused, (kh, kw))
        else:
            pooled = F.adaptive_avg_pool2d(
                fused, (self.grid_h, self.grid_w))
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.token_mlp(tokens + self.pos_embed)
        tokens = self.token_attn(tokens)
        geom = self.geom_head(tokens)
        return tokens, geom

    def _render(self, geom: torch.Tensor, size: int) -> torch.Tensor:
        b = geom.shape[0]
        gh, gw = self.grid_h, self.grid_w
        device = geom.device
        offsets = torch.tanh(geom[..., :2]) * 0.5
        theta = math.pi * torch.tanh(geom[..., 2])
        length = 0.02 + 0.08 * torch.sigmoid(geom[..., 3])
        width = 0.0008 + 0.006 * torch.sigmoid(geom[..., 4])
        opacity = 0.95 * torch.sigmoid(geom[..., 5])
        ys, xs = torch.meshgrid(
            (torch.arange(size, device=device, dtype=geom.dtype) + 0.5)
            / size,
            (torch.arange(size, device=device, dtype=geom.dtype) + 0.5)
            / size,
            indexing='ij')
        cell_x = (torch.arange(gw, device=device, dtype=geom.dtype) + 0.5) / gw
        cell_y = (torch.arange(gh, device=device, dtype=geom.dtype) + 0.5) / gh
        base_y, base_x = torch.meshgrid(cell_y, cell_x, indexing='ij')
        base_x = base_x.reshape(1, -1)
        base_y = base_y.reshape(1, -1)
        cx = base_x + offsets[..., 0] / gw
        cy = base_y + offsets[..., 1] / gh
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        dx = xs.view(1, 1, size, size) - cx.view(b, -1, 1, 1)
        dy = ys.view(1, 1, size, size) - cy.view(b, -1, 1, 1)
        u = dx * cos_t.view(b, -1, 1, 1) + dy * sin_t.view(b, -1, 1, 1)
        v = -dx * sin_t.view(b, -1, 1, 1) + dy * cos_t.view(b, -1, 1, 1)
        g = torch.exp(-0.5 * (
            (u / length.view(b, -1, 1, 1)).square()
            + (v / width.view(b, -1, 1, 1)).square()))
        occ = (opacity.view(b, -1, 1, 1) * g).clamp(max=0.999)
        log_free = torch.log1p(-occ).sum(dim=1, keepdim=True)
        return 1.0 - torch.exp(log_free.clamp_min(-20.0))

    def forward(
        self,
        adapter_features: list[torch.Tensor] | None,
        images: torch.Tensor,
        stage2: torch.Tensor,
        residual_mult: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        evidence = self.evidence(images)
        _, geom = self._tokens(adapter_features, images, evidence)
        structure = self._render(geom, self.render_size)
        structure_native = F.interpolate(
            structure, size=stage2.shape[-2:], mode='bilinear',
            align_corners=False)
        stage2 = stage2.clone()
        # Zero-mean centering avoids a global bias on the recall-heavy
        # candidate 3.  Structure-supported pixels receive positive
        # evidence, unsupported background receives negative evidence, and
        # the image-level mean correction stays near zero.
        center = structure_native.mean(dim=(1, 2, 3), keepdim=True)
        p3 = torch.sigmoid(stage2[:, 3:4])
        uncertainty = (4.0 * p3 * (1.0 - p3)).clamp(0.0, 1.0)
        stage2[:, 3:4] = stage2[:, 3:4] + (
            self.residual_scale * float(residual_mult)
            * (structure_native - center) * uncertainty)
        cache = {
            'structure': structure,
            'evidence': evidence,
            'geom': geom,
            'uncertainty': uncertainty,
        }
        return stage2, structure, cache

    def highres(
        self,
        stage2: torch.Tensor,
        images: torch.Tensor,
        structure: torch.Tensor,
        evidence: torch.Tensor,
        highres_mult: float = 1.0,
        uncertainty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        size = self.highres_size
        high = F.interpolate(
            stage2, size=(size, size), mode='bilinear',
            align_corners=False)
        mid = size // 2
        low = F.interpolate(
            stage2, size=(mid, mid), mode='bilinear',
            align_corners=False)
        img = F.interpolate(
            images, size=(mid, mid), mode='bilinear',
            align_corners=False)
        ev = F.interpolate(
            evidence, size=(mid, mid), mode='bilinear',
            align_corners=False)
        st = F.interpolate(
            structure, size=(mid, mid), mode='bilinear',
            align_corners=False)
        feat = self.image_stem(img)
        residual = self.high_refiner(torch.cat([low, feat, ev, st], 1))
        # Remove the image-level foreground bias.  Without this, the dense
        # BCE+Dice gradient can turn the refiner into a global recall booster,
        # which raises Dice but breaks clDice/HD95.
        residual = residual - residual.mean(dim=(2, 3), keepdim=True)
        # Restrict the high-resolution correction to image-derived structure
        # and to the low-confidence decision band.  Confident background and
        # confident foreground stay exactly as DCD produced them, so the
        # refiner cannot introduce a global recall or boundary-shift bias.
        gate = st.clamp(0.0, 1.0)
        if uncertainty is not None:
            u = F.interpolate(
                uncertainty, size=(mid, mid), mode='bilinear',
                align_corners=False)
            gate = gate * u.clamp(0.0, 1.0)
        residual = residual * gate
        residual = F.interpolate(
            residual, size=(size, size), mode='bilinear',
            align_corners=False)
        high[:, 3:4] = high[:, 3:4] + (
            self.high_scale * float(highres_mult) * residual)
        return high

    def decision_threshold(
        self,
        structure: torch.Tensor,
        size: tuple[int, int],
        base: float = 0.5,
        weight: float | None = None,
        uncertainty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s = F.interpolate(
            structure, size=size, mode='bilinear',
            align_corners=False)
        w = self.decision_weight if weight is None else float(weight)
        if uncertainty is not None:
            u = F.interpolate(
                uncertainty, size=size, mode='bilinear',
                align_corners=False)
            s = s * u.clamp(0.0, 1.0)
        return (base - w * s).clamp(0.05, 0.95)
