"""TCSU-lite: image-guided snake upsampling for BCTD candidate 3.

This is a prompt-free adaptation of PEPA's reconstruction bottleneck.  The
deployed candidate 3 is bilinearly upsampled and a zero-initialized residual
is learned from multi-scale image structure evidence and a dynamic snake
convolution.  The residual is supervised only by the source dense mask loss;
the base BCTD loss, candidate set, fixed candidate-3 deployment, 3-shot
protocol, and evaluator are unchanged.  At initialization the output is
exactly the canonical bilinear prediction.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F



def _fixed_kernel(values: list[list[float]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).view(1, 1, 3, 3)


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if channels % 8 == 0 else 4, channels)


class StructuralEvidence(nn.Module):
    """Label-free grayscale, gradient, Laplacian, variance, ridge maps."""

    def __init__(self, out_size: int = 1024) -> None:
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
        out = torch.cat([gray, mag, lap, var] + ridges, dim=1)
        mean = out.mean(dim=(2, 3), keepdim=True)
        std = out.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
        return ((out - mean) / std).clamp(-4.0, 4.0)


class TubularDeformConv2D(nn.Module):
    """Dynamic snake convolution with cumulative horizontal/vertical offsets."""

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
        return self.norm(self.reduce(torch.cat(samples, dim=1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self._sample_line(x, True)
                      + self._sample_line(x, False))


class SnakeUpsampler(nn.Module):
    """Bilinear base plus a zero-initialized image-guided residual."""

    def __init__(self, render_size: int = 1024, feature_dim: int = 16,
                 coarse_size: int = 512) -> None:
        super().__init__()
        self.render_size = int(render_size)
        self.coarse_size = int(coarse_size)
        self.evidence = StructuralEvidence(out_size=self.render_size)
        self.image_stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            _group_norm(16),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
        )
        self.evidence_proj = nn.Conv2d(7, 16, 3, padding=1)
        self.coarse_proj = nn.Conv2d(1, 8, 3, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(16 + 16 + 8, feature_dim, 3, padding=1),
            _group_norm(feature_dim),
            nn.GELU(),
        )
        self.snake = TubularDeformConv2D(feature_dim, kernel_size=5)
        self.snake_fuse = nn.Sequential(
            nn.Conv2d(2 * feature_dim, feature_dim, 3, padding=1),
            _group_norm(feature_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, stage2: torch.Tensor,
                images: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(
            stage2.detach()[:, 3:4],
            size=(self.render_size, self.render_size),
            mode='bilinear', align_corners=False)
        coarse = F.interpolate(
            stage2.detach()[:, 3:4],
            size=(self.coarse_size, self.coarse_size),
            mode='bilinear', align_corners=False)
        evidence = self.evidence(images)
        ev = F.interpolate(
            evidence, size=(self.coarse_size, self.coarse_size),
            mode='bilinear', align_corners=False)
        img = F.interpolate(
            images, size=(self.coarse_size, self.coarse_size),
            mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat(
            [self.image_stem(img), self.evidence_proj(ev),
             self.coarse_proj(coarse)], dim=1))
        snake = self.snake(F.interpolate(
            fused, size=(self.coarse_size // 2, self.coarse_size // 2),
            mode='bilinear', align_corners=False))
        snake = F.interpolate(
            snake, size=(self.coarse_size, self.coarse_size),
            mode='bilinear', align_corners=False)
        fused = self.snake_fuse(torch.cat([fused, snake], dim=1))
        residual = F.interpolate(
            self.decoder(fused), size=(self.render_size, self.render_size),
            mode='bilinear', align_corners=False)
        return base + residual


