"""TopoWeave: single-stage topology-aware curvilinear segmentation.

The model is a GroupNorm residual U-Net with a gated graph refiner.  Unlike
MaskVSC's two-stage ViG, the refiner is trained jointly with the segmenter
and sees both image features and the coarse probability map, so it can use
image evidence when correcting breaks and false positives.

Tensor contract:
    images: [B, 3, H, W] float in [0, 1]
    coarse/final logits: [B, 1, H, W]
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _norm(channels: int) -> nn.Module:
    groups = 8
    while groups > 1 and channels % groups:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1,
                      bias=False),
            _norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1,
                      bias=False),
            _norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int,
                 out_channels: int):
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(
            x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class GraphBlock(nn.Module):
    """Lightweight kNN graph block with no BatchNorm."""

    def __init__(self, dim: int, k: int = 9):
        super().__init__()
        self.k = int(k)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.src = nn.Linear(dim, dim)
        self.dst = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def _knn(self, x):
        # x: [B, N, D]
        distances = torch.cdist(x, x, p=2)
        n = distances.shape[-1]
        if n <= 1:
            return torch.zeros(
                x.shape[:2] + (1,), dtype=torch.long, device=x.device)
        k = min(self.k, n - 1)
        # Drop the self edge so the k neighbours are informative.
        eye = torch.eye(n, device=x.device, dtype=torch.bool)
        distances = distances.masked_fill(eye.unsqueeze(0), float('inf'))
        return distances.topk(k, dim=-1, largest=False).indices

    def forward(self, x):
        # x: [B, N, D]
        normal = self.norm1(x)
        idx = self._knn(normal)
        tokens, dim = normal.shape[1], normal.shape[2]
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, dim)
        neighbours = torch.gather(
            normal.unsqueeze(2).expand(-1, -1, tokens, -1), 2,
            gather_idx)
        center = normal.unsqueeze(2)
        message = F.relu(self.dst(neighbours) - self.src(center))
        message = message.max(dim=2).values
        x = x + self.out(message)
        return x + self.ffn(self.norm2(x))


class TopoWeaveNet(nn.Module):
    """GroupNorm residual U-Net plus an optional gated graph refiner."""

    def __init__(self, input_size: int = 512, base_width: int = 32,
                 refiner: str = 'graph', graph_blocks: int = 4,
                 graph_dim: int = 128, graph_k: int = 9,
                 graph_stride: int = 16):
        super().__init__()
        if refiner not in ('graph', 'none'):
            raise ValueError(f'unknown refiner: {refiner!r}')
        self.input_size = int(input_size)
        self.refiner = refiner
        self.graph_stride = int(graph_stride)
        self.refiner_enabled = refiner == 'graph'

        widths = [
            base_width,
            base_width * 2,
            base_width * 4,
            base_width * 8,
            base_width * 16,
        ]
        self.stem = ConvBlock(3, widths[0])
        self.down1 = Down(widths[0], widths[1])
        self.down2 = Down(widths[1], widths[2])
        self.down3 = Down(widths[2], widths[3])
        self.down4 = Down(widths[3], widths[4])

        self.up4 = Up(widths[4], widths[3], widths[3])
        self.up3 = Up(widths[3], widths[2], widths[2])
        self.up2 = Up(widths[2], widths[1], widths[1])
        self.up1 = Up(widths[1], widths[0], widths[0])
        self.coarse_head = nn.Conv2d(widths[0], 1, 1)

        self.graph = None
        self.detail = None
        if refiner == 'graph':
            self.graph = nn.ModuleDict({
                'token': nn.Conv2d(widths[4] + 1, graph_dim, 1),
                'pos': _PositionEncoding(graph_dim),
                'blocks': nn.ModuleList([
                    GraphBlock(graph_dim, graph_k)
                    for _ in range(graph_blocks)
                ]),
                'out': nn.Linear(graph_dim, 1),
            })
            self.detail = nn.Sequential(
                nn.Conv2d(widths[0] + 2, widths[0], 3, padding=1,
                          bias=False),
                _norm(widths[0]),
                nn.SiLU(inplace=True),
                nn.Conv2d(widths[0], 1, 1),
            )
            self.gate = nn.Parameter(torch.zeros(1))

    def set_refiner_enabled(self, enabled: bool) -> None:
        self.refiner_enabled = bool(enabled) and self.refiner == 'graph'

    def _refine(self, coarse, feature, decoder):
        # feature: [B, C, H/16, W/16], decoder: [B, C0, H, W]
        probability = torch.sigmoid(coarse).detach()
        prob16 = F.interpolate(
            probability, size=feature.shape[-2:], mode='area')
        tokens = self.graph['token'](torch.cat([feature, prob16], dim=1))
        batch, dim, height, width = tokens.shape
        x = tokens.flatten(2).transpose(1, 2)  # [B, N, D]
        x = x + self.graph['pos'](height, width, x.device, x.dtype)
        for block in self.graph['blocks']:
            x = block(x)
        correction = self.graph['out'](x).transpose(1, 2).reshape(
            batch, 1, height, width)
        correction = F.interpolate(
            correction, size=coarse.shape[-2:], mode='bilinear',
            align_corners=False)
        detail = self.detail(torch.cat(
            [correction, decoder, probability], dim=1))
        scale = 0.1 * torch.sigmoid(self.gate)
        return coarse + scale * detail

    def forward(self, images, return_aux: bool = False):
        x0 = self.stem(images)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        d = self.up4(x4, x3)
        d = self.up3(d, x2)
        d = self.up2(d, x1)
        d = self.up1(d, x0)
        coarse = self.coarse_head(d)

        final = coarse
        if self.refiner == 'graph' and self.refiner_enabled:
            final = self._refine(coarse, x4, d)
        if return_aux:
            return {'logits': final, 'coarse_logits': coarse}
        return final


class _PositionEncoding(nn.Module):
    """Fixed 2D sinusoidal positional encoding for graph tokens."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 4:
            raise ValueError('graph_dim must be divisible by 4')
        self.dim = dim
        half = dim // 2
        div = torch.exp(torch.arange(0, half, 2).float()
                        * (-math.log(10000.0) / max(half - 2, 1)))
        self.register_buffer('div', div, persistent=False)

    def forward(self, height, width, device, dtype):
        y = torch.arange(height, device=device, dtype=torch.float32)
        x = torch.arange(width, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        yy = yy.reshape(-1, 1)
        xx = xx.reshape(-1, 1)
        div = self.div.to(device=device, dtype=torch.float32)
        pe_y = torch.cat([torch.sin(yy * div), torch.cos(yy * div)], dim=1)
        pe_x = torch.cat([torch.sin(xx * div), torch.cos(xx * div)], dim=1)
        encoding = torch.cat([pe_y, pe_x], dim=1)
        if encoding.shape[1] < self.dim:
            encoding = F.pad(encoding, (0, self.dim - encoding.shape[1]))
        return encoding[:, :self.dim].unsqueeze(0).to(dtype=dtype)
