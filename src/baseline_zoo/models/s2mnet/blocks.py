"""PyTorch rewrite of S2M-Net's three novel blocks.

Translated from the TensorFlow/Keras release
(``s2mnet/models/blocks.py``).  The architecture, channel counts, kernel
sizes, SE reduction, BFP routing and residual connections are unchanged.
Keras defaults are matched where they affect behavior: Conv2D biases, ELU
(alpha 1), BatchNorm eps 1e-3 / momentum 0.01, elementwise dropout and
LayerNorm over channels.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _bn(channels: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)


class ChannelLayerNorm(nn.Module):
    """Keras ``LayerNormalization(axis=-1)`` on NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SpectralSelectiveTokenMixer(nn.Module):
    """FFT spectral filter + gated dense projection, fused residually."""

    def __init__(self, channels: int, num_frequencies: int = 32,
                 spatial_size: int | None = None,
                 ssm_state_dim: int = 16, use_spectral: bool = True,
                 use_ssm: bool = True, dropout: float = 0.0):
        super().__init__()
        if spatial_size is None:
            spatial_size = num_frequencies
        self.channels = int(channels)
        self.spatial_size = int(spatial_size)
        self.k = int(min(num_frequencies, spatial_size))
        self.use_spectral = bool(use_spectral)
        self.use_ssm = bool(use_ssm)
        self.ssm_state_dim = int(ssm_state_dim)
        self.dropout = nn.Dropout(dropout)

        if self.use_spectral:
            self.freq_weights = nn.Parameter(
                torch.from_numpy(self._gaussian_init()).float())
            self.spectral_norm = ChannelLayerNorm(channels)
        if self.use_ssm:
            self.ssm_proj = nn.Linear(channels, channels)
            self.ssm_gate = nn.Linear(channels, channels)
            self.ssm_norm = ChannelLayerNorm(channels)
        if self.use_spectral and self.use_ssm:
            self.fusion = nn.Conv2d(2 * channels, channels, 1)
            self.fusion_norm = ChannelLayerNorm(channels)
        self.out_norm = ChannelLayerNorm(channels)

    def _gaussian_init(self) -> np.ndarray:
        frequencies = np.fft.fftfreq(self.k)
        magnitude = np.sqrt(
            frequencies[:, None] ** 2 + frequencies[None, :] ** 2)
        gaussian = np.exp(-((magnitude - 0.25) ** 2) / (2 * 0.15 ** 2))
        gaussian = np.repeat(gaussian[:, :, None], self.channels, axis=2)
        return (gaussian * 0.5).astype(np.float32)

    def _spectral_path(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        k = min(self.k, height, width)
        spectrum = torch.fft.fft2(x.float(), dim=(-2, -1))
        real = F.interpolate(
            spectrum.real, size=(k, k), mode='bilinear',
            align_corners=False)
        imag = F.interpolate(
            spectrum.imag, size=(k, k), mode='bilinear',
            align_corners=False)
        filt = self.freq_weights[:k, :k, :].permute(2, 0, 1).unsqueeze(0)
        filtered = torch.complex(real, imag) * filt.to(real.dtype)
        filtered_real = F.interpolate(
            filtered.real, size=(height, width), mode='bilinear',
            align_corners=False)
        filtered_imag = F.interpolate(
            filtered.imag, size=(height, width), mode='bilinear',
            align_corners=False)
        output = torch.fft.ifft2(
            torch.complex(filtered_real, filtered_imag),
            dim=(-2, -1)).real
        return self.spectral_norm(output)

    def _ssm_path(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.permute(0, 2, 3, 1)
        gate = torch.sigmoid(self.ssm_gate(tokens))
        projected = self.ssm_proj(tokens * gate)
        return self.ssm_norm(projected.permute(0, 3, 1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.use_spectral:
            parts.append(self._spectral_path(x))
        if self.use_ssm:
            parts.append(self._ssm_path(x))
        if len(parts) == 2:
            fused = self.fusion_norm(
                self.fusion(torch.cat(parts, dim=1)))
        elif len(parts) == 1:
            fused = parts[0]
        else:
            fused = x
        return x + self.dropout(self.out_norm(fused))


class MRFSEBlock(nn.Module):
    """Multi-receptive-field depthwise expansion with SE attention."""

    def __init__(self, filters: int, kernels=(3, 5, 7),
                 se_reduction: int = 16, expand_ratio: int = 6,
                 activation: str = 'elu', dropout: float = 0.0):
        super().__init__()
        if activation.lower() != 'elu':
            raise ValueError('S2M-Net uses ELU activations')
        expanded = filters * expand_ratio
        self.expand = nn.Conv2d(filters, expanded, 1)
        self.expand_bn = _bn(expanded)
        self.expand_act = nn.ELU()
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(expanded, expanded, kernel, padding=kernel // 2,
                      groups=expanded)
            for kernel in kernels
        ])
        self.dw_bns = nn.ModuleList([_bn(expanded) for _ in kernels])
        self.dw_act = nn.ELU()
        self.fuse = None
        if len(kernels) > 1:
            self.fuse = nn.Conv2d(
                expanded * len(kernels), expanded, 1)
            self.fuse_bn = _bn(expanded)
            self.fuse_act = nn.ELU()
        se_mid = max(expanded // se_reduction, 8)
        self.se_reduce = nn.Conv2d(expanded, se_mid, 1)
        self.se_expand = nn.Conv2d(se_mid, expanded, 1)
        self.proj = nn.Conv2d(expanded, filters, 1)
        self.proj_bn = _bn(filters)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.expand_act(self.expand_bn(self.expand(x)))
        branches = [
            act(bn(conv(h)))
            for conv, bn, act in zip(
                self.dw_convs, self.dw_bns, [self.dw_act] * len(self.dw_bns))
        ]
        if self.fuse is not None:
            combined = self.fuse_act(self.fuse_bn(
                self.fuse(torch.cat(branches, dim=1))))
        else:
            combined = branches[0]
        squeezed = combined.mean(dim=(2, 3), keepdim=True)
        attention = torch.sigmoid(
            self.se_expand(F.elu(self.se_reduce(squeezed))))
        combined = combined * attention
        out = self.proj_bn(self.proj(combined))
        return self.dropout(out) + x


class BFPDecoderStage(nn.Module):
    """Boundary-focused progressive decoder stage."""

    ROUTING_MODES = ('soft', 'hard', 'learned', 'none')

    def __init__(self, in_channels: int, filters: int,
                 routing: str = 'soft'):
        super().__init__()
        if routing not in self.ROUTING_MODES:
            raise ValueError(f'unsupported BFP routing: {routing}')
        self.filters = int(filters)
        self.routing = routing
        self.region_conv1 = nn.Conv2d(
            in_channels, filters, 3, padding=1)
        self.region_bn1 = _bn(filters)
        self.region_conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.region_bn2 = _bn(filters)
        self.bnd_conv = nn.Conv2d(filters, filters // 2, 3, padding=1)
        self.bnd_map = nn.Conv2d(filters // 2, 1, 1)
        self.refine_conv = nn.Conv2d(filters, filters, 3, padding=1)
        self.refine_bn = _bn(filters)
        fusion_in = filters * 2 if routing == 'none' else filters
        self.fusion_conv = nn.Conv2d(fusion_in, filters, 1)
        self.fusion_bn = _bn(filters)
        self.route_conv = None
        if routing == 'learned':
            self.route_conv = nn.Conv2d(filters * 2, 2, 1)

    def forward(self, decoder_input: torch.Tensor,
                skip_features: torch.Tensor):
        upsampled = F.interpolate(
            decoder_input, size=skip_features.shape[-2:],
            mode='bilinear', align_corners=False)
        x = torch.cat([upsampled, skip_features], dim=1)
        region = F.relu(self.region_bn1(self.region_conv1(x)))
        region = F.relu(self.region_bn2(self.region_conv2(region)))
        boundary_features = F.relu(self.bnd_conv(region))
        boundary = torch.sigmoid(self.bnd_map(boundary_features))
        refined = F.relu(self.refine_bn(
            self.refine_conv(region * boundary)))
        if self.routing == 'soft':
            blended = region * (1.0 - boundary) + refined * boundary
        elif self.routing == 'hard':
            beta = (boundary > 0.5).float()
            blended = region * (1.0 - beta) + refined * beta
        elif self.routing == 'learned':
            weights = torch.softmax(
                self.route_conv(torch.cat([region, refined], dim=1)),
                dim=1)
            blended = (region * weights[:, 0:1]
                       + refined * weights[:, 1:2])
        else:
            blended = torch.cat([region, refined], dim=1)
        out = F.relu(self.fusion_bn(self.fusion_conv(blended)))
        return out, boundary
