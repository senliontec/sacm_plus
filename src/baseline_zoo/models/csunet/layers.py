"""CS-UNet layers, translated from the official TensorFlow notebook.

The notebook's ``DynamicSnakeConv2D`` is explicitly a placeholder:
``ConvGNRelu`` is a plain 3x3 convolution (no deformable offsets) followed by
GroupNorm(16), ReLU and the fixed-alpha Snake activation.  This module
reproduces that released behavior; the paper describes a periodic-activated
convolution block (PACB) with batch normalization, but the code uses
GroupNorm(16) + ReLU, which is what the port follows.

The Gegenbauer Feature Refinement (GFR/FSF) module here is the notebook's
``fsf_block``: concatenate [decoder, skip], channel SE with global max,
reduction 8, degree-5 Gegenbauer activations with a learnable softplus alpha
and sigmoid-bounded inputs, a 1x1 compression to half channels, a global-max
sigmoid gate on the skip, and ``skip * gate + decoder``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gegenbauer_polynomial(x: torch.Tensor, degree: int,
                          alpha: torch.Tensor) -> torch.Tensor:
    """Gegenbauer C_degree^(alpha)(x) by the notebook's recurrence."""
    c_prev = torch.ones_like(x)
    if degree == 0:
        return c_prev
    c_curr = 2.0 * alpha * x
    if degree == 1:
        return c_curr
    for order in range(2, degree + 1):
        coef1 = 2.0 * (order + alpha - 1.0) / order
        coef2 = (order + 2.0 * alpha - 2.0) / order
        c_next = coef1 * x * c_curr - coef2 * c_prev
        c_prev, c_curr = c_curr, c_next
    return c_curr


class GegenbauerNeuralBlock(nn.Module):
    def __init__(self, degree: int = 5):
        super().__init__()
        self.degree = int(degree)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(x)
        alpha = F.softplus(self.alpha)
        return gegenbauer_polynomial(x, self.degree, alpha)


class SnakeActivation(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x).pow(2)


class ConvGNReLU(nn.Module):
    """Plain 3x3 conv + GroupNorm(16) + ReLU (the notebook's placeholder)."""

    def __init__(self, out_ch: int, kernel_size: int,
                 groups: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(
            out_ch, out_ch, kernel_size, padding=kernel_size // 2,
            bias=False)
        self.gn = nn.GroupNorm(groups, out_ch, eps=1e-3)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.gn(self.conv(x)))


class PeriodicConvBlock(nn.Module):
    """Notebook ``DynamicSnakeConv2D``: ConvGNReLU + fixed Snake."""

    def __init__(self, filters: int, kernel_size: int = 3,
                 groups: int = 16):
        super().__init__()
        self.dsconv = ConvGNReLU(filters, kernel_size, groups=groups)
        self.snake = SnakeActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.snake(self.dsconv(x))


class SENet(nn.Module):
    """Global-max channel SE with two degree-5 Gegenbauer blocks."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.gen1 = GegenbauerNeuralBlock(5)
        self.gen2 = GegenbauerNeuralBlock(5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeezed = x.amax(dim=(2, 3))
        excitation = F.relu(self.fc1(squeezed))
        excitation = self.gen1(excitation)
        excitation = torch.sigmoid(self.fc2(excitation))
        excitation = self.gen2(excitation)
        return x * excitation[:, :, None, None]


class FSFBlock(nn.Module):
    """Feature selective fusion: ``skip * gate + decoder``."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = SENet(2 * channels, reduction=reduction)
        self.compress = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, low: torch.Tensor,
                high: torch.Tensor) -> torch.Tensor:
        merged = torch.cat([high, low], dim=1)
        attention = self.se(merged)
        fused = self.compress(attention)
        gate = torch.sigmoid(fused.amax(dim=(2, 3)))[:, :, None, None]
        return low * gate + high
