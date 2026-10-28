"""Global persistence-threshold prior for A2SAM.

Paper Eq. (4) defines a one-dimensional persistence threshold sampled from
``xi ~ N(mu, sigma^2)``.  The official release does not include the prior
network, so this is a paper-level reconstruction: a small U-Net-style
convolutional encoder conditioned on the image and the coarse likelihood map,
predicting a global scalar Gaussian.  It is not the old conditional-VAE latent
sampler from the approximation.

Training uses heteroscedastic Gaussian NLL against a Dice-optimal threshold
computed from training data only.  Sampling accepts an explicit
``torch.Generator`` so validation/prediction are deterministic per sample.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import ConvBlock

DEFAULT_PRIOR_WIDTHS = (24, 48, 96, 192, 384)


class PersistencePrior(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        widths: tuple[int, int, int, int, int] = DEFAULT_PRIOR_WIDTHS,
        hidden: int = 256,
        xi_min: float = 0.0,
        xi_max: float = 1.0,
        min_sigma: float = 0.05,
    ):
        super().__init__()
        c1, c2, c3, c4, c5 = widths
        if min_sigma <= 0:
            raise ValueError('min_sigma must be positive')
        self.xi_min = float(xi_min)
        self.xi_max = float(xi_max)
        self.min_sigma = float(min_sigma)
        self.log_var_min = 2.0 * math.log(self.min_sigma)
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.enc5 = ConvBlock(c4, c5)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c5, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )

    def forward(self, image: torch.Tensor,
                probability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or probability.ndim != 4:
            raise ValueError('image and probability must be [B,C,H,W]')
        if probability.shape[1] != 1:
            raise ValueError('probability must have one channel')
        if probability.shape[-2:] != image.shape[-2:]:
            probability = F.interpolate(
                probability, size=image.shape[-2:], mode='bilinear',
                align_corners=False)
        x = torch.cat([image, probability], dim=1)
        x = self.enc1(x)
        x = self.enc2(self.pool(x))
        x = self.enc3(self.pool(x))
        x = self.enc4(self.pool(x))
        x = self.enc5(self.pool(x))
        x = x.mean(dim=(2, 3))
        stats = self.head(x)
        raw_mu, log_var = stats[:, 0], stats[:, 1]
        # The persistence threshold is defined on probability units.  Keep
        # the predicted mean inside [xi_min, xi_max] instead of letting an
        # untrained head emit mu > 1, which would clamp every sample to the
        # upper bound and make the structural uncertainty identically zero.
        mu = self.xi_min + (self.xi_max - self.xi_min) * torch.sigmoid(raw_mu)
        return mu, log_var.clamp(min=self.log_var_min, max=10.0)

    def nll_loss(self, mu: torch.Tensor, log_var: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        log_var = log_var.clamp(min=self.log_var_min, max=10.0)
        return (0.5 * (log_var + (target - mu).pow(2)
                       / log_var.exp())).mean()

    def sample_xi(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        k: int = 5,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return ``[k, B]`` thresholds sampled from ``N(mu, sigma^2)``."""
        if k < 1:
            raise ValueError('k must be >= 1')
        std = torch.exp(0.5 * log_var.clamp(
            min=self.log_var_min, max=10.0))
        shape = (k,) + tuple(mu.shape)
        if generator is None:
            noise = torch.randn(shape, device=mu.device, dtype=mu.dtype)
        else:
            noise = torch.randn(
                shape, device=generator.device, dtype=mu.dtype,
                generator=generator).to(mu.device)
        return (mu.unsqueeze(0) + noise * std.unsqueeze(0)).clamp(
            self.xi_min, self.xi_max)
