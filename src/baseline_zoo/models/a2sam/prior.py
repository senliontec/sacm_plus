"""Probabilistic U-Net prior used by A2SAM's structural prompt generator.

The A2SAM paper trains a lightweight probabilistic U-Net (Kohl et al., 2018)
on the target masks, then samples it to obtain a coarse likelihood map and a
structural uncertainty map.  The released repository only reads precomputed
``logit``/``uncertainty`` ``.npy`` files and does not include that network;
this module supplies the missing prior.

The implementation is a compact conditional VAE-style U-Net: a shared
encoder, a prior head (image only) and a posterior head (image + GT) that
predict a Gaussian latent, and a FiLM-modulated decoder.  Training uses the
posterior with a KL term; inference samples the prior ``K`` times.  This is a
paper-level reimplementation, not a copy of an unavailable official network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels: int) -> nn.Module:
    groups = 8 if channels % 8 == 0 else 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int,
                 out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                          align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ProbabilisticUNet(nn.Module):
    """Conditional VAE U-Net for coarse likelihood and uncertainty."""

    def __init__(self, in_ch: int = 3, base: int = 32,
                 latent_dim: int = 16):
        super().__init__()
        self.latent_dim = int(latent_dim)
        c1, c2, c3, c4, cb = (base, base * 2, base * 4, base * 8,
                              base * 8)
        self.enc1 = ConvBlock(in_ch, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.bottleneck = ConvBlock(c4, cb)
        self.pool = nn.MaxPool2d(2)

        self.prior_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cb, cb // 4),
            nn.ReLU(inplace=True),
            nn.Linear(cb // 4, 2 * self.latent_dim),
        )
        self.posterior_conv = nn.Sequential(
            nn.Conv2d(cb + 1, cb, 3, padding=1, bias=False),
            _norm(cb),
            nn.ReLU(inplace=True),
        )
        self.posterior_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cb, cb // 4),
            nn.ReLU(inplace=True),
            nn.Linear(cb // 4, 2 * self.latent_dim),
        )

        self.film_b = nn.Linear(self.latent_dim, 2 * cb)
        self.film4 = nn.Linear(self.latent_dim, 2 * c4)
        self.film3 = nn.Linear(self.latent_dim, 2 * c3)
        self.film2 = nn.Linear(self.latent_dim, 2 * c2)
        self.film1 = nn.Linear(self.latent_dim, 2 * c1)

        self.dec4 = UpBlock(cb, c4, c3)
        self.dec3 = UpBlock(c3, c3, c2)
        self.dec2 = UpBlock(c2, c2, c1)
        self.dec1 = UpBlock(c1, c1, base)
        self.out = nn.Conv2d(base, 1, 1)

    @staticmethod
    def _film(feature: torch.Tensor, z: torch.Tensor,
              layer: nn.Linear) -> torch.Tensor:
        scale_shift = layer(z)
        scale, shift = scale_shift.chunk(2, dim=1)
        return feature * (1 + scale[..., None, None]) + shift[..., None, None]

    def _encode(self, image: torch.Tensor, gt: torch.Tensor | None):
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        if gt is None:
            stats = self.prior_head(bottleneck)
        else:
            down_gt = F.interpolate(
                gt, size=bottleneck.shape[-2:], mode='nearest')
            stats = self.posterior_head(
                self.posterior_conv(torch.cat([bottleneck, down_gt], dim=1)))
        mu, logvar = stats.chunk(2, dim=1)
        return e1, e2, e3, e4, bottleneck, mu, logvar

    def _decode(self, features, z: torch.Tensor) -> torch.Tensor:
        e1, e2, e3, e4, bottleneck = features
        b = self._film(bottleneck, z, self.film_b)
        d4 = self.dec4(b, self._film(e4, z, self.film4))
        d3 = self.dec3(d4, self._film(e3, z, self.film3))
        d2 = self.dec2(d3, self._film(e2, z, self.film2))
        d1 = self.dec1(d2, self._film(e1, z, self.film1))
        return self.out(d1)

    @staticmethod
    def reparameterize(mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar).clamp(min=1e-3, max=10.0)
        return mu + std * torch.randn_like(mu)

    @staticmethod
    def kl_divergence(mu: torch.Tensor,
                      logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).mean()

    def forward(self, image: torch.Tensor, gt: torch.Tensor | None = None,
                k: int = 1, use_prior: bool = False):
        """Return logits (or K sampled logits), mu and logvar.

        ``gt`` selects the posterior head for training; inference passes
        ``use_prior=True`` (or omits ``gt``).  With ``k > 1`` the output is
        ``[K, B, 1, H, W]`` sampled independently from the prior/posterior.
        """
        if k < 1:
            raise ValueError(f'k must be >= 1, got {k}')
        posteriors = gt is not None and not use_prior
        features = self._encode(image, gt if posteriors else None)
        e1, e2, e3, e4, bottleneck, mu, logvar = features
        base_features = (e1, e2, e3, e4, bottleneck)
        if k == 1:
            z = self.reparameterize(mu, logvar)
            return self._decode(base_features, z), mu, logvar
        samples = []
        for _ in range(k):
            z = self.reparameterize(mu, logvar)
            samples.append(self._decode(base_features, z))
        return torch.stack(samples, dim=0), mu, logvar

    @torch.no_grad()
    def sample_prior(self, image: torch.Tensor, k: int = 5) -> torch.Tensor:
        """Sample ``k`` prior segmentations for one batch."""
        samples, _, _ = self.forward(image, gt=None, k=k, use_prior=True)
        return samples
