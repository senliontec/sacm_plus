"""CS-UNet: Curvilinear Spectral U-Net (ICASSP 2026), PyTorch rewrite.

The official code is a TensorFlow notebook.  This port reproduces its U-Net:
32-filter base, encoder 32/64/128/256 with plain Conv+ReLU then
PeriodicConvBlock (the notebook's placeholder "dynamic snake" =
Conv-GroupNorm(16)-ReLU-Snake), a 512x2 bottleneck, decoder blocks with
ConvTranspose -> FSF/GFR skip fusion -> Conv+ReLU -> PeriodicConvBlock, and
a 1x1 sigmoid output.  The FSF/GFR module uses global-max channel SE,
reduction 8, degree-5 Gegenbauer activations with learnable softplus alpha,
a 1x1 half-channel compression and a global-max sigmoid gate on the skip.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import FSFBlock, PeriodicConvBlock


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, filters: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, filters, 3, padding=1)
        self.relu = nn.ReLU()
        self.periodic = PeriodicConvBlock(filters)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        x = self.periodic(self.relu(self.conv(x)))
        return x, self.pool(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, filters: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, filters, 2, stride=2)
        self.fsf = FSFBlock(filters)
        self.conv = nn.Conv2d(filters, filters, 3, padding=1)
        self.relu = nn.ReLU()
        self.periodic = PeriodicConvBlock(filters)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.fsf(skip, x)
        return self.periodic(self.relu(self.conv(x)))


class CSUNet(nn.Module):
    def __init__(self, input_size: int = 512, filters: int = 32,
                 num_classes: int = 1):
        super().__init__()
        if num_classes != 1:
            raise ValueError('CS-UNet port supports the binary head only')
        self.image_size = int(input_size)
        self.filters = int(filters)
        f = filters
        self.enc1 = _EncoderBlock(3, f)
        self.enc2 = _EncoderBlock(f, f * 2)
        self.enc3 = _EncoderBlock(f * 2, f * 4)
        self.enc4 = _EncoderBlock(f * 4, f * 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f * 8, f * 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(f * 16, f * 16, 3, padding=1),
            nn.ReLU(),
        )
        self.dec1 = _DecoderBlock(f * 16, f * 8)
        self.dec2 = _DecoderBlock(f * 8, f * 4)
        self.dec3 = _DecoderBlock(f * 4, f * 2)
        self.dec4 = _DecoderBlock(f * 2, f)
        self.output = nn.Conv2d(f, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        x = self.bottleneck(p4)
        x = self.dec1(x, s4)
        x = self.dec2(x, s3)
        x = self.dec3(x, s2)
        x = self.dec4(x, s1)
        return torch.sigmoid(self.output(x))
