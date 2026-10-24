"""SAM ViT-L + lightweight U-Net decoder (project extension baseline).

This is our own probe, not one of the paper's ten baselines.  The frozen
encoder comes from the self-contained ``models.sam_network`` package; the
decoder is the one used by the previous ``baselines/adapters`` version.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sam_network.build_sam import build_sam_vit_l

_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1) / 255.0
_PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1) / 255.0


def normalize(image: torch.Tensor) -> torch.Tensor:
    return (image - _PIXEL_MEAN.to(image.device)) / _PIXEL_STD.to(
        image.device)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                          align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SAMViTLUNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.ModuleList(
            [nn.Conv2d(1024, 64, 1) for _ in range(4)] +
            [nn.Conv2d(256, 64, 1)])
        self.proj_bn = nn.ModuleList(
            [nn.BatchNorm2d(64) for _ in range(5)])
        self.stem_conv = nn.ModuleList([
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
        ])
        self.stem_bn = nn.ModuleList([
            nn.BatchNorm2d(32), nn.BatchNorm2d(64),
            nn.BatchNorm2d(64), nn.BatchNorm2d(64),
        ])
        self.bottleneck = DoubleConv(64, 64)
        self.up1 = Up(64, 64, 64)
        self.up2 = Up(64, 64, 64)
        self.up3 = Up(64, 32, 32)
        self.up4 = DoubleConv(32, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def run_stem(self, image):
        x = F.relu(self.stem_bn[0](self.stem_conv[0](image)), inplace=True)
        skip_12 = x
        x = F.relu(self.stem_bn[1](self.stem_conv[1](x)), inplace=True)
        skip_14 = x
        x = F.relu(self.stem_bn[2](self.stem_conv[2](x)), inplace=True)
        skip_18 = x
        x = F.relu(self.stem_bn[3](self.stem_conv[3](x)), inplace=True)
        return skip_12, skip_14, skip_18, x

    def forward(self, image, features):
        skip_12, skip_14, skip_18, stem_116 = self.run_stem(image)
        fused = 0
        for feature, projection, norm in zip(features, self.proj,
                                             self.proj_bn):
            fused = fused + F.relu(norm(projection(feature.float())),
                                   inplace=True)
        fused = fused + stem_116
        fused = self.bottleneck(fused)
        fused = self.up1(fused, skip_18)
        fused = self.up2(fused, skip_14)
        fused = self.up3(fused, skip_12)
        fused = F.interpolate(fused, scale_factor=2, mode='bilinear',
                              align_corners=False)
        return self.head(self.up4(fused))


class SAMViTLUNet(nn.Module):
    def __init__(self, sam_checkpoint: str):
        super().__init__()
        sam = build_sam_vit_l(checkpoint=sam_checkpoint, use_adapter=False)
        self.encoder = sam.image_encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.decoder = SAMViTLUNetDecoder()

    @torch.no_grad()
    def encode_features(self, image):
        """Frozen ViT-L features: 4 intermediate maps + neck output."""
        self.encoder.eval()
        output, features = self.encoder(normalize(image),
                                        return_features=True)
        maps = [feature.permute(0, 3, 1, 2).contiguous()
                for feature in features]
        maps.append(output)
        return maps

    def forward(self, image, features=None):
        if features is None:
            features = self.encode_features(image)
        return self.decoder(image, features)
