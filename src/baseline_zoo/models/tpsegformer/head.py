"""TPSegformer decode heads.

The released ``TPHead`` inherits MMSeg's ``BaseDecodeHead`` but calls the
non-existent ``self.cls_seg`` and configures ``in_channels=3`` while the
LCCM emits five class-specific channels.  This port defines the missing
classifier explicitly (``conv_seg``: 5 -> ``num_classes``) and keeps the
released SEM/HLFF/LCCM stack.  ``FCNHead`` is the release's five-conv
auxiliary head with its incorrect 19-class output fixed for this binary
task.
"""

from __future__ import annotations

import torch.nn as nn

from .modules import (
    ECALayer,
    HighLowFeatureFusionModule,
    LayerCorrelationCalculation,
)


class TPHead(nn.Module):
    def __init__(self, hlff_in_channels=(384, 768), lccm_size=32,
                 lccm_mode='hilbert', num_classes=1, class_channels=5):
        super().__init__()
        high_channels, low_channels = hlff_in_channels
        self.sem1 = ECALayer(channel=low_channels, k_size=3)
        self.sem2 = ECALayer(channel=high_channels, k_size=3)
        self.hlff = HighLowFeatureFusionModule(high_channels, low_channels)
        self.lccm = nn.Sequential(
            nn.Conv2d(high_channels, class_channels, kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(class_channels),
            LayerCorrelationCalculation(lccm_size, mode=lccm_mode),
        )
        self.conv_seg = nn.Conv2d(class_channels, num_classes, 1)

    def forward(self, inputs):
        x3, x4 = inputs[-2], inputs[-1]
        fused = self.hlff(self.sem1(x4), self.sem2(x3))
        return self.conv_seg(self.lccm(fused))


class FCNHead(nn.Module):
    def __init__(self, in_channels=384, channels=5, num_convs=5,
                 num_classes=1, dropout=0.1):
        super().__init__()
        layers = []
        for index in range(num_convs):
            input_channels = in_channels if index == 0 else channels
            layers.extend([
                nn.Conv2d(input_channels, channels, 3, padding=1,
                          bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
            ])
        self.convs = nn.Sequential(*layers)
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x):
        return self.conv_seg(self.convs(x))
