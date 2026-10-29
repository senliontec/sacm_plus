""" Parts of the U-Net model """
import copy
import math

from torchvision.transforms import ToPILImage, transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Conv(nn.Module):
    """(convolution => [BN] => ReLU) """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)




class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )
        self.max = nn.MaxPool2d(2)
    def forward(self, x1):
        x = self.maxpool_conv(x1)
        return self.relu(x)

class Up(nn.Module):
    """Upscaling then double conv.

    The released bilinear branch was broken: ``Upsample`` keeps all
    ``in_channels`` while the skip ``x2`` has ``in_channels // 2``, so the
    concatenated width is ``1.5 * in_channels`` but ``DoubleConv`` was
    constructed with ``in_channels``.  The paper specifies bilinear
    upsampling, so the conv input is corrected to the actual concatenated
    width.  The ConvTranspose (release) branch is unchanged.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv_in = in_channels + in_channels // 2
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            conv_in = in_channels

        self.conv = DoubleConv(conv_in, out_channels)
        # self.contra=nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)




def conv1x1(in_channels, out_channels, groups=1):
    return nn.Sequential(nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1),
        nn.BatchNorm2d(out_channels))
class OutConv(nn.Module):
    """Project the full-resolution U-Net branch to class logits.

    Upstream called an undefined ``upconv2x2`` and then applied Sigmoid.
    The paper's MRL uses softmax/RecallCE on both branch outputs, so this
    port keeps raw logits (no Sigmoid); the 1x1 projection is the minimal
    reconstruction of the broken upstream head.
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.up = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.up(x)


class RecallCrossEntropy(nn.Module):
    """Multilabel recall loss (MRL), paper Eq. 1.

    ``weight_c = FN_c / (TP_c + FN_c) = 1 - R_c`` is computed from the
    current batch and applied to the per-pixel cross-entropy of class ``c``.
    The upstream implementation hardcodes ``.cuda()`` and has fragile
    ignore-index indexing; this version is device-safe and filters ignored
    pixels explicitly. It must receive raw logits (not log-softmax).
    """

    def __init__(self, n_classes=2, ignore_index=255):
        super(RecallCrossEntropy, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if input.ndim != 4 or input.shape[1] != self.n_classes:
            raise ValueError(
                f'recall CE expects [B,{self.n_classes},H,W] logits, got '
                f'{tuple(input.shape)}')
        target = target.long()
        valid = target != self.ignore_index
        safe_target = target.clone()
        safe_target[~valid] = 0

        pred = input.argmax(dim=1)
        gt_count = torch.bincount(
            safe_target[valid], minlength=self.n_classes).to(input.dtype)
        false_negative = valid & (pred != safe_target)
        fn_count = torch.bincount(
            safe_target[false_negative],
            minlength=self.n_classes).to(input.dtype)
        # 1 - recall; absent classes have no target pixels and therefore no
        # effect on the masked mean below.
        weight = fn_count / gt_count.clamp(min=1)

        per_pixel = F.cross_entropy(
            input, target, reduction='none', ignore_index=self.ignore_index)
        loss = weight[safe_target] * per_pixel
        loss = loss[valid]
        if loss.numel() == 0:
            return input.sum() * 0.0
        return loss.mean()

