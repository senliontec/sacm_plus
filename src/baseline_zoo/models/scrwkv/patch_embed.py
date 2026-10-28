'''
Author: Chenhongyi Yang
Reference: https://github.com/OliverRensu/Shunted-Transformer

mmcv-free adaptation for the SCRWKV port.  Only the configuration used by
SFE (`padding='corner'`, fixed square inputs) is needed; `AdaptivePadding`
is replaced by an explicit bottom/right pad computed from kernel, stride and
dilation.
'''

import math
from itertools import repeat
import collections.abc

import torch
import torch.nn.functional as F
from torch import nn


def to_2tuple(value):
    if isinstance(value, collections.abc.Iterable):
        return tuple(value)
    return tuple(repeat(value, 2))


def _corner_pad_amount(size, kernel, stride, dilation):
    effective = dilation * (kernel - 1) + 1
    out = math.ceil(size / stride)
    return max((out - 1) * stride + effective - size, 0)


class ConvPatchEmbed(nn.Module):
    """Image to patch embedding with a convolutional stem.

    The mmcv original used ``AdaptivePadding(padding='corner')``; here the
    bottom/right padding is computed directly.  For the SCRWKV 512 input the
    padding is zero at every stage.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 num_convs=0,
                 conv_type='Conv2d',
                 patch_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super().__init__()
        assert patch_size % 2 == 0
        self.embed_dims = embed_dims
        stride = patch_size // 2 if stride is None else stride // 2

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.GroupNorm(num_channels=64, num_groups=4),
            nn.ReLU(True))

        if num_convs > 0:
            convs = []
            for _ in range(num_convs):
                convs.extend([
                    nn.Conv2d(64, 64, 3, 1, padding=1, bias=False),
                    nn.GroupNorm(num_channels=64, num_groups=4),
                    nn.ReLU(True)])
            self.convs = nn.Sequential(*convs)
        else:
            self.convs = None

        kernel_size = to_2tuple(patch_size // 2)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        self.adaptive_padding = padding == 'corner'
        self.padding_arg = padding
        padding = to_2tuple(0 if self.adaptive_padding else padding)

        self.projection = nn.Conv2d(
            in_channels=64, out_channels=embed_dims,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=bias)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        if norm_cfg is not None:
            if norm_cfg.get('type', 'LN') != 'LN':
                raise ValueError(f'unsupported norm type: {norm_cfg}')
            self.norm = nn.LayerNorm(
                embed_dims, eps=norm_cfg.get('eps', 1e-6))
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            self.init_input_size = input_size
            input_h, input_w = input_size[0] // 2, input_size[1] // 2
            if self.adaptive_padding:
                pl = _corner_pad_amount(
                    input_w, kernel_size[0], stride[0], dilation[0])
                pt = _corner_pad_amount(
                    input_h, kernel_size[1], stride[1], dilation[1])
                input_h, input_w = input_h + pt, input_w + pl
            h_out = (
                input_h + 2 * padding[0] - dilation[0]
                * (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (
                input_w + 2 * padding[1] - dilation[1]
                * (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        x = self.stem(x)
        if self.convs is not None:
            x = self.convs(x)
        if self.adaptive_padding:
            height, width = x.shape[-2:]
            pad_h = _corner_pad_amount(
                height, self.kernel_size[0], self.stride[0],
                self.dilation[0])
            pad_w = _corner_pad_amount(
                width, self.kernel_size[1], self.stride[1],
                self.dilation[1])
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size
