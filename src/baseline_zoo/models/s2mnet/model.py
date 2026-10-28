"""S2M-Net: Spectral-Spatial Mixing Network, PyTorch rewrite.

Translated from the released TensorFlow/Keras ``s2mnet/models/s2mnet.py``:
a 3x3 stem, five strided encoder stages (MRF-SE + SSTM), four BFP decoder
stages over the reversed skips, a 2x nearest upsampling head and a sigmoid
binary output.  The MAL criterion is attached as a submodule so its five
learnable scalar weights are optimized and checkpointed with the network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import BFPDecoderStage, MRFSEBlock, SpectralSelectiveTokenMixer
from .loss import MorphologyAwareLoss


def _conv_bn_elu(in_channels: int, out_channels: int,
                 stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                  padding=1),
        nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        nn.ELU(),
    )


class S2MNet(nn.Module):
    def __init__(
        self,
        input_size: int = 352,
        filters=(24, 32, 64, 80, 128),
        use_mrfse: bool = True,
        mrfse_kernels=(3, 5, 7),
        se_reduction: int = 16,
        expand_ratio: int = 6,
        use_sstm: bool = True,
        sstm_k: int = 32,
        sstm_ssm_dim: int = 16,
        sstm_stages=(True, True, True, True, True),
        sstm_use_spectral=(True, True, True, True, True),
        sstm_use_ssm=(False, False, True, True, True),
        sstm_dropout: float = 0.1,
        use_bfp: bool = True,
        bfp_routing: str = 'soft',
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(filters) != 5:
            raise ValueError('S2M-Net filters must have five entries')
        self.image_size = int(input_size)
        self.filters = tuple(filters)
        self.use_bfp = bool(use_bfp)
        self.bfp_routing = bfp_routing

        self.stem = _conv_bn_elu(3, 16)
        self.encoder = nn.ModuleList()
        self.mrfse = nn.ModuleList()
        self.sstm = nn.ModuleList()
        in_channels = 16
        for index, channels in enumerate(filters):
            self.encoder.append(
                _conv_bn_elu(in_channels, channels, stride=2))
            if use_mrfse:
                self.mrfse.append(MRFSEBlock(
                    filters=channels,
                    kernels=mrfse_kernels,
                    se_reduction=se_reduction,
                    expand_ratio=expand_ratio,
                    dropout=dropout,
                ))
            else:
                self.mrfse.append(nn.Identity())
            spatial = max(input_size // (2 ** (index + 1)), 1)
            if use_sstm and sstm_stages[index]:
                self.sstm.append(SpectralSelectiveTokenMixer(
                    channels=channels,
                    num_frequencies=sstm_k,
                    spatial_size=spatial,
                    ssm_state_dim=sstm_ssm_dim,
                    use_spectral=sstm_use_spectral[index],
                    use_ssm=sstm_use_ssm[index],
                    dropout=sstm_dropout,
                ))
            else:
                self.sstm.append(nn.Identity())
            in_channels = channels

        reversed_skips = list(reversed(filters))
        decoder_filters = reversed_skips[1:]
        skip_channels = reversed_skips[1:]
        decoder_in = reversed_skips[0]
        self.bfp = nn.ModuleList()
        self.decoder_fallback = nn.ModuleList()
        for skip_channel, out_channels in zip(
                skip_channels, decoder_filters):
            in_ch = decoder_in + skip_channel
            if use_bfp:
                self.bfp.append(BFPDecoderStage(
                    in_channels=in_ch, filters=out_channels,
                    routing=bfp_routing))
            else:
                self.decoder_fallback.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                ))
            decoder_in = out_channels

        self.head_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.head_conv1 = nn.Conv2d(decoder_in, 32, 3, padding=1)
        self.head_conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.output = nn.Conv2d(16, 1, 1)
        self.criterion = MorphologyAwareLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips = []
        for down, mrfse, sstm in zip(self.encoder, self.mrfse, self.sstm):
            x = down(x)
            x = mrfse(x)
            x = sstm(x)
            skips.append(x)
        skips = list(reversed(skips))
        decoder = skips[0]
        modules = self.bfp if self.use_bfp else self.decoder_fallback
        for skip, module in zip(skips[1:], modules):
            if self.use_bfp:
                decoder = module(decoder, skip)[0]
            else:
                upsampled = F.interpolate(
                    decoder, size=skip.shape[-2:], mode='nearest')
                decoder = module(torch.cat([upsampled, skip], dim=1))
        decoder = self.head_up(decoder)
        decoder = F.relu(self.head_conv1(decoder))
        decoder = F.relu(self.head_conv2(decoder))
        return torch.sigmoid(self.output(decoder))
