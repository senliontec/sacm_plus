"""Self-contained CS-UNet PyTorch port (ICASSP 2026).

Curvilinear Spectral U-Net for road extraction: a 32-filter U-Net with
PeriodicConvBlock (the notebook's placeholder Conv-GroupNorm(16)-ReLU-Snake,
not a true deformable snake convolution), and GFR/FSF skip fusion using
Gegenbauer-polynomial spectral channel attention.  Dice loss, SGD 0.01 /
momentum 0.9, no augmentation, 512 inputs.

The official code is a TensorFlow notebook; this package is a PyTorch
rewrite without TensorFlow.

Source: https://github.com/Jayku88/CSUNet
Paper: "Curvilinear Spectral U-Net: A Framework for Structure-Aware Road
Extraction from Very High Resolution Imagery", ICASSP 2026.
"""

from .layers import (
    FSFBlock,
    GegenbauerNeuralBlock,
    PeriodicConvBlock,
    SENet,
    SnakeActivation,
    gegenbauer_polynomial,
)
from .loss import dice_loss
from .model import CSUNet

__all__ = [
    'CSUNet',
    'FSFBlock',
    'GegenbauerNeuralBlock',
    'PeriodicConvBlock',
    'SENet',
    'SnakeActivation',
    'dice_loss',
    'gegenbauer_polynomial',
]
