"""Self-contained S2M-Net PyTorch port (ICML 2026).

The TensorFlow/Keras release provides the architecture and Morphology-Aware
Adaptive Loss; this package translates both to PyTorch without TensorFlow.
Blocks: MRF-SE (multi-receptive-field depthwise + SE), SSTM (per-channel FFT
filter + gated dense projection) and the BFP decoder (boundary map and
region/boundary soft routing).  The five MAL components (core Dice+IoU+
boundary BCE, multi-scale boundary gradients, compactness, scale-aware
focal, Laplacian texture) keep their learnable weights clipped to [0.1, 10].

Source: https://github.com/.../S2M-Net (local tree:
``3rd/S2M-Net-Spectral-Spatial-Mixing-for-Medical-Segmentation``).
Paper: "S2M-Net: Spectral-Spatial Mixing with Morphology-Aware Adaptive
Loss for Medical Image Segmentation", ICML 2026.
"""

from .blocks import (
    BFPDecoderStage,
    MRFSEBlock,
    SpectralSelectiveTokenMixer,
)
from .loss import (
    BoundaryLoss,
    CoreLoss,
    MorphologyAwareLoss,
    ScaleAwareFocalLoss,
    StructureLoss,
    TextureLoss,
)
from .model import S2MNet

__all__ = [
    'BFPDecoderStage',
    'BoundaryLoss',
    'CoreLoss',
    'MRFSEBlock',
    'MorphologyAwareLoss',
    'S2MNet',
    'ScaleAwareFocalLoss',
    'SpectralSelectiveTokenMixer',
    'StructureLoss',
    'TextureLoss',
]
