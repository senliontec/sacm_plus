"""Self-contained TPSegformer port (CVPR 2026).

Swin-T backbone with the release's Topology-Preserving Decoder: SEM
(ECA + spatial branch), HLFF (upsample X4, concat with X3, depthwise
separable 1152 -> 384), and LCCM (5 class-specific layers, 1x1 QKV, Hilbert
flattening, Q^T K softmax, V weighting, inverse Hilbert, sigmoid gate).  A
five-conv FCN auxiliary head is supervised by Dice 0.5 + Sobel edge-aware
1.0; the main head uses binary cross-entropy.

The released code is an MMSegmentation overlay with an undefined
``self.cls_seg``, an incorrect auxiliary class count and a missing
``hilbertcurve`` dependency.  This port vendors the decoder into pure
PyTorch, fixes those defects and uses timm Swin-T, so only the project's
existing torch/timm environment is required.

Source: https://github.com/mumu-k/TPSegformer
Paper: "Hilbert Curve-Based Attention Enabling Topology-Preserving Image
Tensor Representation for Semantic Segmentation Network", CVPR 2026.
"""

from .head import FCNHead, TPHead
from .loss import edge_aware_loss, joint_aux_loss, soft_dice_loss
from .model import TPSegformerModel, build_tpsegformer

__all__ = [
    'FCNHead',
    'TPHead',
    'TPSegformerModel',
    'build_tpsegformer',
    'edge_aware_loss',
    'joint_aux_loss',
    'soft_dice_loss',
]
