"""TopoWeave: single-stage topology-aware curvilinear segmentation.

This is a project extension derived from MaskVSC (TMI 2025), not an official
MaskVSC port and not an official baseline.  The package is self-contained:
GroupNorm residual U-Net, gated dual-input graph refiner, width-aware
TopoMask and a bounded soft-clDice loss.
"""

from .augment import augment_pair
from .losses import TopoWeaveLoss, skeleton_weighted_bce
from .model import GraphBlock, TopoWeaveNet
from .topomask import topo_ratio, width_aware_topomask

__all__ = [
    'GraphBlock',
    'TopoWeaveLoss',
    'TopoWeaveNet',
    'augment_pair',
    'skeleton_weighted_bce',
    'topo_ratio',
    'width_aware_topomask',
]
