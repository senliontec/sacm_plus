"""BCTD-SAM: a self-contained boundary-calibrated curvilinear SAM.

The package owns its SAM ViT-L variant, fusion module, topology losses,
boundary loss and decoding implementation.  It does not import any other
baseline model or the experimental ``methods`` tree at runtime.

BCTD-SAM = TopoStrand-SAM's width-calibrated four-candidate training plus a
distance-transform boundary loss on the Stage-2 boundary head, followed by
resolution-consensus decoding and Adaptive Skeleton Calibration (AS) at
prediction time.
"""

from .segment_anything import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l

__all__ = ['build_sam_vit_b', 'build_sam_vit_h', 'build_sam_vit_l']
