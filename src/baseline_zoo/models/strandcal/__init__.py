"""StrandCal-SAM: a self-contained topology-aware curvilinear SAM.

The package owns its SAM ViT-L variant, fusion module, topology losses and
CAPE implementation.  It does not import any other baseline model or the
experimental ``methods`` tree at runtime.

StrandCal-SAM = TopoStrand-SAM's width-calibrated four-candidate training and
resolution-consensus decoding, followed by Adaptive Topology Calibration
(ATC) at prediction time.
"""

from .segment_anything import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l

__all__ = ['build_sam_vit_b', 'build_sam_vit_h', 'build_sam_vit_l']
