"""TopoStrand-SAM: structure-aware copy-paste curvilinear SAM.

This package is a physically independent curvilinear SAM ViT-L
architecture.  It owns its ``segment_anything`` code, fusion module and
topology losses, so no baseline model is imported at runtime.  The method
change is structure-aware copy-paste plus optional multi-view TTA in
``baseline_zoo.adapters.curvimix``; architecture, losses and fixed candidate-3
deployment are unchanged.
"""

from .segment_anything import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l

__all__ = ['build_sam_vit_b', 'build_sam_vit_h', 'build_sam_vit_l']
