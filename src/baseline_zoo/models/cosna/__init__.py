"""CoSNA: a self-contained consensus-guided snake-reconstruction SAM.

The package owns its SAM ViT-L variant, four-candidate fusion, topology and
boundary losses, decoding, TCSU-lite residual, and adaptive-skeleton
calibration.  It does not import any other baseline model or the
experimental ``methods`` tree at runtime.

CoSNA combines a gated candidate-consensus mask with a TCSU-h3 mask through
a constrained near-union and a resolution-adaptive added-component filter.
"""

from .segment_anything import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l

__all__ = ['build_sam_vit_b', 'build_sam_vit_h', 'build_sam_vit_l']
