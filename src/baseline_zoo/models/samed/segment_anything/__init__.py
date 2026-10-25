"""Minimal vendored SAMed SAM package.

Only the builders and modeling modules are exported; the predictor and
automatic mask generator (which pull in cv2/pycocotools) are omitted.
"""

from .build_sam import (
    build_sam,
    build_sam_vit_b,
    build_sam_vit_h,
    build_sam_vit_l,
    sam_model_registry,
)
from .modeling import Sam

__all__ = [
    'Sam',
    'build_sam',
    'build_sam_vit_b',
    'build_sam_vit_h',
    'build_sam_vit_l',
    'sam_model_registry',
]
