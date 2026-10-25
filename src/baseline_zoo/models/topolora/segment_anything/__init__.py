"""Minimal vendored Meta SAM package used by TopoLoRA-SAM.

Only the builder and modeling modules are imported; the predictor, automatic
mask generator and ONNX helpers are intentionally omitted.
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
