"""LSA-SAM: independent SAM-only package for learned structural admission.

The package owns its SAM ViT-L code and the base scaffold physically.  It
does not import another model package.  The first candidate mechanism is a
learned admit/prune/bridge policy on top of a frozen SAM-only base.
"""

from .segment_anything import (
    build_sam,
    build_sam_vit_b,
    build_sam_vit_h,
    build_sam_vit_l,
    sam_model_registry,
)

__all__ = [
    'build_sam',
    'build_sam_vit_b',
    'build_sam_vit_h',
    'build_sam_vit_l',
    'sam_model_registry',
]
