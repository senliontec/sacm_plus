"""WPRF-SAM: self-contained SAM baseline with reachability supervision.

The package physically owns its SAM ViT-L code, the GFA/BCTD scaffold
copy, the WPRF affinity head and Max-Min reachability losses.  It imports
only the shared ``baseline_zoo.data`` / ``baseline_zoo.runtime`` harness;
it does not import the experimental ``methods`` tree or any other model
package.

WPRF-SAM trains the copied scaffold with edge/reachability losses on a
GT-skeleton support graph; inference emits only the union mask, so the
affinity head adds no test-time cost.
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
