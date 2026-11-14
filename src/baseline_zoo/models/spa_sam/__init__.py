"""SPA-SAM: Structure-Preserving Augmentation SAM.

This package is a physically independent copy of the DCD SAM ViT-L
architecture.  It owns its ``segment_anything`` code, fusion module and
topology losses, so any future change to DCD cannot affect SPA-SAM.  The
method change is the structure-preserving augmentation policy in
``baseline_zoo.adapters.spa_sam``; architecture, losses and fixed candidate-3
deployment are unchanged.
"""

from .segment_anything import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l

__all__ = ['build_sam_vit_b', 'build_sam_vit_h', 'build_sam_vit_l']
