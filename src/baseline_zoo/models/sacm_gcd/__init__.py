"""SACM-GCD: structure-task composition on the SACM decoder.

``segment_anything/`` is derived from
``baseline_zoo.models.sacm_official.segment_anything`` (upstream
``kylechuuuuu/SACM``).  It provides the SACM-GCD typed Stage-2 extension;
the faithful port ``sacm_official`` remains the main-table baseline.
"""

from .segment_anything import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l

__all__ = ['build_sam_vit_b', 'build_sam_vit_h', 'build_sam_vit_l']
