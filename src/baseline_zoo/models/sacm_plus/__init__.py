"""SACM+ : the official SACM port with documented correctness fixes.

``segment_anything/`` is derived from
``baseline_zoo.models.sacm_official.segment_anything`` (upstream
``kylechuuuuu/SACM``, snapshot ``9865049``) and applies the fixes listed in
``UPSTREAM.md``.  It is a separate analysis baseline; the faithful port
``sacm_official`` remains the main-table SACM entry.
"""

from .segment_anything import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l

__all__ = ['build_sam_vit_b', 'build_sam_vit_h', 'build_sam_vit_l']
