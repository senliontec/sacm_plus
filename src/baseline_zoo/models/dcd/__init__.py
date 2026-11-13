"""DCD: Deterministic Candidate Decoding on the SACM decoder.

``segment_anything/`` is derived from the SACM port
(upstream ``kylechuuuuu/SACM``).  DCD trains all four Stage-2 candidates
with dense and geometry-consistent losses, then deploys one fixed candidate
at inference.  The official SACM port remains the independent baseline.
"""

from .segment_anything import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l

__all__ = ['build_sam_vit_b', 'build_sam_vit_h', 'build_sam_vit_l']
