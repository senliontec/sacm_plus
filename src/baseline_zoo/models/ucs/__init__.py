"""Self-contained UCS (Universal Curvilinear Structure) appendix baseline.

Vendored from ``kylechuuuuu/UCS`` (arXiv:2504.04034).  The released code is an
oracle-prompted SAM variant: it derives boxes/points from the GT mask and
refines them with GT-guided error clicks.  See ``UPSTREAM.md``.
"""

from .segment_anything.build_sam import build_sam_vit_l

__all__ = ["build_sam_vit_l"]
