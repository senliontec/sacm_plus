"""CurviMix-SAM: structure-aware copy-paste augmentation for curvilinear SAM.

This package is intentionally outside ``baseline_zoo`` model content.  It
reuses the frozen DCD/SPA-SAM scaffold through the public adapter API and
adds only a new training-time augmentation: pasting intact curvilinear
structures from other training images into the current image.  No baseline
model file is modified.
"""

from .adapter import CurviMixAdapter

__all__ = ['CurviMixAdapter']
