"""Self-contained SAM2-UNet (appendix extended baseline).

Vendored from ``WZH0120/SAM2-UNet`` (Apache-2.0).  The architecture pairs a
frozen SAM2 Hiera-L encoder with per-block prompt-learn adapters and a U-Net
decoder.  See ``UPSTREAM.md`` for provenance and deviations.
"""

from .model import SAM2UNet
from .sam2.build_sam import build_sam2

__all__ = ["SAM2UNet", "build_sam2"]
