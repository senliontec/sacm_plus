"""Self-contained SCRWKV model package (ICML 2026)."""

from __future__ import annotations

from .CSHF import CSHF
from .decoder import Decoder
from .SCIU import Block
from .SFE import SFE


def build_scrwkv() -> Decoder:
    """Build the official SCRWKV 'Crack' 1.22M-parameter segmenter."""
    backbone = SFE(
        arch='Crack',
        out_indices=(0, 1, 2, 3, 4),
        drop_path_rate=0.2,
        final_norm=True,
        convert_syncbn=True,
    )
    return Decoder(backbone, args=None)


__all__ = ['Block', 'CSHF', 'Decoder', 'SFE', 'build_scrwkv']
