"""Self-contained SAM-Adapter (ICCV 2023) model package."""

from .model import DEFAULT_ENCODER_MODE, SAM, build_sam_adapter

__all__ = ['DEFAULT_ENCODER_MODE', 'SAM', 'build_sam_adapter']
