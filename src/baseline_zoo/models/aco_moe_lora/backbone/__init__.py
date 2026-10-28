"""Vendored SAM2 Hiera backbone used by ACO-MoE-LoRA.

Only the Hiera trunk is vendored: the official ACO model builds the full
SAM2 model and then deletes the prompt/mask decoder, memory modules and FPN
neck, keeping ``image_encoder.trunk``.  The vendored files are from
``3rd/ACO-MoE-LoRA/sam2`` (Meta SAM2, Apache-2.0); see ``UPSTREAM.md``.
"""

from .hiera import Hiera

__all__ = ['Hiera']
