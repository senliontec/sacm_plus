"""CurvField: geometry-conditioned prompt-free curvilinear segmentation."""

from __future__ import annotations

from .adapter import StructureConditionedAdapter
from .fields import field_loss, structure_targets
from .head import FieldDecoderBias, StructureFieldHead
from .model import CurvFieldSAM

__all__ = [
    'CurvFieldSAM',
    'FieldDecoderBias',
    'StructureConditionedAdapter',
    'StructureFieldHead',
    'field_loss',
    'structure_targets',
]
