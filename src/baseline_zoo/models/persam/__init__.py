"""Self-contained Personalize-SAM (PerSAM) model package."""

from .algorithm import (
    majority_vote,
    point_selection,
    predict_with_embedding,
    prepare_reference_embedding,
)
from .model import PerSAMModel, build_persam

__all__ = [
    'PerSAMModel',
    'build_persam',
    'majority_vote',
    'point_selection',
    'predict_with_embedding',
    'prepare_reference_embedding',
]
