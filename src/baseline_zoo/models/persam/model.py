"""PerSAM model wrapper: vendored SAM ViT-H + modified predictor."""

from __future__ import annotations

import torch.nn as nn

from .per_segment_anything import SamPredictor, sam_model_registry


class PerSAMModel(nn.Module):
    """Contain the SAM model and the modified PerSAM predictor."""

    def __init__(self, sam: nn.Module):
        super().__init__()
        self.sam = sam
        self.predictor = SamPredictor(sam)


def build_persam(sam_checkpoint: str, sam_type: str = 'vit_h') -> PerSAMModel:
    if sam_type not in sam_model_registry:
        raise ValueError(f'unknown SAM type: {sam_type}')
    sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
    return PerSAMModel(sam)
