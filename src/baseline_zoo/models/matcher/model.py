"""Matcher wrapper: DINOv2 ViT-L + modified SAM ViT-H."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

import baseline_zoo.models.matcher.dinov2.utils.utils as dinov2_utils
from .Matcher import Matcher
from .dinov2.models import vision_transformer as vits
from .segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class MatcherModel(nn.Module):
    """Register DINOv2/SAM for device moves; the Matcher object is not a module."""

    def __init__(self, matcher: Matcher):
        super().__init__()
        self.matcher = matcher
        self.encoder = matcher.encoder
        self.sam = matcher.generator.predictor.model

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        device = None
        if args:
            device = args[0]
        elif 'device' in kwargs:
            device = kwargs['device']
        if device is not None:
            self.matcher.device = torch.device(device)
        return result


def build_matcher(
    sam_weights: str | Path,
    dinov2_weights: Optional[str | Path] = None,
    device: str | torch.device = 'cpu',
    random_init: bool = False,
) -> MatcherModel:
    """Build the official Matcher DINOv2 ViT-L + SAM ViT-H configuration."""
    dinov2 = vits.vit_large(
        img_size=518,
        patch_size=14,
        init_values=1e-5,
        ffn_layer='mlp',
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    if random_init:
        pass
    elif dinov2_weights is None or not Path(dinov2_weights).is_file():
        raise FileNotFoundError(
            f'missing DINOv2 ViT-L weights: {dinov2_weights}')
    else:
        dinov2_utils.load_pretrained_weights(
            dinov2, str(dinov2_weights), 'teacher')
    dinov2.eval()

    sam = sam_model_registry['vit_h'](checkpoint=str(sam_weights))
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=64,
        points_per_batch=64,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        stability_score_offset=1.0,
        sel_stability_score_thresh=0.0,
        sel_pred_iou_thresh=0.0,
        box_nms_thresh=1.0,
        sel_output_layer=3,
        output_layer=0,
        dense_pred=False,
        multimask_output=False,
        sel_multimask_output=False,
    )
    score_filter_cfg = {
        'emd': 0.0,
        'purity': 0.0,
        'coverage': 0.0,
        'score_filter': False,
        'score': 0.33,
        'score_norm': 0.1,
        'topk_scores_threshold': 0.7,
    }
    matcher = Matcher(
        encoder=dinov2,
        generator=generator,
        num_centers=8,
        use_box=False,
        use_points_or_centers=False,
        sample_range=(4, 6),
        max_sample_iterations=30,
        alpha=1.0,
        beta=0.0,
        exp=0.0,
        score_filter_cfg=score_filter_cfg,
        num_merging_mask=10,
        device=torch.device(device),
    )
    return MatcherModel(matcher)
