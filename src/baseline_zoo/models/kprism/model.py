"""Prompt-free semantic K-Prism.

The upstream release exposes three modes (semantic, in-context, interactive)
through one dual-prompt decoder.  The benchmark protocol is label-free at
inference, so this wrapper keeps only the semantic query-only branch
(``click_mode='0'``) and feeds a zero dense prompt map.  The backbone,
pixel fuser, MoE decoder and loss are the vendored upstream modules; only the
assembly and the binary class count are adapted here.

Upstream reference:
``3rd/K-Prism/kprism/KPrism_model.py`` (commit 78c3b1b) and
``kprism/config/model/base.yaml``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .backbone.utnet import UTNet
from .loss import PointSampleLoss
from .meta_arch.seghead import SegHead


class CfgNode(dict):
    """Nested dict with attribute access, like the upstream OmegaConf node."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _node(value: Any) -> Any:
    if isinstance(value, dict):
        return CfgNode({key: _node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_node(item) for item in value]
    return value


def default_config(
    *,
    num_classes: int = 2,
    num_queries: int = 2,
    image_size: int = 512,
    dec_layers: int = 6,
    num_experts: int = 5,
) -> CfgNode:
    """Official architecture config with a binary head.

    ``num_classes=2`` keeps the upstream query-group indexing
    (``index = label - 1``); the benchmark foreground uses label 1.  The
    second group is never queried.  Upstream uses 33 classes and 13 labels.

    Only the official 512 input is supported: UTNet's 256 branch emits
    different per-level channel widths than the released 96/192/384 config.
    """
    if image_size != 512:
        raise ValueError('K-Prism is ported at the official 512 input only')
    config = {
        'model': {
            'backbone': {
                'in_channels': 3,
                'base_channels': 48,
                'reduce_size': 8,
                'block_list': '1234',
                'num_blocks': [1, 1, 1, 1],
                'projection': 'interp',
                'num_heads': [4, 4, 4, 4],
                'attn_drop': 0.1,
                'proj_drop': 0.1,
                'bottleneck': False,
                'maxpool': False,
                'rel_pos': True,
                'out_feature_strides': {
                    'res3': 8, 'res4': 16, 'res5': 32},
                'out_feature_channels': {
                    'res3': 96, 'res4': 192, 'res5': 384},
            },
            'mask_encoder': {
                'in_channels': 4,
                'base_channels': 48,
                'out_feature_strides': {
                    'res3': 8, 'res4': 16, 'res5': 32},
                'out_feature_channels': {
                    'res3': 96, 'res4': 192, 'res5': 384},
            },
            'object_summarizer': {
                'embed_dim': 256,
                'num_summaries': 6,
                'num_ref': 1,
                'add_pe': True,
                'pixel_pe_scale': 32,
                'pixel_pe_temperature': 128,
                'feature_idx': {'res5': 0, 'res4': 1, 'res3': 2},
            },
            'task_encoder': {
                'in_channels': 256,
                'embed_dim': 256,
                'num_queries': num_queries,
                'num_heads': 8,
                'dim_feedforward': 256,
                'dec_layers': 1,
                'num_classes': num_classes,
                'num_class': num_classes,
            },
            'transformer_decoder': {
                'in_channels': 256,
                'hidden_dim': 256,
                'mask_dim': 256,
                'dec_layers': dec_layers,
                'dim_feedforward': 256,
                'nheads': 8,
                'rescale': [16, 8, 4],
                'num_experts': num_experts,
            },
            'sem_seg_head': {
                'in_features': ['res3', 'res4', 'res5'],
                'conv_dim': 256,
                'num_classes': num_classes,
                'transformer_in_feature': ['res3', 'res4', 'res5'],
                'foreground_index': 1,
            },
        },
        'setting': {
            'image_size': [image_size, image_size],
            'long_side_size': image_size,
            'size_divisibility': 32,
            # Inputs are already normalized by the data pipeline; upstream
            # keeps these buffers as a no-op, matching model/base.yaml.
            'pixel_mean': [0.0, 0.0, 0.0],
            'pixel_std': [1.0, 1.0, 1.0],
            'point_sample_method': 'largest_component',
            'dice_weight': 1.0,
            'mask_weight': 1.0,
        },
    }
    return _node(config)


class KPrismSemantic(nn.Module):
    """UTNet + SegHead semantic branch with a zero dense prompt map.

    ``forward`` returns the raw upstream output dict so the adapter can apply
    the official deep-supervised loss.  ``pred_masks`` has decoder resolution
    (128x128 for a 512x512 input) and must be upsampled by the caller.
    """

    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.cfg = cfg
        self.backbone = UTNet(cfg)
        self.sem_seg_head = SegHead(cfg, self.backbone.output_shape())
        self.criterion = PointSampleLoss()
        self.foreground_index = int(
            cfg.model.sem_seg_head.get('foreground_index', 1))
        self.input_size = int(cfg.setting.image_size[0])

    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f'K-Prism expects [B,3,H,W] RGB input, got '
                f'{tuple(images.shape)}')
        features = self.backbone(images)
        dense_prompt = images.new_zeros(
            (images.shape[0], 1, 3, images.shape[-2], images.shape[-1]))
        return self.sem_seg_head(
            features,
            point_feature=dense_prompt,
            click_mode='0',
            query_index=[self.foreground_index] * images.shape[0],
        )


def build_kprism_semantic(
    *,
    num_classes: int = 2,
    num_queries: int = 2,
    image_size: int = 512,
    dec_layers: int = 6,
    num_experts: int = 5,
) -> KPrismSemantic:
    return KPrismSemantic(default_config(
        num_classes=num_classes,
        num_queries=num_queries,
        image_size=image_size,
        dec_layers=dec_layers,
        num_experts=num_experts,
    ))
