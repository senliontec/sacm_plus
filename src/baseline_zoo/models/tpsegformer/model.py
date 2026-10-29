"""TPSegformer: Swin-T backbone + TPDecoder + auxiliary FCN head.

The release is an MMSegmentation overlay.  This port uses timm's Swin-T
(same depths/heads/window/patch and 0.3 drop path) and the vendored TPHead,
so neither mmcv nor mmseg needs to be installed.  A local official Swin-T
checkpoint is auto-detected and MMSeg keys are mapped to timm; training
falls back to random initialization only when no checkpoint is present.
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import FCNHead, TPHead


def _strip_backbone_prefix(state: dict) -> dict:
    cleaned = {}
    for key, value in state.items():
        for prefix in ('module.', 'backbone.', 'model.backbone.'):
            if key.startswith(prefix):
                key = key[len(prefix):]
        # MMSeg and timm differ in the patch-embedding projection name and
        # (for the OpenMMLab Swin checkpoint) in the stage/FFN naming.
        key = key.replace('patch_embed.projection.', 'patch_embed.proj.')
        parts = key.split('.')
        if parts[0] == 'stages' and len(parts) >= 3:
            stage = int(parts[1])
            rest = '.'.join(parts[2:])
            if rest.startswith('blocks.'):
                key = f'layers_{stage}.{rest}'
            elif rest.startswith('downsample.'):
                # MMSeg's stage i downsample feeds timm's layers_{i+1}.
                key = f'layers_{stage + 1}.{rest}'
            key = key.replace('.attn.w_msa.', '.attn.')
            key = key.replace('.ffn.layers.0.0.', '.mlp.fc1.')
            key = key.replace('.ffn.layers.1.', '.mlp.fc2.')
        cleaned[key] = value
    return cleaned


class TPSegformerModel(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 512,
        num_classes: int = 1,
        lccm_mode: str = 'hilbert',
        drop_path_rate: float = 0.3,
        backbone: str = 'swin_tiny_patch4_window7_224',
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.backbone = timm.create_model(
            backbone,
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=image_size,
            drop_path_rate=drop_path_rate,
        )
        channels = list(self.backbone.feature_info.channels())
        if len(channels) < 4:
            raise ValueError(f'backbone must expose four stages: {channels}')
        self.in_channels = channels
        self.head = TPHead(
            hlff_in_channels=(channels[2], channels[3]),
            lccm_size=image_size // 16,
            lccm_mode=lccm_mode,
            num_classes=num_classes,
        )
        self.aux_head = FCNHead(
            in_channels=channels[2],
            channels=5,
            num_convs=5,
            num_classes=num_classes,
            dropout=0.1,
        )

    def load_backbone_checkpoint(self, path: str | Path) -> dict[str, int]:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'missing Swin-T checkpoint: {path}')
        payload = torch.load(str(path), map_location='cpu',
                             weights_only=False)
        state = payload.get('state_dict', payload)
        if not isinstance(state, dict):
            raise RuntimeError(f'{path}: unsupported checkpoint format')
        state = _strip_backbone_prefix(state)
        result = self.backbone.load_state_dict(state, strict=False)
        if result.missing_keys:
            raise RuntimeError(
                f'{path}: Swin-T checkpoint did not provide '
                f'{len(result.missing_keys)} backbone tensors; first: '
                f'{result.missing_keys[:5]}')
        return {
            'missing': 0,
            'unexpected': len(result.unexpected_keys),
        }

    def forward(self, image: torch.Tensor):
        features = self.backbone(image)
        # timm's Swin returns NHWC feature maps; the decoder expects NCHW.
        features = [
            feature.permute(0, 3, 1, 2).contiguous()
            if feature.shape[-1] == channels
            and feature.shape[1] != channels else feature
            for feature, channels in zip(features, self.in_channels)
        ]
        main = self.head(features)
        aux = self.aux_head(features[2])
        size = image.shape[-2:]
        main = F.interpolate(
            main, size=size, mode='bilinear', align_corners=False)
        aux = F.interpolate(
            aux, size=size, mode='bilinear', align_corners=False)
        return main, aux


def build_tpsegformer(
    *,
    image_size: int = 512,
    num_classes: int = 1,
    lccm_mode: str = 'hilbert',
    drop_path_rate: float = 0.3,
    backbone: str = 'swin_tiny_patch4_window7_224',
    backbone_ckpt: str | Path | None = None,
) -> TPSegformerModel:
    model = TPSegformerModel(
        image_size=image_size,
        num_classes=num_classes,
        lccm_mode=lccm_mode,
        drop_path_rate=drop_path_rate,
        backbone=backbone,
    )
    if backbone_ckpt is not None:
        counts = model.load_backbone_checkpoint(backbone_ckpt)
        print(f'loaded Swin-T backbone: {counts}')
    return model
