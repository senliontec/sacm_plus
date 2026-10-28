"""Self-contained SANSA: semantically aligned SAM 2 (NeurIPS 2025).

The official method keeps SAM 2 Hiera-Large frozen, inserts AdaptFormer
bottleneck adapters into the last two hierarchical stages of the image
encoder, and reinterprets reference images plus the target as a pseudo-video
processed by SAM 2's memory pipeline.  This port follows the released
training code: each training episode is one reference plus one target with a
mask prompt; evaluation uses the protocol's three references plus the target.

The released repository's ``preprocess`` does not normalize pixels; its
datasets apply ImageNet normalization before the model.  This port does the
same.  The released ``main.py`` + ``engine.py`` actually train with one
support and focal+Dice loss, although the paper describes BCE+Dice and a
multi-target J=3 objective; the port follows the released training code and
documents the discrepancy.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

from baseline_zoo.models.sansa.model_utils import (
    BackboneOutput,
    DecoderOutput,
)
from baseline_zoo.models.sansa.sam2.modeling.sam2_utils import preprocess

CONFIG_DIR = Path(__file__).resolve().parent / 'sam2' / 'sam2_configs'


def focal_dice_loss(outputs: torch.Tensor, masks: torch.Tensor,
                    num_frames: int) -> torch.Tensor:
    """Official ``loss_masks``: sigmoid focal (0.25, 2.0) + Dice."""
    bs, total = masks.shape[:2]
    start = max(0, total - num_frames)
    src = outputs.view(bs, total, *outputs.shape[-2:])[:, start:total]
    tgt = masks[:, start:total].to(src)
    src = src.flatten(1).float()
    tgt = tgt.flatten(1).to(src.dtype)

    prob = src.sigmoid()
    ce = F.binary_cross_entropy_with_logits(src, tgt, reduction='none')
    p_t = prob * tgt + (1 - prob) * (1 - tgt)
    focal = ce * ((1 - p_t) ** 2.0)
    alpha_t = 0.25 * tgt + 0.75 * (1 - tgt)
    focal = (alpha_t * focal).mean(1).sum() / bs

    numerator = 2 * (prob * tgt).sum(1)
    denominator = prob.sum(1) + tgt.sum(1)
    dice = (1 - (numerator + 1) / (denominator + 1)).sum() / bs
    return focal + dice


class SANSA(nn.Module):
    """SANSA pseudo-video forward around a frozen SAM 2 backbone."""

    def __init__(self, sam: nn.Module):
        super().__init__()
        self.sam = sam

    def forward(self, samples: torch.Tensor,
                prompt_dict: dict) -> torch.Tensor:
        """``samples`` is [B,T,3,H,W] and prompts are support masks.

        Returns logits for all B*T frames at the input resolution.
        """
        if prompt_dict.get('prompt_type', 'mask') != 'mask':
            raise ValueError('this port uses support mask prompts only')
        samples, batch, frames, orig_size = self._preprocess_visual_features(
            samples, self.sam.image_size)
        backbone_output = self._forward_backbone(samples, orig_size)
        outputs = []
        n_shots = prompt_dict['shots']
        for b in range(batch):
            memory_bank = {}
            for idx in range(frames):
                absolute_idx = b * frames + idx
                if idx < n_shots:
                    frame_prompt = prompt_dict[b][idx]['prompt']
                    frame_prompt = self._resize_mask(
                        frame_prompt, self.sam.image_size)
                    decoder_out = self.sam._use_mask_as_output(
                        backbone_output, frame_prompt, absolute_idx)
                else:
                    decoder_out = self._decode_with_memory(
                        backbone_output, absolute_idx, idx, memory_bank)
                memory_bank[idx] = self._memory_entry(
                    decoder_out, backbone_output, absolute_idx)
                outputs.append(decoder_out.masks[0])

        masks = torch.cat(outputs)
        masks = F.interpolate(
            masks[None], size=orig_size[0], mode='bilinear',
            align_corners=False)[0]
        return masks

    def _preprocess_visual_features(self, samples, image_size):
        batch, frames, channels, height, width = samples.shape
        flat = samples.view(batch * frames, channels, height, width)
        orig_size = [tuple(item.shape[-2:]) for item in flat]
        flat = torch.stack(
            [preprocess(item, image_size) for item in flat], dim=0)
        return flat, batch, frames, orig_size

    @staticmethod
    def _resize_mask(mask: torch.Tensor, image_size: int) -> torch.Tensor:
        if mask.shape[-2:] == (image_size, image_size):
            return mask
        return F.interpolate(
            mask.float(), size=(image_size, image_size), mode='nearest')

    def _decode_no_memory(self, backbone_out: BackboneOutput, idx: int):
        current = backbone_out.get_current_feats(idx)
        high_res = backbone_out.get_high_res_features(current)
        pix_feat = current[-1:][-1] + self.sam.no_mem_embed
        pix_feat = pix_feat.permute(1, 2, 0).view(1, 256, 64, 64)
        return self.sam._forward_sam_heads(
            backbone_features=pix_feat, high_res_features=high_res)

    def _decode_with_memory(self, backbone_out: BackboneOutput, idx: int,
                            memory_idx: int, memory_bank: dict):
        current = backbone_out.get_current_feats(idx)
        current_pos = backbone_out.get_current_pos_embeds(idx)
        high_res = backbone_out.get_high_res_features(current)
        pix_feat = self.sam._prepare_memory_conditioned_features(
            frame_idx=memory_idx,
            current_vision_feats=current[-1:],
            current_vision_pos_embeds=current_pos[-1:],
            feat_sizes=backbone_out.feat_sizes[-1:],
            num_frames=memory_idx + 1,
            memory_bank=memory_bank,
        )
        return self.sam._forward_sam_heads(
            backbone_features=pix_feat,
            high_res_features=high_res,
            multimask_output=memory_idx > 0,
        )

    def _memory_entry(self, decoder_out: DecoderOutput,
                      backbone_out: BackboneOutput, idx: int) -> dict:
        current = backbone_out.get_current_feats(idx)
        mem_feats, mem_pos = self.sam._encode_new_memory(
            current_vision_feats=current,
            feat_sizes=backbone_out.feat_sizes,
            pred_masks_high_res=decoder_out.high_res_masks,
            is_mask_from_pts=False,
        )
        return {
            'maskmem_features': mem_feats,
            'maskmem_pos_enc': mem_pos,
            'pred_masks': decoder_out.low_res_masks,
            'obj_ptr': decoder_out.obj_ptr,
        }

    def _forward_backbone(self, samples, orig_size) -> BackboneOutput:
        vis = self.sam.image_encoder.trunk(samples)
        feats, pos = self.sam.image_encoder.neck(vis)
        feats, pos = feats[:-1], pos[:-1]
        feats[0] = self.sam.sam_mask_decoder.conv_s0(feats[0])
        feats[1] = self.sam.sam_mask_decoder.conv_s1(feats[1])
        backbone = {
            'vision_features': feats[-1],
            'vision_pos_enc': pos,
            'backbone_fpn': feats,
        }
        vision_feats, vision_pos, sizes = (
            self.sam._prepare_backbone_features(backbone))
        return BackboneOutput(orig_size, vision_feats, vision_pos, sizes)


def build_sansa(
    sam2_checkpoint: str | Path,
    sam2_config: str = 'sam2_hiera_l.yaml',
    adaptformer_stages: tuple[int, ...] = (2, 3),
    channel_factor: float = 0.3,
) -> SANSA:
    """Build SANSA with the official Hiera-Large + AdaptFormer settings."""
    config_path = CONFIG_DIR / Path(sam2_config).name
    if not config_path.is_file():
        raise FileNotFoundError(f'missing SANSA SAM2 config: {config_path}')
    cfg = OmegaConf.load(config_path)
    cfg.model.image_encoder.trunk.adaptformer_stages = list(
        adaptformer_stages)
    cfg.model.image_encoder.trunk.adapt_dim = float(channel_factor)
    cfg.model.pred_obj_scores = False
    cfg.model.pred_obj_scores_mlp = False
    cfg.model.fixed_no_obj_ptr = False
    OmegaConf.resolve(cfg)
    sam = instantiate(cfg.model, _recursive_=True)

    path = Path(sam2_checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f'missing SAM2 checkpoint: {path}')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    missing, unexpected = sam.load_state_dict(state, strict=False)
    bad_missing = [key for key in missing if 'adapter' not in key]
    # SANSA disables the object-score branch, so the checkpoint's unused
    # object-score parameters are expected to be absent from the model.
    allowed_unexpected = (
        'no_obj_ptr', 'obj_score_token', 'pred_obj_score_head')
    bad_unexpected = [key for key in unexpected
                      if not any(token in key
                                 for token in allowed_unexpected)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f'{path}: SANSA SAM2 weights do not match the config '
            f'({len(bad_missing)} non-adapter missing, '
            f'{len(bad_unexpected)} unexpected); first missing: '
            f'{bad_missing[:5]}; first unexpected: {bad_unexpected[:5]}')

    for name, parameter in sam.named_parameters():
        parameter.requires_grad_('adapter' in name)
    sam.eval()
    return SANSA(sam)


__all__ = ['SANSA', 'build_sansa', 'focal_dice_loss']
