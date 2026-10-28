"""Self-contained FSSAM: SAM 2 few-shot segmentation (ICML 2025).

This is a port of the official ``model/FSSAM5s.py`` (the variable-shot
variant) to the project's binary, support-based 3-shot protocol.  The
paper's three components are kept verbatim:

1. Pseudo Prompt Generator (PPG): DINOv2 features + support masks produce
   per-support FG and Disc prior masks, averaged over the support set.
2. Iterative Memory Refinement (IMR): the Disc memory is refined from the
   FG memory, gated by FG-memory/support-memory cosine similarities.
3. Support-Calibrated Memory Attention (SCMA): support masks calibrate the
   query-to-query-memory attention scores inside the vendored SAM 2 memory
   attention.

The released FSSAM repository calls ``self.Weighted_GAP`` without defining
it and breaks multi-shot calibration by overwriting the support-mask tensor
inside the shot loop; both are repaired in the vendored SAM 2 copy.  The
official DINOv2-B backbone is replaced by a locally available DINOv2-L in
this port when requested (the paper's Table 7 reports the L variant).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_zoo.models.fssam.sam2.build_sam import (
    build_sam2_video_predictor,
)
from baseline_zoo.models.fssam.sam2.utils.misc import (
    load_video_frames_from_data,
)
from baseline_zoo.models.matcher.dinov2.models import (
    vision_transformer as vits,
)


def Weighted_GAP(supp_feat: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
    """Mask-weighted global average pooling (official ``Weighted_GAP``)."""
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:]
    area = (F.avg_pool2d(mask, (feat_h, feat_w))
            * feat_h * feat_w + 0.0005)
    return (F.avg_pool2d(supp_feat, (feat_h, feat_w))
            * feat_h * feat_w / area)


def weighted_dice_loss(prediction: torch.Tensor, target_seg: torch.Tensor,
                       weighted_val: float = 1.0,
                       reduction: str = 'sum',
                       eps: float = 1e-8) -> torch.Tensor:
    """Official ``weighted_dice_loss`` (sigmoid + soft Dice)."""
    target_seg = (target_seg == 1).float()
    n, h, w = target_seg.shape
    prediction = prediction.reshape(-1, h, w)
    target_seg = target_seg.reshape(-1, h, w)
    prediction = torch.sigmoid(prediction).reshape(-1, h * w)
    target_seg = target_seg.reshape(-1, h * w)
    loss_part = (prediction ** 2).sum(dim=-1) + (target_seg ** 2).sum(dim=-1)
    loss = 1 - 2 * (target_seg * prediction).sum(dim=-1) / torch.clamp(
        loss_part, min=eps)
    loss = loss * weighted_val
    if reduction == 'sum':
        return loss.sum() / n
    if reduction == 'mean':
        return loss.mean()
    return loss


class FSSAM(nn.Module):
    """FSSAM forward: priors, SAM 2 memories, IMR, SCMA, mask decoder."""

    def __init__(self, sam2: nn.Module, dino: nn.Module,
                 num_layers: int = 24, num_refine: int = 3,
                 ver_refine: str = 'v1', input_size: int = 512):
        super().__init__()
        self.sam2 = sam2
        self.dino = dino
        self.num_layers = int(num_layers)
        self.num_refine = int(num_refine)
        self.ver_refine = str(ver_refine)
        self.input_size = int(input_size)
        # Official DINOv2 normalization, scaled to 0-255 inputs.
        mean = [item * 255 for item in (0.485, 0.456, 0.406)]
        std = [item * 255 for item in (0.229, 0.224, 0.225)]
        self.register_buffer('dino_mean',
                             torch.tensor(mean).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer('dino_std',
                             torch.tensor(std).view(1, 3, 1, 1),
                             persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        # The official pipeline keeps the image encoder, prompt encoder and
        # DINOv2 frozen; putting them in eval avoids updating any inference
        # behaviour when the runtime calls model.train().
        self.dino.eval()
        self.sam2.image_encoder.eval()
        self.sam2.sam_prompt_encoder.eval()
        self.sam2.obj_ptr_proj.eval()
        self.sam2.mask_downsample.eval()
        return self

    # ------------------------------------------------------------------
    # Prior generation
    # ------------------------------------------------------------------
    @staticmethod
    def _cos_sim(query_feat_high: torch.Tensor,
                 tmp_supp_feat: torch.Tensor,
                 cosine_eps: float = 1e-7) -> torch.Tensor:
        q = query_feat_high.flatten(2).transpose(-2, -1)
        s = tmp_supp_feat.flatten(2).transpose(-2, -1)
        tmp_query = q.contiguous().permute(0, 2, 1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
        tmp_supp_norm = torch.norm(s, 2, 2, True)
        return (s @ tmp_query) / (tmp_supp_norm @ tmp_query_norm
                                  + cosine_eps)

    def generate_prior_proto(self, query_feat_high, final_supp_list,
                             mask_list, fts_size, normalize=False):
        """Official Pseudo Prompt Generator."""
        bsize, _, sp_sz, _ = query_feat_high.size()[:]
        fg_list, bg_list, fg_sim_maxs = [], [], []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(
                mask_list[i], size=(resize_size, resize_size),
                mode='bilinear', align_corners=True)
            fg_supp_feat = Weighted_GAP(tmp_supp_feat, tmp_mask)
            bg_supp_feat = Weighted_GAP(tmp_supp_feat, 1 - tmp_mask)

            fg_sim = self._cos_sim(query_feat_high, fg_supp_feat, cosine_eps)
            bg_sim = self._cos_sim(query_feat_high, bg_supp_feat, cosine_eps)
            fg_sim = fg_sim.max(1)[0].view(bsize, sp_sz * sp_sz)
            bg_sim = bg_sim.max(1)[0].view(bsize, sp_sz * sp_sz)
            fg_sim_maxs.append(fg_sim.max(1)[0].unsqueeze(-1))

            fg_sim = (fg_sim - fg_sim.min(1)[0].unsqueeze(1)) / (
                fg_sim.max(1)[0].unsqueeze(1)
                - fg_sim.min(1)[0].unsqueeze(1) + cosine_eps)
            bg_sim = (bg_sim - bg_sim.min(1)[0].unsqueeze(1)) / (
                bg_sim.max(1)[0].unsqueeze(1)
                - bg_sim.min(1)[0].unsqueeze(1) + cosine_eps)

            fg_sim = fg_sim.view(bsize, 1, sp_sz, sp_sz)
            bg_sim = bg_sim.view(bsize, 1, sp_sz, sp_sz)
            fg_list.append(F.interpolate(
                fg_sim, size=fts_size, mode='bilinear',
                align_corners=True))
            bg_list.append(F.interpolate(
                bg_sim, size=fts_size, mode='bilinear',
                align_corners=True))

        fg_corr = torch.cat(fg_list, 1)
        bg_corr = torch.cat(bg_list, 1)
        corr = fg_corr - bg_corr
        corr = torch.clamp(corr, min=0.0)
        corr_max = corr.view(bsize, len(final_supp_list), -1).max(-1)[0]

        if normalize:
            # FSSAM5s normalizes per support sample.
            corr = corr.view(bsize, len(final_supp_list),
                             fts_size[0] * fts_size[1])
            corr = (corr - corr.min(-1)[0].unsqueeze(-1)) / (
                corr.max(-1)[0].unsqueeze(-1)
                - corr.min(-1)[0].unsqueeze(-1) + cosine_eps)
            corr = corr.view(bsize, len(final_supp_list), *fts_size)

        fg_sim_maxs = torch.cat(fg_sim_maxs, dim=-1)
        return fg_corr, bg_corr, corr, fg_sim_maxs, corr_max

    def generate_prior_mem(self, qry_feat, supp_fg, supp_mask, fts_size):
        """Cosine prior between a query memory and support memories."""
        bsize, _, sp_sz, _ = qry_feat.size()[:]
        cosine_eps = 1e-7
        shot = supp_fg.size(0) // bsize

        if shot == 1:
            fg_supp_feat = Weighted_GAP(supp_fg, supp_mask)
            fg_sim = self._cos_sim(qry_feat, fg_supp_feat, cosine_eps)
            fg_sim = fg_sim.max(1)[0].view(bsize, sp_sz * sp_sz)
            fg_sim = (fg_sim - fg_sim.min(1)[0].unsqueeze(1)) / (
                fg_sim.max(1)[0].unsqueeze(1)
                - fg_sim.min(1)[0].unsqueeze(1) + cosine_eps)
            fg_sim = fg_sim.view(bsize, 1, sp_sz, sp_sz)
            return F.interpolate(
                fg_sim, size=fts_size, mode='bilinear', align_corners=True)

        supp_fg = supp_fg.view(bsize, shot, -1, sp_sz, sp_sz)
        supp_mask = supp_mask.view(bsize, shot, 1, sp_sz, sp_sz)
        fg_sims = []
        for i in range(shot):
            fg_supp_feat = Weighted_GAP(supp_fg[:, i, ...],
                                        supp_mask[:, i, ...])
            fg_sim = self._cos_sim(qry_feat, fg_supp_feat, cosine_eps)
            fg_sim = fg_sim.max(1)[0].view(bsize, sp_sz * sp_sz)
            fg_sim = (fg_sim - fg_sim.min(1)[0].unsqueeze(1)) / (
                fg_sim.max(1)[0].unsqueeze(1)
                - fg_sim.min(1)[0].unsqueeze(1) + cosine_eps)
            fg_sim = fg_sim.view(bsize, 1, sp_sz, sp_sz)
            fg_sims.append(F.interpolate(
                fg_sim, size=fts_size, mode='bilinear', align_corners=True))
        return torch.cat(fg_sims, dim=1).mean(1, True)

    def mem_refine(self, sup_fg_mem_feats, qry_fg_mem_feats,
                   qry_ae_mem_feats, sup_fg, prior_fg, prior_ae, size,
                   prior_mem_qry_sup_fg=None, cos_eps=1e-7):
        """Official Iterative Memory Refinement step."""
        if prior_mem_qry_sup_fg is None:
            prior_mem_qry_sup_fg = self.generate_prior_mem(
                qry_fg_mem_feats, sup_fg_mem_feats, sup_fg, size)
        prior_mem_qry_qry_fg = self.generate_prior_mem(
            qry_fg_mem_feats, qry_ae_mem_feats, prior_ae, size)
        prior_mem_fg = prior_mem_qry_qry_fg + (prior_mem_qry_sup_fg - 1.)
        prior_mem_fg = torch.clamp(prior_mem_fg, min=0.0)

        if self.ver_refine == 'v2':
            flat = prior_mem_fg.flatten(2)
            flat = (flat - flat.min(-1)[0].unsqueeze(-1)) / (
                flat.max(-1)[0].unsqueeze(-1)
                - flat.min(-1)[0].unsqueeze(-1) + cos_eps)
            prior_mem_fg = flat.view_as(prior_mem_fg)

        qry_ae_mem_feats = (qry_fg_mem_feats * prior_mem_fg
                            + qry_ae_mem_feats * (1. - prior_mem_fg))
        prior_ae = prior_fg * prior_mem_fg + prior_ae * (1. - prior_mem_fg)
        flat = prior_ae.flatten(2)
        flat = (flat - flat.min(-1)[0].unsqueeze(-1)) / (
            flat.max(-1)[0].unsqueeze(-1)
            - flat.min(-1)[0].unsqueeze(-1) + cos_eps)
        prior_ae = flat.view_as(prior_ae)
        return qry_ae_mem_feats, prior_ae, prior_mem_qry_sup_fg

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _priors(self, query, support_images, support_masks):
        b, _, h, w = query.shape
        shot = support_images.shape[1]
        flat_supports = support_images.reshape(-1, 3, h, w)
        with torch.no_grad():
            dino_q = F.interpolate(
                query, size=(560, 560), mode='bilinear', align_corners=True)
            dino_s = F.interpolate(
                flat_supports, size=(560, 560), mode='bilinear',
                align_corners=True)
            dino_q = (dino_q - self.dino_mean) / self.dino_std
            dino_s = (dino_s - self.dino_mean) / self.dino_std
            # Only the last DINOv2 block is consumed; requesting one block
            # is numerically identical to the upstream range(0, num_layers)
            # and avoids storing all intermediate maps.
            qry_feat = self.dino.get_intermediate_layers(
                dino_q, n=1, reshape=True)[-1]
            supp_feat = self.dino.get_intermediate_layers(
                dino_s, n=1, reshape=True)[-1]

            feat_size = qry_feat.shape[-2:]
            if shot > 1:
                supp_feat = supp_feat.view(b, shot, -1, *feat_size)
                supp_masks = support_masks.view(
                    b, shot, 1, support_masks.shape[-2],
                    support_masks.shape[-1])
                supp_feat_list, supp_mask_list = [], []
                for i in range(shot):
                    supp_feat_list.append(supp_feat[:, i, ...])
                    supp_mask_list.append(F.interpolate(
                        supp_masks[:, i, ...], size=feat_size,
                        mode='bilinear', align_corners=True))
                prior_fg, _, prior_ae, _, _ = self.generate_prior_proto(
                    qry_feat, supp_feat_list, supp_mask_list, (h, w),
                    normalize=True)
            else:
                supp_mask = F.interpolate(
                    support_masks.float(), size=feat_size,
                    mode='bilinear', align_corners=True)
                prior_fg, _, prior_ae, _, _ = self.generate_prior_proto(
                    qry_feat, [supp_feat], [supp_mask], (h, w),
                    normalize=True)
        prior_fg = prior_fg.mean(1, True)
        prior_ae = prior_ae.mean(1, True)
        return prior_fg, prior_ae

    def forward(self, query, support_images, support_masks):
        """Return binary logits ``[B, H, W]`` at the input resolution."""
        b, _, h, w = query.shape
        flat_supports = support_images.reshape(-1, 3, h, w)
        masks = support_masks.float().clone()
        masks[masks == 255] = 0

        prior_fg, prior_ae = self._priors(query, support_images, masks)

        query_norm = load_video_frames_from_data(
            query, offload_video_to_cpu=False)
        support_norm = load_video_frames_from_data(
            flat_supports, offload_video_to_cpu=False)
        with torch.no_grad():
            (_, _, qry_feats, qry_poss,
             qry_sizes) = self.sam2.get_image_feature_batch(query_norm)
            (_, _, sup_feats, sup_poss,
             sup_sizes) = self.sam2.get_image_feature_batch(support_norm)

        sup_fg = masks.reshape(-1, 1, h, w)
        (sup_fg_preds, sup_fg_obj_ptrs, sup_fg_mem_feats,
         sup_fg_mem_poss) = self.sam2.add_new_mask_batch(
            sup_feats, sup_sizes, sup_fg)
        (qry_fg_preds, qry_fg_obj_ptrs, qry_fg_mem_feats,
         qry_fg_mem_poss) = self.sam2.add_new_mask_batch(
            qry_feats, qry_sizes, prior_fg)
        (qry_ae_preds, qry_ae_obj_ptrs, qry_ae_mem_feats,
         qry_ae_mem_poss) = self.sam2.add_new_mask_batch(
            qry_feats, qry_sizes, prior_ae)

        # The released SAM2 helper always casts mask-memory features to
        # bfloat16 because the official trainer runs under bf16 autocast.
        # This port runs in float32, so bring all memory tensors to the
        # parameter dtype; the cast is differentiable.
        dtype = prior_fg.dtype
        sup_fg_mem_feats = sup_fg_mem_feats.to(dtype)
        qry_fg_mem_feats = qry_fg_mem_feats.to(dtype)
        qry_ae_mem_feats = qry_ae_mem_feats.to(dtype)
        sup_fg_mem_poss = [x.to(dtype) for x in sup_fg_mem_poss]
        qry_fg_mem_poss = [x.to(dtype) for x in qry_fg_mem_poss]
        qry_ae_mem_poss = [x.to(dtype) for x in qry_ae_mem_poss]
        sup_fg_preds = sup_fg_preds.to(dtype)
        sup_fg_obj_ptrs = sup_fg_obj_ptrs.to(dtype)
        qry_ae_preds = qry_ae_preds.to(dtype)
        qry_ae_obj_ptrs = qry_ae_obj_ptrs.to(dtype)

        mem_size = qry_sizes[-1]
        sup_fg_mem = F.interpolate(
            sup_fg, size=mem_size, mode='bilinear', align_corners=True)
        prior_ae_mem = F.interpolate(
            prior_ae, size=mem_size, mode='bilinear', align_corners=True)
        prior_fg_mem = F.interpolate(
            prior_fg, size=mem_size, mode='bilinear', align_corners=True)

        prior_mem_qry_sup_fg = None
        for _ in range(self.num_refine):
            (qry_ae_mem_feats, prior_ae_mem,
             prior_mem_qry_sup_fg) = self.mem_refine(
                sup_fg_mem_feats, qry_fg_mem_feats, qry_ae_mem_feats,
                sup_fg_mem, prior_fg_mem, prior_ae_mem,
                size=mem_size,
                prior_mem_qry_sup_fg=prior_mem_qry_sup_fg)

        # SCMA receives every support mask as [B*shot, 1, h, w]; the
        # vendored attention averages the per-shot support similarities.
        support_masks_mem = F.interpolate(
            sup_fg, size=mem_size, mode='nearest')

        output, _ = self.sam2.propagate_in_video_batch_final(
            qry_feats, qry_poss, qry_sizes,
            sup_fg_mem_feats, sup_fg_mem_poss, support_masks_mem,
            sup_fg_preds, sup_fg_obj_ptrs,
            qry_ae_mem_feats, qry_ae_mem_poss,
            qry_ae_preds, qry_ae_obj_ptrs)
        return output.squeeze(1)


def _build_dinov2(size: str, weights: str | Path,
                  image_size: int = 518) -> nn.Module:
    common = dict(
        img_size=image_size, patch_size=14, init_values=1e-5,
        ffn_layer='mlp', block_chunks=0, qkv_bias=True,
        proj_bias=True, ffn_bias=True)
    if size == 'vitl':
        model = vits.vit_large(**common)
        layers = 24
    elif size == 'vitb':
        model = vits.vit_base(**common)
        layers = 12
    else:
        raise ValueError(f'unsupported DINOv2 size: {size}')
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(f'missing DINOv2 weights: {path}')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'teacher' in state:
        state = state['teacher']
    state = {k.replace('module.', '').replace('backbone.', ''): v
             for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f'{path}: DINOv2 weights do not match {size} '
            f'({len(missing)} missing, {len(unexpected)} unexpected); '
            f'first missing: {missing[:5]}; first unexpected: '
            f'{unexpected[:5]}')
    model.eval()
    return model, layers


def build_fssam(
    sam2_checkpoint: str | Path,
    sam2_config: str = 'sam2_hiera_s.yaml',
    dino_weights: str | Path = 'checkpoints/dinov2_vitl14_pretrain.pth',
    dino_size: str = 'vitl',
    num_refine: int = 3,
    ver_refine: str = 'v1',
    input_size: int = 512,
) -> FSSAM:
    """Build FSSAM around local SAM 2 and DINOv2 checkpoints."""
    sam2 = build_sam2_video_predictor(
        config_file=sam2_config,
        ckpt_path=sam2_checkpoint,
        device='cpu',
        mode='eval',
    )
    dino, layers = _build_dinov2(dino_size, dino_weights)

    # Official policy: only memory encoder, memory attention and mask
    # decoder are optimized; everything else (including DINOv2) is frozen.
    for parameter in sam2.parameters():
        parameter.requires_grad_(False)
    for module in (sam2.memory_encoder, sam2.memory_attention,
                   sam2.sam_mask_decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    for parameter in dino.parameters():
        parameter.requires_grad_(False)
    sam2.eval()
    dino.eval()

    return FSSAM(
        sam2, dino, num_layers=layers, num_refine=num_refine,
        ver_refine=ver_refine, input_size=input_size)


__all__ = ['FSSAM', 'build_fssam', 'weighted_dice_loss', 'Weighted_GAP']
