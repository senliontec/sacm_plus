"""Self-contained RSRM: training-free cross-domain FSS (ECCV 2026).

Official reference: ``Sparkling-Water/RSRM``.  This port keeps the three
training-free modules on a frozen DINOv3 ViT-B/16:

- SAFR: per-image variance-proportion layer selection plus Fisher-ratio
  weighting, re-fusing raw attention features;
- ASE: robust support-derived thresholds define query FG/BG regions, then
  masked cross-attention aggregates query information into the support;
- HPM: global, regional (k-means) and pixel-level prototypes are matched
  and fused by a softmax confidence.

The official code uses ``fast_pytorch_kmeans``; this port uses scikit-learn
k-means++ on CPU for a dependency-free, deterministic equivalent.  The
timm ViT-S compatibility wrapper reproduces Meta DINOv3's ``F_sum`` and
``F_attn`` (``gamma_1 * attn(norm1(x))``) extraction.  The model has no
trainable parameters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from baseline_zoo.models.rsrm.dinov3.models.vision_transformer import (
    DinoVisionTransformer,
)


class _TimmDINOv3(torch.nn.Module):
    """timm DINOv3 wrapper exposing RSRM's intermediate-layer API.

    timm's ViT returns block outputs but not the post-attention output
    (``F_attn`` in RSRM, i.e. ``gamma_1 * attn(norm1(x))``).  A forward hook
    on every ``block.attn`` captures that tensor; both tensors are
    final-normed, prefix-stripped and reshaped to match the official
    ``get_intermediate_layers(..., reshape=True)``.
    """

    def __init__(self, model, depth: int):
        super().__init__()
        self.model = model
        self.depth = int(depth)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def get_intermediate_layers(self, x, n=1, reshape=False,
                                norm=True, **_):
        if isinstance(n, int):
            indices = list(range(self.depth - n, self.depth))
        else:
            indices = list(n)
        captured = {}

        def hook(index):
            def _capture(_module, _inputs, output):
                captured[index] = output
            return _capture

        handles = [
            self.model.blocks[i].attn.register_forward_hook(hook(i))
            for i in indices
        ]
        try:
            _, intermediates = self.model.forward_intermediates(
                x, indices=indices, return_prefix_tokens=True,
                norm=norm, output_fmt='NCHW' if reshape else 'NLC')
        finally:
            for handle in handles:
                handle.remove()

        prefix = self.model.num_prefix_tokens
        height, width = self.model.patch_embed.dynamic_feat_size(
            (x.shape[-2], x.shape[-1]))
        attn = []
        for index in indices:
            value = captured[index]
            # timm's EvaBlock applies layer-scale outside the attention
            # module: x = x + gamma_1 * attn(norm1(x)).  Meta DINOv3's
            # F_attn (x_c) is exactly gamma_1 * attn(norm1(x)), so include
            # the scale before the final norm.
            gamma = getattr(self.model.blocks[index], 'gamma_1', None)
            if gamma is not None:
                value = value * gamma
            if norm:
                value = self.model.norm(value)
            value = value[:, prefix:]
            if reshape:
                value = value.reshape(
                    x.shape[0], height, width, -1).permute(
                        0, 3, 1, 2).contiguous()
            attn.append(value)
        sums = [item[0] for item in intermediates]
        return tuple(sums), tuple(attn)


def _build_dinov3(weights: str | Path):
    """Build a DINOv3 ViT from Meta- or timm-format weights.

    The official RSRM weights are the Meta DINOv3 ViT-B/16 checkpoint.  The
    locally available ``checkpoints/dinov3/pytorch_model.bin`` is the timm
    DINOv3 ViT-S/16 export (``reg_token`` / ``gamma_1/2`` keys), so the
    loader dispatches to the matching architecture instead of remapping
    timm weights onto the Meta module layout.
    """
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(f'missing DINOv3 weights: {path}')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    if not isinstance(state, dict):
        raise RuntimeError(f'{path}: expected a DINOv3 state dict')
    if 'reg_token' in state or any(
            key.endswith('.gamma_1') for key in state):
        return _build_timm_dinov3(state)
    return _build_meta_dinov3(state, path)


def _build_timm_dinov3(state):
    import timm
    embed_dim = int(state['patch_embed.proj.weight'].shape[0])
    depth = 1 + max(int(key.split('.')[1]) for key in state
                    if key.startswith('blocks.'))
    if embed_dim == 384:
        name = 'vit_small_patch16_dinov3'
    elif embed_dim == 768:
        name = 'vit_base_patch16_dinov3'
    elif embed_dim == 1024:
        name = 'vit_large_patch16_dinov3'
    else:
        raise ValueError(f'unsupported timm DINOv3 width: {embed_dim}')
    model = timm.create_model(name, pretrained=False, num_classes=0)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f'timm DINOv3 {name} weights do not match '
            f'({len(missing)} missing, {len(unexpected)} unexpected); '
            f'first missing: {missing[:5]}; first unexpected: '
            f'{unexpected[:5]}')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return _TimmDINOv3(model, depth)


def _build_meta_dinov3(state, path: Path) -> DinoVisionTransformer:
    """Build the vendored Meta DINOv3 architecture."""
    state = dict(state)
    embed_dim = int(state['patch_embed.proj.weight'].shape[0])
    depth = 1 + max(int(key.split('.')[1]) for key in state
                    if key.startswith('blocks.'))
    num_heads = max(1, embed_dim // 64)
    n_storage = (int(state['storage_tokens'].shape[1])
                 if 'storage_tokens' in state else 0)
    qkv_bias = any(key.endswith('attn.qkv.bias') for key in state)
    mask_k_bias = any(key.endswith('attn.qkv.bias_mask') for key in state)
    model = DinoVisionTransformer(
        img_size=480,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords='separate',
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype='fp32',
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        ffn_ratio=4.0,
        qkv_bias=qkv_bias,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer='layernormbf16',
        ffn_layer='mlp',
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=n_storage,
        mask_k_bias=mask_k_bias,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {'mask_token', 'rope_embed.periods'}
    bad_missing = [key for key in missing if key not in allowed_missing]
    if bad_missing or unexpected:
        raise RuntimeError(
            f'{path}: DINOv3 weights do not match the inferred architecture '
            f'({len(bad_missing)} missing, {len(unexpected)} unexpected); '
            f'first missing: {bad_missing[:5]}; first unexpected: '
            f'{unexpected[:5]}')
    # ``mask_token`` is never used without masks, but prepare_tokens adds
    # ``0 * mask_token``; zero it so an uninitialized buffer cannot poison
    # the input.
    torch.nn.init.zeros_(model.mask_token)
    model.eval()
    return model


class RSRM(torch.nn.Module):
    """Official RSRM forward (SAFR + ASE + HPM)."""

    def __init__(self, backbone: torch.nn.Module, layers: int = 12,
                 seed: int = 0):
        super().__init__()
        self.backbone = backbone
        self.layers = int(layers)
        self.seed = int(seed)
        self._kmeans_call = 0

    def forward(self, img_s_list, mask_s_list, img_q):
        h, w = img_q.shape[-2:]
        feature_s_last_list = []
        feature_s_attn_list = []
        resize_mask_s_list = []
        for k in range(len(img_s_list)):
            features_sum, features_attn = self.backbone.get_intermediate_layers(
                x=img_s_list[k], n=range(0, self.layers), reshape=True)
            feature_s_attn_list.append(
                torch.stack(features_attn, dim=1))
            feature_s_last_list.append(features_sum[-1])
            resize_mask = F.interpolate(
                mask_s_list[k].unsqueeze(1).float(),
                size=features_attn[-1].shape[-2:], mode='bilinear',
                align_corners=True)
            resize_mask_s_list.append(
                (resize_mask > 0.5).long().squeeze(1))

        _, features_attn = self.backbone.get_intermediate_layers(
            x=img_q, n=range(0, self.layers), reshape=True)
        feature_q_attn = torch.stack(features_attn, dim=1)

        # SAFR
        channel_weights = self.features_refusion(
            feature_s_last_list, feature_s_attn_list, resize_mask_s_list)
        feature_q = (
            feature_q_attn * channel_weights[:, :, None, None, None]
            / feature_q_attn.std(dim=(2, 3, 4), keepdim=True)
        ).sum(dim=1)
        feature_s_list = []
        for k in range(len(img_s_list)):
            feature_s_list.append((
                feature_s_attn_list[k]
                * channel_weights[:, :, None, None, None]
                / feature_s_attn_list[k].std(dim=(2, 3, 4), keepdim=True)
            ).sum(dim=1))

        # ASE
        feature_fg_list, feature_bg_list = [], []
        fg_thresh_ls, bg_thresh_ls = [], []
        for k in range(len(img_s_list)):
            feature_fg = self.masked_average_pooling(
                feature_s_list[k],
                (resize_mask_s_list[k] == 1).float())[None, :]
            feature_bg = self.masked_average_pooling(
                feature_s_list[k],
                (resize_mask_s_list[k] == 0).float())[None, :]
            feature_fg_list.append(feature_fg)
            feature_bg_list.append(feature_bg)

            support_sim = self.similarity_func(
                feature_s_list[k], feature_fg[0].unsqueeze(-1).unsqueeze(-1),
                feature_bg[0].unsqueeze(-1).unsqueeze(-1))
            support_prob = support_sim.softmax(1)
            fg_thresh_ls.append(self.find_robust_threshold_iou(
                support_prob[:, 1], resize_mask_s_list[k]))
            bg_thresh_ls.append(self.find_robust_threshold_iou(
                support_prob[:, 0], 1 - resize_mask_s_list[k]))

        fg_thresh = torch.tensor(fg_thresh_ls).mean().item()
        bg_thresh = torch.tensor(bg_thresh_ls).mean().item()
        fg_proto = torch.mean(
            torch.cat(feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        bg_proto = torch.mean(
            torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        prior_sim = self.similarity_func(feature_q, fg_proto, bg_proto)
        prior_mask = self.generate_prior_mask(
            prior_sim.softmax(1), fg_threshold=fg_thresh,
            bg_threshold=bg_thresh)

        cross_feat_s_list = []
        for k in range(len(feature_s_list)):
            cross = self.support_enhance(
                feature_q, prior_mask, feature_s_list[k],
                resize_mask_s_list[k])
            cross_feat_s_list.append((cross + feature_s_list[k]) / 2)
        cross_feat_q = feature_q

        # HPM
        fg_sim_ls, bg_sim_ls = [], []
        for k in range(len(img_s_list)):
            fg_sim, bg_sim = self.multi_scale_similarity(
                cross_feat_q, cross_feat_s_list[k], resize_mask_s_list[k])
            fg_sim_ls.append(fg_sim.unsqueeze(1))
            bg_sim_ls.append(bg_sim.unsqueeze(1))
        fg_sim = torch.mean(torch.cat(fg_sim_ls, dim=1), dim=1, keepdim=True)
        bg_sim = torch.mean(torch.cat(bg_sim_ls, dim=1), dim=1, keepdim=True)
        out = torch.cat((bg_sim, fg_sim), dim=1)
        out = F.interpolate(out, size=(h, w), mode='bilinear',
                            align_corners=True)
        return [out]

    # SAFR ---------------------------------------------------------------
    def features_refusion(self, feature_s_last_list, feature_s_attn_list,
                          mask_s_list):
        fusion_weights = 0
        for k in range(len(feature_s_attn_list)):
            batch, layers, channels, height, width = \
                feature_s_attn_list[k].shape
            mask_s = mask_s_list[k]
            channel_num = round(channels / 200)
            final_feature = feature_s_last_list[k]
            channel_std = final_feature.var(dim=(-1, -2))
            top_indices = torch.topk(
                channel_std, channel_num, dim=1).indices
            layer_var = feature_s_attn_list[k].var(dim=(-1, -2))
            batch_indices = torch.arange(
                batch, device=layer_var.device).view(batch, 1, 1).expand(
                    batch, layers, channel_num)
            layer_indices = torch.arange(
                layers, device=layer_var.device).view(1, layers, 1).expand(
                    batch, layers, channel_num)
            channel_indices = top_indices.unsqueeze(1).expand(
                batch, layers, channel_num)
            selected_variances = layer_var[
                batch_indices, layer_indices, channel_indices]
            selected_sum = selected_variances.sum(dim=2)
            total_sum = layer_var.sum(dim=2)
            var_rate = selected_sum / (total_sum + 1e-8)

            indicator = self.find_best_semantic_layer(var_rate)
            fdr_ls = []
            for layer in range(layers):
                if indicator[layer]:
                    feat_s = feature_s_attn_list[k][:, layer, :, :, :]
                    fdr_ls.append(
                        self.fisher_ratio(feat_s, mask_s).unsqueeze(-1))
                else:
                    fdr_ls.append(torch.zeros(
                        batch, 1, device=layer_var.device))
            fdr_ls = torch.cat(fdr_ls, dim=-1)
            fusion_weights = fusion_weights + (
                fdr_ls * total_sum) / (selected_sum + 1e-8)
        fusion_weights = fusion_weights / len(feature_s_attn_list)
        return fusion_weights / fusion_weights.sum(dim=1, keepdim=True)

    @staticmethod
    def find_best_semantic_layer(var_rate):
        batch, layers = var_rate.shape
        mean_rate = var_rate.mean(dim=1, keepdim=True)
        is_location = var_rate > mean_rate
        indicator = torch.zeros_like(var_rate, dtype=torch.bool)
        for b in range(batch):
            location_idx = torch.nonzero(
                is_location[b], as_tuple=False).flatten()
            for i in range(location_idx.size(0)):
                start = location_idx[i].item() + 1
                if start >= layers:
                    continue
                end = layers
                if i + 1 < location_idx.size(0):
                    end = location_idx[i + 1].item()
                if end <= start:
                    continue
                candidate = var_rate[b, start:end]
                target = start + torch.argmin(candidate).item()
                indicator[b, target] = True
        return indicator.any(dim=0)

    @staticmethod
    def fisher_ratio(feature, mask):
        batch, channels, height, width = feature.shape
        scores = []
        for i in range(batch):
            feat_flat = feature[i].view(channels, -1).T
            mask_flat = mask[i].view(-1)
            fg_feat = feat_flat[mask_flat == 1]
            bg_feat = feat_flat[mask_flat == 0]
            if len(fg_feat) < 2 or len(bg_feat) < 2:
                continue
            fg_center = fg_feat.mean(dim=0)
            bg_center = bg_feat.mean(dim=0)
            between = torch.outer(fg_center - bg_center,
                                  fg_center - bg_center)
            within = torch.cov(fg_feat.T) + torch.cov(bg_feat.T)
            fisher = torch.trace(between) / torch.trace(within + 1e-8)
            scores.append(fisher)
        return torch.stack(scores)

    # ASE ----------------------------------------------------------------
    @staticmethod
    def similarity_func(feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)
        return torch.cat(
            (similarity_bg[:, None, ...], similarity_fg[:, None, ...]),
            dim=1) * 10.0

    @staticmethod
    def masked_average_pooling(feature, mask):
        mask = F.interpolate(
            mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear',
            align_corners=True)
        masked = torch.sum(feature * mask, dim=(2, 3))
        return masked / (mask.sum(dim=(2, 3)) + 1e-5)

    @staticmethod
    def find_robust_threshold_iou(prob_map, true_mask, num_thresholds=21):
        def compute_iou(pred, target, eps=1e-6):
            intersection = (pred * target).sum(dim=(1, 2))
            union = (pred + target - pred * target).sum(dim=(1, 2))
            return (intersection + eps) / (union + eps)

        device = prob_map.device
        thresholds = torch.linspace(
            0.0, 1.0, steps=num_thresholds, device=device)
        batch = prob_map.size(0)
        best_ious = torch.zeros(batch, device=device)
        best_thresholds = torch.zeros(batch, device=device)
        all_ious = []
        for threshold in thresholds:
            pred_binary = (prob_map >= threshold).float()
            ious = compute_iou(pred_binary, true_mask)
            all_ious.append(ious.unsqueeze(0))
            better = ious > best_ious
            best_ious = torch.where(better, ious, best_ious)
            best_thresholds = torch.where(
                better, threshold, best_thresholds)
        iou_curve = torch.cat(all_ious, dim=0)
        threshold_mask = iou_curve >= (0.75 * best_ious.unsqueeze(0))
        robust = thresholds.unsqueeze(1).expand_as(iou_curve)[threshold_mask]
        return robust.mean().item()

    @staticmethod
    def generate_prior_mask(prob_map, fg_threshold=0.7,
                            bg_threshold=0.6, topk=12):
        device = prob_map.device
        batch, _, height, width = prob_map.shape
        fg_mask = torch.zeros(
            (batch, height, width), dtype=torch.bool, device=device)
        bg_mask = torch.zeros(
            (batch, height, width), dtype=torch.bool, device=device)
        fg_prob = prob_map[:, 1]
        bg_prob = prob_map[:, 0]
        for i in range(batch):
            high_fg = fg_prob[i] > fg_threshold
            if high_fg.any():
                fg_mask[i] = high_fg
            else:
                flat = fg_prob[i].flatten()
                _, indices = torch.topk(flat, min(topk, flat.numel()))
                temp = torch.zeros_like(flat, dtype=torch.bool)
                temp[indices] = True
                fg_mask[i] = temp.reshape(height, width)
            high_bg = bg_prob[i] > bg_threshold
            if high_bg.any():
                bg_mask[i] = high_bg
            else:
                flat = bg_prob[i].flatten()
                _, indices = torch.topk(flat, min(topk, flat.numel()))
                temp = torch.zeros_like(flat, dtype=torch.bool)
                temp[indices] = True
                bg_mask[i] = temp.reshape(height, width)
        return torch.stack([bg_mask, fg_mask], dim=1).float()

    @staticmethod
    def support_enhance(x_q, mask_q, x_s, mask_s):
        batch, channels, height, width = x_q.shape
        feat_q = x_q.view(batch, channels, height * width)
        feat_s = x_s.view(batch, channels, height * width)
        fg_mask_q = mask_q[:, 1].view(batch, -1)
        bg_mask_q = mask_q[:, 0].view(batch, -1)
        fg_mask_s = mask_s.view(batch, -1)
        bg_mask_s = 1 - fg_mask_s

        sq_mask = torch.logical_or(
            torch.logical_and(fg_mask_q.unsqueeze(1) > 0.5,
                              fg_mask_s.unsqueeze(2) > 0.5),
            torch.logical_and(bg_mask_q.unsqueeze(1) > 0.5,
                              bg_mask_s.unsqueeze(2) > 0.5))
        sq_mask = (1 - sq_mask.float()) * -1e9

        attn_s = torch.bmm(feat_s.transpose(1, 2), feat_q)
        attn_s = attn_s / (channels ** 0.5) + sq_mask
        attn_s = F.softmax(attn_s, dim=-1)
        cross = torch.bmm(attn_s, feat_q.transpose(1, 2)).transpose(1, 2)
        return cross.view(batch, channels, height, width)

    # HPM ----------------------------------------------------------------
    def multi_scale_similarity(self, feature_q, feature_s, mask_s):
        batch, channels, height, width = feature_q.shape
        fg_scales = [1, 9, -1]
        bg_scales = [1, 18, -1]
        fg_sim_ls, bg_sim_ls = [], []
        for b in range(batch):
            cur_feature = feature_s[b].view(channels, -1).permute(1, 0)
            cur_mask = mask_s[b].view(-1)
            cur_feature_q = feature_q[b].view(
                channels, -1).permute(1, 0)
            fg_idx = torch.where(cur_mask > 0.5)[0]
            bg_idx = torch.where(cur_mask <= 0.5)[0]
            fg_feature = cur_feature[fg_idx]
            bg_feature = cur_feature[bg_idx]

            fg_sim_maps, bg_sim_maps = [], []
            for fg_scale, bg_scale in zip(fg_scales, bg_scales):
                if fg_scale == -1 and bg_scale == -1:
                    sim_fg = torch.mm(
                        F.normalize(fg_feature, p=2, dim=1),
                        F.normalize(cur_feature_q, p=2, dim=1).permute(1, 0))
                    sim_bg = torch.mm(
                        F.normalize(bg_feature, p=2, dim=1),
                        F.normalize(cur_feature_q, p=2, dim=1).permute(1, 0))
                    fg_map, _ = torch.max(sim_fg, dim=0)
                    bg_map, _ = torch.max(sim_bg, dim=0)
                    sim_map = torch.cat(
                        [bg_map.view(1, height, width),
                         fg_map.view(1, height, width)], dim=0)
                    fg_sim_maps.append(sim_map[1:])
                    bg_sim_maps.append(sim_map[0:1])
                else:
                    fg_n = min(fg_scale, fg_feature.shape[0])
                    bg_n = min(bg_scale, bg_feature.shape[0])
                    fg_map = self.region_similarity(
                        fg_feature, feature_q[b].unsqueeze(0), fg_n)
                    bg_map = self.region_similarity(
                        bg_feature, feature_q[b].unsqueeze(0), bg_n)
                    sim_map = torch.cat([bg_map, fg_map], dim=0)
                    fg_sim_maps.append(sim_map[1:])
                    bg_sim_maps.append(sim_map[0:1])

            fg_sim_maps = torch.cat(fg_sim_maps, dim=0)
            bg_sim_maps = torch.cat(bg_sim_maps, dim=0)
            conf_map = fg_sim_maps - bg_sim_maps
            mask = (conf_map > 0).float()
            fg_norm = (fg_sim_maps - fg_sim_maps.min()) / (
                fg_sim_maps.max() - fg_sim_maps.min() + 1e-5)
            fg_confs = self.masked_average_pooling(
                fg_norm.unsqueeze(1), mask)
            bg_confs = self.masked_average_pooling(
                fg_norm.unsqueeze(1), 1 - mask)
            confs = (fg_confs - bg_confs).unsqueeze(-1)
            confs = (confs / fg_sim_maps.abs().std()).softmax(0)
            fg_sim_ls.append(
                torch.mean(confs * fg_sim_maps, dim=0, keepdim=True))
            bg_sim_ls.append(
                torch.mean(confs * bg_sim_maps, dim=0, keepdim=True))
        return torch.cat(fg_sim_ls, dim=0), torch.cat(bg_sim_ls, dim=0)

    def region_similarity(self, feature_s, feature_q, n_clusters):
        if feature_s.shape[0] == 0:
            return torch.zeros_like(feature_q[:, :1])
        n_clusters = int(min(max(n_clusters, 1), feature_s.shape[0]))
        if n_clusters == 1:
            labels = np.zeros(feature_s.shape[0], dtype=np.int64)
        else:
            data = feature_s.detach().cpu().float().numpy()
            self._kmeans_call += 1
            kmeans = KMeans(
                n_clusters=n_clusters, init='k-means++', n_init=1,
                max_iter=100,
                random_state=self.seed + self._kmeans_call)
            labels = kmeans.fit_predict(data)
        labels = torch.from_numpy(labels).to(feature_s.device)
        sim_maps = []
        for region_id in range(n_clusters):
            region = feature_s[labels == region_id]
            if region.shape[0] == 0:
                continue
            proto = region.mean(dim=0, keepdim=True)
            sim_maps.append(F.cosine_similarity(
                feature_q, proto.unsqueeze(-1).unsqueeze(-1), dim=1))
        sim_maps = torch.cat(sim_maps, dim=0)
        sim_maps, _ = torch.max(sim_maps, dim=0, keepdim=True)
        return sim_maps


def build_rsrm(dinov3_weights: str | Path, seed: int = 0) -> RSRM:
    backbone = _build_dinov3(dinov3_weights)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone.eval()
    layers = getattr(
        backbone, 'depth', getattr(backbone, 'n_blocks', 12))
    return RSRM(backbone, layers=int(layers), seed=seed)


__all__ = ['RSRM', 'build_rsrm']
