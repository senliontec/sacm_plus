"""SP-SAM: self-prompting SAM for curvilinear structure segmentation.

The framework keeps the official SAM ViT-L image encoder and the official
promptable mask-decoder path, and adds:

* a prompt-proposal head on the 64x64 image embedding that predicts where a
  click should be placed on a curvilinear structure;
* point-conditioned decoding with the official prompt encoder, two-way
  transformer, mask tokens and stage-1 hypernetworks;
* label-free aggregation of several point-conditioned masks by predicted
  IoU.

Unlike the DCD/CoSNA scaffold there is no prompt-free four-candidate
consensus: every output mask is anchored at an explicit image-derived
point.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segment_anything import build_sam_vit_l


class PromptProposalHead(nn.Module):
    """Click heatmap at 1/4 resolution (256x256 at 1024 input).

    The SAM embedding is 64x64; two bilinear upsamples with convolutions
    restore a 256x256 grid so inference clicks are within ~2 px of the
    heatmap maximum instead of ~8 px at the coarse cell centre.
    """

    def __init__(self, in_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 3, padding=1),
            nn.GroupNorm(16, hidden),
            nn.GELU(),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.GroupNorm(8, hidden // 2),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(hidden // 2, hidden // 4, 3, padding=1),
            nn.GroupNorm(8, hidden // 4),
            nn.GELU(),
        )
        self.out = nn.Conv2d(hidden // 4, 1, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = self.body(features)
        h = F.interpolate(h, scale_factor=2, mode='bilinear',
                          align_corners=False)
        h = self.up1(h)
        h = F.interpolate(h, scale_factor=2, mode='bilinear',
                          align_corners=False)
        h = self.up2(h)
        return self.out(h)


def sam_preprocess(sam, images: torch.Tensor) -> torch.Tensor:
    """[0,1] RGB -> official SAM normalization (0-255 input scale)."""
    return sam.preprocess(images * 255.0)


def prompt_mask_forward(mask_decoder, image_embeddings, image_pe,
                        sparse_prompt_embeddings, dense_prompt_embeddings):
    """Official SAM mask-decoder math (stage-1 hypernetworks).

    ``b`` prompt sets each receive their own image embedding, as in
    ``MaskDecoder.predict_masks``.  The stage-2 / SACM additions in the
    copied scaffold are not used, so every parameter touched here has an
    official SAM checkpoint value.
    """
    b = image_embeddings.shape[0]
    src = image_embeddings + dense_prompt_embeddings
    pos_src = (image_pe.repeat_interleave(b, dim=0)
               if image_pe.shape[0] == 1 else image_pe)
    tokens = torch.cat([
        mask_decoder.iou_token.weight.unsqueeze(0).expand(b, -1, -1),
        mask_decoder.mask_tokens.weight.unsqueeze(0).expand(b, -1, -1),
        sparse_prompt_embeddings,
    ], dim=1)
    hs, src = mask_decoder.transformer(src, pos_src, tokens)
    iou_token_out = hs[:, 0, :]
    mask_tokens_out = hs[:, 1:1 + mask_decoder.num_mask_tokens, :]
    h, w = image_embeddings.shape[-2:]
    src = src.transpose(1, 2).view(b, -1, h, w)
    upscaled = mask_decoder.output_upscaling(src)
    hyper = torch.stack([
        mask_decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
        for i in range(mask_decoder.num_mask_tokens)], dim=1)
    uh, uw = upscaled.shape[-2:]
    masks = (hyper @ upscaled.flatten(2)).view(
        b, mask_decoder.num_mask_tokens, uh, uw)
    iou = mask_decoder.iou_prediction_head(iou_token_out)
    return masks, iou


class SelfPromptSAM(nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.sam = sam
        self.proposal = PromptProposalHead(sam.prompt_encoder.embed_dim)

    @property
    def prompt_encoder(self):
        return self.sam.prompt_encoder

    @property
    def mask_decoder(self):
        return self.sam.mask_decoder

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        out = self.sam.image_encoder(sam_preprocess(self.sam, images))
        return out[0] if isinstance(out, tuple) else out

    def decode_points(self, features: torch.Tensor,
                      point_coords: torch.Tensor,
                      point_labels: torch.Tensor):
        """Decode one point prompt per set.

        ``point_coords`` is [B, P, 2] in the 1024-pixel input frame and
        ``point_labels`` is [B, P] (1 foreground, 0 background).  Returns
        masks [B, P, 4, 256, 256] and IoU logits [B, P, 4].
        """
        b, p, _ = point_coords.shape
        flat_coords = point_coords.reshape(b * p, 1, 2).float()
        flat_labels = point_labels.reshape(b * p, 1).to(torch.int64)
        sparse, dense = self.sam.prompt_encoder(
            points=(flat_coords, flat_labels), boxes=None, masks=None)
        feats = features.repeat_interleave(p, dim=0)
        masks, iou = prompt_mask_forward(
            self.mask_decoder, feats,
            self.sam.prompt_encoder.get_dense_pe(), sparse, dense)
        return (masks.view(b, p, *masks.shape[1:]),
                iou.view(b, p, iou.shape[-1]))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encode(images)
        return self.proposal(features)


def build_model(args) -> SelfPromptSAM:
    sam = build_sam_vit_l(
        checkpoint=str(args.sam_checkpoint),
        use_adapter=bool(getattr(args, 'sp_adapters', 1)),
        adapter_dim_ratio=float(getattr(args, 'adapter_dim_ratio', 0.1)))
    return SelfPromptSAM(sam)


def select_peaks(heatmap: torch.Tensor, k: int, radius: int) -> torch.Tensor:
    """Top-k local maxima with a minimum Euclidean distance.

    ``heatmap`` is [H, W] probabilities.  Returns [n, 2] pixel coordinates
    (x, y) in heatmap units, at most ``k`` peaks.
    """
    if heatmap.numel() == 0 or k <= 0:
        return heatmap.new_zeros((0, 2))
    peak_max = float(heatmap.max())
    if peak_max < 1e-6:
        return heatmap.new_zeros((0, 2))
    pooled = F.max_pool2d(
        heatmap[None, None], kernel_size=2 * radius + 1, stride=1,
        padding=radius)[0, 0]
    floor = max(1e-6, 0.05 * peak_max)
    ys, xs = torch.nonzero(
        (heatmap >= pooled - 1e-6) & (heatmap >= floor), as_tuple=True)
    if ys.numel() == 0:
        return heatmap.new_zeros((0, 2))
    values = heatmap[ys, xs]
    order = torch.argsort(values, descending=True)
    kept: list[tuple[float, float]] = []
    for index in order.tolist():
        x, y = float(xs[index]), float(ys[index])
        if all((x - px) ** 2 + (y - py) ** 2 >= radius * radius
               for px, py in kept):
            kept.append((x, y))
            if len(kept) == k:
                break
    coords = heatmap.new_tensor(kept, dtype=torch.float32)
    return coords + 0.5
