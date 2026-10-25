"""Official PerSAM inference steps, factored out for the baseline adapter.

Adapted from ``3rd/Personalize-SAM/persam.py`` (MIT).  PerSAM is a
training-free, one-reference method.  The adapter runs each support image
independently and majority-votes the three binary masks for the project's
3-shot protocol.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def point_selection(mask_sim: torch.Tensor, topk: int = 1):
    """Top-1 foreground and least-similar background point (official)."""
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = topk_xy - topk_x * h
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()

    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = last_xy - last_x * h
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()
    return topk_xy, topk_label, last_xy, last_label


def prepare_reference_embedding(predictor, ref_image: np.ndarray,
                                ref_mask_rgb: np.ndarray) -> torch.Tensor:
    """Encode one reference image and return its normalized object feature."""
    ref_mask = predictor.set_image(ref_image, ref_mask_rgb)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)
    ref_mask = F.interpolate(
        ref_mask, size=ref_feat.shape[0:2], mode='bilinear')
    ref_mask = ref_mask.squeeze()[0]
    target_feat = ref_feat[ref_mask > 0]
    if target_feat.numel() == 0:
        raise RuntimeError('PerSAM reference mask has no foreground features')
    embedding = target_feat.mean(0).unsqueeze(0)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.unsqueeze(0)


def predict_with_embedding(predictor, target_features: torch.Tensor,
                           original_size, input_size,
                           target_embedding: torch.Tensor) -> np.ndarray:
    """Run the official PerSAM point/box refinement for one support."""
    test_feat = target_features.squeeze()
    c, h, w = test_feat.shape
    test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
    test_feat = test_feat.reshape(c, h * w)
    sim = target_embedding.reshape(1, c) @ test_feat
    sim = sim.reshape(1, 1, h, w)
    sim = F.interpolate(sim, scale_factor=4, mode='bilinear')
    sim = predictor.model.postprocess_masks(
        sim, input_size=input_size, original_size=original_size).squeeze()

    topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(
        sim, topk=1)
    topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
    topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

    sim = (sim - sim.mean()) / torch.std(sim)
    sim = F.interpolate(
        sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode='bilinear')
    attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

    masks, scores, logits, _ = predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        multimask_output=False,
        attn_sim=attn_sim,
        target_embedding=target_embedding,
    )
    best_idx = 0
    first_mask = masks[best_idx].astype(bool)
    if not first_mask.any():
        return first_mask

    masks, scores, logits, _ = predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        mask_input=logits[best_idx:best_idx + 1, :, :],
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    refined = masks[best_idx].astype(bool)
    if not refined.any():
        return refined

    y, x = np.nonzero(refined)
    try:
        box = np.array([x.min(), y.min(), x.max(), y.max()])
        masks, scores, logits, _ = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=box[None, :],
            mask_input=logits[best_idx:best_idx + 1, :, :],
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        final = masks[best_idx].astype(bool)
        return final if final.any() else refined
    except Exception:
        return refined


def majority_vote(masks: list[np.ndarray], threshold: float = 0.5
                  ) -> np.ndarray:
    """Boolean majority vote over independently predicted support masks."""
    if not masks:
        raise RuntimeError('no PerSAM predictions to vote on')
    votes = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        votes += mask.astype(np.uint8)
    return votes > (len(masks) * threshold)
