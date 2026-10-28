"""Contrastive Parameter Selection (CPS) for A2SAM.

The released A2SAM code accumulates squared gradients on a validation loader,
normalizes influence and magnitude globally, ranks every element, and freezes
the low-score 96.5% (keeping the top 3.5%).  Norm, position-embedding and 1-D
neck parameters are forced frozen, and the IoU prediction head is excluded.

This module keeps that element-wise behavior.  The chosen elements receive
an identity backward hook; all other elements receive a zero mask.  Tensors
whose mask is entirely zero are frozen with ``requires_grad=False`` so they
are skipped by the optimizer (and by decoupled weight decay).  Partially kept
tensors still undergo AdamW weight decay on their frozen elements, exactly as
the released implementation does.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn


def _flatten(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.detach().reshape(-1).float()
                      for tensor in tensors.values()])


def contrastive_scores(
    named_grads: Mapping[str, torch.Tensor],
    named_params: Mapping[str, torch.Tensor],
    *,
    alpha: float = 0.4,
) -> dict[str, torch.Tensor]:
    """Compute the paper's influence/magnitude contrastive score.

    ``named_grads`` holds accumulated gradients; the released code squares
    the accumulated gradient before normalizing, which is what this function
    does as well.
    """
    if not named_grads:
        return {}
    names = [name for name in named_grads if name in named_params]
    if not names:
        return {}
    influence = {name: named_grads[name].detach().float().pow(2)
                 for name in names}
    magnitude = {name: named_params[name].detach().float().abs()
                 for name in names}
    all_influence = _flatten(influence)
    all_magnitude = _flatten(magnitude)
    i_min = float(all_influence.min())
    i_max = float(all_influence.max())
    p_min = float(all_magnitude.min())
    p_max = float(all_magnitude.max())
    scale_i = (i_max - i_min) or 1.0
    scale_p = (p_max - p_min) or 1.0
    scores = {}
    for name in names:
        influence_n = (influence[name] - i_min) / scale_i
        magnitude_n = (magnitude[name] - p_min) / scale_p
        scores[name] = (alpha * (influence_n / (magnitude_n + 1.0))
                        + (1.0 - alpha) * influence_n)
    return scores


def select_contrastive_masks(
    scores: Mapping[str, torch.Tensor],
    named_params: Mapping[str, torch.Tensor],
    *,
    keep_ratio: float = 0.035,
    freeze_norm: bool = True,
) -> dict[str, torch.Tensor]:
    """Return per-element 0/1 masks keeping the top ``keep_ratio``."""
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f'keep_ratio must be in (0, 1], got {keep_ratio}')
    if not scores:
        return {}
    names = [name for name in scores if name in named_params]
    flat = torch.cat([scores[name].detach().reshape(-1) for name in names])
    keep_num = max(1, int(math.ceil(flat.numel() * keep_ratio)))
    keep_num = min(keep_num, flat.numel())
    threshold = torch.topk(flat, keep_num, largest=True).values[-1]
    masks = {}
    for name in names:
        parameter = named_params[name]
        score = scores[name].detach().to(parameter.device)
        mask = (score >= threshold).float()
        if freeze_norm and (
                'iou_prediction_head' in name
                or 'norm' in name
                or 'pos_embed' in name
                or 'pos' in name
                or (parameter.ndim == 1 and 'neck' in name)):
            mask = torch.zeros_like(mask)
        masks[name] = mask
    return masks


def apply_cps_masks(model: nn.Module,
                    masks: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Install element-wise CPS masks and freeze fully masked tensors.

    Returns a small statistics dictionary (trainable element count and
    fraction) for logging.
    """
    parameters = dict(model.named_parameters())
    kept_elements = 0
    total_elements = 0
    for name, parameter in parameters.items():
        if not parameter.requires_grad:
            continue
        mask = masks.get(name)
        if mask is None:
            continue
        mask = mask.to(device=parameter.device, dtype=torch.float32)
        total_elements += mask.numel()
        kept_elements += int(mask.sum().item())
        if not bool(mask.any()):
            parameter.requires_grad_(False)
            continue

        def hook(grad, mask=mask):
            return grad * mask

        parameter.register_hook(hook)
    return {
        'kept_elements': kept_elements,
        'total_elements': total_elements,
        'kept_fraction': (kept_elements / total_elements
                          if total_elements else 0.0),
    }
