"""Contrastive Parameter Selection (CPS) for A2SAM.

The official release builds a **freeze mask**:

* ``mask == 1`` means "zero this element's gradient";
* ``mask == 0`` means "let this element train";
* the lowest 96.5% of contrastive scores are frozen, leaving the top 3.5%;
* ``iou_prediction_head`` is excluded from scoring and is never masked;
* image-encoder norm/position parameters and 1-D neck parameters are forced
  frozen after ranking;
* parameters are never switched to ``requires_grad=False``.

This module reproduces that convention.  ``mode='paper'`` exposes the paper's
raw ``|w| + 1`` score and floor(kN) train-mask rule as an ablation; the
default is the official release implementation.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

SAM_PREFIXES = ('image_encoder.', 'prompt_encoder.', 'mask_decoder.')
CPS_EXCLUDED_PREFIXES = ('mask_decoder.iou_prediction_head.',)


def _flatten(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.detach().reshape(-1).float()
                      for tensor in tensors.values()])


def sam_cps_candidates(
    model: nn.Module,
    *,
    prefixes: tuple[str, ...] = SAM_PREFIXES,
) -> dict[str, nn.Parameter]:
    """Return the official CPS candidate set.

    Only the SAM image encoder, prompt encoder and mask decoder participate.
    The IoU prediction head is deliberately excluded; reconstructed U-Net,
    prior and prompt-fusion modules are outside CPS.
    """
    candidates = {}
    for name, parameter in model.named_parameters():
        if not name.startswith(prefixes):
            continue
        if name.startswith(CPS_EXCLUDED_PREFIXES):
            continue
        candidates[name] = parameter
    return candidates


def contrastive_scores(
    named_grads: Mapping[str, torch.Tensor],
    named_params: Mapping[str, torch.Tensor],
    *,
    alpha: float = 0.4,
    mode: str = 'official',
) -> dict[str, torch.Tensor]:
    """Compute per-element contrastive scores.

    ``official`` mirrors ``utils_/pruning.calculate_contrastive_score``:
    squared accumulated gradient and raw ``|w|`` are each globally min-max
    normalized, then ``alpha * i / (rho + 1) + (1 - alpha) * i``.

    ``paper`` mirrors Eq. (7)-(18): ``S_i = g**2``,
    ``S_p = S_i / (|w| + 1)`` and ``S = alpha*S_p + (1-alpha)*S_i`` without
    normalization.
    """
    if mode not in {'official', 'paper'}:
        raise ValueError(f"unknown CPS mode: {mode!r}")
    if not named_grads:
        return {}
    names = [name for name in named_grads if name in named_params]
    if not names:
        return {}
    influence = {name: named_grads[name].detach().float().pow(2)
                 for name in names}
    magnitude = {name: named_params[name].detach().float().abs()
                 for name in names}
    if mode == 'paper':
        return {
            name: alpha * (influence[name] / (magnitude[name] + 1.0))
            + (1.0 - alpha) * influence[name]
            for name in names
        }

    all_influence = _flatten(influence)
    all_magnitude = _flatten(magnitude)
    i_min, i_max = all_influence.min(), all_influence.max()
    p_min, p_max = all_magnitude.min(), all_magnitude.max()
    scores = {}
    for name in names:
        influence_n = influence[name]
        magnitude_n = magnitude[name]
        if i_max > i_min:
            influence_n = (influence_n - i_min) / (i_max - i_min)
        if p_max > p_min:
            magnitude_n = (magnitude_n - p_min) / (p_max - p_min)
        scores[name] = (alpha * (influence_n / (magnitude_n + 1.0))
                        + (1.0 - alpha) * influence_n)
    return scores


def _force_image_encoder_freeze(name: str, parameter: nn.Parameter) -> bool:
    if not name.startswith('image_encoder.'):
        return False
    if 'norm' in name or 'pos_embed' in name or 'pos' in name:
        return True
    return parameter.ndim == 1 and 'neck' in name


def select_freeze_masks(
    scores: Mapping[str, torch.Tensor],
    named_params: Mapping[str, torch.Tensor],
    *,
    keep_ratio: float = 0.035,
    mode: str = 'official',
    force_image_encoder: bool = True,
) -> dict[str, torch.Tensor]:
    """Build official per-element freeze masks (``1 = frozen``).

    For ``official``, the lowest ``int(N * (1 - keep_ratio))`` scores are
    frozen, exactly like the release's ``keep_ratio=0.965`` ascending sort.
    For ``paper``, the top ``floor(N * keep_ratio)`` scores form the train
    set; ties at the threshold are trainable as in Eq. (21).
    """
    if mode not in {'official', 'paper'}:
        raise ValueError(f"unknown CPS mode: {mode!r}")
    if not 0.0 < keep_ratio < 1.0:
        raise ValueError(f'keep_ratio must be in (0, 1), got {keep_ratio}')
    if not scores:
        return {}
    names = [name for name in scores if name in named_params]
    if not names:
        return {}
    flat = torch.cat([scores[name].detach().reshape(-1) for name in names])
    total = flat.numel()
    masks = {name: torch.zeros_like(
        scores[name].detach(), dtype=torch.float32) for name in names}

    if mode == 'official':
        freeze_num = int(total * (1.0 - keep_ratio))
        if freeze_num:
            frozen = torch.argsort(flat)[:freeze_num]
            flat_mask = torch.zeros(total, dtype=torch.float32)
            flat_mask[frozen] = 1.0
            offset = 0
            for name in names:
                size = scores[name].numel()
                masks[name] = flat_mask[offset:offset + size].reshape(
                    scores[name].shape)
                offset += size
    else:
        train_num = int(total * keep_ratio)
        train_num = max(0, min(train_num, total))
        if train_num:
            threshold = torch.topk(flat, train_num, largest=True).values[-1]
            flat_mask = (flat < threshold).float()
            offset = 0
            for name in names:
                size = scores[name].numel()
                masks[name] = flat_mask[offset:offset + size].reshape(
                    scores[name].shape)
                offset += size

    for name in names:
        parameter = named_params[name]
        if force_image_encoder and _force_image_encoder_freeze(
                name, parameter):
            masks[name] = torch.ones_like(masks[name])
        else:
            masks[name] = masks[name].to(parameter.device)
    return masks


def zero_cps_gradients(model: nn.Module,
                       masks: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Apply freeze masks to gradients after ``backward``.

    This intentionally does not set ``requires_grad=False``: the official
    implementation keeps all parameters in AdamW, so decoupled weight decay
    still applies to fully masked tensors.
    """
    frozen_elements = 0
    trainable_elements = 0
    total_elements = 0
    parameters = dict(model.named_parameters())
    for name, mask in masks.items():
        parameter = parameters.get(name)
        if parameter is None:
            continue
        elements = mask.numel()
        frozen = int(mask.sum().item())
        total_elements += elements
        frozen_elements += frozen
        trainable_elements += elements - frozen
        if parameter.grad is not None:
            mask_device = mask.to(device=parameter.grad.device)
            if mask_device.dtype != torch.bool:
                mask_device = mask_device > 0.5
            parameter.grad.masked_fill_(mask_device, 0.0)
    return {
        'frozen_elements': frozen_elements,
        'trainable_elements': trainable_elements,
        'total_elements': total_elements,
        'trainable_fraction': (trainable_elements / total_elements
                               if total_elements else 0.0),
    }


def cps_rounding(total: int, keep_ratio: float, mode: str = 'official') -> int:
    """Return the official freeze count or paper train count for tests."""
    if mode == 'official':
        return int(total * (1.0 - keep_ratio))
    return int(total * keep_ratio)
