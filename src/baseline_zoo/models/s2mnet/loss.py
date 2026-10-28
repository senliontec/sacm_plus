"""Morphology-Aware Adaptive Loss (MAL), PyTorch rewrite.

Translated from the released ``s2mnet/losses/{components,mal}.py``.  The five
components, morphology modulation factors, learnable scalar weights
(initialized core=1, boundary=1, structure=1, scale=0.5, texture=0.5) and
the [0.1, 10] clipping are preserved.  Predictions are probabilities, as in
the TensorFlow model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_COMPONENTS = ('core', 'boundary', 'structure', 'scale', 'texture')
_DEFAULT_COEFFICIENTS = dict(
    core=0.5, boundary=1.5, structure=1.0, scale=1.5, texture=1.0)
_INIT_WEIGHTS = dict(
    core=1.0, boundary=1.0, structure=1.0, scale=0.5, texture=0.5)


def _dilation(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size, stride=1,
                        padding=kernel_size // 2)


def _erosion(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size, stride=1,
                         padding=kernel_size // 2)


def detect_boundary(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    return (_dilation(mask, kernel_size)
            - _erosion(mask, kernel_size)).clamp(0.0, 1.0)


def analyze_morphology(y_true: torch.Tensor,
                       eps: float = 1e-6) -> dict[str, torch.Tensor]:
    area = y_true.sum(dim=(1, 2, 3)) + eps
    total_pixels = float(y_true.shape[-2] * y_true.shape[-1])
    dy = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    dx = y_true[:, :, :, 1:] - y_true[:, :, :, :-1]
    dy = F.pad(dy, (0, 0, 0, 1))
    dx = F.pad(dx, (0, 1, 0, 0))
    perimeter = (dy ** 2 + dx ** 2 + eps).sqrt().sum(
        dim=(1, 2, 3)) + eps
    skeleton = _erosion(y_true, kernel_size=3)
    skeleton_area = skeleton.sum(dim=(1, 2, 3)) + eps
    tubularity = (skeleton_area / area).mean().clamp(0.0, 1.0)
    compactness = (4.0 * math.pi * area
                   / (perimeter ** 2 + eps)).mean().clamp(0.0, 1.0)
    boundary = detect_boundary(y_true, kernel_size=5)
    ddy = (boundary[:, :, 2:, :] - 2 * boundary[:, :, 1:-1, :]
           + boundary[:, :, :-2, :])
    ddx = (boundary[:, :, :, 2:] - 2 * boundary[:, :, :, 1:-1]
           + boundary[:, :, :, :-2])
    irregularity = (ddy.abs().mean() + ddx.abs().mean()).clamp(0.0, 1.0)
    object_size = (area / total_pixels).mean().clamp(0.0, 1.0)
    return {
        'tubularity': tubularity,
        'compactness': compactness,
        'irregularity': irregularity,
        'object_size': object_size,
    }


class CoreLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_true: torch.Tensor,
                y_pred: torch.Tensor) -> torch.Tensor:
        y_true = y_true.float()
        y_pred = y_pred.float()
        eps = self.eps
        intersection = (y_true * y_pred).sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + eps) / (
            y_true.sum(dim=(1, 2, 3)) + y_pred.sum(dim=(1, 2, 3)) + eps)
        dice_loss = 1.0 - dice.mean()
        union = (y_true.sum(dim=(1, 2, 3))
                 + y_pred.sum(dim=(1, 2, 3)) - intersection)
        iou = (intersection + eps) / (union + eps)
        iou_loss = 1.0 - iou.mean()
        boundary = detect_boundary(y_true, kernel_size=5)
        weights = 1.0 + 5.0 * boundary
        bce = -(y_true * torch.log(y_pred + eps)
                + (1.0 - y_true) * torch.log(1.0 - y_pred + eps))
        return (0.4 * dice_loss + 0.3 * iou_loss
                + 0.3 * (weights * bce).mean())


class BoundaryLoss(nn.Module):
    def __init__(self, scales=(1, 2, 4), scale_weights=(0.5, 0.3, 0.2)):
        super().__init__()
        if len(scales) != len(scale_weights):
            raise ValueError('scales and scale_weights must match')
        self.scales = list(scales)
        self.scale_weights = list(scale_weights)

    def forward(self, y_true: torch.Tensor,
                y_pred: torch.Tensor) -> torch.Tensor:
        y_true = y_true.float()
        y_pred = y_pred.float()
        total = y_true.new_zeros(())
        for scale, weight in zip(self.scales, self.scale_weights):
            dy_true = y_true[:, :, scale:, :] - y_true[:, :, :-scale, :]
            dy_pred = y_pred[:, :, scale:, :] - y_pred[:, :, :-scale, :]
            dx_true = y_true[:, :, :, scale:] - y_true[:, :, :, :-scale]
            dx_pred = y_pred[:, :, :, scale:] - y_pred[:, :, :, :-scale]
            total = total + weight * (
                (dy_true - dy_pred).abs().mean()
                + (dx_true - dx_pred).abs().mean())
        return total


class StructureLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _compactness(self, mask: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        area = mask.sum(dim=(1, 2, 3)) + eps
        dy = mask[:, :, 1:, :] - mask[:, :, :-1, :]
        dx = mask[:, :, :, 1:] - mask[:, :, :, :-1]
        dy = F.pad(dy, (0, 0, 0, 1))
        dx = F.pad(dx, (0, 1, 0, 0))
        perimeter = (dy ** 2 + dx ** 2 + eps).sqrt().sum(
            dim=(1, 2, 3)) + eps
        return area / (perimeter ** 2 + eps)

    def forward(self, y_true: torch.Tensor,
                y_pred: torch.Tensor) -> torch.Tensor:
        y_true = y_true.float()
        y_pred = y_pred.float()
        morphology = analyze_morphology(y_true, eps=self.eps)
        true_compactness = self._compactness(y_true)
        pred_compactness = self._compactness(y_pred)
        return morphology['compactness'] * (
            true_compactness - pred_compactness).abs().mean()


class ScaleAwareFocalLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_true: torch.Tensor,
                y_pred: torch.Tensor) -> torch.Tensor:
        y_true = y_true.float()
        y_pred = y_pred.float()
        eps = self.eps
        size = float(analyze_morphology(
            y_true, eps=eps)['object_size'].detach().cpu())
        gamma = 3.0 if size < 0.05 else (2.0 if size < 0.20 else 1.5)
        probability = (y_true * y_pred
                       + (1.0 - y_true) * (1.0 - y_pred))
        weight = (1.0 - probability).pow(gamma)
        bce = -(y_true * torch.log(y_pred + eps)
                + (1.0 - y_true) * torch.log(1.0 - y_pred + eps))
        return (weight * bce).mean()


class TextureLoss(nn.Module):
    def forward(self, y_true: torch.Tensor,
                y_pred: torch.Tensor) -> torch.Tensor:
        y_true = y_true.float()
        y_pred = y_pred.float()
        ddy_true = (y_true[:, :, 2:, :] - 2 * y_true[:, :, 1:-1, :]
                    + y_true[:, :, :-2, :])
        ddy_pred = (y_pred[:, :, 2:, :] - 2 * y_pred[:, :, 1:-1, :]
                    + y_pred[:, :, :-2, :])
        ddx_true = (y_true[:, :, :, 2:] - 2 * y_true[:, :, :, 1:-1]
                    + y_true[:, :, :, :-2])
        ddx_pred = (y_pred[:, :, :, 2:] - 2 * y_pred[:, :, :, 1:-1]
                    + y_pred[:, :, :, :-2])
        return ((ddy_true - ddy_pred).abs().mean()
                + (ddx_true - ddx_pred).abs().mean())


class MorphologyAwareLoss(nn.Module):
    def __init__(self, components=_COMPONENTS, learned_weights=True,
                 morphology_modulation=True, coefficients=None,
                 eps=1e-6):
        super().__init__()
        for name in components:
            if name not in _COMPONENTS:
                raise ValueError(f'unknown MAL component: {name}')
        self.components = list(components)
        self.learned_weights = bool(learned_weights)
        self.morphology_modulation = bool(morphology_modulation)
        self.coefficients = {**_DEFAULT_COEFFICIENTS,
                             **(coefficients or {})}
        self.eps = float(eps)
        self.losses = nn.ModuleDict()
        if 'core' in self.components:
            self.losses['core'] = CoreLoss(eps=eps)
        if 'boundary' in self.components:
            self.losses['boundary'] = BoundaryLoss()
        if 'structure' in self.components:
            self.losses['structure'] = StructureLoss(eps=eps)
        if 'scale' in self.components:
            self.losses['scale'] = ScaleAwareFocalLoss(eps=eps)
        if 'texture' in self.components:
            self.losses['texture'] = TextureLoss()
        self.weights = nn.ParameterDict()
        if self.learned_weights:
            for name in self.components:
                self.weights[name] = nn.Parameter(
                    torch.tensor(float(_INIT_WEIGHTS[name])))

    def _modulation(self, morphology: dict[str, torch.Tensor]):
        if not self.morphology_modulation:
            return {name: 1.0 for name in self.components}
        c = self.coefficients
        return {
            'core': 1.0 + c['core'] * morphology['compactness'],
            'boundary': (1.0 + c['boundary'] * morphology['tubularity']
                         + morphology['compactness']),
            'structure': 1.0 + c['structure'] * morphology['tubularity'],
            'scale': 1.0 + c['scale'] * morphology['irregularity'],
            'texture': 1.0 + c['texture'] * morphology['irregularity'],
        }

    def clip_weights(self) -> None:
        if not self.learned_weights:
            return
        with torch.no_grad():
            for parameter in self.weights.values():
                parameter.clamp_(0.1, 10.0)

    def forward(self, y_true: torch.Tensor,
                y_pred: torch.Tensor) -> torch.Tensor:
        y_true = y_true.float()
        y_pred = y_pred.float()
        if (self.components == ['core'] and not self.learned_weights
                and not self.morphology_modulation):
            intersection = (y_true * y_pred).sum()
            dice = (2.0 * intersection + self.eps) / (
                y_true.sum() + y_pred.sum() + self.eps)
            return 1.0 - dice
        morphology = analyze_morphology(y_true, eps=self.eps)
        modulation = self._modulation(morphology)
        total = y_pred.new_zeros(())
        normalizer = y_pred.new_zeros(())
        for name in self.components:
            value = self.losses[name](y_true, y_pred)
            if self.learned_weights:
                weight = self.weights[name].clamp(0.1, 10.0)
            else:
                weight = y_pred.new_tensor(1.0)
            alpha = torch.as_tensor(
                modulation[name], dtype=y_pred.dtype,
                device=y_pred.device)
            total = total + weight * alpha * value
            normalizer = normalizer + weight * alpha
        return total / (normalizer + self.eps)

    def learned_weight_values(self) -> dict[str, float]:
        if not self.learned_weights:
            return {name: 1.0 for name in self.components}
        return {name: float(parameter.detach().cpu())
                for name, parameter in self.weights.items()}
