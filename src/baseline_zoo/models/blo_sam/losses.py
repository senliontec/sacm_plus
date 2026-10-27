"""Minimal BLO-SAM Dice loss.

Upstream ``utils.py`` also imports SimpleITK/medpy/imageio for unrelated
evaluation helpers.  Only the class-wise Dice loss is needed by the bilevel
prompt optimizer, so it is kept here without those dependencies.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = int(n_classes)

    def _one_hot_encoder(self, input_tensor):
        tensors = [
            (input_tensor == index).unsqueeze(1)
            for index in range(self.n_classes)
        ]
        return torch.cat(tensors, dim=1).float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        return 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        if inputs.size() != target.size():
            raise ValueError(
                f'predict {inputs.size()} & target {target.size()} '
                'shape do not match')
        loss = 0.0
        for index in range(self.n_classes):
            loss = loss + self._dice_loss(
                inputs[:, index], target[:, index]) * weight[index]
        return loss / self.n_classes
