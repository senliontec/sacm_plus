"""Official SAMed Dice loss (MIT, from 3rd/SAMed/utils.py)."""

from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    """Class-wise soft Dice loss on logits/softmax probabilities."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            tensor_list.append(
                (input_tensor == i).unsqueeze(1))
        return torch.cat(tensor_list, dim=1).float()

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
        for i in range(self.n_classes):
            loss = loss + self._dice_loss(
                inputs[:, i], target[:, i]) * weight[i]
        return loss / self.n_classes
