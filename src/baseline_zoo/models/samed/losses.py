"""Official SAMed Dice loss (MIT, from 3rd/SAMed/utils.py)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ForegroundDiceLoss(nn.Module):
    """Soft Dice loss on the foreground channel only.

    SAMed's official class-wise Dice includes the background class.  For a
    binary curvilinear task with a very small foreground (0.17%-8% of the
    image), averaging background and foreground equally makes an
    all-background prediction too cheap.  This loss keeps the semantic
    foreground objective but removes that background shortcut.
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits, target, softmax=True):
        if softmax:
            prob = torch.softmax(logits, dim=1)[:, 1]
        else:
            prob = logits[:, 1]
        target = target.float()
        prob = prob.flatten(1)
        target = target.flatten(1)
        intersection = (prob * target).sum(dim=1)
        denominator = prob.square().sum(dim=1) + target.square().sum(dim=1)
        dice = ((2 * intersection + self.smooth)
                / (denominator + self.smooth))
        return (1 - dice).mean()


class SoftClDiceLoss(nn.Module):
    """Differentiable centreline Dice loss for thin curvilinear structures.

    SAMed's original objective is region Dice/CE, which is dominated by the
    background on 0.17%-8% foreground.  Soft-clDice adds a topology term
    computed from iterative soft skeletons, a task adaptation for vessels,
    cracks and membranes.  The skeletonization follows the official clDice
    implementation (Shit et al., CVPR 2021): residual accumulation from
    ``relu(img - soft_open(img))`` over iterative erosions.
    """

    def __init__(self, iters: int = 10, smooth: float = 1e-6):
        super().__init__()
        self.iters = max(int(iters), 1)
        self.smooth = float(smooth)

    @staticmethod
    def _soft_erode(img):
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    @staticmethod
    def _soft_dilate(img):
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

    def _soft_open(self, img):
        return self._soft_dilate(self._soft_erode(img))

    def _soft_skel(self, img):
        img1 = self._soft_open(img)
        skel = F.relu(img - img1)
        for _ in range(self.iters):
            img = self._soft_erode(img)
            img1 = self._soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, logits, target, softmax=True):
        if softmax:
            pred = torch.softmax(logits, dim=1)[:, 1]
        else:
            pred = logits[:, 1]
        target = target.float()
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        skel_pred = self._soft_skel(pred)
        skel_true = self._soft_skel(target)
        dims = (1, 2, 3)
        tprec = (((skel_pred * target).sum(dims) + self.smooth)
                 / (skel_pred.sum(dims) + self.smooth))
        tsens = (((skel_true * pred).sum(dims) + self.smooth)
                 / (skel_true.sum(dims) + self.smooth))
        cl = ((2 * tprec * tsens + self.smooth)
              / (tprec + tsens + self.smooth))
        loss = 1 - cl
        empty = target.sum(dims) <= 0
        return torch.where(empty, torch.zeros_like(loss), loss).mean()


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
