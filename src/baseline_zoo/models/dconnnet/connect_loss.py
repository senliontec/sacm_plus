"""DconnNet connectivity target, loss and bilateral voting.

This is the release's ``connect_loss.py`` with three portability fixes:

* tensors are created on the input device instead of calling ``.cuda()``;
* ``connectivity_matrix`` keeps ``[B, 8*C, H, W]`` for batch size 1 (the
  release's ``squeeze()`` silently removed the batch dimension); and
* the edge-loss denominator is clamped, avoiding NaN on an empty prediction.

The connectivity bit order, CHASE single-class objective, and bilateral
voting equations are unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def connectivity_matrix(multimask, class_num):
    """Convert one-hot masks to the 8-direction connectivity target."""
    batch, _, rows, cols = multimask.shape
    device = multimask.device
    dtype = multimask.dtype
    conn = torch.zeros(
        [batch, class_num * 8, rows, cols], device=device, dtype=dtype)
    for i in range(class_num):
        mask = multimask[:, i, :, :].float()
        up = torch.zeros([batch, rows, cols], device=device)
        down = torch.zeros([batch, rows, cols], device=device)
        left = torch.zeros([batch, rows, cols], device=device)
        right = torch.zeros([batch, rows, cols], device=device)
        up_left = torch.zeros([batch, rows, cols], device=device)
        up_right = torch.zeros([batch, rows, cols], device=device)
        down_left = torch.zeros([batch, rows, cols], device=device)
        down_right = torch.zeros([batch, rows, cols], device=device)

        up[:, :rows - 1, :] = mask[:, 1:rows, :]
        down[:, 1:rows, :] = mask[:, 0:rows - 1, :]
        left[:, :, :cols - 1] = mask[:, :, 1:cols]
        right[:, :, 1:cols] = mask[:, :, :cols - 1]
        up_left[:, 0:rows - 1, 0:cols - 1] = mask[:, 1:rows, 1:cols]
        up_right[:, 0:rows - 1, 1:cols] = mask[:, 1:rows, 0:cols - 1]
        down_left[:, 1:rows, 0:cols - 1] = mask[:, 0:rows - 1, 1:cols]
        down_right[:, 1:rows, 1:cols] = mask[:, 0:rows - 1, 0:cols - 1]

        conn[:, (i * 8) + 0, :, :] = mask * down_right
        conn[:, (i * 8) + 1, :, :] = mask * down
        conn[:, (i * 8) + 2, :, :] = mask * down_left
        conn[:, (i * 8) + 3, :, :] = mask * right
        conn[:, (i * 8) + 4, :, :] = mask * left
        conn[:, (i * 8) + 5, :, :] = mask * up_right
        conn[:, (i * 8) + 6, :, :] = mask * up
        conn[:, (i * 8) + 7, :, :] = mask * up_left
    return conn.float()


def Bilateral_voting(c_map, hori_translation, verti_translation):
    """Convert 8-direction connectivity maps to a segmentation mask."""
    batch, class_num, channel, row, column = c_map.size()
    vote_out = torch.zeros(
        [batch, class_num, channel, row, column],
        device=c_map.device, dtype=c_map.dtype)

    right = torch.bmm(
        c_map[:, :, 4].contiguous().view(-1, row, column),
        hori_translation.view(-1, column, column)).view(
            batch, class_num, row, column)
    left = torch.bmm(
        c_map[:, :, 3].contiguous().view(-1, row, column),
        hori_translation.transpose(3, 2).view(-1, column, column)).view(
            batch, class_num, row, column)
    left_bottom = torch.bmm(
        verti_translation.transpose(3, 2).view(-1, row, row),
        c_map[:, :, 5].contiguous().view(-1, row, column)).view(
            batch, class_num, row, column)
    left_bottom = torch.bmm(
        left_bottom.view(-1, row, column),
        hori_translation.transpose(3, 2).view(-1, column, column)).view(
            batch, class_num, row, column)
    right_above = torch.bmm(
        verti_translation.view(-1, row, row),
        c_map[:, :, 2].contiguous().view(-1, row, column)).view(
            batch, class_num, row, column)
    right_above = torch.bmm(
        right_above.view(-1, row, column),
        hori_translation.view(-1, column, column)).view(
            batch, class_num, row, column)
    left_above = torch.bmm(
        verti_translation.view(-1, row, row),
        c_map[:, :, 0].contiguous().view(-1, row, column)).view(
            batch, class_num, row, column)
    left_above = torch.bmm(
        left_above.view(-1, row, column),
        hori_translation.transpose(3, 2).view(-1, column, column)).view(
            batch, class_num, row, column)
    bottom = torch.bmm(
        verti_translation.transpose(3, 2).view(-1, row, row),
        c_map[:, :, 6].contiguous().view(-1, row, column)).view(
            batch, class_num, row, column)
    up = torch.bmm(
        verti_translation.view(-1, row, row),
        c_map[:, :, 1].contiguous().view(-1, row, column)).view(
            batch, class_num, row, column)
    right_bottom = torch.bmm(
        verti_translation.transpose(3, 2).view(-1, row, row),
        c_map[:, :, 7].contiguous().view(-1, row, column)).view(
            batch, class_num, row, column)
    right_bottom = torch.bmm(
        right_bottom.view(-1, row, column),
        hori_translation.view(-1, column, column)).view(
            batch, class_num, row, column)

    vote_out[:, :, 0] = c_map[:, :, 0] * right_bottom
    vote_out[:, :, 1] = c_map[:, :, 1] * bottom
    vote_out[:, :, 2] = c_map[:, :, 2] * left_bottom
    vote_out[:, :, 3] = c_map[:, :, 3] * right
    vote_out[:, :, 4] = c_map[:, :, 4] * left
    vote_out[:, :, 5] = c_map[:, :, 5] * right_above
    vote_out[:, :, 6] = c_map[:, :, 6] * up
    vote_out[:, :, 7] = c_map[:, :, 7] * left_above

    pred_mask, _ = torch.max(vote_out, dim=2)
    return pred_mask, vote_out


class dice_loss(nn.Module):
    """Release soft-Dice term (per-image mean, smooth=1e-4)."""

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.0001
        i = torch.sum(y_true, dim=(1, 2))
        j = torch.sum(y_pred, dim=(1, 2))
        intersection = torch.sum(y_true * y_pred, dim=(1, 2))
        return (2.0 * intersection + smooth) / (i + j + smooth)

    def soft_dice_loss(self, y_pred, y_true):
        return (1.0 - self.soft_dice_coeff(y_pred, y_true)).mean()

    def __call__(self, y_pred, y_true):
        return self.soft_dice_loss(y_pred, y_true)


class connect_loss(nn.Module):
    """Release connectivity objective (binary/CHASE and multi-class paths)."""

    def __init__(self, num_class, hori_translation, verti_translation,
                 dataset='chase'):
        super().__init__()
        self.BCEloss = nn.BCELoss(reduction='none')
        self.dice_loss = dice_loss()
        self.num_class = int(num_class)
        self.dataset = str(dataset)
        self.verti_translation = verti_translation
        self.hori_translation = hori_translation

    def edge_loss(self, vote_out, edge):
        pred_mask_min, _ = torch.min(vote_out, dim=2)
        pred_mask_min = pred_mask_min * edge
        minloss = self.BCEloss(
            pred_mask_min, torch.full_like(pred_mask_min, 0))
        # The release divides by the foreground sum without a guard.  A
        # fully empty prediction would otherwise produce NaN.
        return minloss.sum() / pred_mask_min.sum().clamp_min(1e-8)

    def forward(self, c_map, target):
        if self.num_class == 1:
            return self.single_class_forward(c_map, target)
        return self.multi_class_forward(c_map, target)

    def multi_class_forward(self, c_map, target):
        target = target.to(device=c_map.device).long()
        batch_num = c_map.shape[0]
        onehotmask = F.one_hot(target, self.num_class)
        onehotmask = onehotmask.permute(0, 3, 1, 2).float()

        con_target = connectivity_matrix(onehotmask, self.num_class)
        hori_translation = self.hori_translation.repeat(
            batch_num, 1, 1, 1)
        verti_translation = self.verti_translation.repeat(
            batch_num, 1, 1, 1)

        class_conn = con_target.view(
            [c_map.shape[0], self.num_class, 8, c_map.shape[2],
             c_map.shape[3]])
        sum_conn = torch.sum(class_conn, dim=2)
        edge = torch.where(
            (sum_conn < 8) & (sum_conn > 0),
            torch.full_like(sum_conn, 1), torch.full_like(sum_conn, 0))

        class_pred = c_map.view(
            [c_map.shape[0], self.num_class, 8, c_map.shape[2],
             c_map.shape[3]])
        final_pred, vote_out = Bilateral_voting(
            class_pred, hori_translation, verti_translation)
        _, bicon_map = Bilateral_voting(
            torch.sigmoid(class_pred), hori_translation,
            verti_translation)
        bicon_map = bicon_map.view(con_target.shape)

        edge_l = self.edge_loss(torch.sigmoid(vote_out), edge)
        pred = F.softmax(final_pred, dim=1)

        dice_l = 0
        for j in range(1, self.num_class):
            dice_l += self.dice_loss(pred[:, j, :, :], onehotmask[:, j])

        ce_loss = F.cross_entropy(final_pred, target)
        conn_l = self.BCEloss(torch.sigmoid(c_map), con_target).mean()
        bicon_l = self.BCEloss(bicon_map, con_target).mean()
        return ce_loss + conn_l + edge_l + 0.2 * bicon_l + dice_l

    def single_class_forward(self, c_map, target):
        batch_num = c_map.shape[0]
        target = target.float()

        con_target = connectivity_matrix(target, self.num_class)
        hori_translation = self.hori_translation.repeat(
            batch_num, 1, 1, 1)
        verti_translation = self.verti_translation.repeat(
            batch_num, 1, 1, 1)

        c_map = torch.sigmoid(c_map)
        class_conn = con_target.view(
            [c_map.shape[0], self.num_class, 8, c_map.shape[2],
             c_map.shape[3]])
        sum_conn = torch.sum(class_conn, dim=2)
        edge = torch.where(
            (sum_conn < 8) & (sum_conn > 0),
            torch.full_like(sum_conn, 1), torch.full_like(sum_conn, 0))

        class_pred = c_map.view(
            [c_map.shape[0], self.num_class, 8, c_map.shape[2],
             c_map.shape[3]])
        pred, bicon_map = Bilateral_voting(
            class_pred, hori_translation, verti_translation)

        edge_l = self.edge_loss(bicon_map, edge)
        dice_l = self.dice_loss(pred[:, 0], target[:, 0])
        bce_loss = self.BCEloss(pred, target).mean()
        conn_l = self.BCEloss(c_map, con_target).mean()

        if self.dataset == 'chase':
            return bce_loss + conn_l + edge_l + dice_l
        bicon_l = self.BCEloss(
            bicon_map.squeeze(1), con_target).mean()
        return bce_loss + conn_l + edge_l + 0.2 * bicon_l + dice_l
