"""Regression tests for the U-Net baseline's official Dice loss."""

import torch

from baseline_zoo.adapters.unet import dice_loss


def test_dice_loss_is_global_over_batch():
    # One empty image and one full image.  The official milesial loss is
    # computed once over the whole B,H,W tensor, not as a per-image mean.
    logits = torch.zeros(2, 1, 1, 1, requires_grad=True)
    target = torch.tensor([0.0, 1.0]).view(2, 1, 1, 1)

    loss = dice_loss(logits, target)
    expected = 1.0 - (1.0 + 1e-6) / (2.0 + 1e-6)
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-8)

    # The old per-image reduction produced a different value on this batch.
    prob = torch.sigmoid(logits.detach())
    inter = 2.0 * (prob * target).sum(dim=(1, 2, 3))
    sets_sum = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    per_image = 1.0 - ((inter + 1e-6) / (sets_sum + 1e-6)).mean()
    assert not torch.allclose(loss.detach(), per_image, atol=1e-6)

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
