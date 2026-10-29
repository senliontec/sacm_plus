"""SAM ViT-L + U-Net extension loss/config tests (no encoder forward)."""

import torch

from baseline_zoo.adapters.sam_vit_l_unet import dice_loss


def test_dice_loss_identical_masks_is_near_zero():
    target = torch.zeros(1, 1, 16, 16)
    target[..., 4:12, 4:12] = 1
    logits = torch.where(target > 0, 20.0, -20.0)
    loss = dice_loss(logits, target)
    assert torch.isfinite(loss)
    assert float(loss) < 1e-3
