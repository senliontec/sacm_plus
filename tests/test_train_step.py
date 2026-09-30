"""End-to-end training-step test (slow, skipped by default).

Builds the model (vit_b for CPU speed; identical architecture switches),
runs ONE full training step with all loss terms — main DiceBCE, Stage-1
deep supervision, IoU-head MSE and soft-clDice — and asserts the loss is
finite and gradients reach at least one parameter. This catches
loss-assembly breakage that the forward-only smoke test cannot.

Run with: make test-slow
"""

import pytest
import torch
import torch.nn.functional as F

from models.sam import build_sam_vit_b
from core.io import get_prompt_embeddings, predict_all
from core.losses import DiceBCELoss, SoftclDiceLoss


@pytest.mark.slow
@pytest.mark.skipif("not config.getoption('--run-slow')")
def test_single_training_step():
    sam = build_sam_vit_b(
        checkpoint=None,
        use_adapter=True,
        adapter_dim_ratio=0.1,
        use_geo_i=True,
        use_geo_e=True,
        use_coarse_to_fine=True,
        use_fusion_v2=True,
        use_multi_depth=True,
    )

    # Freezing protocol identical to train_sam.py
    for name, param in sam.image_encoder.named_parameters():
        if 'adapter' not in name:
            param.requires_grad = False
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in sam.parameters() if p.requires_grad], lr=1e-5
    )

    images = sam.preprocess(torch.randn(1, 3, 1024, 1024))
    masks = (torch.rand(1, 1, 1024, 1024) > 0.9).float()

    sparse, dense = get_prompt_embeddings(sam, 1)
    m2, iou_pred, m1 = predict_all(sam, images, sparse, dense, return_stage1=True)

    criterion = DiceBCELoss()
    soft_cl = SoftclDiceLoss()
    m2_1024 = F.interpolate(m2, size=(1024, 1024), mode='bilinear', align_corners=False)
    m1_1024 = F.interpolate(m1, size=(1024, 1024), mode='bilinear', align_corners=False)

    loss_main = criterion(m2_1024[:, 0:1], masks)
    loss_ds = sum(criterion(m1_1024[:, k:k + 1], masks) for k in range(4)) / 4
    with torch.no_grad():
        m2_bin = (m2_1024 > 0).float()
        gt_bin = (masks > 0.5).float()
        inter = (m2_bin * gt_bin).sum(dim=(2, 3))
        union = m2_bin.sum(dim=(2, 3)) + gt_bin.sum(dim=(2, 3)) - inter
        iou_gt = inter / (union + 1e-6)
    loss_iou = F.mse_loss(iou_pred, iou_gt)
    loss_cl = soft_cl(m2_1024[:, 0:1], masks)

    loss = loss_main + 0.3 * loss_ds + 1.0 * loss_iou + 0.5 * loss_cl

    assert torch.isfinite(loss), f"non-finite loss: {loss.item()}"
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in sam.parameters() if p.requires_grad
    ), "no trainable parameter received a gradient"
    optimizer.step()
