"""Paper-profile BCU-Net tests: MRL, bilinear Up, fusion, augmentation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from baseline_zoo.data import Sample
from baseline_zoo.models.bcunet import (
    BCUNet,
    BCUNetTrainDataset,
    RecallCrossEntropy,
)
from baseline_zoo.models.bcunet.unet_parts import Up


def test_recall_ce_matches_paper_equation():
    logits = torch.tensor([[
        [[2.0, -1.0], [0.0, 1.0]],
        [[1.0, 2.0], [-2.0, 0.0]],
    ]])
    target = torch.tensor([[[0, 1], [1, 0]]])
    criterion = RecallCrossEntropy(n_classes=2)
    actual = criterion(logits, target)

    pred = logits.argmax(1)
    gt = torch.bincount(target.flatten(), minlength=2).float()
    fn = torch.bincount(
        target[pred != target], minlength=2).float()
    weight = fn / gt.clamp(min=1)
    ce = F.cross_entropy(logits, target, reduction='none')[0]
    expected = (weight[target[0]] * ce).mean()
    assert actual.item() == pytest.approx(expected.item(), rel=1e-6)


def test_recall_ce_is_device_safe_and_filters_ignore_index():
    logits = torch.randn(1, 2, 3, 3)
    target = torch.tensor([[[0, 1, 255], [1, 0, 1], [0, 0, 1]]])
    loss = RecallCrossEntropy(n_classes=2)(logits, target)
    assert torch.isfinite(loss)
    if torch.cuda.is_available():
        gpu_loss = RecallCrossEntropy(n_classes=2)(
            logits.cuda(), target.cuda())
        assert gpu_loss.item() == pytest.approx(loss.item(), rel=1e-5)


def test_bilinear_up_uses_concatenated_channel_width():
    up = Up(512, 256, bilinear=True)
    x1 = torch.randn(1, 512, 8, 8)
    x2 = torch.randn(1, 256, 16, 16)
    output = up(x1, x2)
    assert output.shape == (1, 256, 16, 16)


def test_model_uses_paper_average_fusion():
    model = BCUNet(
        n_channels=3, n_classes=2, bilinear=True, pretrained=False,
        in_22k=False, fusion_unet=0.5, fusion_convnext=0.5)
    with torch.no_grad():
        out1, out2, fused = model(torch.randn(1, 3, 64, 64))
    assert fused.shape == out1.shape == out2.shape == (1, 2, 64, 64)
    assert torch.allclose(fused, 0.5 * out1 + 0.5 * out2, atol=1e-6)


def test_offline_18_variant_dataset_is_deterministic(tmp_path):
    image = np.random.default_rng(0).integers(
        0, 255, (64, 80, 3), dtype=np.uint8)
    mask = np.zeros((64, 80), np.uint8)
    mask[20:50, 10:70] = 255
    image_path = tmp_path / 'case.png'
    mask_path = tmp_path / 'case_mask.png'
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)
    sample = Sample(image_path, mask_path, 'DRIVE', 'case')
    dataset = BCUNetTrainDataset(
        [sample], input_size=512, variants=18, seed=0)
    assert len(dataset) == 18
    first = dataset[0]
    second = dataset[1]
    assert first[0].shape == (3, 512, 512)
    assert first[1].shape == (512, 512)
    assert first[1].dtype == torch.int64
    assert set(torch.unique(first[1]).tolist()) <= {0, 1}
    assert torch.equal(dataset[1][0], second[0])
    assert not torch.equal(first[0], second[0])
    # Variant 5 is the horizontal flip.
    assert torch.equal(dataset[5][1], torch.flip(first[1], dims=[-1]))


def test_augmentation_scope_is_retinal_by_default(tmp_path):
    image = np.random.default_rng(1).integers(
        0, 255, (64, 80, 3), dtype=np.uint8)
    mask = np.zeros((64, 80), np.uint8)
    mask[20:50, 10:70] = 255
    image_path = tmp_path / 'case.png'
    mask_path = tmp_path / 'case_mask.png'
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)
    retinal = Sample(image_path, mask_path, 'DRIVE', 'case')
    other = Sample(image_path, mask_path, 'CORN', 'case')
    paper_scope = BCUNetTrainDataset(
        [retinal, other], input_size=512, variants=18, seed=0,
        augment_scope='retinal')
    # Retinal samples use the full variants.
    assert not torch.equal(paper_scope[0][0], paper_scope[1][0])
    # Non-retinal samples repeat identity: items 18 and 19 are identical.
    assert torch.equal(paper_scope[18][0], paper_scope[19][0])
    all_scope = BCUNetTrainDataset(
        [other], input_size=512, variants=18, seed=0,
        augment_scope='all')
    assert not torch.equal(all_scope[0][0], all_scope[1][0])
