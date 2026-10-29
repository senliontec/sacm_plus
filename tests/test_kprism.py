"""K-Prism semantic-branch port tests."""

from __future__ import annotations

import argparse
import random

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from baseline_zoo.adapters.kprism import KPrismAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.models.kprism import build_kprism_semantic
from baseline_zoo.models.kprism.compat import c2_xavier_fill
from baseline_zoo.registry import load_models


@pytest.fixture
def sample(tmp_path):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    mask = np.zeros((60, 80), np.uint8)
    mask[20:40, 25:60] = 255
    image_path = tmp_path / 'case.png'
    mask_path = tmp_path / 'case_mask.png'
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)
    return Sample(image_path, mask_path, 'toy', 'case')


def test_registry_and_config():
    assert 'kprism' in available_adapters()
    adapter = load_adapter('kprism')
    assert isinstance(adapter, KPrismAdapter)
    config = load_models()['kprism']
    assert config.available
    assert config.defaults['epochs'] == 800
    assert 'zero 2-D dense prompt map' in ' '.join(adapter.deviations)


def test_train_batch_official_augmentation(sample):
    adapter = KPrismAdapter()
    adapter.device = torch.device('cpu')
    adapter._train_transform = None
    first = adapter.prepare_train_batch([sample], random.Random(7))
    second = adapter.prepare_train_batch([sample], random.Random(7))
    images, masks = first
    assert images.shape == (1, 3, 512, 512)
    assert masks.shape == (1, 1, 512, 512)
    assert images.dtype == torch.float32
    assert set(torch.unique(masks).tolist()) <= {0.0, 1.0}
    # Padding is normalized after zero-filling: (0 - 0.26) / 0.30.
    assert images.min().item() == pytest.approx(-0.26 / 0.30, abs=1e-5)
    assert torch.equal(images, second[0])
    assert torch.equal(masks, second[1])


def test_scheduler_matches_official_warmup_and_floor():
    adapter = KPrismAdapter()
    model = nn.Linear(1, 1)
    args = argparse.Namespace(
        epochs=20, kprism_warmup_epochs=10,
        lr=1e-4, weight_decay=0.0)
    optimizer = adapter.make_optimizer(model, args)
    scheduler = adapter.make_scheduler(optimizer, args)
    learning_rates = []
    for _ in range(20):
        learning_rates.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    assert learning_rates[0] == pytest.approx(1e-5)
    assert learning_rates[9] == pytest.approx(1e-4)
    assert learning_rates[-1] == pytest.approx(1e-5)


def test_c2_xavier_fill_matches_detectron2():
    torch.manual_seed(0)
    actual = nn.Conv2d(4, 6, 1)
    c2_xavier_fill(actual)
    torch.manual_seed(0)
    expected = nn.Conv2d(4, 6, 1)
    nn.init.kaiming_uniform_(expected.weight, a=1)
    nn.init.constant_(expected.bias, 0)
    assert torch.equal(actual.weight, expected.weight)
    assert torch.equal(actual.bias, expected.bias)


def test_non_official_input_size_is_rejected():
    with pytest.raises(ValueError, match='512'):
        build_kprism_semantic(image_size=256)


def test_model_forward_backward_and_zero_dense_prompt():
    model = build_kprism_semantic()
    captured = {}
    original_forward = model.sem_seg_head.forward

    def spy(features, point_feature=None, **kwargs):
        captured['point_feature'] = point_feature.detach().clone()
        captured['click_mode'] = kwargs.get('click_mode')
        return original_forward(features, point_feature=point_feature, **kwargs)

    model.sem_seg_head.forward = spy
    images = torch.randn(1, 3, 512, 512)
    outputs = model(images)
    assert outputs['pred_masks'].shape == (1, 1, 128, 128)
    assert len(outputs['aux_outputs']) == 5
    assert captured['click_mode'] == '0'
    assert captured['point_feature'].shape == (1, 1, 3, 512, 512)
    assert torch.count_nonzero(captured['point_feature']) == 0

    losses = model.criterion(outputs, [{
        'masks': torch.zeros(1, 512, 512)}])
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert model.backbone.inc[0].conv1.weight.grad is not None


def test_prediction_is_restored_to_gt_resolution(sample):
    class ConstantModel(nn.Module):
        def forward(self, images):
            return {'pred_masks': torch.full(
                (images.shape[0], 1, 32, 32), -10.0)}

    adapter = KPrismAdapter()
    prepared = adapter.prepare_eval_sample(sample, torch.device('cpu'))
    assert prepared['image'].shape == (1, 3, 512, 512)
    prediction = adapter.predict_sample(
        ConstantModel(), prepared, sample)
    assert prediction.shape == (60, 80)
    assert prediction.dtype == np.bool_
    assert not prediction.any()
