"""Fast DSCNet adapter tests (no full-size training)."""

import random
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from baseline_zoo.adapters.dscnet import DSCNetAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.models.dscnet import build_dscnet, cross_loss


def _sample(tmp_path):
    image = tmp_path / 'TEST_0_image.png'
    mask = tmp_path / 'TEST_0_mask.png'
    Image.new('RGB', (32, 24), (120, 120, 120)).save(image)
    label = Image.new('L', (32, 24), 0)
    for x in range(16):
        for y in range(24):
            label.putpixel((x, y), 255)
    label.save(mask)
    return Sample(image_path=image, mask_path=mask,
                  dataset='TEST', stem='TEST_0')


class _DummyModel(nn.Module):
    def forward(self, x):
        return torch.ones_like(x) * 0.6


def test_registry_and_metadata():
    assert 'dscnet' in available_adapters()
    adapter = load_adapter('dscnet')
    assert isinstance(adapter, DSCNetAdapter)
    assert adapter.id == 'dscnet'
    assert adapter.input_size == 224
    assert adapter.train_batch_size == 1
    assert 'DSCNet' in adapter.official_source


def test_tiny_forward_and_loss():
    model = build_dscnet(basic_layer=4)
    images = torch.randn(1, 1, 64, 64)
    masks = torch.zeros(1, 1, 64, 64)
    masks[..., :32] = 1
    pred = model(images)
    assert pred.shape == (1, 1, 64, 64)
    loss = cross_loss()(masks, pred)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_train_step_adds_soft_dice():
    adapter = DSCNetAdapter()
    adapter.args = SimpleNamespace(dsc_dice_weight=1.0)

    class ProbModel(nn.Module):
        def forward(self, x):
            return torch.full_like(x, 0.6)

    images = torch.zeros(1, 1, 8, 8)
    masks = torch.zeros(1, 1, 8, 8)
    masks[..., :4] = 1
    pred = torch.full_like(images, 0.6)
    expected_bce = adapter._criterion(masks, pred)
    expected_dice = adapter._soft_dice(pred, masks)
    loss = adapter.train_step(ProbModel(), (images, masks))
    assert torch.allclose(loss, expected_bce + expected_dice)

    adapter.args.dsc_dice_weight = 0.0
    loss = adapter.train_step(ProbModel(), (images, masks))
    assert torch.allclose(loss, expected_bce)


def test_preprocess_shapes(tmp_path):
    adapter = DSCNetAdapter()
    adapter.input_size = 224
    adapter._mean = 0.0
    adapter._std = 1.0
    image, mask = adapter._preprocess(
        _sample(tmp_path), random.Random(0), train=True)
    assert image.shape == (1, 224, 224)
    assert mask.shape == (1, 224, 224)
    assert set(mask.unique().tolist()) <= {0, 1}
    assert torch.isfinite(image).all()


def test_patch_prediction_and_config(tmp_path):
    adapter = DSCNetAdapter()
    adapter.input_size = 224
    adapter._mean = 0.0
    adapter._std = 1.0
    adapter.args = SimpleNamespace(dsc_stride=112)
    prepared = {
        'image': torch.zeros(1, 1, 64, 48),
        'original_size': (64, 48),
    }
    pred = adapter.predict_sample(_DummyModel(), prepared, _sample(tmp_path))
    assert pred.shape == (64, 48)
    assert pred.dtype == bool
    assert pred.all()

    save_args = SimpleNamespace(dsc_basic_layer=16, dsc_kernel_size=9,
                                dsc_stride=112)
    config = adapter.save_config(save_args)
    assert config['input_size'] == 224
    assert config['mean'] == 0.0
    restore = SimpleNamespace()
    adapter.restore_config(restore, config)
    assert restore.input_size == 224
    assert restore.dsc_basic_layer == 16


def test_patch_prediction_rejects_stride_gap(tmp_path):
    adapter = DSCNetAdapter()
    adapter.input_size = 224
    adapter.args = SimpleNamespace(dsc_stride=300)
    prepared = {
        'image': torch.zeros(1, 1, 64, 48),
        'original_size': (64, 48),
    }
    with pytest.raises(SystemExit, match='dsc-stride'):
        adapter.predict_sample(_DummyModel(), prepared, _sample(tmp_path))
