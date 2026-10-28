"""CS-UNet tests (PyTorch rewrite of the TF notebook)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.csunet import CSUNetAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.models.csunet import (
    CSUNet,
    FSFBlock,
    GegenbauerNeuralBlock,
    PeriodicConvBlock,
    dice_loss,
    gegenbauer_polynomial,
)


def _sample(tmp_path, width=64, height=48):
    image = tmp_path / 'TEST_0_image.png'
    mask = tmp_path / 'TEST_0_mask.png'
    Image.new('RGB', (width, height), (120, 120, 120)).save(image)
    label = Image.new('L', (width, height), 0)
    for x in range(8, width - 8):
        label.putpixel((x, height // 2), 255)
    label.save(mask)
    return Sample(image_path=image, mask_path=mask,
                  dataset='TEST', stem='TEST_0')


def _args(**overrides):
    values = dict(
        input_size=512,
        csunet_filters=32,
        csunet_lr_decay=1e-6,
        csunet_random_init=False,
        epochs=800, lr=0.01, weight_decay=0.0, grad_clip=0.0,
        accum_steps=1, seed=42, data_root='data/sacm_3shot',
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_registry_and_metadata():
    assert 'csunet' in available_adapters()
    adapter = load_adapter('csunet')
    assert isinstance(adapter, CSUNetAdapter)
    assert adapter.id == 'csunet'
    assert adapter.input_size == 512
    assert adapter.official_lr == pytest.approx(0.01)
    assert adapter.official_weight_decay == 0.0
    assert any('placeholder' in item for item in adapter.deviations)


def test_gegenbauer_values_and_alpha():
    x = torch.tensor([0.5])
    alpha = torch.tensor([1.0])
    assert float(gegenbauer_polynomial(x, 0, alpha)) == pytest.approx(1.0)
    assert float(gegenbauer_polynomial(x, 1, alpha)) == pytest.approx(1.0)
    assert float(gegenbauer_polynomial(x, 2, alpha)) == pytest.approx(0.0)
    block = GegenbauerNeuralBlock(5)
    out = block(torch.randn(2, 8))
    assert out.shape == (2, 8)
    assert float(torch.nn.functional.softplus(block.alpha)) > 0


def test_block_shapes():
    x = torch.randn(1, 32, 16, 16)
    assert PeriodicConvBlock(32)(x).shape == x.shape
    fused = FSFBlock(32)(x, torch.randn(1, 32, 16, 16))
    assert fused.shape == (1, 32, 16, 16)


def test_model_forward_backward_and_parameters():
    model = CSUNet(input_size=64)
    assert sum(p.numel() for p in model.parameters()) == 7_240_385
    prediction = model(torch.randn(2, 3, 64, 64))
    assert prediction.shape == (2, 1, 64, 64)
    assert 0 <= float(prediction.min()) <= float(prediction.max()) <= 1
    gt = torch.zeros(2, 1, 64, 64)
    gt[:, :, 20:40, 10:50] = 1
    loss = dice_loss(prediction, gt)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_adapter_data_and_config(tmp_path):
    adapter = CSUNetAdapter()
    adapter.args = _args()
    adapter.device = torch.device('cpu')
    batch = adapter.prepare_train_batch(
        [_sample(tmp_path)], None)
    assert batch['image'].shape == (1, 3, 512, 512)
    assert batch['gt'].shape == (1, 1, 512, 512)
    assert batch['image'].min() >= 0 and batch['image'].max() <= 1
    config = adapter.save_config(adapter.args)
    restored = CSUNetAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert target.input_size == 512
    assert target.csunet_filters == 32
    assert target.csunet_lr_decay == pytest.approx(1e-6)


def test_optimizer_and_constant_lr_decay():
    model = CSUNet(input_size=64)
    adapter = CSUNetAdapter()
    adapter.args = _args()
    optimizer = adapter.make_optimizer(model, adapter.args)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]['momentum'] == pytest.approx(0.9)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.01)
    scheduler = adapter.make_scheduler(optimizer, adapter.args)
    assert scheduler is not None
    scheduler.step(1000)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.009)


def test_predict_sample_restores_shape():
    model = CSUNet(input_size=64).eval()
    adapter = CSUNetAdapter()
    adapter.args = _args()
    prepared = {
        'image': torch.rand(3, 80, 100),
        'size': (80, 100),
    }
    sample = Sample(image_path=None, mask_path=None,
                    dataset='TEST', stem='TEST_0')
    prediction = adapter.predict_sample(model, prepared, sample)
    assert prediction.shape == (80, 100)
    assert prediction.dtype == np.bool_
