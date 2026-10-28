"""DconnNet tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.dconnnet import DconnNetAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.models.dconnnet import (
    Bilateral_voting,
    DconnNet,
    connect_loss,
    connectivity_matrix,
    dice_loss,
)


def _translations(size=32):
    hori = torch.zeros(1, 1, size, size)
    verti = torch.zeros(1, 1, size, size)
    for index in range(size - 1):
        hori[:, :, index, index + 1] = 1
        verti[:, :, index, index + 1] = 1
    return hori, verti


def _sample(tmp_path, width=100, height=80):
    image = tmp_path / 'DRIVE_0_image.png'
    mask = tmp_path / 'DRIVE_0_mask.png'
    array = np.random.default_rng(0).integers(
        0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(array).save(image)
    label = np.zeros((height, width), dtype=np.uint8)
    label[height // 2, 10:width - 10] = 255
    Image.fromarray(label, mode='L').save(mask)
    return Sample(image_path=image, mask_path=mask,
                  dataset='DRIVE', stem='DRIVE_0')


def _args(**overrides):
    values = dict(
        input_size=64, dconnnet_backbone_ckpt='missing.pth',
        dconnnet_no_augment=False, dconnnet_random_init=False,
        epochs=800, lr=3.8e-3, weight_decay=0.0, grad_clip=0.0,
        accum_steps=1, seed=42, data_root='data/sacm_3shot',
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_registry_and_metadata():
    assert 'dconnnet' in available_adapters()
    adapter = load_adapter('dconnnet')
    assert isinstance(adapter, DconnNetAdapter)
    assert adapter.id == 'dconnnet'
    assert adapter.input_size == 960
    assert adapter.official_lr == pytest.approx(3.8e-3)
    assert adapter.train_batch_size == 4
    assert any('bilateral' in item.lower() for item in adapter.deviations)


def test_connectivity_matrix_keeps_batch_and_layout():
    mask = torch.zeros(1, 1, 5, 5)
    mask[0, 0, 2, 1:4] = 1
    conn = connectivity_matrix(mask, 1)
    assert conn.shape == (1, 8, 5, 5)
    assert conn.dtype == torch.float32
    # The center pixel's right/left directions are set for a horizontal line.
    assert float(conn[0, 3, 2, 2]) == 1.0
    assert float(conn[0, 4, 2, 2]) == 1.0
    assert float(conn.sum()) > 0


def test_bilateral_voting_and_loss_batch_one():
    hori, verti = _translations(32)
    mask = torch.zeros(1, 1, 32, 32)
    mask[0, 0, 15, 8:24] = 1
    conn = connectivity_matrix(mask, 1)
    logits = (conn * 2.0 - 1.0).requires_grad_(True)
    pred, votes = Bilateral_voting(
        torch.sigmoid(logits.view(1, 1, 8, 32, 32)), hori, verti)
    assert pred.shape == (1, 1, 32, 32)
    assert votes.shape == (1, 1, 8, 32, 32)
    criterion = connect_loss(1, hori, verti, dataset='chase')
    loss = criterion(logits, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(dice_loss()(pred[:, 0], mask[:, 0]))


def test_model_forward_parameters_and_backward():
    model = DconnNet(num_class=1)
    assert sum(p.numel() for p in model.parameters()) == 28_382_912
    logits, auxiliary = model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 8, 64, 64)
    assert auxiliary.shape == (1, 8, 64, 64)
    assert torch.isfinite(logits).all()
    target = torch.zeros(1, 1, 64, 64)
    target[:, :, 30:34, 10:50] = 1
    hori, verti = _translations(64)
    criterion = connect_loss(1, hori, verti, dataset='chase')
    loss = criterion(logits, target) + 0.3 * criterion(auxiliary, target)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_adapter_batch_loss_prediction_and_config(tmp_path):
    adapter = DconnNetAdapter()
    adapter.device = torch.device('cpu')
    adapter.args = _args()
    model = adapter.build_model(adapter.args)
    batch = adapter.prepare_train_batch([_sample(tmp_path)], None)
    assert batch['image'].shape == (1, 3, 64, 64)
    assert batch['gt'].shape == (1, 1, 64, 64)
    assert batch['image'].min() >= -1.6
    assert batch['image'].max() <= 1.6
    loss = adapter.train_step(model, batch)
    assert torch.isfinite(loss)
    sample = _sample(tmp_path)
    prepared = adapter.prepare_eval_sample(sample, torch.device('cpu'))
    assert prepared['image'].shape == (3, 64, 64)
    prediction = adapter.predict_sample(model.eval(), prepared, sample)
    assert prediction.shape == (80, 100)
    assert prediction.dtype == np.bool_
    config = adapter.save_config(adapter.args)
    restored = DconnNetAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert target.input_size == 64
    assert restored.input_size == 64


def test_optimizer_and_polynomial_scheduler():
    model = DconnNet(num_class=1)
    adapter = DconnNetAdapter()
    adapter.args = _args(epochs=100)
    optimizer = adapter.make_optimizer(model, adapter.args)
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(3.8e-3)
    scheduler = adapter.make_scheduler(optimizer, adapter.args)
    scheduler.step(50)
    expected = 3.8e-3 * (0.5 ** 0.9)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(expected)
