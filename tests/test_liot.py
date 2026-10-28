"""LIOT + IterNet tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.liot import LIOTAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.models.liot import (
    BRIGHT_FOREGROUND_DATASETS,
    DARK_FOREGROUND_DATASETS,
    Iternet,
    local_intensity_order,
    polarity_for_dataset,
)


def _reference_liot(image):
    """Literal translation of the release's numba loop."""
    image = np.asarray(image)
    gray = image[:, :, 1]
    padded = np.pad(gray, ((8, 8)), 'constant')
    weight, height = padded.shape
    codes = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.uint8)
    direction_map = np.zeros((gray.shape[0], gray.shape[1], 8),
                             dtype=np.uint8)
    for direction in range(4):
        for position in range(8):
            if direction == 0:
                neighbour = padded[position + 9:weight - 7 + position, 8:-8]
            elif direction == 1:
                neighbour = padded[position:-1 * (16 - position), 8:-8]
            elif direction == 2:
                neighbour = padded[8:-8, position + 9:height - 7 + position]
            else:
                neighbour = padded[8:-8, position:-1 * (16 - position)]
            bit = gray.astype(np.int64) - neighbour.astype(np.int64)
            bit[bit > 0] = 1
            bit[bit <= 0] = 0
            direction_map[:, :, position] = bit * 2 ** position
        codes[:, :, direction] = np.sum(direction_map, axis=2).astype(np.uint8)
    return codes


def _sample(tmp_path, width=150, height=120):
    image = tmp_path / 'DRIVE_0_image.png'
    mask = tmp_path / 'DRIVE_0_mask.png'
    image_array = np.random.default_rng(0).integers(
        0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(image_array).save(image)
    label = np.zeros((height, width), dtype=np.uint8)
    label[height // 2, 10:width - 10] = 255
    Image.fromarray(label, mode='L').save(mask)
    return Sample(image_path=image, mask_path=mask,
                  dataset='DRIVE', stem='DRIVE_0')


def _args(**overrides):
    values = dict(
        input_size=128, liot_polarity='auto', liot_no_augment=False,
        liot_random_init=False, liot_repeat=30, epochs=800, lr=1e-3,
        weight_decay=0.0, grad_clip=0.0, accum_steps=4, seed=42,
        data_root='data/sacm_3shot',
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_registry_and_metadata():
    assert 'liot' in available_adapters()
    adapter = load_adapter('liot')
    assert isinstance(adapter, LIOTAdapter)
    assert adapter.id == 'liot'
    assert adapter.input_size == 128
    assert adapter.official_lr == pytest.approx(1e-3)
    assert adapter.train_batch_size == 8
    assert adapter.accumulation_steps == 4
    assert any('polarity' in item.lower() for item in adapter.deviations)


def test_liot_matches_release_loop_and_polarity():
    image = np.random.default_rng(1).integers(
        0, 256, (21, 19, 3), dtype=np.uint8)
    expected = _reference_liot(image)
    actual = local_intensity_order(image)
    assert actual.shape == (21, 19, 4)
    assert actual.dtype == np.uint8
    assert np.array_equal(actual, expected)
    inverted = local_intensity_order(image, invert=True)
    inverted_image = 255 - image[:, :, 1][:, :, None].repeat(3, 2)
    assert np.array_equal(inverted, _reference_liot(inverted_image))
    assert polarity_for_dataset('DRIVE') is True
    assert polarity_for_dataset('CREMI') is False
    assert polarity_for_dataset('DRIVE', 'dark') is False
    assert polarity_for_dataset('CREMI', 'bright') is True
    assert 'CORN' in BRIGHT_FOREGROUND_DATASETS
    assert 'DeepCrack' in DARK_FOREGROUND_DATASETS


def test_model_forward_backward_and_parameters():
    model = Iternet(n_channels=4, n_classes=2)
    assert sum(p.numel() for p in model.parameters()) == 13_604_360
    logits = model(torch.randn(2, 4, 64, 64))
    assert logits.shape == (2, 2, 64, 64)
    assert torch.isfinite(logits).all()
    target = torch.zeros(2, 64, 64, dtype=torch.long)
    target[:, 30:34, 10:50] = 1
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_adapter_batch_loss_and_prediction(tmp_path):
    adapter = LIOTAdapter()
    adapter.args = _args()
    adapter.device = torch.device('cpu')
    adapter.input_size = 128
    adapter._polarity_mode = 'auto'
    sample = _sample(tmp_path)
    batch = adapter.prepare_train_batch([sample], None)
    assert batch['image'].shape == (1, 4, 128, 128)
    assert batch['gt'].shape == (1, 128, 128)
    assert batch['gt'].dtype == torch.long
    assert batch['image'].min() >= 0 and batch['image'].max() <= 1
    model = Iternet(n_channels=4, n_classes=2)
    loss = adapter.train_step(model, batch)
    assert torch.isfinite(loss)
    prepared = adapter.prepare_eval_sample(sample, torch.device('cpu'))
    assert prepared['image'].shape[0] == 4
    prediction = adapter.predict_sample(model.eval(), prepared, sample)
    assert prediction.shape == (120, 150)
    assert prediction.dtype == np.bool_


def test_optimizer_scheduler_and_config():
    model = Iternet(n_channels=4, n_classes=2)
    adapter = LIOTAdapter()
    adapter.args = _args()
    optimizer = adapter.make_optimizer(model, adapter.args)
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-3)
    scheduler = adapter.make_scheduler(optimizer, adapter.args)
    scheduler.step(100)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-3)
    adapter.args = _args(liot_repeat=2)
    sampler = adapter.make_train_sampler([0, 1], adapter.args)
    assert len(sampler) == 4
    config = adapter.save_config(adapter.args)
    restored = LIOTAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert target.input_size == 128
    assert target.liot_polarity == 'auto'
    assert target.liot_repeat == 2
    assert restored.input_size == 128
