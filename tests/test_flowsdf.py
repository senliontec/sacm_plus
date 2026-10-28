"""Fast FlowSDF tests (SDF targets, sampler, tiling, adapter plumbing)."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.flowsdf import FlowSDFAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.models.flowsdf import (
    build_flowsdf,
    euler_sample,
    predict_tiled,
    predict_whole,
    signed_distance_map,
    tile_starts,
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
        input_size=128,
        flowsdf_model_channels=128,
        flowsdf_rrdb_blocks=12,
        flowsdf_sigma_min=1e-5,
        flowsdf_ema_decay=0.999,
        flowsdf_ode_steps=4,
        flowsdf_k_samples=4,
        flowsdf_val_k_samples=1,
        flowsdf_sdf_delta=15.0,
        flowsdf_threshold=0.03,
        flowsdf_tile_stride=64,
        flowsdf_tile_chunk=4,
        flowsdf_no_augment=False,
        lr=1e-4,
        weight_decay=0.0,
        grad_clip=1.0,
        accum_steps=1,
        seed=42,
        epochs=800,
        data_root='data/sacm_3shot',
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_registry_and_metadata():
    assert 'flowsdf' in available_adapters()
    adapter = load_adapter('flowsdf')
    assert isinstance(adapter, FlowSDFAdapter)
    assert adapter.id == 'flowsdf'
    assert adapter.input_size == 128
    assert adapter.official_lr == pytest.approx(1e-4)
    assert adapter.selection == 'dice'
    assert any('on the fly' in item for item in adapter.deviations)


def test_signed_distance_convention():
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:7] = True
    sdf = signed_distance_map(mask, delta=15.0)
    assert sdf.dtype == np.float32
    assert sdf[4, 4] < 0
    assert sdf[2, 2] < 0
    assert sdf[0, 0] > 0
    assert sdf[0, 0] > sdf[1, 1] > 0
    assert sdf.min() >= -1 and sdf.max() <= 1
    np.testing.assert_allclose(sdf, signed_distance_map(mask, 15.0))
    assert np.all(signed_distance_map(np.zeros((4, 4), bool)) == 1.0)
    assert np.all(signed_distance_map(np.ones((4, 4), bool)) == -1.0)


def test_tile_starts_cover_full_extent():
    assert tile_starts(100, 128, 64) == [0]
    assert tile_starts(128, 128, 64) == [0]
    starts = tile_starts(300, 128, 64)
    assert starts[0] == 0
    assert starts[-1] == 300 - 128
    assert all(b - a <= 64 for a, b in zip(starts, starts[1:]))


def test_small_model_forward_and_sampler():
    model = build_flowsdf(model_channels=32, rrdb_blocks=1).eval()
    x = torch.randn(2, 1, 64, 64)
    t = torch.rand(2, 1)
    image = torch.rand(2, 3, 64, 64) * 2 - 1
    with torch.no_grad():
        out = model(x, t, img_cond=image)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    noise = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        first = euler_sample(model, torch.rand(2, 3, 32, 32) * 2 - 1,
                             noise, nfe=2)
        second = euler_sample(model, torch.rand(2, 3, 32, 32) * 2 - 1,
                              noise, nfe=2)
    assert first.shape == noise.shape
    assert second.shape == noise.shape
    assert torch.isfinite(first).all()


def test_tiled_prediction_preserves_resolution():
    model = build_flowsdf(model_channels=32, rrdb_blocks=1).eval()
    image = torch.rand(3, 100, 140) * 2 - 1
    with torch.no_grad():
        sdf = predict_tiled(
            model, image, patch=32, stride=16, k_samples=2, nfe=2,
            tile_chunk=3, seed=0)
        tiny = predict_tiled(
            model, image[:, :20, :30], patch=32, stride=16, k_samples=1,
            nfe=1, tile_chunk=1, seed=0)
    assert sdf.shape == (100, 140)
    assert tiny.shape == (20, 30)
    assert torch.isfinite(sdf).all()


def test_whole_image_prediction_preserves_resolution():
    model = build_flowsdf(model_channels=32, rrdb_blocks=1).eval()
    image = torch.rand(3, 100, 140) * 2 - 1
    with torch.no_grad():
        sdf = predict_whole(
            model, image, size=32, k_samples=2, nfe=2, seed=0)
    assert sdf.shape == (100, 140)
    assert torch.isfinite(sdf).all()


def test_train_batch_shapes_and_ranges(tmp_path):
    adapter = FlowSDFAdapter()
    adapter.args = _args()
    adapter.device = torch.device('cpu')
    image, sdf = adapter._train_pair(_sample(tmp_path), random.Random(0))
    assert image.shape == (3, 128, 128)
    assert sdf.shape == (1, 128, 128)
    assert image.min() >= -1.0001 and image.max() <= 1.0001
    assert sdf.min() >= -1 and sdf.max() <= 1
    assert sdf.min() < 0


def test_train_step_loss_and_gradients():
    adapter = FlowSDFAdapter()
    adapter.args = _args()
    adapter.device = torch.device('cpu')
    model = build_flowsdf(model_channels=32, rrdb_blocks=1)
    batch = {
        'image': torch.rand(2, 3, 64, 64) * 2 - 1,
        'sdf': torch.rand(2, 1, 64, 64) * 2 - 1,
    }
    loss = adapter.train_step(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_ema_state_roundtrip():
    adapter = FlowSDFAdapter()
    adapter.args = _args()
    adapter.device = torch.device('cpu')
    model = build_flowsdf(model_channels=32, rrdb_blocks=1)
    ema = adapter._ensure_ema(model)
    before = [p.detach().clone() for p in ema.parameters()]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.5)
    adapter._update_ema(model)
    after = [p.detach().clone() for p in ema.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))

    state = adapter.state_dict(model)
    assert set(state) == {'model', 'ema'}
    restored = FlowSDFAdapter()
    restored.device = torch.device('cpu')
    target = build_flowsdf(model_channels=32, rrdb_blocks=1)
    restored.load_state_dict(target, state)
    for left, right in zip(restored.ema_model.parameters(),
                           state['ema'].values()):
        assert torch.equal(left.detach(), right)


def test_predict_sample_returns_bool_at_gt_resolution(tmp_path):
    adapter = FlowSDFAdapter()
    adapter.args = _args()
    adapter.device = torch.device('cpu')
    model = build_flowsdf(model_channels=32, rrdb_blocks=1).eval()
    adapter.ema_model = model
    sample = _sample(tmp_path)
    prepared = adapter.prepare_eval_sample(sample, torch.device('cpu'))
    prediction = adapter.predict_sample(model, prepared, sample)
    assert prediction.dtype == np.bool_
    assert prediction.shape == (48, 64)


def test_config_roundtrip():
    adapter = FlowSDFAdapter()
    adapter.input_size = 128
    args = _args()
    config = adapter.save_config(args)
    restored = FlowSDFAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert target.input_size == 128
    assert target.flowsdf_model_channels == 128
    assert target.flowsdf_threshold == pytest.approx(0.03)
    assert target.flowsdf_no_augment is False
