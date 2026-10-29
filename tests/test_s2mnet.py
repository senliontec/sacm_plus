"""S2M-Net tests (PyTorch model, MAL, adapter plumbing)."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.adapters.s2mnet import S2MNetAdapter
from baseline_zoo.data import Sample
from baseline_zoo.models.s2mnet import (
    BFPDecoderStage,
    MRFSEBlock,
    S2MNet,
    SpectralSelectiveTokenMixer,
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
        input_size=352,
        s2mnet_dropout=0.1,
        s2mnet_sstm_k=32,
        s2mnet_warmup_epochs=10,
        s2mnet_tile_stride=176,
        s2mnet_no_augment=False,
        s2mnet_random_init=False,
        epochs=800, lr=1e-4, weight_decay=1e-4, grad_clip=1.0,
        accum_steps=1, seed=42, data_root='data/sacm_3shot',
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_registry_and_metadata():
    assert 's2mnet' in available_adapters()
    adapter = load_adapter('s2mnet')
    assert isinstance(adapter, S2MNetAdapter)
    assert adapter.id == 's2mnet'
    assert adapter.input_size == 352
    assert adapter.official_lr == pytest.approx(1e-4)
    assert adapter.official_weight_decay == pytest.approx(1e-4)
    assert any('TensorFlow' in item for item in adapter.deviations)


def test_model_forward_and_parameters():
    model = S2MNet(input_size=64)
    total = sum(p.numel() for p in model.parameters())
    assert 4_000_000 < total < 5_000_000
    model.eval()
    with torch.no_grad():
        prediction = model(torch.randn(2, 3, 64, 64))
    assert prediction.shape == (2, 1, 64, 64)
    assert prediction.min() >= 0 and prediction.max() <= 1
    assert sum(p.numel() for p in S2MNet(
        input_size=352).parameters()) < 5_000_000


def test_blocks_shapes():
    sstm = SpectralSelectiveTokenMixer(
        channels=24, num_frequencies=32, spatial_size=44,
        use_ssm=False)
    x = torch.randn(1, 24, 44, 44)
    assert sstm(x).shape == x.shape
    mrf = MRFSEBlock(filters=24)
    assert mrf(torch.randn(1, 24, 32, 32)).shape == (1, 24, 32, 32)
    bfp = BFPDecoderStage(64 + 32, 32)
    out, boundary = bfp(
        torch.randn(1, 64, 8, 8), torch.randn(1, 32, 16, 16))
    assert out.shape == (1, 32, 16, 16)
    assert boundary.shape == (1, 1, 16, 16)


def test_mal_loss_and_weight_clipping():
    model = S2MNet(input_size=64)
    gt = torch.zeros(2, 1, 64, 64)
    gt[:, :, 20:40, 10:50] = 1
    prediction = model(torch.randn(2, 3, 64, 64))
    loss = model.criterion(gt, prediction)
    assert torch.isfinite(loss)
    loss.backward()
    for name in model.criterion.weights:
        assert model.criterion.weights[name].grad is not None
    with torch.no_grad():
        model.criterion.weights['core'].fill_(20.0)
        model.criterion.weights['boundary'].fill_(0.0)
    model.criterion.clip_weights()
    assert float(model.criterion.weights['core']) == pytest.approx(10.0)
    assert float(model.criterion.weights['boundary']) == pytest.approx(0.1)
    values = model.criterion.learned_weight_values()
    assert values['scale'] == pytest.approx(0.5)


def test_adapter_data_and_config(tmp_path):
    adapter = S2MNetAdapter()
    adapter.args = _args()
    adapter.device = torch.device('cpu')
    batch = adapter.prepare_train_batch(
        [_sample(tmp_path)], random.Random(0))
    assert batch['image'].shape == (1, 3, 352, 352)
    assert batch['gt'].shape == (1, 1, 352, 352)
    assert batch['image'].min() >= 0 and batch['image'].max() <= 1
    config = adapter.save_config(adapter.args)
    restored = S2MNetAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert target.input_size == 352
    assert target.s2mnet_sstm_k == 32
    assert target.s2mnet_no_augment is False


def test_optimizer_and_scheduler():
    model = S2MNet(input_size=64)
    adapter = S2MNetAdapter()
    adapter.args = _args()
    optimizer = adapter.make_optimizer(model, adapter.args)
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-4)
    assert optimizer.param_groups[0]['weight_decay'] == pytest.approx(1e-4)
    scheduler = adapter.make_scheduler(optimizer, adapter.args)
    assert scheduler is not None
    for _ in range(1000):
        scheduler.step()
    assert min(scheduler.get_last_lr()) == pytest.approx(1e-6, rel=1e-3)


def test_predict_sample_tiles_and_restores_shape():
    model = S2MNet(input_size=64).eval()
    adapter = S2MNetAdapter()
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
    tiny = {
        'image': torch.rand(3, 40, 50),
        'size': (40, 50),
    }
    prediction = adapter.predict_sample(model, tiny, sample)
    assert prediction.shape == (40, 50)


@pytest.mark.slow
def test_train_step_backward_small():
    model = S2MNet(input_size=64)
    adapter = S2MNetAdapter()
    adapter.args = _args()
    batch = {
        'image': torch.randn(1, 3, 64, 64),
        'gt': torch.zeros(1, 1, 64, 64),
    }
    batch['gt'][:, :, 20:40, 10:50] = 1
    loss = adapter.train_step(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
