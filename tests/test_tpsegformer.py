"""TPSegformer tests (decoder, Hilbert traversal, adapter plumbing)."""

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.adapters.tpsegformer import TPSegformerAdapter
from baseline_zoo.data import Sample
from baseline_zoo.models.tpsegformer import (
    TPHead,
    build_tpsegformer,
    edge_aware_loss,
    joint_aux_loss,
    soft_dice_loss,
)
from baseline_zoo.models.tpsegformer.hilbert import (
    expand_hilbert,
    flatten_hilbert,
    hilbert_permutation,
)
from baseline_zoo.models.tpsegformer.modules import LayerCorrelationCalculation


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
        tpsegformer_backbone='swin_tiny_patch4_window7_224',
        tpsegformer_drop_path=0.3,
        tpsegformer_lccm_mode='hilbert',
        tpsegformer_backbone_ckpt=None,
        tpsegformer_head_lr_mult=10.0,
        tpsegformer_warmup_epochs=5,
        tpsegformer_no_augment=False,
        tpsegformer_no_photometric=False,
        epochs=800, lr=6e-5, weight_decay=0.01, grad_clip=0.0,
        accum_steps=1, seed=42, data_root='data/sacm_3shot',
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_registry_and_metadata():
    assert 'tpsegformer' in available_adapters()
    adapter = load_adapter('tpsegformer')
    assert isinstance(adapter, TPSegformerAdapter)
    assert adapter.id == 'tpsegformer'
    assert adapter.input_size == 512
    assert adapter.official_lr == pytest.approx(6e-5)
    assert adapter.official_weight_decay == pytest.approx(0.01)
    assert any('cls_seg' in item for item in adapter.deviations)


def test_hilbert_permutation_and_roundtrip():
    indices, inverse = hilbert_permutation(8)
    assert len(indices) == 64
    assert np.array_equal(indices[inverse], np.arange(64))
    points = np.stack(
        [indices // 8, indices % 8], axis=1)
    steps = np.abs(np.diff(points, axis=0)).sum(axis=1)
    assert np.all(steps == 1)
    x = torch.randn(2, 3, 8, 8)
    assert torch.allclose(
        expand_hilbert(flatten_hilbert(x, indices), inverse, 8), x)


def test_lccm_hilbert_matches_row_first():
    x = torch.randn(1, 5, 32, 32)
    hilbert = LayerCorrelationCalculation(32, 'hilbert').eval()
    row = LayerCorrelationCalculation(32, 'row-first').eval()
    row.load_state_dict(hilbert.state_dict())
    with torch.no_grad():
        left = hilbert(x)
        right = row(x)
    assert left.shape == (1, 5, 32, 32)
    assert torch.allclose(left, right, atol=1e-5)


def test_head_shapes():
    head = TPHead(
        hlff_in_channels=(384, 768), lccm_size=32, num_classes=1).eval()
    x3 = torch.randn(2, 384, 32, 32)
    x4 = torch.randn(2, 768, 16, 16)
    with torch.no_grad():
        out = head([torch.randn(2, 96, 128, 128),
                    torch.randn(2, 192, 64, 64), x3, x4])
    assert out.shape == (2, 1, 32, 32)


def test_model_forward_small():
    model = build_tpsegformer(
        image_size=256, num_classes=1, drop_path_rate=0.0).eval()
    with torch.no_grad():
        main, aux = model(torch.randn(1, 3, 256, 256))
    assert main.shape == (1, 1, 256, 256)
    assert aux.shape == (1, 1, 256, 256)


def test_losses_finite_and_differentiable():
    logits = torch.randn(1, 1, 64, 64, requires_grad=True)
    gt = torch.zeros(1, 1, 64, 64)
    gt[:, :, 30, 10:54] = 1
    loss = (torch.nn.functional.binary_cross_entropy_with_logits(
        logits, gt)
        + joint_aux_loss(logits, gt))
    assert torch.isfinite(loss)
    assert torch.isfinite(soft_dice_loss(logits, gt))
    assert torch.isfinite(edge_aware_loss(logits, gt))
    loss.backward()
    assert logits.grad is not None


def test_predict_sample_returns_gt_shape():
    model = build_tpsegformer(
        image_size=256, num_classes=1, drop_path_rate=0.0).eval()
    adapter = TPSegformerAdapter()
    adapter.args = _args()
    prepared = {
        'image': torch.randn(1, 3, 256, 256),
        'size': (48, 64),
    }
    sample = Sample(image_path=Path('x'), mask_path=Path('y'),
                    dataset='TEST', stem='TEST_0')
    prediction = adapter.predict_sample(model, prepared, sample)
    assert prediction.shape == (48, 64)
    assert prediction.dtype == np.bool_


def test_adapter_data_and_config(tmp_path):
    adapter = TPSegformerAdapter()
    adapter.args = _args()
    adapter.device = torch.device('cpu')
    batch = adapter.prepare_train_batch(
        [_sample(tmp_path)], random.Random(0))
    assert batch['image'].shape == (1, 3, 512, 512)
    assert batch['gt'].shape == (1, 1, 512, 512)
    assert torch.isfinite(batch['image']).all()

    adapter.input_size = 512
    config = adapter.save_config(adapter.args)
    restored = TPSegformerAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert target.input_size == 512
    assert target.tpsegformer_lccm_mode == 'hilbert'
    assert target.tpsegformer_head_lr_mult == pytest.approx(10.0)
    assert target.tpsegformer_no_augment is False


def test_optimizer_and_scheduler_groups():
    model = build_tpsegformer(
        image_size=256, num_classes=1, drop_path_rate=0.0)
    adapter = TPSegformerAdapter()
    adapter.args = _args()
    optimizer = adapter.make_optimizer(model, adapter.args)
    learning_rates = sorted({round(group['lr'], 8)
                             for group in optimizer.param_groups})
    assert any(abs(lr - 6e-5) < 1e-12 for lr in learning_rates)
    assert any(abs(lr - 6e-4) < 1e-12 for lr in learning_rates)
    assert any(group['weight_decay'] == 0.0
               for group in optimizer.param_groups)
    assert any(group['weight_decay'] == pytest.approx(0.01)
               for group in optimizer.param_groups)
    scheduler = adapter.make_scheduler(optimizer, adapter.args)
    assert scheduler is not None


@pytest.mark.slow
def test_train_step_backward_small():
    model = build_tpsegformer(
        image_size=256, num_classes=1, drop_path_rate=0.0)
    adapter = TPSegformerAdapter()
    adapter.args = _args()
    adapter.device = torch.device('cpu')
    batch = {
        'image': torch.randn(1, 3, 256, 256),
        'gt': torch.zeros(1, 1, 256, 256),
    }
    batch['gt'][:, :, 100:160, 60:200] = 1
    loss = adapter.train_step(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.head.parameters())
