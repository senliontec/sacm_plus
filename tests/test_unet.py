"""Regression tests for the official U-Net baseline port."""

import argparse
import random
from pathlib import Path

import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.unet import UNetAdapter, dice_loss
from baseline_zoo.data import Sample
from baseline_zoo.runtime import build_parser


def _adapter() -> UNetAdapter:
    adapter = UNetAdapter()
    adapter.device = torch.device('cpu')
    adapter.args = argparse.Namespace(
        selection='dice', lr=1e-5, weight_decay=1e-8, momentum=0.999)
    return adapter


def _sample(tmp_path: Path, width: int = 200, height: int = 100,
            name: str = 'img') -> Sample:
    image_path = tmp_path / f'{name}.png'
    mask_path = tmp_path / f'{name}_mask.png'
    Image.new('RGB', (width, height), (10, 20, 30)).save(image_path)
    mask = Image.new('L', (width, height), 0)
    # Off-centre so a flip/rotation would be visible in the no-augmentation
    # regression test.
    cx, cy = width // 4, height // 4
    for x in range(cx - 2, cx + 2):
        for y in range(cy - 2, cy + 2):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)
    return Sample(image_path, mask_path, 'TEST', name)


def test_dice_loss_is_global_over_batch():
    # One empty image and one full image.  The official milesial loss is
    # computed once over the whole B,H,W tensor, not as a per-image mean.
    logits = torch.zeros(2, 1, 1, 1, requires_grad=True)
    target = torch.tensor([0.0, 1.0]).view(2, 1, 1, 1)

    loss = dice_loss(logits, target)
    expected = 1.0 - (1.0 + 1e-6) / (2.0 + 1e-6)
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-8)

    # The old per-image reduction produced a different value on this batch.
    prob = torch.sigmoid(logits.detach())
    inter = 2.0 * (prob * target).sum(dim=(1, 2, 3))
    sets_sum = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    per_image = 1.0 - ((inter + 1e-6) / (sets_sum + 1e-6)).mean()
    assert not torch.allclose(loss.detach(), per_image, atol=1e-6)

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_default_recipe_matches_official():
    args = build_parser(UNetAdapter()).parse_args(['--stage', 'train'])
    assert args.lr == 1e-5
    assert args.weight_decay == 1e-8
    assert args.momentum == 0.999
    assert args.scale == 0.5
    assert args.batch_size == 1
    assert args.grad_clip == 1.0


def test_optimizer_is_official_rmsprop():
    adapter = UNetAdapter()
    args = build_parser(adapter).parse_args(['--stage', 'train'])
    optimizer = adapter.make_optimizer(torch.nn.Linear(2, 2), args)
    assert isinstance(optimizer, torch.optim.RMSprop)
    group = optimizer.param_groups[0]
    assert group['lr'] == 1e-5
    assert group['weight_decay'] == 1e-8
    assert group['momentum'] == 0.999


def test_scheduler_is_plateau_on_dice():
    adapter = UNetAdapter()
    args = build_parser(adapter).parse_args(['--stage', 'train'])
    optimizer = adapter.make_optimizer(torch.nn.Linear(2, 2), args)
    scheduler = adapter.make_scheduler(optimizer, args)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.mode == 'max'
    assert scheduler.patience == 5
    assert adapter.scheduler_after_validation is True


def test_step_scheduler_receives_validation_dice():
    adapter = _adapter()
    optimizer = torch.optim.SGD(
        torch.nn.Linear(1, 1).parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5)
    calls = []
    scheduler.step = lambda value: calls.append(value)

    adapter.step_scheduler(scheduler, {'dice': 0.63})
    assert calls == [0.63]


def test_step_scheduler_generic_scheduler_uses_no_argument_step():
    adapter = _adapter()
    calls = []

    class FakeScheduler:
        def step(self):
            calls.append('stepped')

    adapter.step_scheduler(FakeScheduler(), {'dice': 0.63})
    assert calls == ['stepped']


def test_official_preprocessing_is_scale_half_without_augmentation(tmp_path):
    adapter = _adapter()
    sample = _sample(tmp_path, 200, 100)
    images, masks = adapter.prepare_train_batch([sample], random.Random(0))
    assert tuple(images.shape) == (1, 3, 50, 100)
    assert tuple(masks.shape) == (1, 1, 50, 100)
    # No augmentation: the same sample is identical for any RNG seed.
    images2, masks2 = adapter.prepare_train_batch([sample], random.Random(999))
    assert torch.equal(images, images2)
    assert torch.equal(masks, masks2)
    # 0/255 mask -> 0/1 float; the foreground block survives NEAREST resize.
    assert masks.max().item() == 1.0
    assert masks.sum().item() > 0


def test_variable_input_sizes_require_batch_one(tmp_path):
    adapter = _adapter()
    first = _sample(tmp_path, 200, 100, name='first')
    second = _sample(tmp_path, 120, 80, name='second')
    with pytest.raises(RuntimeError, match='batch-size 1'):
        adapter.prepare_train_batch([first, second], random.Random(0))


def test_predict_upsamples_logits_to_gt(tmp_path):
    adapter = _adapter()
    sample = _sample(tmp_path, 200, 100)

    class FixedLogits(torch.nn.Module):
        def forward(self, images):
            return torch.zeros(1, 1, 50, 100)

    pred = adapter.predict_sample(
        FixedLogits(), torch.zeros(1, 3, 50, 100), sample)
    assert pred.shape == (100, 200)
    assert pred.dtype == bool


def test_scale_config_roundtrip():
    args = build_parser(UNetAdapter()).parse_args(
        ['--stage', 'predict', '--test-datasets', 'X:y'])
    adapter = UNetAdapter()
    adapter.scale = 0.25
    config = adapter.save_config(args)
    restored = UNetAdapter()
    restored.restore_config(args, config)
    assert restored.scale == 0.25
    assert args.scale == 0.25


def test_default_grad_clip_matches_official():
    args = build_parser(UNetAdapter()).parse_args(['--stage', 'train'])
    assert args.grad_clip == 1.0
