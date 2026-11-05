"""Regression tests for the 3-shot U-Net baseline port."""

import argparse
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.unet import UNetAdapter, dice_loss
from baseline_zoo.data import Sample
from baseline_zoo.runtime import build_parser


def _adapter(patch: bool = True) -> UNetAdapter:
    adapter = UNetAdapter()
    adapter.device = torch.device('cpu')
    if patch:
        # The preprocessing tests below exercise the legacy patch ablation;
        # the deployed default is full native-resolution training.
        adapter.train_mode = 'patch'
    adapter.args = argparse.Namespace(
        selection='dice', lr=3e-4, weight_decay=1e-4)
    return adapter


def _sample(tmp_path: Path, width: int = 200, height: int = 100,
            name: str = 'img') -> Sample:
    image_path = tmp_path / f'{name}.png'
    mask_path = tmp_path / f'{name}_mask.png'
    image = Image.new('RGB', (width, height), (10, 20, 30))
    mask = Image.new('L', (width, height), 0)
    # Matching bright pixel block in image and mask, off-centre so a
    # flip/rotation would be visible if the transforms were not joint.
    cx, cy = width // 4, height // 4
    for x in range(cx - 2, cx + 2):
        for y in range(cy - 2, cy + 2):
            image.putpixel((x, y), (255, 255, 255))
            mask.putpixel((x, y), 255)
    image.save(image_path)
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


def test_default_recipe_matches_3shot():
    args = build_parser(UNetAdapter()).parse_args(['--stage', 'train'])
    assert args.lr == 3e-4
    assert args.weight_decay == 1e-4
    assert args.scale == 1.0
    assert args.batch_size == 1
    assert args.grad_clip == 1.0
    assert args.epochs == 300
    # The shared flag defaults to None; the adapter's official warmup is 20,
    # and configs/models/unet.yaml also passes --warmup-epochs 20.
    assert args.warmup_epochs is None
    assert UNetAdapter.warmup_epochs == 20
    assert args.unet_train_mode == 'full'
    assert args.unet_norm == 'instance'
    # Patch parameters remain parsed for the documented ablation.
    assert args.unet_patch_size == 512
    assert args.unet_crops_per_image == 8
    assert args.unet_fg_crop_prob == 0.5
    assert args.unet_photometric == 1


def test_optimizer_is_adamw():
    adapter = UNetAdapter()
    args = build_parser(adapter).parse_args(['--stage', 'train'])
    optimizer = adapter.make_optimizer(torch.nn.Linear(2, 2), args)
    assert isinstance(optimizer, torch.optim.AdamW)
    group = optimizer.param_groups[0]
    assert group['lr'] == 3e-4
    assert group['weight_decay'] == 1e-4


def test_scheduler_is_warmup_then_cosine():
    adapter = UNetAdapter()
    args = build_parser(adapter).parse_args(['--stage', 'train'])
    optimizer = adapter.make_optimizer(torch.nn.Linear(2, 2), args)
    scheduler = adapter.make_scheduler(optimizer, args)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert adapter.scheduler_after_validation is True

    warmup = UNetAdapter.warmup_epochs
    lrs = []
    for _ in range(args.epochs):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == pytest.approx(args.lr / warmup)
    assert lrs[warmup - 1] == pytest.approx(args.lr)
    assert lrs[warmup] <= args.lr
    assert lrs[-1] < 1e-6


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


def test_3shot_preprocessing_uses_native_square_patches(tmp_path):
    adapter = _adapter()
    sample = _sample(tmp_path, 200, 200)
    images, masks = adapter.prepare_train_batch([sample], random.Random(0))
    assert tuple(images.shape) == (1, 3, 512, 512)
    assert tuple(masks.shape) == (1, 1, 512, 512)
    # The 4x4 foreground block survives flips/90-degree rotations.
    assert masks.sum().item() == 16.0
    # Image and mask must receive the same transform.
    fg = masks[0, 0] > 0
    assert images[0][:, fg].mean() > images[0][:, ~fg].mean()


def test_mixed_input_sizes_produce_uniform_patches(tmp_path):
    adapter = _adapter()
    first = _sample(tmp_path, 200, 100, name='first')
    second = _sample(tmp_path, 120, 80, name='second')
    images, masks = adapter.prepare_train_batch(
        [first, second], random.Random(0))
    # Native patches normalise shapes, so batch > 1 no longer needs equal
    # source resolutions.
    assert tuple(images.shape) == (2, 3, 512, 512)
    assert tuple(masks.shape) == (2, 1, 512, 512)


def test_foreground_biased_crops_keep_foreground(tmp_path):
    adapter = _adapter()
    adapter.patch_size = 256
    adapter.fg_crop_prob = 1.0
    sample = _sample(tmp_path, 900, 600)
    for seed in range(12):
        _, masks = adapter.prepare_train_batch(
            [sample], random.Random(seed))
        assert masks.sum().item() > 0


def test_train_dataset_repeats_each_image(tmp_path):
    adapter = _adapter()
    args = build_parser(UNetAdapter()).parse_args(['--stage', 'train'])
    samples = [_sample(tmp_path, 64, 64, name=str(i)) for i in range(3)]
    dataset = adapter.make_train_dataset(samples, args)
    assert len(dataset) == 3 * args.unet_crops_per_image
    stems = [dataset[index].stem for index in range(len(dataset))]
    for sample in samples:
        assert stems.count(sample.stem) == args.unet_crops_per_image


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
    adapter.scale = 0.5
    adapter.patch_size = 384
    adapter.crops_per_image = 4
    adapter.fg_crop_prob = 0.75
    adapter.photometric = False
    config = adapter.save_config(args)
    restored = UNetAdapter()
    restored.restore_config(args, config)
    assert restored.scale == 0.5
    assert args.scale == 0.5
    assert restored.patch_size == 384
    assert args.unet_patch_size == 384
    assert restored.crops_per_image == 4
    assert args.unet_crops_per_image == 4
    assert restored.fg_crop_prob == 0.75
    assert args.unet_fg_crop_prob == 0.75
    assert restored.photometric is False
    assert args.unet_photometric == 0


def test_default_grad_clip_matches_official():
    args = build_parser(UNetAdapter()).parse_args(['--stage', 'train'])
    assert args.grad_clip == 1.0


def test_full_mode_uses_instance_norm_and_native_images(tmp_path):
    adapter = _adapter(patch=False)
    args = build_parser(UNetAdapter()).parse_args(['--stage', 'train'])
    model = adapter.build_model(args)
    assert not any(isinstance(module, torch.nn.BatchNorm2d)
                   for module in model.modules())
    assert any(isinstance(module, torch.nn.InstanceNorm2d)
               for module in model.modules())

    sample = _sample(tmp_path, 200, 100)
    images, masks = adapter.prepare_train_batch(
        [sample], random.Random(0))
    assert images.shape[0] == 1
    assert tuple(images.shape[1:]) in ((3, 100, 200), (3, 200, 100))
    assert tuple(masks.shape) == (1, 1) + tuple(images.shape[2:])
    # The 4x4 foreground block survives flips/90-degree rotations.
    assert masks.sum().item() == 16.0
    with pytest.raises(ValueError, match='--batch-size 1'):
        adapter.prepare_train_batch([sample, sample], random.Random(1))


def test_literal_binary_mask_is_foreground(tmp_path):
    sample = _sample(tmp_path, 80, 64)
    with Image.open(sample.mask_path) as mask:
        values = (np.asarray(mask) > 0).astype(np.uint8)
    Image.fromarray(values).save(sample.mask_path)

    adapter = _adapter()
    _, masks = adapter.prepare_train_batch(
        [sample], random.Random(0))
    assert masks.sum().item() == 16.0


def test_legacy_checkpoint_infers_batch_patch_recipe(tmp_path):
    args = build_parser(UNetAdapter()).parse_args(['--stage', 'predict'])
    adapter = UNetAdapter()
    adapter.restore_config(args, {'scale': 0.5})
    assert adapter.train_mode == 'patch'
    assert adapter.norm == 'batch'
    assert args.unet_train_mode == 'patch'
    assert args.unet_norm == 'batch'
    assert adapter.scale == 0.5


def test_patch_batch_ablation_builds_batchnorm(tmp_path):
    adapter = UNetAdapter()
    args = build_parser(adapter).parse_args([
        '--stage', 'train', '--unet-train-mode', 'patch',
        '--unet-norm', 'batch'])
    model = adapter.build_model(args)
    assert any(isinstance(module, torch.nn.BatchNorm2d)
               for module in model.modules())
    assert not any(isinstance(module, torch.nn.InstanceNorm2d)
                   for module in model.modules())
