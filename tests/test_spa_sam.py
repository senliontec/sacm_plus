"""Unit tests for the independent SPA-SAM baseline module."""

from __future__ import annotations

import argparse
import inspect
import random
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.adapters.registry import load_adapter  # noqa: E402
from baseline_zoo.adapters.spa_sam import (  # noqa: E402
    AUG_MODES,
    SpaSamAdapter,
    augment_batch,
)
from baseline_zoo.models.spa_sam.derive_fixed3 import derive  # noqa: E402


class _Always:
    def random(self):
        return 0.1

    def randrange(self, stop):
        return 1 % stop


def test_augmentation_modes():
    images = torch.rand(1, 3, 16, 16)
    masks = (torch.rand(1, 1, 16, 16) > 0.5).float()
    for mode in AUG_MODES:
        rng = random.Random(0) if mode != 'flip' else _Always()
        out_images, out_masks = augment_batch(images, masks, rng, mode)
        assert out_images.shape == images.shape
        assert out_masks.shape == masks.shape
        assert torch.isfinite(out_images).all()
        assert torch.isfinite(out_masks).all()
        assert out_images.min() >= 0 and out_images.max() <= 1
        assert out_masks.min() >= 0 and out_masks.max() <= 1
    none_images, none_masks = augment_batch(
        images, masks, random.Random(0), 'none')
    assert torch.equal(none_images, images)
    assert torch.equal(none_masks, masks)
    neutral_images, neutral_masks = augment_batch(
        images, masks, random.Random(0), 'mild', flip_prob=0.0,
        photo_strength=0.0, noise_std=0.0)
    assert torch.allclose(neutral_images, images)
    assert torch.equal(neutral_masks, masks)
    try:
        augment_batch(images, masks, random.Random(0), 'bad')
    except ValueError:
        pass
    else:
        raise AssertionError('invalid augmentation mode must raise')


def test_flip_is_synchronized():
    images = torch.zeros(1, 3, 8, 8)
    images[:, :, 1, 1] = 1.0
    masks = torch.zeros(1, 1, 8, 8)
    masks[:, :, 1, 1] = 1.0
    out_images, out_masks = augment_batch(
        images, masks, _Always(), 'flip')
    expected_image = torch.flip(torch.flip(images, [3]), [2])
    expected_mask = torch.flip(torch.flip(masks, [3]), [2])
    assert torch.equal(out_images, expected_image)
    assert torch.equal(out_masks, expected_mask)


def test_registry_and_independence():
    adapter = load_adapter('spa_sam')
    assert isinstance(adapter, SpaSamAdapter)
    assert adapter.id == 'spa_sam'
    assert adapter.display_name == 'SPA-SAM'
    assert adapter.train_aug == 'mild'
    source = inspect.getsource(SpaSamAdapter)
    assert 'baseline_zoo.models.dcd' not in source
    assert 'baseline_zoo.adapters.dcd' not in source


def test_config_roundtrip():
    adapter = SpaSamAdapter()
    args = argparse.Namespace(train_aug='mild')
    adapter.restore_config(args, {'train_aug': 'photo'})
    assert args.train_aug == 'photo'
    adapter._read_aug(args)
    assert adapter.train_aug == 'photo'


def test_fixed3_derivation(tmp_path):
    source = tmp_path / 'best_model.pth'
    target = tmp_path / 'best_model_fixed3.pth'
    payload = {
        'adapter': 'spa_sam',
        'config': {'adapter_config': {'head_fusion': 1}},
        'model_state_dict': {
            'head_fusion.fuse.weight': torch.zeros(1, 4, 1, 1),
            'head_fusion.fuse.bias': torch.zeros(1),
            'mask_decoder.x': torch.zeros(1),
        },
        'best_epoch': 7,
        'val_dice': 0.5,
    }
    torch.save(payload, source)
    result = derive(source, target, 3)
    assert not result['skipped']
    out = torch.load(target, map_location='cpu', weights_only=False)
    state = out['model_state_dict']
    assert out['adapter'] == 'spa_sam'
    assert 'head_fusion.fuse.weight' not in state
    assert out['config']['adapter_config']['head_select'] == 'fixed'
    assert out['config']['adapter_config']['deployment_head'] == 3
    assert out['config']['adapter_config']['head_fusion'] == 0
