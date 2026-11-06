"""Tests for the CS2Net task-adapted adapter.

Covers the full-resolution geometry path, the BatchNorm -> InstanceNorm
replacement, and the checkpoint config round-trip.  The official model file
is not modified; only the adapter wraps it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.adapters.cs2net import (  # noqa: E402
    CS2NET_MULTIPLE,
    CS2NetAdapter,
    _full_meta,
    _full_transform,
    _replace_batchnorm,
    _restore_full,
)
from baseline_zoo.models.cs2net import CSNet  # noqa: E402


def test_full_meta_pads_to_multiple_of_16():
    meta = _full_meta(100, 70, 1024)
    assert (meta['new_w'], meta['new_h']) == (100, 70)
    assert meta['scale'] == 1.0
    assert (meta['new_w'] + meta['pad_w']) % CS2NET_MULTIPLE == 0
    assert (meta['new_h'] + meta['pad_h']) % CS2NET_MULTIPLE == 0


def test_full_meta_caps_long_side_and_keeps_ratio():
    meta = _full_meta(2048, 1024, 1024)
    assert max(meta['new_w'], meta['new_h']) == 1024
    assert abs(meta['new_w'] / meta['new_h'] - 2.0) < 1e-3
    assert (meta['new_w'] + meta['pad_w']) % CS2NET_MULTIPLE == 0


def test_full_transform_restore_roundtrip():
    width, height = 100, 70
    image = Image.fromarray(
        np.full((height, width, 3), 12, dtype=np.uint8))
    mask_array = np.zeros((height, width), dtype=np.uint8)
    mask_array[10:30, 20:60] = 255
    mask = Image.fromarray(mask_array)

    transformed, transformed_mask, meta = _full_transform(
        image, mask, 1024)
    assert transformed.size == (
        meta['new_w'] + meta['pad_w'], meta['new_h'] + meta['pad_h'])
    assert transformed_mask.size == transformed.size

    prediction = np.asarray(transformed_mask) > 127
    restored = _restore_full(prediction, meta)
    assert restored.shape == (height, width)
    assert np.array_equal(restored, mask_array > 127)


def test_instance_norm_replacement_and_forward():
    model = _replace_batchnorm(CSNet(classes=1, channels=3), 'instance')
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert any(isinstance(m, nn.InstanceNorm2d) for m in model.modules())
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 80, 112))
    assert output.shape == (1, 1, 80, 112)


def test_batch_norm_mode_is_unchanged():
    model = _replace_batchnorm(CSNet(classes=1, channels=3), 'batch')
    assert any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert not any(isinstance(m, nn.InstanceNorm2d) for m in model.modules())


def test_config_save_restore_roundtrip():
    adapter = CS2NetAdapter()
    source = argparse.Namespace(
        cs2net_geometry='full', cs2net_max_side=1024,
        cs2net_norm='instance', cs2net_augment='mild',
        cs2net_threshold='fixed', cs2net_threshold_value=0.5)
    config = adapter.save_config(source)

    target = argparse.Namespace(
        cs2net_geometry='padded', cs2net_max_side=1024,
        cs2net_norm='batch', cs2net_augment='official',
        cs2net_threshold='otsu', cs2net_threshold_value=0.5)
    adapter.restore_config(target, config)

    assert target.cs2net_geometry == 'full'
    assert target.cs2net_norm == 'instance'
    assert target.cs2net_augment == 'mild'
    assert target.cs2net_threshold == 'fixed'
