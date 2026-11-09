"""Regression tests for the nnWNet BatchNorm/InstanceNorm adaptation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from baseline_zoo.adapters.nnwnet import (
    _infer_checkpoint_norm,
    _install_norm_architecture,
    _replace_batchnorm,
)
from baseline_zoo.adapters.registry import load_adapter
from baseline_zoo.models.nnwnet import get_trainer_class
from baseline_zoo.models.nnwnet.trainer import WNet2D
from baseline_zoo.runtime import build_parser


def test_default_cli_uses_instance_norm():
    adapter = load_adapter('nnwnet')
    args = build_parser(adapter).parse_args([])
    assert args.nnwnet_norm == 'instance'
    args = build_parser(adapter).parse_args(
        ['--nnwnet-norm', 'batch'])
    assert args.nnwnet_norm == 'batch'


def test_replace_batchnorm_makes_forward_consistent():
    model = WNet2D(in_channel=3, num_classes=2,
                   deep_supervised=False)
    assert any(isinstance(module, nn.BatchNorm2d)
               for module in model.modules())
    _replace_batchnorm(model)
    assert not any(isinstance(module, nn.BatchNorm2d)
                   for module in model.modules())
    assert any(isinstance(module, nn.InstanceNorm2d)
               for module in model.modules())
    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 2, 64, 64)


def test_checkpoint_norm_is_inferred_from_buffers(tmp_path):
    batch = tmp_path / 'batch.pth'
    torch.save({
        'network_weights': {
            'inc.bn.running_mean': torch.zeros(1),
            'inc.bn.weight': torch.ones(1),
        }
    }, batch)
    assert _infer_checkpoint_norm(batch) == 'batch'

    instance = tmp_path / 'instance.pth'
    torch.save({
        'network_weights': {
            'inc.norm.weight': torch.ones(1),
            'inc.norm.bias': torch.zeros(1),
        }
    }, instance)
    assert _infer_checkpoint_norm(instance) == 'instance'


def _write_split(root: Path, split: str, stems: list[str]) -> None:
    image_dir = root / split / 'images'
    mask_dir = root / split / 'masks'
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for index, stem in enumerate(stems):
        image = np.full((12, 16, 3), 40 + index, dtype=np.uint8)
        Image.fromarray(image).save(image_dir / f'{stem}.png')
        mask = np.zeros((12, 16), dtype=np.uint8)
        mask[4:8, 3:12] = 255
        Image.fromarray(mask).save(mask_dir / f'{stem}.png')
    manifest = root / 'split_manifest.csv'
    write_header = not manifest.is_file()
    with manifest.open('a', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['dataset', 'split', 'output', 'group'])
        if write_header:
            writer.writeheader()
        for stem in stems:
            writer.writerow({
                'dataset': 'DRIVE', 'split': split,
                'output': stem, 'group': 'DRIVE'})


def test_stage_raw_writes_binary_labels_and_fixed_split(tmp_path):
    """nnU-Net labels must be 0/1 and the 18/6 split must stay explicit."""
    data_root = tmp_path / 'data'
    _write_split(data_root, 'train', ['DRIVE_1', 'DRIVE_2'])
    _write_split(data_root, 'val', ['DRIVE_3'])
    adapter = load_adapter('nnwnet')
    raw = tmp_path / 'nnunet' / 'raw'

    train_cases, _ = adapter._stage_raw(
        raw, data_root, include_val=False, channels='grayscale')
    train_cases, val_cases = adapter._stage_raw(
        raw, data_root, include_val=True, channels='grayscale')

    assert train_cases == ['DRIVE_1', 'DRIVE_2']
    assert val_cases == ['DRIVE_3']
    labels = np.asarray(Image.open(
        raw / adapter_dataset_dir(raw) / 'labelsTr' / 'DRIVE_1.png'))
    assert set(np.unique(labels)) <= {0, 1}
    dataset_json = json.loads(
        (raw / adapter_dataset_dir(raw) / 'dataset.json').read_text())
    assert dataset_json['numTraining'] == 3
    assert dataset_json['labels'] == {'background': 0, 'foreground': 1}
    assert dataset_json['channel_names'] == {'0': 'grayscale'}

    preprocessed = tmp_path / 'nnunet' / 'preprocessed'
    adapter._write_splits(preprocessed, train_cases, val_cases)
    splits = json.loads((preprocessed / 'splits_final.json').read_text())
    assert splits == [{'train': train_cases, 'val': val_cases}]


def adapter_dataset_dir(raw: Path) -> str:
    from baseline_zoo.adapters.nnwnet import DATASET_NAME
    return DATASET_NAME


def test_trainer_builder_installs_instance_norm():
    trainer_class = get_trainer_class()
    original = trainer_class.build_network_architecture
    try:
        assert _install_norm_architecture('instance') is trainer_class
        model = trainer_class.build_network_architecture(
            'unused', {}, [], 3, 2, False)
        assert not any(isinstance(module, nn.BatchNorm2d)
                       for module in model.modules())
        assert any(isinstance(module, nn.InstanceNorm2d)
                   for module in model.modules())

        assert _install_norm_architecture('batch') is trainer_class
        model = trainer_class.build_network_architecture(
            'unused', {}, [], 3, 2, False)
        assert any(isinstance(module, nn.BatchNorm2d)
                   for module in model.modules())
        assert not any(isinstance(module, nn.InstanceNorm2d)
                       for module in model.modules())
    finally:
        trainer_class.build_network_architecture = staticmethod(original)
        for attribute in ('_adapter_norm_builder', '_adapter_norm_mode',
                          '_adapter_norm_original'):
            if hasattr(trainer_class, attribute):
                delattr(trainer_class, attribute)
