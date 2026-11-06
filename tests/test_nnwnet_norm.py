"""Regression tests for the nnWNet BatchNorm/InstanceNorm adaptation."""

from __future__ import annotations

import torch
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
