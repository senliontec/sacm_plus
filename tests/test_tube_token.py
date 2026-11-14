"""Unit tests for the isolated TubeToken post-decode control."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from methods.tube_token.adapter import TubeTokenAdapter  # noqa: E402
from methods.tube_token.derive_fixed3 import derive  # noqa: E402
from methods.tube_token.model import StructureControl  # noqa: E402


def _control(**overrides):
    kwargs = dict(
        adapter_dim=1024,
        feature_dim=32,
        grid_h=2,
        grid_w=4,
        render_size=32,
        highres_size=64,
        residual_weight=0.2,
        residual_prior=0.5,
        decision_weight=0.1,
    )
    kwargs.update(overrides)
    return StructureControl(**kwargs)


def test_structure_control_shapes_and_gradients():
    control = _control()
    images = torch.rand(1, 3, 64, 64)
    features = [torch.randn(1, 8, 8, 1024) for _ in range(4)]
    stage2 = torch.randn(1, 4, 16, 16, requires_grad=True)
    refined, structure, cache = control(features, images, stage2)
    assert refined.shape == stage2.shape
    assert structure.shape == (1, 1, 32, 32)
    assert torch.all(structure >= 0) and torch.all(structure <= 1)
    assert not torch.allclose(refined[:, 3:4], stage2[:, 3:4])
    # Only candidate 3 is touched.
    assert torch.allclose(refined[:, :3], stage2[:, :3])
    high = control.highres(
        refined, images, structure, cache['evidence'])
    assert high.shape == (1, 4, 64, 64)
    threshold = control.decision_threshold(structure, (64, 64))
    assert threshold.shape == (1, 1, 64, 64)
    assert torch.all(threshold >= 0.05) and torch.all(threshold <= 0.95)
    (high[:, 3:4].mean() + structure.mean()).backward()
    assert control.residual_scale.grad is not None
    assert control.geom_head[0].weight.grad is not None


def test_adapter_structure_config_roundtrip():
    adapter = TubeTokenAdapter()
    args = argparse.Namespace(
        structure_mode='post', structure_tokens=8,
        structure_grid_h=2, structure_grid_w=4,
        structure_render_size=64, structure_highres_size=128,
        structure_residual_weight=0.3, structure_residual_prior=0.4,
        structure_decision_weight=0.2,
    )
    adapter._read_structure_args(args)
    assert adapter.structure_tokens == 8
    assert adapter.structure_grid_h * adapter.structure_grid_w == 8
    restored = argparse.Namespace()
    adapter.restore_config(restored, {
        'structure_mode': 'post',
        'structure_tokens': 8,
        'structure_grid_h': 2,
        'structure_grid_w': 4,
        'structure_render_size': 64,
        'structure_highres_size': 128,
        'structure_residual_weight': 0.3,
        'structure_residual_prior': 0.4,
        'structure_decision_weight': 0.2,
    })
    assert restored.structure_tokens == 8
    assert restored.structure_decision_weight == 0.2


def test_derive_fixed3_preserves_structure_weights(tmp_path):
    source = tmp_path / 'best_model.pth'
    target = tmp_path / 'best_model_fixed3.pth'
    payload = {
        'adapter': 'tube_token',
        'config': {'adapter_config': {'head_fusion': 1}},
        'model_state_dict': {
            'head_fusion.fuse.weight': torch.zeros(1, 4, 1, 1),
            'head_fusion.fuse.bias': torch.zeros(1),
            'structure_control.residual_scale': torch.tensor(0.2),
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
    assert 'head_fusion.fuse.weight' not in state
    assert 'structure_control.residual_scale' in state
    assert out['config']['adapter_config']['head_select'] == 'fixed'
    assert out['config']['adapter_config']['deployment_head'] == 3
    assert out['config']['adapter_config']['head_fusion'] == 0
