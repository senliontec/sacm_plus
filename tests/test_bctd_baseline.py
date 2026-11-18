"""Independence and behavior tests for the baseline BCTD-SAM module."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.adapters.registry import load_adapter  # noqa: E402
from baseline_zoo.adapters.bctd import BCTDAdapter  # noqa: E402
from baseline_zoo.runtime import build_parser  # noqa: E402
from baseline_zoo.models.bctd.adaptive_skel import (  # noqa: E402
    apply_adaptive_skel,
)
from baseline_zoo.models.bctd.boundary_loss import BoundaryLoss  # noqa: E402
from baseline_zoo.models.bctd.derive_fixed3 import derive  # noqa: E402


def test_registry_and_independence():
    adapter = load_adapter('bctd')
    assert isinstance(adapter, BCTDAdapter)
    assert adapter.id == 'bctd'
    assert adapter.display_name == 'BCTD-SAM'
    assert type(adapter).__module__ == 'baseline_zoo.adapters.bctd'
    source = (REPO_ROOT / 'src' / 'baseline_zoo' / 'adapters' /
              'bctd.py').read_text()
    source += '\n'.join(
        path.read_text()
        for path in (REPO_ROOT / 'src' / 'baseline_zoo' / 'models' /
                     'bctd').rglob('*.py'))
    for forbidden in (
            'methods.topostrand', 'adapters.strandcal', 'models.strandcal',
            'adapters.dcd', 'models.dcd', 'adapters.spa_sam',
            'models.spa_sam', 'adapters.curvimix', 'models.curvimix',
            'adapters.sacm', 'models.sacm',
            'TopoStrandAdapter', 'StrandCalAdapter', 'DCDAdapter',
            'SpaSamAdapter', 'CurviMixAdapter', 'SACMAdapter'):
        assert forbidden not in source, forbidden


def test_parser_and_defaults():
    adapter = BCTDAdapter()
    parser = build_parser(adapter)
    args = parser.parse_args([
        '--stage', 'train', '--epochs', '1', '--use_wandb', '0',
        '--boundary-loss-weight', '0.05',
        '--boundary-loss-head', 'boundary',
        '--boundary-loss-resolution', '512',
        '--boundary-loss-warmup-epochs', '5',
        '--as-calibration', '1', '--atc', '0'])
    assert args.epochs == 1
    assert args.head_fusion == 0
    assert args.lr == 3e-4
    assert args.boundary_loss_weight == 0.05
    assert args.boundary_loss_head == 'boundary'
    assert args.boundary_loss_resolution == 512
    assert args.boundary_loss_warmup_epochs == 5
    assert args.as_calibration == 1
    assert args.atc == 0


def test_adaptive_skel_matches_reference():
    reference_path = (REPO_ROOT / 'scripts' / 'experiments' /
                      'topostrand' / 'transform_predictions.py')
    spec = importlib.util.spec_from_file_location(
        'topostrand_transform_reference', reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    args = argparse.Namespace(
        radius=1, max_radius=2.0, width_low=7.0, width_high=15.0,
        fg_min=0.04, fg_dense=0.09, fg_thick=0.15,
        thick_radius=2, dense_thin_radius=1,
        fg_dense_thin=0.09, width_dense_thin=8.0)
    rng = np.random.default_rng(0)
    cases = 0
    for shape in ((64, 64), (96, 128), (33, 77)):
        for probability in (0.001, 0.01, 0.04, 0.08, 0.12, 0.2, 0.4):
            mask = rng.random(shape) < probability
            got = apply_adaptive_skel(mask)
            want = reference.adaptive_skel_dilate(mask, args)
            assert np.array_equal(got, want), (shape, probability)
            cases += 1
    assert cases == 21


def test_boundary_loss_is_finite_and_differentiable():
    loss = BoundaryLoss(resolution=32)
    logits = torch.zeros(1, 1, 32, 32, requires_grad=True)
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 8:24, 12:20] = 1.0
    value = loss(logits, target)
    assert torch.isfinite(value)
    value.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_fixed3_derive(tmp_path):
    source = tmp_path / 'best_model.pth'
    target = tmp_path / 'best_model_fixed3.pth'
    payload = {
        'adapter': 'topostrand',
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
    assert out['adapter'] == 'bctd'
    assert 'head_fusion.fuse.weight' not in state
    cfg = out['config']['adapter_config']
    assert cfg['head_select'] == 'fixed'
    assert cfg['deployment_head'] == 3
    assert cfg['head_fusion'] == 0
    assert out['fixed3_model'] == 'bctd'


def test_train_step_boundary_loss_smoke(monkeypatch):
    adapter = BCTDAdapter()
    adapter.device = torch.device('cpu')
    adapter.args = argparse.Namespace(
        bce_weight=0.7, dice_weight=0.3, bce_pos_weight=2.0)
    adapter.head_select = 'gating'
    adapter.deployment_head = 3
    adapter.specialized_heads = True
    adapter.specialized_head_weight = 0.1
    adapter.specialized_topology_weight = 0.1
    adapter.cl_dice_weight = 0.1
    adapter.skeleton_recall_weight = 0.1
    adapter.topology_resolution = 32
    adapter.topology_warmup_epochs = 0
    adapter.deep_sup_weight = 0.3
    adapter.cape_weight = 0.0
    adapter.deployed_tversky_weight = 0.0
    adapter.area_loss_weight = 0.0
    adapter.boundary_loss_weight = 0.05
    adapter.boundary_loss_resolution = 32
    adapter.boundary_loss_head = 'boundary'
    adapter.boundary_loss_warmup_epochs = 0
    adapter.head_fusion = False
    adapter.stage2_loss_scope = 'all'
    adapter.topology_head = 'all'
    adapter._epoch = 1

    def fake_forward(model, images, return_stage1=False):
        del model
        base = torch.rand(
            1, 4, 64, 64, generator=torch.Generator().manual_seed(0))
        base.requires_grad_(True)
        return (base, None, base) if return_stage1 else (base, None)

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    images = torch.rand(1, 3, 64, 64)
    masks = (torch.rand(1, 1, 64, 64) >
             0.8).float()
    loss = adapter.train_step(None, (images, masks))
    assert torch.isfinite(loss)
    loss.backward()
