"""Independence and behavior tests for the baseline CoSNA package."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.adapters.registry import load_adapter  # noqa: E402
from baseline_zoo.method_groups import group_for  # noqa: E402
from baseline_zoo.models.cosna.adapter import CoSNAAdapter  # noqa: E402
from baseline_zoo.models.cosna.tcsu import SnakeUpsampler  # noqa: E402
from baseline_zoo.runtime import build_parser  # noqa: E402


def test_registry_and_independence():
    adapter = load_adapter('cosna')
    assert isinstance(adapter, CoSNAAdapter)
    assert adapter.id == 'cosna'
    assert adapter.display_name == 'CoSNA'
    assert type(adapter).__module__ == 'baseline_zoo.models.cosna.adapter'
    assert group_for('cosna') == 'proposed'

    root = REPO_ROOT / 'src' / 'baseline_zoo'
    source = (root / 'adapters' / 'cosna.py').read_text()
    source += '\n'.join(
        path.read_text()
        for path in (root / 'models' / 'cosna').rglob('*.py'))
    for forbidden in (
            'methods.cosna',
            'from methods',
            'import methods',
            'baseline_zoo.models.bctd',
            'baseline_zoo.adapters.bctd',
            'from .bctd',
            'from ..models.bctd',
            'BCTDAdapter'):
        assert forbidden not in source, forbidden


def test_parser_defaults_match_canonical_cosna():
    adapter = CoSNAAdapter()
    parser = build_parser(adapter)
    args = parser.parse_args([
        '--stage', 'predict',
        '--data-root', 'data/sacm_3shot_seed0',
        '--test-datasets', 'DRIVE:/tmp',
        '--use_wandb', '0',
    ])
    assert args.cosna_gate == 0.10
    assert args.cosna_near_radius == 4
    assert args.cosna_train_scope == 'tcsu'
    assert args.cosna_filter_min_size == 80
    assert args.cosna_filter_aspect == 2.0
    assert args.cosna_filter_maxdim == 1200
    assert args.rcd == 1
    assert args.as_calibration == 1
    assert args.close_radius == 2


def test_tcsu_zero_residual_identity():
    module = SnakeUpsampler(render_size=128, coarse_size=64)
    stage2 = torch.randn(1, 4, 32, 32)
    image = torch.rand(1, 3, 128, 128)
    out = module(stage2, image)
    base = F.interpolate(
        stage2.detach()[:, 3:4], size=(128, 128), mode='bilinear',
        align_corners=False)
    assert torch.equal(out, base)
