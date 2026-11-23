"""Independence and behavior tests for the baseline WPRF-SAM package."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.adapters.registry import load_adapter  # noqa: E402
from baseline_zoo.method_groups import group_for  # noqa: E402
from baseline_zoo.models.wprf_sam.wprf import (  # noqa: E402
    AffinityHead, WPRFLoss, graph_distance, reachability,
    symmetrized_affinity)
from baseline_zoo.models.wprf_sam.wprf_adapter import WPRFAdapter  # noqa: E402
from baseline_zoo.runtime import build_parser  # noqa: E402


def test_registry_and_independence():
    adapter = load_adapter('wprf_sam')
    assert isinstance(adapter, WPRFAdapter)
    assert adapter.id == 'wprf_sam'
    assert adapter.display_name == 'WPRF-SAM reachability'
    assert type(adapter).__module__ == (
        'baseline_zoo.models.wprf_sam.wprf_adapter')
    assert group_for('wprf_sam') == 'prompt_free'

    root = REPO_ROOT / 'src' / 'baseline_zoo'
    source = (root / 'adapters' / 'wprf_sam.py').read_text()
    source += '\n'.join(
        path.read_text()
        for path in (root / 'models' / 'wprf_sam').rglob('*.py'))
    for forbidden in (
            'methods.wprf_sam',
            'from methods',
            'import methods',
            'baseline_zoo.models.cosna',
            'baseline_zoo.adapters.cosna',
            'from .cosna',
            'from ..models.cosna'):
        assert forbidden not in source, forbidden


def test_parser_defaults():
    adapter = WPRFAdapter()
    parser = build_parser(adapter)
    args = parser.parse_args([
        '--stage', 'predict',
        '--data-root', 'data/sacm_3shot_seed0',
        '--test-datasets', 'DRIVE:/tmp',
        '--use_wandb', '0',
    ])
    assert args.wprf == 1
    assert args.wprf_edge_weight == 1.0
    assert args.wprf_reach_weight == 1.0
    assert args.wprf_stride == 8
    assert args.wprf_sources == 16
    assert args.wprf_warmup_epochs == 0


def test_affinity_head_shape():
    head = AffinityHead(256)
    out = head(torch.zeros(2, 256, 64, 64))
    assert out.shape == (2, 8, 128, 128)


def test_wprf_loss_supports_batch():
    head = AffinityHead(256)
    embeddings = torch.zeros(2, 256, 64, 64)
    masks = torch.zeros(2, 1, 1024, 1024)
    masks[:, :, 500, 200:800] = 1.0
    loss = WPRFLoss(edge_weight=0.5, reach_weight=0.5)(
        head, embeddings, masks)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_wprf_edge_loss_uses_every_batch_item():
    head = AffinityHead(256)
    embeddings = torch.zeros(2, 256, 64, 64)
    two = torch.zeros(2, 1, 1024, 1024)
    two[:, :, 500, 200:800] = 1.0
    one = two.clone()
    one[1] = 0.0
    loss_two = WPRFLoss(edge_weight=1.0, reach_weight=0.0)(
        head, embeddings, two)
    loss_one = WPRFLoss(edge_weight=1.0, reach_weight=0.0)(
        head, embeddings, one)
    assert not torch.allclose(loss_two, loss_one)


def test_graph_distance_boundary_padding():
    support = torch.zeros(5, 5)
    support[2, :] = 1
    dist = graph_distance(support, np.array([[2, 0]]), 3)[0]
    assert float(dist[2, 3]) == 3.0
    # A zero-padded shift would make the boundary cell look distance 1.
    assert float(dist[2, 4]) >= 1e4 - 1


def test_maxmin_reachability_propagates():
    support = torch.zeros(1, 5, 5)
    support[0, 2, :] = 1
    weights = symmetrized_affinity(torch.zeros(1, 8, 5, 5), support)
    history = reachability(
        weights[0], support[0], np.array([[2, 0]]), 3)
    assert float(history[0][0, 2, 1]) > 0.4
    assert float(history[2][0, 2, 3]) > 0.4
    assert float(history[2][0, 2, 4]) == 0.0
