"""Unit tests for the CoSNA package (CPU only)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from methods.cosna.adapter import CoSNAAdapter, _gated_mask
from methods.cosna.tcsu import SnakeUpsampler
from baseline_zoo.runtime import build_parser


def test_cosna_zero_residual():
    module = SnakeUpsampler(render_size=128, coarse_size=64)
    stage2 = torch.randn(1, 4, 32, 32)
    image = torch.rand(1, 3, 128, 128)
    out = module(stage2, image)
    base = F.interpolate(
        stage2.detach()[:, 3:4], size=(128, 128), mode='bilinear',
        align_corners=False)
    assert torch.equal(out, base)


def test_gated_mask_prefers_consensus_on_disagreement():
    probs = np.array([
        [[0.9, 0.9]],
        [[0.9, 0.1]],
        [[0.9, 0.1]],
        [[0.9, 0.1]],
    ], dtype=np.float32)
    mask = _gated_mask(probs, gate=0.10)
    assert mask[0, 0]
    assert not mask[0, 1]


def test_cosna_identity_and_cli():
    adapter = CoSNAAdapter()
    assert adapter.id == 'cosna'
    parser = build_parser(adapter)
    args = parser.parse_args([
        '--stage', 'predict', '--checkpoint', '/tmp/x.pth',
        '--data-root', '/tmp', '--test-datasets', 'X:/tmp',
        '--cosna-weight', '1', '--cosna-near-radius', '4',
    ])
    assert args.cosna_weight == 1.0
    assert args.cosna_near_radius == 4
