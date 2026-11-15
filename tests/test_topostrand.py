"""Smoke and contract tests for the independent TopoStrand-SAM module."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.runtime import build_parser  # noqa: E402
from methods.topostrand.adapter import TopoStrandAdapter  # noqa: E402
from methods.topostrand.derive_fixed3 import derive  # noqa: E402


def test_parser_and_recipe_defaults():
    adapter = TopoStrandAdapter()
    parser = build_parser(adapter)
    args = parser.parse_args([
        '--stage', 'train', '--epochs', '1', '--use_wandb', '0',
        '--deployed-tversky-weight', '0.1', '--area-loss-weight', '0.1',
        '--copy-paste-scale-jitter', '0.35',
    ])
    assert args.epochs == 1
    assert args.deployed_tversky_weight == 0.1
    assert args.area_loss_weight == 0.1
    assert args.copy_paste_scale_jitter == 0.35
    assert args.head_select is None
    assert args.deployment_head == 3


def test_fit_center_and_scale_jitter_shape():
    image = torch.rand(1, 3, 48, 64)
    fitted = TopoStrandAdapter._fit_center(image, 32, 32)
    assert fitted.shape == (1, 3, 32, 32)
    adapter = TopoStrandAdapter()
    adapter.device = torch.device('cpu')
    adapter.input_size = 32
    adapter.copy_paste_prob = 1.0
    adapter.copy_paste_intensity_match = False
    adapter.copy_paste_scale_jitter = 0.35
    adapter._source_bank = [(torch.rand(3, 32, 32), torch.ones(1, 32, 32))]
    images = torch.rand(1, 3, 32, 32)
    masks = torch.zeros(1, 1, 32, 32)
    out_images, out_masks = adapter._copy_paste(
        images, masks, random.Random(0))
    assert out_images.shape == images.shape
    assert out_masks.shape == masks.shape
    assert torch.isfinite(out_images).all()


def test_fixed3_rewrite(tmp_path):
    source = tmp_path / 'best_model.pth'
    target = tmp_path / 'best_model_fixed3.pth'
    torch.save({
        'adapter': 'topostrand',
        'config': {'adapter_config': {'head_fusion': 1}},
        'model_state_dict': {
            'head_fusion.fuse.weight': torch.zeros(1, 4, 1, 1),
            'head_fusion.fuse.bias': torch.zeros(1),
            'mask_decoder.x': torch.zeros(1),
        },
        'best_epoch': 3,
        'val_dice': 0.5,
    }, source)
    result = derive(source, target, 3)
    assert not result['skipped']
    out = torch.load(target, map_location='cpu', weights_only=False)
    assert out['adapter'] == 'topostrand'
    assert 'head_fusion.fuse.weight' not in out['model_state_dict']
    cfg = out['config']['adapter_config']
    assert cfg['head_select'] == 'fixed'
    assert cfg['deployment_head'] == 3
    assert cfg['head_fusion'] == 0
