"""Independence and behavior tests for the baseline CurviMix-SAM module."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.adapters.registry import load_adapter  # noqa: E402
from baseline_zoo.adapters.curvimix import CurviMixAdapter  # noqa: E402
from baseline_zoo.runtime import build_parser  # noqa: E402
from baseline_zoo.models.curvimix.derive_fixed3 import derive  # noqa: E402


def test_registry_and_independence():
    adapter = load_adapter('curvimix')
    assert isinstance(adapter, CurviMixAdapter)
    assert adapter.id == 'curvimix'
    assert adapter.display_name == 'CurviMix-SAM'
    assert type(adapter).__module__ == 'baseline_zoo.adapters.curvimix'
    # The baseline module must not inherit from, or import, another model.
    source = (REPO_ROOT / 'src' / 'baseline_zoo' / 'adapters' /
              'curvimix.py').read_text()
    source += '\n'.join(
        path.read_text()
        for path in (REPO_ROOT / 'src' / 'baseline_zoo' / 'models' /
                     'curvimix').rglob('*.py'))
    for forbidden in (
            'adapters.spa_sam', 'adapters.dcd', 'adapters.sacm',
            'models.spa_sam', 'models.dcd', 'models.sacm',
            'SpaSamAdapter', 'DCDAdapter', 'SACMAdapter'):
        assert forbidden not in source, forbidden


def test_parser_and_defaults():
    adapter = CurviMixAdapter()
    parser = build_parser(adapter)
    args = parser.parse_args([
        '--stage', 'train', '--epochs', '1', '--use_wandb', '0',
        '--copy-paste-prob', '0.25', '--tta-flip'])
    assert args.epochs == 1
    assert args.copy_paste_prob == 0.25
    assert args.tta_flip is True
    assert args.head_fusion == 0
    assert args.lr == 3e-4


def test_copy_paste_and_tta():
    adapter = CurviMixAdapter()
    adapter.device = torch.device('cpu')
    adapter.input_size = 32
    adapter.copy_paste_prob = 1.0
    adapter.copy_paste_intensity_match = True
    source_image = torch.rand(3, 32, 32)
    source_mask = torch.zeros(1, 32, 32)
    source_mask[:, 8:24, 8:24] = 1.0
    adapter._source_bank = [(source_image, source_mask)]
    images = torch.rand(1, 3, 32, 32)
    masks = (torch.rand(1, 1, 32, 32) > 0.8).float()
    out_images, out_masks = adapter._copy_paste(
        images.clone(), masks.clone(), random.Random(0))
    assert out_images.shape == images.shape
    assert out_masks.shape == masks.shape
    assert torch.isfinite(out_images).all()
    assert float(out_masks.max()) > 0.5

    # A tiny fake decoder lets us test the TTA / selector path without SAM.
    adapter.head_fusion = False
    adapter.head_select = 'fixed'
    adapter.deployment_head = 3
    adapter.args = argparse.Namespace(deploy_threshold=0.5)
    adapter.tta_flip = True
    adapter.tta_select = False

    def fake_forward(model, tensor, return_stage1=False):
        del model, return_stage1
        batch = tensor.shape[0]
        base = torch.rand(batch, 4, 16, 16,
                          generator=torch.Generator().manual_seed(0))
        return base, None

    adapter._forward = fake_forward
    mask_path = REPO_ROOT / 'results' / '.tmp' / 'curvimix_tta_test_mask.png'
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('L', (20, 24), 0).save(mask_path)
    sample = argparse.Namespace(mask_path=mask_path)
    prepared = torch.rand(1, 3, 64, 64)
    pred = adapter.predict_sample(None, prepared, sample)
    assert pred.shape == (24, 20)
    assert pred.dtype == np.bool_
    adapter.tta_select = True
    pred2 = adapter.predict_sample(None, prepared, sample)
    assert pred2.shape == (24, 20)


def test_fixed3_derive(tmp_path):
    source = tmp_path / 'best_model.pth'
    target = tmp_path / 'best_model_fixed3.pth'
    payload = {
        'adapter': 'curvimix',
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
    assert out['adapter'] == 'curvimix'
    assert 'head_fusion.fuse.weight' not in state
    cfg = out['config']['adapter_config']
    assert cfg['head_select'] == 'fixed'
    assert cfg['deployment_head'] == 3
    assert cfg['head_fusion'] == 0
