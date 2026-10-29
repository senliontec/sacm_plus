"""Full two-stage MaskVSC (stage-1 UNet + stage-2 ViG) tests."""

from __future__ import annotations

import argparse
import csv
import math

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from baseline_zoo.adapters.maskvsc import MaskVSCAdapter
from baseline_zoo.models.maskvsc import MaskVSCFull, UNet, ViGSeg_Fundus
from baseline_zoo.models.maskvsc.gcn import _tokens_to_feature_map


def _write_pair(root, split, stem):
    image_dir = root / split / 'images'
    mask_dir = root / split / 'masks'
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[16:48, 20:50] = 255
    Image.fromarray(image).save(image_dir / f'{stem}.png')
    Image.fromarray(mask).save(mask_dir / f'{stem}.png')
    return stem


@pytest.fixture
def split_root(tmp_path):
    root = tmp_path / 'split'
    train_stem = _write_pair(root, 'train', 'toy_train')
    val_stem = _write_pair(root, 'val', 'toy_val')
    with open(root / 'split_manifest.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['dataset', 'split',
                                                    'source_image', 'output',
                                                    'group'])
        writer.writeheader()
        writer.writerow({'dataset': 'toy', 'split': 'train',
                         'source_image': 'toy_train.png',
                         'output': train_stem, 'group': 'toy/__img_0'})
        writer.writerow({'dataset': 'toy', 'split': 'val',
                         'source_image': 'toy_val.png',
                         'output': val_stem, 'group': 'toy/__img_1'})
    return root


def test_vig_token_grid_and_forward_backward():
    with pytest.raises(ValueError, match='not a square'):
        _tokens_to_feature_map(torch.zeros(1, 5, 8))
    model = ViGSeg_Fundus(
        num_channels=1, img_size=128, patch_size=32, num_ViGBlocks=1)
    images = torch.rand(1, 1, 128, 128)
    output = model(images)
    assert output.shape == (1, 1, 128, 128)
    loss = nn.BCELoss()(output, torch.zeros_like(output))
    loss.backward()
    assert model.backbone.blocks[0].in_layer1.layer[0].weight.grad is not None


def test_full_wrapper_stage_transitions():
    seg = UNet(n_channels=3, n_classes=1)
    vig = ViGSeg_Fundus(
        num_channels=1, img_size=128, patch_size=32, num_ViGBlocks=1)
    model = MaskVSCFull(seg, vig)
    assert seg.training and not vig.training
    assert any(p.requires_grad for p in seg.parameters())
    assert not any(p.requires_grad for p in vig.parameters())

    images = torch.rand(1, 3, 128, 128)
    assert model(images).shape == (1, 1, 128, 128)

    model.set_stage('stage2')
    model.train()
    assert not seg.training and vig.training
    assert not any(p.requires_grad for p in seg.parameters())
    assert any(p.requires_grad for p in vig.parameters())
    output = model(images)
    assert output.shape == (1, 1, 128, 128)
    loss = nn.BCELoss()(output, torch.zeros_like(output))
    loss.backward()
    assert any(p.grad is not None for p in vig.parameters())
    assert not any(p.grad is not None for p in seg.parameters())

    state = model.state_dict()
    assert any(key.startswith('seg.') for key in state)
    assert any(key.startswith('vig.') for key in state)


def _training_args(root, workdir):
    return argparse.Namespace(
        data_root=root,
        workdir=workdir,
        stage='train',
        device='cpu',
        epochs=2,
        maskvsc_stage1_epochs=1,
        maskvsc_stage2_epochs=1,
        maskvsc_vig_blocks=1,
        maskvsc_vig_edges=9,
        maskvsc_vig_heads=1,
        maskvsc_vig_patch_size=32,
        maskvsc_vig_batch_size=1,
        maskvsc_vig_lr=None,
        batch_size=1,
        lr=2e-4,
        weight_decay=0.0,
        accum_steps=1,
        grad_clip=0.0,
        scheduler='official',
        warmup_epochs=None,
        selection='dice',
        seed=0,
        resume=False,
        num_workers=0,
        limit_train=0,
        limit_val=0,
        use_wandb=False,
        wandb_mode='disabled',
    )


def test_two_stage_training_and_prediction(split_root, tmp_path):
    from baseline_zoo.runtime import seed_everything

    adapter = MaskVSCAdapter()
    workdir = tmp_path / 'run'
    args = _training_args(split_root, workdir)
    seed_everything(0)
    reference = MaskVSCAdapter().build_model(args)
    reference_vig = reference.vig.backbone.blocks[0].in_layer1.layer[0].weight
    reference_vig = reference_vig.detach().clone()
    checkpoint = adapter.run_training(args)
    assert checkpoint.is_file()
    payload = torch.load(checkpoint, map_location='cpu',
                         weights_only=False)
    state = payload['model_state_dict']
    assert any(key.startswith('seg.') for key in state)
    assert any(key.startswith('vig.') for key in state)
    vig_key = 'vig.backbone.blocks.0.in_layer1.layer.0.weight'
    assert not torch.equal(state[vig_key], reference_vig)
    assert payload['config']['stage1_epochs'] == 1
    assert payload['config']['stage2_epochs'] == 1
    assert payload['config']['vig_arch']['blocks'] == 1
    assert payload['split_manifest_sha256']
    assert (workdir / 'metrics_history.json').is_file()
    assert (workdir / 'val_metrics_by_dataset.csv').is_file()

    adapter.prediction_mode = True
    model = adapter.build_model(_training_args(split_root, workdir))
    adapter.load_state_dict(model, state)
    sample = adapter.prepare_eval_sample(
        type('S', (), {
            'image_path': split_root / 'val' / 'images' / 'toy_val.png',
            'mask_path': split_root / 'val' / 'masks' / 'toy_val.png',
        })(), torch.device('cpu'))
    prediction = adapter.predict_sample(
        model, sample, type('S', (), {
            'mask_path': split_root / 'val' / 'masks' / 'toy_val.png',
        })())
    assert prediction.shape == (64, 64)
    assert prediction.dtype == np.bool_


def test_stage2_optimizer_is_not_empty():
    """Regression: stage-1 freezes ViG, so the optimizer must switch first."""
    adapter = MaskVSCAdapter()
    args = _training_args(None, None)
    adapter._apply_arch(args)
    model = adapter.build_model(args)
    model.set_stage('stage2')
    optimizer = torch.optim.Adam(
        [p for p in model.vig.parameters() if p.requires_grad], lr=2e-4)
    assert optimizer.param_groups[0]['params']
    assert math.isclose(optimizer.param_groups[0]['lr'], 2e-4)
