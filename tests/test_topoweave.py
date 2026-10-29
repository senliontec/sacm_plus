"""TopoWeave tests: model, topology loss, TopoMask and shared-runtime smoke."""

from __future__ import annotations

import csv
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.adapters.topoweave import TopoWeaveAdapter
from baseline_zoo.data import Sample
from baseline_zoo.models.topoweave import (
    TopoWeaveLoss,
    TopoWeaveNet,
    augment_pair,
    skeleton_weighted_bce,
    topo_ratio,
    width_aware_topomask,
)
from baseline_zoo.runtime import run_prediction, run_training


def _write_sample(root, split, stem, width=64, height=64, thick=False):
    image_dir = root / split / 'images'
    mask_dir = root / split / 'masks'
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), np.uint8)
    if thick:
        mask[height // 2 - 3:height // 2 + 3, 8:width - 8] = 255
    else:
        mask[height // 2, 8:width - 8] = 255
        mask[height // 2:height - 8, width // 2] = 255
    Image.fromarray(image).save(image_dir / f'{stem}.png')
    Image.fromarray(mask).save(mask_dir / f'{stem}.png')


@pytest.fixture
def toy_split(tmp_path):
    root = tmp_path / 'split'
    _write_sample(root, 'train', 'TOY_train')
    _write_sample(root, 'val', 'TOY_val')
    with open(root / 'split_manifest.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['dataset', 'split', 'source_image',
                                'output', 'group'])
        writer.writeheader()
        writer.writerow({'dataset': 'TOY', 'split': 'train',
                         'source_image': 'TOY_train.png',
                         'output': 'TOY_train', 'group': 'TOY/0'})
        writer.writerow({'dataset': 'TOY', 'split': 'val',
                         'source_image': 'TOY_val.png',
                         'output': 'TOY_val', 'group': 'TOY/1'})
    return root


def _args(root, workdir, **overrides):
    values = dict(
        data_root=root,
        workdir=workdir,
        stage='train',
        device='cpu',
        epochs=2,
        batch_size=1,
        lr=2e-4,
        weight_decay=0.01,
        grad_clip=1.0,
        accum_steps=1,
        scheduler='official',
        warmup_epochs=None,
        selection='dice',
        seed=0,
        resume=False,
        save_optimizer=False,
        num_workers=0,
        limit_train=0,
        limit_val=0,
        use_wandb=False,
        wandb_mode='disabled',
        input_size=64,
        topoweave_refiner='graph',
        topoweave_graph_blocks=1,
        topoweave_graph_dim=32,
        topoweave_graph_k=4,
        topoweave_graph_stride=16,
        topoweave_base_width=8,
        topoweave_refiner_warmup=0,
        topoweave_mask_ratio=0.35,
        topoweave_mask_jitter=0.05,
        topoweave_loss_dice=1.0,
        topoweave_loss_topo=0.5,
        topoweave_coarse_weight=0.4,
        topoweave_topo_iters=2,
        topoweave_no_strong_augment=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _sample(tmp_path, thick=False):
    _write_sample(tmp_path, 'split', 'TOY_0', thick=thick)
    return Sample(
        image_path=tmp_path / 'split' / 'images' / 'TOY_0.png',
        mask_path=tmp_path / 'split' / 'masks' / 'TOY_0.png',
        dataset='TOY', stem='TOY_0')


def test_registry_and_metadata():
    assert 'topoweave' in available_adapters()
    adapter = load_adapter('topoweave')
    assert isinstance(adapter, TopoWeaveAdapter)
    assert adapter.id == 'topoweave'
    assert adapter.display_name == 'TopoWeave'
    assert adapter.input_size == 512
    assert adapter.official_lr == pytest.approx(2e-4)
    assert any('not an official baseline' in item
               for item in adapter.deviations)


def test_model_forward_backward_and_refiner_switch():
    model = TopoWeaveNet(
        input_size=64, base_width=8, refiner='graph', graph_blocks=1,
        graph_dim=32, graph_k=4)
    images = torch.rand(2, 3, 64, 64)
    outputs = model(images, return_aux=True)
    assert outputs['logits'].shape == (2, 1, 64, 64)
    assert outputs['coarse_logits'].shape == (2, 1, 64, 64)
    assert torch.mean(torch.abs(
        outputs['logits'] - outputs['coarse_logits'])) < 0.5
    loss = TopoWeaveLoss(topo_iters=2)(outputs, torch.zeros_like(
        outputs['logits']))
    loss.backward()
    assert model.stem.block[0].weight.grad is not None
    assert model.graph['blocks'][0].src.weight.grad is not None

    plain = TopoWeaveNet(
        input_size=64, base_width=8, refiner='none', graph_blocks=1,
        graph_dim=32, graph_k=4)
    out = plain(images, return_aux=True)
    assert torch.equal(out['logits'], out['coarse_logits'])


def test_full_parameter_count_is_compact():
    model = TopoWeaveNet(
        input_size=128, base_width=32, refiner='graph',
        graph_blocks=4, graph_dim=128, graph_k=9)
    total = sum(p.numel() for p in model.parameters())
    assert 6_000_000 < total < 14_000_000


def test_config_roundtrip():
    adapter = TopoWeaveAdapter()
    args = _args(None, None)
    adapter._apply_arch(args)
    config = adapter.save_config(args)
    target = TopoWeaveAdapter()
    restored_args = _args(None, None)
    target.restore_config(restored_args, config)
    assert target.input_size == 64
    assert target._refiner == 'graph'
    assert target._graph_blocks == 1
    assert target._graph_dim == 32
    assert target._graph_k == 4
    assert target._base_width == 8
    assert target._mask_ratio == pytest.approx(0.35)
    assert restored_args.topoweave_graph_k == 4


def test_width_aware_topomask_invariants():
    image = np.random.default_rng(1).integers(
        0, 255, (64, 64, 3), dtype=np.uint8)
    label = np.zeros((64, 64), np.uint8)
    label[32, 8:56] = 255
    label[32:56, 32] = 255
    original = label.copy()
    changed = width_aware_topomask(
        image, label, random.Random(3), 0.35)
    assert np.array_equal(label, original)
    assert np.any(changed != image)
    assert np.array_equal(
        width_aware_topomask(image, label, random.Random(3), 0.0), image)
    empty = np.zeros((32, 32), np.uint8)
    assert np.array_equal(
        width_aware_topomask(image[:32, :32], empty, random.Random(0), 0.5),
        image[:32, :32])

    thick = np.zeros((64, 64), np.uint8)
    thick[28:36, 8:56] = 255
    thin_changed = np.count_nonzero(changed != image)
    thick_changed = np.count_nonzero(
        width_aware_topomask(image, thick, random.Random(3), 0.35) != image)
    assert thick_changed > thin_changed
    assert 0.0 <= topo_ratio(50, 100, rng=random.Random(0)) <= 0.5


def test_loss_is_bounded_and_empty_safe():
    criterion = TopoWeaveLoss(topo_iters=2)
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 12:20, 8:24] = 1
    random_logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    loss = criterion({'logits': random_logits,
                      'coarse_logits': random_logits}, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert random_logits.grad is not None

    empty = torch.zeros(2, 1, 32, 32)
    empty_loss = criterion({'logits': random_logits,
                            'coarse_logits': random_logits}, empty)
    assert torch.isfinite(empty_loss)

    skeleton = (torch.rand(2, 1, 32, 32) > 0.8).float()
    weighted = skeleton_weighted_bce(
        random_logits, target, skeleton, weight=3.0)
    assert torch.isfinite(weighted)


def test_checkpoint_eligibility_during_warmup():
    adapter = TopoWeaveAdapter()
    args = _args(None, None, topoweave_refiner_warmup=1)
    adapter._apply_arch(args)
    assert not adapter.checkpoint_eligible(1)
    assert adapter.checkpoint_eligible(2)
    assert adapter.checkpoint_eligible(3)
    plain = TopoWeaveAdapter()
    plain._apply_arch(_args(None, None, topoweave_refiner='none'))
    assert plain.checkpoint_eligible(1)


def test_augment_and_ratio_are_deterministic():
    rng = np.random.default_rng(2)
    image = Image.fromarray(
        rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    label = Image.fromarray(np.zeros((64, 64), np.uint8))
    first = augment_pair(image, label, random.Random(7))
    second = augment_pair(image, label, random.Random(7))
    assert np.array_equal(np.asarray(first[0]), np.asarray(second[0]))


def test_tiny_cpu_training_and_prediction(toy_split, tmp_path):
    adapter = TopoWeaveAdapter()
    args = _args(toy_split, tmp_path / 'run')
    checkpoint = run_training(adapter, args)
    assert checkpoint.is_file()
    assert (tmp_path / 'run' / 'metrics_history.json').is_file()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['adapter'] == 'topoweave'
    assert payload['config']['adapter_config']['refiner'] == 'graph'
    assert payload['config']['adapter_config']['input_size'] == 64

    pred_args = _args(toy_split, tmp_path / 'run')
    pred_args.stage = 'predict'
    pred_args.pred_root = tmp_path / 'pred'
    pred_args.test_datasets = [f'TOY:{toy_split / "val"}']
    pred_args.checkpoint = checkpoint
    run_prediction(adapter, pred_args, checkpoint)
    assert (tmp_path / 'pred' / 'TOY' / 'TOY_val.png').is_file()
    assert (tmp_path / 'pred' / '.prediction_meta.json').is_file()
