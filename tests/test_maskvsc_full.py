"""Full two-stage MaskVSC (stage-1 UNet + stage-2 ViG) tests."""

from __future__ import annotations

import argparse
import csv
import math
import warnings

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from baseline_zoo.adapters.maskvsc import MaskVSCAdapter, _augment_pair
from baseline_zoo.data import Sample
from baseline_zoo.models.maskvsc import (
    Connect_Loss,
    MaskVSCFull,
    UNet,
    ViGSeg_Fundus,
    mask_vessel_corrupt,
)
from baseline_zoo.models.maskvsc.gcn import (
    _knn_graph,
    _release_tokens_to_feature_map,
    _tokens_to_feature_map,
)


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


def test_official_release_decode_forward_backward():
    features = torch.zeros(2, 64, 256)
    folded = _release_tokens_to_feature_map(features, 16)
    assert folded.shape == (2, 64, 16, 16)
    with pytest.raises(ValueError, match='release decode needs'):
        _release_tokens_to_feature_map(torch.zeros(1, 64, 8), 16)

    model = ViGSeg_Fundus(
        num_channels=1, img_size=128, patch_size=16, num_ViGBlocks=1,
        decode='release')
    images = torch.rand(1, 1, 128, 128)
    output = model(images)
    assert output.shape == (1, 1, 128, 128)
    loss = nn.BCELoss()(output, torch.zeros_like(output))
    loss.backward()
    assert model.backbone.blocks[0].in_layer1.layer[0].weight.grad is not None


def test_official_fundus_vig_parameter_count():
    model = ViGSeg_Fundus(
        num_channels=1, img_size=512, patch_size=16, num_ViGBlocks=16,
        decode='release')
    assert model.predictor[0].in_channels == 1024
    assert model.backbone.blocks[0].graph_metric == 'paper'
    assert sum(p.numel() for p in model.parameters()) == 23_841_764


def test_paper_and_release_knn_graphs_differ():
    features = torch.tensor([[[1.0, 0.0], [1.0, 0.1], [2.0, 0.0]]])
    paper = _knn_graph(features, 2, 'paper')
    release = _knn_graph(features, 2, 'release')
    assert set(paper[0, 0].tolist()) == {0, 1}
    assert set(release[0, 0].tolist()) == {0, 2}
    with pytest.raises(ValueError, match='unknown ViG graph metric'):
        _knn_graph(features, 2, 'cosine')


def test_paper_connect_loss_is_scale_comparable():
    torch.manual_seed(0)
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., 20:44, 24:40] = 1
    pred = (0.5 * mask + 0.5 * torch.rand_like(mask)).clamp(0, 1)
    paper = Connect_Loss(mode='paper')(pred, mask)
    release = Connect_Loss(mode='release')(pred, mask)
    capped = Connect_Loss(mode='paper', cap=0.5)(pred, mask)
    bce = torch.nn.BCELoss()(pred, mask)
    assert paper.item() > 0.01
    assert paper.item() > release.item() * 1e4
    assert paper.item() < bce.item() * 50
    assert capped.item() <= 0.5
    with pytest.raises(ValueError, match='unknown Connect_Loss mode'):
        Connect_Loss(mode='cosine')
    with pytest.raises(ValueError, match='cap must be >= 0'):
        Connect_Loss(mode='paper', cap=-1)
    empty = torch.zeros(1, 1, 32, 32)
    assert Connect_Loss(mode='paper')(
        torch.rand_like(empty), empty).item() == 0.0


def test_paper_connect_loss_penalizes_empty_thin_structure():
    # The released 3x3 opening erases a one-pixel vessel, making an empty
    # prediction have zero L_C. Paper mode must skip the opening.
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 32, 8:56] = 1
    empty = torch.zeros_like(mask)
    paper = Connect_Loss(mode='paper')(empty, mask)
    release = Connect_Loss(mode='release')(empty, mask)
    assert paper.item() > 0.01
    assert release.item() < 1e-5


class _FixedRng:
    def __init__(self, angle, flip):
        self.angle = angle
        self.flip = flip

    def uniform(self, low, high):
        assert low == -high
        return self.angle

    def random(self):
        return 0.0 if self.flip else 1.0


def test_mask_vessel_corrupt_silences_skan_separator_warning():
    # skan warns once per augmented sample when separator is left implicit.
    image = np.full((64, 64, 3), 128, np.uint8)
    label = np.zeros((64, 64), np.uint8)
    label[8:56, 32] = 255
    label[32, 32:56] = 255
    with warnings.catch_warnings():
        warnings.simplefilter(
            'error', np.exceptions.VisibleDeprecationWarning)
        corrupted = mask_vessel_corrupt(image, label, 0.5)
    assert corrupted.shape == image.shape
    assert corrupted.dtype == np.uint8


def test_paper_rotation_and_flip_augmentation_are_exact():
    rng = np.random.default_rng(3)
    image = Image.fromarray(
        rng.integers(0, 255, (32, 40, 3), dtype=np.uint8))
    mask = np.zeros((32, 40), np.uint8)
    mask[8:24, 12:28] = 255
    label = Image.fromarray(mask)

    rotated_image, rotated_label = _augment_pair(
        image, label, _FixedRng(30.0, False), 30.0)
    expected_image = image.rotate(
        30.0, resample=Image.BILINEAR, expand=False,
        fillcolor=(0, 0, 0))
    expected_label = label.rotate(
        30.0, resample=Image.NEAREST, expand=False, fillcolor=0)
    assert np.array_equal(np.asarray(rotated_image),
                          np.asarray(expected_image))
    assert np.array_equal(np.asarray(rotated_label),
                          np.asarray(expected_label))
    assert set(np.unique(np.asarray(rotated_label))).issubset({0, 255})

    flipped_image, flipped_label = _augment_pair(
        image, label, _FixedRng(0.0, True), 0.0)
    assert np.array_equal(
        np.asarray(flipped_image),
        np.asarray(image.transpose(Image.FLIP_LEFT_RIGHT)))
    assert np.array_equal(
        np.asarray(flipped_label),
        np.asarray(label.transpose(Image.FLIP_LEFT_RIGHT)))

    same_image, same_label = _augment_pair(
        image, label, _FixedRng(0.0, False), 0.0)
    assert np.array_equal(np.asarray(same_image), np.asarray(image))
    assert np.array_equal(np.asarray(same_label), np.asarray(label))


def test_coherent_decode_remains_an_explicit_ablation():
    with pytest.raises(ValueError, match='not a square'):
        _tokens_to_feature_map(torch.zeros(1, 5, 8))
    model = ViGSeg_Fundus(
        num_channels=1, img_size=128, patch_size=32, num_ViGBlocks=1,
        decode='coherent')
    output = model(torch.rand(1, 1, 128, 128))
    assert output.shape == (1, 1, 128, 128)


def test_full_wrapper_stage_transitions():
    seg = UNet(n_channels=3, n_classes=1)
    vig = ViGSeg_Fundus(
        num_channels=1, img_size=128, patch_size=16, num_ViGBlocks=1,
        decode='release')
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
        maskvsc_vig_patch_size=16,
        maskvsc_vig_decode='release',
        maskvsc_vig_graph='release',
        maskvsc_connect_loss='release',
        maskvsc_connect_cap=0.0,
        maskvsc_rotation_degrees=0.0,
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


def test_maskvsc_uses_custom_two_stage_pipeline():
    # The shared runtime trains one model and ignores stage2_epochs; without
    # this flag MaskVSC never trains the ViG and prediction uses random ViG.
    assert MaskVSCAdapter.custom_pipeline is True


def test_maskvsc_wandb_logs_per_dataset_scalars():
    # The custom two-stage loop replaces the shared runtime, so it must
    # reproduce its val/<dataset>/<metric> W&B schema itself.
    scalars = MaskVSCAdapter._add_validation_scalars(
        {'epoch': 3},
        {'dice': 0.5, 'iou': 0.4, 'cldice': 0.3, 'hd95': 20.0},
        {'DRIVE': {'dice': 0.6, 'iou': 0.5, 'cldice': 0.4, 'hd95': 10.0},
         'CREMI': {'dice': 0.4, 'iou': 0.3, 'cldice': 0.2, 'hd95': 30.0}})
    assert scalars['epoch'] == 3
    assert scalars['val/dice'] == 0.5
    assert scalars['val/DRIVE/dice'] == 0.6
    assert scalars['val/DRIVE/iou'] == 0.5
    assert scalars['val/DRIVE/cldice'] == 0.4
    assert scalars['val/DRIVE/hd95'] == 10.0
    assert scalars['val/CREMI/dice'] == 0.4
    assert scalars['val/CREMI/hd95'] == 30.0


def test_maskvsc_main_defaults_are_release():
    parser = argparse.ArgumentParser()
    MaskVSCAdapter().add_args(parser)
    args = parser.parse_args([])
    assert args.maskvsc_connect_loss == 'release'
    assert args.maskvsc_connect_cap == 0.0
    assert args.maskvsc_rotation_degrees == 0.0
    assert args.maskvsc_vig_graph == 'release'
    assert args.maskvsc_vig_patch_size == 16
    assert args.maskvsc_vig_decode == 'release'


def test_paper_mode_config_roundtrip():
    adapter = MaskVSCAdapter()
    args = _training_args(None, None)
    args.maskvsc_connect_loss = 'paper'
    args.maskvsc_connect_cap = 1.0
    args.maskvsc_rotation_degrees = 30.0
    args.maskvsc_vig_graph = 'paper'
    adapter._apply_arch(args)
    config = adapter.save_config(args)
    target = _training_args(None, None)
    adapter.restore_config(target, config)
    assert target.maskvsc_connect_loss == 'paper'
    assert target.maskvsc_connect_cap == 1.0
    assert target.maskvsc_rotation_degrees == 30.0
    assert target.maskvsc_vig_graph == 'paper'


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
    assert payload['config']['vig_arch']['decode'] == 'release'
    assert payload['config']['vig_arch']['graph_metric'] == 'release'
    assert payload['config']['adapter_config']['vig_decode'] == 'release'
    assert payload['config']['adapter_config']['vig_graph'] == 'release'
    assert payload['config']['adapter_config']['connect_loss'] == 'release'
    assert payload['config']['adapter_config']['connect_cap'] == 0.0
    assert payload['config']['adapter_config']['rotation_degrees'] == 0.0
    assert payload['split_manifest_sha256']
    assert (workdir / 'metrics_history.json').is_file()
    assert (workdir / 'val_metrics_by_dataset.csv').is_file()

    adapter.prediction_mode = True
    pred_args = _training_args(split_root, workdir)
    # Simulate a fresh process: restore the architecture from the checkpoint.
    pred_args.maskvsc_vig_patch_size = 32
    pred_args.maskvsc_vig_decode = 'coherent'
    pred_args.maskvsc_vig_graph = 'paper'
    pred_args.maskvsc_connect_loss = 'paper'
    pred_args.maskvsc_connect_cap = 1.0
    pred_args.maskvsc_rotation_degrees = 30.0
    adapter.restore_config(pred_args, payload['config']['adapter_config'])
    assert pred_args.maskvsc_vig_patch_size == 16
    assert pred_args.maskvsc_vig_decode == 'release'
    assert pred_args.maskvsc_vig_graph == 'release'
    assert pred_args.maskvsc_connect_loss == 'release'
    assert pred_args.maskvsc_connect_cap == 0.0
    assert pred_args.maskvsc_rotation_degrees == 0.0
    model = adapter.build_model(pred_args)
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


def test_stage2_uses_official_soft_labels(tmp_path):
    image = np.full((41, 53, 3), 128, np.uint8)
    mask = np.zeros((41, 53), np.uint8)
    mask[10:30, 15:40] = 255
    image_path = tmp_path / 'case.png'
    mask_path = tmp_path / 'case_mask.png'
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)
    adapter = MaskVSCAdapter()
    adapter.input_size = 64
    sample = Sample(image_path, mask_path, 'toy', 'case')
    _, labels = adapter._prepare_clean_batch(
        [sample], torch.device('cpu'))
    assert labels.shape == (1, 1, 64, 64)
    assert labels.max().item() == 1.0
    assert labels.min().item() == 0.0
    # Official GCN labels are PIL-resized soft masks, not thresholded.
    assert torch.any((labels > 0) & (labels < 1))


def test_legacy_patch32_checkpoint_infers_coherent_decode():
    adapter = MaskVSCAdapter()
    args = _training_args(None, None)
    adapter.restore_config(args, {'vig_patch_size': 32})
    assert adapter._vig_decode == 'coherent'
    assert args.maskvsc_vig_decode == 'coherent'
    assert adapter._vig_graph == 'release'
    assert adapter._connect_loss_mode == 'release'
    assert adapter._rotation_degrees == 0.0
    assert args.maskvsc_vig_graph == 'release'
    assert args.maskvsc_connect_loss == 'release'
    assert args.maskvsc_connect_cap == 0.0
    assert args.maskvsc_rotation_degrees == 0.0
    config = adapter.save_config(args)
    assert config['vig_decode'] == 'coherent'
    assert config['vig_graph'] == 'release'
