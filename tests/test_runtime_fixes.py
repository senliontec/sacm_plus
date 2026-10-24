"""Regression tests for issues found in the full porting audit."""

import argparse
import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.registry import load_adapter
from baseline_zoo.data import load_split_samples
from baseline_zoo.protocols import load_protocols
from baseline_zoo.runtime import binary_metrics, build_parser
from core.metrics import cldice_score, dice_score, hd95_score, iou_score


def _write_pair(root: Path, stem: str, size=(16, 16)):
    image_dir, mask_dir = root / 'images', root / 'masks'
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[4:12, 4:12] = 200
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    Image.fromarray(image).save(image_dir / f'{stem}.png')
    Image.fromarray(mask).save(mask_dir / f'{stem}.png')


def _manifest(root: Path, rows):
    with (root / 'split_manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['dataset', 'split', 'source_image',
                                'output', 'group'])
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_membership_rejects_stray_and_missing(tmp_path):
    train = tmp_path / 'train'
    for stem in ('DRIVE_a', 'DRIVE_b', 'DRIVE_c'):
        _write_pair(train, stem)
    _manifest(tmp_path, [
        {'dataset': 'DRIVE', 'split': 'train', 'source_image': 'a.tif',
         'output': 'DRIVE_a', 'group': 'DRIVE/a'},
    ])
    with pytest.raises(SystemExit):
        load_split_samples(tmp_path, 'train')

    _manifest(tmp_path, [
        {'dataset': 'DRIVE', 'split': 'train', 'source_image': 'a.tif',
         'output': 'DRIVE_a', 'group': 'DRIVE/a'},
        {'dataset': 'DRIVE', 'split': 'train', 'source_image': 'b.tif',
         'output': 'DRIVE_b', 'group': 'DRIVE/b'},
        {'dataset': 'DRIVE', 'split': 'train', 'source_image': 'c.tif',
         'output': 'DRIVE_c', 'group': 'DRIVE/c'},
    ])
    samples = load_split_samples(tmp_path, 'train')
    assert [sample.stem for sample in samples] == [
        'DRIVE_a', 'DRIVE_b', 'DRIVE_c']


def test_fixed_protocol_resolves_frozen_split():
    protocol = load_protocols()['fixed']
    split = protocol.split_for(42)
    assert split.split_root.name == 'sacm_3shot'
    assert split.test_manifest.name == 'protocol_manifest.csv'


def test_sacm_official_clip_default_is_effective():
    parser = build_parser(load_adapter('sacm_official'))
    assert parser.parse_args([]).grad_clip == 1.0
    assert parser.parse_args(['--grad-clip', '0.25']).grad_clip == 0.25


def test_cs2net_schedule_does_not_decay_before_first_epoch():
    adapter = load_adapter('cs2net')
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = adapter.make_scheduler(
        optimizer, argparse.Namespace(epochs=4))
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-4)
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-4)
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] < 1e-4


def test_sam_octa_does_not_prepad_non_square_image(tmp_path):
    _write_pair(tmp_path, 'LES-AV_1', size=(60, 80))
    from baseline_zoo.data import RawPairDataset
    sample = RawPairDataset(tmp_path).samples[0]
    adapter = load_adapter('sam_octa')
    adapter.args = argparse.Namespace()
    adapter.device = torch.device('cpu')
    prepared = adapter.prepare_eval_sample(sample, torch.device('cpu'))
    # Longest side is resized to 1024; the short side must stay unpadded.
    assert prepared['image'].shape == (1, 3, 1024, 768)
    assert prepared['original_size'] == (80, 60)


def test_sam_vit_l_unet_state_dict_is_decoder_only():
    adapter = load_adapter('sam_vit_l_unet')

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.decoder = torch.nn.Linear(2, 2)

    model = Dummy()
    state = adapter.state_dict(model)
    assert all(not key.startswith('encoder.') for key in state)
    assert all(key in model.decoder.state_dict() for key in state)


def test_knowsam_unlabeled_pool_excludes_protocol_images(tmp_path):
    adapter = load_adapter('knowsam')
    root = tmp_path / 'stage'
    adapter._stage(root, Path('data/sacm_3shot_seed0'), 8)
    staged = {path.name for path in (root / 'unlabeled' / 'image').glob('*')}
    manifest = Path('data/sacm_3shot_seed0/split_manifest.csv')
    with manifest.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['split'] not in ('train', 'val'):
                continue
            name = f"{row['dataset']}_{Path(row['source_image']).stem}_image.png"
            assert name not in staged


def test_segdino_restores_backbone_config():
    adapter = load_adapter('segdino')
    args = argparse.Namespace(
        dino_size='large', dino_ckpt=Path('weights.pth'), random_init=False)
    saved = adapter.save_config(args)
    target = argparse.Namespace(
        dino_size='small', dino_ckpt=None, random_init=False)
    adapter.restore_config(target, saved)
    assert target.dino_size == 'large'
    assert target.random_init is True
    assert target.dino_ckpt is None


def test_binary_metrics_match_canonical_paper_metrics():
    pred = np.zeros((32, 32), dtype=bool)
    gt = np.zeros((32, 32), dtype=bool)
    pred[4:20, 5:18] = True
    gt[6:22, 8:20] = True
    metrics = binary_metrics(pred, gt)
    assert metrics['dice'] == dice_score(pred, gt)
    assert metrics['iou'] == iou_score(pred, gt)
    assert metrics['cldice'] == cldice_score(pred, gt)
    assert metrics['hd95'] == hd95_score(pred, gt)


def test_nnwnet_discovers_checkpoint_extension(tmp_path):
    adapter = load_adapter('nnwnet')
    fold = tmp_path / 'fold_0'
    fold.mkdir()
    (fold / 'checkpoint_best.pth').write_bytes(b'pth')
    assert adapter._find_checkpoint(tmp_path, 'best').name == 'checkpoint_best.pth'
    (fold / 'checkpoint_best.pth').unlink()
    (fold / 'checkpoint_best.safetensors').write_bytes(b'safe')
    assert adapter._find_checkpoint(tmp_path, 'best').name == \
        'checkpoint_best.safetensors'
