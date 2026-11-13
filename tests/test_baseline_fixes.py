"""Regression tests for the 2026-11 baseline task-adaptation fixes."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'scripts' / 'pipeline'))

from baseline_zoo.data import RawPairDataset, load_mask  # noqa: E402
# Sample is intentionally not imported; tests build RawPairDataset pairs.
from baseline_zoo.method_groups import (  # noqa: E402
    group_for, validate_groups,
)
from baseline_zoo.registry import load_models  # noqa: E402
from baseline_zoo.report import load_run, write_report  # noqa: E402
from baseline_zoo.runtime import binary_metrics  # noqa: E402
from core.metrics import binary_metrics as core_binary_metrics  # noqa: E402
from metrics import compute_selected  # noqa: E402


def test_load_mask_accepts_literal_01_and_0255(tmp_path):
    for value in (1, 255):
        root = tmp_path / f'masks_{value}'
        root.mkdir()
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 2:5] = value
        path = root / 'mask.png'
        Image.fromarray(mask).save(path)
        loaded = load_mask(path)
        assert loaded.dtype == bool
        assert loaded.sum() == 9


def test_raw_pair_dataset_explicit_name_overrides_stem(tmp_path):
    root = tmp_path / 'FIVES_root'
    (root / 'images').mkdir(parents=True)
    (root / 'masks').mkdir()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 255
    Image.fromarray(image).save(root / 'images' / '100_D.png')
    Image.fromarray(mask).save(root / 'masks' / '100_D.png')
    sample = RawPairDataset(root, dataset_name='FIVES').samples[0]
    assert sample.dataset == 'FIVES'
    # Without the override the legacy stem-prefix convention still applies.
    assert RawPairDataset(root).samples[0].dataset == '100'


def test_binary_metrics_empty_penalty_is_finite():
    gt = np.zeros((10, 12), dtype=bool)
    gt[2:6, 3:6] = True
    pred = np.zeros_like(gt)
    zero = core_binary_metrics(pred, gt)
    assert zero['dice'] == 0.0
    assert zero['cldice'] == 0.0
    assert zero['hd95'] == pytest.approx(np.hypot(10, 12))
    assert zero['empty'] == 1.0

    both = core_binary_metrics(np.zeros_like(gt), np.zeros_like(gt))
    assert both['dice'] == 1.0
    assert both['cldice'] == 1.0
    assert both['hd95'] == 0.0

    raw = core_binary_metrics(pred, gt, empty_policy='nan')
    assert np.isnan(raw['cldice'])
    assert np.isnan(raw['hd95'])

    # The runtime wrapper and the paper collector use the same policy.
    assert binary_metrics(pred, gt)['hd95'] == pytest.approx(np.hypot(10, 12))
    selected = compute_selected(pred.astype(float), gt.astype(float),
                                ['dice', 'cldice', 'hd95'])
    assert selected['dice'] == 0.0
    assert selected['cldice'] == 0.0
    assert selected['hd95'] == pytest.approx(np.hypot(10, 12))


def test_scrwkv_threshold_uses_uint8_scale(tmp_path):
    from baseline_zoo.adapters.scrwkv import SCRWKVAdapter

    class Dummy(torch.nn.Module):
        def forward(self, image):
            return torch.tensor(
                [[[[0.5, 0.1], [-0.5, 1.0]]]], dtype=torch.float32)

    mask_path = tmp_path / 'mask.png'
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(mask_path)
    adapter = SCRWKVAdapter()
    adapter.args = SimpleNamespace(scrwkv_threshold=0.2 / 255.0)
    sample = SimpleNamespace(mask_path=mask_path)
    pred = adapter.predict_sample(
        Dummy(),
        {'image': torch.zeros(1, 3, 2, 2), 'size': (2, 2)},
        sample)
    # With the official 0.2/255 ratio, the 0.1 and 1.0 logits are positive;
    # the old 0.2-of-max interpretation would keep only 1.0.
    assert pred.shape == (2, 2)
    assert int(pred.sum()) == 3


def test_cwsam_loss_has_foreground_weight_and_dice():
    from baseline_zoo.adapters.cwsam import CWSAMAdapter

    class Dummy(torch.nn.Module):
        def infer(self, images):
            return torch.zeros(1, 2, 8, 8, requires_grad=True)

    adapter = CWSAMAdapter()
    adapter.args = SimpleNamespace(
        cwsam_bg_weight=0.1, cwsam_fg_weight=1.0, cwsam_dice_weight=0.5)
    masks = torch.zeros(1, 8, 8, dtype=torch.long)
    masks[:, 2:5, 2:5] = 1
    loss = adapter.train_step(Dummy(), (torch.zeros(1, 3, 8, 8), masks))
    assert torch.isfinite(loss)
    assert float(loss) > 0


def test_tpsegformer_main_head_gets_dice_term():
    from baseline_zoo.adapters.tpsegformer import TPSegformerAdapter

    class Dummy(torch.nn.Module):
        def forward(self, images):
            main = torch.zeros(
                1, 1, images.shape[-2], images.shape[-1],
                requires_grad=True)
            return main, main

    adapter = TPSegformerAdapter()
    adapter.args = SimpleNamespace(tpsegformer_main_dice_weight=0.5)
    gt = torch.zeros(1, 1, 16, 16)
    gt[:, :, 4:10, 4:10] = 1
    loss = adapter.train_step(Dummy(), {
        'image': torch.zeros(1, 3, 16, 16), 'gt': gt})
    assert torch.isfinite(loss)
    assert float(loss) > 0


def test_method_groups_cover_every_registered_model():
    assert validate_groups(load_models()) == []
    assert group_for('sacm_gcd') == 'proposed'
    assert group_for('ucs') == 'oracle'
    assert group_for('fssam') == 'support'
    assert group_for('c2p') == 'zero_shot'


def _write_metrics(run_root: Path, *, status: str | None = None):
    tables = run_root / 'eval' / 'tables'
    tables.mkdir(parents=True)
    with (tables / 'all_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['dataset', 'role', 'dice_mean', 'dice_std',
                        'iou_mean', 'iou_std', 'cldice_mean', 'cldice_std',
                        'hd95_mean', 'hd95_std'])
        writer.writeheader()
        writer.writerow({'dataset': 'DRIVE', 'role': 'Seen',
                         'dice_mean': 0.8, 'dice_std': 0.0,
                         'iou_mean': 0.7, 'iou_std': 0.0,
                         'cldice_mean': 0.82, 'cldice_std': 0.0,
                         'hd95_mean': 5.0, 'hd95_std': 0.0})
    if status is not None:
        (run_root / 'eval' / 'run_manifest.json').write_text(
            json.dumps({'status': status}) + '\n')


def test_report_skips_partial_manifest_and_separates_groups(tmp_path):
    partial = tmp_path / 'unet' / 'partial' / 'seed_0'
    _write_metrics(partial, status='partial')
    assert load_run(partial, 'unet', 0) == []
    assert load_run(partial, 'unet', 0, allow_partial=True)

    rows = []
    for model, variant in (('unet', 'default'), ('sacm_gcd', 'all_dense')):
        root = tmp_path / model / variant / 'seed_0'
        _write_metrics(root, status='done')
        rows.extend(load_run(root, model, 0))
    assert rows
    out = write_report(rows, tmp_path / 'tables')
    text = out['main_results'].read_text()
    assert '## Prompt-free supervised' in text
    assert '## Proposed method' in text
    assert 'unet' in text and 'sacm_gcd' in text
