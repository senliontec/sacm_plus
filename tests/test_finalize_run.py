"""Canonical results-layout tests for finalize_run.py."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'scripts' / 'pipeline'))

import finalize_run  # noqa: E402


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_info(path: Path, **extra) -> None:
    payload = {
        'family': 'baseline',
        'method': 'unet',
        'seed': 0,
        'best_epoch': 7,
        'last_epoch': 10,
        'selection': 'dice',
        'epochs': 10,
    }
    payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + '\n')


def _default_args(**overrides):
    values = {
        'family': 'baseline',
        'method': 'unet',
        'variant': 'default',
        'seed': 0,
        'source_train': None,
        'source_eval': None,
        'checkpoint': None,
        'out_root': None,
        'cleanup': True,
        'protocol': None,
        'pred_root': None,
        'status': None,
    }
    values.update(overrides)
    return finalize_run.argparse.Namespace(**values)


def _write_baseline_source(root: Path) -> tuple[Path, Path]:
    run_root = root / 'unet' / 'default' / 'seed_0'
    run_root.mkdir(parents=True)
    (run_root / 'best_model.pth').write_bytes(b'checkpoint')
    _write_run_info(run_root / 'run_info.json')
    (run_root / 'metrics_history.json').write_text('{"history": []}\n')
    (run_root / 'train.log').write_text('train\n')
    (run_root / 'eval.log').write_text('eval\n')
    eval_root = run_root / 'eval'
    _write_csv(
        eval_root / 'tables' / 'all_metrics.csv',
        ['family', 'method', 'variant', 'seed', 'dataset', 'role',
         'dice_mean', 'dice_std', 'dice_n',
         'iou_mean', 'iou_std', 'iou_n',
         'cldice_mean', 'cldice_std', 'cldice_n',
         'hd95_mean', 'hd95_std', 'hd95_n'],
        [
            {'family': 'baseline', 'method': 'unet', 'variant': 'default',
             'seed': 0, 'dataset': 'DRIVE', 'role': 'Seen',
             'dice_mean': 0.8, 'dice_std': 0.01, 'dice_n': 2,
             'iou_mean': 0.7, 'iou_std': 0.01, 'iou_n': 2,
             'cldice_mean': 0.82, 'cldice_std': 0.01, 'cldice_n': 2,
             'hd95_mean': 5.0, 'hd95_std': 0.5, 'hd95_n': 2},
            {'family': 'baseline', 'method': 'unet', 'variant': 'default',
             'seed': 0, 'dataset': 'WIRE', 'role': 'Novel',
             'dice_mean': 0.5, 'dice_std': 0.02, 'dice_n': 2,
             'iou_mean': 0.4, 'iou_std': 0.02, 'iou_n': 2,
             'cldice_mean': 0.6, 'cldice_std': 0.02, 'cldice_n': 2,
             'hd95_mean': 60.0, 'hd95_std': 1.0, 'hd95_n': 2},
        ])
    for dataset in ('DRIVE', 'WIRE'):
        role = 'Seen' if dataset == 'DRIVE' else 'Novel'
        _write_csv(
            eval_root / dataset / 'metrics.csv',
            ['dataset', 'image', 'role', 'dice', 'iou', 'cldice', 'hd95'],
            [
                {'dataset': dataset, 'image': 'a', 'role': role,
                 'dice': 0.8, 'iou': 0.7, 'cldice': 0.82, 'hd95': 5.0},
                {'dataset': dataset, 'image': 'b', 'role': role,
                 'dice': 0.8, 'iou': 0.7, 'cldice': 0.82, 'hd95': 5.0},
            ])
        (eval_root / dataset / 'results_summary.txt').write_text(
            'Num Samples: 2\n')
    return run_root, eval_root


def test_finalize_baseline_in_place(tmp_path):
    run_root, eval_root = _write_baseline_source(tmp_path / 'runs')
    args = _default_args(source_train=run_root, source_eval=eval_root,
                         out_root=tmp_path / 'runs')
    target = finalize_run.finalize(args)

    assert target == run_root
    assert (target / 'best_model.pth').is_file()
    assert (target / 'run_info.json').is_file()
    assert (target / 'metrics_history.json').is_file()
    assert (target / 'train.log').is_file()
    assert (target / 'eval' / 'eval.log').is_file()
    assert (target / 'eval' / 'jobs.csv').is_file()
    assert (target / 'eval' / 'run_manifest.json').is_file()
    assert (target / 'eval' / 'DRIVE' / 'metrics.csv').is_file()
    assert (target / 'eval' / 'DRIVE' / 'summary.csv').is_file()

    with open(target / 'eval' / 'tables' / 'all_metrics.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    assert {row['dataset'] for row in rows} == {'DRIVE', 'WIRE'}
    assert {row['variant'] for row in rows} == {'default'}
    assert {row['seed'] for row in rows} == {'0'}

    with open(target / 'eval' / 'tables' / 'macro_metrics.csv',
              newline='') as f:
        macro = list(csv.DictReader(f))
    assert 'All' in {row['role'] for row in macro}

    manifest = json.loads(
        (target / 'eval' / 'run_manifest.json').read_text())
    assert manifest['status'] == 'done'
    assert manifest['checkpoint_kind'] == 'file'
    assert not (run_root / 'eval.log').exists()


def test_finalize_sacm_x_promotes_and_flattens(tmp_path):
    out_root = tmp_path / 'runs'
    run_root = out_root / 'sacm_x' / 'full_typed' / 'seed_0'
    run_root.mkdir(parents=True)
    # A previous finalize left an old checkpoint at the run root.  The fresh
    # per-preset training output must win.
    (run_root / 'best_model.pth').write_bytes(b'old-checkpoint')
    raw = run_root / 'full_typed'
    raw.mkdir(parents=True)
    (raw / 'best_model.pth').write_bytes(b'new-checkpoint')
    _write_run_info(raw / 'run_info.json', family='sacm_x',
                    method='sacm_x', preset='full_typed')
    (raw / 'metrics_history.json').write_text('{"history": []}\n')
    (raw / 'run.log').write_text('train\n')
    eval_root = run_root / 'eval'
    (eval_root / 'full_typed' / 'DRIVE').mkdir(parents=True)
    (eval_root / 'full_typed' / 'DRIVE' / 'metrics.csv').write_text(
        'image,dice,iou,cldice,hd95\na,0.8,0.7,0.82,5.0\n')
    (eval_root / 'full_typed' / 'DRIVE' / 'results_summary.txt').write_text(
        'Num Samples: 1\n')
    (eval_root / 'full_typed' / 'run_eval.log').write_text('eval\n')
    _write_csv(
        eval_root / 'tables' / 'all_metrics.csv',
        ['preset', 'dataset', 'role', 'dice_mean', 'dice_std',
         'iou_mean', 'iou_std', 'cldice_mean', 'cldice_std',
         'hd95_mean', 'hd95_std'],
        [{'preset': 'full_typed', 'dataset': 'DRIVE', 'role': 'Seen',
          'dice_mean': 0.8, 'dice_std': 0.0,
          'iou_mean': 0.7, 'iou_std': 0.0,
          'cldice_mean': 0.82, 'cldice_std': 0.0,
          'hd95_mean': 5.0, 'hd95_std': 0.0}])
    _write_csv(eval_root / 'jobs_eval.csv',
               ['preset', 'dataset', 'gpu', 'status'],
               [{'preset': 'full_typed', 'dataset': 'DRIVE', 'gpu': '0',
                 'status': 'done'}])

    args = _default_args(family='sacm_x', method='sacm_x',
                         variant='full_typed', source_train=run_root,
                         source_eval=eval_root, out_root=out_root)
    target = finalize_run.finalize(args)

    assert target == run_root
    assert (target / 'best_model.pth').read_bytes() == b'new-checkpoint'
    assert (target / 'train.log').is_file()
    assert (target / 'eval' / 'DRIVE' / 'metrics.csv').is_file()
    assert (target / 'eval' / 'DRIVE' / 'summary.csv').is_file()
    assert (target / 'eval' / 'eval.log').is_file()
    assert (target / 'eval' / 'jobs.csv').is_file()
    assert not raw.exists()
    assert not (eval_root / 'full_typed').exists()
    assert not (eval_root / 'jobs_eval.csv').exists()

    with open(target / 'eval' / 'DRIVE' / 'metrics.csv', newline='') as f:
        row = next(csv.DictReader(f))
    assert row['dataset'] == 'DRIVE'
    assert row['role'] == 'Seen'


def test_explicit_missing_checkpoint_fails(tmp_path):
    run_root, eval_root = _write_baseline_source(tmp_path / 'runs')
    missing = tmp_path / 'does-not-exist.pth'
    args = _default_args(
        source_train=run_root, source_eval=eval_root,
        out_root=tmp_path / 'runs', checkpoint=missing)
    with pytest.raises(SystemExit):
        finalize_run.finalize(args)


def test_canonical_seed_spelling(tmp_path):
    path = finalize_run._run_root(
        'sacm_x', 'full_typed', 0, root=tmp_path)
    assert path.name == 'seed_0'
