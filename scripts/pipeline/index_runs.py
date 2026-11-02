#!/usr/bin/env python3
"""Build one seed-aware index over SACM-X and baseline run manifests.

The two pipelines keep their original storage layouts (baseline predictions
and paper tables vs SACM-X per-dataset metrics).  This script does not move
or rewrite either layout; it reads ``run_manifest.json`` and emits one CSV so
a user can locate every method/seed/checkpoint/metrics file in one place.

Usage:
    python scripts/pipeline/index_runs.py \
        --roots results/paper results/sacm_x \
        --out results/runs_index.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    'family', 'method', 'seed', 'status', 'checkpoint', 'best_epoch',
    'eval_root', 'metrics_csv', 'macro_csv', 'manifest',
]


def _load(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _best_epoch(checkpoint: str | None) -> int | None:
    if not checkpoint:
        return None
    info = Path(checkpoint).parent / 'run_info.json'
    payload = _load(info)
    if payload is None:
        return None
    value = payload.get('best_epoch')
    return int(value) if value is not None else None


def _baseline_row(manifest: dict, path: Path) -> dict:
    checkpoint = manifest.get('checkpoint')
    return {
        'family': 'baseline',
        'method': manifest.get('method', path.parent.name),
        'seed': manifest.get('seed'),
        'status': manifest.get('status', 'unknown'),
        'checkpoint': checkpoint,
        'best_epoch': _best_epoch(checkpoint),
        'eval_root': manifest.get('paper_root', str(path.parent)),
        'metrics_csv': manifest.get('paper_metrics'),
        'macro_csv': manifest.get('paper_macro'),
        'manifest': str(path),
    }


def _sacm_x_row(manifest: dict, path: Path) -> dict:
    eval_root = manifest.get('output_root', str(path.parent))
    checkpoints = manifest.get('checkpoints') or {}
    checkpoint = (
        next(iter(checkpoints.values())) if len(checkpoints) == 1 else None)
    jobs = manifest.get('jobs') or []
    statuses = {str(row.get('status', '')) for row in jobs}
    if statuses and statuses <= {'done', 'train_only'}:
        status = 'done'
    elif 'failed' in ' '.join(statuses).lower():
        status = 'failed'
    else:
        status = 'unknown'
    return {
        'family': 'sacm_x',
        'method': 'sacm_x',
        'seed': manifest.get('seed'),
        'status': status,
        'checkpoint': checkpoint,
        'best_epoch': _best_epoch(checkpoint),
        'eval_root': eval_root,
        'metrics_csv': str(Path(eval_root) / 'tables' / 'all_metrics.csv'),
        'macro_csv': str(Path(eval_root) / 'tables' / 'macro_metrics.csv'),
        'manifest': str(path),
    }


def _legacy_baseline_rows(root: Path) -> list[dict]:
    """Index pre-manifest baseline runs: <method>/seed_<seed>/best_model.pth."""
    rows: list[dict] = []
    for checkpoint in sorted(root.glob('*/seed_*/best_model.pth')):
        seed_dir = checkpoint.parent
        method_dir = seed_dir.parent
        try:
            seed = int(seed_dir.name.rsplit('_', 1)[-1])
        except ValueError:
            continue
        rows.append({
            'family': 'baseline',
            'method': method_dir.name,
            'seed': seed,
            'status': 'legacy',
            'checkpoint': str(checkpoint),
            'best_epoch': _best_epoch(str(checkpoint)),
            'eval_root': str(seed_dir),
            'metrics_csv': str(seed_dir / 'paper_metrics.csv'),
            'macro_csv': str(seed_dir / 'paper_macro.csv'),
            'manifest': '',
        })
    return rows


def collect(roots: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for root in roots:
        if not root.is_dir():
            print(f'[warn] root not found: {root}')
            continue
        for path in sorted(root.rglob('run_manifest.json')):
            manifest = _load(path)
            if manifest is None:
                print(f'[warn] unreadable manifest: {path}')
                continue
            family = str(manifest.get('family', '')).lower()
            if family == 'baseline':
                rows.append(_baseline_row(manifest, path))
            elif family in ('sacm_x', 'sacmx') or 'presets' in manifest:
                rows.append(_sacm_x_row(manifest, path))
            else:
                print(f'[warn] unknown manifest family in {path}')
        rows.extend(_legacy_baseline_rows(root))

    seen: set[str] = set()
    unique: list[dict] = []
    for row in rows:
        key = str(row.get('checkpoint') or '') + '|' + str(
            row.get('metrics_csv') or '')
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    unique.sort(key=lambda row: (
        str(row.get('family', '')),
        str(row.get('method', '')),
        -1 if row.get('seed') is None else int(row['seed']),
        str(row.get('eval_root', '')),
    ))
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', type=Path,
                        default=[Path('results/paper'),
                                 Path('results/sacm_x')])
    parser.add_argument('--out', type=Path,
                        default=Path('results/runs_index.csv'))
    args = parser.parse_args()

    rows = collect(args.roots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'indexed {len(rows)} run(s) -> {args.out}')
    if not rows:
        print('No run_manifest.json found. Run training/evaluation with the '
              'current scripts first.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
