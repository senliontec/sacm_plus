#!/usr/bin/env python3
"""Build one seed-aware index over canonical baseline/SACM-X run manifests.

Every current run lives at ``results/runs/<method>/<variant>/seed_<seed>``.
This script reads each ``eval/run_manifest.json`` (and legacy manifests when
explicitly requested) and emits one CSV so a user can locate every
method/variant/seed checkpoint and metrics table in one place.

Usage:
    python scripts/pipeline/index_runs.py
    # legacy migration: --roots results/runs results/paper results/sacm_x
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    'family', 'method', 'variant', 'seed', 'status', 'checkpoint',
    'best_epoch', 'eval_root', 'metrics_csv', 'macro_csv', 'manifest',
    'commit', 'code_fingerprint',
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


def _code_info(manifest: dict, path: Path) -> tuple[str | None, str | None]:
    """Read commit/code_fingerprint from the run's provenance directory."""
    candidates: list[Path] = []
    checkpoint = manifest.get('checkpoint')
    if checkpoint:
        candidates.append(Path(checkpoint).parent)
    eval_root = manifest.get('eval_root')
    if eval_root:
        candidates.append(Path(eval_root))
        candidates.append(Path(eval_root).parent)
    candidates.append(path.parent)
    candidates.append(path.parent.parent)
    for root in candidates:
        for name in ('code_version.json', 'git.json'):
            payload = _load(root / 'provenance' / name)
            if payload:
                return (
                    payload.get('commit'),
                    payload.get('code_fingerprint'),
                )
    return None, None


def _baseline_row(manifest: dict, path: Path) -> dict:
    checkpoint = manifest.get('checkpoint')
    code_commit, code_fp = _code_info(manifest, path)
    return {
        'family': 'baseline',
        'method': manifest.get('method', path.parent.name),
        'variant': manifest.get('variant', 'default'),
        'seed': manifest.get('seed'),
        'status': manifest.get('status', 'unknown'),
        'checkpoint': checkpoint,
        'best_epoch': _best_epoch(checkpoint),
        'eval_root': manifest.get('paper_root', str(path.parent)),
        'metrics_csv': manifest.get('paper_metrics'),
        'macro_csv': manifest.get('paper_macro'),
        'manifest': str(path),
        'commit': code_commit,
        'code_fingerprint': code_fp,
    }


def _canonical_row(manifest: dict, path: Path) -> dict:
    status = str(manifest.get('status', 'done'))
    if (path.parent / 'run_manifest.failed.json').is_file():
        status = 'failed'
    code_commit, code_fp = _code_info(manifest, path)
    return {
        'family': manifest.get('family', ''),
        'method': manifest.get('method', ''),
        'variant': manifest.get('variant', 'default'),
        'seed': manifest.get('seed'),
        'status': status,
        'checkpoint': manifest.get('checkpoint'),
        'best_epoch': _best_epoch(manifest.get('checkpoint')),
        'eval_root': manifest.get('eval_root', str(path.parent)),
        'metrics_csv': manifest.get('metrics_csv'),
        'macro_csv': manifest.get('macro_csv'),
        'manifest': str(path),
        'commit': code_commit,
        'code_fingerprint': code_fp,
    }


def _sacm_x_row(manifest: dict, path: Path) -> dict:
    eval_root = manifest.get('output_root', str(path.parent))
    checkpoints = manifest.get('checkpoints') or {}
    checkpoint = (
        next(iter(checkpoints.values())) if len(checkpoints) == 1 else None)
    jobs = manifest.get('jobs') or []
    statuses = {str(row.get('status', '')) for row in jobs}
    if statuses and statuses == {'done'}:
        status = 'done'
    elif statuses and statuses <= {'done', 'train_only'}:
        status = 'train_only'
    elif 'missing_checkpoint' in statuses:
        status = 'missing_checkpoint'
    elif 'failed' in ' '.join(statuses).lower():
        status = 'failed'
    else:
        status = 'unknown'
    metrics = Path(eval_root) / 'tables' / 'all_metrics.csv'
    macro = Path(eval_root) / 'tables' / 'macro_metrics.csv'
    presets = manifest.get('presets') or [manifest.get('variant', 'default')]
    code_commit, code_fp = _code_info(manifest, path)
    return {
        'family': 'sacm_x',
        'method': 'sacm_x',
        'variant': presets[0] if len(presets) == 1 else 'default',
        'seed': manifest.get('seed'),
        'status': status,
        'checkpoint': checkpoint,
        'best_epoch': _best_epoch(checkpoint),
        'eval_root': eval_root,
        'metrics_csv': str(metrics) if metrics.is_file() else None,
        'macro_csv': str(macro) if macro.is_file() else None,
        'manifest': str(path),
        'commit': code_commit,
        'code_fingerprint': code_fp,
    }


def _train_only_rows(root: Path) -> list[dict]:
    """Index canonical runs that have a checkpoint but no eval manifest yet."""
    rows: list[dict] = []
    for info_path in sorted(root.glob('*/*/seed_*/run_info.json')):
        run_dir = info_path.parent
        if (run_dir / 'eval' / 'run_manifest.json').is_file():
            continue
        payload = _load(info_path)
        if payload is None:
            continue
        checkpoint = run_dir / 'best_model.pth'
        if not checkpoint.exists():
            checkpoint = run_dir / 'best_model'
        if not checkpoint.exists():
            value = payload.get('checkpoint')
            if value:
                checkpoint = Path(str(value))
        if not checkpoint.exists():
            continue
        failed = (run_dir / 'eval' / 'run_manifest.failed.json').is_file()
        code_commit, code_fp = _code_info(payload, info_path)
        rows.append({
            'family': payload.get('family', ''),
            'method': payload.get('method', run_dir.parent.parent.name),
            'variant': payload.get('variant', run_dir.parent.name),
            'seed': payload.get('seed'),
            'status': 'failed' if failed else 'train_only',
            'checkpoint': str(checkpoint),
            'best_epoch': _best_epoch(str(checkpoint)),
            'eval_root': '',
            'metrics_csv': None,
            'macro_csv': None,
            'manifest': '',
            'commit': code_commit,
            'code_fingerprint': code_fp,
        })
    return rows


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
        code_commit, code_fp = _code_info({}, checkpoint)
        rows.append({
            'family': 'baseline',
            'method': method_dir.name,
            'variant': 'default',
            'seed': seed,
            'status': 'legacy',
            'checkpoint': str(checkpoint),
            'best_epoch': _best_epoch(str(checkpoint)),
            'eval_root': str(seed_dir),
            'commit': code_commit,
            'code_fingerprint': code_fp,
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
            if manifest.get('metrics_csv') and manifest.get('variant'):
                rows.append(_canonical_row(manifest, path))
            elif family == 'baseline':
                rows.append(_baseline_row(manifest, path))
            elif family in ('sacm_x', 'sacmx') or 'presets' in manifest:
                rows.append(_sacm_x_row(manifest, path))
            else:
                print(f'[warn] unknown manifest family in {path}')
        rows.extend(_train_only_rows(root))
        rows.extend(_legacy_baseline_rows(root))

    def _norm(value) -> str:
        if not value:
            return ''
        return str(Path(str(value)).resolve())

    seen: set[str] = set()
    unique: list[dict] = []
    for row in rows:
        key = _norm(row.get('checkpoint')) + '|' + _norm(
            row.get('metrics_csv'))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    # A canonical results/runs row supersedes its raw-cache legacy row for
    # the same method+seed.  Keep the canonical row only.
    def _is_canonical(row: dict) -> bool:
        value = str(row.get('manifest', '')).replace('\\', '/')
        return value.startswith('results/runs/') or '/results/runs/' in value

    canonical_keys = {
        (str(row.get('method')), str(row.get('variant', 'default')),
         row.get('seed'))
        for row in unique if _is_canonical(row)
    }
    if canonical_keys:
        unique = [
            row for row in unique
            if (str(row.get('method')), str(row.get('variant', 'default')),
                row.get('seed')) not in canonical_keys
            or _is_canonical(row)
        ]
    unique.sort(key=lambda row: (
        str(row.get('family', '')),
        str(row.get('method', '')),
        str(row.get('variant', '')),
        -1 if row.get('seed') is None else int(row['seed']),
        str(row.get('eval_root', '')),
    ))
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', type=Path,
                        default=[Path('results/runs')],
                        help='canonical results root(s); pass legacy roots '
                             'explicitly when migrating old runs')
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
