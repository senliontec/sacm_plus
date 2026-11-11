#!/usr/bin/env python3
"""Promote one baseline run into the canonical results layout.

The only user-facing tree is::

    results/runs/<method>/<variant>/seed_<seed>/
      best_model.pth            # the only evaluation checkpoint
      best_model                # directory checkpoints (symlink)
      checkpoint.json           # kind + native checkpoint path
      train_config.json         # full args + provenance pointers
      provenance/               # data/environment/git/manifests
      wandb_run.json
      run_info.json
      metrics_history.json
      train.log
      predictions/              # baselines, when present
      eval/
        <DATASET>/
          metrics.csv           # per-image
          summary.csv           # one canonical summary row
          results_summary.txt ...
          prediction_index.csv  # prediction -> GT mask mapping
          eval_manifest.json
        tables/
          all_metrics.csv       # unified schema
          macro_metrics.csv     # unified long schema
          final_report.md
          final_report.csv
        run_manifest.json
        eval.log
        jobs.csv

New runs write into this tree directly.  ``--source-train`` and
``--source-eval`` may also point at the legacy raw trees; in that case the
finalizer copies the artifacts into the canonical tree and leaves the source
untouched.  With ``--cleanup`` and in-place paths it removes only the
pipeline-specific intermediate directories after every canonical artifact has
been written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from core.run_layout import RUNS_ROOT, run_root as _run_root  # noqa: E402

META_FIELDS = ['family', 'method', 'variant', 'seed']
RESERVED_DIRS = {
    '.run_status', '__pycache__', 'candidates', 'eval', 'logs',
    'paper', 'predictions', 'tables', 'torchrun_logs', 'visualizations',
}
ROLE_ORDER = ('Seen', 'Unseen', 'Novel', 'All')
MAIN_METRICS = ('dice', 'iou', 'cldice', 'hd95')


# ---------------------------------------------------------------------------
# small IO helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _same_path(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return False
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _link_or_copy(source: Path, target: Path) -> None:
    """Hardlink a file when possible; never duplicate a 1.4 GiB checkpoint."""
    if _same_path(source, target):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.is_file():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _copy_file(source: Path | None, target: Path) -> None:
    if source is not None and source.is_file() and not _same_path(source, target):
        _link_or_copy(source, target)


def _merge_or_move(source: Path, target: Path, cleanup: bool) -> None:
    """Expose a dataset directory at ``target`` while preserving sidecars."""
    if _same_path(source, target):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.copytree(source, target, dirs_exist_ok=True)
        if cleanup:
            shutil.rmtree(source)
        return
    if cleanup:
        shutil.move(str(source), str(target))
    else:
        # Copy rather than symlink: the canonical metrics.csv is rewritten
        # in place, and a symlink would mutate the legacy source directory.
        shutil.copytree(source, target)


def _pointer_path(path: Path) -> Path:
    text = path.read_text().strip().splitlines()
    value = Path(text[0]) if text else path
    if not value.is_absolute():
        value = (path.parent / value).resolve()
    return value


# ---------------------------------------------------------------------------
# checkpoint + metadata
# ---------------------------------------------------------------------------

@dataclass
class CheckpointInfo:
    native: Path
    kind: str  # 'file' or 'directory'


def _checkpoint_candidate(path: Path | None) -> CheckpointInfo | None:
    if path is None or not path.exists():
        return None
    if path.is_dir():
        return CheckpointInfo(path, 'directory')
    if path.suffix == '.txt':
        resolved = _pointer_path(path)
        if not resolved.exists():
            return None
        kind = 'directory' if resolved.is_dir() else 'file'
        return CheckpointInfo(resolved, kind)
    return CheckpointInfo(path, 'file')


def _resolve_checkpoint(source_train: Path, variant: str,
                        explicit: Path | None) -> CheckpointInfo:
    if explicit is not None:
        found = _checkpoint_candidate(explicit)
        if found is None:
            raise SystemExit(
                f'explicit checkpoint does not exist or has a broken '
                f'pointer: {explicit}')
        return found
    # Prefer the fresh per-preset training output over a checkpoint that was
    # promoted by an earlier finalize; otherwise retraining into an existing
    # run root silently reuses the old model.
    candidates = [
        source_train / variant / 'best_model.pth',
        source_train / 'best_model.pth',
        source_train / variant / 'SGDL_best_model.pth',
        source_train / 'SGDL_best_model.pth',
        source_train / variant / 'nnunet_model_dir.txt',
        source_train / 'nnunet_model_dir.txt',
        source_train / variant / 'knowsam_model.txt',
        source_train / 'knowsam_model.txt',
    ]
    for path in candidates:
        found = _checkpoint_candidate(path)
        if found is not None:
            return found
    checkpoints = sorted(source_train.glob('*/best_model.pth'))
    if len(checkpoints) == 1:
        return CheckpointInfo(checkpoints[0], 'file')
    raise SystemExit(
        f'no checkpoint under {source_train} (variant={variant}); '
        'pass --checkpoint explicitly')


def _promote_checkpoint(info: CheckpointInfo, run_root: Path) -> Path:
    """Put the checkpoint at a stable canonical path and describe it."""
    run_root.mkdir(parents=True, exist_ok=True)
    if info.kind == 'directory':
        target = run_root / 'best_model'
        if not _same_path(info.native, target):
            if target.is_symlink():
                if _same_path(target, info.native):
                    pass
                else:
                    target.unlink()
            elif target.is_dir():
                # A real copy from an earlier finalize must not shadow the
                # fresh checkpoint directory.
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            if not target.exists():
                try:
                    os.symlink(info.native.resolve(), target,
                               target_is_directory=True)
                except OSError:
                    shutil.copytree(info.native, target)
    else:
        target = run_root / 'best_model.pth'
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        _link_or_copy(info.native, target)
    payload = {
        'kind': info.kind,
        'native': str(info.native.resolve()),
        'canonical': str(target.resolve()),
    }
    (run_root / 'checkpoint.json').write_text(
        json.dumps(payload, indent=2) + '\n')
    return target


def _find_file(*candidates: Path | None) -> Path | None:
    for path in candidates:
        if path is not None and path.is_file():
            return path
    return None


def _read_source_run_info(source_train: Path, variant: str,
                          checkpoint: CheckpointInfo) -> tuple[dict, Path | None]:
    candidates = [
        checkpoint.native.parent / 'run_info.json',
        source_train / variant / 'run_info.json',
        source_train / 'run_info.json',
    ]
    for path in candidates:
        payload = _read_json(path)
        if payload:
            return payload, path
    return {}, None


# ---------------------------------------------------------------------------
# evaluation artifacts
# ---------------------------------------------------------------------------

def _dataset_dirs(source_eval: Path, method: str,
                  variant: str, expected: set[str] | None = None
                  ) -> list[Path]:
    """Find raw per-dataset result dirs across current and legacy layouts."""
    roots = [source_eval / variant, source_eval / method, source_eval]
    found: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name in RESERVED_DIRS:
                continue
            if expected is not None and child.name not in expected:
                continue
            if child.name in found:
                continue
            if ((child / 'metrics.csv').is_file()
                    or (child / 'results_summary.txt').is_file()):
                found[child.name] = child
    return [found[name] for name in sorted(found)]


def _aggregate_source(source_eval: Path) -> Path | None:
    return _find_file(
        source_eval / 'tables' / 'all_metrics.csv',
        source_eval / 'all_metrics.csv',
        source_eval / 'paper_metrics.csv',
    )


def _is_metric_key(key: str) -> bool:
    return key.endswith(('_mean', '_std', '_n'))


def _resolve_protocol(args: argparse.Namespace, source_info: dict,
                      source_train: Path) -> Path | None:
    if args.protocol:
        path = Path(args.protocol)
        return path if path.is_file() else path
    data_root = source_info.get('data_root')
    candidates = []
    if data_root:
        candidates.append(Path(str(data_root)) / 'protocol_manifest.csv')
    candidates += [
        source_train / 'protocol_manifest.csv',
        source_train.parent / 'protocol_manifest.csv',
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _role_from_protocol(protocol: Path | None) -> dict[str, str]:
    roles: dict[str, str] = {}
    if protocol is None or not protocol.is_file():
        return roles
    try:
        with protocol.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if not str(row.get('split', '')).endswith('_test'):
                    continue
                dataset = str(row.get('dataset', '')).strip()
                role = str(row.get('role', '')).strip()
                if dataset and role:
                    roles[dataset] = role.capitalize()
    except OSError:
        pass
    return roles


def _per_image_rows(dataset_dir: Path) -> tuple[list[dict], list[str]]:
    path = dataset_dir / 'metrics.csv'
    if not path.is_file():
        return [], []
    try:
        return _read_csv(path)
    except OSError:
        return [], []


def _numeric_columns(fields: list[str]) -> list[str]:
    return [key for key in fields
            if key not in ('dataset', 'image', 'role', 'family', 'method',
                           'variant', 'seed', 'preset', 'selected_head')
            and not key.endswith(('_mean', '_std', '_n', '_meta'))]


def _summary_from_rows(rows: list[dict], metrics: list[str]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for metric in metrics:
        values = []
        for row in rows:
            try:
                value = float(row.get(metric, 'nan'))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        summary[metric] = {
            'mean': statistics.fmean(values) if values else float('nan'),
            'std': statistics.pstdev(values) if len(values) > 1 else 0.0,
            'n': len(values),
        }
    return summary


def _aggregate_rows(source: Path | None, variant: str) -> tuple[list[dict], list[str]]:
    if source is None:
        return [], []
    rows, fields = _read_csv(source)
    filtered = []
    for row in rows:
        for key in ('variant', 'preset'):
            value = row.get(key)
            if value not in (None, '', variant):
                break
        else:
            filtered.append(row)
    return filtered, fields


def _canonical_all_rows(aggregate_rows: list[dict],
                        aggregate_fields: list[str],
                        dataset_dirs: list[Path],
                        roles: dict[str, str],
                        family: str,
                        method: str,
                        variant: str,
                        seed: int) -> tuple[list[dict], list[str]]:
    """Build the unified all_metrics table.

    Source aggregate means/std are authoritative; per-image metrics fill in
    missing datasets, roles, and sample counts.
    """
    by_dataset: dict[str, dict] = {}
    raw_by_dataset: dict[str, list[dict]] = {}
    for dataset_dir in dataset_dirs:
        rows, fields = _per_image_rows(dataset_dir)
        raw_by_dataset[dataset_dir.name] = rows
        if not rows:
            continue
        metrics = _numeric_columns(fields)
        summary = _summary_from_rows(rows, metrics)
        item = {
            'dataset': dataset_dir.name,
            'role': '',
        }
        for role_key in ('role',):
            values = [str(row.get(role_key, '')).strip() for row in rows]
            values = [value for value in values if value]
            if values:
                item['role'] = values[0].capitalize()
        for metric in metrics:
            item[f'{metric}_mean'] = summary[metric]['mean']
            item[f'{metric}_std'] = summary[metric]['std']
            item[f'{metric}_n'] = summary[metric]['n']
        by_dataset[dataset_dir.name] = item

    for row in aggregate_rows:
        dataset = str(row.get('dataset', '')).strip()
        if not dataset:
            continue
        item = by_dataset.setdefault(dataset, {'dataset': dataset,
                                               'role': ''})
        role = str(row.get('role', '')).strip()
        if role:
            item['role'] = role.capitalize()
        for key in aggregate_fields:
            if _is_metric_key(key):
                value = row.get(key, '')
                # Per-image metrics are authoritative and give every family
                # the same mean/std/n convention.  The aggregate row fills
                # metrics that the per-image table does not expose.
                if (value not in ('', None)
                        and (key not in item
                             or item[key] in ('', None))):
                    item[key] = value

    for name, raw in raw_by_dataset.items():
        if not raw:
            continue
        item = by_dataset.setdefault(name, {'dataset': name, 'role': ''})
        fields = list(raw[0]) if raw else []
        metrics = _numeric_columns(fields)
        for metric in metrics:
            key = f'{metric}_n'
            if key not in item or item[key] in ('', None):
                item[key] = sum(
                    1 for row in raw
                    if math.isfinite(_float(row.get(metric))))

    for dataset, item in by_dataset.items():
        if not item.get('role'):
            item['role'] = roles.get(dataset, '')
        item.update({
            'family': family,
            'method': method,
            'variant': variant,
            'seed': seed,
        })

    # Canonical paper tables always have the same four metrics and the same
    # column order, regardless of family or how many extra metrics the raw
    # evaluator produced.  Extra metrics remain in per-image metrics.csv.
    fields = META_FIELDS + ['dataset', 'role']
    for metric in MAIN_METRICS:
        fields += [f'{metric}_mean', f'{metric}_std', f'{metric}_n']
    rows = []
    for dataset in sorted(by_dataset):
        item = by_dataset[dataset]
        row = {key: item.get(key, '') for key in fields}
        rows.append(row)
    return rows, fields


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _macro_rows(all_rows: list[dict], fields: list[str],
                family: str, method: str, variant: str,
                seed: int) -> list[dict]:
    metric_bases = sorted({
        key[:-5] for key in fields
        if key.endswith('_mean') and key != 'mean'
    })
    rows: list[dict] = []
    present_roles = {str(row.get('role', '')) for row in all_rows}
    roles = [role for role in ROLE_ORDER if role in present_roles or role == 'All']
    if 'All' not in present_roles and all_rows:
        roles = [role for role in roles if role != 'All'] + ['All']
    for role in roles:
        if role == 'All':
            group = list(all_rows)
        else:
            group = [row for row in all_rows if row.get('role') == role]
            if not group and role != 'All':
                continue
        for metric in metric_bases:
            values = []
            for row in group:
                value = _float(row.get(f'{metric}_mean'))
                if math.isfinite(value):
                    values.append(value)
            if not values:
                continue
            rows.append({
                'family': family,
                'method': method,
                'variant': variant,
                'seed': seed,
                'role': role,
                'metric': metric,
                'n_datasets': len(values),
                'mean': statistics.fmean(values),
                'std': (statistics.stdev(values)
                        if len(values) > 1 else 0.0),
            })
    return rows


def _normalise_dataset_metrics(dataset_dir: Path, target: Path,
                               dataset: str, role: str) -> None:
    rows, fields = _per_image_rows(dataset_dir)
    if not rows:
        return
    metrics = [key for key in fields
               if key not in ('dataset', 'image', 'role')]
    out_fields = ['dataset', 'image', 'role'] + metrics
    out_rows = []
    for row in rows:
        item = {
            'dataset': dataset,
            'image': row.get('image', ''),
            'role': str(row.get('role', '') or role),
        }
        item.update({key: row.get(key, '') for key in metrics})
        out_rows.append(item)
    _write_csv(target / 'metrics.csv', out_fields, out_rows)


def _write_csv_if_missing(source: Path | None, target: Path) -> None:
    _copy_file(source, target)


def _write_report(path: Path, all_rows: list[dict],
                  macro_rows: list[dict], family: str, method: str,
                  variant: str, seed: int) -> None:
    metrics = [metric for metric in MAIN_METRICS
               if any(f'{metric}_mean' in row for row in all_rows)]
    lines = [
        '# Unified run report',
        '',
        f'- Family: `{family}`',
        f'- Method: `{method}`',
        f'- Variant: `{variant}`',
        f'- Seed: `{seed}`',
        '',
        'Values are fractions for Dice/IoU/clDice and pixels for HD95.',
        '',
        '## Macro (equal dataset weight)',
        '',
        '| Role | ' + ' | '.join(metrics) + ' | n |',
        '|---|' + '---:|' * len(metrics) + '---:|',
    ]
    macro_by = {(row.get('role'), row.get('metric')): row
                for row in macro_rows}
    present_roles = {str(row.get('role', '')) for row in all_rows}
    roles = [role for role in ROLE_ORDER
             if role == 'All' or role in present_roles]
    for role in roles:
        cells = []
        count = ''
        for metric in metrics:
            row = macro_by.get((role, metric))
            cells.append('—' if row is None else f"{_float(row.get('mean')):.4f}")
            if row is not None:
                count = str(row.get('n_datasets', ''))
        lines.append(f'| {role} | ' + ' | '.join(cells) + f' | {count} |')

    lines += [
        '',
        '## Per dataset',
        '',
        '| Dataset | Role | ' + ' | '.join(metrics) + ' |',
        '|---|---|' + '---:|' * len(metrics),
    ]
    for row in all_rows:
        cells = []
        for metric in metrics:
            value = _float(row.get(f'{metric}_mean'))
            cells.append('—' if not math.isfinite(value) else f'{value:.4f}')
        lines.append(f"| {row.get('dataset', '')} | {row.get('role', '')} | "
                     + ' | '.join(cells) + ' |')
    path.write_text('\n'.join(lines) + '\n')


def _write_report_csv(path: Path, all_rows: list[dict]) -> None:
    fields = ['dataset', 'role']
    for metric in MAIN_METRICS:
        if any(f'{metric}_mean' in row for row in all_rows):
            fields.append(metric)
    rows = []
    for row in all_rows:
        item = {'dataset': row.get('dataset', ''),
                'role': row.get('role', '')}
        for metric in MAIN_METRICS:
            item[metric] = row.get(f'{metric}_mean', '')
        rows.append(item)
    _write_csv(path, fields, rows)


# ---------------------------------------------------------------------------
# logs + jobs
# ---------------------------------------------------------------------------

def _promote_train_log(source_train: Path, variant: str,
                       checkpoint: CheckpointInfo, run_root: Path) -> None:
    source = _find_file(
        checkpoint.native.parent / 'train.log',
        checkpoint.native.parent / 'run.log',
        source_train / 'train.log',
        source_train / 'run.log',
        source_train / variant / 'train.log',
        source_train / variant / 'run.log',
    )
    _copy_file(source, run_root / 'train.log')


def _artifact_source(source_train: Path, variant: str,
                     checkpoint: CheckpointInfo) -> Path | None:
    """Find the directory that holds train_config/provenance."""
    candidates = [
        checkpoint.native.parent,
        source_train / variant,
        source_train,
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_dir():
            continue
        seen.add(candidate)
        if any((candidate / name).exists() for name in (
                'train_config.json', '.resume_state.pth',
                'wandb_run.json', 'provenance', 'checkpoints')):
            return candidate
    return None


def _promote_extra_artifacts(source_train: Path, variant: str,
                             checkpoint: CheckpointInfo,
                             run_root: Path) -> dict[str, str | None]:
    """Copy training provenance into the canonical run root.

    Temporary resume/periodic checkpoints are never promoted: only the
    selected ``best_model.pth`` is a result artifact.
    """
    source = _artifact_source(source_train, variant, checkpoint)
    result: dict[str, str | None] = {
        'train_config': None,
        'last_checkpoint': None,
        'wandb': None,
        'provenance': None,
        'checkpoints': None,
    }
    if source is None:
        return result

    for name, key in (
            ('train_config.json', 'train_config'),
            ('wandb_run.json', 'wandb')):
        source_path = _find_file(
            source / name,
            source_train / name,
            source_train / variant / name,
            checkpoint.native.parent / name,
        )
        target = run_root / name
        if source_path is not None and not _same_path(source_path, target):
            _copy_file(source_path, target)
        if target.is_file():
            result[key] = str(target.resolve())
    for resume_config in sorted(source.glob('train_config.resume_*.json')):
        target = run_root / resume_config.name
        if not _same_path(resume_config, target):
            _copy_file(resume_config, target)

    # Never promote .resume_state.pth or periodic snapshots.  Remove stale
    # targets left by older finalizer versions so a rerun leaves exactly one
    # model artifact in the canonical tree.
    stale_paths = [run_root / '.resume_state.pth',
                   run_root / 'checkpoints',
                   *run_root.glob('best_model.before_*.pth'),
                   *run_root.glob('*last*.pth')]
    for stale in stale_paths:
        if stale.is_symlink() or stale.is_file():
            stale.unlink()
        elif stale.is_dir():
            shutil.rmtree(stale)

    provenance_source = source / 'provenance'
    provenance_target = run_root / 'provenance'
    if provenance_source.is_dir() and not _same_path(
            provenance_source, provenance_target):
        provenance_target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            provenance_source, provenance_target, dirs_exist_ok=True)
    if provenance_target.is_dir():
        result['provenance'] = str(provenance_target.resolve())
    return result


def _promote_eval_log(source_eval: Path, source_train: Path,
                      variant: str, eval_root: Path,
                      cleanup: bool) -> None:
    """Move the planner's live log when possible.

    The planner redirects the child process to ``<run>/eval.log``.  Moving
    that open file to ``eval/eval.log`` keeps the parent's file descriptor
    writing into the canonical file after the child returns.  Copying a
    still-open log would truncate the final lines.
    """
    target = eval_root / 'eval.log'
    live = source_eval.parent / 'eval.log'
    if (source_eval.name == 'eval' and live.is_file()
            and not _same_path(live, target)):
        if cleanup:
            shutil.move(str(live), str(target))
        else:
            _copy_file(live, target)
        if target.is_file():
            return
    source = _find_file(
        source_eval / 'eval.log',
        source_eval / variant / 'run_eval.log',
        source_eval / 'run_eval.log',
        source_train / 'eval.log',
        source_train / variant / 'run_eval.log',
    )
    _copy_file(source, target)


def _read_jobs(path: Path | None, phase: str) -> list[dict]:
    if path is None or not path.is_file():
        return []
    try:
        rows, _ = _read_csv(path)
    except OSError:
        return []
    out = []
    for row in rows:
        out.append({
            'phase': row.get('phase') or phase,
            'preset': row.get('preset', ''),
            'dataset': row.get('dataset', ''),
            'gpu': row.get('gpu', ''),
            'status': row.get('status', ''),
        })
    return out


def _canonical_jobs(source_train: Path, source_eval: Path, variant: str,
                    datasets: list[str], status: str,
                    eval_root: Path) -> None:
    rows = []
    rows.extend(_read_jobs(source_eval / 'jobs_eval.csv', 'eval'))
    rows.extend(_read_jobs(source_eval / 'jobs.csv', 'eval'))
    rows.extend(_read_jobs(source_train / 'jobs.csv', 'train'))
    if not rows:
        for dataset in datasets:
            rows.append({'phase': 'eval', 'preset': variant,
                         'dataset': dataset, 'gpu': '', 'status': status})
        if not datasets:
            rows.append({'phase': 'train', 'preset': variant,
                         'dataset': '', 'gpu': '', 'status': status})
    _write_csv(eval_root / 'jobs.csv',
               ['phase', 'preset', 'dataset', 'gpu', 'status'], rows)


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------

def _safe_unlink(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except OSError:
        pass


def _cleanup(source_train: Path, source_eval: Path, run_root: Path,
             eval_root: Path, variant: str, method: str,
             checkpoint: CheckpointInfo) -> None:
    in_place_train = _same_path(source_train, run_root)
    in_place_eval = _same_path(source_eval, eval_root)

    if in_place_eval:
        tables = eval_root / 'tables'
        tables.mkdir(parents=True, exist_ok=True)
        # Preserve useful legacy reports/tables, then remove the raw views.
        for tex in sorted(eval_root.glob('table_*.tex')):
            _copy_file(tex, tables / tex.name)
            _safe_unlink(tex)
        _copy_file(eval_root / 'paper_tables.md', tables / 'source_report.md')
        _copy_file(eval_root / 'paper_stats.json',
                   tables / 'collect_stats.json')
        for name in ('paper_metrics.csv', 'paper_macro.csv',
                     'paper_tables.md', 'paper_stats.json',
                     'jobs_eval.csv', 'run_eval.log',
                     'run_manifest.failed.json'):
            _safe_unlink(eval_root / name)
        for name in (variant, method):
            raw = eval_root / name
            if (raw.is_dir() and not (raw / 'metrics.csv').is_file()
                    and name not in RESERVED_DIRS):
                _safe_unlink(raw)
        _safe_unlink(eval_root / 'run_eval.log')

    if in_place_train:
        # Remove the raw per-preset directory only when the promoted
        # checkpoint came from it.  A root/external checkpoint may coexist
        # with a fresh training directory that must be preserved.
        variant_dir = source_train / variant
        if (checkpoint.kind == 'directory'
                or not _is_below(checkpoint.native, variant_dir)):
            pass
        else:
            _safe_unlink(variant_dir)
        _safe_unlink(source_train / 'run_manifest.json')
        _safe_unlink(source_train / 'jobs.csv')
        _safe_unlink(source_train / 'eval.log')


def _is_below(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def finalize(args: argparse.Namespace) -> Path:
    source_train = Path(args.source_train)
    source_eval = Path(args.source_eval)
    out_root = Path(args.out_root)
    run_root = _run_root(args.method, args.variant, int(args.seed),
                         root=out_root)
    eval_root = run_root / 'eval'
    tables = eval_root / 'tables'
    run_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    checkpoint = _resolve_checkpoint(
        source_train, args.variant, args.checkpoint)
    canonical_checkpoint = _promote_checkpoint(checkpoint, run_root)
    source_info, source_info_path = _read_source_run_info(
        source_train, args.variant, checkpoint)
    protocol_path = _resolve_protocol(args, source_info, source_train)

    history_source = _find_file(
        checkpoint.native.parent / 'metrics_history.json',
        source_train / args.variant / 'metrics_history.json',
        source_train / 'metrics_history.json',
    )
    _copy_file(history_source, run_root / 'metrics_history.json')
    history = _read_json(run_root / 'metrics_history.json')

    _promote_train_log(source_train, args.variant, checkpoint, run_root)
    extra_artifacts = _promote_extra_artifacts(
        source_train, args.variant, checkpoint, run_root)
    checkpoint_meta_path = run_root / 'checkpoint.json'
    checkpoint_meta = _read_json(checkpoint_meta_path)
    if checkpoint_meta:
        checkpoint_meta.update({
            'last_checkpoint': extra_artifacts.get('last_checkpoint'),
            'periodic_checkpoints': extra_artifacts.get('checkpoints'),
        })
        checkpoint_meta_path.write_text(
            json.dumps(checkpoint_meta, indent=2) + '\n')
    _promote_eval_log(source_eval, source_train, args.variant, eval_root,
                      cleanup=bool(args.cleanup))

    roles = _role_from_protocol(protocol_path)
    dataset_dirs = _dataset_dirs(
        source_eval, args.method, args.variant,
        expected=set(roles) if roles else None)
    aggregate_source = _aggregate_source(source_eval)
    aggregate_rows, aggregate_fields = _aggregate_rows(
        aggregate_source, args.variant)
    all_rows, all_fields = _canonical_all_rows(
        aggregate_rows, aggregate_fields, dataset_dirs, roles,
        args.family, args.method, args.variant, int(args.seed))
    _write_csv(tables / 'all_metrics.csv', all_fields, all_rows)

    macro_rows = _macro_rows(all_rows, all_fields, args.family, args.method,
                             args.variant, int(args.seed))
    _write_csv(
        tables / 'macro_metrics.csv',
        ['family', 'method', 'variant', 'seed', 'role', 'metric',
         'n_datasets', 'mean', 'std'],
        macro_rows)

    all_by_dataset = {row.get('dataset'): row for row in all_rows}
    for dataset_dir in dataset_dirs:
        dataset = dataset_dir.name
        target = eval_root / dataset
        _merge_or_move(dataset_dir, target, cleanup=bool(args.cleanup))
        row = all_by_dataset.get(dataset, {})
        _normalise_dataset_metrics(
            target, target, dataset, str(row.get('role', '')))
        if row:
            _write_csv(target / 'summary.csv', all_fields, [row])
        for sidecar in ('results_summary.txt', 'metrics_meta.json',
                        'head_audit.csv', 'threshold_audit.csv',
                        'head_full_audit.csv', 'head_threshold_audit.csv'):
            _copy_file(dataset_dir / sidecar, target / sidecar)
        candidates = dataset_dir / 'candidates'
        if candidates.is_dir() and not (target / 'candidates').exists():
            try:
                os.symlink(candidates.resolve(), target / 'candidates',
                           target_is_directory=True)
            except OSError:
                pass

    report_source = _find_file(
        source_eval / 'tables' / 'final_report.md',
        source_eval / 'paper_tables.md',
        source_eval / 'tables' / 'report.md',
    )
    if report_source is not None and not _same_path(
            report_source, tables / 'final_report.md'):
        _copy_file(report_source, tables / 'source_report.md')
    _write_report(tables / 'final_report.md', all_rows, macro_rows,
                  args.family, args.method, args.variant, int(args.seed))
    _write_report_csv(tables / 'final_report.csv', all_rows)

    _canonical_jobs(source_train, source_eval, args.variant,
                    [row.get('dataset') for row in all_rows],
                    args.status or ('done' if all_rows else 'train_only'),
                    eval_root)

    best_epoch = source_info.get('best_epoch')
    last_epoch = source_info.get('last_epoch')
    if last_epoch is None:
        entries = history.get('history') or []
        last_epoch = entries[-1].get('epoch') if entries else None
    if best_epoch is None:
        best_epoch = history.get('best_epoch')

    run_info = dict(source_info)
    run_info.update({
        'family': args.family,
        'method': args.method,
        'variant': args.variant,
        'seed': int(args.seed),
        'checkpoint': str(canonical_checkpoint.resolve()),
        'checkpoint_sha256': getattr(args, 'checkpoint_sha256', None),
        'checkpoint_kind': checkpoint.kind,
        'native_checkpoint': str(checkpoint.native.resolve()),
        'run_root': str(run_root.resolve()),
        'eval_root': str(eval_root.resolve()),
        'best_epoch': best_epoch,
        'last_epoch': last_epoch,
        'source_train': str(source_train.resolve()),
        'source_eval': str(source_eval.resolve()),
        'source_run_info': (str(source_info_path.resolve())
                            if source_info_path else None),
        'protocol': (str(protocol_path) if protocol_path else None),
        'status': args.status or ('done' if all_rows else 'train_only'),
        'last_checkpoint': extra_artifacts.get('last_checkpoint'),
        'train_config': extra_artifacts.get('train_config'),
        'provenance': extra_artifacts.get('provenance'),
        'periodic_checkpoints': extra_artifacts.get('checkpoints'),
        'checkpoints': {
            'best': str(canonical_checkpoint.resolve()),
            'last': extra_artifacts.get('last_checkpoint'),
            'periodic_dir': extra_artifacts.get('checkpoints'),
        },
        'wandb': extra_artifacts.get('wandb'),
        'generated_at': datetime.now().isoformat(timespec='seconds'),
    })
    (run_root / 'run_info.json').write_text(
        json.dumps(run_info, indent=2, default=str) + '\n')

    manifest = {
        'family': args.family,
        'method': args.method,
        'variant': args.variant,
        'seed': int(args.seed),
        'status': run_info['status'],
        'run_root': str(run_root.resolve()),
        'checkpoint': str(canonical_checkpoint.resolve()),
        'checkpoint_sha256': getattr(args, 'checkpoint_sha256', None),
        'checkpoint_kind': checkpoint.kind,
        'native_checkpoint': str(checkpoint.native.resolve()),
        'last_checkpoint': extra_artifacts.get('last_checkpoint'),
        'train_config': extra_artifacts.get('train_config'),
        'provenance': extra_artifacts.get('provenance'),
        'periodic_checkpoints': extra_artifacts.get('checkpoints'),
        'checkpoints': {
            'best': str(canonical_checkpoint.resolve()),
            'last': extra_artifacts.get('last_checkpoint'),
            'periodic_dir': extra_artifacts.get('checkpoints'),
        },
        'wandb': extra_artifacts.get('wandb'),
        'run_info': str((run_root / 'run_info.json').resolve()),
        'eval_root': str(eval_root.resolve()),
        'metrics_csv': str((tables / 'all_metrics.csv').resolve()),
        'macro_csv': str((tables / 'macro_metrics.csv').resolve()),
        'final_report': str((tables / 'final_report.md').resolve()),
        'datasets': [row.get('dataset') for row in all_rows],
        'protocol': (str(protocol_path) if protocol_path else None),
        'pred_root': args.pred_root,
        'source_train': str(source_train.resolve()),
        'source_eval': str(source_eval.resolve()),
        'generated_at': datetime.now().isoformat(timespec='seconds'),
    }
    (eval_root / 'run_manifest.json').write_text(
        json.dumps(manifest, indent=2, default=str) + '\n')

    if args.cleanup:
        _cleanup(source_train, source_eval, run_root, eval_root,
                 args.variant, args.method, checkpoint)
    return run_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True,
                        choices=['baseline'])
    parser.add_argument('--method', required=True)
    parser.add_argument('--variant', default='default')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--source-train', type=Path, required=True)
    parser.add_argument('--source-eval', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument(
        '--checkpoint-sha256', default=None,
        help='fingerprint of the deployed checkpoint file, recorded in the '
             'canonical eval manifest')
    parser.add_argument('--out-root', type=Path, default=RUNS_ROOT)
    parser.add_argument('--cleanup', action='store_true',
                        help='remove pipeline-specific intermediates after '
                             'the canonical tree is complete')
    parser.add_argument('--protocol', default=None)
    parser.add_argument('--pred-root', default=None)
    parser.add_argument('--status', default=None)
    args = parser.parse_args()
    run_root = finalize(args)
    print(f'unified run: {run_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
