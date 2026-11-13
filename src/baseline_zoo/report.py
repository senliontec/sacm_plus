"""Cross-model / cross-seed metric tables.

Canonical runs expose ``<run>/eval/tables/all_metrics.csv`` for both
families.  Legacy ``<run>/tables/all_metrics.csv`` and
``<run>/paper_metrics.csv`` remain readable during migration.

The report writes a tidy CSV, macro CSV, seed-variance CSV and a Markdown
summary.  Values are kept as fractions for Dice/IoU/clDice (the Markdown
display converts them to percentages); HD95 is in pixels.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .method_groups import GROUP_LABELS, group_for, validate_groups
from .protocols import ProtocolConfig
from .registry import REPO_ROOT

METRICS = ('dice', 'iou', 'cldice', 'hd95')
ROLES = ('Seen', 'Unseen', 'Novel')
GROUP_ORDER = (
    'prompt_free', 'proposed', 'support', 'oracle', 'semi_supervised',
    'zero_shot', 'appendix', 'reconstructed', 'unclassified',
)
PRIMARY = {'dice': 'Dice', 'iou': 'IoU', 'cldice': 'clDice', 'hd95': 'HD95'}


@dataclass(frozen=True)
class MetricRow:
    model: str
    seed: int
    dataset: str
    role: str
    metric: str
    mean: float
    std: float
    group: str = 'unclassified'


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _role(value: str) -> str:
    text = (value or '').strip().lower()
    return {'seen': 'Seen', 'unseen': 'Unseen', 'novel': 'Novel'}.get(
        text, value)


def _run_manifest(run_root: Path) -> dict | None:
    """Return the parsed run manifest, or None when absent/unreadable."""
    for candidate in (run_root / 'eval' / 'run_manifest.json',
                      run_root / 'run_manifest.json'):
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _run_status(run_root: Path) -> str | None:
    """Return the manifest status, or None for a legacy run without one."""
    failed = (run_root / 'eval' / 'run_manifest.failed.json')
    if failed.is_file():
        return 'failed'
    for candidate in (run_root / 'eval' / 'run_manifest.json',
                      run_root / 'run_manifest.json'):
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            return 'invalid_manifest'
        return str(payload.get('status', 'done'))
    return None


def load_run(run_root: Path, model: str, seed: int,
             preset: str | None = None,
             allow_partial: bool = False) -> list[MetricRow]:
    """Read one (model, variant, seed) canonical result directory.

    Canonical rows live at ``<run>/eval/tables/all_metrics.csv``.  The
    legacy fallbacks remain readable so a partially migrated tree can still
    be reported.  Runs whose manifest says ``partial``/``train_only``/
    ``failed`` are skipped unless ``allow_partial`` is explicit; otherwise a
    19/20-dataset table would silently share a macro denominator with
    complete runs.
    """
    status = _run_status(run_root)
    if status is None:
        print(f'[warn] {run_root}: no run_manifest.json; dataset '
              'completeness cannot be verified')
    if status is not None and status != 'done' and not allow_partial:
        print(f'[warn] skipping incomplete run {run_root} '
              f'(status={status}); pass --allow-partial to include it')
        return []
    path = run_root / 'eval' / 'tables' / 'all_metrics.csv'
    if not path.is_file():
        path = run_root / 'tables' / 'all_metrics.csv'
    if not path.is_file():
        path = run_root / 'paper_metrics.csv'
    if not path.is_file():
        return []

    rows: list[MetricRow] = []
    # A preset selects one model variant (for example DCD's all_dense
    # run).  Rows without a variant field remain eligible.
    filter_preset = preset
    with open(path, newline='') as f:
        for raw in csv.DictReader(f):
            row_method = (raw.get('method') or '').strip()
            if row_method and row_method != model:
                raise SystemExit(
                    f'{path}: row method={row_method!r} but requested '
                    f'{model!r}')
            row_seed = raw.get('seed')
            if row_seed not in (None, ''):
                try:
                    row_seed_int = int(row_seed)
                except (TypeError, ValueError):
                    raise SystemExit(
                        f'{path}: invalid seed={row_seed!r}') from None
                if row_seed_int != int(seed):
                    raise SystemExit(
                        f'{path}: row seed={row_seed_int} but requested '
                        f'{seed}')
            if (filter_preset is not None
                    and raw.get('preset') not in (None, filter_preset)):
                continue
            dataset = raw.get('dataset', '').strip()
            role = _role(raw.get('role', ''))
            if not dataset or not role:
                continue
            # Canonical rows already carry the variant; a legacy table may
            # contain several presets in one file.
            if (filter_preset is not None
                    and raw.get('variant') not in (None, filter_preset)):
                continue
            for metric in METRICS:
                mean = _float(raw.get(f'{metric}_mean'))
                if math.isnan(mean):
                    continue
                rows.append(MetricRow(
                    model=model,
                    seed=seed,
                    dataset=dataset,
                    role=role,
                    metric=metric,
                    mean=mean,
                    std=_float(raw.get(f'{metric}_std')),
                    group=group_for(model),
                ))

    manifest = _run_manifest(run_root)
    if manifest is not None:
        manifest_variant = manifest.get('variant')
        if (filter_preset is not None and manifest_variant
                and str(manifest_variant) != filter_preset):
            raise SystemExit(
                f'{run_root}: manifest variant={manifest_variant!r} but '
                f'requested {filter_preset!r}')
        expected = manifest.get('datasets')
        if isinstance(expected, list) and expected:
            actual = {row.dataset for row in rows}
            missing = sorted(set(expected) - actual)
            extra = sorted(actual - set(expected))
            if missing or extra:
                print(f'[warn] {path}: dataset set differs from '
                      f'run_manifest.json (missing={missing}, extra={extra})')
    return rows


def _run_variant(model: str, preset: str | None) -> str:
    if preset:
        return preset
    # Use the model config's default variant so a bare report does not mix
    # several model variants or baseline ablations.
    try:
        from .registry import load_models
        config = load_models().get(model)
        defaults = config.defaults if config is not None else {}
        return str(defaults.get('preset') or 'default')
    except Exception:
        return 'default'


def load_runs(models: Iterable[str], seeds: Iterable[int],
              result_root: Path,
              preset: str | None = None,
              allow_partial: bool = False
              ) -> tuple[list[MetricRow], list[str]]:
    rows: list[MetricRow] = []
    missing: list[str] = []
    unknown = validate_groups(models)
    if unknown:
        print(f'[warn] unclassified method group(s): {unknown}; '
              'add them to method_groups.py before a paper table')
    for model in models:
        # --preset selects a model variant; otherwise use the config default.
        variant = preset or _run_variant(model, None)
        for seed in seeds:
            run_root = result_root / model / variant / f'seed_{seed}'
            # Only the legacy default layout is allowed to fall back; never
            # substitute a different variant for the configured preset.
            legacy = result_root / model / f'seed_{seed}'
            if (not run_root.is_dir() and variant == 'default'
                    and legacy.is_dir()):
                print(f'[warn] using legacy layout {legacy}')
                run_root = legacy
            current = load_run(run_root, model, seed, preset=variant,
                               allow_partial=allow_partial)
            if not current:
                missing.append(str(run_root))
            rows.extend(current)
    return rows, missing


def _value(item, key: str):
    return item[key] if isinstance(item, dict) else getattr(item, key)


def _group(rows: Iterable, keys: tuple[str, ...]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(_value(row, key) for key in keys)].append(row)
    return grouped


def build_tables(rows: list[MetricRow]):
    macro = []
    for (group, model, seed, role, metric), items in sorted(_group(
            rows, ('group', 'model', 'seed', 'role', 'metric')).items()):
        values = [item.mean for item in items if math.isfinite(item.mean)]
        if not values:
            continue
        macro.append({
            'group': group,
            'model': model,
            'seed': seed,
            'role': role,
            'metric': metric,
            'n_datasets': len(values),
            'mean': statistics.fmean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
        })

    variance = []
    for (group, model, role, metric), items in sorted(_group(
            macro, ('group', 'model', 'role', 'metric')).items()):
        values = [item['mean'] for item in items]
        n_datasets = [item['n_datasets'] for item in items]
        variance.append({
            'group': group,
            'model': model,
            'role': role,
            'metric': metric,
            'n_seeds': len(values),
            'n_datasets': min(n_datasets) if n_datasets else 0,
            'mean': statistics.fmean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    return macro, variance


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    try:
        with open(tmp, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _fmt(value: float, metric: str) -> str:
    if not math.isfinite(value):
        return '--'
    if metric == 'hd95':
        return f'{value:.2f}'
    return f'{value * 100.0:.2f}'


def write_markdown(path: Path, variance: list[dict]) -> None:
    by_key = {(row['group'], row['model'], row['role'], row['metric']): row
              for row in variance}
    groups = [
        group for group in GROUP_ORDER
        if any(row.get('group', 'unclassified') == group
               for row in variance)
    ]
    lines = [
        '# Paper results',
        '',
        'Mean over seeds; ± is the standard deviation across seeds. '
        'Dice/IoU/clDice are percentages; HD95 is pixels (lower is better). '
        'Groups use different task/supervision settings and must not be '
        'compared as one main table.',
        '',
    ]
    for group in groups:
        group_rows = [row for row in variance
                      if row.get('group', 'unclassified') == group]
        models = sorted({row['model'] for row in group_rows})
        lines += [f'## {GROUP_LABELS.get(group, "Unclassified")}', '']
        for role in ROLES:
            if not any(row['role'] == role for row in group_rows):
                continue
            lines += [f'### {role}', '',
                      '| Model | Dice (%) | IoU (%) | clDice (%) | HD95 |',
                      '|---|---:|---:|---:|---:|']
            for model in models:
                cells = []
                for metric in METRICS:
                    item = by_key.get((group, model, role, metric))
                    if item is None:
                        cells.append('--')
                    else:
                        sd = (item['std'] * 100.0
                              if metric != 'hd95' else item['std'])
                        cells.append(
                            f"{_fmt(item['mean'], metric)} ± {sd:.2f}")
                lines.append(f'| {model} | ' + ' | '.join(cells) + ' |')
            lines.append('')
    _atomic_write_text(path, '\n'.join(lines) + '\n')


def write_report(rows: list[MetricRow], out_dir: Path) -> dict[str, Path]:
    macro, variance = build_tables(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        'all_metrics': out_dir / 'all_metrics.csv',
        'macro_metrics': out_dir / 'macro_metrics.csv',
        'seed_variance': out_dir / 'seed_variance.csv',
        'main_results': out_dir / 'main_results.md',
    }
    write_csv(paths['all_metrics'],
              ['group', 'model', 'seed', 'dataset', 'role', 'metric',
               'mean', 'std'],
              [row.__dict__ for row in rows])
    write_csv(paths['macro_metrics'],
              ['group', 'model', 'seed', 'role', 'metric', 'n_datasets',
               'mean', 'std'],
              macro)
    write_csv(paths['seed_variance'],
              ['group', 'model', 'role', 'metric', 'n_seeds', 'n_datasets',
               'mean', 'std'],
              variance)
    write_markdown(paths['main_results'], variance)
    return paths


def resolve_result_root(path: str | Path | None) -> Path:
    if path is None:
        return REPO_ROOT / 'results' / 'runs'
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def check_protocol_seeds(protocol: ProtocolConfig, seeds: list[int]) -> None:
    unknown = sorted(set(seeds) - set(protocol.seeds))
    if unknown:
        raise ValueError(
            f'protocol {protocol.name!r} has seeds {list(protocol.seeds)}, '
            f'requested unknown {unknown}')
