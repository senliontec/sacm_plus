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
import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .protocols import ProtocolConfig
from .registry import REPO_ROOT

METRICS = ('dice', 'iou', 'cldice', 'hd95')
ROLES = ('Seen', 'Unseen', 'Novel')
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


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _role(value: str) -> str:
    text = (value or '').strip().lower()
    return {'seen': 'Seen', 'unseen': 'Unseen', 'novel': 'Novel'}.get(
        text, value)


def load_run(run_root: Path, model: str, seed: int,
             preset: str | None = None) -> list[MetricRow]:
    """Read one (model, variant, seed) canonical result directory.

    Canonical rows live at ``<run>/eval/tables/all_metrics.csv``.  The
    legacy fallbacks remain readable so a partially migrated tree can still
    be reported.
    """
    path = run_root / 'eval' / 'tables' / 'all_metrics.csv'
    if not path.is_file():
        path = run_root / 'tables' / 'all_metrics.csv'
    if not path.is_file():
        path = run_root / 'paper_metrics.csv'
    if not path.is_file():
        return []

    rows: list[MetricRow] = []
    # A preset only selects an SACM-X variant.  Baselines always live under
    # variant=default, so applying --preset to them would drop every row.
    filter_preset = preset if model == 'sacm_x' else None
    with open(path, newline='') as f:
        for raw in csv.DictReader(f):
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
                ))
    return rows


def _run_variant(model: str, preset: str | None) -> str:
    if preset:
        return preset
    # Use the model config's default variant so a bare report does not mix
    # several SACM-X presets or several baseline ablations.
    try:
        from .registry import load_models
        config = load_models().get(model)
        defaults = config.defaults if config is not None else {}
        return str(defaults.get('preset') or 'default')
    except Exception:
        return 'default'


def load_runs(models: Iterable[str], seeds: Iterable[int],
              result_root: Path,
              preset: str | None = None) -> tuple[list[MetricRow], list[str]]:
    rows: list[MetricRow] = []
    missing: list[str] = []
    for model in models:
        # --preset selects SACM-X's variant; baseline variants are always
        # default regardless of the requested SACM-X preset.
        variant = (preset if model == 'sacm_x' and preset
                   else _run_variant(model, None))
        for seed in seeds:
            run_root = result_root / model / variant / f'seed_{seed}'
            if not run_root.is_dir():
                # A run may use a non-default preset (e.g. SACM-X
                # full_typed).  With no explicit --preset, accept the only
                # available variant instead of silently reporting it missing.
                matches = sorted((result_root / model).glob(
                    f'*/seed_{seed}'))
                if len(matches) == 1:
                    run_root = matches[0]
            legacy = result_root / model / f'seed_{seed}'
            if not run_root.is_dir() and legacy.is_dir():
                run_root = legacy
            current = load_run(run_root, model, seed, preset=preset)
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
    for (model, seed, role, metric), items in sorted(_group(
            rows, ('model', 'seed', 'role', 'metric')).items()):
        values = [item.mean for item in items if math.isfinite(item.mean)]
        if not values:
            continue
        macro.append({
            'model': model,
            'seed': seed,
            'role': role,
            'metric': metric,
            'n_datasets': len(values),
            'mean': statistics.fmean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
        })

    variance = []
    for (model, role, metric), items in sorted(_group(
            macro, ('model', 'role', 'metric')).items()):
        values = [item['mean'] for item in items]
        variance.append({
            'model': model,
            'role': role,
            'metric': metric,
            'n_seeds': len(values),
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
    by_key = {(row['model'], row['role'], row['metric']): row
              for row in variance}
    models = sorted({row['model'] for row in variance})
    lines = [
        '# Paper results',
        '',
        'Mean over seeds; ± is the standard deviation across seeds. '
        'Dice/IoU/clDice are percentages; HD95 is pixels (lower is better).',
        '',
    ]
    for role in ROLES:
        if not any(row['role'] == role for row in variance):
            continue
        lines += [f'## {role}', '',
                  '| Model | Dice (%) | IoU (%) | clDice (%) | HD95 |',
                  '|---|---:|---:|---:|---:|']
        for model in models:
            cells = []
            for metric in METRICS:
                item = by_key.get((model, role, metric))
                if item is None:
                    cells.append('--')
                else:
                    sd = item['std'] * 100.0 if metric != 'hd95' else item['std']
                    cells.append(f"{_fmt(item['mean'], metric)} ± {sd:.2f}")
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
              ['model', 'seed', 'dataset', 'role', 'metric', 'mean', 'std'],
              [row.__dict__ for row in rows])
    write_csv(paths['macro_metrics'],
              ['model', 'seed', 'role', 'metric', 'n_datasets', 'mean', 'std'],
              macro)
    write_csv(paths['seed_variance'],
              ['model', 'role', 'metric', 'n_seeds', 'mean', 'std'],
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
