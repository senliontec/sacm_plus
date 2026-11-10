#!/usr/bin/env python3
"""Paired analysis for the SACM-STC B0-B3 ablation.

Reads the canonical per-run tables written by ``run_baseline_test.py``:

    results/runs/sacm_stc/<arm>/seed_<s>/eval/tables/macro_metrics.csv
    results/runs/sacm_stc/<arm>/seed_<s>/eval/tables/all_metrics.csv

and, when present, the candidate diagnostics:

    results/ablations/sacm_stc/candidate_diagnostics/<arm>/seed_<s>/
        selection_by_role.csv

The output is descriptive and paired by seed.  It never trains, selects
checkpoints, or changes an experiment.  Contrasts, arms and metrics come from
the YAML config.

Controls:
  * primary metric/role are pre-specified in the config;
  * all metrics and roles are still reported;
  * paired t interval is seed-level, with exact win/loss counts;
  * the dataset bootstrap resamples the 20 macro datasets, holding seeds
    paired, and is labelled separately from seed-level uncertainty;
  * a missing arm/seed is reported as a missing pair, never silently imputed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / 'configs' / 'ablations' /
                  'sacm_stc_scope_structure.yaml')
METRICS = ('dice', 'iou', 'cldice', 'hd95')
ROLES = ('All', 'Seen', 'Unseen', 'Novel')
DISPLAY = {'dice': 'Dice', 'iou': 'IoU', 'cldice': 'clDice', 'hd95': 'HD95'}


def _resolve(path_text) -> Path:
    path = Path(str(path_text))
    return path if path.is_absolute() else REPO_ROOT / path


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def _float(value, default=float('nan')) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value, metric: str, digits=3) -> str:
    if value is None or not math.isfinite(float(value)):
        return '—'
    value = float(value)
    if metric == 'hd95':
        return f'{value:.{digits}f}'
    return f'{value * 100:.{digits}f}'


def _signed(value, metric: str, digits=3) -> str:
    if value is None or not math.isfinite(float(value)):
        return '—'
    value = float(value)
    if metric == 'hd95':
        return f'{value:+.{digits}f}'
    return f'{value * 100:+.{digits}f}'


def _metric_suffix(metric: str) -> str:
    return '%' if metric != 'hd95' else 'px'


def _run_root(config, arm: str, seed: int) -> Path:
    return _resolve(config['output_root']) / arm / f'seed_{seed}'


def _training_incomplete_reason(run_root: Path, epochs) -> str | None:
    """Return why a run must not enter the paired analysis, or None."""
    path = run_root / 'run_info.json'
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return 'missing or invalid run_info.json'
    if payload.get('status') != 'done':
        return f'training status={payload.get("status")!r}, expected done'
    if epochs is not None:
        try:
            recorded = int(payload.get('epochs'))
        except (TypeError, ValueError):
            return 'run_info.json has no valid epochs'
        if recorded != int(epochs):
            return f'training epochs={recorded}, expected {int(epochs)}'
    return None


def _eval_incomplete_reason(config, arm: str, seed: int) -> str | None:
    """Return why a 20-dataset eval table is not safe to analyse, or None."""
    run_root = _run_root(config, arm, seed)
    table = run_root / 'eval' / 'tables' / 'all_metrics.csv'
    try:
        rows = _read_csv(table)
    except (OSError, csv.Error):
        return 'missing or invalid eval/tables/all_metrics.csv'
    expected = int(config.get('expected_datasets', 20))
    datasets = {row.get('dataset') for row in rows if row.get('dataset')}
    if len(datasets) < expected:
        return (f'eval table has {len(datasets)}/{expected} datasets '
                '(eval incomplete)')
    manifest = run_root / 'eval' / 'run_manifest.json'
    try:
        payload = json.loads(manifest.read_text())
    except (OSError, ValueError, TypeError):
        return 'missing or invalid eval/run_manifest.json'
    if payload.get('status') not in (None, 'done'):
        return f'eval status={payload.get("status")!r}, expected done'
    checkpoint_name = config.get('eval_checkpoint')
    if checkpoint_name:
        expected_name = Path(str(checkpoint_name)).name
        recorded = payload.get('checkpoint')
        if not recorded or Path(recorded).name != expected_name:
            return (f'eval checkpoint={recorded!r}, '
                    f'expected {expected_name}')
    return None


def _load_macro(config, arm: str, seed: int) -> dict | None:
    path = (_run_root(config, arm, seed) / 'eval' / 'tables' /
            'macro_metrics.csv')
    if not path.is_file():
        return None
    values = {}
    for row in _read_csv(path):
        role = row.get('role', '').strip()
        metric = row.get('metric', '').strip().lower()
        if role in ROLES and metric in METRICS:
            values[(role, metric)] = _float(row.get('mean'))
    return values


def _load_dataset(config, arm: str, seed: int) -> dict | None:
    path = (_run_root(config, arm, seed) / 'eval' / 'tables' /
            'all_metrics.csv')
    if not path.is_file():
        return None
    values = {}
    for row in _read_csv(path):
        dataset = row.get('dataset')
        role = row.get('role')
        if not dataset:
            continue
        for metric in METRICS:
            key = f'{metric}_mean'
            if key in row:
                values[(dataset, role, metric)] = _float(row.get(key))
    return values


def _load_candidate_role(config, arm: str, seed: int) -> dict | None:
    diagnostics = config.get('diagnostics') or {}
    root = _resolve(diagnostics.get(
        'output_root', 'results/ablations/sacm_stc/candidate_diagnostics'))
    path = root / arm / f'seed_{seed}' / 'selection_by_role.csv'
    if not path.is_file():
        return None
    rows = _read_csv(path)
    values = {}
    keys = (
        'selected_dice', 'selected_cldice', 'oracle_dice', 'worst_dice',
        'gating_top1', 'dice_regret', 'dice_spread',
        'pairwise_disagreement', 'pairwise_iou',
    )
    for row in rows:
        split = str(row.get('split', '')).strip().lower()
        if split != 'test' and not split.endswith('_test'):
            continue
        role = row.get('role')
        if role not in ROLES:
            continue
        for key in keys:
            field = f'{key}_mean'
            if field in row:
                values[(role, key)] = _float(row.get(field))
    return values


def _paired_stats(left: list[float], right: list[float]) -> dict:
    pairs = [(a, b) for a, b in zip(left, right)
             if math.isfinite(a) and math.isfinite(b)]
    if not pairs:
        return {
            'n': 0, 'mean_diff': float('nan'), 'sd_diff': float('nan'),
            'ci_low': float('nan'), 'ci_high': float('nan'),
            't': float('nan'), 'p': float('nan'), 'sign_test_p': float('nan'),
            'win': 0, 'loss': 0, 'tie': 0,
        }
    a = np.asarray([pair[0] for pair in pairs], dtype=float)
    b = np.asarray([pair[1] for pair in pairs], dtype=float)
    diff = a - b
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else float('nan')
    if n > 1:
        try:
            from scipy import stats
            se = sd / math.sqrt(n)
            crit = float(stats.t.ppf(0.975, n - 1))
            t_stat, p_value = stats.ttest_rel(a, b)
            ci_low, ci_high = mean - crit * se, mean + crit * se
            t_stat = float(t_stat)
            p_value = float(p_value)
            try:
                sign_p = float(stats.binomtest(
                    int((diff > 0).sum()), n, 0.5,
                    alternative='two-sided').pvalue)
            except Exception:
                sign_p = float('nan')
        except Exception:
            ci_low = ci_high = t_stat = p_value = sign_p = float('nan')
    else:
        ci_low = ci_high = t_stat = p_value = sign_p = float('nan')
    return {
        'n': n,
        'mean_diff': mean,
        'sd_diff': sd,
        'ci_low': ci_low,
        'ci_high': ci_high,
        't': t_stat,
        'p': p_value,
        'sign_test_p': sign_p,
        'win': int((diff > 0).sum()),
        'loss': int((diff < 0).sum()),
        'tie': int((diff == 0).sum()),
    }


def _bootstrap_dataset(diffs: dict[str, float], samples: int,
                       seed: int) -> tuple[float, float]:
    datasets = sorted(diffs)
    if not datasets:
        return float('nan'), float('nan')
    values = np.asarray([diffs[dataset] for dataset in datasets],
                        dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(max(1, int(samples))):
        indices = rng.integers(0, len(values), size=len(values))
        means.append(float(values[indices].mean()))
    return float(np.percentile(means, 2.5)), float(
        np.percentile(means, 97.5))


def _dataset_diff_rows(config, contrast: dict, arms: list[str],
                       seeds: list[int], data: dict) -> list[dict]:
    del arms
    left, right = contrast['left'], contrast['right']
    rows = []
    left_data = data.get(left, {})
    right_data = data.get(right, {})
    for seed in seeds:
        left_seed = left_data.get(seed)
        right_seed = right_data.get(seed)
        if left_seed is None or right_seed is None:
            continue
        keys = set(left_seed) & set(right_seed)
        for (dataset, role, metric) in sorted(keys):
            if metric not in METRICS:
                continue
            a = left_seed[(dataset, role, metric)]
            b = right_seed[(dataset, role, metric)]
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            rows.append({
                'contrast': contrast['id'],
                'label': contrast.get('label', ''),
                'dataset': dataset,
                'role': role,
                'metric': metric,
                'seed': seed,
                'left': a,
                'right': b,
                'diff': a - b,
            })
    return rows


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _arm_summary_rows(config, macro: dict, arms: list[str],
                      seeds: list[int]) -> list[dict]:
    """Mean/std of each arm's own test metrics across seeds."""
    rows = []
    for arm in arms:
        display = str(
            (config.get('arms', {}).get(arm) or {}).get(
                'display_name', arm))
        for role in ROLES:
            for metric in METRICS:
                values = []
                for seed in seeds:
                    value = macro.get(arm, {}).get(seed, {}).get(
                        (role, metric))
                    if value is not None and math.isfinite(value):
                        values.append(value)
                if not values:
                    continue
                rows.append({
                    'arm': arm,
                    'display_name': display,
                    'role': role,
                    'metric': metric,
                    'n_seeds': len(values),
                    'mean': float(np.mean(values)),
                    'std': (float(np.std(values, ddof=1))
                            if len(values) > 1 else float('nan')),
                })
    return rows


def _markdown_report(config, contrast_summaries: list[dict],
                     dataset_summaries: list[dict],
                     candidate_summaries: list[dict],
                     arm_summaries: list[dict],
                     missing: list[str]) -> str:
    primary_metric = str((config.get('analysis') or {}).get(
        'primary_metric', 'dice'))
    primary_role = str((config.get('analysis') or {}).get(
        'primary_role', 'All'))
    lines = [
        f'# {config["name"]}: paired ablation report',
        '',
        f'- Protocol: `{config["protocol"]}`',
        f'- Seeds: {config.get("seeds")}',
        f'- Primary endpoint: **{primary_role} '
        f'{DISPLAY.get(primary_metric, primary_metric)}**',
        '- Differences are left minus right. Dice/IoU/clDice are percentage '
        'points; HD95 is pixels.',
        '',
    ]
    if missing:
        lines += ['## Missing runs', '']
        lines += [f'- {item}' for item in missing]
        lines.append('')
    if arm_summaries:
        lines += [
            '',
            f'## Arm test results ({primary_role} '
            f'{DISPLAY.get(primary_metric, primary_metric)})',
            '',
            '| arm | description | mean | std | seeds |',
            '|---|---|---:|---:|---:|',
        ]
        for row in arm_summaries:
            if (row['role'] == primary_role
                    and row['metric'] == primary_metric):
                lines.append(
                    f"| {row['arm']} | {row['display_name']} | "
                    f"{_fmt(row['mean'], primary_metric)} "
                    f"{_metric_suffix(primary_metric)} | "
                    f"{_fmt(row['std'], primary_metric)} | "
                    f"{row['n_seeds']} |")
        lines.append('')
    lines += [
        f'## Primary ({primary_role} '
        f'{DISPLAY.get(primary_metric, primary_metric)})',
        '',
        '| contrast | label | mean diff | 95% seed CI | win/loss/tie | seeds | '
        'pair t p | sign p |',
        '|---|---|---:|---|---:|---:|---:|---:|',
    ]
    for row in contrast_summaries:
        if (row['metric'] == primary_metric
                and row['role'] == primary_role):
            lines.append(
                f"| {row['contrast']} | {row['label']} | "
                f"{_signed(row['mean_diff'], primary_metric)} "
                f"{_metric_suffix(primary_metric)} | "
                f"[{_signed(row['ci_low'], primary_metric)}, "
                f"{_signed(row['ci_high'], primary_metric)}] | "
                f"{row['win']}/{row['loss']}/{row['tie']} | "
                f"{row['n']} | {row['p']:.3f} | {row['sign_test_p']:.3f} |")
    lines += [
        '',
        '## All metrics and roles',
        '',
        '| contrast | role | metric | mean diff | 95% seed CI | '
        'win/loss/tie | n |',
        '|---|---|---|---:|---|---:|---:|',
    ]
    for row in contrast_summaries:
        lines.append(
            f"| {row['contrast']} | {row['role']} | "
            f"{DISPLAY.get(row['metric'], row['metric'])} | "
            f"{_signed(row['mean_diff'], row['metric'])} "
            f"{_metric_suffix(row['metric'])} | "
            f"[{_signed(row['ci_low'], row['metric'])}, "
            f"{_signed(row['ci_high'], row['metric'])}] | "
            f"{row['win']}/{row['loss']}/{row['tie']} | {row['n']} |")
    if dataset_summaries:
        lines += [
            '',
            '## Dataset-level paired bootstrap',
            '',
            'The interval resamples the macro datasets while keeping seeds '
            'paired; it is not a substitute for independent seeds.',
            '',
            '| contrast | role | metric | mean dataset diff | 95% dataset CI | '
            'win/loss | datasets |',
            '|---|---|---|---:|---|---:|---:|',
        ]
        for row in dataset_summaries:
            lines.append(
                f"| {row['contrast']} | {row['role']} | "
                f"{DISPLAY.get(row['metric'], row['metric'])} | "
                f"{_signed(row['mean_diff'], row['metric'])} "
                f"{_metric_suffix(row['metric'])} | "
                f"[{_signed(row['ci_low'], row['metric'])}, "
                f"{_signed(row['ci_high'], row['metric'])}] | "
                f"{row['win']}/{row['loss']} | {row['n_datasets']} |")
    if candidate_summaries:
        lines += [
            '',
            '## Candidate-head diagnostics',
            '',
            '| contrast | role | key | mean diff | 95% seed CI | '
            'win/loss/tie | n |',
            '|---|---|---|---:|---|---:|---:|',
        ]
        for row in candidate_summaries:
            lines.append(
                f"| {row['contrast']} | {row['role']} | {row['key']} | "
                f"{row['mean_diff']:+.4f} | "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
                f"{row['win']}/{row['loss']}/{row['tie']} | {row['n']} |")
    lines.append('')
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--arms', nargs='*', default=None)
    parser.add_argument('--seeds', nargs='*', default=None)
    args = parser.parse_args(argv)

    config_path = args.config
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with open(config_path) as handle:
        config = yaml.safe_load(handle) or {}
    arms = list(config.get('arms', {}))
    if args.arms:
        requested = [item for value in args.arms
                     for item in str(value).split(',') if item]
        unknown = [arm for arm in requested if arm not in config['arms']]
        if unknown:
            raise SystemExit(f'unknown arm(s): {unknown}')
        arms = list(dict.fromkeys(requested))
    seeds = [int(value) for value in (
        args.seeds if args.seeds is not None
        else config.get('seeds', []))]
    analysis = config.get('analysis') or {}
    required = [arm for contrast in config.get('contrasts', [])
                for arm in (contrast['left'], contrast['right'])]
    arms = list(dict.fromkeys(arms + [
        arm for arm in required if arm in config['arms']]))
    output_root = _resolve(analysis.get(
        'output_root', 'results/ablations/sacm_stc/analysis'))
    samples = int(analysis.get('bootstrap_samples', 2000))
    bootstrap_seed = int(analysis.get('seed', 20261109))

    macro = {}
    dataset = {}
    candidate = {}
    missing = []
    require_complete = bool(
        analysis.get('require_train_complete', False))
    require_eval_complete = bool(
        analysis.get('require_eval_complete', False))
    expected_epochs = config.get('epochs')
    for arm in arms:
        for seed in seeds:
            if require_complete:
                reason = _training_incomplete_reason(
                    _run_root(config, arm, seed), expected_epochs)
                if reason is not None:
                    missing.append(f'{arm}/seed_{seed}: {reason}')
                    continue
            if require_eval_complete:
                reason = _eval_incomplete_reason(config, arm, seed)
                if reason is not None:
                    missing.append(f'{arm}/seed_{seed}: {reason}')
                    continue
            values = _load_macro(config, arm, seed)
            if values is None:
                missing.append(f'{arm}/seed_{seed}: missing macro_metrics.csv')
            else:
                macro.setdefault(arm, {})[seed] = values
            values = _load_dataset(config, arm, seed)
            if values is None:
                missing.append(f'{arm}/seed_{seed}: missing all_metrics.csv')
            else:
                dataset.setdefault(arm, {})[seed] = values
            values = _load_candidate_role(config, arm, seed)
            if values is not None:
                candidate.setdefault(arm, {})[seed] = values

    arm_summaries = _arm_summary_rows(config, macro, arms, seeds)
    contrast_summaries = []
    by_seed_rows = []
    dataset_rows = []
    dataset_summaries = []
    candidate_summaries = []
    for contrast in config.get('contrasts', []):
        left, right = contrast['left'], contrast['right']
        if left not in macro or right not in macro:
            missing.append(
                f'contrast {contrast["id"]}: missing arm {left} or {right}')
            continue
        paired_seeds = sorted(set(macro[left]) & set(macro[right]))
        for role in ROLES:
            for metric in METRICS:
                left_values = []
                right_values = []
                for seed in paired_seeds:
                    a = macro[left][seed].get((role, metric))
                    b = macro[right][seed].get((role, metric))
                    if a is None or b is None:
                        continue
                    left_values.append(a)
                    right_values.append(b)
                    by_seed_rows.append({
                        'contrast': contrast['id'],
                        'label': contrast.get('label', ''),
                        'role': role,
                        'metric': metric,
                        'seed': seed,
                        'left': a,
                        'right': b,
                        'diff': a - b,
                    })
                stats = _paired_stats(left_values, right_values)
                if not stats['n']:
                    continue
                contrast_summaries.append({
                    'contrast': contrast['id'],
                    'label': contrast.get('label', ''),
                    'role': role,
                    'metric': metric,
                    **stats,
                })
        # Per-dataset paired differences and dataset bootstrap.
        diff_rows = _dataset_diff_rows(
            config, contrast, arms, paired_seeds, dataset)
        dataset_rows.extend(diff_rows)
        if diff_rows:
            grouped = defaultdict(dict)
            for row in diff_rows:
                grouped[(row['role'], row['metric'])][
                    (row['dataset'], row['seed'])] = row['diff']
            for (role, metric), cells in grouped.items():
                by_dataset = defaultdict(list)
                for (dataset_name, _seed), diff in cells.items():
                    by_dataset[dataset_name].append(diff)
                if not by_dataset:
                    continue
                dataset_means = {
                    name: float(np.mean(values))
                    for name, values in by_dataset.items()
                    if values and np.isfinite(values).any()}
                if not dataset_means:
                    continue
                values = np.asarray(list(dataset_means.values()),
                                    dtype=float)
                low, high = _bootstrap_dataset(
                    dataset_means, samples, bootstrap_seed)
                dataset_summaries.append({
                    'contrast': contrast['id'],
                    'label': contrast.get('label', ''),
                    'role': role,
                    'metric': metric,
                    'n_datasets': len(values),
                    'mean_diff': float(values.mean()),
                    'ci_low': low,
                    'ci_high': high,
                    'win': int((values > 0).sum()),
                    'loss': int((values < 0).sum()),
                })
        # Candidate diagnostics are only meaningful for contrasts with
        # available per-arm diagnostics; missing data is skipped.
        if (left in candidate and right in candidate):
            paired_candidate_seeds = sorted(
                set(candidate[left]) & set(candidate[right]))
            keys = set()
            for seed in paired_candidate_seeds:
                keys |= set(candidate[left][seed]) & set(candidate[right][seed])
            for role in ROLES:
                for key in sorted(k for k in keys if k[0] == role):
                    _, metric = key
                    a_values, b_values = [], []
                    for seed in paired_candidate_seeds:
                        a = candidate[left][seed].get(key)
                        b = candidate[right][seed].get(key)
                        if a is None or b is None:
                            continue
                        a_values.append(a)
                        b_values.append(b)
                    stats = _paired_stats(a_values, b_values)
                    if not stats['n']:
                        continue
                    candidate_summaries.append({
                        'contrast': contrast['id'],
                        'label': contrast.get('label', ''),
                        'role': role,
                        'key': metric,
                        **stats,
                    })

    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_root / 'arm_summary.csv', arm_summaries,
        ['arm', 'display_name', 'role', 'metric', 'n_seeds', 'mean', 'std'])
    _write_csv(
        output_root / 'contrasts_by_seed.csv', by_seed_rows,
        ['contrast', 'label', 'role', 'metric', 'seed', 'left', 'right',
         'diff'])
    _write_csv(
        output_root / 'contrast_summary.csv', contrast_summaries,
        ['contrast', 'label', 'role', 'metric', 'n', 'mean_diff', 'sd_diff',
         'ci_low', 'ci_high', 't', 'p', 'sign_test_p', 'win', 'loss', 'tie'])
    _write_csv(
        output_root / 'contrast_by_dataset.csv', dataset_rows,
        ['contrast', 'label', 'dataset', 'role', 'metric', 'seed',
         'left', 'right', 'diff'])
    _write_csv(
        output_root / 'contrast_by_dataset_summary.csv', dataset_summaries,
        ['contrast', 'label', 'role', 'metric', 'n_datasets', 'mean_diff',
         'ci_low', 'ci_high', 'win', 'loss'])
    _write_csv(
        output_root / 'candidate_contrast_summary.csv', candidate_summaries,
        ['contrast', 'label', 'role', 'key', 'n', 'mean_diff', 'sd_diff',
         'ci_low', 'ci_high', 't', 'p', 'sign_test_p', 'win', 'loss', 'tie'])
    (output_root / 'ablation_report.md').write_text(
        _markdown_report(config, contrast_summaries, dataset_summaries,
                         candidate_summaries, arm_summaries, missing))
    (output_root / 'summary.json').write_text(json.dumps({
        'config': str(config_path),
        'arms': arms,
        'seeds': seeds,
        'n_arm_rows': len(arm_summaries),
        'n_contrasts': len(config.get('contrasts', [])),
        'n_contrast_rows': len(contrast_summaries),
        'n_dataset_rows': len(dataset_rows),
        'n_candidate_rows': len(candidate_summaries),
        'missing': missing,
        'output_root': str(output_root),
    }, indent=2) + '\n')

    print(f'wrote ablation analysis to {output_root}')
    print(f'  arms={len(arm_summaries)} '
          f'contrasts={len(contrast_summaries)} '
          f'dataset_rows={len(dataset_rows)} '
          f'candidate_rows={len(candidate_summaries)}')
    if missing:
        print(f'  missing/incomplete: {len(missing)}')
        for item in missing[:10]:
            print(f'    - {item}')
    return 0 if contrast_summaries else 2


if __name__ == '__main__':
    raise SystemExit(main())
