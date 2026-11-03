#!/usr/bin/env python3
"""Build a cross-model comparison from audit_predictions.py outputs.

Each ``--audit NAME=DIR`` must point at a directory containing the
``per_dataset.csv`` and ``macro_summary.csv`` written by
``scripts/pipeline/audit_predictions.py``.  The audit trees are read-only.

Outputs, under ``--out``:

* ``macro_comparison.csv``: long-form macro rows for every scope/model;
* ``per_dataset_comparison.csv``: long-form per-dataset rows;
* ``comparison.md``: human-readable macro, per-dataset and failure tables;
* ``summary.json``: coverage warnings plus the same macro/per-dataset rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

SCOPE_ORDER = {'Seen': 0, 'Unseen': 1, 'Novel': 2, 'All': 3}
ROLE_ORDER = {'Seen': 0, 'Unseen': 1, 'Novel': 2}

# (CSV field, short label in Markdown)
METRIC_FIELDS = [
    ('dice_mean', 'Dice'),
    ('iou_mean', 'IoU'),
    ('cldice_mean', 'clDice'),
    ('hd95_mean', 'HD95'),
    ('precision_mean', 'P'),
    ('recall_mean', 'R'),
    ('components_pred_mean', 'Comp pred'),
    ('components_gt_mean', 'Comp GT'),
    ('components_ratio_mean', 'Comp ratio'),
    ('small_component_rate_mean', 'Small comp'),
    ('fragmented_rate', 'Frag'),
    ('overseg_rate', 'Over'),
    ('underseg_rate', 'Under'),
    ('hole_mismatch_rate', 'Hole'),
    ('betti_mean', 'Betti'),
    ('skeleton_precision_mean', 'Skel P'),
    ('skeleton_recall_mean', 'Skel R'),
]
METRIC_KEYS = [key for key, _label in METRIC_FIELDS]
COUNT_FIELDS = ['n_images', 'n_ok', 'n_size_mismatch', 'n_unmatched_pred']


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return number if math.isfinite(number) else float('nan')


def _int(value: Any) -> int:
    number = _float(value)
    return int(number) if math.isfinite(number) else 0


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    return '—' if not math.isfinite(number) else f'{number:.{digits}f}'


def _pct(value: Any) -> str:
    number = _float(value)
    return '—' if not math.isfinite(number) else f'{100.0 * number:.1f}%'


def _signed_pp(value: Any) -> str:
    number = _float(value)
    return '—' if not math.isfinite(number) else f'{100.0 * number:+.2f} pp'


def _signed_num(value: Any, digits: int = 2) -> str:
    number = _float(value)
    return '—' if not math.isfinite(number) else f'{number:+.{digits}f}'


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: list[str],
               rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _parse_audits(specs: list[str]) -> list[tuple[str, Path]]:
    audits: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in specs:
        name, sep, raw_path = spec.partition('=')
        name = name.strip()
        raw_path = raw_path.strip()
        if not sep or not name or not raw_path:
            raise SystemExit(f'expected NAME=DIR, got {spec!r}')
        if name in seen:
            raise SystemExit(f'duplicate audit name: {name!r}')
        root = Path(raw_path)
        if not (root / 'per_dataset.csv').is_file():
            raise SystemExit(f'{name}: missing {root / "per_dataset.csv"}')
        if not (root / 'macro_summary.csv').is_file():
            raise SystemExit(f'{name}: missing {root / "macro_summary.csv"}')
        seen.add(name)
        audits.append((name, root))
    if not audits:
        raise SystemExit('pass at least one --audit NAME=DIR')
    return audits


def _scope_key(scope: str) -> tuple[int, str]:
    return (SCOPE_ORDER.get(scope, 9), scope)


def _dataset_key(item: tuple[str, str]) -> tuple[int, str]:
    dataset, role = item
    return (ROLE_ORDER.get(role, 9), dataset)


def _best_models(
    rows: dict[str, dict[str, str]], key: str, higher: bool,
) -> set[str]:
    values: list[tuple[str, float]] = []
    for model, row in rows.items():
        value = _float(row.get(key))
        if math.isfinite(value):
            values.append((model, value))
    if not values:
        return set()
    target = max(value for _model, value in values) if higher else min(
        value for _model, value in values)
    return {
        model for model, value in values
        if math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12)
    }


def _cell(
    rows: dict[str, dict[str, str]], model: str, key: str,
    best: set[str], *, digits: int = 4, percent: bool = False,
) -> str:
    value = rows.get(model, {}).get(key)
    text = _pct(value) if percent else _fmt(value, digits)
    return f'**{text}**' if model in best else text


def _coverage_warnings(
    model_order: list[str],
    per_by_model: dict[str, list[dict[str, str]]],
) -> list[str]:
    warnings: list[str] = []
    by_model = {
        name: {row.get('dataset', ''): row for row in rows}
        for name, rows in per_by_model.items()
    }
    all_datasets = set().union(*(set(rows) for rows in by_model.values()))
    for name in model_order:
        missing = sorted(all_datasets - set(by_model[name]))
        if missing:
            warnings.append(
                f'{name}: missing datasets: {", ".join(missing)}')
    for dataset in sorted(all_datasets):
        counts = {}
        for name in model_order:
            row = by_model[name].get(dataset)
            if row is not None:
                counts[name] = _int(row.get('n_images'))
        if len(set(counts.values())) > 1:
            detail = ', '.join(
                f'{name}={count}' for name, count in counts.items())
            warnings.append(f'{dataset}: n_images differ ({detail})')
        for name in model_order:
            row = by_model[name].get(dataset)
            if row is None:
                continue
            mismatch = _int(row.get('n_size_mismatch'))
            unmatched = _int(row.get('n_unmatched_pred'))
            if mismatch:
                warnings.append(
                    f'{dataset}/{name}: {mismatch} size mismatch')
            if unmatched:
                warnings.append(
                    f'{dataset}/{name}: {unmatched} unmatched prediction')
    return warnings


def _build_macro_rows(
    model_order: list[str],
    macro_by_model: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, Any]]:
    scopes = sorted(
        set().union(*(set(rows) for rows in macro_by_model.values())),
        key=_scope_key,
    )
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        for model in model_order:
            source = macro_by_model[model].get(scope, {})
            row: dict[str, Any] = {
                'scope': scope,
                'model': model,
                'n_datasets': source.get('n_datasets', ''),
            }
            for key in METRIC_KEYS:
                row[key] = source.get(key, '')
            rows.append(row)
    return rows


def _build_per_dataset_rows(
    model_order: list[str],
    per_by_model: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    datasets: dict[str, dict[str, Any]] = {}
    for model in model_order:
        for source in per_by_model[model]:
            dataset = str(source.get('dataset', ''))
            if not dataset:
                continue
            item = datasets.setdefault(dataset, {
                'role': str(source.get('role', '')), 'models': {},
            })
            if not item['role']:
                item['role'] = str(source.get('role', ''))
            item['models'][model] = source

    rows: list[dict[str, Any]] = []
    for dataset, item in sorted(
            datasets.items(), key=lambda pair: _dataset_key(
                (pair[0], pair[1]['role']))):
        for model in model_order:
            source = item['models'].get(model, {})
            row: dict[str, Any] = {
                'dataset': dataset,
                'role': item['role'],
                'model': model,
            }
            for field in COUNT_FIELDS:
                row[field] = source.get(field, '')
            for key in METRIC_KEYS:
                row[key] = source.get(key, '')
            rows.append(row)
    return rows


def _write_report(
    path: Path,
    model_order: list[str],
    audits: list[tuple[str, Path]],
    primary: str,
    macro_rows: list[dict[str, Any]],
    per_rows: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    macro_by_scope: dict[str, dict[str, dict[str, Any]]] = {}
    for row in macro_rows:
        macro_by_scope.setdefault(str(row['scope']), {})[str(row['model'])] = row

    dataset_models: dict[str, dict[str, dict[str, Any]]] = {}
    roles: dict[str, str] = {}
    for row in per_rows:
        dataset = str(row['dataset'])
        roles[dataset] = str(row['role'])
        dataset_models.setdefault(dataset, {})[str(row['model'])] = row

    lines = [
        '# Audit comparison',
        '',
        'Sources:',
        '',
    ]
    for name, root in audits:
        lines.append(f'- **{name}**: `{root}`')
    lines += [
        '',
        f'Primary model: **{primary}**',
        '',
        '## Coverage',
        '',
    ]
    if warnings:
        lines.extend(f'- {item}' for item in warnings)
    else:
        lines.append('- All models share the same dataset set; '
                     'no size/unmatched warnings.')

    lines += [
        '',
        '## Macro (equal dataset weight)',
        '',
        '| Scope | Model | n | Dice | IoU | clDice | HD95 ↓ | P | R | '
        'Comp ratio | Small comp | Frag | Over | Under | Hole |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for scope in sorted(macro_by_scope, key=_scope_key):
        rows = macro_by_scope[scope]
        for model in model_order:
            row = rows.get(model, {})
            lines.append(
                f"| {scope} | {model} | {row.get('n_datasets', '')} | "
                f"{_fmt(row.get('dice_mean'))} | "
                f"{_fmt(row.get('iou_mean'))} | "
                f"{_fmt(row.get('cldice_mean'))} | "
                f"{_fmt(row.get('hd95_mean'))} | "
                f"{_fmt(row.get('precision_mean'))} | "
                f"{_fmt(row.get('recall_mean'))} | "
                f"{_fmt(row.get('components_ratio_mean'), 2)} | "
                f"{_pct(row.get('small_component_rate_mean'))} | "
                f"{_pct(row.get('fragmented_rate'))} | "
                f"{_pct(row.get('overseg_rate'))} | "
                f"{_pct(row.get('underseg_rate'))} | "
                f"{_pct(row.get('hole_mismatch_rate'))} |"
            )

    others = [name for name in model_order if name != primary]
    if others:
        lines += [
            '',
            f'## Macro deltas: {primary} minus comparison model',
            '',
            '| Scope | Compared with | Dice Δ | IoU Δ | clDice Δ | '
            'HD95 Δ (px) | Comp ratio Δ |',
            '|---|---|---:|---:|---:|---:|---:|',
        ]
        for scope in sorted(macro_by_scope, key=_scope_key):
            rows = macro_by_scope[scope]
            primary_row = rows.get(primary, {})
            for other in others:
                other_row = rows.get(other, {})
                lines.append(
                    f'| {scope} | {other} | '
                    f"{_signed_pp(_float(primary_row.get('dice_mean')) - _float(other_row.get('dice_mean')))} | "
                    f"{_signed_pp(_float(primary_row.get('iou_mean')) - _float(other_row.get('iou_mean')))} | "
                    f"{_signed_pp(_float(primary_row.get('cldice_mean')) - _float(other_row.get('cldice_mean')))} | "
                    f"{_signed_num(_float(primary_row.get('hd95_mean')) - _float(other_row.get('hd95_mean')), 2)} | "
                    f"{_signed_num(_float(primary_row.get('components_ratio_mean')) - _float(other_row.get('components_ratio_mean')), 2)} |"
                )

    lines += [
        '',
        '## Per-dataset Dice / clDice / HD95',
        '',
    ]
    header = ['Dataset', 'Role']
    for metric in ('dice_mean', 'cldice_mean', 'hd95_mean'):
        label = {
            'dice_mean': 'Dice',
            'cldice_mean': 'clDice',
            'hd95_mean': 'HD95',
        }[metric]
        header.extend(f'{label} {name}' for name in model_order)
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('|---' * len(header) + '|')
    for dataset in sorted(dataset_models, key=lambda name: _dataset_key(
            (name, roles.get(name, '')))):
        models = dataset_models[dataset]
        dice_best = _best_models(models, 'dice_mean', higher=True)
        cldice_best = _best_models(models, 'cldice_mean', higher=True)
        hd95_best = _best_models(models, 'hd95_mean', higher=False)
        cells = [dataset, roles.get(dataset, '')]
        for model in model_order:
            cells.append(_cell(models, model, 'dice_mean', dice_best))
        for model in model_order:
            cells.append(_cell(models, model, 'cldice_mean', cldice_best))
        for model in model_order:
            cells.append(_cell(models, model, 'hd95_mean', hd95_best))
        lines.append('| ' + ' | '.join(cells) + ' |')

    if primary in model_order:
        lines += [
            '',
            f'## {primary} failure profile',
            '',
            '| Dataset | n | Dice | P | R | Comp pred/GT | Comp ratio | '
            'Small comp | Frag | Over | Under | Hole |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        ]
        for dataset in sorted(
                dataset_models,
                key=lambda name: _dataset_key(
                    (name, roles.get(name, '')))):
            row = dataset_models[dataset].get(primary, {})
            lines.append(
                f"| {dataset} | {row.get('n_images', '')} | "
                f"{_fmt(row.get('dice_mean'))} | "
                f"{_fmt(row.get('precision_mean'))} | "
                f"{_fmt(row.get('recall_mean'))} | "
                f"{_fmt(row.get('components_pred_mean'), 2)}/"
                f"{_fmt(row.get('components_gt_mean'), 2)} | "
                f"{_fmt(row.get('components_ratio_mean'), 2)} | "
                f"{_pct(row.get('small_component_rate_mean'))} | "
                f"{_pct(row.get('fragmented_rate'))} | "
                f"{_pct(row.get('overseg_rate'))} | "
                f"{_pct(row.get('underseg_rate'))} | "
                f"{_pct(row.get('hole_mismatch_rate'))} |"
            )

    lines += [
        '',
        '## Notes',
        '',
        '- Dice, clDice, P and R are higher-is-better; HD95 is pixels and '
        'lower-is-better.',
        '- `Frag` is the audit definition (predicted components >= '
        'max(5, 3 x GT) and at least 5 more than GT); a high rate on a '
        'high-Dice dataset can be caused by tiny extra components.',
        '- `Comp ratio` is components_pred / components_gt; use the absolute '
        'component counts in the failure profile to judge severity.',
        '- This comparison is diagnostic. It does not perform significance '
        'testing or confidence intervals.',
        '',
    ]
    path.write_text('\n'.join(lines) + '\n')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--audit', action='append', required=True, metavar='NAME=DIR',
        help='audit output directory; repeat for each model')
    parser.add_argument(
        '--primary', default='sacm_x',
        help='model used for the failure profile and delta tables '
             '(default: sacm_x)')
    parser.add_argument(
        '--out', type=Path, required=True,
        help='output directory for comparison.md/CSV/JSON')
    args = parser.parse_args()

    audits = _parse_audits(args.audit)
    model_order = [name for name, _root in audits]

    per_by_model: dict[str, list[dict[str, str]]] = {}
    macro_by_model: dict[str, dict[str, dict[str, str]]] = {}
    for name, root in audits:
        per_by_model[name] = _read_csv(root / 'per_dataset.csv')
        macro_rows = _read_csv(root / 'macro_summary.csv')
        macro_by_model[name] = {
            str(row.get('scope', '')): row for row in macro_rows
        }

    primary = args.primary
    if primary not in model_order:
        primary = model_order[-1]

    warnings = _coverage_warnings(model_order, per_by_model)
    macro_rows = _build_macro_rows(model_order, macro_by_model)
    per_rows = _build_per_dataset_rows(model_order, per_by_model)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    macro_fields = ['scope', 'model', 'n_datasets'] + METRIC_KEYS
    per_fields = ['dataset', 'role', 'model'] + COUNT_FIELDS + METRIC_KEYS
    _write_csv(out_dir / 'macro_comparison.csv', macro_fields, macro_rows)
    _write_csv(
        out_dir / 'per_dataset_comparison.csv', per_fields, per_rows)
    _write_report(
        out_dir / 'comparison.md', model_order, audits, primary,
        macro_rows, per_rows, warnings)

    payload = {
        'audits': {name: str(root) for name, root in audits},
        'primary': primary,
        'coverage_warnings': warnings,
        'macro': macro_rows,
        'per_dataset': per_rows,
    }
    (out_dir / 'summary.json').write_text(
        json.dumps(payload, indent=2, default=str) + '\n')

    print(f'Wrote: {out_dir / "macro_comparison.csv"}')
    print(f'Wrote: {out_dir / "per_dataset_comparison.csv"}')
    print(f'Wrote: {out_dir / "comparison.md"}')
    print(f'Wrote: {out_dir / "summary.json"}')
    if warnings:
        print(f'Coverage warnings: {len(warnings)} (see comparison.md)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
