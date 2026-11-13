#!/usr/bin/env python3
"""Build results.tex in SACM Table-2 format from canonical eval tables.

Reads one ``results/runs/<model>/<variant>/seed_<seed>/eval/tables/
all_metrics.csv`` per selected seed.  Each cell is the mean over the selected
seeds; ``--with-std`` writes mean $\\pm$ std.  The layout is the same
per-dataset Table-2 layout used by scripts/pipeline/build_paper_table.py:
Dataset + Role rows, one four-metric block (Dice/IoU/clDice/HD95) per method,
Seen/Unseen/Novel/All macro rows, best bold and second best underlined.

Examples:
  python scripts/build_results_tex.py
  python scripts/build_results_tex.py --seeds 0 1 2 3
  python scripts/build_results_tex.py --seeds 0 --models sacm_gcd:all_dense
  python scripts/build_results_tex.py --with-std --no-bold
  python scripts/build_results_tex.py --out results.tex

The default method list matches the server runs:
  sacm_gcd:all_dense sacm_official:default unet:default cs2net:default
  maskvsc:default
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'scripts' / 'pipeline'))

from build_paper_table import (  # noqa: E402
    METRICS,
    ROLE_ORDER,
    _aggregate,
    _latex,
    _load_method,
    _parse_models,
    _protocol_roles,
)

DEFAULT_MODELS = [
    'sacm_gcd:all_dense',
    'sacm_official:default',
    'unet:default',
    'cs2net:default',
    'maskvsc:default',
]
METHOD_LABELS = {
    ('sacm_official', 'default'): 'SACM (official)',
    ('sacm_gcd', 'all_dense'): 'SACM-GCD',
    ('unet', 'default'): 'U-Net',
    ('cs2net', 'default'): 'CS2Net',
    ('maskvsc', 'default'): 'MaskVSC',
    ('bcunet', 'default'): 'BCU-Net',
}


def _label(model: str, variant: str) -> str:
    return METHOD_LABELS.get(
        (model, variant),
        model if variant == 'default' else f'{model}/{variant}')


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Build results.tex in SACM Table-2 format.')
    parser.add_argument('--repo-root', type=Path, default=None)
    parser.add_argument(
        '--models', nargs='+', default=DEFAULT_MODELS,
        help='model:variant specs (default: all server models)')
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=[0, 1, 2, 3],
        help='seeds to average (default: 0 1 2 3)')
    parser.add_argument('--expected-datasets', type=int, default=20)
    parser.add_argument(
        '--out', type=Path, default=Path('results.tex'),
        help='output .tex path (default: results.tex at repo root)')
    parser.add_argument(
        '--with-std', action='store_true',
        help='write mean $\\pm$ std instead of mean only')
    parser.add_argument(
        '--no-percent', dest='percent', action='store_false', default=True,
        help='show Dice/IoU/clDice as [0,1] fractions (default: percent)')
    parser.add_argument(
        '--decimals', type=int, default=None,
        help='display decimals (default: 2 for percent, 4 for fractions)')
    parser.add_argument('--no-bold', action='store_true')
    args = parser.parse_args()

    repo = (args.repo_root.resolve() if args.repo_root is not None
            else REPO_ROOT)
    out = args.out if args.out.is_absolute() else repo / args.out
    seeds = sorted(dict.fromkeys(args.seeds))
    if not seeds:
        raise SystemExit('--seeds cannot be empty')
    decimals = (args.decimals if args.decimals is not None
                else (2 if args.percent else 4))

    specs = _parse_models(args.models)
    labels = [_label(model, variant) for model, variant in specs]
    if len(set(labels)) != len(labels):
        raise SystemExit(
            f'duplicate method labels: {labels}; pass each model:variant once')

    method_data = {}
    roles = {}
    datasets = None
    for (model, variant), label in zip(specs, labels):
        data = _load_method(repo, model, variant, seeds,
                            args.expected_datasets)
        method_data[label] = data
        current = set(data[seeds[0]])
        if datasets is None:
            datasets = current
            roles = {dataset: data[seeds[0]][dataset]['role']
                     for dataset in datasets}
        elif current != datasets:
            raise SystemExit(
                f'{label}: dataset set differs from {labels[0]}; '
                f'missing={sorted(datasets - current)}, '
                f'extra={sorted(current - datasets)}')
        else:
            for dataset in datasets:
                if data[seeds[0]][dataset]['role'] != roles[dataset]:
                    raise SystemExit(
                        f'{label}: role for {dataset} differs from '
                        f'{labels[0]}')

    unknown_roles = sorted({role for role in roles.values()
                            if role not in ROLE_ORDER})
    if unknown_roles:
        raise SystemExit(
            f'unknown or missing dataset roles {unknown_roles}; expected '
            f'{sorted(ROLE_ORDER)}')

    protocol = (repo / 'data' / f'sacm_3shot_seed{seeds[0]}' /
                'protocol_manifest.csv')
    if protocol.is_file():
        expected_roles = _protocol_roles(protocol)
        if expected_roles:
            if set(expected_roles) != set(datasets):
                raise SystemExit(
                    f'{protocol}: test dataset set differs; '
                    f'missing={sorted(set(expected_roles) - set(datasets))}, '
                    f'extra={sorted(set(datasets) - set(expected_roles))}')
            for dataset in datasets:
                if expected_roles[dataset] != roles[dataset]:
                    raise SystemExit(
                        f'{protocol}: role for {dataset} is '
                        f'{roles[dataset]!r}, protocol says '
                        f'{expected_roles[dataset]!r}')
        else:
            print(f'[warn] {protocol}: no *_test rows; skipping name check')
    else:
        print(f'[warn] protocol manifest not found: {protocol}; '
              'skipping dataset-name check')

    datasets = sorted(
        datasets,
        key=lambda dataset: (ROLE_ORDER.get(roles[dataset], 9),
                             dataset.lower()))
    per = {}
    macro = {}
    for label in labels:
        label_per, label_macro = _aggregate(
            method_data[label], datasets, seeds, roles)
        for dataset, metric in label_per:
            per[(label, dataset, metric)] = label_per[(dataset, metric)]
        for role, metric in label_macro:
            macro[(label, role, metric)] = label_macro[(role, metric)]

    macro_roles = [role for role in ROLE_ORDER
                   if any(roles[d] == role for d in datasets)] + ['All']
    text = _latex(
        datasets, macro_roles, labels, roles, per, macro,
        decimals, args.with_std, not args.no_bold, args.percent)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8')

    print(f'models: {", ".join(labels)}')
    print(f'seeds:  {seeds}')
    print(f'datasets: {len(datasets)}  metrics: {", ".join(METRICS)}')
    print(f'wrote: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
