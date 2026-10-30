#!/usr/bin/env python3
"""Generate cue-ml YAML for one baseline stage (train or eval).

Each run invokes ``scripts/tools/cue_job.py``.  Two queue orders are
supported:

* ``--order model`` (default): one experiment per model, longest model first.
  This minimizes the idle tail.
* ``--order seed``: one experiment per seed, longest model first inside each
  seed.  cue's FIFO queue therefore dispatches every model for seed 0 before
  seed 1, then seed 2, and so on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.protocols import load_protocol  # noqa: E402
from baseline_zoo.registry import load_models  # noqa: E402

# Keep this list explicit: --model all would also include SACM-X.
MODELS = (
    'unet', 'cs2net', 'bcunet', 'maskvsc', 'nnwnet', 'jtfn', 'dscnet',
    'segdino', 'samed', 'topolora', 'sam_adapter', 'sam_octa', 'cwsam',
    'sacm_official', 'persam', 'knowsam', 'sam2_unet',
    'sam_med2d', 'ucs', 'matcher', 'sam_vit_l_unet',
)

# Iteration-based models cannot be ordered by their YAML epoch field.
COST_OVERRIDES = {
    'knowsam': 50_000,
    'nnwnet': 2_000,
}


def _parse_models(value: str) -> list[str]:
    if value.strip() == 'all':
        return list(MODELS)
    requested = [item.strip() for item in value.split(',') if item.strip()]
    unknown = sorted(set(requested) - set(MODELS))
    if unknown:
        raise SystemExit(f'unknown model(s): {", ".join(unknown)}')
    return list(dict.fromkeys(requested))


def _cost(model_id: str, models) -> int:
    if model_id in COST_OVERRIDES:
        return COST_OVERRIDES[model_id]
    epochs = int(models[model_id].defaults.get('epochs', 0) or 0)
    return max(epochs, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('train', 'eval'), required=True)
    parser.add_argument('--models', default='all',
                        help='comma-separated model ids or all')
    parser.add_argument('--seeds', nargs='+', default=['0'])
    parser.add_argument('--protocol', default='seeded5')
    parser.add_argument(
        '--order', choices=('model', 'seed'), default='model',
        help='queue order: model-major (longest first) or seed-major '
             '(all models for seed 0, then seed 1)')
    parser.add_argument(
        '--no-resume', action='store_true',
        help='make every generated job ignore matching done markers')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--log-dir', type=Path,
                        default=REPO_ROOT / 'results' / 'cue' / 'logs')
    parser.add_argument('--python-cmd', default=sys.executable)
    parser.add_argument(
        '--script', type=Path,
        default=REPO_ROOT / 'scripts' / 'tools' / 'cue_job.py')

    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.seeds == ['all']:
        seeds = list(protocol.seeds)
    else:
        try:
            seeds = [int(seed) for seed in args.seeds]
        except ValueError as exc:
            raise SystemExit(f'--seeds must be integers or all: {exc}') from exc
        seeds = protocol.select_seeds(seeds)

    loaded = load_models()
    model_ids = _parse_models(args.models)
    for model_id in model_ids:
        if model_id not in loaded:
            raise SystemExit(f'model config not found: {model_id}')
    model_ids.sort(key=lambda model_id: (-_cost(model_id, loaded), model_id))

    script_path = Path(args.script).resolve()
    if not script_path.is_file():
        raise SystemExit(f'cue wrapper not found: {script_path}')

    suffix = ' --no-resume' if args.no_resume else ''

    def run_args(model_id: str, seed: int) -> dict:
        return {'args': (f'--stage {args.stage} --model {model_id} '
                         f'--seed {seed} --protocol {args.protocol}{suffix}')}

    experiments = []
    if args.order == 'seed':
        # cue uses a FIFO job queue: queue every model for seed 0 first.
        for seed in seeds:
            experiments.append({
                'name': f'{args.stage}_seed_{seed}',
                'script': str(script_path),
                'runs': [run_args(model_id, seed) for model_id in model_ids],
            })
    else:
        for model_id in model_ids:
            experiments.append({
                'name': f'{args.stage}_{model_id}',
                'script': str(script_path),
                'runs': [run_args(model_id, seed) for seed in seeds],
            })

    payload = {
        'config': {
            'log_dir': str(Path(args.log_dir).resolve()),
            'python_cmd': str(args.python_cmd),
        },
        'experiments': experiments,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, width=1000)
    print(f'cue config -> {out} ({len(model_ids)} models, '
          f'{len(seeds)} seed(s), order={args.order}, stage={args.stage})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
