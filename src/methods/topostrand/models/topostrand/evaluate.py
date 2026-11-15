"""Predict and evaluate a CurviMix-SAM fixed3 checkpoint on 20 datasets.

The shared ``run_baseline_test.py`` intentionally knows only registered
baseline adapters.  CurviMix-SAM is a new local model, so this wrapper
performs its prediction step with the local runner first, then asks the
canonical metric collector to evaluate the existing prediction tree with
``--skip-predict``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from baseline_zoo.runtime import REPO_ROOT
PIPELINE = REPO_ROOT / 'scripts' / 'pipeline'
sys.path.insert(0, str(PIPELINE))

from protocol_specs import dataset_specs  # noqa: E402


def run(cmd):
    print('+', ' '.join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], cwd=REPO_ROOT,
                   check=True, env=os.environ.copy())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--use-wandb', type=int, choices=[0, 1], default=0)
    parser.add_argument('--tta-flip', action='store_true')
    parser.add_argument('--tta-select', action='store_true',
                        help='label-free cross-view candidate selector')
    parser.add_argument('--tag', default='',
                        help='suffix for prediction/eval roots, e.g. _tta')
    parser.add_argument('--datasets', nargs='+', default=None)
    args = parser.parse_args(argv)

    selected = set(args.datasets) if args.datasets else None
    specs = dataset_specs(args.protocol, selected, REPO_ROOT)
    prediction_root = (args.workdir / f'predictions{args.tag}' /
                       'curvimix')
    predict_cmd = [sys.executable, '-m', 'baseline_zoo.run_one',
                   '--model', 'curvimix', '--stage', 'predict',
                   '--checkpoint', args.checkpoint,
                   '--workdir', args.workdir,
                   '--pred-root', prediction_root,
                   '--data-root', args.data_root,
                   '--seed', args.seed,
                   '--device', args.device,
                   '--test-datasets', *specs]
    if args.tta_flip:
        predict_cmd.append('--tta-flip')
    if args.tta_select:
        predict_cmd.append('--tta-select')
    run(predict_cmd)

    eval_root = args.workdir / f'eval{args.tag}'
    metrics_cmd = [sys.executable, REPO_ROOT / 'scripts' / 'pipeline' /
                   'run_baseline_test.py',
                   '--method', 'curvimix',
                   '--skip-predict',
                   '--checkpoint', args.checkpoint,
                   '--pred-root', args.workdir / f'predictions{args.tag}',
                   '--paper-root', eval_root,
                   '--protocol', args.protocol,
                   '--data-root', args.data_root,
                   '--seed', args.seed,
                   '--variant', 'curvimix',
                   '--device', args.device,
                   '--use_wandb', str(args.use_wandb),
                   '--skip-visualize']
    if args.datasets:
        metrics_cmd += ['--datasets', *args.datasets]
    run(metrics_cmd)
    print(f'eval tables -> {eval_root / "tables" / "all_metrics.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
