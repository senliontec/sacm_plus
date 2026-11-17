"""Predict and evaluate a StrandCal-SAM fixed3 checkpoint.

The shared ``run_baseline_test.py`` knows registered baseline adapters, but
StrandCal-SAM needs its model-specific inference switches (flip TTA, RCD,
closing and ATC).  This wrapper runs the baseline prediction step first, then
asks the canonical metric collector to score the prediction tree.
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
    parser.add_argument('--tta-flip', action='store_true', default=True)
    parser.add_argument('--no-tta-flip', dest='tta_flip',
                        action='store_false')
    parser.add_argument('--rcd', type=int, choices=[0, 1], default=1)
    parser.add_argument('--atc', type=int, choices=[0, 1], default=1)
    parser.add_argument('--tile', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=896)
    parser.add_argument('--routing-skel-ratio', type=float, default=1.0)
    parser.add_argument('--routing-area-ratio', type=float, default=0.7)
    parser.add_argument('--tag', default='')
    parser.add_argument('--datasets', nargs='+', default=None)
    args = parser.parse_args(argv)

    selected = set(args.datasets) if args.datasets else None
    specs = dataset_specs(args.protocol, selected, REPO_ROOT)
    prediction_root = (args.workdir / f'predictions{args.tag}' /
                       'strandcal')
    predict_cmd = [
        sys.executable, '-m', 'baseline_zoo.run_one',
        '--model', 'strandcal', '--stage', 'predict',
        '--checkpoint', args.checkpoint,
        '--workdir', args.workdir,
        '--pred-root', prediction_root,
        '--data-root', args.data_root,
        '--seed', args.seed,
        '--device', args.device,
        '--test-datasets', *specs,
        '--rcd', str(args.rcd),
        '--atc', str(args.atc),
        '--tile', str(args.tile),
        '--stride', str(args.stride),
        '--routing-skel-ratio', str(args.routing_skel_ratio),
        '--routing-area-ratio', str(args.routing_area_ratio),
    ]
    if args.tta_flip:
        predict_cmd.append('--tta-flip')
    run(predict_cmd)

    eval_root = args.workdir / f'eval{args.tag}'
    metrics_cmd = [
        sys.executable, REPO_ROOT / 'scripts' / 'pipeline' /
        'run_baseline_test.py',
        '--method', 'strandcal',
        '--skip-predict',
        '--checkpoint', args.checkpoint,
        '--pred-root', args.workdir / f'predictions{args.tag}',
        '--paper-root', eval_root,
        '--protocol', args.protocol,
        '--data-root', args.data_root,
        '--seed', args.seed,
        '--variant', 'strandcal',
        '--device', args.device,
        '--use_wandb', str(args.use_wandb),
        '--skip-visualize',
    ]
    if args.datasets:
        metrics_cmd += ['--datasets', *args.datasets]
    run(metrics_cmd)
    print(f'eval tables -> {eval_root / "tables" / "all_metrics.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
