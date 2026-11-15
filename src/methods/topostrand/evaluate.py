"""Predict and evaluate a TopoStrand-SAM fixed3 checkpoint on 20 datasets.

The shared ``run_baseline_test.py`` intentionally knows only registered
baseline adapters.  TopoStrand-SAM is a new local model, so this wrapper
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

REPO_ROOT = Path(__file__).resolve().parents[3]
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
                       'topostrand')
    predict_cmd = [sys.executable, '-m', 'methods.topostrand.run_one',
                   '--stage', 'predict',
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
                   'collect_paper_metrics.py',
                   '--methods', 'topostrand',
                   '--pred-root', args.workdir / f'predictions{args.tag}',
                   '--out-root', eval_root,
                   '--metrics', 'dice,iou,cldice,hd95',
                   '--empty-metric-policy', 'penalty',
                   '--protocol', args.protocol,
                   '--layout', 'canonical',
                   '--family', 'baseline',
                   '--variant', 'topostrand',
                   '--seed', str(args.seed)]
    if args.datasets:
        metrics_cmd += ['--datasets', *args.datasets]
    run(metrics_cmd)
    print(f'eval tables -> {eval_root / "tables" / "all_metrics.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
