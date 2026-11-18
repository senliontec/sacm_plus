"""Train BCTD-SAM and derive its fixed3 checkpoint.

All arguments after the module name are forwarded to
``baseline_zoo.run_one --model bctd --stage train``.  Running both steps
through ``sys.executable`` keeps the baseline independent of the caller's
``python`` PATH.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    known, _ = parser.parse_known_args(argv)

    train_cmd = [
        sys.executable, '-m', 'baseline_zoo.run_one',
        '--model', 'bctd', '--stage', 'train', *argv,
    ]
    subprocess.run(train_cmd, check=True)

    workdir = known.workdir
    derive_cmd = [
        sys.executable, '-m', 'baseline_zoo.models.bctd.derive_fixed3',
        '--input', str(workdir / 'best_model.pth'),
        '--output', str(workdir / 'best_model_fixed3.pth'),
        '--deployment-head', '3',
    ]
    subprocess.run(derive_cmd, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
