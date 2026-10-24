#!/usr/bin/env python3
"""Print ``NAME:TEST_ROOT`` specs from a protocol manifest.

The shell launchers used to duplicate this parser and hard-code
``data/sacm_3shot/protocol_manifest.csv``.  Keeping one parser makes
seed-specific roots explicit: callers pass the manifest belonging to the
same split root as training.

The root is the parent of the mask directory (the directory that contains
``images/`` and ``masks/``), which is what ``run_experiments.py
--test_datasets`` expects.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def dataset_specs(protocol: Path, datasets: set[str] | None = None,
                  repo: Path | None = None) -> list[str]:
    repo = Path.cwd() if repo is None else repo
    roots: dict[str, Path] = {}
    with open(protocol, newline='') as f:
        for row in csv.DictReader(f):
            if not row['split'].endswith('_test'):
                continue
            name = row['dataset']
            if datasets and name not in datasets:
                continue
            mask = Path(row['mask_path'])
            if not mask.is_absolute():
                mask = repo / mask
            roots[name] = mask.parent.parent
    if not roots:
        raise SystemExit('no test datasets selected')
    return [f'{name}:{roots[name]}' for name in sorted(roots)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True,
                        help='protocol_manifest.csv for the active split')
    parser.add_argument('--datasets', default='',
                        help='space-separated dataset names; empty means all')
    parser.add_argument('--repo', type=Path, default=Path.cwd(),
                        help='base for relative mask paths (default: cwd)')
    args = parser.parse_args()

    if not args.protocol.is_file():
        raise SystemExit(f'missing protocol manifest: {args.protocol}')
    selected = set(args.datasets.split()) if args.datasets.strip() else None
    print(' '.join(dataset_specs(args.protocol, selected, args.repo)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
