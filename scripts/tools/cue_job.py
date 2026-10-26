#!/usr/bin/env python3
"""Run one baseline train/eval job under cue-ml.

cue sets ``CUDA_VISIBLE_DEVICES`` to the physical GPU(s) assigned to the
job.  This wrapper rebuilds the normal baseline command with ``cuda:0`` (the
only visible device) and reuses the project's marker/resume logic.  It never
imports or modifies the vendored cue source under ``3rd/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--stage', choices=('train', 'eval'), required=True)
    parser.add_argument('--protocol', default='seeded5')
    parser.add_argument(
        '--no-resume', action='store_true',
        help='do not skip a job whose <stage>_done marker matches')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='print the command without running it')
    args = parser.parse_args(argv)

    os.chdir(REPO_ROOT)
    src = str(REPO_ROOT / 'src')
    existing_path = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = (
        src + (os.pathsep + existing_path if existing_path else ''))
    # Import after chdir so all relative project paths are resolved from
    # the repository root, exactly as the normal pipeline does.
    from baseline_zoo.__main__ import _build_jobs, _run_job

    namespace = argparse.Namespace(
        model=args.model,
        protocol=args.protocol,
        seeds=[str(args.seed)],
        gpus='0',
    )
    jobs, _, _, _ = _build_jobs(namespace, args.stage)
    job = next((
        item for item in jobs
        if item['model'].id == args.model and item['seed'] == args.seed
    ), None)
    if job is None:
        print(
            f'cue_job: no job for model={args.model} seed={args.seed} '
            f'stage={args.stage}',
            file=sys.stderr,
        )
        return 2

    # cue already injected CUDA_VISIBLE_DEVICES.  The command was built with
    # --device cuda:0, which maps to the assigned physical GPU.
    ok = _run_job(
        job, args.stage,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
