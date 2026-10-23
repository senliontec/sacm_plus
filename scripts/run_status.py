#!/usr/bin/env python3
"""Show live progress for sacm_plus prediction/metric/visualization jobs.

The pipeline writes ``results/.run_status/<method>.json`` through
``core.run_status.RunStatus``.  This command renders those files:

    python scripts/run_status.py
    python scripts/run_status.py --method unet
    watch -n 2 python scripts/run_status.py --method unet
    python scripts/run_status.py --json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_NAMES = {
    'predict': 'predictions',
    'paper_metrics': 'paper metrics',
    'visualize': 'visualizations',
}


def duration(seconds):
    if seconds is None:
        return '?'
    seconds = max(float(seconds), 0)
    if seconds < 60:
        return f'{seconds:.0f}s'
    minutes, seconds = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f'{minutes}m{seconds:02d}s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h{minutes:02d}m'


def pct(done, total):
    if not total:
        return ''
    return f' ({100.0 * done / total:5.1f}%)'


def render(data, now=None):
    now = time.time() if now is None else now
    method = data.get('method', '?')
    status = data.get('status', 'running')
    stage = data.get('stage') or '?'
    stage_name = STAGE_NAMES.get(stage, stage)
    si = data.get('stage_index')
    st = data.get('stage_total')
    stage_pos = f' ({si}/{st})' if si and st else ''
    lines = [f'{method}: {status}  stage={stage_name}{stage_pos}']

    dataset = data.get('dataset')
    d_done = data.get('dataset_done') or 0
    d_total = data.get('dataset_total') or 0
    di = data.get('dataset_index')
    dt = data.get('datasets_total')
    if dataset:
        where = f'  dataset: {dataset}'
        if di and dt:
            where += f' ({di}/{dt})'
        if d_total:
            where += f'  {d_done}/{d_total}{pct(d_done, d_total)}'
        lines.append(where)

    active = data.get('active_datasets') or []
    if active and (not dataset or len(active) > 1):
        lines.append('  active: ' + ', '.join(active))

    overall_done = data.get('overall_done') or 0
    overall_total = data.get('overall_total') or 0
    parts = []
    if data.get('datasets_total'):
        parts.append(f"datasets {data.get('datasets_done', 0)}/"
                     f"{data['datasets_total']}")
    if overall_total:
        parts.append(f'images {overall_done}/{overall_total}'
                     f'{pct(overall_done, overall_total)}')
    rate = data.get('rate') or 0
    if rate:
        parts.append(f'{rate:.1f} img/s')
    if parts:
        lines.append('  ' + '  |  '.join(parts))

    timing = [f"elapsed {duration(data.get('elapsed_s'))}"]
    if data.get('eta_s') is not None and status == 'running':
        timing.append(f"ETA {duration(data.get('eta_s'))}")
    if data.get('stage_elapsed_s') is not None:
        timing.append(f"stage {duration(data.get('stage_elapsed_s'))}")
    lines.append('  ' + '  |  '.join(timing))

    age = max(now - float(data.get('updated_epoch') or 0), 0)
    stale = status == 'running' and age > 60
    suffix = f'  [stale {duration(age)}]' if stale else ''
    lines.append(f'  updated {duration(age)} ago{suffix}')

    if data.get('error'):
        lines.append(f"  error: {data['error']}")
    elif data.get('message'):
        lines.append(f"  message: {data['message']}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=REPO_ROOT / 'results' / '.run_status')
    parser.add_argument('--method', default=None,
                        help='only show one method')
    parser.add_argument('--json', action='store_true',
                        help='print raw JSON instead of a summary')
    args = parser.parse_args()

    paths = sorted(args.root.glob('*.json'))
    if args.method:
        paths = [p for p in paths if p.stem == args.method]
    if not paths:
        print(f'no status files under {args.root}')
        return 1

    payload = {}
    for path in paths:
        try:
            payload[path.stem] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f'{path}: unreadable ({exc})')
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    for index, (name, data) in enumerate(payload.items()):
        if index:
            print()
        print(render(data))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
