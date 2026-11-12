#!/usr/bin/env python3
"""One-click runner for the lean fixed3 ablation.

The script trains the three new arms in
``configs/ablations/sacm_stc_fixed3_lean.yaml``, derives fixed H3 checkpoints
for the two SSH arms, evaluates every row with the paper metric pipeline, and
writes one compact ablation table.

The five paper rows are:

  fixed3              existing 4-seed eval tables (read-only)
  gating + fusion     existing SSH checkpoint, eval only
  nofusion            train, derive H3, eval
  native_control      train, eval the checkpoint as-is (gating, no fusion)
  notopo               train, derive H3, eval

The official 4-seed table is included as a reference row.  Per-role and
per-dataset analysis is deliberately out of scope for this lean sheet; the
full report remains ``results/reports/final_fixed3_report.html``.

Usage:
    bash scripts/run_sacm_stc_fixed3_lean.sh
    bash scripts/run_sacm_stc_fixed3_lean.sh --dry-run
    SKIP_TRAIN=1 bash scripts/run_sacm_stc_fixed3_lean.sh
    EVAL_GPUS=0,1,2,3 bash scripts/run_sacm_stc_fixed3_lean.sh

Environment overrides:
    SEEDS="0 1 2 3"       EPOCHS=800
    TRAIN_GPUS=0,1,2,3,4,5,6,7
    EVAL_GPUS=0,1,2,3,4,5,6,7
    CONFIG=configs/ablations/sacm_stc_fixed3_lean.yaml
    LEAN_ROOT=results/runs/sacm_stc_fixed3_lean
    SSH_ROOT=results/runs/sacm_stc_ssh_full_ab/ssh
    FIXED3_ROOT=results/runs/sacm_stc_fixed3_800/eval
    OFFICIAL_ROOT=results/runs/sacm_official/default
    SUMMARY_ROOT=results/ablations/sacm_stc_fixed3_lean
    SKIP_TRAIN=0 SKIP_EVAL=0 DRY_RUN=0 RESUME=1
    FORCE_DERIVE=0 FORCE_EVAL=0
    EXPECTED_DATASETS=20
    MAX_USED_MIB=2000 ALLOW_BUSY=0
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import queue
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'ablations' / 'sacm_stc_fixed3_lean.yaml'
METRICS = [
    ('dice', 'Dice', True),
    ('iou', 'IoU', True),
    ('cldice', 'clDice', True),
    ('hd95', 'HD95', False),
]
ARMS = ('nofusion', 'native_control', 'notopo')


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _parse_values(raw, default):
    values = raw if raw is not None else default
    if isinstance(values, str):
        values = [values]
    out = []
    for item in values:
        for token in str(item).replace(',', ' ').split():
            out.append(token)
    return out


def _parse_seeds(raw, default):
    tokens = _parse_values(raw, default)
    try:
        seeds = [int(token) for token in tokens]
    except ValueError as exc:
        raise SystemExit(f'invalid seed list: {tokens}') from exc
    if not seeds:
        raise SystemExit('seed list is empty')
    return list(dict.fromkeys(seeds))


def _parse_gpus(raw: str) -> list[str]:
    gpus = [item.strip() for item in str(raw).split(',') if item.strip()]
    if not gpus:
        raise SystemExit('GPU list is empty')
    if len(set(gpus)) != len(gpus):
        raise SystemExit(f'duplicate GPU ids: {gpus}')
    return gpus


def _resolve(path_text) -> Path:
    path = Path(str(path_text))
    return path if path.is_absolute() else REPO_ROOT / path


def _eval_complete(eval_root: Path, expected: int) -> bool:
    table = eval_root / 'tables' / 'all_metrics.csv'
    if not table.is_file():
        return False
    try:
        with table.open(newline='') as handle:
            rows = [
                row for row in csv.DictReader(handle)
                if row.get('dataset')
            ]
    except (OSError, csv.Error):
        return False
    if len({row['dataset'] for row in rows}) < expected:
        return False
    for row in rows:
        for key, _, _ in METRICS:
            raw = row.get(f'{key}_mean')
            if raw in (None, ''):
                return False
            try:
                float(raw)
            except ValueError:
                return False
    return True


def _read_rows(path: Path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def _run(cmd: list[str], log_path: Path | None = None,
         dry_run: bool = False) -> None:
    printable = ' '.join(str(item) for item in cmd)
    print(f'+ {printable}', flush=True)
    if dry_run:
        return
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('w') as handle:
            proc = subprocess.run(
                cmd, cwd=REPO_ROOT, stdout=handle,
                stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(
            f'command failed ({proc.returncode}): {printable}'
            + (f'; see {log_path}' if log_path else ''))


def _check_gpus(gpus: list[str], max_used_mib: int,
                allow_busy: bool, dry_run: bool) -> None:
    if dry_run:
        return
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used',
             '--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f'[warn] nvidia-smi unavailable; GPU-busy check skipped: {exc}',
              file=sys.stderr)
        return
    available = {}
    for line in out.strip().splitlines():
        parts = [item.strip() for item in line.split(',')]
        if len(parts) < 2:
            continue
        try:
            available[parts[0]] = int(float(parts[1]))
        except ValueError:
            available[parts[0]] = 0
    missing = [gpu for gpu in gpus if gpu not in available]
    if missing:
        raise SystemExit(f'GPU(s) not visible to nvidia-smi: {missing}')
    busy = [(gpu, available[gpu]) for gpu in gpus
            if available[gpu] > max_used_mib]
    if busy and not allow_busy:
        detail = ', '.join(f'{gpu}={mib}MiB' for gpu, mib in busy)
        raise SystemExit(
            f'selected GPU(s) look busy: {detail}; free them or set '
            'ALLOW_BUSY=1 / MAX_USED_MIB=...')


@dataclass(frozen=True)
class EvalJob:
    arm: str
    seed: int
    variant: str
    checkpoint: Path
    eval_root: Path
    pred_root: Path


def _eval_command(job: EvalJob, gpu: str, seed: int,
                  wandb_project: str, wandb_entity: str,
                  no_resume: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / 'scripts' / 'pipeline' / 'run_baseline_test.py'),
        '--method', 'sacm_stc',
        '--protocol', str(
            REPO_ROOT / 'data' / f'sacm_3shot_seed{seed}' /
            'protocol_manifest.csv'),
        '--seed', str(seed),
        '--variant', job.variant,
        '--checkpoint', str(job.checkpoint),
        '--pred-root', str(job.pred_root),
        '--paper-root', str(job.eval_root),
        '--device', f'cuda:{gpu}',
        '--use_wandb', 'true',
        '--wandb_project', wandb_project,
        '--wandb_entity', wandb_entity,
        '--wandb_group', 'sacm_stc_fixed3_lean_eval',
        '--wandb_name', f'paper_metrics-sacm_stc-{job.arm}-seed{seed}',
        '--wandb_tags', 'sacm_stc,ablation,fixed3,lean,eval',
        '--metrics', 'dice,iou,cldice,hd95',
        '--skip-visualize',
    ]
    if no_resume:
        cmd.append('--no-resume')
    return cmd


def _read_macro(eval_root: Path):
    table = eval_root / 'tables' / 'all_metrics.csv'
    if not table.is_file():
        return None
    rows = _read_rows(table)
    rows = [row for row in rows if row.get('dataset')]
    if not rows:
        return None
    out = {}
    for key, _, _ in METRICS:
        values = []
        for row in rows:
            raw = row.get(f'{key}_mean')
            if raw in (None, ''):
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
        if len(values) != len(rows):
            return None
        out[key] = statistics.fmean(values)
    return out


def _mean_std(values: list[float]):
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _display(metric: str, value: float | None) -> str:
    if value is None:
        return '—'
    if metric == 'hd95':
        return f'{value:.2f}'
    return f'{value * 100.0:.2f}'


def _delta(metric: str, value: float | None, ref: float | None) -> str:
    if value is None or ref is None:
        return '—'
    diff = value - ref
    if metric != 'hd95':
        diff *= 100.0
    return f'{diff:+.2f}'


def _summary_rows(seed_roots: dict[str, dict[int, Path]], seeds: list[int]):
    rows = []
    fixed_by_seed = {}
    for seed in seeds:
        fixed_by_seed[seed] = _read_macro(seed_roots['fixed3'][seed])
    fixed_values = {
        key: _mean_std([
            values[key] for values in fixed_by_seed.values()
            if values and key in values
        ])[0]
        for key, _, _ in METRICS
    }
    for arm in ('fixed3', 'gating_fusion', 'nofusion', 'native_control',
                'notopo', 'official'):
        per_seed = {
            seed: _read_macro(seed_roots[arm][seed]) for seed in seeds
        }
        present = [values for values in per_seed.values() if values]
        record = {
            'arm': arm,
            'n_seeds': len(present),
            'n_requested': len(seeds),
            'status': ('OK' if len(present) == len(seeds)
                       else ('PARTIAL' if present else 'MISSING')),
        }
        for key, _, _ in METRICS:
            vals = [values[key] for values in present if key in values]
            mean, std = _mean_std(vals)
            record[f'{key}_mean'] = mean
            record[f'{key}_std'] = std
            record[f'{key}_delta'] = (
                None if mean is None or fixed_values[key] is None
                else mean - fixed_values[key])
        rows.append(record)
    return rows, fixed_values


def _write_summary(rows, fixed_values, output_root: Path,
                   seeds: list[int]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fields = ['arm', 'n_seeds', 'n_requested', 'status']
    for key, _, _ in METRICS:
        fields += [f'{key}_mean', f'{key}_std', f'{key}_delta']
    with (output_root / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    labels = {
        'fixed3': 'fixed3',
        'gating_fusion': 'gating + fusion',
        'nofusion': 'w/o fusion training',
        'native_control': 'native four-head control',
        'notopo': 'w/o topology loss',
        'official': 'SACM official (reference)',
    }
    lines = [
        '# fixed3 Lean Ablation — 4-seed macro summary',
        '',
        f'- Seeds: `{seeds}`',
        '- Scope: 20 test datasets, 800 epochs, All-role macro.',
        '- fixed3 and official are existing eval tables; new arms are '
        'trained by this run.',
        '',
        '| Method | Dice | IoU | clDice | HD95 | seeds | status |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        cells = []
        for key, _, _ in METRICS:
            mean = row[f'{key}_mean']
            std = row[f'{key}_std']
            if mean is None:
                cells.append('—')
            else:
                cells.append(
                    f'{_display(key, mean)} ± {_display(key, std)}')
        lines.append(
            f'| {labels.get(row["arm"], row["arm"])} | '
            + ' | '.join(cells)
            + f' | {row["n_seeds"]}/{row["n_requested"]} | '
            f'{row["status"]} |')
    lines += [
        '',
        '## Deltas vs fixed3',
        '',
        '| Method | ΔDice | ΔIoU | ΔclDice | ΔHD95 |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        if row['arm'] == 'fixed3':
            continue
        lines.append(
            f'| {labels.get(row["arm"], row["arm"])} | '
            + ' | '.join(_delta(key, row[f'{key}_mean'], fixed_values[key])
                         for key, _, _ in METRICS)
            + ' |')
    lines += [
        '',
        'HD95 is lower-better. p-values are not computed in this lean '
        'table; use the full report for paired tests.',
        '',
    ]
    (output_root / 'summary.md').write_text(
        '\n'.join(lines), encoding='utf-8')

    html_rows = []
    for row in rows:
        cells = [
            f'<td>{html.escape(labels.get(row["arm"], row["arm"]))}</td>'
        ]
        for key, _, _ in METRICS:
            mean = row[f'{key}_mean']
            std = row[f'{key}_std']
            text = ('—' if mean is None
                    else f'{_display(key, mean)} ± {_display(key, std)}')
            cells.append(f'<td>{html.escape(text)}</td>')
        cells.append(f'<td>{row["n_seeds"]}/{row["n_requested"]}</td>')
        cells.append(f'<td>{html.escape(row["status"])}</td>')
        html_rows.append('<tr>' + ''.join(cells) + '</tr>')
    html_body = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>fixed3 lean ablation</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;margin:32px;color:#111}}
h1{{font-size:24px}} table{{border-collapse:collapse;margin-top:18px}}
th,td{{border:1px solid #ccc;padding:6px 10px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} th{{background:#f2f2ef}}
.note{{color:#555;max-width:900px}}
</style></head><body>
<h1>fixed3 Lean Ablation</h1>
<p class="note">4 seeds × 20 datasets, All-role macro. HD95 lower-better.
This is the lean paper table; the full report is
<code>final_fixed3_report.html</code>.</p>
<table><thead><tr><th>Method</th><th>Dice</th><th>IoU</th>
<th>clDice</th><th>HD95</th><th>seeds</th><th>status</th></tr></thead>
<tbody>{''.join(html_rows)}</tbody></table>
</body></html>'''
    (output_root / 'summary.html').write_text(html_body, encoding='utf-8')
    (output_root / 'summary.json').write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n' + '\n'.join(lines))
    print(f'wrote {output_root}/summary.md')
    print(f'wrote {output_root}/summary.csv')
    print(f'wrote {output_root}/summary.html')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    config_default = os.environ.get('CONFIG')
    parser.add_argument(
        '--config', type=Path,
        default=Path(config_default) if config_default else DEFAULT_CONFIG)
    parser.add_argument('--seeds', nargs='*', default=None)
    parser.add_argument('--train-gpus', default=os.environ.get(
        'TRAIN_GPUS', os.environ.get('GPU_LIST', '0,1,2,3,4,5,6,7')))
    parser.add_argument('--eval-gpus', default=os.environ.get(
        'EVAL_GPUS', os.environ.get('GPU_LIST', '0,1,2,3,4,5,6,7')))
    parser.add_argument('--epochs', type=int,
                        default=int(os.environ.get('EPOCHS', '800')))
    parser.add_argument('--skip-train', action='store_true',
                        default=_as_bool(os.environ.get('SKIP_TRAIN', '0')))
    parser.add_argument('--skip-eval', action='store_true',
                        default=_as_bool(os.environ.get('SKIP_EVAL', '0')))
    parser.add_argument('--dry-run', action='store_true',
                        default=_as_bool(os.environ.get('DRY_RUN', '0')))
    parser.add_argument('--no-resume', action='store_true',
                        default=not _as_bool(os.environ.get('RESUME', '1')))
    parser.add_argument(
        '--force-derive', action='store_true',
        default=_as_bool(os.environ.get('FORCE_DERIVE', '0')))
    parser.add_argument(
        '--force-eval', action='store_true',
        default=_as_bool(os.environ.get('FORCE_EVAL', '0')))
    parser.add_argument('--force', action='store_true',
                        help='force both derivation and evaluation')
    parser.add_argument('--allow-busy', action='store_true',
                        default=_as_bool(os.environ.get('ALLOW_BUSY', '0')))
    parser.add_argument('--max-used-mib', type=int,
                        default=int(os.environ.get('MAX_USED_MIB', '2000')))
    args = parser.parse_args(argv)
    force_derive = bool(args.force or args.force_derive)
    force_eval = bool(args.force or args.force_eval)

    config_path = args.config if args.config.is_absolute() else (
        REPO_ROOT / args.config)
    if not config_path.is_file():
        raise SystemExit(f'missing config: {config_path}')
    config = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    arms = config.get('arms') or {}
    missing_arms = [arm for arm in ARMS if arm not in arms]
    if missing_arms:
        raise SystemExit(f'config missing arms: {missing_arms}')

    seed_source = args.seeds
    if seed_source is None:
        seed_source = os.environ.get('SEEDS')
    raw_seeds = _parse_values(seed_source, config.get('seeds', [0, 1, 2, 3]))
    seeds = _parse_seeds(raw_seeds, None)
    train_gpus = _parse_gpus(args.train_gpus)
    eval_gpus = _parse_gpus(args.eval_gpus)
    expected = int(os.environ.get('EXPECTED_DATASETS', '20'))
    config_lean_root = _resolve(
        config.get('output_root', 'results/runs/sacm_stc_fixed3_lean'))
    lean_override = os.environ.get('LEAN_ROOT')
    if lean_override and _resolve(lean_override) != config_lean_root:
        raise SystemExit(
            'LEAN_ROOT must match config output_root: '
            f'{_resolve(lean_override)} != {config_lean_root}')
    lean_root = config_lean_root
    ssh_root = _resolve(os.environ.get(
        'SSH_ROOT', 'results/runs/sacm_stc_ssh_full_ab/ssh'))
    fixed3_root = _resolve(os.environ.get(
        'FIXED3_ROOT', 'results/runs/sacm_stc_fixed3_800/eval'))
    official_root = _resolve(os.environ.get(
        'OFFICIAL_ROOT', 'results/runs/sacm_official/default'))
    summary_root = _resolve(os.environ.get(
        'SUMMARY_ROOT', 'results/ablations/sacm_stc_fixed3_lean'))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = _resolve(os.environ.get(
        'LOG_DIR', f'results/logs/sacm_stc_fixed3_lean_{stamp}'))
    output_root = summary_root
    wandb_project = os.environ.get(
        'WANDB_PROJECT', str((config.get('wandb') or {}).get('project',
                                                              'sacm')))
    wandb_entity = os.environ.get(
        'WANDB_ENTITY', str((config.get('wandb') or {}).get('entity',
                                                             'buaazqk')))

    print('=' * 60)
    print(' fixed3 lean ablation')
    print(f' config:      {config_path}')
    print(f' seeds:       {seeds}')
    print(f' epochs:      {args.epochs}')
    print(f' train GPUs:  {train_gpus}')
    print(f' eval GPUs:   {eval_gpus}')
    print(f' lean root:   {lean_root}')
    print(f' summary:     {output_root}')
    print(f' log dir:     {log_dir}')
    if args.dry_run:
        print(' mode:        DRY RUN')
    print('=' * 60)

    for seed in seeds:
        protocol = REPO_ROOT / 'data' / f'sacm_3shot_seed{seed}' / (
            'protocol_manifest.csv')
        if not protocol.is_file() and not args.dry_run:
            raise SystemExit(f'missing protocol: {protocol}')
        fixed_eval = fixed3_root / f'seed_{seed}'
        official_eval = official_root / f'seed_{seed}' / 'eval'
        if not args.dry_run:
            if not _eval_complete(fixed_eval, expected):
                raise SystemExit(
                    f'fixed3 eval incomplete: {fixed_eval}/tables/'
                    'all_metrics.csv')
            if not _eval_complete(official_eval, expected):
                raise SystemExit(
                    f'official eval incomplete: {official_eval}/tables/'
                    'all_metrics.csv')
            ssh_checkpoint = ssh_root / f'seed_{seed}' / 'best_model.pth'
            if not ssh_checkpoint.is_file():
                raise SystemExit(
                    f'missing existing SSH checkpoint for gating+fusion: '
                    f'{ssh_checkpoint}')

    if not args.dry_run and not args.skip_train:
        _check_gpus(train_gpus, args.max_used_mib, args.allow_busy,
                    args.dry_run)

    if not args.skip_train:
        train_cmd = [
            'bash', str(REPO_ROOT / 'scripts' /
                        'run_sacm_stc_ablation.sh'),
            '--config', str(config_path),
            '--stage', 'train',
            '--epochs', str(args.epochs),
            '--gpus', ','.join(train_gpus),
            '--seeds', *[str(seed) for seed in seeds],
        ]
        if args.no_resume:
            train_cmd.append('--no-resume')
        if args.dry_run:
            train_cmd.append('--dry-run')
        print('[1/4] training three lean arms')
        _run(train_cmd, log_dir / 'train.log', args.dry_run)
    else:
        print('[1/4] training skipped (SKIP_TRAIN=1)')

    if not args.dry_run:
        _check_gpus(eval_gpus, args.max_used_mib, args.allow_busy,
                    args.dry_run)

    derived = {}
    print('[2/4] deriving fixed H3 for nofusion / notopo')
    for arm in ('nofusion', 'notopo'):
        for seed in seeds:
            run_root = lean_root / arm / f'seed_{seed}'
            source = run_root / 'best_model.pth'
            output = run_root / 'best_model_fixed3_nofusion.pth'
            derived[(arm, seed)] = output
            cmd = [
                sys.executable,
                str(REPO_ROOT / 'scripts' / 'pipeline' /
                    'make_fixed3_checkpoint.py'),
                '--input', str(source),
                '--output', str(output),
                '--deployment-head', '3',
            ]
            if force_derive:
                cmd.append('--force')
            if not args.dry_run and not source.is_file():
                raise SystemExit(
                    f'missing trained checkpoint for {arm}/seed{seed}: '
                    f'{source}')
            _run(cmd, log_dir / 'derive' / f'{arm}_seed{seed}.log',
                 args.dry_run)

    fixed_roots = {
        'fixed3': {
            seed: fixed3_root / f'seed_{seed}' for seed in seeds
        },
        'official': {
            seed: official_root / f'seed_{seed}' / 'eval' for seed in seeds
        },
    }

    jobs = []
    for seed in seeds:
        protocol_root = lean_root / 'gating_fusion' / f'seed_{seed}'
        jobs.append(EvalJob(
            arm='gating_fusion', seed=seed, variant='gating_fusion',
            checkpoint=ssh_root / f'seed_{seed}' / 'best_model.pth',
            eval_root=protocol_root / 'eval',
            pred_root=protocol_root / 'predictions'))
        for arm, variant in (('nofusion', 'nofusion_fixed3'),
                             ('native_control', 'native_control'),
                             ('notopo', 'notopo_fixed3')):
            run_root = lean_root / arm / f'seed_{seed}'
            checkpoint = (
                derived[(arm, seed)] if arm in ('nofusion', 'notopo')
                else run_root / 'best_model.pth')
            suffix = 'eval_fixed3' if arm in ('nofusion', 'notopo') else 'eval'
            jobs.append(EvalJob(
                arm=arm, seed=seed, variant=variant,
                checkpoint=checkpoint,
                eval_root=run_root / suffix,
                pred_root=run_root / ('predictions_fixed3'
                                      if suffix == 'eval_fixed3'
                                      else 'predictions')))

    if not args.skip_eval:
        print('[3/4] evaluating four ablation rows')
        pending = []
        for job in jobs:
            if not force_eval and _eval_complete(job.eval_root, expected):
                print(f'  skip complete: {job.arm}/seed{job.seed}')
                continue
            if not args.dry_run and not job.checkpoint.is_file():
                raise SystemExit(
                    f'missing checkpoint for {job.arm}/seed{job.seed}: '
                    f'{job.checkpoint}')
            pending.append(job)
        if pending:
            gpu_queue: queue.Queue[str] = queue.Queue()
            for gpu in eval_gpus:
                gpu_queue.put(gpu)

            def evaluate(job: EvalJob):
                gpu = gpu_queue.get()
                try:
                    log = (log_dir / 'eval' / job.arm /
                           f'seed{job.seed}.log')
                    _run(_eval_command(job, gpu, job.seed, wandb_project,
                                       wandb_entity, args.no_resume),
                         log, args.dry_run)
                    return job, None
                except Exception as exc:  # noqa: BLE001
                    return job, exc
                finally:
                    gpu_queue.put(gpu)

            if args.dry_run:
                for job in pending:
                    evaluate(job)
            else:
                failures = []
                with ThreadPoolExecutor(
                        max_workers=len(eval_gpus)) as pool:
                    futures = [pool.submit(evaluate, job)
                               for job in pending]
                    for future in as_completed(futures):
                        job, exc = future.result()
                        if exc is not None:
                            failures.append((job, exc))
                            print(f'[fail] {job.arm}/seed{job.seed}: {exc}',
                                  file=sys.stderr)
                        else:
                            print(f'[done] {job.arm}/seed{job.seed}')
                if failures:
                    raise SystemExit(
                        f'{len(failures)} evaluation job(s) failed; see '
                        f'{log_dir}/eval')
        else:
            print('  all evaluation tables already complete')
    else:
        print('[3/4] evaluation skipped (SKIP_EVAL=1)')

    if args.dry_run:
        print('[4/4] dry-run: summary not written')
        return 0

    print('[4/4] writing summary')
    seed_roots = {
        **fixed_roots,
        'gating_fusion': {
            seed: lean_root / 'gating_fusion' / f'seed_{seed}' / 'eval'
            for seed in seeds
        },
        'nofusion': {
            seed: lean_root / 'nofusion' / f'seed_{seed}' / 'eval_fixed3'
            for seed in seeds
        },
        'native_control': {
            seed: lean_root / 'native_control' / f'seed_{seed}' / 'eval'
            for seed in seeds
        },
        'notopo': {
            seed: lean_root / 'notopo' / f'seed_{seed}' / 'eval_fixed3'
            for seed in seeds
        },
    }
    rows, fixed_values = _summary_rows(seed_roots, seeds)
    _write_summary(rows, fixed_values, output_root, seeds)
    print(f'driver log: {log_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
