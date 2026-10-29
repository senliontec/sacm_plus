#!/usr/bin/env python3
"""Collect paper metrics for every baseline from the prediction contract.

Reads ``data/sacm_3shot/protocol_manifest.csv`` and expects
``results/predictions/<method>/<dataset>/<stem>.png`` for every test image.
Writes per-image metrics, per-dataset mean/std, role macro averages, and
LaTeX/Markdown tables.

Examples:
    # One method, four main metrics
    python scripts/pipeline/collect_paper_metrics.py --methods unet

    # All methods found under results/predictions
    python scripts/pipeline/collect_paper_metrics.py --metrics dice,iou,cldice,hd95

    # Fast smoke test
    python scripts/pipeline/collect_paper_metrics.py --methods unet --datasets DRIVE
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'scripts' / 'pipeline'))

from metrics import compute_selected  # noqa: E402

from core.run_status import RunStatus  # noqa: E402

MAIN_METRICS = ['dice', 'iou', 'cldice', 'hd95']
CACHE_VERSION = 2
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
METHOD_META = {
    'a2sam': ('A2SAM (MIA 2026)', '✓'),
    'aco_moe_lora': ('ACO-MoE-LoRA', '✓'),
    'unet': ('U-Net', '✗'),
    'cs2net': ('CS2Net-2D', '✗'),
    'csunet': ('CS-UNet', '✗'),
    'bcunet': ('BCUNet', '✗'),
    'blo_sam': ('BLO-SAM', '✓'),
    'c2p': ('C2P', '✗'),
    'convlora': ('Conv-LoRA', '✓'),
    'dconnnet': ('DconnNet', '✗'),
    'dka': ('DKA', '✗'),
    'dscnet': ('DSCNet', '✗'),
    'flowsdf': ('FlowSDF', '✗'),
    'fssam': ('FSSAM', '✓'),
    'maskvsc': ('MaskVSC', '✗'),
    'nnunet2d': ('nnU-Net 2D', '✗'),
    'nnwnet': ('nnWNet', '✗'),
    'jtfn': ('JTFN', '✗'),
    'liot': ('LIOT-IterNet', '✗'),
    'medcagd': ('MedCAGD', '✗'),
    'persam': ('PerSAM', '✓'),
    'rsrm': ('RSRM', '✓'),
    's2mnet': ('S2M-Net', '✗'),
    'matcher': ('Matcher', '✓'),
    'sam_med2d': ('SAM-Med2D', '✓'),
    'sam_octa': ('SAM-OCTA', '✓'),
    'sansa': ('SANSA', '✓'),
    'scrwkv': ('SCRWKV', '✗'),
    'cwsam': ('CWSAM', '✓'),
    'knowsam': ('KnowSAM', '✓'),
    'kprism': ('K-Prism (ICLR 2026)', '✗'),
    'segdino': ('SegDINO', '✓'),
    'segdino_v2': ('SegDINO v2', '✓'),
    'sacm_official': ('SACM (official)', '✗'),
    'sam_vit_l_unet': ('SAM ViT-L + U-Net', '✗'),
    'sam2_unet': ('SAM2-UNet', '✗'),
    'sam_adapter': ('SAM-Adapter', '✓'),
    'samed': ('SAMed', '✓'),
    'topolora': ('TopoLoRA-SAM', '✓'),
    'topoweave': ('TopoWeave', '✗'),
    'tpsegformer': ('TPSegformer', '✗'),
    'ucs': ('UCS', '✓'),
    'sacm_x': ('SACM-X', '✗'),
}
ROLE_ORDER = ['Seen', 'Unseen', 'Novel']


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def read_rows(protocol_path: Path, dataset_filter):
    rows = []
    with open(protocol_path) as f:
        for row in csv.DictReader(f):
            if not row['split'].endswith('_test'):
                continue
            if dataset_filter and row['dataset'] not in dataset_filter:
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f'no test rows found in {protocol_path}')
    return rows


def discover_methods(pred_root: Path):
    methods = []
    for child in sorted(pred_root.iterdir()):
        if child.is_dir() and any(p.is_dir() for p in child.iterdir()):
            methods.append(child.name)
    return methods


def group_rows(rows, max_images=None):
    grouped = defaultdict(list)
    for row in rows:
        dataset = row['dataset']
        if max_images is not None and len(grouped[dataset]) >= max_images:
            continue
        grouped[dataset].append(row)
    return grouped


def missing_predictions(method, dataset, rows_ds, pred_root):
    missing = []
    for row in rows_ds:
        stem = Path(row['mask_path']).stem
        pred_path = pred_root / method / dataset / f'{stem}.png'
        if not pred_path.is_file():
            missing.append(str(pred_path))
    return missing


def pred_signature(method, dataset, rows_ds, pred_root):
    """Cheap fingerprint of prediction *and* GT files for one dataset."""
    digest = hashlib.sha256()
    for row in rows_ds:
        stem = Path(row['mask_path']).stem
        pred_path = pred_root / method / dataset / f'{stem}.png'
        gt_path = resolve(row['mask_path'])
        tokens = [stem]
        for path in (pred_path, gt_path):
            try:
                stat = path.stat()
                tokens.append(f'{stat.st_size}|{stat.st_mtime_ns}')
            except OSError:
                tokens.append('MISSING')
        digest.update('|'.join(tokens).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def cache_paths(out_root, method, dataset):
    base = out_root / method / dataset
    return base / 'metrics.csv', base / 'metrics_meta.json'


def load_cached(metrics_path, meta_path, metric_names, n_expected,
                pred_root, signature, max_images):
    if not (metrics_path.is_file() and meta_path.is_file()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (meta.get('version') != CACHE_VERSION
            or meta.get('metric_names') != list(metric_names)
            or meta.get('n_expected') != n_expected
            or meta.get('pred_root') != str(pred_root)
            or meta.get('pred_signature') != signature
            or meta.get('max_images') != max_images):
        return None
    try:
        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            rows = []
            for raw in reader:
                item = {'image': raw.get('image', ''),
                        'role': raw.get('role', '')}
                for metric in metric_names:
                    value = raw.get(metric, '')
                    item[metric] = (float('nan') if value in ('', 'nan', 'NaN')
                                    else float(value))
                rows.append(item)
    except (OSError, ValueError, KeyError):
        return None
    if len(rows) != meta.get('n_images'):
        return None
    return rows


def save_meta(out_root, method, dataset, meta):
    _, meta_path = cache_paths(out_root, method, dataset)
    tmp = meta_path.with_name(meta_path.name + '.tmp')
    try:
        with open(tmp, 'w') as f:
            json.dump(meta, f, indent=2)
            f.write('\n')
        os.replace(tmp, meta_path)
    except OSError:
        pass


def _load_binary(path):
    """Accept both 0/255 PNGs and literal 0/1 masks/predictions."""
    values = np.array(Image.open(path).convert('L'))
    if values.size and values.max() <= 1:
        return values.astype(np.float32)
    return values.astype(np.float32) / 255.0


def collect_chunk(method, dataset, chunk_index, rows_chunk, pred_root,
                  metric_names):
    """Worker: compute a contiguous image chunk (process-pool safe)."""
    per_rows = []
    missing = []
    for row in rows_chunk:
        stem = Path(row['mask_path']).stem
        pred_path = pred_root / method / dataset / f'{stem}.png'
        if not pred_path.is_file():
            missing.append(str(pred_path))
            continue
        gt_path = resolve(row['mask_path'])
        pred = _load_binary(pred_path)
        gt = _load_binary(gt_path)
        if pred.shape != gt.shape:
            raise SystemExit(
                f'{pred_path}: prediction shape {pred.shape} != GT {gt.shape}')
        metrics = compute_selected(pred, gt, metric_names)
        per_rows.append({
            'image': stem,
            'role': row['role'],
            **metrics,
        })
    return dataset, chunk_index, per_rows, missing


def collect_method(method, rows, pred_root, out_root, metric_names,
                   max_images=None, workers=DEFAULT_WORKERS, resume=True,
                   status=None, chunk_size=100):
    cache_enabled = metric_names != 'all'
    grouped = group_rows(rows, max_images)
    order = list(grouped)
    total_expected = sum(len(v) for v in grouped.values())
    workers = max(1, int(workers))
    chunk_size = max(1, int(chunk_size))
    print(f'{method}: collecting {total_expected} images across '
          f'{len(order)} datasets (workers={workers}, '
          f'chunk={chunk_size}, resume={resume})', flush=True)

    t0 = time.time()
    per_dataset = {}
    missing = []
    to_run = []
    completed = 0
    reused_rows = 0
    for dataset in order:
        rows_ds = grouped[dataset]
        signature = pred_signature(method, dataset, rows_ds, pred_root)
        metrics_path, meta_path = cache_paths(out_root, method, dataset)
        cached = None
        if resume and cache_enabled:
            cached = load_cached(
                metrics_path, meta_path, metric_names, len(rows_ds),
                pred_root, signature, max_images)
        if cached is not None:
            per_dataset[dataset] = cached
            missing.extend(missing_predictions(
                method, dataset, rows_ds, pred_root))
            completed += 1
            reused_rows += len(rows_ds)
            print(f'  {method}/{dataset}: reused {len(cached)} cached rows',
                  flush=True)
        else:
            to_run.append((dataset, rows_ds, signature))

    if status is not None:
        status.update(
            datasets_total=len(order), overall_total=total_expected,
            datasets_done=completed, overall_done=reused_rows,
            active_datasets=[], dataset=None, dataset_done=0,
            dataset_total=0, rate=0.0, eta_s=None,
            message=f'{method}: metric collection')

    if not to_run:
        return {ds: per_dataset[ds] for ds in order if ds in per_dataset}, missing

    # Submit fixed-size chunks rather than whole datasets: the parent gets
    # live progress every chunk, and small/large datasets share the pool
    # evenly (CHCrack5k no longer occupies one worker for a whole minute).
    dataset_rows = {}
    dataset_signatures = {}
    dataset_chunks_total = {}
    chunk_specs = []
    for dataset, rows_ds, signature in to_run:
        dataset_rows[dataset] = rows_ds
        dataset_signatures[dataset] = signature
        chunks = [rows_ds[i:i + chunk_size]
                  for i in range(0, len(rows_ds), chunk_size)]
        dataset_chunks_total[dataset] = len(chunks)
        for chunk_index, chunk in enumerate(chunks):
            chunk_specs.append((dataset, chunk_index, chunk))
    workers = max(1, min(workers, len(chunk_specs) or 1))

    active = set(dataset_chunks_total)
    processed = reused_rows
    dataset_done_images = defaultdict(int)
    dataset_rows_by_chunk = defaultdict(dict)
    dataset_missing_map = defaultdict(list)
    remaining = dict(dataset_chunks_total)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(collect_chunk, method, dataset, chunk_index, chunk,
                        pred_root, metric_names):
                (dataset, chunk_index, len(chunk))
            for dataset, chunk_index, chunk in chunk_specs
        }
        for future in as_completed(futures):
            dataset, chunk_index, attempted = futures[future]
            _, _, per_rows, chunk_missing = future.result()
            dataset_rows_by_chunk[dataset][chunk_index] = per_rows
            dataset_missing_map[dataset].extend(chunk_missing)
            dataset_done_images[dataset] += attempted
            processed += attempted
            remaining[dataset] -= 1

            total_ds = len(dataset_rows[dataset])
            done_ds = dataset_done_images[dataset]
            if done_ds % 100 == 0 or remaining[dataset] == 0:
                print(f'    {dataset}: {done_ds}/{total_ds}', flush=True)

            if remaining[dataset] == 0:
                per_rows_all = [
                    row for index in sorted(dataset_rows_by_chunk[dataset])
                    for row in dataset_rows_by_chunk[dataset][index]
                ]
                per_dataset[dataset] = per_rows_all
                missing.extend(dataset_missing_map[dataset])
                completed += 1
                active.discard(dataset)
                metrics_path, _ = cache_paths(out_root, method, dataset)
                fields = ['image', 'role']
                if per_rows_all:
                    fields += [k for k in per_rows_all[0]
                               if k not in ('image', 'role')]
                write_csv(metrics_path, fields, per_rows_all)
                if cache_enabled:
                    meta = {
                        'version': CACHE_VERSION,
                        'metric_names': list(metric_names),
                        'n_expected': len(dataset_rows[dataset]),
                        'n_images': len(per_rows_all),
                        'pred_root': str(pred_root),
                        'pred_signature': dataset_signatures[dataset],
                        'max_images': max_images,
                        'generated_at_utc':
                            datetime.now(timezone.utc).isoformat(),
                    }
                    save_meta(out_root, method, dataset, meta)
                print(f'{method}/{dataset}: {len(per_rows_all)} images '
                      f'({len(dataset_missing_map[dataset])} missing)',
                      flush=True)

            elapsed = max(time.time() - t0, 1e-6)
            rate = processed / elapsed
            eta = ((total_expected - processed) / rate
                   if rate > 0 and processed < total_expected else 0.0)
            if status is not None:
                status.update(
                    datasets_total=len(order), overall_total=total_expected,
                    datasets_done=completed, overall_done=processed,
                    active_datasets=sorted(active), dataset=None,
                    dataset_done=0, dataset_total=0, rate=round(rate, 2),
                    eta_s=round(eta, 1),
                    message=f'{method}: {processed}/{total_expected} images')

    ordered = {ds: per_dataset[ds] for ds in order if ds in per_dataset}
    return ordered, missing


def summarize(per_dataset, metric_names):
    summary = {}
    for dataset, rows in per_dataset.items():
        if not rows:
            continue
        summary[dataset] = {}
        for metric in metric_names:
            values = np.asarray([r[metric] for r in rows], dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                summary[dataset][metric] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'n': int(values.size),
                }
            else:
                summary[dataset][metric] = {
                    'mean': float('nan'),
                    'std': float('nan'),
                    'n': 0,
                }
    return summary


def role_of(dataset, rows):
    roles = {row['role'] for row in rows if row['dataset'] == dataset}
    if len(roles) != 1:
        raise SystemExit(f'{dataset}: expected one role, got {roles}')
    return roles.pop().capitalize()


def macro(summary, role_map, metric):
    groups = defaultdict(list)
    for dataset, values in summary.items():
        item = values.get(metric)
        if item and math.isfinite(item['mean']):
            groups[role_map[dataset]].append(item['mean'])
    out = {}
    for role in ROLE_ORDER:
        vals = groups.get(role, [])
        out[role] = {
            'mean': float(np.mean(vals)) if vals else float('nan'),
            'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'n': len(vals),
        }
    all_vals = [v for vals in groups.values() for v in vals]
    out['All'] = {
        'mean': float(np.mean(all_vals)) if all_vals else float('nan'),
        'std': float(np.std(all_vals, ddof=1)) if len(all_vals) > 1 else 0.0,
        'n': len(all_vals),
    }
    return out


PERCENT_METRICS = {
    'dice', 'iou', 'precision', 'recall', 'sensitivity', 'specificity',
    'accuracy', 'mcc', 'cldice', 'nsd', 'dice_auc', 'cldice_auc',
}


def fmt(value, metric, percent=True):
    if not math.isfinite(value):
        return 'NaN'
    if percent and metric in PERCENT_METRICS:
        value *= 100.0
    return f'{value:.2f}'


def _atomic_write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def write_csv(path, fieldnames, rows):
    """Write a CSV atomically so a kill cannot leave a truncated cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    try:
        with open(tmp, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def write_markdown(path, methods, summary, macros, metric_names):
    lines = ['# Paper metrics', '']
    for metric in metric_names:
        lines += [f'## {metric}', '', '| Method | Prompt | ' +
                  ' | '.join(ROLE_ORDER + ['All']) + ' |',
                  '|---|---|' + '---:|' * (len(ROLE_ORDER) + 1)]
        for method in methods:
            display, prompt = METHOD_META.get(method, (method, '?'))
            vals = [fmt(macros[method][metric][role]['mean'], metric)
                    for role in ROLE_ORDER + ['All']]
            lines.append(f'| {display} | {prompt} | ' + ' | '.join(vals) + ' |')
        lines.append('')
    _atomic_write_text(path, '\n'.join(lines) + '\n')


def write_latex(out_dir, methods, summary, macros, metric_names):
    # ``summary`` is method -> dataset -> metrics.  Build the union of
    # datasets once; looking up summary[method] by dataset is required.
    datasets = sorted({dataset for method in methods for dataset in summary[method]})
    for metric in metric_names:
        lines = [
            '\\begin{tabular}{l' + 'c' * len(methods) + '}',
            '\\toprule',
            'Dataset & ' + ' & '.join(
                METHOD_META.get(m, (m, '?'))[0] for m in methods) + ' \\\\',
            '\\midrule',
        ]
        for dataset in datasets:
            vals = []
            for method in methods:
                item = summary[method].get(dataset, {}).get(metric)
                vals.append('--' if item is None else
                            f"{fmt(item['mean'], metric)} $\\pm$ "
                            f"{fmt(item['std'], metric)}")
            lines.append(f'{dataset} & ' + ' & '.join(vals) + ' \\\\')
        lines += ['\\bottomrule', '\\end{tabular}', '']
        _atomic_write_text(out_dir / f'table_dataset_{metric}.tex',
                           '\n'.join(lines))
        lines = [
            '\\begin{tabular}{l' + 'c' * (len(ROLE_ORDER) + 1) + '}',
            '\\toprule',
            'Method & ' + ' & '.join(ROLE_ORDER + ['All']) + ' \\\\',
            '\\midrule',
        ]
        for method in methods:
            vals = [fmt(macros[method][metric][role]['mean'], metric)
                    for role in ROLE_ORDER + ['All']]
            lines.append(f'{METHOD_META.get(method, (method, "?"))[0]} & ' +
                         ' & '.join(vals) + ' \\\\')
        lines += ['\\bottomrule', '\\end{tabular}', '']
        _atomic_write_text(out_dir / f'table_macro_{metric}.tex',
                           '\n'.join(lines))


def print_summary(methods, summaries, macros, roles, metric_names):
    short = {'dice': 'Dice', 'iou': 'IoU', 'cldice': 'clDice', 'hd95': 'HD95'}
    labels = [short.get(m, m) for m in metric_names]
    for method in methods:
        print(f'\n[{method}] per-dataset (mean)')
        print('  ' + f'{"dataset":<14} {"role":<7} ' +
              ' '.join(f'{label:>8}' for label in labels))
        for dataset in sorted(summaries[method]):
            values = summaries[method][dataset]
            cells = [fmt(values.get(m, {}).get('mean', float('nan')), m)
                     for m in metric_names]
            print('  ' + f'{dataset:<14} {roles[method][dataset]:<7} ' +
                  ' '.join(f'{cell:>8}' for cell in cells))
        print(f'[{method}] macro')
        for role in ROLE_ORDER + ['All']:
            cells = [fmt(macros[method][m][role]['mean'], m)
                     for m in metric_names]
            print('  ' + f'{role:<7}          ' +
                  ' '.join(f'{cell:>8}' for cell in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path,
                        default=REPO_ROOT / 'data' / 'sacm_3shot' /
                        'protocol_manifest.csv')
    parser.add_argument('--pred-root', type=Path,
                        default=REPO_ROOT / 'results' / 'predictions')
    parser.add_argument('--out-root', type=Path,
                        default=REPO_ROOT / 'results' / 'paper')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='method directory names (default: discover all)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='optional dataset-name filter')
    parser.add_argument('--metrics', default='dice,iou,cldice,hd95',
                        help="comma-separated metrics or 'all'")
    parser.add_argument('--max-images', type=int, default=None,
                        help='per-dataset cap for smoke tests')
    parser.add_argument('--allow-missing', action='store_true',
                        help='write partial results instead of failing')
    parser.add_argument(
        '--allow-partial-overwrite', action='store_true',
        help='allow a dataset/metric/max-images subset to overwrite an '
             'existing complete paper_metrics.csv')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'parallel dataset workers (default: '
                             f'{DEFAULT_WORKERS})')
    parser.add_argument('--chunk-size', type=int, default=100,
                        help='images per worker task (default: 100)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='ignore per-dataset metric caches')
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    if args.max_images is not None and args.max_images <= 0:
        raise SystemExit('--max-images must be positive (omit it to use all)')
    if args.workers < 1:
        raise SystemExit('--workers must be >= 1')
    if args.chunk_size < 1:
        raise SystemExit('--chunk-size must be >= 1')

    collect_metrics = 'all' if args.metrics == 'all' else [
        m.strip() for m in args.metrics.split(',') if m.strip()]
    if collect_metrics != 'all':
        unknown = [m for m in collect_metrics if m not in MAIN_METRICS]
        if unknown:
            raise SystemExit(f'unknown metrics: {unknown}')

    rows = read_rows(args.protocol, set(args.datasets) if args.datasets else None)
    methods = args.methods or discover_methods(args.pred_root)
    if not methods:
        raise SystemExit(f'no methods under {args.pred_root}')

    metric_names = None if collect_metrics == 'all' else collect_metrics
    use_cache = args.resume and collect_metrics != 'all'
    if collect_metrics == 'all' and args.resume:
        print('[info] --metrics all disables per-dataset metric cache')
    status_path = Path(os.environ.get('SACM_RUN_STATUS') or
                       (args.pred_root.parent / '.run_status' /
                        f'{methods[0]}.json'))
    os.environ.setdefault('SACM_RUN_STATUS', str(status_path))
    status = RunStatus(method=methods[0] if len(methods) == 1
                       else 'paper_metrics',
                       stage='paper_metrics', heartbeat=10.0)

    all_summaries = {}
    all_macros = {}
    all_roles = {}
    stats = {'methods': {}, 'missing': {}}
    try:
        for method in methods:
            worker_metrics = ('all' if collect_metrics == 'all'
                              else metric_names)
            grouped = group_rows(rows, args.max_images)
            status.update(
                datasets_total=len(grouped),
                overall_total=sum(len(v) for v in grouped.values()),
                datasets_done=0, overall_done=0, active_datasets=[],
                dataset=None, dataset_done=0, dataset_total=0,
                rate=0.0, eta_s=None, message=f'{method}: starting')
            per_dataset, missing = collect_method(
                method, rows, args.pred_root, args.out_root,
                worker_metrics, args.max_images, args.workers,
                use_cache, status, args.chunk_size)
            if missing and not args.allow_missing:
                raise SystemExit(
                    f'{method}: {len(missing)} missing predictions; '
                    f'first={missing[0]} (use --allow-missing to continue)')
            if metric_names is None:
                first_rows = next(iter(per_dataset.values()), [])
                if not first_rows:
                    raise SystemExit(
                        f'{method}: no predictions were collected; '
                        'check --pred-root and --allow-missing')
                metric_names = [k for k in first_rows[0]
                                if k not in ('image', 'role')]
            summary = summarize(per_dataset, metric_names)
            role_map = {dataset: role_of(dataset, rows)
                        for dataset in summary}
            macros = {m: macro(summary, role_map, m)
                      for m in metric_names}
            expected = {row['dataset'] for row in rows}
            missing_datasets = sorted(expected - set(summary))
            if missing_datasets:
                print(f'[warn] {method}: missing datasets '
                      f'{missing_datasets}; macro denominators are reduced')
            all_summaries[method] = summary
            all_macros[method] = macros
            all_roles[method] = role_map
            stats['methods'][method] = {
                'datasets': len(summary),
                'images': sum(len(v) for v in per_dataset.values()),
                'missing': len(missing),
                'missing_datasets': missing_datasets,
            }
            stats['missing'][method] = missing[:100]
            print(f'{method}: {stats["methods"][method]["images"]} images, '
                  f'{stats["methods"][method]["datasets"]} datasets, '
                  f'{len(missing)} missing', flush=True)

        if metric_names is None:
            raise SystemExit('no metrics collected')
        args.out_root.mkdir(parents=True, exist_ok=True)
        metric_fields = []
        for m in metric_names:
            metric_fields += [f'{m}_mean', f'{m}_std', f'{m}_n']
        metric_subset = (collect_metrics != 'all'
                         and set(collect_metrics) != set(MAIN_METRICS))
        partial_run = (bool(args.datasets) or args.max_images is not None
                       or metric_subset)
        canonical_metrics = args.out_root / 'paper_metrics.csv'
        if (partial_run and canonical_metrics.is_file()
                and not args.allow_partial_overwrite):
            raise SystemExit(
                f'refusing to overwrite complete {canonical_metrics} with a '
                'partial run (--datasets/--metrics/--max-images); rerun the '
                'full protocol or pass --allow-partial-overwrite')
        dataset_rows = []
        for method in methods:
            for dataset, vals in sorted(all_summaries[method].items()):
                row = {'method': method, 'dataset': dataset,
                       'role': role_of(dataset, rows)}
                for m in metric_names:
                    item = vals.get(m, {})
                    row[f'{m}_mean'] = item.get('mean')
                    row[f'{m}_std'] = item.get('std')
                    row[f'{m}_n'] = item.get('n', 0)
                dataset_rows.append(row)
        write_csv(args.out_root / 'paper_metrics.csv',
                  ['method', 'dataset', 'role'] + metric_fields, dataset_rows)

        macro_rows = []
        for method in methods:
            for role in ROLE_ORDER + ['All']:
                row = {'method': method, 'role': role}
                for m in metric_names:
                    item = all_macros[method][m][role]
                    row[f'{m}_mean'] = item['mean']
                    row[f'{m}_std'] = item['std']
                    row[f'{m}_n_datasets'] = item['n']
                macro_rows.append(row)
        macro_fields = []
        for m in metric_names:
            macro_fields += [f'{m}_mean', f'{m}_std', f'{m}_n_datasets']
        write_csv(args.out_root / 'paper_macro.csv',
                  ['method', 'role'] + macro_fields, macro_rows)

        write_markdown(args.out_root / 'paper_tables.md',
                       methods, all_summaries, all_macros, metric_names)
        write_latex(args.out_root, methods, all_summaries, all_macros,
                    metric_names)
        stats.update({
            'protocol': str(args.protocol),
            'pred_root': str(args.pred_root),
            'metrics': metric_names,
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'test_rows': len(rows),
            'macro': {
                method: {
                    m: {role: all_macros[method][m][role]
                        for role in ROLE_ORDER + ['All']}
                    for m in metric_names
                }
                for method in methods
            },
        })
        with open(args.out_root / 'paper_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
            f.write('\n')
        print_summary(methods, all_summaries, all_macros, all_roles,
                      metric_names)
        status.finish('paper metrics complete')
    except BaseException as exc:
        status.fail(exc)
        raise
    print(f'wrote {args.out_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
