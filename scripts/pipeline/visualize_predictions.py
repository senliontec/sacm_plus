#!/usr/bin/env python3
"""Save qualitative prediction panels and per-dataset grids.

For every method/dataset, a fixed random sample of test images is rendered as:
    input | GT overlay | prediction overlay | error map (TP/FP/FN)
Individual panels and a contact sheet are written, plus a CSV of per-image
metrics.

Examples:
    python scripts/pipeline/visualize_predictions.py --methods unet --num-samples 8
    python scripts/pipeline/visualize_predictions.py --methods unet --datasets DRIVE
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'scripts' / 'pipeline'))

from metrics import compute_selected  # noqa: E402
from core.run_status import RunStatus  # noqa: E402

METHOD_META = {
    'a2sam': ('A2SAM (MIA 2026)', '✓'),
    'aco_moe_lora': ('ACO-MoE-LoRA', '✓'),
    'unet': ('U-Net', '✗'),
    'cs2net': ('CS²Net', '✗'),
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
    'sam-med2d': ('SAM-Med2D', '✓'),
    'sam-octa': ('SAM-OCTA', '✓'),
    'sansa': ('SANSA', '✓'),
    'scrwkv': ('SCRWKV', '✗'),
    'cwsam': ('CWSAM', '✓'),
    'knowsam': ('KnowSAM', '✓'),
    'segdino': ('SegDINO', '✓'),
    'segdino_v2': ('SegDINO v2', '✓'),
    'sacm': ('SACM', '✗'),
    'sam_adapter': ('SAM-Adapter', '✓'),
    'samed': ('SAMed', '✓'),
    'topolora': ('TopoLoRA-SAM', '✓'),
    'tpsegformer': ('TPSegformer', '✗'),
    'ucs': ('UCS', '✓'),
}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def read_test_rows(protocol_path: Path, dataset_filter):
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


def sample_rows(rows, num_samples, seed):
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row['dataset'], []).append(row)
    selected = {}
    for index, dataset in enumerate(sorted(by_dataset)):
        items = sorted(by_dataset[dataset], key=lambda r: r['mask_path'])
        if num_samples is None or num_samples <= 0 or num_samples >= len(items):
            selected[dataset] = items
        else:
            rng = random.Random(seed + index)
            selected[dataset] = sorted(
                rng.sample(items, num_samples), key=lambda r: r['mask_path'])
    return selected


def load_image(path: Path, max_side: int):
    image = Image.open(path).convert('RGB')
    size = image.size
    if max_side and max(size) > max_side:
        scale = max_side / max(size)
        image = image.resize(
            (max(1, int(size[0] * scale)), max(1, int(size[1] * scale))),
            Image.BICUBIC)
    return image, size


def overlay(image, mask, color):
    base = np.asarray(image, dtype=np.float32) / 255.0
    m = mask.astype(bool)
    out = base.copy()
    if m.any():
        out[m] = 0.45 * base[m] + 0.55 * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0, 1)


def error_map(image, pred, gt):
    base = np.asarray(image.convert('L'), dtype=np.float32) / 255.0
    rgb = np.repeat(base[..., None] * 0.35, 3, axis=2)
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    rgb[tp] = [0.0, 0.8, 0.2]
    rgb[fp] = [0.95, 0.15, 0.15]
    rgb[fn] = [0.15, 0.35, 0.95]
    return np.clip(rgb, 0, 1)


def make_panel(image, gt, pred, title, metrics):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), dpi=140)
    panels = [
        (np.asarray(image), 'Input'),
        (overlay(image, gt, (0.0, 0.9, 0.2)), 'GT'),
        (overlay(image, pred, (0.95, 0.1, 0.1)), 'Prediction'),
        (error_map(image, pred, gt), 'Error: TP/FP/FN'),
    ]
    for ax, (arr, label) in zip(axes, panels):
        ax.imshow(arr)
        ax.set_title(label, fontsize=9)
        ax.axis('off')
    fig.suptitle(
        f"{title}  Dice={metrics['dice']:.3f}  IoU={metrics['iou']:.3f}  "
        f"clDice={metrics['cldice']:.3f}  HD95={metrics['hd95']:.1f}",
        fontsize=10)
    fig.tight_layout()
    return fig


def make_grid(panels, title):
    n = len(panels)
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.0 * n), dpi=120,
                             squeeze=False)
    for row_axes, (image, gt, pred, name) in zip(axes, panels):
        row_axes[0].imshow(image)
        row_axes[0].set_title('Input', fontsize=8)
        row_axes[1].imshow(overlay(image, gt, (0.0, 0.9, 0.2)))
        row_axes[1].set_title('GT', fontsize=8)
        row_axes[2].imshow(overlay(image, pred, (0.95, 0.1, 0.1)))
        row_axes[2].set_title('Prediction', fontsize=8)
        row_axes[3].imshow(error_map(image, pred, gt))
        row_axes[3].set_title('TP/FP/FN', fontsize=8)
        for ax in row_axes:
            ax.axis('off')
        row_axes[0].set_ylabel(name, fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path,
                        default=REPO_ROOT / 'data' / 'sacm_3shot' /
                        'protocol_manifest.csv')
    parser.add_argument('--pred-root', type=Path,
                        default=REPO_ROOT / 'results' / 'predictions')
    parser.add_argument('--out-root', type=Path,
                        default=REPO_ROOT / 'results' / 'visualizations')
    parser.add_argument('--methods', nargs='+', required=True)
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--num-samples', type=int, default=8,
                        help='images per dataset; 0/negative means all')
    parser.add_argument('--max-side', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--allow-missing', action='store_true')
    args = parser.parse_args()

    rows = read_test_rows(
        args.protocol, set(args.datasets) if args.datasets else None)
    samples = sample_rows(rows, args.num_samples, args.seed)
    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    total_panels = sum(len(items) for _ in args.methods
                       for items in samples.values())
    units = len(args.methods) * len(samples)
    status = RunStatus(method=args.methods[0], stage='visualize',
                       heartbeat=10.0)
    status.update(datasets_total=units, overall_total=total_panels,
                  overall_done=0)

    unit = 0
    try:
        for method in args.methods:
            display = METHOD_META.get(method, (method, '?'))[0]
            for dataset, items in samples.items():
                unit += 1
                out_dir = args.out_root / method / dataset
                out_dir.mkdir(parents=True, exist_ok=True)
                status.start_dataset(f'{method}/{dataset}', index=unit,
                                     total=len(items),
                                     overall_total=total_panels)
                panels = []
                for row in items:
                    stem = Path(row['mask_path']).stem
                    pred_path = (args.pred_root / method / dataset /
                                 f'{stem}.png')
                    if not pred_path.is_file():
                        msg = f'missing prediction: {pred_path}'
                        if args.allow_missing:
                            print(f'[warn] {msg}')
                            continue
                        raise SystemExit(msg)
                    image, _ = load_image(resolve(row['image_path']),
                                          args.max_side)
                    gt_orig = Image.open(resolve(row['mask_path'])).convert('L')
                    pred_orig = Image.open(pred_path).convert('L')
                    if gt_orig.size != pred_orig.size:
                        raise SystemExit(
                            f'{pred_path}: {pred_orig.size} != GT {gt_orig.size}')
                    metrics = compute_selected(
                        np.asarray(pred_orig, dtype=np.float32) / 255.0,
                        np.asarray(gt_orig, dtype=np.float32) / 255.0,
                        ['dice', 'iou', 'cldice', 'hd95'])
                    gt_image = (gt_orig if gt_orig.size == image.size else
                                gt_orig.resize(image.size, Image.NEAREST))
                    pred_image = (pred_orig if pred_orig.size == image.size else
                                  pred_orig.resize(image.size, Image.NEAREST))
                    gt = np.asarray(gt_image) > 127
                    pred = np.asarray(pred_image) > 127
                    title = f'{display} | {dataset} | {stem}'
                    fig = make_panel(image, gt, pred, title, metrics)
                    panel_path = out_dir / f'{stem}.png'
                    fig.savefig(panel_path, bbox_inches='tight')
                    plt.close(fig)
                    panels.append((image, gt, pred, stem))
                    summary_rows.append({
                        'method': method,
                        'dataset': dataset,
                        'image': stem,
                        'dice': metrics['dice'],
                        'iou': metrics['iou'],
                        'cldice': metrics['cldice'],
                        'hd95': metrics['hd95'],
                        'panel': str(panel_path),
                    })
                    status.advance(1, write=False)
                if panels:
                    # A grid with hundreds of rows is both unreadable and an
                    # OOM risk (e.g. CHCrack5k with --num-samples 0).  Keep all
                    # individual panels but cap the contact sheet.
                    grid_panels = panels[:16]
                    fig = make_grid(grid_panels, f'{display} | {dataset}')
                    fig.savefig(out_dir / 'grid.png', bbox_inches='tight')
                    plt.close(fig)
                    if len(panels) > len(grid_panels):
                        print(f'  grid capped at {len(grid_panels)} of '
                              f'{len(panels)} panels')
                status.finish_dataset()
                print(f'{method}/{dataset}: {len(panels)} panels -> {out_dir}',
                      flush=True)
        status.finish('visualizations complete')
    except BaseException as exc:
        status.fail(exc)
        raise

    with open(args.out_root / 'visualizations.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'method', 'dataset', 'image', 'dice', 'iou', 'cldice', 'hd95',
            'panel'])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f'wrote {args.out_root / "visualizations.csv"}')


if __name__ == '__main__':
    main()
