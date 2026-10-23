#!/usr/bin/env python3
"""Evaluate binary prediction PNGs with the project's unified metrics.

Predictions are expected at:
    <pred-root>/<dataset>/<stem>.png       (0/255, original GT size)

Usage:
    python scripts/evaluate_predictions.py \
        --pred-root results/predictions/unet \
        --datasets DRIVE:data/sacm_3shot/test/DRIVE \
                   WIRE:datasets/novel/WIRE \
        --out-root results/unet
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from core.metrics import (  # noqa: E402
    cldice_score,
    compute_metrics,
    dice_score,
    hd95_score,
    iou_score,
)

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.ppm')
METRIC_FNS = {
    'dice': dice_score,
    'iou': iou_score,
    'cldice': cldice_score,
    'hd95': hd95_score,
}


def pairs(root: Path):
    image_dir, mask_dir = root / 'images', root / 'masks'
    out = []
    for image in sorted(image_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTS:
            continue
        mask = next(
            (mask_dir / (image.stem + ext) for ext in IMAGE_EXTS
             if (mask_dir / (image.stem + ext)).is_file()),
            None)
        if mask is not None:
            out.append((image, mask))
    return out


def compute_selected(pred, gt, metric_names):
    if metric_names == 'all':
        return compute_metrics(pred, gt, threshold=0.5)
    pred_bin = pred > 0.5
    gt_bin = gt > 0.5
    return {name: float(METRIC_FNS[name](pred_bin, gt_bin))
            for name in metric_names}


def evaluate_one(pred_dir: Path, gt_root: Path, metric_names):
    rows = []
    metric_values = {}
    for image, mask_path in pairs(gt_root):
        pred_path = pred_dir / f'{image.stem}.png'
        if not pred_path.is_file():
            raise SystemExit(f'missing prediction: {pred_path}')
        pred = np.array(Image.open(pred_path).convert('L'), dtype=np.float32) / 255.0
        gt = np.array(Image.open(mask_path).convert('L'), dtype=np.float32) / 255.0
        if pred.shape != gt.shape:
            raise SystemExit(
                f'size mismatch for {image.stem}: pred={pred.shape} gt={gt.shape}')
        m = compute_selected(pred, gt, metric_names)
        row = {'image': image.stem}
        row.update(m)
        rows.append(row)
        for key, value in m.items():
            metric_values.setdefault(key, []).append(value)
    summary = {}
    for key, values in metric_values.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            summary[key] = (float(arr.mean()), float(arr.std()))
        else:
            summary[key] = (float('nan'), float('nan'))
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred-root', type=Path, required=True,
                        help='directory containing <dataset>/<stem>.png')
    parser.add_argument('--datasets', nargs='+', required=True,
                        help='NAME:GT_ROOT entries; GT_ROOT has images/ + masks/')
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--metrics', default='dice,iou,cldice,hd95',
                        help="comma-separated subset of "
                             f"{','.join(METRIC_FNS)} or 'all'")
    args = parser.parse_args()

    metric_names = 'all' if args.metrics == 'all' else [
        name.strip() for name in args.metrics.split(',') if name.strip()]
    if metric_names != 'all':
        unknown = [name for name in metric_names if name not in METRIC_FNS]
        if unknown:
            raise SystemExit(
                f'unknown metrics: {unknown}; choose from {list(METRIC_FNS)} or all')

    for spec in args.datasets:
        name, sep, root = spec.partition(':')
        if not sep or not name or not root:
            raise SystemExit(f'--datasets entries must be NAME:DIR, got {spec!r}')
        pred_dir = args.pred_root / name
        gt_root = Path(root)
        summary, rows = evaluate_one(pred_dir, gt_root, metric_names)
        out_dir = args.out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'results_summary.txt', 'w') as f:
            f.write(f'Dataset: {root}\n')
            f.write(f'Num Samples: {len(rows)}\n')
            for key, (mean, std) in summary.items():
                f.write(f'Average {key.capitalize()}: {mean:.4f} ± {std:.4f}\n')
        with open(out_dir / 'metrics.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image'] + list(summary))
            writer.writeheader()
            writer.writerows(rows)
        print(f'{name}: {len(rows)} images -> {out_dir}')
        for key in ('dice', 'iou', 'cldice', 'hd95'):
            if key in summary:
                mean, std = summary[key]
                print(f'  {key}: {mean:.4f} ± {std:.4f}')


if __name__ == '__main__':
    main()
