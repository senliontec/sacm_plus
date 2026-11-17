#!/usr/bin/env python3
"""Evaluate label-free probability postprocessing from exported arrays.

Reads ``<prob-root>/<dataset>/<stem>.npy`` with shape (4, H, W) and compares
plain candidate deployment against hysteresis and gap-repair rules.
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, label
from skimage.morphology import binary_closing, disk

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / 'src'))
from core.metrics import binary_metrics  # noqa: E402

METRICS = ('dice', 'iou', 'cldice', 'hd95')


def infer(probs: np.ndarray) -> dict[str, np.ndarray]:
    high3 = probs[3] > 0.5
    high0 = probs[0] > 0.5
    mean = probs.mean(0)
    out = {
        'h3': high3,
        'h0': high0,
        'mean': mean > 0.5,
        'h3_close2': binary_closing(high3, disk(2)),
    }
    for tau in (0.3, 0.4):
        low = probs[3] > tau
        labels, count = label(low)
        if count:
            touching = np.unique(labels[high3])
            touching = touching[touching > 0]
            keep = np.isin(labels, touching)
        else:
            keep = low
        out[f'h3_hyst{tau}'] = keep
        band = high3 | (low & binary_dilation(high3, disk(2)))
        out[f'h3_band{tau}'] = band
        low_mean = mean > tau
        labels, count = label(low_mean)
        if count:
            touching = np.unique(labels[mean > 0.5])
            touching = touching[touching > 0]
            keep = np.isin(labels, touching)
        else:
            keep = low_mean
        out[f'mean_hyst{tau}'] = keep
    return out


def process(task):
    npy, mask_path, configs = task
    probs = np.load(npy).astype(np.float32)
    from PIL import Image
    with Image.open(mask_path) as image:
        gt = np.asarray(image.convert('L')) > 127
    if gt.shape != probs.shape[1:]:
        return None
    masks = infer(probs)
    return {
        name: binary_metrics(masks[name], gt)
        for name in configs if name in masks
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prob-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--configs', nargs='+', default=None)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(argv)

    rows = list(csv.DictReader(open(args.protocol)))
    rows = [row for row in rows if row['split'].endswith('_test')]
    if args.datasets:
        selected = set(args.datasets)
        rows = [row for row in rows if row['dataset'] in selected]
    records = []
    for row in rows:
        stem = Path(row['mask_path']).stem
        npy = args.prob_root / row['dataset'] / f'{stem}.npy'
        if npy.is_file():
            records.append((npy, row['mask_path'], row['dataset']))
    sample = np.load(records[0][0], mmap_mode='r')
    configs = args.configs or list(infer(np.zeros(sample.shape, np.float32)))
    print(f'{len(records)} images, configs={configs}', flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process, (npy, mask, configs)): dataset
            for npy, mask, dataset in records
        }
        for index, future in enumerate(as_completed(futures), 1):
            value = future.result()
            if value is None:
                continue
            dataset = futures[future]
            for name, metrics in value.items():
                results.setdefault((dataset, name), []).append(metrics)
            if index % 100 == 0:
                print(f'  {index}/{len(records)}', flush=True)

    datasets = sorted({key[0] for key in results})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'config', *METRICS])
        for dataset in datasets:
            for name in configs:
                values = results.get((dataset, name), [])
                if not values:
                    continue
                means = [float(np.mean([v[m] for v in values])) for m in METRICS]
                writer.writerow([dataset, name, *means])
    print('macro', flush=True)
    for name in configs:
        values = [
            np.mean([v[m] for v in results.get((d, name), [])])
            for d in datasets
            if results.get((d, name))
        ]
        if values:
            means = [float(np.mean([v[i] for v in values]))
                     for i in range(len(METRICS))]
            print(f'  {name:12s} ' + ' '.join(
                f'{m}={v:.4f}' for m, v in zip(METRICS, means)), flush=True)
    print(f'wrote {args.out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
