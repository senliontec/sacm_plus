#!/usr/bin/env python3
"""Characterize GT mask morphology on the 20-dataset protocol.

This is diagnostic: it reads test masks from the protocol manifest and
summarizes shape properties that determine which structural module is
appropriate:

* thin vs thick (area / skeleton length),
* curvilinear vs blob-like (PCA elongation, largest-component aspect),
* fragmented vs connected (component count, largest-component share),
* holey/membrane-like (background holes),
* branching complexity (skeleton endpoints/branchpoints),
* sparse vs dense foreground.

It does not train, evaluate predictions, or touch checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

try:
    from skimage.morphology import skeletonize
except ImportError:  # pragma: no cover
    skeletonize = None

ROOT = Path(__file__).resolve().parents[2]
EIGHT = np.ones((3, 3), dtype=np.uint8)
FOUR = ndi.generate_binary_structure(2, 1)
SMALL_PX = 20


def _load_mask(path: Path) -> np.ndarray:
    values = np.asarray(Image.open(path).convert('L'))
    if values.size and int(values.max()) <= 1:
        return values > 0
    return values > 127


def _count_holes(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    labels, count = ndi.label(~mask, structure=FOUR)
    if count == 0:
        return 0
    border = np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1],
    ]))
    border = border[border != 0]
    return int(max(count - len(border), 0))


def _pca_elongation(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if ys.size < 2:
        return 0.0
    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)])
    cov = np.cov(coords)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    if eig[1] <= 1e-12:
        return 0.0
    return float(math.sqrt(max(eig[1], 0.0) / max(eig[0], 1e-12)))


def _largest_bbox_aspect(mask: np.ndarray, labels: np.ndarray,
                         n_components: int) -> float:
    if n_components == 0:
        return 0.0
    areas = np.bincount(labels.ravel())[1:]
    largest = int(np.argmax(areas)) + 1
    ys, xs = np.nonzero(labels == largest)
    if ys.size == 0:
        return 0.0
    height = int(ys.max() - ys.min() + 1)
    width = int(xs.max() - xs.min() + 1)
    return float(max(height, width) / max(min(height, width), 1))


def _metrics(task: tuple) -> dict:
    dataset, image, mask_path = task
    mask = _load_mask(Path(mask_path))
    height, width = mask.shape
    fg_area = int(mask.sum())
    fg_ratio = fg_area / max(height * width, 1)

    labels, n_components = ndi.label(mask, structure=EIGHT)
    areas = (np.bincount(labels.ravel())[1:]
             if n_components else np.zeros(0, dtype=np.int64))
    largest_share = (
        float(areas.max() / areas.sum()) if areas.size else 0.0)
    small_rate = (
        float(np.mean(areas < SMALL_PX)) if areas.size else 0.0)

    if skeletonize is not None and fg_area:
        skeleton = skeletonize(mask)
    else:
        skeleton = np.zeros_like(mask, dtype=bool)
    skel_len = int(skeleton.sum())
    mean_width = fg_area / max(skel_len, 1)
    skeleton_density = skel_len / max(fg_area, 1)

    if skel_len:
        neighbors = (
            ndi.convolve(
                skeleton.astype(np.uint8), EIGHT.astype(np.uint8),
                mode='constant', cval=0).astype(np.int16)
            - skeleton.astype(np.int16)
        )
        endpoints = int(np.count_nonzero(skeleton & (neighbors == 1)))
        branchpoints = int(np.count_nonzero(skeleton & (neighbors >= 3)))
    else:
        endpoints = 0
        branchpoints = 0

    ys, xs = np.nonzero(mask)
    if ys.size:
        bbox_area = ((ys.max() - ys.min() + 1)
                     * (xs.max() - xs.min() + 1))
        bbox_extent = fg_area / max(int(bbox_area), 1)
    else:
        bbox_extent = 0.0

    return {
        'dataset': dataset,
        'image': image,
        'height': height,
        'width': width,
        'image_aspect': max(height, width) / max(min(height, width), 1),
        'fg_ratio': fg_ratio,
        'components': n_components,
        'holes': _count_holes(mask),
        'largest_component_share': largest_share,
        'small_component_rate': small_rate,
        'skeleton_length': skel_len,
        'mean_width': mean_width,
        'mean_width_rel': mean_width / max(height, width),
        'skeleton_density': skeleton_density,
        'endpoints': endpoints,
        'branchpoints': branchpoints,
        'endpoint_density': endpoints / max(skel_len, 1),
        'branchpoint_density': branchpoints / max(skel_len, 1),
        'elongation': _pca_elongation(mask),
        'largest_bbox_aspect': _largest_bbox_aspect(
            mask, labels, n_components),
        'bbox_extent': bbox_extent,
    }


def _median(values, digits=6):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return float('nan')
    return round(float(statistics.median(values)), digits)


def _fmt(value, digits=4):
    return '—' if not math.isfinite(value) else f'{value:.{digits}f}'


def _heuristic(rows: dict) -> str:
    """A coarse label for report readability, not a scientific classifier."""
    width = rows['mean_width_rel_median']
    elong = rows['elongation_median']
    components = rows['components_median']
    largest = rows['largest_component_share_median']
    fg = rows['fg_ratio_median']
    holes = rows['holes_median']
    if components >= 20 and largest < 0.6:
        return 'fragmented'
    if width <= 0.008 and elong >= 3.0:
        return 'thin-curvilinear'
    if fg >= 0.12 and width > 0.02:
        return 'dense/blob'
    if holes >= 5:
        return 'membrane/holey'
    if fg < 0.02:
        return 'sparse'
    return 'mixed'


def _write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--datasets', default='',
                        help='space-separated dataset filter')
    parser.add_argument('--limit', type=int, default=0,
                        help='max images per dataset (0 = all)')
    args = parser.parse_args()

    selected = set(args.datasets.split()) if args.datasets.strip() else None
    groups = defaultdict(list)
    with args.protocol.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if not str(row.get('split', '')).endswith('_test'):
                continue
            dataset = str(row.get('dataset', '')).strip()
            if selected is not None and dataset not in selected:
                continue
            mask_path = Path(str(row.get('mask_path', '')))
            if not mask_path.is_absolute():
                mask_path = ROOT / mask_path
            groups[dataset].append(
                (dataset, Path(str(row.get('mask_path', ''))).name,
                 str(mask_path)))
    if not groups:
        raise SystemExit('no test masks found')

    workers = max(1, int(args.workers))
    all_rows = []
    dataset_rows = []
    for dataset in sorted(groups):
        tasks = groups[dataset]
        if args.limit:
            tasks = tasks[:max(0, int(args.limit))]
        print(f'[{dataset}] {len(tasks)} images', flush=True)
        if workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                rows = list(executor.map(_metrics, tasks, chunksize=1))
        else:
            rows = [_metrics(task) for task in tasks]
        all_rows.extend(rows)
        summary = {'dataset': dataset, 'n_images': len(rows)}
        numeric = (
            'image_aspect', 'fg_ratio', 'components', 'holes',
            'largest_component_share', 'small_component_rate',
            'skeleton_length', 'mean_width', 'mean_width_rel',
            'skeleton_density', 'endpoints', 'branchpoints',
            'endpoint_density', 'branchpoint_density', 'elongation',
            'largest_bbox_aspect', 'bbox_extent',
        )
        for key in numeric:
            summary[f'{key}_median'] = _median(
                [row[key] for row in rows])
        summary['shape_class'] = _heuristic(summary)
        dataset_rows.append(summary)

    args.out.mkdir(parents=True, exist_ok=True)
    image_fields = list(all_rows[0].keys())
    dataset_fields = list(dataset_rows[0].keys())
    _write_csv(args.out / 'mask_morphology_per_image.csv',
               image_fields, all_rows)
    _write_csv(args.out / 'mask_morphology_per_dataset.csv',
               dataset_fields, dataset_rows)

    lines = [
        '# GT mask morphology',
        '',
        f'- Protocol: `{args.protocol}`',
        f'- Images: {len(all_rows)}',
        '',
        '> Descriptive only. `shape_class` is a coarse readability label, '
        'not a trained classifier.',
        '',
        '| Dataset | n | FG ratio | components | holes | '
        'largest share | mean width rel | elongation | branch density | '
        'shape class |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in dataset_rows:
        lines.append(
            f"| {row['dataset']} | {row['n_images']} | "
            f"{_fmt(row['fg_ratio_median'])} | "
            f"{_fmt(row['components_median'], 2)} | "
            f"{_fmt(row['holes_median'], 2)} | "
            f"{_fmt(row['largest_component_share_median'], 3)} | "
            f"{_fmt(row['mean_width_rel_median'], 5)} | "
            f"{_fmt(row['elongation_median'], 2)} | "
            f"{_fmt(row['branchpoint_density_median'], 4)} | "
            f"{row['shape_class']} |"
        )
    lines += [
        '',
        '## Reading the metrics',
        '',
        '- `mean_width_rel`: foreground area / skeleton length / max(H,W); '
        'small = thin structures.',
        '- `elongation`: PCA eigenvalue ratio of foreground pixels; '
        'large = anisotropic/curvilinear.',
        '- `components`: GT connected components per image; large with low '
        'largest share = genuinely fragmented GT.',
        '- `holes`: background holes per image; large = membrane/holey GT.',
        '- `branchpoint_density`: branch points per skeleton pixel; '
        'large = branching curvilinear structures.',
        '',
    ]
    (args.out / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f'Wrote: {args.out / "mask_morphology_per_dataset.csv"}')
    print(f'Wrote: {args.out / "mask_morphology_per_image.csv"}')
    print(f'Wrote: {args.out / "report.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
