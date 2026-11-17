#!/usr/bin/env python3
"""Apply label-free morphology to saved binary TopoStrand predictions.

The script preserves the canonical prediction contract
(``<root>/topostrand/<dataset>/<stem>.png``) so the existing metric
collector can score the result unchanged.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (
    convolve,
    distance_transform_edt,
    label,
    minimum,
)
from scipy.spatial import cKDTree
from skimage.draw import line as draw_line
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    disk,
    skeletonize,
)

NEIGHBORS = np.ones((3, 3), dtype=np.uint8)


def load_binary(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert('L'))
    if values.size and int(values.max()) <= 1:
        return values > 0
    return values > 127


def save_binary(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode='L').save(path)


def skeleton_endpoints(skeleton: np.ndarray) -> np.ndarray:
    counts = convolve(skeleton.astype(np.uint8), NEIGHBORS, mode='constant')
    return skeleton & ((counts - skeleton.astype(np.uint8)) == 1)


def local_tangents(skeleton: np.ndarray, points: np.ndarray,
                   radius: int = 4) -> np.ndarray:
    coords = np.argwhere(skeleton)
    if len(coords) == 0:
        return np.zeros((len(points), 2), dtype=np.float32)
    tree = cKDTree(coords)
    tangents = np.zeros((len(points), 2), dtype=np.float32)
    for index, point in enumerate(points):
        local = coords[tree.query_ball_point(point, radius)]
        if len(local) < 3:
            tangents[index] = (0.0, 0.0)
            continue
        centered = local.astype(np.float32) - local.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        tangents[index] = vh[0]
    return tangents


def bridge_endpoints(mask: np.ndarray, max_dist: float = 16.0,
                     angle_deg: float = 45.0,
                     border_margin: int = 2,
                     min_comp_area: int = 20,
                     max_added_frac: float = 0.002) -> np.ndarray:
    """Connect endpoint pairs of different components along their tangents."""
    skeleton = skeletonize(mask)
    components, count = label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(components.ravel())
    endpoints = np.argwhere(skeleton_endpoints(skeleton))
    if len(endpoints) < 2:
        return mask
    height, width = mask.shape
    keep = [
        point for point in endpoints
        if (border_margin <= point[0] < height - border_margin
            and border_margin <= point[1] < width - border_margin
            and sizes[components[tuple(point)]] >= min_comp_area)
    ]
    endpoints = np.asarray(keep, dtype=np.int32)
    if len(endpoints) < 2:
        return mask
    tangents = local_tangents(skeleton, endpoints)
    labels = np.asarray([
        components[tuple(point)] for point in endpoints], dtype=np.int32)
    tree = cKDTree(endpoints.astype(np.float32))
    pairs = sorted(
        tree.query_pairs(max_dist),
        key=lambda pair: float(np.linalg.norm(
            endpoints[pair[1]] - endpoints[pair[0]])))
    cos_limit = float(np.cos(np.deg2rad(angle_deg)))
    limit = max(1, int(max_added_frac * float(mask.sum())))
    used = set()
    bridges = np.zeros_like(mask)
    added = 0
    for first, second in pairs:
        if first in used or second in used:
            continue
        if labels[first] == labels[second]:
            continue
        vector = endpoints[second] - endpoints[first]
        length = float(np.linalg.norm(vector))
        if length <= 1.0:
            continue
        direction = vector / length
        tangent_first = tangents[first]
        tangent_second = tangents[second]
        if (np.linalg.norm(tangent_first) < 0.5
                or np.linalg.norm(tangent_second) < 0.5):
            continue
        alignment_first = abs(float(np.dot(direction, tangent_first)))
        alignment_second = abs(float(np.dot(direction, tangent_second)))
        if (alignment_first < cos_limit
                or alignment_second < cos_limit):
            continue
        rr, cc = draw_line(
            int(endpoints[first][0]), int(endpoints[first][1]),
            int(endpoints[second][0]), int(endpoints[second][1]))
        new_pixels = int((~mask[rr, cc]).sum())
        if added + new_pixels > limit:
            continue
        bridges[rr, cc] = True
        added += new_pixels
        used.add(first)
        used.add(second)
    return mask | bridges


def remove_isolated(mask: np.ndarray, min_area: int = 20,
                    max_distance: float = 5.0) -> np.ndarray:
    """Drop small components that are far from any large component."""
    components, count = label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(components.ravel())
    large_ids = np.flatnonzero(sizes >= min_area)
    large_ids = large_ids[large_ids != 0]
    if len(large_ids) == 0:
        return mask
    large = np.isin(components, large_ids)
    distance = distance_transform_edt(~large)
    nearest = minimum(distance, components, index=np.arange(1, count + 1))
    nearest = np.asarray(nearest, dtype=np.float32)
    keep_ids = np.flatnonzero(
        (sizes[1:] >= min_area) | (nearest <= max_distance)) + 1
    return np.isin(components, keep_ids)


def adaptive_skel_dilate(mask: np.ndarray, args) -> np.ndarray:
    """Skeleton dilation gated by predicted width and foreground density."""
    if not mask.any():
        return mask
    skeleton = skeletonize(mask)
    length = float(skeleton.sum())
    if length <= 0:
        return mask
    width = 2.0 * float(mask.sum()) / length
    foreground = float(mask.mean())
    medium = (args.width_low <= width <= args.width_high
              and foreground < args.fg_dense)
    apply = ((foreground >= args.fg_min and not medium)
             or foreground >= args.fg_thick)
    if not apply:
        return mask
    if foreground >= args.fg_thick and width <= args.width_high:
        return mask | binary_dilation(mask, disk(args.thick_radius))
    if (foreground >= args.fg_dense_thin
            and width <= args.width_dense_thin):
        return mask | binary_dilation(mask, disk(args.dense_thin_radius))
    return mask | binary_dilation(skeleton, disk(args.radius))


def adaptive_calibrate(mask: np.ndarray, args) -> np.ndarray:
    """Density/width-gated skeleton dilation (1 or 2 px)."""
    if not mask.any():
        return mask
    skeleton = skeletonize(mask)
    length = float(skeleton.sum())
    if length <= 0:
        return mask
    width = 2.0 * float(mask.sum()) / length
    foreground = float(mask.mean())
    if foreground < args.fg_sparse:
        radius = 2
    elif (width >= args.width_wide
          and args.fg_wide_min <= foreground < args.fg_dense):
        radius = 2
    elif foreground >= args.fg_thick and width <= args.width_high:
        radius = 2
    elif foreground >= args.fg_min and width < args.width_narrow:
        radius = 1
    elif foreground >= args.fg_thick and width > args.width_high:
        radius = 1
    else:
        return mask
    return mask | binary_dilation(skeleton, disk(radius))


def transform(mask: np.ndarray, name: str, args) -> np.ndarray:
    if name == 'identity':
        return mask
    if name == 'skel_dilate':
        skeleton = skeletonize(mask)
        return mask | binary_dilation(skeleton, disk(args.radius))
    if name == 'thin_skel_dilate':
        skeleton = skeletonize(mask)
        radius_map = distance_transform_edt(mask)
        thin = skeleton & (radius_map <= args.max_radius)
        return mask | binary_dilation(thin, disk(args.radius))
    if name == 'adaptive_skel':
        return adaptive_skel_dilate(mask, args)
    if name == 'adaptive_cal':
        return adaptive_calibrate(mask, args)
    if name == 'close':
        return binary_closing(mask, disk(args.radius))
    if name == 'open':
        return binary_opening(mask, disk(args.radius))
    if name == 'dilate':
        return binary_dilation(mask, disk(args.radius))
    if name == 'erode':
        return binary_erosion(mask, disk(args.radius))
    if name == 'remove_small':
        labels, count = label(mask)
        if count == 0:
            return mask
        sizes = np.bincount(labels.ravel())
        keep = sizes >= args.min_area
        keep[0] = False
        return keep[labels]
    if name == 'remove_isolated':
        return remove_isolated(mask, args.min_area, args.isolated_distance)
    bridge_kwargs = dict(
        max_dist=args.max_dist,
        angle_deg=args.angle_deg,
        border_margin=args.border_margin,
        min_comp_area=args.min_comp_area,
        max_added_frac=args.max_added_frac,
    )
    if name == 'bridge':
        return bridge_endpoints(mask, **bridge_kwargs)
    if name == 'skel_bridge':
        bridged = bridge_endpoints(mask, **bridge_kwargs)
        skeleton = skeletonize(bridged)
        return bridged | binary_dilation(skeleton, disk(args.radius))
    if name == 'thin_skel_bridge':
        skeleton = skeletonize(mask)
        radius_map = distance_transform_edt(mask)
        thin = skeleton & (radius_map <= args.max_radius)
        grown = mask | binary_dilation(thin, disk(args.radius))
        return bridge_endpoints(grown, **bridge_kwargs)
    raise SystemExit(f'unknown transform: {name}')


def process_one(task):
    source, destination, name, args = task
    mask = load_binary(source)
    result = transform(mask, name, args)
    save_binary(destination, result)
    return int(result.sum() != mask.sum())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', type=Path, required=True,
                        help='prediction root containing topostrand/')
    parser.add_argument('--dst', type=Path, required=True,
                        help='destination prediction root')
    parser.add_argument('--transform', default='identity')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--radius', type=int, default=1)
    parser.add_argument('--max-radius', type=float, default=2.0)
    parser.add_argument('--width-low', type=float, default=7.0)
    parser.add_argument('--width-high', type=float, default=15.0)
    parser.add_argument('--fg-min', type=float, default=0.04)
    parser.add_argument('--fg-dense', type=float, default=0.09)
    parser.add_argument('--fg-thick', type=float, default=0.15)
    parser.add_argument('--thick-radius', type=int, default=2)
    parser.add_argument('--dense-thin-radius', type=int, default=1)
    parser.add_argument('--fg-dense-thin', type=float, default=0.09)
    parser.add_argument('--width-dense-thin', type=float, default=8.0)
    parser.add_argument('--fg-sparse', type=float, default=0.005)
    parser.add_argument('--width-wide', type=float, default=10.0)
    parser.add_argument('--fg-wide-min', type=float, default=0.0)
    parser.add_argument('--width-narrow', type=float, default=7.5)
    parser.add_argument('--min-area', type=int, default=16)
    parser.add_argument('--isolated-distance', type=float, default=5.0)
    parser.add_argument('--max-dist', type=float, default=16.0)
    parser.add_argument('--angle-deg', type=float, default=45.0)
    parser.add_argument('--border-margin', type=int, default=2)
    parser.add_argument('--min-comp-area', type=int, default=20)
    parser.add_argument('--max-added-frac', type=float, default=0.002)
    args = parser.parse_args(argv)

    source_root = args.src / 'topostrand'
    if not source_root.is_dir():
        raise SystemExit(f'missing {source_root}')
    datasets = sorted(
        path.name for path in source_root.iterdir() if path.is_dir())
    if args.datasets:
        selected = set(args.datasets)
        datasets = [name for name in datasets if name in selected]
    tasks = []
    for dataset in datasets:
        for source in sorted((source_root / dataset).glob('*.png')):
            destination = args.dst / 'topostrand' / dataset / source.name
            tasks.append((source, destination, args.transform, args))
    print(f'{args.transform}: {len(tasks)} masks -> {args.dst}', flush=True)
    changed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            changed += int(future.result())
            if index % 500 == 0:
                print(f'  {index}/{len(tasks)}', flush=True)
    print(f'done: {changed}/{len(tasks)} masks changed', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
