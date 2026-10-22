#!/usr/bin/env python3
"""Prepare CREMI volumes as 2D boundary segmentation pairs.

Output: datasets/seen/CREMI/{images,masks,manifest.csv}

The target is a one-sided 8-connected boundary derived from neuron_ids.
All-zero raw slices are skipped (volume C has two such slices).
"""
from pathlib import Path

import numpy as np
from common import (
    add_row,
    build_parser,
    finish,
    prepare_output,
    require_dir,
)
from PIL import Image


def _find_hdfs(src: Path):
    try:
        import h5py
    except ImportError as e:
        raise SystemExit('CREMI conversion requires h5py') from e
    candidates = []
    for path in sorted(src.rglob('*.hdf')):
        if 'padded' in path.name or '+' in path.name:
            continue
        with h5py.File(path, 'r') as handle:
            if ('volumes/raw' in handle
                    and 'volumes/labels/neuron_ids' in handle):
                candidates.append(path)
    return candidates


def main() -> int:
    args = build_parser(
        'Prepare CREMI boundary segmentation', 'CREMI', 'seen/CREMI'
    ).parse_args()
    src = require_dir(args.src.resolve(), 'CREMI')
    hdfs = _find_hdfs(src)
    if not hdfs:
        raise SystemExit(f'CREMI: no usable unpadded .hdf found under {src}')

    import h5py
    out = prepare_output(args.out, args.overwrite)
    rows = []
    max_images = args.max_images
    for hdf in hdfs:
        if max_images is not None and len(rows) >= max_images:
            break
        sample = hdf.stem.split('_2016')[0]
        with h5py.File(hdf, 'r') as handle:
            raw = handle['volumes/raw']
            nid = handle['volumes/labels/neuron_ids']
            for z in range(raw.shape[0]):
                if max_images is not None and len(rows) >= max_images:
                    break
                image = np.asarray(raw[z], dtype=np.uint8)
                if not np.any(image):
                    continue
                labels = np.asarray(nid[z])
                boundary = np.zeros_like(labels, dtype=bool)
                boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
                boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
                name = f'{sample}_z{z:03d}'
                image_path = out / 'images' / f'{name}.png'
                mask_path = out / 'masks' / f'{name}.png'
                Image.fromarray(image).save(image_path)
                Image.fromarray((boundary * 255).astype(np.uint8)).save(mask_path)
                add_row(rows, hdf, image_path, mask_path)
    return finish(out, rows, 'CREMI')


if __name__ == '__main__':
    raise SystemExit(main())
