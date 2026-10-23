#!/usr/bin/env python3
"""Prepare the official CORN-1 release for the SACM protocol.

Source layout (segment/datasets/CORN/CORN-1)::

    images/train_orig/  labels/train_label/    1176 pairs
    images/val_orig/    labels/val_label/       188 pairs
    images/test_orig/   labels/test_label/      152 pairs

Output (default)::

    datasets/seen/CORN/train/{images,masks,manifest.csv}   official train
    datasets/seen/CORN/val/{images,masks,manifest.csv}     official val
    datasets/seen/CORN/test/{images,masks,manifest.csv}    official test

The official splits are batch-disjoint (train batches 1-8,12; val 9;
test 10-11), so using them preserves the author-provided leakage barrier.
Masks are written as single-channel 0/255 PNG at threshold >127.

One dataset directory with explicit train/val/test subdirectories keeps the
official split visible without creating three sibling role directories.
prepare_data consumes the subdirectories via named entries:

    CORN:datasets/seen/CORN/train
    --val_dirs  CORN:datasets/seen/CORN/val
    --test_dirs CORN:datasets/seen/CORN/test
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    RAW_ROOT,
    REPO_ROOT,
    add_row,
    copy_image,
    finish,
    index_by_stem,
    iter_files,
    limit,
    prepare_output,
    require_dir,
    save_binary_mask,
)

SPLITS = (
    ('train', 'train_orig', 'train_label'),
    ('val', 'val_orig', 'val_label'),
    ('test', 'test_orig', 'test_label'),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Prepare official CORN-1 train/val/test splits'
    )
    parser.add_argument(
        '--src', type=Path, default=RAW_ROOT / 'CORN',
        help=f'raw CORN root containing CORN-1/ (default: {RAW_ROOT / "CORN"})',
    )
    parser.add_argument(
        '--out-root', type=Path, default=REPO_ROOT / 'datasets',
        help='root for the prepared role directories (default: datasets)',
    )
    parser.add_argument('--overwrite', action='store_true',
                        help='replace existing images/, masks/ and manifest.csv')
    parser.add_argument('--max-images', type=int, default=None,
                        help='only prepare the first N pairs of each split')
    args = parser.parse_args()

    src = args.src.resolve()
    release = src / 'CORN-1'
    if not release.is_dir():
        release = src  # allow --src .../CORN/CORN-1

    for split, image_sub, label_sub in SPLITS:
        image_dir = require_dir(release / 'images' / image_sub,
                                f'CORN {split} images')
        label_dir = require_dir(release / 'labels' / label_sub,
                                f'CORN {split} labels')
        labels = index_by_stem(label_dir)
        out = prepare_output(
            args.out_root / 'seen' / 'CORN' / split, args.overwrite)
        rows = []
        for image in limit(iter_files(image_dir), args.max_images):
            label = labels.get(image.stem)
            if label is None:
                raise SystemExit(
                    f'CORN {split}: no label for {image.name} in {label_dir}')
            image_dst = out / 'images' / image.name
            mask_dst = out / 'masks' / f'{image.stem}.png'
            copy_image(image, image_dst)
            save_binary_mask(label, mask_dst, threshold=127)
            add_row(rows, image, image_dst, mask_dst)
        finish(out, rows, f'CORN/{split}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
