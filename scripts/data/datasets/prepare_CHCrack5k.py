#!/usr/bin/env python3
"""Prepare the CHCrack5k test split, skipping empty GT masks.

Output: datasets/unseen/CHCrack5k/{images,masks,manifest.csv}
"""
import numpy as np
from common import (
    add_row,
    build_parser,
    copy_image,
    finish,
    iter_files,
    limit,
    prepare_output,
    require_dir,
    save_binary_mask,
)
from PIL import Image


def main() -> int:
    args = build_parser(
        'Prepare CHCrack5k test split', 'CHCrack5k', 'unseen/CHCrack5k'
    ).parse_args()
    src = require_dir(args.src.resolve(), 'CHCrack5k')
    image_dir = require_dir(src / 'test' / 'image', 'CHCrack5k test images')
    gt_dir = require_dir(src / 'test' / 'gt', 'CHCrack5k test GT')

    out = prepare_output(args.out, args.overwrite)
    rows = []
    for image in limit(iter_files(image_dir), args.max_images):
        mask = None
        for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
            candidate = gt_dir / f'{image.stem}{ext}'
            if candidate.exists():
                mask = candidate
                break
        if mask is None:
            continue
        arr = np.array(Image.open(mask).convert('L'))
        if not np.any(arr > 0):
            continue
        image_dst = out / 'images' / image.name
        mask_dst = out / 'masks' / f'{image.stem}.png'
        copy_image(image, image_dst)
        save_binary_mask(mask, mask_dst)
        add_row(rows, image, image_dst, mask_dst)
    return finish(out, rows, 'CHCrack5k')


if __name__ == '__main__':
    raise SystemExit(main())
