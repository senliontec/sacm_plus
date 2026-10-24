#!/usr/bin/env python3
"""Prepare DRIVE training images and vessel masks.

Output: datasets/seen/DRIVE/{images,masks,fov,manifest.csv}

The official DRIVE test/mask is a FOV mask, not vessel GT.  This script only
converts the training split, which is what the available-data protocol uses.
"""
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
    validate_vessel_mask,
)


def main() -> int:
    args = build_parser(
        'Prepare DRIVE training split', 'DRIVE', 'seen/DRIVE'
    ).parse_args()
    src = args.src.resolve()
    img_dir = require_dir(src / 'training' / 'images', 'DRIVE training images')
    mask_dir = require_dir(src / 'training' / '1st_manual', 'DRIVE 1st_manual')
    fov_dir = src / 'training' / 'mask'

    out = prepare_output(args.out, args.overwrite)
    (out / 'fov').mkdir(exist_ok=True)
    rows = []
    for image in limit(iter_files(img_dir), args.max_images):
        base = image.stem
        idx = base.split('_')[0]
        mask = mask_dir / f'{idx}_manual1.gif'
        if not mask.exists():
            continue
        validate_vessel_mask(mask)
        image_dst = out / 'images' / f'{base}{image.suffix}'
        mask_dst = out / 'masks' / f'{base}.png'
        copy_image(image, image_dst)
        save_binary_mask(mask, mask_dst)

        fov = fov_dir / f'{base}_mask.gif'
        fov_dst = None
        if fov.exists():
            fov_dst = out / 'fov' / f'{base}.png'
            save_binary_mask(fov, fov_dst)
        add_row(rows, image, image_dst, mask_dst, fov_dst)
    return finish(out, rows, 'DRIVE')


if __name__ == '__main__':
    raise SystemExit(main())
