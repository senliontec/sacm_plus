#!/usr/bin/env python3
"""Prepare the labeled XCAD test split.

Output: datasets/unseen/XCAD/{images,masks,manifest.csv}
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
)


def main() -> int:
    args = build_parser(
        'Prepare XCAD test split', 'XCAD', 'unseen/XCAD'
    ).parse_args()
    src = require_dir(args.src.resolve(), 'XCAD')
    test = next(iter(sorted(src.rglob('test'))), None)
    if test is None:
        raise SystemExit(f'XCAD: test directory not found under {src}')
    image_dir = require_dir(test / 'images', 'XCAD test images')
    mask_dir = require_dir(test / 'masks', 'XCAD test masks')

    out = prepare_output(args.out, args.overwrite)
    rows = []
    for image in limit(iter_files(image_dir), args.max_images):
        mask = None
        for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
            candidate = mask_dir / f'{image.stem}{ext}'
            if candidate.exists():
                mask = candidate
                break
        if mask is None:
            continue
        image_dst = out / 'images' / image.name
        mask_dst = out / 'masks' / f'{image.stem}.png'
        copy_image(image, image_dst)
        save_binary_mask(mask, mask_dst)
        add_row(rows, image, image_dst, mask_dst)
    return finish(out, rows, 'XCAD')


if __name__ == '__main__':
    raise SystemExit(main())
