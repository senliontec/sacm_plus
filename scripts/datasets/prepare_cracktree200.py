#!/usr/bin/env python3
"""Prepare CrackTree200 RGB images and crack masks.

Output: datasets/seen/CrackTree200/{images,masks,manifest.csv}
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
        'Prepare CrackTree200', 'cracktree200', 'seen/CrackTree200'
    ).parse_args()
    src = require_dir(args.src.resolve(), 'cracktree200')
    gt = next(iter(sorted(src.rglob('cracktree200_gt'))), None)
    if gt is None:
        raise SystemExit(f'CrackTree200: cracktree200_gt not found under {src}')
    root = gt.parent
    image_dir = root / 'cracktree200rgb'
    if not image_dir.is_dir():
        image_dir = root / 'cracktree200gray'
    require_dir(image_dir, 'CrackTree200 images')

    out = prepare_output(args.out, args.overwrite)
    rows = []
    for image in limit(iter_files(image_dir), args.max_images):
        mask = gt / f'{image.stem}.png'
        if not mask.exists():
            continue
        image_dst = out / 'images' / image.name
        mask_dst = out / 'masks' / f'{image.stem}.png'
        copy_image(image, image_dst)
        save_binary_mask(mask, mask_dst)
        add_row(rows, image, image_dst, mask_dst)
    return finish(out, rows, 'CrackTree200')


if __name__ == '__main__':
    raise SystemExit(main())
