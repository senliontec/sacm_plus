#!/usr/bin/env python3
"""Prepare the ROAD (Massachusetts Roads) test split.

Output: datasets/novel/ROAD/{images,masks,manifest.csv}
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
        'Prepare ROAD test split', 'ROAD', 'novel/ROAD'
    ).parse_args()
    src = require_dir(args.src.resolve(), 'ROAD')
    test = next(iter(sorted(src.rglob('tiff/test'))), None)
    if test is None:
        raise SystemExit(f'ROAD: tiff/test not found under {src}')
    label_dir = test.parent / 'test_labels'
    require_dir(label_dir, 'ROAD test labels')

    out = prepare_output(args.out, args.overwrite)
    rows = []
    for image in limit(iter_files(test), args.max_images):
        mask = None
        for ext in ('.tif', '.tiff', '.png'):
            candidate = label_dir / f'{image.stem}{ext}'
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
    return finish(out, rows, 'ROAD')


if __name__ == '__main__':
    raise SystemExit(main())
