#!/usr/bin/env python3
"""Prepare DCA1 coronary angiograms.

Output: datasets/seen/DCA1/{images,masks,manifest.csv}
"""
from common import (
    add_row,
    build_parser,
    finish,
    limit,
    prepare_output,
    require_dir,
    save_binary_mask,
    save_rgb_png,
)


def main() -> int:
    args = build_parser(
        'Prepare DCA1', 'DCA1', 'seen/DCA1'
    ).parse_args()
    src = require_dir(args.src.resolve(), 'DCA1')
    gt = next(iter(sorted(src.rglob('*_gt.pgm'))), None)
    if gt is None:
        raise SystemExit(f'DCA1: no *_gt.pgm found under {src}')
    root = gt.parent

    out = prepare_output(args.out, args.overwrite)
    rows = []
    gts = limit(sorted(root.glob('*_gt.pgm')), args.max_images)
    for mask in gts:
        idx = mask.name[:-len('_gt.pgm')]
        image = root / f'{idx}.pgm'
        if not image.exists():
            continue
        image_dst = out / 'images' / f'{idx}.png'
        mask_dst = out / 'masks' / f'{idx}.png'
        save_rgb_png(image, image_dst)
        save_binary_mask(mask, mask_dst)
        add_row(rows, image, image_dst, mask_dst)
    return finish(out, rows, 'DCA1')


if __name__ == '__main__':
    raise SystemExit(main())
