#!/usr/bin/env python3
"""Prepare CHASEDB1 with the first-observer (1stHO) masks.

Output: datasets/seen/CHASEDB1/{images,masks,manifest.csv}
"""
from common import (
    IMAGE_EXTS,
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
        'Prepare CHASEDB1 (1stHO)', 'CHASEDB1', 'seen/CHASEDB1'
    ).parse_args()
    src = require_dir(args.src.resolve(), 'CHASEDB1')
    first = next(iter(sorted(src.rglob('*_1stHO.png'))), None)
    if first is None:
        raise SystemExit(f'CHASEDB1: no *_1stHO.png found under {src}')
    root = first.parent
    masks = {p.stem.replace('_1stHO', ''): p for p in root.glob('*_1stHO.png')}
    images = [
        p for p in sorted(root.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        and '_1stHO' not in p.name and '_2ndHO' not in p.name
    ]

    out = prepare_output(args.out, args.overwrite)
    rows = []
    for image in limit(images, args.max_images):
        mask = masks.get(image.stem)
        if mask is None:
            continue
        image_dst = out / 'images' / f'{image.stem}.png'
        mask_dst = out / 'masks' / f'{image.stem}.png'
        save_rgb_png(image, image_dst)
        save_binary_mask(mask, mask_dst)
        add_row(rows, image, image_dst, mask_dst)
    return finish(out, rows, 'CHASEDB1')


if __name__ == '__main__':
    raise SystemExit(main())
