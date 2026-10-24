#!/usr/bin/env python3
"""Prepare LES-AV images and vessel masks.

Output: datasets/unseen/LES-AV/{images,masks,manifest.csv}
"""
from common import (
    add_row,
    build_parser,
    copy_image,
    finish,
    limit,
    prepare_output,
    require_dir,
    save_binary_mask,
    validate_vessel_mask,
)


def main() -> int:
    args = build_parser(
        'Prepare LES-AV', 'LES-AV', 'unseen/LES-AV'
    ).parse_args()
    src = require_dir(args.src.resolve(), 'LES-AV')
    image_dir = require_dir(src / 'images', 'LES-AV images')
    mask_dir = require_dir(src / 'vessel-segmentations', 'LES-AV vessel masks')

    images = [
        p for p in sorted(image_dir.iterdir())
        if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    ]
    out = prepare_output(args.out, args.overwrite)
    rows = []
    for image in limit(images, args.max_images):
        mask = mask_dir / f'{image.stem}.png'
        if not mask.exists():
            continue
        validate_vessel_mask(mask)
        image_dst = out / 'images' / image.name
        mask_dst = out / 'masks' / f'{image.stem}.png'
        copy_image(image, image_dst)
        save_binary_mask(mask, mask_dst)
        add_row(rows, image, image_dst, mask_dst)
    return finish(out, rows, 'LES-AV')


if __name__ == '__main__':
    raise SystemExit(main())
