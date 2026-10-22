#!/usr/bin/env python3
"""Prepare HRF images, vessel masks and FOV masks.

Output: datasets/unseen/HRF/{images,masks,fov,manifest.csv}
"""
from common import (
    IMAGE_EXTS,
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
        'Prepare HRF', 'HRF', 'unseen/HRF'
    ).parse_args()
    src = require_dir(args.src.resolve(), 'HRF')
    image_dir = require_dir(src / 'images', 'HRF images')
    mask_dir = require_dir(src / 'masks', 'HRF vessel masks')
    roi_dir = require_dir(src / 'roi_masks', 'HRF ROI masks')

    images = [
        p for p in sorted(image_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    out = prepare_output(args.out, args.overwrite)
    (out / 'fov').mkdir(exist_ok=True)
    rows = []
    for image in limit(images, args.max_images):
        mask = None
        for ext in ('.tif', '.tiff', '.png', '.gif'):
            candidate = mask_dir / f'{image.stem}{ext}'
            if candidate.exists():
                mask = candidate
                break
        roi = roi_dir / f'{image.stem}_mask.tif'
        if mask is None or not roi.exists():
            continue
        validate_vessel_mask(mask)
        image_dst = out / 'images' / image.name
        mask_dst = out / 'masks' / f'{image.stem}.png'
        fov_dst = out / 'fov' / f'{image.stem}.png'
        copy_image(image, image_dst)
        save_binary_mask(mask, mask_dst)
        save_binary_mask(roi, fov_dst)
        add_row(rows, image, image_dst, mask_dst, fov_dst)
    return finish(out, rows, 'HRF')


if __name__ == '__main__':
    raise SystemExit(main())
