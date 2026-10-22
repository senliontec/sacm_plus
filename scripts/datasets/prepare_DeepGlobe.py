#!/usr/bin/env python3
"""Prepare the labeled DeepGlobe Road training split.

Output: datasets/unseen/DeepGlobe/{images,masks,manifest.csv}

The local valid/ and test/ folders contain satellite images only; their
ground truth is not available.  This script therefore reads train/, which
has 6,226 image/mask pairs, and can optionally cap the number of pairs with
--max-images.
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
)


def main() -> int:
    args = build_parser(
        'Prepare DeepGlobe Road train split', 'DeepGlobe', 'unseen/DeepGlobe'
    ).parse_args()
    src = require_dir(args.src.resolve(), 'DeepGlobe')
    train_dir = require_dir(src / 'train', 'DeepGlobe train')

    images = sorted(train_dir.glob('*_sat.jpg'))
    out = prepare_output(args.out, args.overwrite)
    rows = []
    for image in limit(images, args.max_images):
        base = image.name[:-len('_sat.jpg')]
        mask = train_dir / f'{base}_mask.png'
        if not mask.exists():
            continue
        image_dst = out / 'images' / f'{base}.jpg'
        mask_dst = out / 'masks' / f'{base}.png'
        copy_image(image, image_dst)
        save_binary_mask(mask, mask_dst)
        add_row(rows, image, image_dst, mask_dst)
    return finish(out, rows, 'DeepGlobe/train')


if __name__ == '__main__':
    raise SystemExit(main())
