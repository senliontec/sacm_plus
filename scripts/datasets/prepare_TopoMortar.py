#!/usr/bin/env python3
"""Prepare TopoMortar images and accurate mortar-network masks.

Output: datasets/novel/TopoMortar/{images,masks,manifest.csv}
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


def _add_args(parser):
    parser.add_argument(
        '--split', choices=('train', 'val', 'test'), default='test',
        help='TopoMortar split to prepare (default: test)',
    )


def main() -> int:
    args = build_parser(
        'Prepare TopoMortar', 'TopoMortar', 'novel/TopoMortar',
        extra=_add_args,
    ).parse_args()
    src = require_dir(args.src.resolve(), 'TopoMortar')
    split_dir = src / args.split
    image_dir = require_dir(split_dir / 'images', f'TopoMortar {args.split} images')
    mask_dir = require_dir(split_dir / 'accurate', f'TopoMortar {args.split} masks')

    out = prepare_output(args.out, args.overwrite)
    rows = []
    for image in limit(iter_files(image_dir), args.max_images):
        mask = mask_dir / f'{image.stem}.png'
        if not mask.exists():
            continue
        image_dst = out / 'images' / image.name
        mask_dst = out / 'masks' / f'{image.stem}.png'
        copy_image(image, image_dst)
        save_binary_mask(mask, mask_dst)
        add_row(rows, image, image_dst, mask_dst)
    return finish(out, rows, f'TopoMortar/{args.split}')


if __name__ == '__main__':
    raise SystemExit(main())
