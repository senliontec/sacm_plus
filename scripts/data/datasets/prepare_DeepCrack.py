#!/usr/bin/env python3
"""Prepare the DeepCrack test (or train) split.

Output: datasets/unseen/DeepCrack/{images,masks,manifest.csv}
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
        '--split', choices=('test', 'train'), default='test',
        help='DeepCrack split to prepare (default: test)',
    )


def main() -> int:
    args = build_parser(
        'Prepare DeepCrack', 'DeepCrack', 'unseen/DeepCrack', extra=_add_args
    ).parse_args()
    src = require_dir(args.src.resolve(), 'DeepCrack')
    root = next(iter(sorted(src.rglob(f'{args.split}_img'))), None)
    if root is None:
        raise SystemExit(f'DeepCrack: {args.split}_img not found under {src}')
    image_dir = root
    mask_dir = root.parent / f'{args.split}_lab'
    require_dir(mask_dir, 'DeepCrack labels')

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
    return finish(out, rows, f'DeepCrack/{args.split}')


if __name__ == '__main__':
    raise SystemExit(main())
