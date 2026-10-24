#!/usr/bin/env python3
"""Prepare the FIVES test (or train) split.

Output: datasets/unseen/FIVES/{images,masks,manifest.csv}
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
        help='FIVES split to prepare (default: test)',
    )


def main() -> int:
    args = build_parser(
        'Prepare FIVES', 'FIVES', 'unseen/FIVES', extra=_add_args
    ).parse_args()
    src = require_dir(args.src.resolve(), 'FIVES')
    original = next(iter(sorted(src.rglob(f'{args.split}/Original'))), None)
    if original is None:
        raise SystemExit(f'FIVES: {args.split}/Original not found under {src}')
    ground_truth = original.parent / 'Ground truth'
    require_dir(ground_truth, 'FIVES Ground truth')

    out = prepare_output(args.out, args.overwrite)
    rows = []
    for image in limit(iter_files(original), args.max_images):
        mask = ground_truth / image.name
        if not mask.exists():
            continue
        image_dst = out / 'images' / image.name
        mask_dst = out / 'masks' / f'{image.stem}.png'
        copy_image(image, image_dst)
        save_binary_mask(mask, mask_dst)
        add_row(rows, image, image_dst, mask_dst)
    return finish(out, rows, f'FIVES/{args.split}')


if __name__ == '__main__':
    raise SystemExit(main())
