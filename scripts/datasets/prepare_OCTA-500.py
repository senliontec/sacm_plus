#!/usr/bin/env python3
"""Prepare the 2D OCTA-500 projections and vessel labels.

Output: datasets/unseen/OCTA-500/{images,masks,manifest.csv}

Only subjects that have both a 2D projection in OCTA(ILM_OPL) and a label
are written.  The default target is GT_LargeVessel; 300 of the 500 labels
currently have a matching 2D projection in the local copy.
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
        '--label',
        choices=('GT_LargeVessel', 'GT_Artery', 'GT_Vein', 'GT_Capillary'),
        default='GT_LargeVessel',
        help='OCTA-500 label channel (default: GT_LargeVessel)',
    )


def main() -> int:
    args = build_parser(
        'Prepare OCTA-500', 'OCTA-500', 'unseen/OCTA-500', extra=_add_args
    ).parse_args()
    src = require_dir(args.src.resolve(), 'OCTA-500')
    projection_dir = require_dir(src / 'OCTA(ILM_OPL)', 'OCTA-500 projections')
    label_dir = require_dir(
        src / 'Label' / 'Label' / args.label, f'OCTA-500 {args.label}'
    )

    out = prepare_output(args.out, args.overwrite)
    rows = []
    for image in limit(iter_files(projection_dir), args.max_images):
        mask = label_dir / f'{image.stem}.bmp'
        if not mask.exists():
            continue
        image_dst = out / 'images' / image.name
        mask_dst = out / 'masks' / f'{image.stem}.png'
        copy_image(image, image_dst)
        save_binary_mask(mask, mask_dst)
        add_row(rows, image, image_dst, mask_dst)
    return finish(out, rows, f'OCTA-500/{args.label}')


if __name__ == '__main__':
    raise SystemExit(main())
