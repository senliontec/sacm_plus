#!/usr/bin/env python3
"""Prepare STARE images and observer masks (default: ah).

Output: datasets/unseen/STARE/{images,masks,manifest.csv}
"""
from common import (
    add_row,
    build_parser,
    finish,
    limit,
    open_image,
    prepare_output,
    require_dir,
    save_binary_mask,
)


def _add_args(parser):
    parser.add_argument(
        '--labels', choices=('ah', 'vk'), default='ah',
        help='STARE observer labels (default: ah)',
    )


def main() -> int:
    args = build_parser(
        'Prepare STARE', 'STARE', 'unseen/STARE', extra=_add_args
    ).parse_args()
    src = require_dir(args.src.resolve(), 'STARE')
    image_dir = next(iter(sorted(src.rglob('stare-images'))), None)
    if image_dir is None:
        raise SystemExit(f'STARE: stare-images not found under {src}')
    label_dir = image_dir.parent / f'labels-{args.labels}'
    if not label_dir.is_dir():
        label_dir = image_dir.parent / 'labels-ah'
    require_dir(label_dir, 'STARE labels')

    images = [
        p for p in sorted(image_dir.iterdir())
        if p.is_file() and (p.name.lower().endswith('.ppm')
                            or p.name.lower().endswith('.ppm.gz'))
    ]
    out = prepare_output(args.out, args.overwrite)
    rows = []
    for image in limit(images, args.max_images):
        key = image.name.split('.')[0]
        mask = None
        for name in (f'{key}.{args.labels}.ppm.gz',
                     f'{key}.{args.labels}.ppm',
                     f'{key}.ppm.gz',
                     f'{key}.ppm'):
            candidate = label_dir / name
            if candidate.exists():
                mask = candidate
                break
        if mask is None:
            continue
        image_dst = out / 'images' / f'{key}.png'
        mask_dst = out / 'masks' / f'{key}.png'
        open_image(image).convert('RGB').save(image_dst)
        save_binary_mask(mask, mask_dst)
        add_row(rows, image, image_dst, mask_dst)
    return finish(out, rows, 'STARE')


if __name__ == '__main__':
    raise SystemExit(main())
