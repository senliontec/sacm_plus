#!/usr/bin/env python3
"""Prepare a fixed test subset of the labeled DeepGlobe Road train split.

Output: datasets/novel/DeepGlobe/{images,masks,manifest.csv}

The local valid/ and test/ folders contain satellite images only; their
ground truth is not available.  This script therefore reads train/, which
has 6,226 image/mask pairs, and by default samples a fixed 500-pair subset
with seed 42.  Use --full to keep all 6,226 pairs; --max-images N is treated
as a sample size here (not as first-N slicing) for consistency with the
fixed-subset contract.
"""
import random

from common import (
    add_row,
    build_parser,
    copy_image,
    finish,
    prepare_output,
    require_dir,
    save_binary_mask,
)


def _add_args(parser):
    parser.add_argument('--subset-size', type=int, default=500,
                        help='randomly sample N pairs (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='sampling seed (default: 42)')
    parser.add_argument('--full', action='store_true',
                        help='use all pairs instead of --subset-size')


def main() -> int:
    args = build_parser(
        'Prepare DeepGlobe Road train split', 'DeepGlobe', 'novel/DeepGlobe',
        extra=_add_args,
    ).parse_args()
    src = require_dir(args.src.resolve(), 'DeepGlobe')
    train_dir = require_dir(src / 'train', 'DeepGlobe train')

    images = sorted(train_dir.glob('*_sat.jpg'))
    pairs = []
    missing = []
    for image in images:
        base = image.name[:-len('_sat.jpg')]
        mask = train_dir / f'{base}_mask.png'
        if mask.exists():
            pairs.append((image, mask))
        else:
            missing.append(str(mask))
    if missing:
        raise SystemExit(
            f'DeepGlobe: {len(missing)} sampled/source images have no mask; '
            f'first={missing[0]}')
    if args.full:
        if args.max_images is not None:
            raise SystemExit('use either --full or --max-images, not both')
        selected = pairs
    else:
        size = (args.max_images if args.max_images is not None
                else args.subset_size)
        if size <= 0:
            raise SystemExit('--subset-size/--max-images must be positive')
        if size > len(pairs):
            raise SystemExit(
                f'requested {size} pairs but only {len(pairs)} have masks')
        # Fixed random subset: reproducible and not selected by model output.
        selected = sorted(random.Random(args.seed).sample(pairs, size),
                          key=lambda pair: pair[0].name)

    out = prepare_output(args.out, args.overwrite)
    rows = []
    for image, mask in selected:
        base = image.name[:-len('_sat.jpg')]
        image_dst = out / 'images' / f'{base}.jpg'
        mask_dst = out / 'masks' / f'{base}.png'
        copy_image(image, image_dst)
        save_binary_mask(mask, mask_dst)
        add_row(rows, image, image_dst, mask_dst)
    if len(rows) != len(selected):
        raise SystemExit(
            f'DeepGlobe: wrote {len(rows)} pairs, expected {len(selected)}')
    return finish(out, rows, 'DeepGlobe/train')


if __name__ == '__main__':
    raise SystemExit(main())
