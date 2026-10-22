#!/usr/bin/env python3
"""Prepare the in-house TYRE images and masks.

Output: datasets/novel/TYRE/{images,masks,manifest.csv}
"""
from common import build_parser, prepare_ready


def main() -> int:
    args = build_parser(
        'Prepare TYRE', 'tyre', 'novel/TYRE'
    ).parse_args()
    return prepare_ready(
        args.src.resolve(), args.out, args.overwrite,
        args.max_images, 'TYRE',
    )


if __name__ == '__main__':
    raise SystemExit(main())
