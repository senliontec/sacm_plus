#!/usr/bin/env python3
"""Prepare the in-house WIRE images and masks.

Output: datasets/novel/WIRE/{images,masks,manifest.csv}
"""
from common import build_parser, prepare_ready


def main() -> int:
    args = build_parser(
        'Prepare WIRE', 'wire', 'novel/WIRE'
    ).parse_args()
    return prepare_ready(
        args.src.resolve(), args.out, args.overwrite,
        args.max_images, 'WIRE',
    )


if __name__ == '__main__':
    raise SystemExit(main())
