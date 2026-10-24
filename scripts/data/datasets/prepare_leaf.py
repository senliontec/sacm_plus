#!/usr/bin/env python3
"""Prepare the author-provided LEAF set for the SACM Novel protocol.

Source layout (segment/datasets/leaf)::

    images/leaf_*.png    31 RGB images
    masks/leaf_*.png     31 RGB-encoded (R=G=B) masks

Output::

    datasets/novel/LEAF/{images,masks,manifest.csv}

Masks are binarized at >127 and written as single-channel 0/255 PNG, matching
the archived official copy.  LEAF is evaluation-only: it is part of the SACM
Novel set and is never used for training.

Note: LEAF annotations are thin vein tracings with many 1-2 px breaks
(median 305 8-connected components per image).  This is a property of the
official labels, not of this converter; keep the official masks and report
topology metrics with that caveat.
"""

from __future__ import annotations

from common import build_parser, prepare_ready


def main() -> int:
    args = build_parser(
        'Prepare the official LEAF Novel set', 'leaf', 'novel/LEAF'
    ).parse_args()
    return prepare_ready(
        args.src.resolve(), args.out, args.overwrite, args.max_images, 'LEAF'
    )


if __name__ == '__main__':
    raise SystemExit(main())
