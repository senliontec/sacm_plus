#!/usr/bin/env python
"""Apply a trained Deep Closing shape prior to a directory of masks.

This is a model-agnostic post-processing utility: point it at the PNG
predictions of any baseline or at SACM-X's prediction directory, and it
writes refined masks with the same relative layout.

Example:
    PYTHONPATH=src python scripts/tools/apply_deep_closing.py \
        --input results/paper/samed/seed_0/predictions \
        --output results/paper/samed/seed_0/predictions_deep_closing \
        --checkpoint results/deep_closing/seed0_shape_prior.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from core.postprocess.deep_closing import DeepClosingRefiner  # noqa: E402

SUPPORTED_EXTENSIONS = ('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Refine prediction masks with Deep Closing')
    parser.add_argument('--input', required=True,
                        help='Input mask file or directory')
    parser.add_argument('--output', required=True,
                        help='Output file or directory (same layout)')
    parser.add_argument('--checkpoint', required=True,
                        help='Deep Closing shape-prior checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold on input mask values (default 0.5)')
    parser.add_argument('--dilation_threshold', type=float, default=None,
                        help='Override Deep Dilation threshold (default: '
                             'checkpoint value, usually 0.2)')
    parser.add_argument('--resolution', type=int, default=None,
                        help='Override the shape-prior working resolution')
    parser.add_argument('--max_iters', type=int, default=None,
                        help='Override Simple Point Erosion iteration cap')
    parser.add_argument('--device', default=None,
                        help='cuda/cpu; default = cuda when available')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output masks')
    return parser.parse_args()


def iter_inputs(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    return sorted(
        path for path in source.rglob('*')
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def output_path_for(path: Path, source: Path, destination: Path) -> Path:
    if source.is_file():
        return destination
    return destination / path.relative_to(source)


def main() -> int:
    args = parse_args()
    source = Path(args.input)
    destination = Path(args.output)
    refiner = DeepClosingRefiner.from_checkpoint(
        args.checkpoint,
        device=args.device,
        resolution=args.resolution,
        dilation_threshold=args.dilation_threshold,
        max_iters=args.max_iters,
    )
    files = iter_inputs(source)
    if not files:
        raise SystemExit(f'no masks found under {source}')
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for path in files:
        target = output_path_for(path, source, destination)
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        with Image.open(path) as image:
            mask = np.asarray(image.convert('L')) > 127
        refined = refiner.refine(mask, threshold=args.threshold)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            refined.astype(np.uint8) * 255, mode='L').save(target)
        written += 1
        print(f'{path} -> {target}')
    print(
        f'Deep Closing: wrote {written}, skipped {skipped}, '
        f'total {len(files)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
