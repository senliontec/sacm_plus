#!/usr/bin/env python
"""Apply SAMRefiner to a directory of binary prediction masks.

SAMRefiner needs the source image as well as the coarse mask.  For a
prediction tree (``<dataset>/<stem>.png``), pass the protocol manifest so
the original image can be resolved by ``(dataset, stem)``.

Example:
    PYTHONPATH=src python scripts/tools/apply_sam_refiner.py \
        --input results/paper/unet/seed_0/predictions \
        --output results/paper/unet/seed_0/predictions_samrefiner \
        --manifest data/sacm_3shot_seed0/protocol_manifest.csv \
        --checkpoint checkpoints/sam_vit_h_4b8939.pth --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from core.postprocess.sam_refiner import SAMRefiner  # noqa: E402

SUPPORTED_EXTENSIONS = ('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Refine binary prediction masks with SAMRefiner')
    parser.add_argument('--input', required=True,
                        help='Input mask file or directory')
    parser.add_argument('--output', required=True,
                        help='Output file or directory (same layout)')
    parser.add_argument('--checkpoint', required=True,
                        help='SAM or HQ-SAM checkpoint')
    parser.add_argument('--manifest', default=None,
                        help='protocol_manifest.csv with dataset/image_path '
                             '(required for a prediction tree)')
    parser.add_argument('--image', default=None,
                        help='Source image for a single input mask')
    parser.add_argument('--model-type', default='vit_h',
                        choices=['vit_h', 'vit_l', 'vit_b'])
    parser.add_argument('--device', default=None,
                        help='cuda/cpu; default = cuda when available')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold on input mask values (default 0.5)')
    parser.add_argument('--iterations', type=int, default=5,
                        help='SAMRefiner iterative refinement passes')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output masks')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process at most N masks (smoke tests)')
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


def load_manifest(path: Path | None) -> dict[tuple[str, str], Path]:
    if path is None:
        return {}
    mapping: dict[tuple[str, str], Path] = {}
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            image = Path(row['image_path'])
            if not image.is_absolute():
                image = REPO_ROOT / image
            mapping[(row['dataset'], Path(row['image_path']).stem)] = image
            mapping[(row['dataset'], Path(row['mask_path']).stem)] = image
    return mapping


def resolve_image(path: Path, source: Path,
                  manifest: dict[tuple[str, str], Path],
                  explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if source.is_file():
        raise SystemExit('--image is required for a single input mask')
    relative = path.relative_to(source)
    if len(relative.parts) < 2:
        raise SystemExit(
            f'cannot infer dataset/stem from {path}; expected '
            '<dataset>/<stem>.<ext>')
    dataset, stem = relative.parts[-2], path.stem
    image = manifest.get((dataset, stem))
    if image is None or not image.is_file():
        raise SystemExit(
            f'no source image for {dataset}/{stem}; pass --manifest with '
            'matching image_path/mask_path entries')
    return image


def main() -> int:
    args = parse_args()
    source = Path(args.input)
    destination = Path(args.output)
    files = iter_inputs(source)
    if args.limit is not None:
        files = files[:max(0, args.limit)]
    if not files:
        raise SystemExit(f'no masks found under {source}')
    manifest = load_manifest(Path(args.manifest) if args.manifest else None)
    device = args.device
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    refiner = SAMRefiner(
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        device=device,
        iterations=args.iterations,
    )
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
        image_path = resolve_image(path, source, manifest, args.image)
        with Image.open(path) as mask_image:
            mask = np.asarray(mask_image.convert('L')) > int(
                args.threshold * 255)
        refined = refiner.refine(image_path, mask)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(refined.astype(np.uint8) * 255, mode='L').save(target)
        written += 1
        print(f'{path} -> {target}')
    print(f'SAMRefiner: wrote {written}, skipped {skipped}, '
          f'total {len(files)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
