#!/usr/bin/env python3
"""Export every SACM-X prediction as a binary mask and an image overlay."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.ppm')


def load_rows(path: Path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def resolve(path: str, repo: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo / p


def find_result(eval_dir: Path, name: str):
    direct = eval_dir / f'{name}_result.png'
    if direct.is_file():
        return direct
    stem = Path(name).stem
    for candidate in (eval_dir / f'{stem}_result.png',
                      eval_dir / f'{stem}.png'):
        if candidate.is_file():
            return candidate
    return None


def binary_mask(pred_path: Path, size):
    pred = Image.open(pred_path).convert('L')
    if pred.size != size:
        pred = pred.resize(size, Image.Resampling.NEAREST)
    return (np.asarray(pred) > 127)


def overlay(image: np.ndarray, mask: np.ndarray):
    out = image.astype(np.float32).copy()
    red = np.asarray([255, 30, 30], dtype=np.float32)
    out[mask] = 0.55 * out[mask] + 0.45 * red
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-root', type=Path,
                        default=Path('results/sacm_x_official_200'))
    parser.add_argument('--preset', default='full')
    parser.add_argument('--protocol', type=Path,
                        default=Path('data/sacm_3shot/protocol_manifest.csv'))
    parser.add_argument('--out-root', type=Path,
                        default=Path('results/visualizations/sacm_x_all'))
    parser.add_argument('--datasets', nargs='*', default=None)
    args = parser.parse_args()

    repo = Path.cwd()
    protocol = load_rows(args.protocol)
    selected = set(args.datasets) if args.datasets else None
    datasets = sorted({r['dataset'] for r in protocol
                       if r['split'].endswith('_test')
                       and (selected is None or r['dataset'] in selected)})
    if not datasets:
        raise SystemExit('no datasets selected')

    for dataset in datasets:
        eval_dir = args.eval_root / args.preset / dataset
        out_dir = args.out_root / dataset
        mask_dir = out_dir / 'masks'
        overlay_dir = out_dir / 'overlay'
        mask_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        metrics = {}
        metrics_path = eval_dir / 'metrics.csv'
        if metrics_path.is_file():
            for row in load_rows(metrics_path):
                metrics[row.get('image', '')] = row

        rows = [r for r in protocol
                if r['dataset'] == dataset and r['split'].endswith('_test')]
        done = missing = 0
        manifest = []
        for row in rows:
            image_path = resolve(row['image_path'], repo)
            name = image_path.name
            pred_path = find_result(eval_dir, name)
            if pred_path is None or not image_path.is_file():
                missing += 1
                continue
            image = np.asarray(Image.open(image_path).convert('RGB'))
            size = (image.shape[1], image.shape[0])
            mask = binary_mask(pred_path, size)
            stem = Path(name).stem
            mask_out = mask_dir / f'{stem}.png'
            overlay_out = overlay_dir / f'{stem}.png'
            Image.fromarray((mask.astype(np.uint8) * 255), mode='L').save(mask_out)
            Image.fromarray(overlay(image, mask)).save(overlay_out)
            mrow = metrics.get(name) or metrics.get(Path(name).stem, {})
            manifest.append({
                'dataset': dataset,
                'role': row.get('role', ''),
                'image': name,
                'image_path': str(image_path),
                'prediction': str(pred_path),
                'mask': str(mask_out),
                'overlay': str(overlay_out),
                'dice': mrow.get('dice', ''),
                'cldice': mrow.get('cldice', ''),
                'hd95': mrow.get('hd95', ''),
            })
            done += 1

        if manifest:
            manifest_path = out_dir / 'manifest.csv'
            with open(manifest_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
                writer.writeheader()
                writer.writerows(manifest)
        print(f'{dataset}: exported={done} missing={missing} out={out_dir}',
              flush=True)

    print(f'\nall visualizations: {args.out_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
