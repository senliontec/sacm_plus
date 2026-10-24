#!/usr/bin/env python3
"""Overlay SACM-X predictions and GT on original images for the worst cases."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.ppm')


def read_csv(path: Path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def find_by_name(directory: Path, name: str):
    direct = directory / name
    if direct.is_file():
        return direct
    stem = Path(name).stem
    for ext in IMAGE_EXTS:
        candidate = directory / f'{stem}{ext}'
        if candidate.is_file():
            return candidate
    return None


def read_mask(path: Path, size):
    image = Image.open(path).convert('L')
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def read_rgb(path: Path):
    return np.asarray(Image.open(path).convert('RGB'))


def overlay(image, mask, color, alpha=0.45):
    out = image.astype(np.float32).copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def error_map(image, gt, pred):
    gray = np.asarray(Image.fromarray(image).convert('L')).astype(np.float32)
    base = np.stack([gray, gray, gray], axis=-1) * 0.45
    tp = gt & pred
    fp = pred & ~gt
    fn = gt & ~pred
    base[tp] = [0, 200, 0]
    base[fp] = [230, 0, 0]
    base[fn] = [0, 80, 230]
    return np.clip(base, 0, 255).astype(np.uint8)


def panel(axes, image, gt, pred, row):
    axes[0].imshow(image)
    axes[0].set_title('original')
    axes[1].imshow(overlay(image, gt, (0, 220, 0)))
    axes[1].set_title('GT (green)')
    axes[2].imshow(overlay(image, pred, (230, 0, 0)))
    axes[2].set_title('prediction (red)')
    axes[3].imshow(error_map(image, gt, pred))
    axes[3].set_title('TP green / FP red / FN blue')
    for ax in axes:
        ax.axis('off')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--eval-root', type=Path,
                        default=Path('results/sacm_x_official_200'))
    parser.add_argument('--preset', default='full')
    parser.add_argument('--protocol', type=Path,
                        default=Path('data/sacm_3shot/protocol_manifest.csv'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--top-n', type=int, default=8)
    args = parser.parse_args()

    eval_dir = args.eval_root / args.preset / args.dataset
    metrics_path = eval_dir / 'metrics.csv'
    if not metrics_path.is_file():
        raise SystemExit(
            f'missing {metrics_path}; wait until this dataset finishes evaluation')

    root = None
    with open(args.protocol) as f:
        for row in csv.DictReader(f):
            if row['dataset'] == args.dataset and row['split'].endswith('_test'):
                mask = Path(row['mask_path'])
                if not mask.is_absolute():
                    mask = Path.cwd() / mask
                root = mask.parent.parent
                break
    if root is None:
        raise SystemExit(f'no test rows for {args.dataset} in {args.protocol}')
    image_dir, mask_dir = root / 'images', root / 'masks'

    rows = []
    for row in read_csv(metrics_path):
        name = row.get('image', '')
        pred = eval_dir / f'{name}_result.png'
        if not pred.is_file():
            pred = eval_dir / f'{Path(name).stem}_result.png'
        if not pred.is_file():
            continue
        try:
            dice = float(row.get('dice', 'nan'))
        except ValueError:
            dice = float('nan')
        rows.append((dice, row, pred))
    if not rows:
        raise SystemExit(f'no prediction PNGs in {eval_dir}')
    rows.sort(key=lambda item: (not math.isfinite(item[0]), item[0]))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = rows[:max(1, args.top_n)]
    manifest = []

    for idx, (dice, row, pred_path) in enumerate(selected, 1):
        name = row.get('image', '')
        image_path = find_by_name(image_dir, name)
        gt_path = find_by_name(mask_dir, name)
        if image_path is None or gt_path is None:
            print(f'[warn] missing image/GT for {name}; skip')
            continue
        image = read_rgb(image_path)
        size = (image.shape[1], image.shape[0])
        gt = read_mask(gt_path, size)
        pred = read_mask(pred_path, size)

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        panel(axes, image, gt, pred, idx)
        cldice = row.get('cldice', 'nan')
        hd95 = row.get('hd95', 'nan')
        fig.suptitle(
            f'{args.dataset} | {name} | Dice={dice:.4f} '
            f'clDice={cldice} HD95={hd95}', fontsize=12)
        fig.tight_layout()
        stem = Path(name).stem
        save_path = out_dir / f'{idx:02d}_{stem}_viz.png'
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        manifest.append({
            'image': name,
            'dice': dice,
            'cldice': cldice,
            'hd95': hd95,
            'visualization': str(save_path),
            'prediction': str(pred_path),
        })

    if not manifest:
        raise SystemExit('nothing visualized')

    # Contact sheet: one row per selected image, same four columns.
    n = len(manifest)
    fig, axes = plt.subplots(n, 4, figsize=(18, 4.5 * n))
    if n == 1:
        axes = np.asarray(axes).reshape(1, 4)
    for i, item in enumerate(manifest):
        name = item['image']
        image = read_rgb(find_by_name(image_dir, name))
        size = (image.shape[1], image.shape[0])
        gt = read_mask(find_by_name(mask_dir, name), size)
        pred = read_mask(Path(item['prediction']), size)
        panel(axes[i], image, gt, pred, i)
        axes[i][0].set_ylabel(
            f"{name}\nDice={float(item['dice']):.3f}", fontsize=9)
    fig.suptitle(f'{args.dataset}: worst {n} by Dice', fontsize=14)
    fig.tight_layout()
    contact = out_dir / f'worst_{n}_contact.png'
    fig.savefig(contact, dpi=100, bbox_inches='tight')
    plt.close(fig)

    manifest_path = out_dir / 'worst_manifest.csv'
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    print(f'individual visualizations: {out_dir}')
    print(f'contact sheet: {contact}')
    print(f'manifest: {manifest_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
