"""Build the few-shot training split for the curvilinear protocol.

Layout expected by SegmentationDataset (train_sam.py):

    <out_dir>/train/images/*  +  <out_dir>/train/masks/*
    <out_dir>/val/images/*    +  <out_dir>/val/masks/*

Each source dataset directory must contain images/ and masks/ subdirs
(the Curvilinear_Structure_Datasets layout). Test datasets keep their own
layout and are passed directly to test_sam.py --data_root.

The split is recorded in <out_dir>/split_manifest.csv for reproducibility.

Usage:
    python prepare_data.py \
        --train_dirs <d1> <d2> <d3> <d4> <d5> <d6> \
        --shots 3 --val_shots 1 \
        --out_dir data/sacm_3shot --seed 42
"""

import argparse
import csv
import os
import random
import shutil

SUPPORTED = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def find_pairs(src_dir):
    """List (image_path, mask_path) pairs in a dataset directory."""
    image_dir = os.path.join(src_dir, 'images')
    mask_dir = os.path.join(src_dir, 'masks')
    pairs = []
    for fn in sorted(os.listdir(image_dir)):
        if fn.lower().endswith(SUPPORTED):
            base = os.path.splitext(fn)[0]
            for ext in SUPPORTED:
                mask_path = os.path.join(mask_dir, base + ext)
                if os.path.exists(mask_path):
                    pairs.append((os.path.join(image_dir, fn), mask_path))
                    break
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Build the few-shot (or full) training split')
    parser.add_argument('--train_dirs', nargs='+', required=True,
                        help='Source dataset dirs (each with images/ and masks/); '
                             '3-shot per dataset is sampled from the first 6')
    parser.add_argument('--shots', type=int, default=3, help='Train images per dataset (few-shot mode)')
    parser.add_argument('--val_shots', type=int, default=1, help='Val images per dataset (few-shot mode)')
    parser.add_argument('--use_all', action='store_true',
                        help='全量模式(架构研究层): 不做 few-shot 采样, 按 --val_ratio 切 train/val')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='全量模式下验证集比例(默认 0.2)')
    parser.add_argument('--out_dir', type=str, required=True, help='Output root')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    for split in ('train', 'val'):
        os.makedirs(os.path.join(args.out_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, split, 'masks'), exist_ok=True)

    manifest = []
    for src in args.train_dirs:
        name = os.path.basename(os.path.normpath(src))
        pairs = find_pairs(src)
        if not pairs:
            raise ValueError(f"No image-mask pairs found in {src}")
        random.shuffle(pairs)
        if args.use_all:
            n_val = max(1, int(len(pairs) * args.val_ratio))
            subsets = [
                ('train', pairs[n_val:]),
                ('val', pairs[:n_val]),
            ]
        else:
            if len(pairs) < args.shots + args.val_shots:
                raise ValueError(
                    f"{name}: only {len(pairs)} pairs available, "
                    f"need at least {args.shots + args.val_shots}"
                )
            subsets = [
                ('train', pairs[:args.shots]),
                ('val', pairs[args.shots:args.shots + args.val_shots]),
            ]
        for split, subset in subsets:
            for img_path, mask_path in subset:
                img_ext = os.path.splitext(img_path)[1]
                mask_ext = os.path.splitext(mask_path)[1]
                # Prefix the dataset name to avoid collisions across sources
                new_base = f"{name}_{os.path.splitext(os.path.basename(img_path))[0]}"
                new_img = os.path.join(args.out_dir, split, 'images', new_base + img_ext)
                new_mask = os.path.join(args.out_dir, split, 'masks', new_base + mask_ext)
                shutil.copy2(img_path, new_img)
                shutil.copy2(mask_path, new_mask)
                manifest.append({
                    'dataset': name,
                    'split': split,
                    'source_image': os.path.basename(img_path),
                    'output': new_base,
                })
                print(f"[{split}] {name}: {new_base}")

    manifest_path = os.path.join(args.out_dir, 'split_manifest.csv')
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'split', 'source_image', 'output'])
        writer.writeheader()
        writer.writerows(manifest)

    counts = {}
    for m in manifest:
        counts[m['split']] = counts.get(m['split'], 0) + 1
    print(f"Done: {counts.get('train', 0)} train / {counts.get('val', 0)} val images -> {args.out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
