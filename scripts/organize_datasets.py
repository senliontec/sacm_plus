"""Organize raw downloaded datasets into the pipeline layout.

The pipeline expects, per dataset root:
    <root>/images/*  +  <root>/masks/*     (TestDataset / prepare_data source)

Converters (auto-detected by layout):
  - ready:      already has images/ + masks/ -> symlinked as-is
  - DRIVE:      official train/test split; masks are .gif with suffix
                names (_test_mask / _manual1) -> converted to binary .png
                with names matching the image basename. Outputs:
                DRIVE_train (training/images + training/1st_manual) and
                DRIVE_test (test/images + test/mask).
  - DIS5K:      split dirs (DIS-TR / DIS-TE1..4 / DIS-VD) each with
                im/ + gt/; gt is instance-labelled -> binarized (>0).
                Outputs: DIS5K_train (DIS-TR) and DIS5K_test (TE1..4
                merged, filenames prefixed with the split name).

Usage:
    python scripts/organize_datasets.py --src <dir-with-raw-datasets> \
        --out datasets
"""

import argparse
import os
import re
import shutil

import numpy as np
from PIL import Image


def convert_to_binary_png(src, dst, threshold=0):
    """Convert any image to a binary (0/255) PNG."""
    arr = np.array(Image.open(src).convert('L'))
    Image.fromarray(((arr > threshold) * 255).astype(np.uint8)).save(dst)


def copy_image(src, dst):
    shutil.copy2(src, dst)


def _convert_pairs(pairs, out_root):
    """pairs: list of (src_image, src_mask, out_basename); converts the
    mask to binary png named like the image."""
    os.makedirs(os.path.join(out_root, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_root, 'masks'), exist_ok=True)
    n = 0
    for src_img, src_mask, base in pairs:
        img_ext = os.path.splitext(src_img)[1]
        copy_image(src_img, os.path.join(out_root, 'images', base + img_ext))
        convert_to_binary_png(src_mask, os.path.join(out_root, 'masks', base + '.png'))
        n += 1
    return n


def organize_ready(src, out_root):
    """Symlink a dataset that already has images/ + masks/."""
    for sub in ('images', 'masks'):
        src_sub = os.path.join(src, sub)
        if not os.path.isdir(src_sub):
            raise ValueError(f"{src}: missing {sub}/")
        dst = os.path.join(out_root, sub)
        if not os.path.exists(dst):
            os.symlink(src_sub, dst)
    n = len(os.listdir(os.path.join(src, 'images')))
    print(f"[ready] {os.path.basename(out_root)}: {n} 对(软链接,未复制)")
    return n


def organize_drive(src, out_root):
    """DRIVE official layout -> DRIVE_train + DRIVE_test."""
    total = 0
    # train: training/images + training/1st_manual
    img_dir = os.path.join(src, 'training', 'images')
    mask_dir = os.path.join(src, 'training', '1st_manual')
    pairs = []
    for f in sorted(os.listdir(img_dir)):
        base = os.path.splitext(f)[0]                      # 21_training
        idx = base.split('_')[0]                            # 21
        mask = os.path.join(mask_dir, f'{idx}_manual1.gif')
        if os.path.exists(mask):
            pairs.append((os.path.join(img_dir, f), mask, base))
    n = _convert_pairs(pairs, os.path.join(out_root, 'DRIVE_train'))
    print(f"[DRIVE] DRIVE_train: {n} 对")
    total += n
    # test: test/images + test/mask (XX_test_mask.gif)
    img_dir = os.path.join(src, 'test', 'images')
    mask_dir = os.path.join(src, 'test', 'mask')
    pairs = []
    for f in sorted(os.listdir(img_dir)):
        base = os.path.splitext(f)[0]                      # 01_test
        mask = os.path.join(mask_dir, f'{base}_mask.gif')
        if os.path.exists(mask):
            pairs.append((os.path.join(img_dir, f), mask, base))
    n = _convert_pairs(pairs, os.path.join(out_root, 'DRIVE_test'))
    print(f"[DRIVE] DRIVE_test: {n} 对")
    total += n
    return total


def organize_dis5k(src, out_root):
    """DIS5K official layout -> DIS5K_train + DIS5K_test (gt binarized)."""
    total = 0
    for split, out_name, prefix in [
        ('DIS-TR', 'DIS5K_train', ''),
        ('DIS-VD', 'DIS5K_val', ''),
    ]:
        img_dir = os.path.join(src, split, 'im')
        gt_dir = os.path.join(src, split, 'gt')
        if not os.path.isdir(img_dir):
            continue
        pairs = []
        for f in sorted(os.listdir(img_dir)):
            base = os.path.splitext(f)[0]
            gt = os.path.join(gt_dir, base + '.png')
            if os.path.exists(gt):
                pairs.append((os.path.join(img_dir, f), gt,
                              prefix + base if prefix else base))
        n = _convert_pairs(pairs, os.path.join(out_root, out_name))
        print(f"[DIS5K] {out_name}: {n} 对")
        total += n
    # test: merge TE1..4, prefixed to avoid name collisions
    pairs = []
    for split in ('DIS-TE1', 'DIS-TE2', 'DIS-TE3', 'DIS-TE4'):
        img_dir = os.path.join(src, split, 'im')
        gt_dir = os.path.join(src, split, 'gt')
        if not os.path.isdir(img_dir):
            continue
        for f in sorted(os.listdir(img_dir)):
            base = os.path.splitext(f)[0]
            gt = os.path.join(gt_dir, base + '.png')
            if os.path.exists(gt):
                pairs.append((os.path.join(img_dir, f), gt, f'{split}_{base}'))
    n = _convert_pairs(pairs, os.path.join(out_root, 'DIS5K_test'))
    print(f"[DIS5K] DIS5K_test: {n} 对")
    total += n
    return total


def main():
    parser = argparse.ArgumentParser(description='把原始数据集整理成 images/+masks/ 布局')
    parser.add_argument('--src', type=str, required=True,
                        help='含原始数据集的目录(每个子目录一个数据集)')
    parser.add_argument('--out', type=str, default='datasets', help='输出根目录')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total = 0
    for name in sorted(os.listdir(args.src)):
        src = os.path.join(args.src, name)
        if not os.path.isdir(src) or name.startswith('.'):
            continue
        out_root = os.path.join(args.out, name)
        if os.path.isdir(os.path.join(src, 'images')) and os.path.isdir(os.path.join(src, 'masks')):
            if os.path.isdir(os.path.join(out_root, 'images')):
                print(f"[skip] {name}: 目标已具备 images/+masks/ 布局")
                continue
            total += organize_ready(src, out_root)
        elif os.path.isdir(os.path.join(src, 'training')) and os.path.isdir(os.path.join(src, 'test')):
            if os.path.isdir(os.path.join(args.out, 'DRIVE_train')) and \
               os.path.isdir(os.path.join(args.out, 'DRIVE_test')):
                print(f"[skip] {name}: DRIVE_train/DRIVE_test 已存在")
                continue
            total += organize_drive(src, args.out)
        elif os.path.isdir(os.path.join(src, 'DIS-TR')):
            if os.path.isdir(os.path.join(args.out, 'DIS5K_test')):
                print(f"[skip] {name}: DIS5K_* 已存在")
                continue
            total += organize_dis5k(src, args.out)
        else:
            print(f"[unknown] {name}: 无法识别的布局,跳过")
    print(f"\n完成: 共整理 {total} 对 → {args.out}")


if __name__ == '__main__':
    main()
