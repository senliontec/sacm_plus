"""Organize raw downloaded datasets into the pipeline layout.

The pipeline expects, per dataset root:
    <root>/images/*  +  <root>/masks/*     (TestDataset / prepare_data source)

Converters (auto-detected by layout):
  - ready:      already has images/ + masks/ -> symlinked as-is
  - DRIVE:      official train/test split. training/1st_manual is real
                vessel GT -> DRIVE_train. test/mask is a FOV mask, NOT
                vessel GT: it is written to DRIVE_test/fov/ and masks/
                is only created when real test GT is supplied via
                --drive_test_gt (or found in test/1st_manual,
                test/groundtruth, ...). Vessel masks are checked by
                foreground ratio so a FOV mask cannot be used silently.
  - DIS5K:      split dirs (DIS-TR / DIS-TE1..4 / DIS-VD) each with
                im/ + gt/; gt is instance-labelled -> binarized (>0).
                Outputs: DIS5K_train (DIS-TR) and DIS5K_test (TE1..4
                merged, filenames prefixed with the split name).

Usage:
    python scripts/organize_datasets.py --src <dir-with-raw-datasets> \
        --out datasets
"""

import argparse
import gzip
import io
import os
import shutil
import tarfile
import zipfile

import numpy as np
from PIL import Image


def convert_to_binary_png(src, dst, threshold=0):
    """Convert any image to a binary (0/255) PNG."""
    arr = np.array(Image.open(src).convert('L'))
    Image.fromarray(((arr > threshold) * 255).astype(np.uint8)).save(dst)


def copy_image(src, dst):
    shutil.copy2(src, dst)


VESSEL_MASK_FG_RANGE = (0.005, 0.30)
FOV_LIKE_FG_RATIO = 0.40
DRIVE_IMAGE_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg',
                    '.bmp', '.ppm', '.pgm'}
DRIVE_TEST_GT_SUBDIRS = ('1st_manual', 'groundtruth', 'ground_truth',
                         'gt', 'labels')


def mask_foreground_ratio(path):
    """Fraction of nonzero pixels in a mask (binary or 0/255)."""
    arr = np.array(Image.open(path).convert('L'))
    if arr.size == 0:
        return float('nan')
    return float((arr > 0).mean())


def validate_vessel_mask(path, lo=None, hi=None):
    """Reject FOV-like or otherwise implausible DRIVE vessel masks.

    DRIVE vessel GT is ~5-12% foreground; the official test/mask is a
    FOV mask (~68%). Raise instead of silently using the wrong target.
    """
    lo = VESSEL_MASK_FG_RANGE[0] if lo is None else lo
    hi = VESSEL_MASK_FG_RANGE[1] if hi is None else hi
    ratio = mask_foreground_ratio(path)
    if not np.isfinite(ratio) or not (lo <= ratio <= hi):
        raise ValueError(
            f"{path}: foreground ratio {ratio:.1%} is outside the vessel "
            f"range {lo:.1%}-{hi:.1%}. DRIVE test/mask is a FOV mask, not "
            "vessel ground truth; use --drive_test_gt with real labels."
        )
    return ratio


def _directory_median_ratio(mask_dir):
    if not os.path.isdir(mask_dir):
        return float('nan')
    ratios = []
    for f in sorted(os.listdir(mask_dir)):
        if os.path.splitext(f)[1].lower() not in ('.png', '.gif'):
            continue
        try:
            ratios.append(mask_foreground_ratio(os.path.join(mask_dir, f)))
        except Exception:
            continue
    return float(np.median(ratios)) if ratios else float('nan')


def _mask_dir_is_vessel_like(mask_dir):
    med = _directory_median_ratio(mask_dir)
    return (np.isfinite(med)
            and VESSEL_MASK_FG_RANGE[0] <= med <= VESSEL_MASK_FG_RANGE[1])


def _drive_test_gt_candidates(src, base, idx, explicit_gt=None):
    """Yield candidate test-GT paths for one DRIVE test image.

    test/mask is never auto-detected because it is FOV, not vessel GT.
    An explicit --drive_test_gt directory is still content-checked by
    validate_vessel_mask(), so pointing it at test/mask fails loudly.
    """
    dirs = []
    if explicit_gt:
        dirs.append(explicit_gt)
    else:
        for sub in DRIVE_TEST_GT_SUBDIRS:
            d = os.path.join(src, 'test', sub)
            if os.path.isdir(d):
                dirs.append(d)
    patterns = (
        f'{idx}_manual1.gif', f'{idx}_manual1.png',
        f'{base}_manual1.gif', f'{base}_manual1.png',
        f'{base}_manual.gif', f'{base}_manual.png',
        f'{base}_gt.gif', f'{base}_gt.png',
        f'{base}_mask.gif', f'{base}_mask.png',
        f'{base}.gif', f'{base}.png',
        f'{idx}.gif', f'{idx}.png',
    )
    seen = set()
    for d in dirs:
        for pat in patterns:
            p = os.path.join(d, pat)
            if p in seen or not os.path.isfile(p):
                continue
            seen.add(p)
            yield p


def _copy_images(img_dir, out_images):
    """Copy regular image files, skipping hidden/OS metadata files."""
    os.makedirs(out_images, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(img_dir)):
        p = os.path.join(img_dir, f)
        ext = os.path.splitext(f)[1].lower()
        if not os.path.isfile(p) or ext not in DRIVE_IMAGE_EXTS:
            continue
        copy_image(p, os.path.join(out_images, f))
        n += 1
    return n


def _quarantine_fov_masks(test_root):
    """Move an old DRIVE_test/masks directory to fov/ if it is FOV-like."""
    mask_dir = os.path.join(test_root, 'masks')
    if not os.path.isdir(mask_dir):
        return False
    med = _directory_median_ratio(mask_dir)
    if not np.isfinite(med) or med <= FOV_LIKE_FG_RATIO:
        return False
    fov_dir = os.path.join(test_root, 'fov')
    os.makedirs(fov_dir, exist_ok=True)
    for f in sorted(os.listdir(mask_dir)):
        src = os.path.join(mask_dir, f)
        dst = os.path.join(fov_dir, f)
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)
    os.rmdir(mask_dir)
    print(f"[DRIVE] 旧 DRIVE_test/masks 前景中位数 {med:.1%} 不像血管,"
          "已移至 DRIVE_test/fov/")
    return True


def _convert_drive_fov(src_mask_dir, img_dir, out_fov_dir):
    """Convert official FOV masks for reference; never use them as GT."""
    os.makedirs(out_fov_dir, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(img_dir)):
        base = os.path.splitext(f)[0]
        mask = os.path.join(src_mask_dir, f'{base}_mask.gif')
        if os.path.exists(mask):
            convert_to_binary_png(mask, os.path.join(out_fov_dir, base + '.png'))
            n += 1
    return n


def _convert_drive_training_fov(src_mask_dir, img_dir, out_fov_dir):
    """training/mask/<id>_training_mask.gif -> DRIVE_train/fov/<id>.png."""
    if not os.path.isdir(src_mask_dir):
        return 0
    os.makedirs(out_fov_dir, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(img_dir)):
        base = os.path.splitext(f)[0]
        mask = os.path.join(src_mask_dir, f'{base}_mask.gif')
        if os.path.exists(mask):
            convert_to_binary_png(mask, os.path.join(out_fov_dir, base + '.png'))
            n += 1
    return n


def _write_drive_test_readme(test_root):
    path = os.path.join(test_root, 'README.md')
    text = """# DRIVE test: no public vessel ground truth

`images/` contains the official DRIVE test images.
`fov/` contains the official FOV masks from `test/mask` (foreground ~68%).
FOV masks are **not** vessel labels and must not be used for Dice, clDice,
HD95, or any segmentation metric.

`masks/` is intentionally absent. To build a valid test set, either:

1. Use a fixed held-out split of `DRIVE_train` (official `1st_manual` labels),
   e.g. 3-shot train / 17-image test; or
2. Evaluate through the official DRIVE challenge server; or
3. Rebuild with a verified rehosted test GT:

       python scripts/organize_datasets.py \\
           --src datasets --out datasets \\
           --drive_test_gt /path/to/verified/vessel/gt

The converter rejects masks whose foreground ratio is outside 0.5%-30%,
which catches the FOV mask (~68%) automatically.
"""
    with open(path, 'w') as f:
        f.write(text)


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


def organize_drive(src, out_root, drive_test_gt=None):
    """DRIVE official layout -> DRIVE_train + DRIVE_test.

    training/1st_manual is real vessel GT. test/mask is FOV, not GT:
    it is written to DRIVE_test/fov/, and DRIVE_test/masks/ is only
    created when real vessel GT is supplied (explicit directory or
    test/1st_manual, test/groundtruth, ...).
    """
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
            validate_vessel_mask(mask)
            pairs.append((os.path.join(img_dir, f), mask, base))
    n = _convert_pairs(pairs, os.path.join(out_root, 'DRIVE_train'))
    print(f"[DRIVE] DRIVE_train: {n} 对")
    n_fov = _convert_drive_training_fov(
        os.path.join(src, 'training', 'mask'), img_dir,
        os.path.join(out_root, 'DRIVE_train', 'fov'))
    if n_fov:
        print(f"[DRIVE] DRIVE_train fov: {n_fov} 张")
    total += n

    test_img_dir = os.path.join(src, 'test', 'images')
    if not os.path.isdir(test_img_dir):
        print("[DRIVE] test/images 不存在,跳过 DRIVE_test")
        return total

    test_root = os.path.join(out_root, 'DRIVE_test')
    n_img = _copy_images(test_img_dir, os.path.join(test_root, 'images'))
    # An old masks/ directory may contain FOV masks from the buggy
    # converter; quarantine it before deciding what to rebuild.
    _quarantine_fov_masks(test_root)

    fov_src = os.path.join(src, 'test', 'mask')
    n_fov = 0
    if os.path.isdir(fov_src):
        n_fov = _convert_drive_fov(fov_src, test_img_dir,
                                   os.path.join(test_root, 'fov'))

    pairs = []
    rejected = []
    for f in sorted(os.listdir(test_img_dir)):
        if os.path.splitext(f)[1].lower() not in DRIVE_IMAGE_EXTS:
            continue
        base = os.path.splitext(f)[0]                      # 01_test
        idx = base.split('_')[0]                            # 01
        for cand in _drive_test_gt_candidates(src, base, idx, drive_test_gt):
            try:
                validate_vessel_mask(cand)
            except ValueError as e:
                rejected.append(str(e))
                continue
            pairs.append((os.path.join(test_img_dir, f), cand, base))
            break

    if pairs:
        # Rebuild masks from verified GT; drop stale masks first.
        mask_out = os.path.join(test_root, 'masks')
        if os.path.isdir(mask_out):
            shutil.rmtree(mask_out)
        n = _convert_pairs(pairs, test_root)
        print(f"[DRIVE] DRIVE_test: {n} 对 (真实血管 GT)")
        if n != n_img:
            print(f"[DRIVE] 警告: test images={n_img}, GT pairs={n};"
                  " 缺 GT 的图不会被评测")
        readme = os.path.join(test_root, 'README.md')
        if os.path.exists(readme):
            os.remove(readme)  # stale "masks absent" note
        total += n
    elif _mask_dir_is_vessel_like(os.path.join(test_root, 'masks')):
        print("[DRIVE] DRIVE_test: 保留已有的血管 GT masks/")
        readme = os.path.join(test_root, 'README.md')
        if os.path.exists(readme):
            os.remove(readme)
    else:
        print(f"[DRIVE] DRIVE_test: images/{n_img} 张 + fov/{n_fov} 张;"
              " 官方 test/mask 是 FOV,不是 GT,因此不生成 masks/")
        _write_drive_test_readme(test_root)

    for msg in rejected[:3]:
        print(f"[DRIVE] 忽略无效测试 GT: {msg}")
    return total


def _find_dir_with(src, predicate):
    for root, _, files in os.walk(src):
        if predicate(root, files):
            return root
    return None


def _save_rgb(img, dst):
    img.convert('RGB').save(dst)


def _save_binary(img, dst):
    arr = np.array(img.convert('L'))
    Image.fromarray(((arr > 0) * 255).astype(np.uint8)).save(dst)


def _tar_image(tar, name):
    f = tar.extractfile(name)
    if f is None:
        return None
    data = f.read()
    if name.endswith('.gz'):
        data = gzip.decompress(data)
    return Image.open(io.BytesIO(data))


def organize_dca1(src, out_root):
    """DCA1 Database_134_Angiograms -> DCA1/images + DCA1/masks."""
    pair_dir = _find_dir_with(
        src, lambda r, fs: any(f.endswith('_gt.pgm') for f in fs))
    if pair_dir is None:
        raise ValueError(f"DCA1: no *_gt.pgm pairs found under {src}")
    out = os.path.join(out_root, 'DCA1')
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
    n = 0
    for gt_name in sorted(os.listdir(pair_dir)):
        if not gt_name.endswith('_gt.pgm'):
            continue
        idx = gt_name[:-len('_gt.pgm')]
        img_path = os.path.join(pair_dir, idx + '.pgm')
        gt_path = os.path.join(pair_dir, gt_name)
        if not os.path.exists(img_path):
            continue
        _save_rgb(Image.open(img_path), os.path.join(out, 'images', idx + '.png'))
        convert_to_binary_png(gt_path, os.path.join(out, 'masks', idx + '.png'))
        n += 1
    print(f"[DCA1] {n} 对")
    return n


def _read_maybe_gz(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            return Image.open(io.BytesIO(f.read()))
    return Image.open(path)


def _organize_stare_extracted(root, out_root, label_set='ah'):
    img_dir = os.path.join(root, 'stare-images')
    label_dir = os.path.join(root, f'labels-{label_set}')
    if not os.path.isdir(label_dir):
        label_dir = os.path.join(root, 'labels-vk')
    if not os.path.isdir(label_dir):
        raise ValueError(f"STARE: no labels-{label_set} / labels-vk under {root}")
    out = os.path.join(out_root, 'STARE')
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
    n = 0
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith(('.ppm.gz', '.ppm')):
            continue
        key = f.split('.')[0]
        label = None
        for cand in (f'{key}.{label_set}.ppm.gz', f'{key}.{label_set}.ppm',
                     f'{key}.ppm.gz', f'{key}.ppm'):
            p = os.path.join(label_dir, cand)
            if os.path.exists(p):
                label = p
                break
        if label is None:
            continue
        _save_rgb(_read_maybe_gz(os.path.join(img_dir, f)),
                  os.path.join(out, 'images', key + '.png'))
        _save_binary(_read_maybe_gz(label),
                     os.path.join(out, 'masks', key + '.png'))
        n += 1
    print(f"[STARE] {n} 对 (extracted, labels {os.path.basename(label_dir)})")
    return n


def organize_stare(src, out_root, label_set='ah'):
    """STARE tar or extracted .ppm.gz -> STARE/images + STARE/masks."""
    extracted = _find_dir_with(
        src, lambda r, fs: os.path.isdir(os.path.join(r, 'stare-images')))
    if extracted is not None:
        return _organize_stare_extracted(extracted, out_root, label_set)
    root = _find_dir_with(src, lambda r, fs: 'stare-images.tar' in fs)
    if root is None:
        raise ValueError(f"STARE: no stare-images dir/tar under {src}")
    img_tar = os.path.join(root, 'stare-images.tar')
    label_tar = os.path.join(root, f'labels-{label_set}.tar')
    if not os.path.exists(label_tar):
        label_tar = os.path.join(root, 'labels-vk.tar')
    if not os.path.exists(label_tar):
        raise ValueError(f"STARE: no labels-{label_set}.tar / labels-vk.tar")
    out = os.path.join(out_root, 'STARE')
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out, 'masks'), exist_ok=True)

    images = {}
    with tarfile.open(img_tar, 'r:*') as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            key = os.path.basename(m.name).split('.')[0]
            img = _tar_image(tar, m.name)
            if img is not None:
                images[key] = img
    labels = {}
    with tarfile.open(label_tar, 'r:*') as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            key = os.path.basename(m.name).split('.')[0]
            img = _tar_image(tar, m.name)
            if img is not None:
                labels[key] = img
    n = 0
    for key in sorted(images):
        if key not in labels:
            continue
        _save_rgb(images[key], os.path.join(out, 'images', key + '.png'))
        _save_binary(labels[key], os.path.join(out, 'masks', key + '.png'))
        n += 1
    print(f"[STARE] {n} 对 (labels {os.path.basename(label_tar)})")
    return n


def _chase_mask_for(base, root):
    for suffix in ('_1stHO.png', '_2ndHO.png'):
        p = os.path.join(root, base + suffix)
        if os.path.exists(p):
            return p
    return None


def organize_chasedb1(src, out_root):
    """Extracted CHASEDB1 -> CHASEDB1/images + masks (1stHO preferred)."""
    root = _find_dir_with(
        src, lambda r, fs: any(f.endswith('_1stHO.png') for f in fs))
    if root is None:
        raise ValueError(f"CHASEDB1: no *_1stHO.png found under {src}")
    out = os.path.join(out_root, 'CHASEDB1')
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
    n = 0
    for f in sorted(os.listdir(root)):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        if '_1stHO' in f or '_2ndHO' in f:
            continue
        base = os.path.splitext(f)[0]
        mask = _chase_mask_for(base, root)
        if mask is None:
            continue
        _save_rgb(Image.open(os.path.join(root, f)),
                  os.path.join(out, 'images', base + '.png'))
        _save_binary(Image.open(mask),
                     os.path.join(out, 'masks', base + '.png'))
        n += 1
    print(f"[CHASEDB1] {n} 对")
    return n


def organize_chasedb1_zip(zip_path, out_root):
    """CHASEDB1.zip -> CHASEDB1/images + masks without manual extraction."""
    out = os.path.join(out_root, 'CHASEDB1')
    if os.path.isdir(os.path.join(out, 'images')) and \
            os.listdir(os.path.join(out, 'images')):
        print("[skip] CHASEDB1.zip: CHASEDB1 已存在")
        return 0
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        by_base = {os.path.basename(n): n for n in names}
        for name in names:
            base_name = os.path.basename(name)
            if not base_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if '_1stHO' in base_name or '_2ndHO' in base_name:
                continue
            base = os.path.splitext(base_name)[0]
            mask_name = by_base.get(base + '_1stHO.png') or \
                by_base.get(base + '_2ndHO.png')
            if mask_name is None:
                continue
            img = Image.open(io.BytesIO(z.read(name)))
            mask = Image.open(io.BytesIO(z.read(mask_name)))
            _save_rgb(img, os.path.join(out, 'images', base + '.png'))
            _save_binary(mask, os.path.join(out, 'masks', base + '.png'))
            n += 1
    print(f"[CHASEDB1.zip] {n} 对")
    return n


def organize_fives(src, out_root):
    """FIVES train/test Original+Ground truth -> FIVES_train/FIVES_test."""
    root = _find_dir_with(
        src,
        lambda r, fs: os.path.isdir(os.path.join(r, 'train', 'Original'))
        and os.path.isdir(os.path.join(r, 'test', 'Original')))
    if root is None:
        raise ValueError(f"FIVES: train/Original + test/Original not found under {src}")
    total = 0
    for split, out_name in (('train', 'FIVES_train'), ('test', 'FIVES_test')):
        img_dir = os.path.join(root, split, 'Original')
        gt_dir = os.path.join(root, split, 'Ground truth')
        out = os.path.join(out_root, out_name)
        if os.path.isdir(os.path.join(out, 'images')) and \
                os.listdir(os.path.join(out, 'images')):
            print(f"[skip] {out_name}: 已存在")
            continue
        os.makedirs(os.path.join(out, 'images'), exist_ok=True)
        os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
        n = 0
        for f in sorted(os.listdir(img_dir)):
            if not f.lower().endswith('.png'):
                continue
            img = os.path.join(img_dir, f)
            gt = os.path.join(gt_dir, f)
            if not os.path.exists(gt):
                continue
            # FIVES images are already PNG; copying is much faster than
            # re-encoding 800 2048x2048 images.
            copy_image(img, os.path.join(out, 'images', f))
            _save_binary(Image.open(gt), os.path.join(out, 'masks', f))
            n += 1
        print(f"[FIVES] {out_name}: {n} 对")
        total += n
    return total


def organize_cremi(src, out_root):
    """CREMI HDF5 -> CREMI/images + CREMI/masks (curvilinear boundary).

    Some released volumes contain all-zero raw sections (e.g. volume C
    z014/z074 in the 20160501 files) while their labels are still
    non-empty. Those images carry no signal and must not enter training
    or evaluation, so they are skipped here.
    """
    hdfs = []
    for root, _, files in os.walk(src):
        for f in files:
            if f.endswith('.hdf') and 'padded' not in f:
                hdfs.append(os.path.join(root, f))
    if not hdfs:
        raise ValueError(f"CREMI: no unpadded .hdf found under {src}")
    try:
        import h5py
    except ImportError as e:
        raise ImportError("CREMI conversion requires h5py: pip install h5py") from e
    out = os.path.join(out_root, 'CREMI')
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
    n = 0
    n_blank = 0
    for hdf in sorted(hdfs):
        stem = os.path.basename(hdf)[:-4]
        sample = stem.split('_2016')[0]
        with h5py.File(hdf, 'r') as f:
            raw = f['volumes/raw']
            nid = f['volumes/labels/neuron_ids']
            for z in range(raw.shape[0]):
                img = np.asarray(raw[z], dtype=np.uint8)
                name = f'{sample}_z{z:03d}.png'
                if not np.any(img):
                    # Remove stale outputs from an earlier conversion.
                    for sub in ('images', 'masks'):
                        stale = os.path.join(out, sub, name)
                        if os.path.exists(stale):
                            os.remove(stale)
                    print(f"[CREMI] skip blank raw slice {name}")
                    n_blank += 1
                    continue
                lab = np.asarray(nid[z])
                boundary = np.zeros_like(lab, dtype=bool)
                boundary[:, :-1] |= lab[:, :-1] != lab[:, 1:]
                boundary[:-1, :] |= lab[:-1, :] != lab[1:, :]
                Image.fromarray(img).save(os.path.join(out, 'images', name))
                Image.fromarray((boundary * 255).astype(np.uint8)).save(
                    os.path.join(out, 'masks', name))
                n += 1
    print(f"[CREMI] {n} 对 (跳过全黑切片 {n_blank} 张)")
    return n


def organize_cracktree(src, out_root, variant='rgb'):
    """CrackTree200 rgb/gray + gt -> CrackTree200/images + masks."""
    root = _find_dir_with(
        src,
        lambda r, fs: os.path.isdir(os.path.join(r, 'cracktree200_gt'))
        and (os.path.isdir(os.path.join(r, 'cracktree200rgb'))
             or os.path.isdir(os.path.join(r, 'cracktree200gray'))))
    if root is None:
        raise ValueError(f"CrackTree200: rgb/gray + gt dirs not found under {src}")
    img_dir = os.path.join(root, f'cracktree200{variant}')
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(root, 'cracktree200rgb')
    gt_dir = os.path.join(root, 'cracktree200_gt')
    out = os.path.join(out_root, 'CrackTree200')
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
    n = 0
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        base = os.path.splitext(f)[0]
        gt = os.path.join(gt_dir, base + '.png')
        if not os.path.exists(gt):
            continue
        copy_image(os.path.join(img_dir, f),
                   os.path.join(out, 'images', f))
        convert_to_binary_png(gt, os.path.join(out, 'masks', base + '.png'))
        n += 1
    print(f"[CrackTree200] {n} 对")
    return n


def organize_deepcrack(src, out_root):
    """DeepCrack train_img/lab + test_img/lab -> DeepCrack_train/test."""
    root = _find_dir_with(
        src,
        lambda r, fs: all(os.path.isdir(os.path.join(r, d)) for d in
                          ('train_img', 'train_lab', 'test_img', 'test_lab')))
    if root is None:
        raise ValueError(f"DeepCrack: train/test img/lab dirs not found under {src}")
    total = 0
    for split, out_name in (('train', 'DeepCrack_train'),
                            ('test', 'DeepCrack_test')):
        img_dir = os.path.join(root, f'{split}_img')
        lab_dir = os.path.join(root, f'{split}_lab')
        out = os.path.join(out_root, out_name)
        os.makedirs(os.path.join(out, 'images'), exist_ok=True)
        os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
        n = 0
        for f in sorted(os.listdir(img_dir)):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            base = os.path.splitext(f)[0]
            lab = None
            for ext in ('.png', '.jpg', '.jpeg'):
                cand = os.path.join(lab_dir, base + ext)
                if os.path.exists(cand):
                    lab = cand
                    break
            if lab is None:
                continue
            copy_image(os.path.join(img_dir, f),
                       os.path.join(out, 'images', f))
            convert_to_binary_png(lab, os.path.join(out, 'masks', base + '.png'))
            n += 1
        print(f"[DeepCrack] {out_name}: {n} 对")
        total += n
    return total


def organize_xcad(src, out_root):
    """XCAD test/images + test/masks -> XCAD_test (train is unlabeled)."""
    root = _find_dir_with(
        src,
        lambda r, fs: os.path.isdir(os.path.join(r, 'test', 'images'))
        and os.path.isdir(os.path.join(r, 'test', 'masks')))
    if root is None:
        raise ValueError(f"XCAD: test/images + test/masks not found under {src}")
    img_dir = os.path.join(root, 'test', 'images')
    mask_dir = os.path.join(root, 'test', 'masks')
    out = os.path.join(out_root, 'XCAD_test')
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
    n = 0
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        base = os.path.splitext(f)[0]
        mask = None
        for ext in ('.png', '.jpg', '.jpeg'):
            cand = os.path.join(mask_dir, base + ext)
            if os.path.exists(cand):
                mask = cand
                break
        if mask is None:
            continue
        copy_image(os.path.join(img_dir, f),
                   os.path.join(out, 'images', f))
        convert_to_binary_png(mask, os.path.join(out, 'masks', base + '.png'))
        n += 1
    print(f"[XCAD] test: {n} 对 (train unlabeled, ignored)")
    return n


def organize_road(src, out_root):
    """ROAD/Massachusetts Roads test split -> ROAD_test/images + masks."""
    root = _find_dir_with(
        src,
        lambda r, fs: os.path.isdir(os.path.join(r, 'tiff', 'test'))
        and os.path.isdir(os.path.join(r, 'tiff', 'test_labels')))
    if root is None:
        raise ValueError(f"ROAD: tiff/test + tiff/test_labels not found under {src}")
    img_dir = os.path.join(root, 'tiff', 'test')
    lab_dir = os.path.join(root, 'tiff', 'test_labels')
    out = os.path.join(out_root, 'ROAD_test')
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
    n = 0
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith(('.tif', '.tiff', '.png')):
            continue
        base = os.path.splitext(f)[0]
        lab = None
        for ext in ('.tif', '.tiff', '.png'):
            cand = os.path.join(lab_dir, base + ext)
            if os.path.exists(cand):
                lab = cand
                break
        if lab is None:
            continue
        copy_image(os.path.join(img_dir, f),
                   os.path.join(out, 'images', f))
        convert_to_binary_png(lab, os.path.join(out, 'masks', base + '.png'))
        n += 1
    print(f"[ROAD] test: {n} 对")
    return n


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
    parser.add_argument('--drive_test_gt', type=str, default=None,
                        help='DRIVE test 血管 GT 目录(默认不把 test/mask 当 GT)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total = 0
    # Zip-only CHASEDB1 is common; read it without manual extraction.
    for name in sorted(os.listdir(args.src)):
        p = os.path.join(args.src, name)
        if os.path.isfile(p) and name.lower() == 'chasedb1.zip':
            total += organize_chasedb1_zip(p, args.out)
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
            # Never skip DRIVE: an old DRIVE_test/masks may be FOV and
            # must be validated/quarantined on every run.
            total += organize_drive(src, args.out,
                                    drive_test_gt=args.drive_test_gt)
        elif os.path.isdir(os.path.join(src, 'DIS-TR')):
            if os.path.isdir(os.path.join(args.out, 'DIS5K_test')):
                print(f"[skip] {name}: DIS5K_* 已存在")
                continue
            total += organize_dis5k(src, args.out)
        elif _find_dir_with(src, lambda r, fs: any(f.endswith('_gt.pgm') for f in fs)):
            total += organize_dca1(src, args.out)
        elif _find_dir_with(
                src, lambda r, fs: 'stare-images.tar' in fs
                or os.path.isdir(os.path.join(r, 'stare-images'))):
            total += organize_stare(src, args.out)
        elif _find_dir_with(src, lambda r, fs: any(f.endswith('_1stHO.png') for f in fs)):
            total += organize_chasedb1(src, args.out)
        elif _find_dir_with(
                src, lambda r, fs: os.path.isdir(os.path.join(r, 'train', 'Original'))
                and os.path.isdir(os.path.join(r, 'test', 'Original'))):
            total += organize_fives(src, args.out)
        elif _find_dir_with(src, lambda r, fs: any(f.endswith('.hdf') for f in fs)):
            total += organize_cremi(src, args.out)
        elif _find_dir_with(
                src, lambda r, fs: os.path.isdir(os.path.join(r, 'cracktree200_gt'))
                and (os.path.isdir(os.path.join(r, 'cracktree200rgb'))
                     or os.path.isdir(os.path.join(r, 'cracktree200gray')))):
            total += organize_cracktree(src, args.out)
        elif _find_dir_with(
                src, lambda r, fs: all(os.path.isdir(os.path.join(r, d)) for d in
                                       ('train_img', 'train_lab', 'test_img', 'test_lab'))):
            total += organize_deepcrack(src, args.out)
        elif _find_dir_with(
                src, lambda r, fs: os.path.isdir(os.path.join(r, 'test', 'images'))
                and os.path.isdir(os.path.join(r, 'test', 'masks'))):
            total += organize_xcad(src, args.out)
        elif _find_dir_with(
                src, lambda r, fs: os.path.isdir(os.path.join(r, 'tiff', 'test'))
                and os.path.isdir(os.path.join(r, 'tiff', 'test_labels'))):
            total += organize_road(src, args.out)
        else:
            print(f"[unknown] {name}: 无法识别的布局,跳过")
    print(f"\n完成: 共整理 {total} 对 → {args.out}")


if __name__ == '__main__':
    main()
