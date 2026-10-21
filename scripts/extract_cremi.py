#!/usr/bin/env python3
"""Export CREMI HDF5 slices as image and boundary-mask PNG files.

CREMI HDF5 files contain:

    volumes/raw                 uint8  [Z, H, W]
    volumes/labels/neuron_ids   uint64 [Z, H, W]
    volumes/labels/clefts       uint64 [Z, H, W]

CREMI does not provide an official 2-D boundary mask. This script derives a
0/255 boundary mask from neuron_ids: a pixel is a boundary when its right or
lower neighbour has a different neuron id. This is the same target used by
``organize_datasets.py::organize_cremi``.

Usage:
    # All unpadded HDF files under --src
    python scripts/extract_cremi.py --src /path/to/datasets/CREMI \
        --out /path/to/cremi_png --skip-blank

    # Only selected HDF file(s); --src is optional when --hdf is given
    python scripts/extract_cremi.py \
        --hdf /path/to/datasets/CREMI/sample_A_20160501.hdf \
        --out /path/to/cremi_png --skip-blank

Output layout (one folder per HDF, named after the original file):
    <out>/<hdf_name>/            # e.g. sample_A_20160501.hdf/
        images_gray/z<ZZZ>.png   # single-channel L
        images_rgb/z<ZZZ>.png    # 3-channel RGB, R=G=B
        masks/z<ZZZ>.png
        manifest.csv             # paths relative to this HDF folder
    <out>/manifest.csv           # aggregate manifest with paths relative to out

CREMI ``volumes/raw`` is single-channel electron microscopy, so the RGB
export is the grayscale image replicated across R/G/B (R=G=B). It does not
invent color information.

By default all slices are exported, including all-zero raw slices. Pass
``--skip-blank`` to omit them (this reproduces the current 373-pair CREMI
conversion). Padded ``*_padded_*.hdf`` files are never selected by automatic
discovery: their raw volumes are padded to 3072x3072 while their labels stay
1250x1250, so they cannot be paired without padding-aware alignment. Use the
unpadded files.

Automatic discovery only selects HDF files that contain both
``volumes/raw`` and ``volumes/labels/neuron_ids``. CREMI test volumes
(``sample_A+/B+/C+_20160601.hdf``) contain raw images but no labels and are
skipped with a warning. Use ``--hdf`` to select files explicitly; the
script still requires neuron labels to produce boundary masks.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import h5py
except ImportError:  # pragma: no cover - exercised only without the data extra
    h5py = None


RAW_KEY = 'volumes/raw'
NEURON_KEY = 'volumes/labels/neuron_ids'


def require_h5py() -> None:
    if h5py is None:
        raise SystemExit(
            "h5py is required for CREMI conversion; install it with "
            "`pip install h5py` or `pip install -e '.[data]'`."
        )


def discover_hdfs(src: os.PathLike) -> list[Path]:
    """Return labeled unpadded CREMI .hdf files under *src*.

    Padded files cannot be paired with the unpadded labels without
    padding-aware alignment and are excluded. Test volumes such as
    ``sample_A+_20160601.hdf`` contain raw images but no neuron_ids; they
    cannot produce boundary masks and are skipped with a warning. Use
    ``--hdf`` to select a file explicitly.
    """
    src_path = Path(src)
    if not src_path.is_dir():
        raise SystemExit(f"source directory does not exist: {src_path}")
    files = [
        p for p in sorted(src_path.rglob('*.hdf'))
        if 'padded' not in p.name.lower()
    ]
    if not files:
        raise SystemExit(f"no CREMI .hdf files found under {src_path}")

    labeled = []
    unlabeled = []
    for path in files:
        try:
            with h5py.File(path, 'r') as handle:
                has_labels = RAW_KEY in handle and NEURON_KEY in handle
        except OSError as exc:
            raise SystemExit(f"cannot open {path}: {exc}") from exc
        if has_labels:
            labeled.append(path)
        else:
            unlabeled.append(path)
    if unlabeled:
        names = ', '.join(p.name for p in unlabeled)
        print(
            f"[CREMI] skip HDF without {RAW_KEY} and {NEURON_KEY}: {names}"
        )
    if not labeled:
        raise SystemExit(
            f"no labeled CREMI HDF files found under {src_path}; "
            "test volumes without neuron_ids cannot produce masks"
        )
    return labeled


def validate_hdf(path: Path) -> None:
    """Fail before any output is written if an HDF cannot produce masks."""
    with h5py.File(path, 'r') as handle:
        for key in (RAW_KEY, NEURON_KEY):
            if key not in handle:
                raise SystemExit(
                    f"{path}: missing dataset {key}; this HDF cannot "
                    "produce image/mask pairs"
                )
        raw = handle[RAW_KEY]
        neuron_ids = handle[NEURON_KEY]
        if raw.dtype != np.uint8:
            raise SystemExit(f"{path}: {RAW_KEY} must be uint8, got {raw.dtype}")
        if raw.ndim != 3 or raw.shape != neuron_ids.shape:
            raise SystemExit(
                f"{path}: expected raw/neuron_ids shapes [Z,H,W] and "
                f"equal, got {raw.shape} / {neuron_ids.shape}"
            )


def resolve_hdfs(src: os.PathLike | None, hdf_paths: list[str]) -> list[Path]:
    """Resolve explicitly selected HDF files.

    Relative paths are tried against the current directory and then against
    *src* (including a recursive match by basename). Explicit selections may
    be outside *src*, but they must still contain neuron_ids; padded files
    are not supported because their raw/label shapes differ.
    """
    src_path = Path(src).resolve() if src else None
    resolved: list[Path] = []
    for item in hdf_paths:
        requested = Path(item).expanduser()
        path: Path | None = None
        if requested.is_file():
            path = requested.resolve()
        elif src_path is not None:
            candidate = (src_path / requested).resolve()
            if candidate.is_file():
                path = candidate
            else:
                matches = sorted(src_path.rglob(requested.name))
                if len(matches) == 1:
                    path = matches[0].resolve()
                elif len(matches) > 1:
                    sample = ', '.join(str(p) for p in matches[:5])
                    raise SystemExit(
                        f"ambiguous --hdf {item!r}: {len(matches)} matches "
                        f"under {src_path} ({sample}, ...)"
                    )
        if path is None:
            raise SystemExit(f"--hdf not found: {item}")
        if path.suffix.lower() != '.hdf':
            raise SystemExit(f"--hdf must point to an .hdf file: {item}")
        resolved.append(path)
    if not resolved:
        raise SystemExit("--hdf was given but no files resolved")
    # Preserve selection order while dropping duplicates.
    return list(dict.fromkeys(resolved))


def _source_root(paths: list[Path]) -> Path:
    """Common parent used for portable ``source`` paths in the manifest."""
    return Path(os.path.commonpath([str(p.parent) for p in paths]))


def boundary_mask(neuron_ids: np.ndarray) -> np.ndarray:
    """Derive a 2-D boundary mask from an integer neuron-id slice.

    A pixel is marked when the label to its right or below differs. This
    marks every interface exactly once and keeps the mask one pixel wide.
    """
    labels = np.asarray(neuron_ids)
    if labels.ndim != 2:
        raise ValueError(f"neuron_ids slice must be 2-D, got shape {labels.shape}")
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, :-1] = labels[:, :-1] != labels[:, 1:]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    return boundary


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def extract(
    src: os.PathLike | None,
    out: os.PathLike,
    *,
    hdf_paths: list[str] | None = None,
    skip_blank: bool = False,
    overwrite: bool = False,
) -> dict:
    """Export all selected CREMI slices.

    If *hdf_paths* is given, only those files are read; otherwise every
    unpadded ``*.hdf`` under *src* is used. Returns a summary dict with
    counts: volumes, slices, written, blank_skipped, blank_written.
    """
    require_h5py()
    src_path = Path(src).resolve() if src else None
    out_path = Path(out).resolve()
    if hdf_paths:
        hdfs = resolve_hdfs(src_path, hdf_paths)
    else:
        if src_path is None:
            raise SystemExit("--src is required when --hdf is not given")
        hdfs = discover_hdfs(src_path)
    if src_path is None:
        src_path = _source_root(hdfs)
    for hdf in hdfs:
        validate_hdf(hdf)

    volume_names = [Path(hdf).name for hdf in hdfs]
    if len(set(volume_names)) != len(volume_names):
        duplicates = sorted({
            name for name in volume_names if volume_names.count(name) > 1
        })
        raise SystemExit(f"duplicate HDF filenames: {duplicates}")
    volume_dirs = {hdf: out_path / Path(hdf).name for hdf in hdfs}
    manifest_path = out_path / 'manifest.csv'

    if out_path.exists() and not out_path.is_dir():
        raise SystemExit(f"output path is not a directory: {out_path}")

    if out_path.exists() and any(out_path.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"output directory is not empty: {out_path}\n"
                "Pass --overwrite to replace the per-HDF folders and "
                "manifest.csv."
            )
        for hdf in hdfs:
            volume_dir = volume_dirs[hdf]
            if volume_dir.is_dir():
                shutil.rmtree(volume_dir)
            # Migrate the previous stem-only layout on overwrite.
            legacy_volume_dir = out_path / Path(hdf).stem
            if legacy_volume_dir != volume_dir and legacy_volume_dir.is_dir():
                shutil.rmtree(legacy_volume_dir)
        # Remove the old flat layout from earlier versions of this script.
        for legacy in ('images', 'images_gray', 'images_rgb', 'masks'):
            legacy_dir = out_path / legacy
            if legacy_dir.is_dir():
                shutil.rmtree(legacy_dir)
        if manifest_path.exists():
            manifest_path.unlink()
    out_path.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'hdf', 'z', 'image_gray', 'image_rgb', 'mask', 'source',
        'blank', 'fg_ratio',
    ]
    summary = {
        'volumes': 0,
        'slices': 0,
        'written': 0,
        'blank_skipped': 0,
        'blank_written': 0,
    }
    all_rows = []

    for hdf in hdfs:
        volume_name = Path(hdf).name
        volume_dir = volume_dirs[hdf]
        image_gray_dir = volume_dir / 'images_gray'
        image_rgb_dir = volume_dir / 'images_rgb'
        mask_dir = volume_dir / 'masks'
        for directory in (image_gray_dir, image_rgb_dir, mask_dir):
            directory.mkdir(parents=True, exist_ok=True)
        volume_rows = []

        with h5py.File(hdf, 'r') as handle:
            if RAW_KEY not in handle:
                raise SystemExit(f"{hdf}: missing dataset {RAW_KEY}")
            if NEURON_KEY not in handle:
                raise SystemExit(f"{hdf}: missing dataset {NEURON_KEY}")
            raw = handle[RAW_KEY]
            neuron_ids = handle[NEURON_KEY]
            if raw.ndim != 3 or raw.shape != neuron_ids.shape:
                raise SystemExit(
                    f"{hdf}: expected raw/neuron_ids shapes [Z,H,W] and "
                    f"equal, got {raw.shape} / {neuron_ids.shape}"
                )

            summary['volumes'] += 1
            n_slices = raw.shape[0]
            for z in range(n_slices):
                image = np.asarray(raw[z])
                labels = np.asarray(neuron_ids[z])
                if image.ndim != 2:
                    raise SystemExit(
                        f"{hdf}: raw[{z}] must be 2-D, got {image.shape}"
                    )
                if image.dtype != np.uint8:
                    raise SystemExit(
                        f"{hdf}: raw[{z}] must be uint8, got {image.dtype}"
                    )

                stem = f'z{z:03d}'
                is_blank = not np.any(image)
                if is_blank and skip_blank:
                    summary['blank_skipped'] += 1
                    summary['slices'] += 1
                    print(
                        f'[CREMI] skip blank raw slice '
                        f'{volume_name}/{stem}.png'
                    )
                    continue

                boundary = boundary_mask(labels)
                image_gray_path = image_gray_dir / f'{stem}.png'
                image_rgb_path = image_rgb_dir / f'{stem}.png'
                mask_path = mask_dir / f'{stem}.png'
                gray_image = Image.fromarray(image)
                gray_image.save(image_gray_path)
                gray_image.convert('RGB').save(image_rgb_path)
                Image.fromarray((boundary * 255).astype(np.uint8)).save(
                    mask_path
                )

                base_row = {
                    'hdf': volume_name,
                    'z': z,
                    'source': _relative(Path(hdf), src_path),
                    'blank': int(is_blank),
                    'fg_ratio': f'{boundary.mean():.8f}',
                }
                volume_rows.append({
                    **base_row,
                    'image_gray': _relative(image_gray_path, volume_dir),
                    'image_rgb': _relative(image_rgb_path, volume_dir),
                    'mask': _relative(mask_path, volume_dir),
                })
                all_rows.append({
                    **base_row,
                    'image_gray': _relative(image_gray_path, out_path),
                    'image_rgb': _relative(image_rgb_path, out_path),
                    'mask': _relative(mask_path, out_path),
                })
                summary['written'] += 1
                summary['slices'] += 1
                if is_blank:
                    summary['blank_written'] += 1

                if (z + 1) % 25 == 0 or z + 1 == n_slices:
                    print(
                        f'[CREMI] {volume_name}: {z + 1}/{n_slices} slices '
                        f'({summary["written"]} written total)'
                    )

        with open(volume_dir / 'manifest.csv', 'w', newline='') as vf:
            writer = csv.DictWriter(vf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(volume_rows)

    with open(manifest_path, 'w', newline='') as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(
        f"[CREMI] done: {summary['written']} image/mask pairs from "
        f"{summary['volumes']} volume(s); "
        f"blank written={summary['blank_written']}, "
        f"blank skipped={summary['blank_skipped']}"
    )
    print(f"[CREMI] manifest: {manifest_path}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Export CREMI HDF5 image slices and neuron-boundary masks to PNG.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  # all unpadded HDF files under --src\n'
            '  python scripts/extract_cremi.py \\\n'
            '      --src /path/to/CREMI --out /path/to/cremi_png '
            '--skip-blank\n'
            '\n'
            '  # only selected HDF file(s); --src is optional\n'
            '  python scripts/extract_cremi.py \\\n'
            '      --hdf sample_A_20160501.hdf sample_C_20160501.hdf \\\n'
            '      --src /path/to/CREMI --out /path/to/cremi_png\n'
        ),
    )
    parser.add_argument(
        '--src', required=False,
        help='directory containing sample_*.hdf (searched recursively); '
             'optional when --hdf is given',
    )
    parser.add_argument(
        '--hdf', nargs='+', default=None,
        help='specific HDF file(s) to export; relative names are resolved '
             'against cwd and then --src',
    )
    parser.add_argument(
        '--out', required=True,
        help='output root; each HDF gets its own <hdf_name>/ folder',
    )
    parser.add_argument(
        '--skip-blank', action='store_true',
        help='skip all-zero raw slices (matches the current 373-pair CREMI set)',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='replace the selected per-HDF folders and manifest.csv',
    )
    args = parser.parse_args()

    if not args.src and not args.hdf:
        parser.error('provide --src and/or --hdf')
    if args.src and not Path(args.src).is_dir():
        parser.error(f'--src is not a directory: {args.src}')

    extract(
        args.src,
        args.out,
        hdf_paths=args.hdf,
        skip_blank=args.skip_blank,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
