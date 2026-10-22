# Per-dataset preparation scripts

Each dataset has its own entry point so a single dataset can be rebuilt
without running the full organizer:

```bash
# Rebuild all 18 source pools (no 3-shot split) and print a summary table
bash scripts/datasets/prepare_all.sh

# Rebuild only selected datasets
bash scripts/datasets/prepare_all.sh DRIVE HRF TopoMortar

# Individual dataset
python scripts/datasets/prepare_DRIVE.py
python scripts/datasets/prepare_CREMI.py --max-images 4 --out /tmp/cremi_smoke
python scripts/datasets/prepare_HRF.py --overwrite
```

Defaults:

- `--src` points to the raw dataset under `../datasets/<name>`.
- `--out` points to the active role path under `datasets/{seen,unseen,novel}`.
- Existing `images/`, `masks/`, `fov/` or `manifest.csv` are not replaced
  unless `--overwrite` is passed.
- `--max-images N` is available for smoke tests and quick inspection.

Every script produces the same layout:

```text
<out>/
  images/
  masks/          single-channel PNG, 0/255
  fov/            optional (DRIVE, HRF)
  manifest.csv    source_image,image,mask,fov,fg_ratio
```

## Scripts

Each raw directory under `segment/datasets/` has exactly one script:

| Raw dataset | Script | Output role | Notes |
|---|---|---|---|
| `DRIVE` | `prepare_DRIVE.py` | `seen/DRIVE` | Training split only; official test has no vessel GT |
| `CHASEDB1` | `prepare_CHASEDB1.py` | `seen/CHASEDB1` | 1stHO observer masks |
| `DCA1` | `prepare_DCA1.py` | `seen/DCA1` | Coronary angiograms |
| `CREMI` | `prepare_CREMI.py` | `seen/CREMI` | Boundary target; skips all-zero raw slices |
| `cracktree200` | `prepare_cracktree200.py` | `seen/CrackTree200` | RGB variant |
| `FIVES` | `prepare_FIVES.py` | `unseen/FIVES` | `--split test` by default |
| `STARE` | `prepare_STARE.py` | `unseen/STARE` | `--labels ah|vk`, default `ah` |
| `XCAD` | `prepare_XCAD.py` | `unseen/XCAD` | Test split only; train is unlabeled |
| `DeepCrack` | `prepare_DeepCrack.py` | `unseen/DeepCrack` | `--split test` by default |
| `HRF` | `prepare_HRF.py` | `unseen/HRF` | Also writes `fov/` from ROI masks |
| `LES-AV` | `prepare_LES-AV.py` | `unseen/LES-AV` | Uses `vessel-segmentations/` |
| `CHCrack5k` | `prepare_CHCrack5k.py` | `unseen/CHCrack5k` | Test split; skips empty GT |
| `ROAD` | `prepare_ROAD.py` | `novel/ROAD` | Massachusetts Roads test split |
| `tyre` | `prepare_tyre.py` | `novel/TYRE` | In-house data |
| `wire` | `prepare_wire.py` | `novel/WIRE` | In-house data |
| `TopoMortar` | `prepare_TopoMortar.py` | `novel/TopoMortar` | Test split by default |
| `OCTA-500` | `prepare_OCTA-500.py` | `unseen/OCTA-500` | 300 paired 2D projections; `GT_LargeVessel` default |
| `DeepGlobe` | `prepare_DeepGlobe.py` | `unseen/DeepGlobe` | Train split with masks; valid/test have no GT |

TYRE/WIRE source masks are anti-aliased RGB; they are binarized at `>127`
to match the frozen protocol (not at `>0`).

The scripts do not create or modify `data/sacm_3shot_v0/`; that split is
built separately by `scripts/prepare_data.py`.

`LEAF` is intentionally not included: it was removed from the active
protocol and archived under `datasets_legacy/novel_LEAF`.
