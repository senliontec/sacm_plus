# BCTD-SAM — Boundary-Calibrated Topology Decoding

BCTD is the strongest current model in the project. It is an evolution of
TopoStrand-SAM, not a baseline-zoo modification.

## Method

- Base: frozen SAM ViT-L + four-candidate decoder, WCD (precision Tversky +
  foreground-area loss on slot 3), RCD resolution routing, structure-aware
  copy-paste, fixed3 deployment, flip TTA, close2.
- Training addition: distance-transform boundary loss on the Stage-2
  **boundary head (head 2)**: weight 0.05, resolution 512, warmup 5 epochs.
- Inference addition: **adaptive-skeleton calibration (AS)**, a label-free
  1/2-px skeleton dilation gated by predicted width/foreground.
- No target labels, supports, or test-time tuning. AS thresholds are
  label-free but inherited/benchmark-selected; validate on a held-out split
  before a paper claim.

## Result (4 seeds, 200 epochs, 20 datasets)

| Method | Dice | IoU | clDice | HD95 |
|---|---:|---:|---:|---:|
| TopoStrand + ATC | 61.48 | 47.90 | 69.91 | 59.64 |
| BCTD base (boundary loss) | 61.93 | 48.29 | 69.61 | 58.89 |
| **BCTD + AS** | **61.93** | **48.29** | **70.14** | **59.23** |

12-dataset paper subset (8 Unseen + 4 Novel): **64.57 / 51.37 / 73.63 /
57.13**, vs TopoStrand+ATC 64.21 / 51.03 / 73.41 / 57.60.

Per-seed BCTD+AS:

- seed 0: 62.93 / 49.33 / 71.01 / 58.28
- seed 1: 61.29 / 47.73 / 69.96 / 61.64
- seed 2: 61.30 / 47.67 / 69.94 / 60.10
- seed 3: 62.20 / 48.42 / 69.65 / 56.91

The macro gains are not uniform: 13/20 datasets improve Dice/IoU/clDice and
12/20 improve HD95; paired per-dataset Wilcoxon p-values are Dice 0.062,
IoU 0.049, clDice 0.249, HD95 0.165. Do not claim every dataset improved.
See `results/reports/topostrand_bctd200_4seed_20.md` for the full table.

## Reproduction

Train/evaluate base BCTD:

```bash
bash scripts/run_bctd_seeds03_200.sh
```

The training command is the TopoStrand command plus:

```text
--boundary-loss-weight 0.05 --boundary-loss-head boundary
--boundary-loss-resolution 512 --boundary-loss-warmup-epochs 5
```

Apply AS after base evaluation:

```bash
python scripts/experiments/topostrand/transform_predictions.py \
  --src results/runs/topostrand/bctd200/seed_<s>/predictions_bctd200 \
  --dst results/runs/topostrand/bctd200/seed_<s>/predictions_bctd200_as \
  --transform adaptive_skel
python scripts/pipeline/collect_paper_metrics.py \
  --methods topostrand \
  --pred-root results/runs/topostrand/bctd200/seed_<s>/predictions_bctd200_as \
  --out-root results/runs/topostrand/bctd200/seed_<s>/eval_bctd200_as \
  --metrics dice,iou,cldice,hd95 --empty-metric-policy penalty \
  --protocol data/sacm_3shot_seed<s>/protocol_manifest.csv \
  --layout canonical --family baseline --variant topostrand --seed <s>
```

Do not reintroduce a learned per-image selector; keep the fixed3/RCD/TTA
inference contract.
