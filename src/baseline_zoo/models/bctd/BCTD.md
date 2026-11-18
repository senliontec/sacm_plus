# BCTD-SAM

**BCTD-SAM** = *Boundary-Calibrated Topology Decoding*. It is a
self-contained baseline-zoo model. The package owns its SAM ViT-L variant,
fusion module, topology losses, boundary loss, decoding code and
adaptive-skeleton calibration. It does not import `methods.topostrand`,
StrandCal, DCD, SPA-SAM, CurviMix or SACM at runtime.

Registry id: `bctd`
Display name: `BCTD-SAM`
Checkpoint: `<run-root>/best_model_fixed3.pth`

## Recipe

Training is the TopoStrand-SAM width-calibrated four-candidate recipe plus a
distance-transform boundary loss:

- frozen SAM ViT-L with adapters;
- four Stage-2 candidates, Stage-1 gating, fixed candidate-3 deployment;
- dense + soft-clDice + Skeleton Recall + role losses + deep supervision;
- precision-heavy Tversky and foreground-area calibration on slot 3;
- structure-aware copy-paste, mild augmentation, 200 epochs;
- boundary loss on the Stage-2 boundary head (slot 2): weight 0.05,
  512 resolution, five-epoch warmup.

The boundary loss is

```
phi_G(x) = dist(x, foreground) - dist(x, background)
L_b      = sum(phi_G(x) * p(x)) / (sum(p(x)) + eps)
```

Inside the GT, `phi_G` is negative; outside it is positive. The loss pushes
probability mass inside the annotation and directly penalises predictions
far from the true boundary. It is training-only and adds no inference
parameters. The deployed head remains slot 3; the boundary objective
regularises shared decoder features.

Prediction adds:

1. four-view flip TTA;
2. resolution-consensus decoding (full view versus native-resolution tiles,
   routed by skeleton/area statistics);
3. close2;
4. Adaptive Skeleton Calibration (AS).

## AS

AS uses only the predicted binary mask. For each image:

```
f    = foreground pixels / image pixels
skel = skeletonize(mask)
w    = 2 * foreground pixels / skeleton pixels
```

It leaves medium-width, non-dense masks unchanged. Dense thick structures
are dilated by 2 px on the full mask, dense thin structures by 1 px, and
other eligible structures by a 1 px skeleton dilation unioned with the
mask. The rule is label-free, but its thresholds were selected on the
benchmark; validate them on a held-out split before a paper claim.

Ablation switches: `--as-calibration 0` disables AS; `--atc 1`
enables ATC; `--rcd 0` disables resolution-consensus decoding.

## Commands

Train four seeds from scratch and evaluate 20 datasets:

```
bash scripts/run_bctd_baseline_seeds03_200.sh
```

Reproduce from the existing method-tree BCTD checkpoints without
retraining (the script derives a bctd-labelled checkpoint, then evaluates):

```
MODE=repro SRC_RUN_ROOT=results/runs/topostrand/bctd200 \
  EVAL_GPUS=0,1 bash scripts/run_bctd_baseline_seeds03_200.sh
```

Direct baseline commands:

```
bash scripts/run_train.sh --model bctd --seeds 0 1 2 3 --gpus 0,1,2,3
bash scripts/run_eval.sh  --model bctd --seeds 0 1 2 3 --gpus 4,5,6,7
```

Convert an existing checkpoint manually:

```
python -m baseline_zoo.models.bctd.derive_fixed3 \
  --input results/runs/topostrand/bctd200/seed_0/best_model_fixed3.pth \
  --output results/runs/bctd/repro/seed_0/best_model_fixed3.pth \
  --deployment-head 3
```

The conversion preserves every model tensor, relabels the adapter to
`bctd`, and removes unused fusion tensors.

## Reference four-seed result

20 datasets (dataset-equal macro, 4 seeds):

```
Dice 61.93  IoU 48.29  clDice 70.14  HD95 59.23
```

12-dataset paper subset (8 Unseen + 4 Novel):

```
Dice 64.57  IoU 51.37  clDice 73.63  HD95 57.13
```

Per-seed AS result:

```
seed 0: 62.93 / 49.33 / 71.01 / 58.28
seed 1: 61.29 / 47.73 / 69.96 / 61.64
seed 2: 61.30 / 47.67 / 69.94 / 60.10
seed 3: 62.20 / 48.42 / 69.65 / 56.91
```

Reference TopoStrand+ATC: 61.48/47.90/69.91/59.64 (20 datasets) and
64.21/51.03/73.41/57.60 (12 datasets).

## Caveats

- AS thresholds are label-free but benchmark-selected.
- Macro gains are not uniform: 13/20 datasets improve Dice/IoU/clDice and
  12/20 improve HD95; report per-dataset wins.
- Paired per-dataset Wilcoxon vs ATC: Dice p=0.062, IoU p=0.049,
  clDice p=0.249, HD95 p=0.165. Do not claim every dataset improved or
  that every metric is individually significant.
- Seed 1 has weaker HD95.
- These are 200-epoch runs; do not mix them with 50-epoch pilots.
