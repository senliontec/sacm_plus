# StrandCal-SAM

**StrandCal-SAM** = *Topology-Strand Segmentation with Adaptive Topology
Calibration*.  It is a self-contained baseline: the package owns its SAM
ViT-L variant, fusion module, topology losses, CAPE and decoding code.  It
does not import `methods.topostrand` or any other baseline model at runtime.

Registry id: `strandcal`
Display name: `StrandCal-SAM`
Checkpoint: `<run-root>/best_model_fixed3.pth`

## Recipe

Training is the width-calibrated four-candidate recipe inherited from
TopoStrand-SAM:

- frozen SAM ViT-L with adapters;
- four Stage-2 candidates, Stage-1 gating, fixed candidate-3 deployment;
- dense + soft-clDice + Skeleton Recall + role losses + deep supervision;
- precision-heavy Tversky and foreground-area calibration on slot 3;
- structure-aware copy-paste, mild augmentation, 200 epochs.

Prediction adds three label-free stages:

1. four-view flip TTA (optional; canonical eval uses it);
2. resolution-consensus decoding (full view versus native-resolution tiles,
   routed by skeleton/area statistics);
3. Adaptive Topology Calibration (ATC): density/width-gated skeleton
   dilation by 1-2 px.

## ATC

ATC uses only the predicted binary mask.  For each image:

```
f = foreground pixels / image pixels
skel = skeletonize(mask)
w = 2 * foreground pixels / skeleton pixels
```

Branches, in order:

1. `f < 0.5%` -> skeleton dilation 2 px;
2. `w >= 10` and `1% <= f < 10%` -> 2 px;
3. `f >= 15%` and `w <= 15` -> 2 px;
4. `f >= 4%` and `w < 7.5` -> 1 px;
5. `f >= 15%` and `w > 15` -> 1 px;
6. otherwise unchanged.

Output: `mask OR dilated_skeleton`.  The rule is label-free but its thresholds
were selected on the benchmark; validate on a held-out split before a paper
claim.

## Commands

One-click server run (train 4 seeds, derive fixed3, evaluate 20 datasets):

```
bash scripts/run_strandcal_seeds03_200.sh
```

Direct baseline commands:

```
bash scripts/run_train.sh --model strandcal --seeds 0 1 2 3 --gpus 0,1,2,3
bash scripts/run_eval.sh  --model strandcal --seeds 0 1 2 3 --gpus 4,5,6,7
```

Convert an existing TopoStrand checkpoint for evaluation:

```
python -m baseline_zoo.models.strandcal.derive_fixed3 \
  --input results/runs/topostrand/final200_pa/seed_0/best_model_fixed3.pth \
  --output results/runs/strandcal/<variant>/seed_0/best_model_fixed3.pth \
  --deployment-head 3
```

The rewrite sets `payload["adapter"] = "strandcal"` and removes unused
fusion tensors; candidate-head weights are unchanged.

Ablate ATC/RCD at evaluation:

```
--atc 0 --rcd 0 --no-tta-flip
```

## Reference four-seed result

These numbers come from the original TopoStrand+ATC prediction artifacts;
this self-contained baseline should be re-evaluated to generate its own table.

20 datasets (dataset-equal macro):

```
Dice 61.48  IoU 47.90  clDice 69.91  HD95 59.64
```

12-dataset paper subset (8 Unseen + 4 Novel):

```
Dice 64.21  IoU 51.03  clDice 73.41  HD95 57.60
```

CurviMix-SAM reference: 60.47/46.88/69.31/60.49 (20) and
62.79/49.56/72.83/58.45 (12).
