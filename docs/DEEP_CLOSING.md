# Deep Closing post-processing

Deep Closing is integrated as an **optional post-processing refiner**, not
as a segmentation backbone.  It is applied after a model has produced a
binary mask and is particularly useful for broken thin tubular structures.

Paper: Qian Wu, Yufei Chen, Wei Liu, Xiaodong Yue, Xiahai Zhuang,
"Deep Closing: Enhancing Topological Connectivity in Medical Tubular
Segmentation", IEEE TMI, 2024.  Upstream code: `3rd/DeepClosing` (MIT).
Implementation/provenance: `src/core/postprocess/UPSTREAM.md`.

## Method kept from the paper

1. Train a MIM U-Net shape prior on **training masks only**.  Training uses
   random block masking, Recurrent TubeScaler augmentation and the
   topology-aware reconstruction loss.
2. At inference, `Deep Dilation` applies the shape prior to the predicted
   mask and thresholds it (default 0.2).
3. `Simple Component Erosion` labels the positive residue
   `dilated - prediction`, erodes simple points recursively, and keeps only
   non-simple components.  Original foreground is never removed.
4. The result is reported as `<method> + Deep Closing`.  It is not a new
   baseline architecture.

## One-click shape-prior training

The shape prior is trained on the 18 protocol training masks; validation
masks select the checkpoint.  Test masks are never read.

```bash
GPU=0 bash scripts/sacm_x/train_deep_closing.sh
```

Defaults: `TRAIN_ROOT=data/sacm_3shot_seed0`,
`OUTPUT=results/deep_closing/seed0_shape_prior.pth`, 300 epochs, 512
resolution.  Useful overrides:

```bash
GPU=0 TRAIN_ROOT=data/sacm_3shot_seed1 \
  OUTPUT=results/deep_closing/seed1_shape_prior.pth \
  EPOCHS=300 RESOLUTION=512 ALPHA=0.1 \
  bash scripts/sacm_x/train_deep_closing.sh
```

Smoke test:

```bash
GPU=0 EPOCHS=1 RESOLUTION=128 CHANNELS=16,32,64 \
  LIMIT_TRAIN=2 LIMIT_VAL=1 \
  bash scripts/sacm_x/train_deep_closing.sh
```

## Evaluate SACM-X with Deep Closing

Set the checkpoint without changing the model checkpoint:

```bash
GPU=0 PRESET=full \
  DEEP_CLOSING_CKPT=results/deep_closing/seed0_shape_prior.pth \
  bash scripts/sacm_x/eval_sacm_x_20.sh
```

The evaluator applies the refiner after FOV masking and before metrics. The
reported segmentation loss remains the raw model loss; Dice/IoU/clDice/
HD95/Betti and the saved `*_result.png` use the refined mask.  The output
directory gains `Deep Closing: ...` in `results_summary.txt` and W&B records
the checkpoint.

Quick train+eval path:

```bash
GPU=0 PRESET=full \
  DEEP_CLOSING_CKPT=results/deep_closing/seed0_shape_prior.pth \
  bash scripts/sacm_x/run_sacm_x_quick.sh
```

Fully one-click (train the shape prior if it is missing, then run SACM-X):

```bash
GPU=0 PRESET=full OUTPUT_ROOT=results/sacm_x_dc_seed0 \
  bash scripts/sacm_x/run_sacm_x_deep_closing.sh
```

Add `RETRAIN_DEEP_CLOSING=true` to force a new shape prior; use
`DEEP_EPOCHS`, `DEEP_RESOLUTION`, `DEEP_CHANNELS` to override its recipe.

## Apply to any existing prediction directory

This works for baseline predictions and for SACM-X predictions after the
fact:

```bash
PYTHONPATH=src python scripts/tools/apply_deep_closing.py \
  --input results/paper/samed/seed_0/predictions \
  --output results/paper/samed/seed_0/predictions_deep_closing \
  --checkpoint results/deep_closing/seed0_shape_prior.pth
```

Output is a 0/255 PNG tree with the same relative paths.  Use `--overwrite`
to replace existing outputs; `--threshold` controls binarization of the
input mask (default 0.5); `--dilation_threshold` overrides the shape-prior
threshold; `--resolution` and `--max_iters` override checkpoint inference
settings.

## Evaluation protocol

- Train shape prior on `train` masks only; use `val` for checkpoint
  selection; never train on `test`.
- Apply the same shape-prior checkpoint to base and post-processed runs;
  compare identical test sets, seeds, checkpoint and threshold.
- Report Dice/IoU **and** topology metrics (clDice, HD95, Betti, connected
  components).  Deep Closing mainly adds recall/connectivity and can
  slightly reduce precision; do not select it on test.
- Treat it as an ablation: `base` vs `base + Deep Closing`.  Apply it to
  all compared methods if it is used as a common post-processing stage;
  otherwise label it as part of the proposed method.

## Caveats

- The shape prior is trained from 18 seen-domain masks; generalization to
  unseen/novel domains is empirical and must be measured.
- It never removes original foreground, so it can add false positives in
  regions the shape prior over-dilates.  Validate the dilation threshold
  (0.2 default) and report precision/recall.
- The port uses a 512 square working resolution and nearest resizing by
  default; the official implementation uses 224 crops and sliding windows.
  See `UPSTREAM.md`.
- The upstream repository is incomplete and releases no weights; our
  checkpoint is trained from scratch on the protocol masks.
