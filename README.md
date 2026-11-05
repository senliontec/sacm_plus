# Skein

Skein is a PyTorch-based research framework for prompt-free curvilinear structure segmentation, built on the SACM baseline and the Segment Anything Model (SAM). The project provides custom training and testing scripts, and wraps SAM for easy extension and application.

> Skein is the working name for the improved framework. The official SACM baseline is self-contained under `src/baseline_zoo/models/sacm_official/` (registry id `sacm_official`); `--preset sacm` is the SACM-equivalent preset inside Skein. The corrected analysis variant is `src/baseline_zoo/models/sacm_plus/` (registry id `sacm_plus`, see `UPSTREAM.md`); it never replaces the official main-table entry. The local checkout/server directory may still be named `sacm_plus`/`sacm-plus`.
>
> **Model name: Skein.** Canonical preset `skein`; `full_fixed2_aux` is a compatibility alias. **Skein** is both the project/framework name and the deployed model name. The final deployment checkpoints are tagged `full_fixed2_gate`; their residual experts are frozen at exactly zero, so their forward output is the structural Skein model. Gate/residual-MoE and post-hoc repair are experimental diagnostics, not the deployed method.

## Features

- ViT-based image segmentation model (SAM)
- Custom dataset training and evaluation
- Automatic mask generation and prediction
- Detailed logging and evaluation metrics (F1, Precision, Recall)

## Directory Structure

```
segment_anything/         # SAM model and components
    __init__.py
    build_sam.py          # Build different SAM model variants
    predictor.py          # Inference and mask prediction
    automatic_mask_generator.py # Automatic mask generation
    modeling/             # Model architecture
    utils/                # Utility functions
src/train/trainer.py              # Training script
src/eval/evaluate.py               # Testing script
README.md
```

## Datasets

The frozen available-data protocol lives in `datasets/` (evaluation roots)
and `data/sacm_3shot/` (train/val/Seen split; the earlier five-domain draft
has been removed):

- **Train**: DRIVE, CHASEDB1, DCA1, CREMI, CrackTree200, CORN; 3-shot each = 18.
- **Val**: 1 per domain = 6 (CORN uses its official validation batch).
- **Seen**: held-out images from the five original training domains = 491,
  plus the official CORN test split = 152; total 643.
- **Unseen**: FIVES 200, STARE 20, XCAD 126, DeepCrack 237,
  HRF 45, LES-AV 22, CHCrack5k 990, OCTA-500 300 = 1,940.
- **Novel**: ROAD 49, LEAF 31, TYRE 36, WIRE 31, DeepGlobe 500,
  TopoMortar 350 = 997.

The final extended protocol therefore has **20 datasets** and
**3,580 test images** (643 Seen + 1,940 Unseen + 997 Novel).  DeepGlobe
uses a fixed seed-42 subset of its labeled train split; its official
valid/test splits have no GT.

This now matches the SACM paper on the two dataset-level training/evaluation
budgets we can reproduce: **6 domains × 3-shot = 18** training images and the
four-dataset Novel set (`ROAD, LEAF, TYRE, WIRE = 147`). The Seen/Unseen
composition still differs: DSCA is access-restricted, the official CRACK
split is unavailable, and the official DRIVE test set has no vessel GT, so
published numbers remain reference-only. CORN uses the author-provided
batch-disjoint official splits (train 1176 / val 188 / test 152; the paper
reports 100 Seen images, we use all 152). LEAF is the author-provided
official SACM set; its vein masks are thin and fragmented (median 305
8-connected components per image), so topology metrics on LEAF should be
reported with that annotation caveat.

Individual dataset converters live in `scripts/data/datasets/`. Each script can
rebuild one dataset independently; see `scripts/data/datasets/README.md`.

## Engineering quick start

```bash
make install   # pip install -e ".[dev]"
make smoke     # build the full model + one forward (first thing on a new machine)
make test      # unit tests: hand-computed metrics, loss smoke tests, configs, augmentation
make registry  # print the available topology-loss registry
```

Wandb logging is optional: set `WANDB_API_KEY` (or run `wandb login`)
before training/evaluation; no credential is stored in the source tree.

Engineering conventions: `docs/ENGINEERING.md`.  Reproducible seeding,
deterministic kernels, DataLoader workers, and environment manifests:
`docs/REPRODUCIBILITY.md`.  Optional post-processing:
`docs/DEEP_CLOSING.md`, `docs/SAM_REFINER.md`.

## Quick Start

1. Install dependencies:
   ```bash
   pip install torch torchvision scikit-learn pillow tqdm scipy matplotlib scikit-image networkx
   # optional topology-loss extras: gudhi torch-topological POT opencv-python monai
   ```

   `scikit-image` is recommended (not required): the clDice metric then
   uses `skimage.morphology.skeletonize`, exactly matching the official
   clDice repository convention; without it, a bundled Zhang-Suen
   implementation is used instead.

2. Train the model (presets reproduce the ablation matrix; the default
   `full` preset is the complete model):

   ```bash
   # Bug-fixed SACM baseline reproduction
   python src/train/trainer.py --preset sacm --data_root <root> --checkpoint <sam_vit_l.pth>

   # Stage 1: + coarse-to-fine, deep supervision, clDice loss, IoU head
   python src/train/trainer.py --preset stage1 --data_root <root> --checkpoint <sam_vit_l.pth>

   # Stage 2: + geometric strip adapters
   python src/train/trainer.py --preset stage2 --data_root <root> --checkpoint <sam_vit_l.pth>

   # Full model (default)
   python src/train/trainer.py --preset full --data_root <root> --checkpoint <sam_vit_l.pth>

   # Typed structural candidates (B prototype)
   python src/train/trainer.py --preset full_typed --data_root <root> --checkpoint <sam_vit_l.pth>
   ```

   Single-module ablations: `--preset no_geo_i / no_geo_e / no_c2f /
   no_fusion_v2 / no_multi_depth / no_cl / no_ds / no_iou`.

3. Test the model (architecture flags must match the trained checkpoint):

   ```bash
   python src/eval/evaluate.py --preset full --data_root <root> \
       --trained_weights best_model.pth --output_dir results/<name> \
       --selection iou --tta
   ```

   Evaluation reports 16 metrics (mean ± std) in `results_summary.txt`
   and per-image rows in `metrics.csv`: Dice / IoU / Precision / Recall /
   Sensitivity / Specificity / Accuracy / MCC / clDice / HD / HD95 /
   ASSD / ASD / RAVD / NSD / Betti error. Every metric is a
   strict-fidelity reimplementation of its official source (medpy,
   DeepMind surface-distance, or the official clDice repository) —
   conventions documented in `eval_metrics.py`. Add `--auc` for
   threshold-sweep robustness curves (dice_auc / cldice_auc, ~9x
   slower); `--nsd_tolerance` sets the NSD tolerance in pixels.
   `--selection gating` replicates the original SACM head-0 selection;
   `--selection iou` uses the trained IoU head.

## Unified results (baselines + Skein)

Use the wrappers for paper runs.  Every run, baseline or Skein, writes one
canonical directory:

```text
results/runs/<method>/<variant>/seed_<seed>/
  best_model.pth            # file checkpoints (best_model/ for dirs)
  checkpoint.json
  run_info.json
  metrics_history.json
  train.log
  predictions/              # baselines
  eval/
    <DATASET>/metrics.csv
    <DATASET>/summary.csv
    tables/all_metrics.csv
    tables/macro_metrics.csv
    tables/final_report.md
    run_manifest.json
    eval.log
    jobs.csv
```

- baselines: `bash scripts/run_train.sh --model unet --seeds 0` then
  `bash scripts/run_eval.sh --model unet --seeds 0`
- Skein: `SEED=0 PRESET=full_typed SKIP_TEST=true bash scripts/skein/run_skein_quick.sh`
  trains and promotes the best checkpoint; run
  `bash scripts/skein/eval_skein_20.sh` for the 20-dataset eval.
- index every run: `python scripts/pipeline/index_runs.py` →
  `results/runs_index.csv`

See `scripts/skein/README.md` for the complete commands.  The low-level
`src/train/trainer.py` / `src/eval/evaluate.py` examples below are direct
architecture entrypoints; they do not promote results into the canonical tree.

## New files

- `core/metrics/` — evaluation metrics (pixel/distance/surface/
  topology/persistence/aggregate); every metric is a strict-fidelity
  reimplementation of its official source (medpy, DeepMind
  surface-distance, official clDice, official engines). The root
  `eval_metrics.py` is a backward-compatibility shim.
- `losses.py` — DiceBCELoss and the differentiable SoftclDiceLoss,
  verified line-by-line against the official clDice repository
  (Shit et al., CVPR 2021): soft_open + residual-accumulation
  soft-skeleton, 10 iterations, smooth=1.0.
- `augmentation.py` — joint image-mask augmentation (flips, 90° rotations,
  elastic deformation; brightness/contrast jitter at p=0.2, automatically
  skipped for grayscale images so mixed-modality training splits work).
- `configs.py` — architecture/training presets for the ablation matrix.

## Topology losses (strict official ports)

`--topology_loss {none,decl,betti,dice_betti,wasserstein,
composed_wasserstein,centerline_ce,dice_cldice,ce_cldice,ce_clce,
satloss,topograph,dice_topograph,exact_topograph}` adds an extra
topology loss to the training objective (weight via
`--topology_loss_weight`):

- `decl` — DECL endpoint-connectivity loss (official port of
  DECL-Loss/loss/end_distance_loss.py). Internally combines its own
  Dice; keep the weight small (e.g. 0.1).
- `betti` — Betti matching loss (official engine vendored verbatim in
  `betti_matching.py` from the Betti-matching repository, Stucki et
  al.). Evaluated at 256x256: the pure-Python persistence engine is
  impractically slow at 1024 (the official repo trains on small crops).
- `centerline_ce` — centerline-CE loss (MICCAI 2024, official port of
  cldice_loss.py's dice_clCE_loss). Internally combines its own Dice;
  keep the weight small. Caveat: the pulled repo copy lacks its
  soft_skeleton.py; the verified official clDice skeleton is used (see
  topology_losses.py).
- `satloss` — spatial-aware persistent feature matching loss (ICCVW
  2025, official vendor in `satloss.py`). Requires `gudhi`,
  `torch-topological` and `POT`; without them the module imports
  cleanly but does not register. Evaluated at 256 resolution (gudhi
  persistence per iteration at 1024 is expensive).
- `topograph` — component-graph topology loss (official vendor in
  `core/losses/topograph.py`, pure-Python path use_c=False; the official
  C++ relabel-mask extension is optional and guarded). Requires
  `networkx` + `scipy` + Python >= 3.10; evaluates at 256 resolution
  (per-image graph construction is CPU-heavy). DiceTopographLoss and
  ExactTopographLoss are ported as well (`dice_topograph`,
  `exact_topograph`).

All topology losses live in `core/losses/` (one module per family,
registered on import); each is a registered entry, so adding more ports
does not touch the training script.

## Loss configuration notes

The base loss is `DiceBCELoss(bce=0.7, dice=0.3)`, i.e. L = BCE + 0.43·Dice.
SACM's paper reports λ = 0.4 while its own code uses the same 0.7/0.3
ratio (λ ≈ 0.43); the `sacm` preset deliberately follows the code (the
actual source of the published numbers). The new method adds Stage-1 deep
supervision (0.3), IoU-head supervision (1.0) and soft-clDice (0.5, with
a 10-epoch warmup) on top of the same base loss — report these exact
weights when writing up the method.

## P0 Skein improvements (2026-11)

The first high-ROI improvement pass is implemented as opt-in/controlled
switches in the Skein training/evaluation path:

- `--train_input_mode {resize,aspect_pad,native_crop}` / `--eval_input_mode {resize,aspect_pad}`:
  `aspect_pad` preserves the source aspect ratio and pads (no square
  stretch); `native_crop` keeps native pixel scale for large images via
  foreground-biased 1024x1024 crops.  Defaults are `aspect_pad` for both
  so training and evaluation use the same geometry.
- `--boundary_loss_weight W --boundary_loss_resolution R --boundary_loss_warmup N --boundary_loss_ramp N`:
  distance-transform boundary term targeting HD95/boundary outliers.
  The default is `0` (legacy); the Skein launcher enables `0.05`.
- `--empty_guard`: if the selected candidate is empty, fall back to the
  candidate with the largest foreground area (label-free; helps
  thick/dense domains where a head collapses).  The launcher enables it.
- `--tta`: existing four-way flip TTA.  Off by default for fair comparison;
  set `USE_TTA=true` for an X+TTA run and report it separately.
- `native_crop` is available for experiments, but final evaluation should
  currently use `aspect_pad`; native-resolution tiled inference is the next
  planned step, otherwise train/test pixel scale differs on high-res data.

The Skein launcher defaults are now:
`INPUT_MODE=aspect_pad`, `EVAL_INPUT_MODE=aspect_pad`, `FG_CROP_PROB=0.5`,
`BOUNDARY_LOSS_WEIGHT=0.05`, `EMPTY_GUARD=true`, `USE_TTA=false`.

### Typed structural heads (B prototype)

`--preset full_typed` is an opt-in architecture experiment, not the default.
It keeps four stable Stage-2 slots and zero-initialized type embeddings:

| slot | type | extra training signal |
|---:|---|---|
| 0 | thin | clDice / skeleton |
| 1 | blob | base Dice+BCE |
| 2 | boundary | distance-transform boundary loss |
| 3 | recall | Tversky (`alpha=0.3`, `beta=0.7`) |

The base Dice+BCE loss is applied to every typed head; the type-specific
terms only shape their specialisation. `stage2_supervision='typed'` requires
`use_typed_heads=true` (the preset sets both). The old `full` preset remains
untyped and is the control. Start with `full_typed` and compare the oracle
head and fixed-head results on validation before adding any ranking module.

Relevant flags: `--use_typed_heads {true,false}`,
`--stage2_supervision {wta,allhead,soft,typed}`,
`--tversky_weight`, `--tversky_alpha`, `--tversky_beta`.

Minimal validation-first loop (no test tuning):

```bash
SEED=0
RUN_ROOT="results/runs/skein/full_typed/seed_${SEED}"

GPU=7 PRESET=full_typed EPOCHS=100 SEED="$SEED" \
OUTPUT_ROOT="$RUN_ROOT" SKIP_TEST=true \
bash scripts/skein/run_skein_quick.sh
```

The one-click head-audit helpers were removed in the Skein cleanup.  Use
`eval_skein_20.sh` with `AUDIT_HEADS=true` / `AUDIT_THRESHOLDS=...` on a
validation split when you need those diagnostics.

### Fragmentation / candidate audit

`evaluate.py` now supports optional component diagnostics and post-processing:

- `--component_metrics` adds `components_pred/gt`, `component_count_error`,
  `small_component_rate`, `fp_area_ratio`, `fn_area_ratio` to the output.
- `--audit_heads` writes `head_audit.csv` (per-candidate Dice/IoU/P/R and
  component statistics).
- `--audit_thresholds -2 -1 0 1 2` writes `threshold_audit.csv` for
  calibration on validation data (do not tune on test).
- `--audit_head_thresholds -2 -1 0 1 2` writes
  `head_threshold_audit.csv`: every Stage-2 head at every threshold. This is
  the grid needed to compare fixed heads and thresholds without confounding
  the two.
- `--fixed_head {0,1,2,3}` forces one Stage-2 head for every image and
  overrides `--selection`; use it for head-calibration ablations, not final
  test tuning.
- `--save_candidate_masks` writes the four candidate masks under
  `<output>/candidates/`.
- `--component_prune_mode {off,absolute,relative}` with the
  `--component_prune_*` thresholds removes components that are both small
  and short in skeleton length.  It is opt-in; calibrate on validation,
  and do not apply a global rule to LEAF/TYRE/WIRE where GT itself is
  fragmented.
- `--selection quality` combines the IoU score with a small-component /
  component-count penalty.  Compare it against `iou` and `gating` on
  validation before using it for a report.

The one-click evaluation script forwards these through environment
variables, e.g.:

```bash
SEED=0
RUN_ROOT="results/runs/skein/full_typed/seed_${SEED}"
DATASETS="CREMI ROAD DeepGlobe TopoMortar OCTA-500 TYRE WIRE" \
AUDIT_HEADS=true COMPONENT_METRICS=true SAVE_CANDIDATE_MASKS=true \
AUDIT_THRESHOLDS="-2 -1 0 1 2" \
TRAIN_OUT="$RUN_ROOT" OUTPUT_ROOT="$RUN_ROOT/eval" \
PRESET=full_typed SEED="$SEED" GPU=7 \
bash scripts/skein/eval_skein_20.sh
```

For validation calibration, use the diagnostic flags above on the
val/support split (`AUDIT_THRESHOLDS`, `AUDIT_HEAD_THRESHOLDS`).  The
one-click calibration and selector-comparison helpers were removed in the
Skein cleanup.  The val split has only one image per seen dataset, so its
per-dataset winners are noisy; treat the macro table and the fixed-head
reference as evidence, not as an oracle.  Do not tune on the test split.

## Experiment tooling

```bash
# 1. Build the 3-shot train/val split (6 datasets x 3 shots, +1 val shot).
#    CORN uses its official validation/test splits:
python scripts/data/prepare_data.py \
    --train_dirs datasets/seen/DRIVE datasets/seen/CHASEDB1 datasets/seen/DCA1 \
                 datasets/seen/CREMI datasets/seen/CrackTree200 \
                 CORN:datasets/seen/CORN/train \
    --val_dirs CORN:datasets/seen/CORN/val \
    --test_dirs CORN:datasets/seen/CORN/test \
    --shots 3 --val_shots 1 --make_test \
    --out_dir data/sacm_3shot --seed 42

# 2. Motivation diagnostics# 4. Motivation diagnostics (head supervision + gating quality evidence)
python -m models.skein.diagnose --preset sacm --data_root /data/DRIVE \
    --trained_weights results/ablation/sacm/best_model.pth \
    --output_dir diagnosis/sacm_drive
```

## Baseline monitoring, paper statistics, and visualization

Every baseline adapter uses the shared W&B helper in
`src/core/wandb_utils.py` and writes the same prediction contract:

```text
results/runs/<method>/default/seed_<seed>/predictions/<method>/<dataset>/<stem>.png
```

Example for the U-Net adapter:

```bash
# One-click every available model/seed: train -> eval -> tables -> figures
# GPU=... is supported when --gpus is omitted; conflicting values fail early.
# --model all includes the appendix sam2_unet, which needs
# checkpoints/sam2_hiera_large.pt.
bash scripts/run_paper.sh --model all --seeds all --gpus 0,1,2,3 --resume

# One model/seed through the same registry
bash scripts/run_train.sh --model unet --seeds 0 --gpus 1
bash scripts/run_eval.sh --model unet --seeds 0 --gpus 1

# Regenerate every paper table and figure from existing metrics only
bash scripts/run_paper.sh --model all --seeds all --tables-only

# Direct adapter invocation with W&B monitoring
PYTHONPATH=src python -m baseline_zoo.run_one --model unet --stage train \
    --epochs 800 --use_wandb --wandb_mode online --wandb_project sacm

# Predict on a subset (for a full run use scripts/pipeline/run_baseline_test.py)
PYTHONPATH=src python -m baseline_zoo.run_one --model unet --stage predict \
    --test-datasets DRIVE:data/sacm_3shot_seed0/test/DRIVE \
                   WIRE:datasets/novel/WIRE

# Collect paper tables for the same subset (canonical eval root)
SEED=0
RUN_ROOT="results/runs/unet/default/seed_${SEED}"
python scripts/pipeline/collect_paper_metrics.py --methods unet \
    --datasets DRIVE WIRE --pred-root "$RUN_ROOT/predictions" \
    --out-root "$RUN_ROOT/eval" --layout canonical --family baseline \
    --variant default --seed "$SEED"

# Save qualitative panels for the same subset
python scripts/pipeline/visualize_predictions.py --methods unet \
    --datasets DRIVE WIRE --pred-root "$RUN_ROOT/predictions" \
    --out-root "$RUN_ROOT/visualizations" --num-samples 8

# One command: predict on all protocol test sets, collect canonical tables,
# and finalize the run.  Requires the trained checkpoint.
python scripts/pipeline/run_baseline_test.py --method unet --seed "$SEED"
```

Long test/evaluation runs print stage progress and keep a state file under
the run root at `.run_status/<method>.json`.

Prediction prints every 50 images; metric collection prints per dataset and
every 100 images.  Per-dataset metric results are cached
(`results/runs/<method>/default/seed_<seed>/eval/<dataset>/metrics.csv` +
`metrics_meta.json`), so an interrupted run resumes completed datasets
automatically.  Metric collection is parallel across datasets (`--workers`,
default `min(8, CPU count)`):

```bash
# Resume a run that stopped during prediction or metrics
python scripts/pipeline/run_baseline_test.py --method unet --skip-predict
python scripts/pipeline/run_baseline_test.py --method unet --skip-predict --workers 8

# Recompute everything (ignore the cache)
python scripts/pipeline/collect_paper_metrics.py --methods unet --no-resume
```

The collector finishes by printing a per-dataset summary table plus
Seen/Unseen/Novel/All macro rows; the same values are in
`eval/tables/all_metrics.csv` and `eval/tables/macro_metrics.csv` after
`finalize_run.py` promotes the run.

U-Net training prints a rich one-line table (header + one row per epoch)
and persists the same values next to its checkpoint:

```text
results/runs/unet/default/seed_<N>/metrics_history.json
    config + per-epoch overall and per-dataset metrics (AI-readable)
results/runs/unet/default/seed_<N>/val_metrics_by_dataset.csv
    one row per epoch; 6 datasets x Dice/IoU/clDice/HD95 = 24 metric columns
```

Per-dataset values are also logged to W&B every epoch as
`val/<dataset>/<metric>` (the four paper metrics; per-dataset empty rate is
kept in the JSON only).  W&B checkpoint-artifact upload is **off by default**
because large artifacts can stall the end of a run on the self-hosted server;
enable it with `WANDB_LOG_ARTIFACT=true`.  Shutdown waits at most
`WANDB_FINISH_TIMEOUT` seconds (default 60) before returning.

`collect_paper_metrics.py --layout canonical` reads the frozen protocol
manifest and writes per-image `eval/<dataset>/metrics.csv`, per-dataset
`summary.csv`, `eval/tables/all_metrics.csv`, `eval/tables/macro_metrics.csv`,
`eval/tables/final_report.md`, and `collect_stats.json`.
`finalize_run.py` then writes `eval/run_manifest.json`, `eval/eval.log`, and
`eval/jobs.csv`. `visualize_predictions.py` writes
`<run-root>/visualizations/<method>/<dataset>/<stem>.png` and `grid.png`
with input, GT, prediction, and TP/FP/FN error panels.

`src/models/skein/diagnose.py` produces the paper's motivation evidence:
per-head gradient norms under the original winner-take-all loss vs the
full loss (showing 3/4 Stage-2 heads were unsupervised), and the gating
rank vs ground-truth IoU correlation.

## Reference

- [Meta Segment Anything](https://github.com/facebookresearch/segment-anything)
