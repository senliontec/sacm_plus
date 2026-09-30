# SACM

SACM is a PyTorch-based image segmentation project that integrates the Segment Anything Model (SAM) for both training and inference. The project provides custom training and testing scripts, and wraps SAM for easy extension and application.

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

This project uses **4 internal datasets** for evaluation. The datasets can be found at:

- [Curvilinear_Structure_Datasets](https://github.com/dianshuoli/Curvilinear_Structure_Datasets)

## Engineering quick start

```bash
make install   # pip install -e ".[dev]"
make smoke     # build the full model + one forward (first thing on a new machine)
make test      # unit tests: hand-computed metrics, loss smoke tests, configs, augmentation
make verify    # numerical cross-check against the official repos (CPU only)
make registry  # print the available topology-loss registry
```

Engineering conventions: `docs/ENGINEERING.md`.

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

## Experiment tooling

```bash
# 1. Build the 3-shot train/val split (6 datasets x 3 shots, +1 val shot)
python scripts/prepare_data.py --train_dirs <d1> ... <d6> --shots 3 --val_shots 1 \
    --out_dir data/sacm_3shot --seed 42

# 2. Run the whole ablation matrix on multiple GPUs (train + test per job)
python scripts/run_experiments.py --gpus 0 1 2 3 4 5 6 7 \
    --checkpoint /path/sam_vit_l_0b3195.pth --train_root data/sacm_3shot \
    --test_datasets DRIVE:/data/DRIVE CHASEDB1:/data/CHASEDB1 \
    --presets sacm stage1 stage2 full --output_root results --epochs 50 --tta

# 3. Aggregate per-job summaries into CSV + LaTeX tables
python scripts/aggregate_results.py --root results --out results/tables

# 4. Motivation diagnostics (head supervision + gating quality evidence)
python -m models.sacm.diagnose --preset sacm --data_root /data/DRIVE \
    --trained_weights results/sacm/DRIVE/best_model.pth \
    --output_dir diagnosis/sacm_drive
```

`diagnose_heads.py` produces the paper's motivation evidence:
per-head gradient norms under the original winner-take-all loss vs the
full loss (showing 3/4 Stage-2 heads were unsupervised), and the gating
rank vs ground-truth IoU correlation.

## Reference

- [Meta Segment Anything](https://github.com/facebookresearch/segment-anything)
