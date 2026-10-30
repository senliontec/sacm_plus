# scripts/

Top level contains only the one-click entry points.  Everything else lives
in four subdirectories.

## 1. All experiments (train + evaluate + tables + figures)

```bash
bash scripts/run_paper.sh --model all --seeds all --gpus 0,1,2,3 --resume
```

`--model all` resolves every `status: available` model in
`configs/models/` (the registered official/reference baselines plus the
SACM-X research framework, including the
appendix `sam2_unet`, the prompt-free `sam_adapter`/`samed`/`topolora`/
`convlora`/`c2p`/`a2sam`/`kprism`, the non-SAM `jtfn`/`dscnet`/`rsrm`/`dka`/
`scrwkv`/`flowsdf`/`tpsegformer`/`s2mnet`/`csunet`/`liot`/`dconnnet`,
the support-based
`persam`/`matcher`/`fssam`/`sansa`, and the oracle-prompted `ucs`).  For
each (model, seed)
job it runs the config's
`train_entrypoint`, then `eval_entrypoint`, and finally writes the
cross-model tables and figures.

Before training, `run_paper.sh` always clears compute processes on the GPUs
selected by `--gpus` via `tools/clear_gpus.sh` (destructive; other GPUs are
never touched). Pass `--no-kill` to skip cleanup; `--kill-gpu` is retained
for backward compatibility. `--dry-run` and `--tables-only` never kill
anything.
`GPU=<indices>` is accepted when `--gpus` is omitted; setting both to
different values fails before any cleanup.

Subsets and short runs:

```bash
bash scripts/run_paper.sh --model unet --seeds 0 --gpus 2
bash scripts/run_paper.sh --model unet,sacm_official --seeds 0 1 --gpus 2,3
bash scripts/run_paper.sh --model all --seeds all --dry-run
```

### A2SAM read-only triage

After an A2SAM run, diagnose branch-cap saturation and DCA1 empty predictions
without retraining:

```bash
GPU=0 bash scripts/tools/run_a2sam_triage.sh
# optional: --a2sam-datasets all --splits val
```

It reads `train`/`val` only and writes `branch_stats.csv`,
`branch_sweep.csv`, and `summary.json` under
`results/diagnostics/a2sam_seed_0/`. The checkpoint/data/output paths can be
overridden with `CHECKPOINT`, `DATA_ROOT`, and `OUTPUT`.

## W&B

Baseline training and evaluation default to W&B. `run_train.sh` and
`run_eval.sh` source `scripts/tools/wandb_env.sh` and require credentials
when `WANDB_MODE` is online; missing credentials fail fast with a clear
message. Use `WANDB_MODE=offline` for local W&B files or
`WANDB_MODE=disabled` to turn it off. Direct `run_one` also defaults to
W&B; pass `--use_wandb false` to disable it for a single run.

## cue-ml GPU queue

`tools/run_paper_cue.sh` is the dynamic-GPU alternative to the static
round-robin in `run_paper.sh`.  It requires `pip install cue-ml` and an idle
set of GPUs (cue does not detect existing VRAM users; the wrapper preflights
`nvidia-smi` and refuses to start while compute processes exist).

```bash
# one-click train queue -> eval queue -> tables/figures
bash scripts/tools/run_paper_cue.sh --seeds 0

# force the two repaired models to rerun even though their old markers exist
bash scripts/tools/run_paper_cue.sh --models samed,dscnet --seeds 0 --no-resume

# generate/validate all YAML without running models
bash scripts/tools/run_paper_cue.sh --seeds 0 --dry-run
```

Each cue job is dispatched through `tools/cue_job.py`, which uses the GPU
assigned by `CUDA_VISIBLE_DEVICES` and maps it to `--device cuda:0`.

## Optuna tuning

`tools/tune_baselines.py` reads per-model sections from
`configs/tuning/spaces.yaml`, runs validation-only trials in isolated
`results/tuning/trials/` workdirs, and writes the best parameters to
`results/tuning/<model>/best_<protocol>_seed<seed>_<selection>_ep<N>.json`
(where `<N>` is the effective budget: `--epochs` if provided, otherwise the
model config default).  Install with `pip install optuna`.

```bash
# inspect spaces without optuna / without running
python scripts/tools/tune_baselines.py --models all --dry-run

# screen a model on one GPU, then rerun the selected config at full budget
python scripts/tools/tune_baselines.py --models dscnet --trials 8 --gpus 0

# parallel trials on four GPUs, using a shorter search budget
python scripts/tools/tune_baselines.py --models unet,samed --trials 12 \
    --gpus 0,1,2,3 --epochs 50

# batch seeds 0-4; each (model, seed) gets its own study/best file
python scripts/tools/tune_baselines.py --models all --seeds 0 1 2 3 4 \
    --trials 8 --gpus 0,1,2,3,4,5,6
```

### One-click batch HPO on GPU 0-6

After uploading the code to the server:

```bash
bash scripts/tools/run_optuna_paper.sh
```

It defaults to all paper baselines, **seed `0`**, 8 trials per
(model, seed), and physical GPUs `0,1,2,3,4,5,6`.  GPU 7 is left free:
the script never queries it or includes it in `CUDA_VISIBLE_DEVICES`.
It refuses to start if any selected GPU already
has more than 512 MiB in use, and installs Optuna if it is missing.

Useful overrides:

```bash
# one-click smoke / inspect the plan
bash scripts/tools/run_optuna_paper.sh --dry-run

# smaller search, then detach to a timestamped log
bash scripts/tools/run_optuna_paper.sh --models dscnet,unet \
    --seeds 0 --trials 4 --epochs 50 --detach

# custom interpreter or storage; per-trial weights are deleted by default
PYTHON=/path/to/tsai/bin/python STORAGE=results/tuning/optuna.db \
    bash scripts/tools/run_optuna_paper.sh

# keep trial weights for debugging and cap a stuck trial at 6 hours
bash scripts/tools/run_optuna_paper.sh --keep-trial-checkpoints \
    --trial-timeout 21600
```

The runner dynamically assigns each trial to the next free GPU in the
selected pool.  `nnWNet` and `KnowSAM` are skipped because their custom
official pipelines do not yet expose the shared scheduler/LR interface;
`Matcher`, `PerSAM`, `RSRM` and `C2P` are training-free or zero-shot.
Thus 24 of the official/reference baselines are in the tuning harness.
The baseline registry contains official or reference implementations only;
project-specific models are kept outside `baseline_zoo`. Newly added ports
must be added to the explicit `BASELINE_MODELS` list and
`configs/tuning/spaces.yaml` before `--models all` can tune them.  Trial
`best_model*.pth` files are deleted after scoring unless
`--keep-trial-checkpoints` is passed; the metrics history and trainer log
are kept.  The selected configuration must be applied to a fresh
full-budget run before reporting.  Test data is never used by the tuning
runner.

## 2. Paper tables and figures only

```bash
bash scripts/run_paper.sh --model all --seeds all --tables-only
```

This skips train/eval and regenerates everything under
`results/paper/tables/` and `results/paper/figures/` from existing
metrics.  Missing runs fail the report unless `--allow-missing` is passed.

## Stage wrappers

| Script | Purpose |
|---|---|
| `run_train.sh` | `python -m baseline_zoo train` |
| `run_eval.sh` | `python -m baseline_zoo eval` (never trains) |
| `run_report.sh` | `python -m baseline_zoo report` (tables only) |
| `run_figures.sh` | `python -m baseline_zoo.figures` (figures only) |

All four accept `--model`, `--seeds`, `--gpus`, `--protocol`, `--resume`,
`--dry-run`; `run_report.sh` also accepts `--allow-missing`.

## Directory layout

```text
scripts/
├── run_paper.sh            # one-click all experiments
├── run_sacm_vitl_compare.sh # official SACM + SACM-X presets (ViT-L)
├── run_train.sh            # stage wrappers
├── run_eval.sh
├── run_report.sh
├── run_figures.sh
├── pipeline/               # evaluation helpers used by the configs
│   ├── run_baseline_test.py
│   ├── collect_paper_metrics.py
│   ├── visualize_predictions.py
│   ├── metrics.py
│   └── protocol_specs.py
├── sacm_x/                 # SACM-X research framework entrypoints
│   ├── run_sacm_x_800.sh   # one-click 800-epoch ViT-L training + W&B
│   ├── run_sacm_x_quick.sh
│   ├── eval_sacm_x_20.sh
│   ├── run_experiments.py
│   ├── aggregate_results.py
│   ├── make_final_report.py
│   ├── run_ablation.sh
│   ├── metrics_report.py
│   ├── kill_runs.sh
│   └── wandb_env.sh
├── data/                   # one-time dataset/protocol construction
│   ├── prepare_data.py
│   ├── organize_datasets.py
│   ├── group_datasets.py
│   ├── build_protocol_manifest.py
│   ├── make_shot3_split.sh
│   └── datasets/           # per-dataset official converters
└── tools/
    ├── setup_baselines.sh
    ├── clear_gpus.sh
    ├── verify_against_official.py
    ├── wandb_env.sh
    ├── night_runner.py
    └── clean_pycache.sh
```

## SACM-X research framework

`sacm_x/` is the SACM-X training/evaluation engine; it is separate from the
official/reference baseline adapters. Its model code lives in
`src/models/sacm_x`, `src/train` and `src/eval`, and its registry entry is
`configs/models/sacm_x.yaml`.

### Skeleton Recall topology arm

SACM-X includes an opt-in Skeleton Recall loss (ECCV 2024, Apache-2.0
adaptation in `src/core/losses/skeleton_recall.py`). The `full` preset is
unchanged; topology presets are explicit:

| Preset | Skeleton Recall weight | Tube |
|---|---:|---|
| `full` | 0.0 | — |
| `full_srec` | 0.2 | yes |
| `full_srec_low` | 0.1 | yes |
| `full_srec_high` | 0.5 | yes |
| `full_srec_notube` | 0.2 | no |

The loss runs on the tubed GT skeleton at 1024 and is warmed up through the
existing `skeleton_recall_warmup/ramp` schedule (the presets start at epoch 0
and ramp over five epochs).

One-click ViT-L comparison on the server (official SACM on GPU0; one SACM-X
preset per subsequent GPU):

```bash
# Baseline only (backward-compatible default)
bash scripts/run_sacm_vitl_compare.sh

# Topology arm: official SACM + SACM-X full + SACM-X full_srec
SACM_X_PRESETS="full full_srec" bash scripts/run_sacm_vitl_compare.sh

# Weight sweep on three GPUs
SACM_X_PRESETS="full full_srec_low full_srec_high" \
  SACM_X_GPUS="1 2 3" bash scripts/run_sacm_vitl_compare.sh
```

The script trains every preset on the same seed/split, evaluates all 20
datasets, and writes one aggregated table under
`results/sacm_vitl_compare/seed<SEED>/sacm_x/tables/`.

### 800-epoch training run (W&B, no test evaluation)

```bash
# Default: ViT-L full, GPU0, seed0, 800 epochs, validation every epoch
bash scripts/sacm_x/run_sacm_x_800.sh

# Topology arm
PRESET=full_srec GPU=1 SEED=0 bash scripts/sacm_x/run_sacm_x_800.sh

# 24GB-card fallback if full OOMs
PRESET=stage2 GPU=0 bash scripts/sacm_x/run_sacm_x_800.sh
```

Outputs: `results/sacm_x_vitl_800/seed<SEED>/<PRESET>/best_model.pth` and
`metrics_history.json`; the W&B run is in entity `buaazqk`, project `sacm`,
run name `sacm_x_<MODEL>_<PRESET>_seed<SEED>`, group
`sacm_x_<MODEL>_<PRESET>_800_seed<SEED>`. W&B logs exactly the baseline
monitor keys: global `val/dice`, `val/iou`, `val/cldice`, `val/hd95`,
`val/empty_prediction_rate`, `val/selection`, plus per-domain
`val/<DATASET>/{dice,iou,cldice,hd95}`. The full metric suite stays in
`metrics_history.json`. Override `CKPT`, `TRAIN_ROOT`, or `OUTPUT_ROOT` for
a different server layout.

## Data preparation (one-time)

```bash
# Frozen seed-42 3-shot protocol already in data/sacm_3shot/
bash scripts/data/make_shot3_split.sh 0 1 2 3 4   # five seeded splits

# Rebuild raw dataset layout / manifests
python scripts/data/organize_datasets.py --src ../datasets --out datasets
python scripts/data/group_datasets.py --root datasets
python scripts/data/build_protocol_manifest.py --split-root data/sacm_3shot \
    --datasets-root datasets --seed 42
```

`scripts/data/datasets/prepare_*.py` build individual official test pools;
`prepare_data.py`, `make_shot3_split.sh` and `build_protocol_manifest.py`
write the train/val/test manifests.

## Notes

- SegDINO needs DINOv3 weights only for training.  The adapter auto-detects
  a size-matched `.bin`/`.pth`/`.pt`/`.safetensors` file under
  `checkpoints/` (e.g. `checkpoints/dinov3/pytorch_model.bin`), so no
  environment variable is required.
- `clear_gpus.sh` stops compute processes on the selected GPU(s);
  `run_paper.sh` invokes it by default for every train/eval run; `--no-kill`
  skips cleanup (`--kill-gpu` is accepted for backward compatibility).
- `setup_baselines.sh` installs every baseline extra;
  `verify_against_official.py` is the `make verify` cross-check;
  `clean_pycache.sh` is `make clean`.
