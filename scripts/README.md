# scripts/

Top level contains only the one-click entry points.  Everything else lives
in four subdirectories.

## 1. All experiments (train + evaluate + tables + figures)

```bash
bash scripts/run_paper.sh --model all --seeds all --gpus 0,1,2,3 --resume
```

`--model all` resolves every `status: available` model in
`configs/models/` (SACM-X plus the 21 runnable baselines, including the
appendix `sam2_unet`, the prompt-free `sam_adapter`/`samed`/`topolora`, the
non-SAM `jtfn`/`dscnet`, the support-based `persam`/`matcher`, and the
oracle-prompted `ucs`).  For each (model, seed)
job it runs the config's
`train_entrypoint`, then `eval_entrypoint`, and finally writes the
cross-model tables and figures.

Before training, `run_paper.sh` clears compute processes on the GPUs
selected by `--gpus` via `tools/clear_gpus.sh`; other GPUs are not touched.
Pass `--no-kill` to skip cleanup; `--dry-run` and `--tables-only` never
kill anything.

Subsets and short runs:

```bash
bash scripts/run_paper.sh --model unet --seeds 0 --gpus 2
bash scripts/run_paper.sh --model unet,sacm_x --seeds 0 1 --gpus 2,3
bash scripts/run_paper.sh --model all --seeds all --dry-run
```

## W&B

Baseline training and evaluation default to W&B. `run_train.sh` and
`run_eval.sh` source `scripts/sacm_x/wandb_env.sh` and require credentials
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
(model, seed), and physical GPUs `0,1,2,3,4,5,6`.  GPU 7 is reserved for
SACM-X: the script never queries it or includes it in
`CUDA_VISIBLE_DEVICES`.  It refuses to start if any selected GPU already
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
`Matcher` and `PerSAM` are training-free.  Thus 17 of the 21 baselines are
tuned.  Trial `best_model*.pth` files are deleted after scoring unless
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
├── sacm_x/                 # SACM-X training/eval engine
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
    └── clean_pycache.sh
```

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

## SACM-X specific

`run_sacm_x_quick.sh` and `eval_sacm_x_20.sh` are the commands embedded in
`configs/models/sacm_x.yaml`; they use the SACM-X training/evaluation engine
(`run_experiments.py`) and its report path (`aggregate_results.py`,
`make_final_report.py`).  `run_ablation.sh` runs the SACM-X preset ablation
matrix and uses the same engine.

## Notes

- SegDINO needs DINOv3 weights only for training.  The adapter auto-detects
  a size-matched `.bin`/`.pth`/`.pt`/`.safetensors` file under
  `checkpoints/` (e.g. `checkpoints/dinov3/pytorch_model.bin`), so no
  environment variable is required.
- `clear_gpus.sh` stops compute processes on the selected GPU(s);
  `run_paper.sh` passes `--gpus` to it before training and accepts
  `--no-kill` to opt out.
- `setup_baselines.sh` installs every baseline extra;
  `verify_against_official.py` is the `make verify` cross-check;
  `clean_pycache.sh` is `make clean`.
