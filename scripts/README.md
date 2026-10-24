# scripts/

Two one-click entry points cover the paper workflow.

## 1. All experiments (train + evaluate + tables + figures)

```bash
bash scripts/run_paper.sh --model all --seeds all --gpus 0,1,2,3 --resume
```

`--model all` resolves every `status: available` model in
`configs/models/` (SACM-X plus the 13 runnable baselines; SAM2-UNet is
`reference_only`).  For each (model, seed) job it runs the config's
`train_entrypoint`, then `eval_entrypoint`, and finally writes the
cross-model tables and figures.

Subsets and short runs:

```bash
bash scripts/run_paper.sh --model unet --seeds 0 --gpus 2
bash scripts/run_paper.sh --model unet,sacm_x --seeds 0 1 --gpus 2,3
bash scripts/run_paper.sh --model all --seeds all --dry-run
```

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

## Data preparation (one-time)

```bash
# Frozen seed-42 3-shot protocol already in data/sacm_3shot/
bash scripts/make_shot3_split.sh 0 1 2 3 4   # five seeded splits

# Rebuild raw dataset layout / manifests
python scripts/organize_datasets.py --src ../datasets --out datasets
python scripts/group_datasets.py --root datasets
python scripts/build_protocol_manifest.py --split-root data/sacm_3shot \
    --datasets-root datasets --seed 42
```

`scripts/datasets/prepare_*.py` build individual official test pools;
`prepare_data.py`, `make_shot3_split.sh` and `build_protocol_manifest.py`
write the train/val/test manifests.

## SACM-X specific

`run_sacm_x_quick.sh` and `eval_sacm_x_20.sh` are the commands embedded in
`configs/models/sacm_x.yaml`; they use the SACM-X training/evaluation engine
(`run_experiments.py`) and its report path (`aggregate_results.py`,
`make_final_report.py`).  `run_ablation.sh` runs the SACM-X preset ablation
matrix and uses the same engine.

## Evaluation helpers

| Script | Purpose |
|---|---|
| `run_baseline_test.py` | predict + paper metrics + visualization for one baseline |
| `collect_paper_metrics.py` | per-dataset Dice/IoU/clDice/HD95 from prediction PNGs |
| `show_paper_metrics.py` | print a completed paper-metric run |
| `visualize_predictions.py` | qualitative panels for baseline predictions |
| `visualize_sacm_x_all.*`, `visualize_sacm_x_errors.*` | SACM-X mask/overlay and worst-case panels |
| `run_status.py` | live progress of a `run_baseline_test.py` job |
| `protocol_specs.py` | single `NAME:TEST_ROOT` parser for evaluation |

## Setup and maintenance

| Script | Purpose |
|---|---|
| `setup_baselines.sh` | install every baseline extra (`pip install -e .[...]`) |
| `diagnose_selection.py` | checkpoint/selection diagnostics |
| `verify_against_official.py` | numerical cross-check against official repos (`make verify`) |
| `kill_runs.sh` | stop this user's training/eval processes |
| `clean_pycache.sh` | remove `__pycache__`/`.pyc` (`make clean`) |
| `wandb_env.sh` | shared W&B environment/credential check |
