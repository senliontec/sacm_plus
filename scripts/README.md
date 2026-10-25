# scripts/

Top level contains only the one-click entry points.  Everything else lives
in four subdirectories.

## 1. All experiments (train + evaluate + tables + figures)

```bash
bash scripts/run_paper.sh --model all --seeds all --gpus 0,1,2,3 --resume
```

`--model all` resolves every `status: available` model in
`configs/models/` (SACM-X plus the 16 runnable baselines, including the
appendix `sam2_unet`, the prompt-free `sam_adapter` and the oracle-prompted
`ucs`).  For each (model, seed)
job it runs the config's
`train_entrypoint`, then `eval_entrypoint`, and finally writes the
cross-model tables and figures.

Before training, `run_paper.sh` clears all NVIDIA compute processes via
`tools/clear_gpus.sh` (the same behaviour as the fencing launcher).
Pass `--no-kill` to skip cleanup; `--dry-run` and `--tables-only` never
kill anything.

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
- `clear_gpus.sh` stops all NVIDIA compute processes; `run_paper.sh`
  calls it before training and accepts `--no-kill` to opt out.
- `setup_baselines.sh` installs every baseline extra;
  `verify_against_official.py` is the `make verify` cross-check;
  `clean_pycache.sh` is `make clean`.
