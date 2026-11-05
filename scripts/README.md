# scripts/

Top level contains only the one-click entry points.  Everything else lives
in three subdirectories.

## 1. All experiments (train + evaluate + tables + figures)

```bash
bash scripts/run_paper.sh --model all --seeds all --gpus 0,1,2,3 --resume
```

`--model all` resolves every `status: available` model in
`configs/models/` (Skein plus the 41 official/reference baselines and two
project extensions/variants: the CurvField extension and the corrected
`sacm_plus` variant, including the
appendix `sam2_unet`, the prompt-free `sam_adapter`/`samed`/`topolora`/
`convlora`/`c2p`/`a2sam`/`kprism`, the non-SAM `jtfn`/`dscnet`/`rsrm`/`dka`/
`scrwkv`/`flowsdf`/`tpsegformer`/`s2mnet`/`csunet`/`liot`/`dconnnet`,
the support-based
`persam`/`matcher`/`fssam`/`sansa`, and the oracle-prompted `ucs`).  For
each (model, seed)
job it runs the config's
`train_entrypoint`, then `eval_entrypoint`, and finally writes the
cross-model tables and figures.

Subsets and short runs:

```bash
bash scripts/run_paper.sh --model unet --seeds 0 --gpus 2
bash scripts/run_paper.sh --model unet,skein --seeds 0 1 --gpus 2,3
bash scripts/run_paper.sh --model all --seeds all --dry-run
```

### Prediction-vs-GT audit

`pipeline/audit_predictions.py` is a read-only diagnostic after evaluation.
It compares saved `*_result.png` predictions with GT masks at original
resolution and reports pixel errors, precision/recall, component
fragmentation, holes/Betti errors, skeleton errors, and failure flags:

```bash
python scripts/pipeline/audit_predictions.py \
  --pred-root results/runs/skein/full_typed/seed_0/eval \
  --protocol data/sacm_3shot_seed0/protocol_manifest.csv \
  --out results/audit/skein_full_typed_seed0
```

It writes `per_image.csv`, `per_dataset.csv`, `macro_summary.csv`,
`unmatched.csv`, `summary.json`, `report.md`, and (by default) comparison PNGs
for the worst five images per dataset.  Set `--save-error-maps` to save all
images.  Comparison PNGs show GT / prediction / pixel error (TP green, FP red,
FN blue) / skeleton error (GT yellow, prediction cyan, overlap white).  This
does not replace `evaluate.py` or change the paper table.

When `evaluate.py` or the baseline runtime has written
`prediction_index.csv`, the audit uses that exact prediction-to-GT mapping
instead of guessing from filenames.  A training run's
`provenance/data_manifest.json` can also supply test roots via
`--data-manifest`.

## W&B

Baseline training and evaluation default to W&B. `run_train.sh` and
`run_eval.sh` source `scripts/skein/wandb_env.sh` and require credentials
when `WANDB_MODE` is online; missing credentials fail fast with a clear
message. Use `WANDB_MODE=offline` for local W&B files or
`WANDB_MODE=disabled` to turn it off. Direct `run_one` also defaults to
W&B; pass `--use_wandb false` to disable it for a single run.

## Removed helpers

GPU cleanup, cue-ml queueing and Optuna HPO helpers were removed together
with `scripts/tools/`.  Free the selected GPUs manually before a run.

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
│   ├── finalize_run.py
│   ├── index_runs.py
│   ├── metrics.py
│   └── protocol_specs.py
├── skein/                  # Skein training/eval engine
│   ├── run_skein_5seeds.sh
│   ├── run_skein_quick.sh
│   ├── eval_skein_20.sh
│   ├── run_variant.sh
│   ├── run_experiments.py / aggregate_results.py / make_final_report.py
│   └── wandb_env.sh
└── data/                   # one-time dataset/protocol construction
    ├── prepare_data.py
    ├── organize_datasets.py
    ├── group_datasets.py
    ├── build_protocol_manifest.py
    ├── make_shot3_split.sh
    └── datasets/           # per-dataset official converters
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

## Skein specific

Skein is trained and evaluated through `run_skein_quick.sh` and
`eval_skein_20.sh`, the entry points embedded in `configs/models/skein.yaml`.
`run_skein_5seeds.sh` launches five seeds through `run_variant.sh`.
`run_experiments.py`, `aggregate_results.py` and `make_final_report.py` are
the low-level train/eval/report helpers used by those entry points.  The
export, matched-comparison and gate/repair diagnostic scripts were removed
together with `scripts/tools/`.

## Notes

- SegDINO needs DINOv3 weights only for training.  The adapter auto-detects
  a size-matched `.bin`/`.pth`/`.pt`/`.safetensors` file under
  `checkpoints/` (e.g. `checkpoints/dinov3/pytorch_model.bin`), so no
  environment variable is required.
- `make clean` removes `__pycache__` and stray `.pyc`/`.pyo` files.
