# scripts/

Top level contains only the one-click entry points.  Everything else lives
in three subdirectories.

## 1. All experiments (train + evaluate + tables + figures)

```bash
bash scripts/run_paper.sh --model all --seeds all --gpus 0,1,2,3 --resume
```

`--model all` resolves every `status: available` model in
`configs/models/` (the 41 official/reference baselines plus the corrected
`sacm_stc` variant, including the
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
bash scripts/run_paper.sh --model unet,sacm_stc --seeds 0 1 --gpus 2,3
bash scripts/run_paper.sh --model all --seeds all --dry-run
```

### Prediction-vs-GT audit

`pipeline/audit_predictions.py` is a read-only diagnostic after evaluation.
It compares saved `*_result.png` predictions with GT masks at original
resolution and reports pixel errors, precision/recall, component
fragmentation, holes/Betti errors, skeleton errors, and failure flags:

```bash
python scripts/pipeline/audit_predictions.py \
  --pred-root results/runs/sacm_stc/all_dense/seed_0/eval \
  --protocol data/sacm_3shot_seed0/protocol_manifest.csv \
  --out results/audit/sacm_stc_all_dense_seed0
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
`run_eval.sh` source `scripts/wandb_env.sh` and require credentials
when `WANDB_MODE` is online; missing credentials fail fast with a clear
message. Use `WANDB_MODE=offline` for local W&B files or
`WANDB_MODE=disabled` to turn it off. Direct `run_one` also defaults to
W&B; pass `--use_wandb false` to disable it for a single run.

## SACM-STC canonical result layout

SACM-STC has one canonical main variant: `all_dense`.  The historical
`sacm_stc/default/` tree was only an evaluation alias of the `all_dense`
checkpoint and has been archived.  Do not pull or build a separate
`sacm_stc:default` row.  A clean main rerun writes both training and eval
tables under:

```text
results/runs/sacm_stc/all_dense/seed_<s>/
```

Ablation runs use their own `<arm>` variants and diagnostics/analysis live
under `results/ablations/sacm_stc/`.  Use
`bash scripts/archive_sacm_stc_main.sh` to preserve and move an old
`all_dense` + `default` pair before rerunning.

## Reproduction runs

Train and evaluate the proposed method and the official baseline separately,
then build Table 2:

```bash
# SACM-STC, seeds 0-3, GPUs 0-3 (one seed per GPU)
bash scripts/run_sacm_stc_seeds03_gpu0123.sh

# SACM-STC, seeds 0-3, GPUs 4-7
bash scripts/run_sacm_stc_seeds03_gpu4567.sh

# official SACM, seeds 0-3, GPUs 4-7
bash scripts/run_sacm_official_seeds03_gpu4567.sh
```

Each script trains every seed first, then evaluates all seeds on the 20
protocol test datasets with Dice/IoU/clDice/HD95.  The recorder writes
`eval/tables/per_dataset_4metrics.csv` per seed, plus combined
`results/reports/<model>_<variant>_seeds*_per_dataset.csv`,
`..._mean_std.csv` and `..._mean_std.md`.  Useful overrides: `GPU_LIST=...`,
`SEEDS="..."`, `RESUME=1` (skip completed jobs), `SKIP_TRAIN=1`,
`SKIP_EVAL=1`, `EXPECTED_DATASETS=20`.

## SACM-STC fixed3 vs sacm_official (800 epochs, seeds 0-3)

fixed3 is an **inference-only deployment** of the SSH checkpoint:
`head_select=fixed`, `deployment_head=3` (the recall/Tversky expert), and
`head_fusion=0`.  The network weights are unchanged; the derived checkpoint
removes the unused `head_fusion.*` tensors so the strict load succeeds.

One command trains the SSH base for seeds 0-3 at 800 epochs, derives the
fixed3 checkpoint for every seed, evaluates fixed3 on all 20 protocol
datasets, and writes a four-seed comparison against the already-completed
`sacm_official` eval tables.  It does **not** train/evaluate `sacm_official`
and does **not** train/evaluate `sacm_stc full`:

```bash
bash scripts/run_sacm_stc_fixed3_seeds03_800.sh
DRY_RUN=1 bash scripts/run_sacm_stc_fixed3_seeds03_800.sh   # print commands
EVAL_GPUS=4,5,6,7 bash scripts/run_sacm_stc_fixed3_seeds03_800.sh
```

Default GPU placement: train on GPUs 0-3, then evaluate fixed3 on the same
GPUs 0-3 after training.  GPUs 4-7 stay free; set `EVAL_GPUS=4,5,6,7` if you
prefer to run the evaluation there.

The official comparison requires the existing tables:

```text
results/runs/sacm_official/default/seed_0..3/eval/tables/all_metrics.csv
results/runs/sacm_official/default/seed_0..3/eval/tables/macro_metrics.csv
```

Key outputs:

```text
results/runs/sacm_stc_ssh_full_ab/ssh/seed_<s>/best_model.pth
results/runs/sacm_stc_ssh_full_ab/ssh/seed_<s>/best_model_fixed3_nofusion.pth
results/runs/sacm_stc_fixed3_800/eval/seed_<s>/tables/all_metrics.csv
results/ablations/sacm_stc_fixed3_800/summary.md
results/ablations/sacm_stc_fixed3_800/summary.json
```

The helpers are `pipeline/make_fixed3_checkpoint.py` (derive the no-fusion
checkpoint), `pipeline/check_ssh_recipe.py` (before reusing a complete
800-epoch SSH run, compare its adapter config with the YAML arm), and
`pipeline/summarize_fixed3_seeds.py` (fixed3 vs official, four metrics,
per-role/per-seed tables, per-dataset wins, paired deltas).  The run uses
local data and the local SAM checkpoint; it never downloads weights.

## SACM-STC scope/structure ablation

The dense-scope × structure-scope factorial is configured in
`configs/ablations/sacm_stc_scope_structure.yaml`; the runner contains no
arm names or model hyperparameters.  Audit the existing checkpoints first,
then run the factorial:

```bash
bash scripts/run_sacm_stc_ablation.sh --stage audit --gpus 4,5,6,7
bash scripts/run_sacm_stc_ablation.sh --stage all --gpus 4,5,6,7
```

Ablation runs use distinct `<arm>` variants under
`results/runs/sacm_stc/`; diagnostics/analysis live under
`results/ablations/sacm_stc/`.  See
[`../configs/ablations/README.md`](../configs/ablations/README.md) for the
arm definitions, decision rules, outputs, and diagnostics.

## Paper Table 2 (all 20 datasets)

After both runs have finished (the combined script above already does this),
build the SACM-style per-dataset table with all four metrics:

```bash
bash scripts/build_paper_table.sh
```

Default methods are `sacm_stc:all_dense sacm_official:default`, seeds `0 1 2 3`.
It reads each run's canonical `eval/tables/all_metrics.csv` and falls back to
`per_dataset_4metrics.csv` only when that is absent.  It aggregates mean/std
across seeds per dataset and per Seen/Unseen/Novel/All role (macro computed
per seed first), and writes to `results/reports/paper_table2/`:

| File | Content |
|---|---|
| `table2.tex` | paper-ready LaTeX, means only, percent, best bold / second underlined |
| `table2_with_std.tex` | LaTeX, mean ± std |
| `table2.md` | Markdown, mean ± std |
| `table2_means.md` | Markdown, means only, percent, best bold / second underlined |
| `table2.csv` | long table: dataset, method, metric, mean, std, unit, n_seeds |
| `table2_macro.csv` | Seen/Unseen/Novel/All macro rows |

The display follows the SACM paper's Table 2: Dice/IoU/clDice are shown in
percent (2 decimals) and HD95 in pixels; best is bold and second best is
underlined.  `table2.csv` / `table2_macro.csv` keep the raw overlap
fractions and mark the unit.  Pass `--no-percent` for [0,1] fractions,
`--no-bold` to disable ranking marks, and `--decimals N` to change precision.

The builder fails loudly if a seed is missing, a seed has the wrong number of
datasets, dataset sets or roles differ across seeds/methods, or the dataset
names/roles disagree with `data/sacm_3shot_seed<s>/protocol_manifest.csv`
(when that manifest is present).  This prevents a silently wrong 20-dataset
table.

Overrides: `MODELS="..."`, `SEEDS="..."`, `OUT_DIR=...`,
`EXPECTED_DATASETS=20`.  The table fails loudly if any seed is missing or a
seed does not contain exactly the expected number of datasets.

Metric conventions: clDice is the official clDice from `core.metrics`, and
HD95 is medpy-style in pixels at the original evaluation resolution.  The
SACM paper's "clDice" is closer to skeleton Dice, so do not compare clDice
values numerically across the two papers.  The collector uses the explicit
`penalty` convention for empty masks; state this convention before making
fine-grained empty-mask claims.

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
├── run_sacm_stc_seeds03_gpu0123.sh    # reproduction: SACM-STC seeds 0-3 (GPUs 0-3)
├── run_sacm_stc_seeds03_gpu4567.sh    # reproduction: SACM-STC seeds 0-3 (GPUs 4-7)
├── run_sacm_official_seeds03_gpu4567.sh  # reproduction: SACM seeds 0-3
├── build_paper_table.sh    # Table 2 from recorded runs
├── pipeline/               # evaluation helpers used by the configs
│   ├── run_baseline_test.py
│   ├── collect_paper_metrics.py
│   ├── record_4metrics.py  # per-dataset Dice/IoU/clDice/HD95 recorder
│   ├── build_paper_table.py # SACM-style Table 2 builder
│   ├── finalize_run.py
│   ├── index_runs.py
│   ├── metrics.py
│   └── protocol_specs.py
├── wandb_env.sh            # shared self-hosted W&B environment
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

## Notes

- SegDINO needs DINOv3 weights only for training.  The adapter auto-detects
  a size-matched `.bin`/`.pth`/`.pt`/`.safetensors` file under
  `checkpoints/` (e.g. `checkpoints/dinov3/pytorch_model.bin`), so no
  environment variable is required.
- `make clean` removes `__pycache__` and stray `.pyc`/`.pyo` files.
