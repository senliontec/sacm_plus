# SACM-X run layout

Use one run directory per experiment.  Training and evaluation are separate
subdirectories so a 20-dataset evaluation never overwrites the training log
or the checkpoint.  The `seed_<seed>` level is mandatory: it keeps every
seed's checkpoint, logs, and 20-dataset tables separate.

```text
results/sacm_x/<run>/seed_<seed>/
  train/<preset>/
    best_model.pth          # val-selection best (not necessarily the last epoch)
    metrics_history.json    # per-epoch metrics + best_epoch
    run_info.json           # checkpoint, best_epoch, last_epoch, selection
    run.log
  eval/<tag>/
    <preset>/<DATASET>/
      metrics.csv
      results_summary.txt
      head_audit.csv ...    # when audit flags are enabled
    tables/
      all_metrics.csv
      macro_metrics.csv
      final_report.md
      final_report.csv
      table_*.tex
    run_eval.log
    jobs_eval.csv
    run_manifest.json
```

## Train

```bash
SEED=0
RUN_NAME=typed_800

OUTPUT_ROOT="results/sacm_x/${RUN_NAME}_seed${SEED}/train" \
GPU=7 SEED="$SEED" RUN_NAME="$RUN_NAME" PRESET=full_typed \
EPOCHS=400 VAL_INTERVAL=1 SKIP_TEST=true \
TRAIN_ROOT="data/sacm_3shot_seed${SEED}" \
bash scripts/sacm_x/run_sacm_x_quick.sh
```

The single best checkpoint is:

```text
$OUTPUT_ROOT/full_typed/best_model.pth
```

`run_info.json` records `best_epoch`, `last_epoch`, `selection_metric`, and the
absolute checkpoint path.  Only the val-selection best is saved; the last
epoch is not a separate file.

## Evaluate all 20 test sets

```bash
SEED=0
RUN_NAME=typed_800
RUN_ROOT="results/sacm_x/${RUN_NAME}_seed${SEED}"
TRAIN_OUT="$RUN_ROOT/train"
OUTPUT_ROOT="$RUN_ROOT/eval/main"

TRAIN_OUT="$TRAIN_OUT" \
GPU=7 SEED="$SEED" PRESET=full_typed \
TRAIN_ROOT="data/sacm_3shot_seed${SEED}" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
EVAL_INPUT_MODE=aspect_pad \
USE_WANDB=true EMPTY_GUARD=true \
COMPONENT_METRICS=true AUDIT_HEADS_FULL=true \
bash scripts/sacm_x/eval_sacm_x_20.sh
```

No checkpoint copy is needed: the eval script uses
`$TRAIN_OUT/$PRESET/best_model.pth` by default.  `OUTPUT_ROOT` is the eval
root; if omitted and `TRAIN_OUT` ends in `/train`, it defaults to the sibling
`eval/` directory.

Results:

```text
$OUTPUT_ROOT/$PRESET/<DATASET>/metrics.csv
$OUTPUT_ROOT/tables/all_metrics.csv
$OUTPUT_ROOT/tables/macro_metrics.csv
$OUTPUT_ROOT/tables/final_report.md
$OUTPUT_ROOT/jobs_eval.csv
$OUTPUT_ROOT/run_manifest.json
```

## Evaluate a subset

```bash
SEED=0
RUN_NAME=typed_800
RUN_ROOT="results/sacm_x/${RUN_NAME}_seed${SEED}"
TRAIN_OUT="$RUN_ROOT/train"
OUTPUT_ROOT="$RUN_ROOT/eval/audit6"

TRAIN_OUT="$TRAIN_OUT" \
GPU=7 SEED="$SEED" PRESET=full_typed \
TRAIN_ROOT="data/sacm_3shot_seed${SEED}" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
DATASETS="CREMI TYRE WIRE LEAF TopoMortar OCTA-500" \
AUDIT_HEADS=true COMPONENT_METRICS=true \
AUDIT_THRESHOLDS="-2 -1 0 1 2" \
bash scripts/sacm_x/eval_sacm_x_20.sh
```

The launcher passes `--datasets` and `--preset` to the aggregator, so stale
results from another dataset set cannot leak into the macro table.

## Explicit checkpoint

```bash
SEED=0
RUN_NAME=typed_800
RUN_ROOT="results/sacm_x/${RUN_NAME}_seed${SEED}"
TRAIN_OUT="$RUN_ROOT/train"
OUTPUT_ROOT="$RUN_ROOT/eval/other"
CKPT=/absolute/path/to/best_model.pth

CKPT="$CKPT" TRAIN_OUT="$TRAIN_OUT" OUTPUT_ROOT="$OUTPUT_ROOT" \
SEED="$SEED" PRESET=full_typed \
bash scripts/sacm_x/eval_sacm_x_20.sh
```

`CKPT` overrides the default checkpoint path.

## Selection variants

Use a separate eval root for every decision rule:

```bash
SEED=0
RUN_NAME=typed_800
RUN_ROOT="results/sacm_x/${RUN_NAME}_seed${SEED}"
TRAIN_OUT="$RUN_ROOT/train"

TRAIN_OUT="$TRAIN_OUT" SEED="$SEED" PRESET=full_typed \
OUTPUT_ROOT="$RUN_ROOT/eval/quality" SELECTION=quality \
bash scripts/sacm_x/eval_sacm_x_20.sh

TRAIN_OUT="$TRAIN_OUT" SEED="$SEED" PRESET=full_typed \
OUTPUT_ROOT="$RUN_ROOT/eval/head3" FIXED_HEAD=3 \
bash scripts/sacm_x/eval_sacm_x_20.sh
```

Do not reuse an eval root for a different selection rule: per-dataset files
would be overwritten.

## Auto-eval after training

`run_sacm_x_quick.sh` defaults to `SKIP_TEST=false`, so after training it
evaluates all datasets from the protocol and writes `tables/` under the
training root.  With `SKIP_TEST=true` (used for the current 800-epoch run),
run the evaluation command above separately.

## Baseline runs

Baselines keep their original layout:

```text
results/paper/<method>/seed_<seed>/
  best_model.pth
  run_info.json
  paper_metrics.csv
  paper_macro.csv
```

New baseline trainings also write `run_info.json` (method, seed, checkpoint,
best_epoch); new baseline evaluations write `run_manifest.json` next to
`paper_metrics.csv`.

## One seed-aware index for both families

```bash
python scripts/pipeline/index_runs.py \
  --roots results/paper results/sacm_x \
  --out results/runs_index.csv
```

The index contains `family, method, seed, checkpoint, best_epoch, eval_root,
metrics_csv, macro_csv, manifest`, so different seeds are separate rows.
