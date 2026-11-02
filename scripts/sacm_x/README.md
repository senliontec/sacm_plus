# Unified runs: baselines and SACM-X

Every run, baseline or SACM-X, lives in exactly one directory:

```text
results/runs/<method>/<variant>/seed_<seed>/
  best_model.pth            # file checkpoints
  best_model                # directory checkpoints (symlink)
  checkpoint.json           # kind + native checkpoint path
  run_info.json
  metrics_history.json
  train.log
  predictions/              # baselines, when present
  eval/
    <DATASET>/
      metrics.csv           # per-image metrics
      summary.csv           # one canonical dataset row
      results_summary.txt ...
    tables/
      all_metrics.csv       # unified schema
      macro_metrics.csv     # unified long schema
      final_report.md
      final_report.csv
    run_manifest.json
    eval.log
    jobs.csv
```

- `method` = baseline id or `sacm_x`
- `variant` = `default` for baselines, preset for SACM-X
- `seed` is always part of the path

There is no second user-facing layout.  ``results/paper`` and
``results/sacm_x`` are not written by the current pipeline.

## Baselines: train and evaluate 20 test sets

```bash
# Train one or more seeds through the unified registry.
bash scripts/run_train.sh --model unet --seeds 0 1 --gpus 0,1

# Evaluate the saved best checkpoint on the 20 test sets.
bash scripts/run_eval.sh --model unet --seeds 0 1 --gpus 0,1
```

Each job writes (for example, seed 1):

```text
results/runs/unet/default/seed_1/
```

The eval stage calls `collect_paper_metrics.py --layout canonical` and then
`finalize_run.py`, so the checkpoint and the 20-dataset tables share one run
root.

## SACM-X: train, then evaluate 20 test sets

```bash
SEED=0
PRESET=full_typed
RUN_ROOT="results/runs/sacm_x/${PRESET}/seed${SEED}"

GPU=7 SEED="$SEED" PRESET="$PRESET" \
TRAIN_ROOT="data/sacm_3shot_seed${SEED}" \
OUTPUT_ROOT="$RUN_ROOT" EPOCHS=800 VAL_INTERVAL=1 SKIP_TEST=true \
bash scripts/sacm_x/run_sacm_x_quick.sh
```

Training promotes the best checkpoint to:

```text
"$RUN_ROOT/best_model.pth"
```

`run_info.json` records `best_epoch`, `last_epoch`, `selection`, `seed`, and
the checkpoint kind/path.  Only the validation-selected best checkpoint is
saved.

Evaluate the saved checkpoint:

```bash
GPU=7 SEED="$SEED" PRESET="$PRESET" \
TRAIN_ROOT="data/sacm_3shot_seed${SEED}" \
TRAIN_OUT="$RUN_ROOT" OUTPUT_ROOT="$RUN_ROOT/eval" \
EVAL_INPUT_MODE=aspect_pad USE_WANDB=true EMPTY_GUARD=true \
COMPONENT_METRICS=true AUDIT_HEADS_FULL=true \
bash scripts/sacm_x/eval_sacm_x_20.sh
```

Or run train + eval in one command by setting `SKIP_TEST=false`; the quick
script trains, finalizes the checkpoint, then invokes the 20-dataset eval.

Do not pass a hard-coded seed path.  Derive it from `SEED`, as above, so each
seed writes to a different directory.

## Latest run index

```bash
python scripts/pipeline/index_runs.py
# results/runs_index.csv
```

Columns: `family,method,variant,seed,status,checkpoint,best_epoch,eval_root,
metrics_csv,macro_csv,manifest`.

## Reproducibility

Use the shell wrappers (`run_train.sh`, `run_eval.sh`,
`run_sacm_x_quick.sh`, `eval_sacm_x_20.sh`) so `PYTHONHASHSEED`,
`CUBLAS_WORKSPACE_CONFIG`, OMP/MKL are fixed before Python starts.  Both
pipelines configure deterministic kernels; SACM-X uses strict mode by
default, baselines use best-effort strict mode (`--deterministic-strict true`
enables strict).

## Legacy results

Old runs under `results/paper` or `results/sacm_x` are not consumed by the
current scripts.  To migrate one manually:

```bash
python scripts/pipeline/finalize_run.py \
  --family sacm_x --method sacm_x --variant "$PRESET" --seed "$SEED" \
  --source-train <legacy-train-root> \
  --source-eval  <legacy-eval-root> \
  --out-root results/runs
```

For baselines use `--family baseline --method <id> --variant default` and the
legacy seed directory as both source roots.
