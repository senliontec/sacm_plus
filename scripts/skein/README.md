# Skein: train and evaluate

> **Canonical preset: `skein`.**  `full_fixed2_aux` is a compatibility alias.
> The unified paper pipeline uses the `train_entrypoint` /
> `eval_entrypoint` in `configs/models/skein.yaml`, which call the two
> wrappers below.

## Files

```text
scripts/skein/
├── run_skein_5seeds.sh      # five seeds via run_variant.sh
├── run_skein_quick.sh       # one seed, canonical run layout
├── eval_skein_20.sh         # 20-dataset evaluation
├── run_variant.sh           # generic preset launcher
├── run_experiments.py       # low-level multi-GPU train/eval engine
├── aggregate_results.py     # per-job summary aggregation
├── make_final_report.py     # four-metric report
└── wandb_env.sh             # self-hosted W&B environment
```

## Commands

```bash
# five seeds (server, 8x3090)
bash scripts/skein/run_skein_5seeds.sh

# one seed, train + validation only
GPU=1 PRESET=skein EPOCHS=800 SEED=0 SKIP_TEST=true \
  bash scripts/skein/run_skein_quick.sh

# one seed, train then 20-dataset evaluation
GPU=1 PRESET=skein EPOCHS=800 SEED=0 SKIP_TEST=false \
  bash scripts/skein/run_skein_quick.sh

# evaluate an existing checkpoint
GPU=1 PRESET=skein SEED=0 \
  bash scripts/skein/eval_skein_20.sh

# any preset through the generic launcher
GPU=1 PRESET=skein bash scripts/skein/run_variant.sh skein full
```

`PRESET` defaults to `full`; pass `PRESET=skein` for the deployed model.
`TRAIN_ROOT` defaults to `data/sacm_3shot_seed${SEED}`; `CANONICAL_ROOT`
defaults to `results/runs`.

## Layout

```text
results/runs/skein/<preset>/seed_<seed>/
  best_model.pth
  last_model.pth
  checkpoints/
  train_config.json
  provenance/
  run_info.json
  metrics_history.json
  train.log
  eval/
    <DATASET>/metrics.csv
    <DATASET>/summary.csv
    tables/all_metrics.csv
    tables/macro_metrics.csv
```

Every run writes `provenance/` (code, environment, split, checkpoint hash)
and `metrics_history.json`.  `RESUME=last` / `RESUME=best` resumes from the
canonical run root.

## Notes

- The wrapper fixes `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG` and
  OMP/MKL before Python starts.
- W&B is sourced from `wandb_env.sh`; set `WANDB_MODE=offline` or
  `WANDB_MODE=disabled` when needed.
- New runs use `results/runs/skein/...`.  Historical runs under legacy
  result paths are not consumed by the current pipeline.
- GPU cleanup, export, matched-comparison and gate/repair diagnostic helpers
  were removed with `scripts/tools/`.
