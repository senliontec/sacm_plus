# Reproducibility

SACM-X now has an explicit deterministic mode. It is enabled by default for
training and evaluation.

## What is fixed

- Python `random`, NumPy, PyTorch CPU RNG, and CUDA RNG are seeded with
  `--seed`.
- `torch.backends.cudnn.deterministic=True`,
  `torch.backends.cudnn.benchmark=False`.
- TF32 is disabled for cuBLAS and cuDNN.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- `torch.use_deterministic_algorithms(True)` in strict mode.
- DataLoader workers are seeded with `worker_init_fn=seed_worker`, and the
  sampler/worker generator is seeded explicitly.
- Image/mask file order is sorted (`_find_pairs`), so the index-to-image
  mapping does not depend on `os.listdir`.
- Launchers export `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`.
- Checkpoints, W&B configs, and `metrics_history.json` record the seed,
  deterministic flags, worker count, PyTorch/CUDA/cuDNN versions, and device
  name.

## Flags

```bash
--deterministic true              # default
--deterministic_strict true       # fail on unsupported deterministic ops
--deterministic_threads 1         # torch CPU threads
--num_workers 4                   # keep fixed across compared runs
--seed 0
```

`--deterministic_strict false` is a best-effort mode: unsupported CUDA
operators may run non-deterministically. Use it only when a known operator
blocks the reproducible run, and report that choice.

## Guarantee scope

Bitwise-identical results are only expected when all of the following match:

- same model/code revision and data manifest;
- same GPU model and count;
- same PyTorch, CUDA, cuDNN, and driver versions;
- same `--seed`, `--num_workers`, and deterministic flags.

Different GPUs, drivers, CUDA/PyTorch versions, or single-GPU vs DDP runs are
not expected to be bitwise identical. For papers, report multiple seeds and
mean ± std; do not treat a single-seed sub-point difference as evidence.

## Reproducible comparison protocol

```bash
SEED=0
RUN_ROOT="results/runs/sacm_x/full_typed/seed_${SEED}"
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
GPU=7 PRESET=full_typed EPOCHS=800 VAL_INTERVAL=1 SKIP_TEST=true \
OUTPUT_ROOT="$RUN_ROOT" \
TRAIN_ROOT="data/sacm_3shot_seed${SEED}" \
bash scripts/sacm_x/run_sacm_x_quick.sh
```

Keep the same `SEED`, `PRESET`, `TRAIN_ROOT`, `EVAL_INPUT_MODE`,
`NUM_WORKERS`, and checkpoint-selection settings for the control run. The
one-click launchers already export the required environment variables.

## Checking a rerun

A rerun can be compared through:

- `results_summary.txt` (now includes Seed/Deterministic/Num workers);
- `metrics_history.json` (`reproducibility_info`);
- checkpoint `config.reproducibility_info`;
- W&B run config.

If strict mode aborts on an operator, report the exact op and use
`--deterministic_strict false` only as a diagnostic; the run is then not
bitwise guaranteed.

## Empty-mask and geometry conventions

- The baseline runtime and canonical paper collector default to
  `--empty-metric-policy penalty`: an empty prediction gets Dice/IoU/clDice=0
  and HD95=image diagonal, so it stays in every denominator.  Use
  `--empty-metric-policy nan` only when reproducing the raw official metric
  semantics; a NaN-based table silently drops empty predictions.
- The report skips runs whose `run_manifest.json` status is not `done`.
  Use `--allow-partial` only for diagnostics, never for a paper table.
- SACM-X defaults to aspect-preserving `aspect_pad` plus an empty guard.
  Baseline rows use their official geometry/guards.  For a matched
  comparison set `MATCH_BASELINE_GEOMETRY=1` when invoking
  `scripts/sacm_x/eval_sacm_x_20.sh` and report that ablation separately.
