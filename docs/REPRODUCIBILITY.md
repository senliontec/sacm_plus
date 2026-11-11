# Reproducibility

SACM-STC now has an explicit deterministic mode. It is enabled by default for
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
RUN_ROOT="results/runs/sacm_stc/all_dense/seed_${SEED}"
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
GPU=7 PRESET=full_typed EPOCHS=300 VAL_INTERVAL=1 SKIP_TEST=true \
OUTPUT_ROOT="$RUN_ROOT" \
TRAIN_ROOT="data/sacm_3shot_seed${SEED}" \
bash scripts/run_sacm_stc_seeds03_gpu0123.sh
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
- SACM-STC defaults to aspect-preserving `aspect_pad` plus an empty guard.
  Baseline rows use their official geometry/guards.  For a matched
  comparison set `MATCH_BASELINE_GEOMETRY=1` when invoking
  `scripts/run_eval.sh --model sacm_stc` and report that ablation separately.
- `REUSE_EVAL_MODEL=1` only changes process/model-loading lifetime; each
  dataset still uses the same seeded DataLoader generator and the same
  deterministic settings.  `PAPER_METRICS_ONLY=1` computes the same Dice,
  IoU, clDice and HD95 definitions and only omits unrequested metrics.
  Both default to off; neither changes the seed or the canonical layout.

## Training artifacts

Every successful training run retains exactly one model artifact:
`best_model.pth`.  While training is in progress, a temporary
`last_model.pth` (crash-resume state) and optional
`checkpoints/epoch_XXXX.pth` snapshots may exist; both are deleted when
training reaches `status=done`.  A failed/interrupted run keeps the temporary
resume state so it can be resumed, but it is never an evaluation candidate.

- `last_model.pth` is temporary only and is rejected by evaluation;
- `checkpoints/` periodic snapshots are temporary only;
- `train_config.json` with the full resolved args, git commit/dirty state,
  environment versions, base-checkpoint SHA256 and data-manifest hash;
- checkpoint `config` records `val_selection`, `val_fixed_head`,
  `val_empty_guard`, the quality-selector weights and topology-loss settings;
  `evaluate.py` reuses fixed/quality selection automatically when the
  checkpoint was trained with it;
- `provenance/data_manifest.json` with train/val/test paths, sizes, mtimes
  and SHA256s (disable hashing with `--hash_data_files false`);
- copies of `protocol_manifest.csv`, `split_manifest.csv` and
  `protocol_summary.json` under `provenance/`;
- `wandb_run.json` with the run id/name/project/entity/url;
- incremental `metrics_history.json` and `run_info.json` after every epoch.

`--resume last` (or `--resume best`) resumes at the next epoch boundary after
restoring model, optimizer, scheduler, RNG, generator and history.  A resume
refuses a different architecture configuration, seed, split hash or data
manifest fingerprint.  Resume granularity is one epoch; mid-epoch DataLoader
worker state is not persisted.

The shared `baseline_zoo.runtime` path enforces best-only final artifacts for
every baseline.  Adapters with their own official training loop (nnU-Net,
nnWNet, KnowSAM, MaskVSC) keep that framework's checkpointing; their
canonical runners must expose only `best_model.pth` to evaluation.

`evaluate.py` writes `prediction_index.csv` and `eval_manifest.json` in every
dataset output directory.  The index stores each prediction relative to its
dataset directory and the absolute GT mask path, so it survives
`finalize_run` moving the directory.  `audit_predictions.py` uses this index
when present.

These artifacts apply to runs started with this code.  Historical runs may
still contain `last_model.pth` or periodic snapshots; those files are audit
artifacts only and must never be used for evaluation or paper tables.
