# DCD

DCD is a prompt-free curvilinear-structure segmentation method built on
the SACM baseline and Segment Anything Model (SAM). The repository also
contains a self-contained zoo of official/reference baselines, a shared
3-shot / 20-dataset evaluation protocol, and the scripts used for paper
tables.

- Proposed method: `src/baseline_zoo/models/dcd/` (registry id
  `dcd`).
- Official comparison row: `src/baseline_zoo/models/sacm/`
  (registry id `sacm`). The official implementation is kept
  independent and is not replaced by the proposed method.
- Baseline zoo: `configs/models/*.yaml` + `src/baseline_zoo/`.

## Directory structure

```text
configs/models/              # model metadata + train/eval entrypoints
src/baseline_zoo/            # adapters, self-contained baseline models, runtime
src/models/sam_network/      # shared SAM network used by some baselines
src/core/                    # metrics, data utils, run layout, W&B helpers
scripts/run_train.sh         # unified train wrapper
scripts/run_eval.sh          # unified eval wrapper
scripts/run_report.sh        # aggregate existing metrics
scripts/pipeline/            # prediction, metric collection, report helpers
scripts/data/                # one-time dataset/protocol construction
data/sacm_3shot_seed*/       # seeded 3-shot train/val + test manifests
datasets/                    # evaluation image/mask roots
checkpoints/                 # local SAM/base weights
results/runs/                # canonical run outputs
```

## Datasets

The frozen protocol has 18 training images (6 seen domains × 3 shots), 6
validation images, and 20 test datasets / 3,580 test images:

- Seen: CHASEDB1, CORN, CREMI, CrackTree200, DCA1, DRIVE (667 images)
- Unseen: CHCrack5k, DeepCrack, FIVES, HRF, LES-AV, OCTA-500, STARE, XCAD
  (1,916 images)
- Novel: DeepGlobe, LEAF, ROAD, TYRE, TopoMortar, WIRE (997 images)

The seeded splits live under `data/sacm_3shot_seed0..4/`; evaluation roots
are under `datasets/`. Dataset-specific converters live in
`scripts/data/datasets/`. See `docs/DATASET_PROTOCOL_REPORT.html` for the
protocol details and caveats (CORN official split, LEAF annotation
fragmentation, TopoMortar GT caveat, DeepGlobe seed-42 subset).

## Quick start

```bash
make install          # pip install -e ".[dev]"
make smoke            # build a model and run one forward
make test             # unit tests
```

W&B is self-hosted and online by default. Credentials are loaded from
`scripts/wandb_env.sh`; use `WANDB_MODE=offline` or `WANDB_MODE=disabled`
only when intentionally logging offline/not logging.

## DCD: frozen canonical fixed3 800

The canonical DCD main is the fixed3 800 recipe:

```bash
# canonical fixed3 800 main + 4-arm component ablation, seeds 0-3, GPUs 0-7
bash scripts/run_dcd_full_800.sh
```

The `ssh` arm is exactly the previous fixed3 800 recipe: train with gating
and auxiliary fusion, then derive `head_select=fixed`,
`deployment_head=3`, `head_fusion=0` for inference.  The 4-seed 20-dataset
result is **56.66 / 43.05 / 63.42 / 64.28** (Dice / IoU / clDice / HD95).
The 200-epoch script (`scripts/run_dcd_expert_200.sh`) is only a fast
budget-reproduction check.  The model registry default in
`configs/models/dcd.yaml` is also 800 epochs.

The model keeps the official four-candidate Stage-2 decoder, dense deep
supervision, Skeleton Recall / soft-clDice consistency on all four dense
candidates, and soft bilinear targets plus foreground-weighted BCE.  It is
frozen: new research branches must use a separate adapter/model.

The official baseline is run through the same registry:

```bash
bash scripts/run_sacm_seeds03_gpu4567.sh
```

## Unified baseline runs

```bash
# One model / one seed
bash scripts/run_train.sh --model unet --seeds 0 --gpus 1
bash scripts/run_eval.sh --model unet --seeds 0 --gpus 1

# All available models, then tables and figures
bash scripts/run_paper.sh --model all --seeds all --gpus 0,1,2,3 --resume
bash scripts/run_paper.sh --model all --seeds all --tables-only
```

Every run writes one canonical directory:

```text
results/runs/<method>/<variant>/seed_<seed>/
    best_model.pth
    metrics_history.json
    val_metrics_by_dataset.csv
    predictions/<method>/<dataset>/*.png
    eval/tables/all_metrics.csv
    eval/tables/macro_metrics.csv
    eval/tables/per_dataset_4metrics.csv
```

Use `scripts/build_paper_table.sh` after the runs to build the SACM-style
per-dataset table (`MODELS` defaults to
`dcd:all_dense sacm:default`).

## Evaluation details

The shared evaluator reports Dice, IoU, clDice (official `core.metrics`
implementation), and HD95 in pixels at GT resolution. Empty predictions use
the explicit `penalty` convention. Do not compare clDice values directly
with the SACM paper: the paper's "clDice" is closer to skeleton Dice.

Prediction and metric collection are resumable:

```bash
python scripts/pipeline/run_baseline_test.py --method unet --skip-predict
python scripts/pipeline/run_baseline_test.py --method unet --skip-predict --workers 8
```

Per-dataset metric caches are stored under
`results/runs/<method>/<variant>/seed_<seed>/eval/<dataset>/` and are reused
unless `--no-resume` is passed. `scripts/pipeline/audit_predictions.py`
provides a read-only prediction-vs-GT diagnostic (pixel errors, components,
holes/Betti, skeleton errors).

## Reproducibility and tests

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src pytest tests -q
bash scripts/run_train.sh --model dcd --seeds 0 --gpus 0 --dry-run
```

The training/evaluation entrypoints seed Python/NumPy/PyTorch, use
deterministic kernels when requested, and store provenance (git state, data
manifest, config) next to every run.

## Documentation

- `docs/BASELINE_PORTING.md` — official baseline ports and deviations
- `docs/BASELINE_FIXES_2026-11.md` — task-adaptation fixes
- `docs/CVPR_STORY.md` — experiment narrative
- `docs/ENGINEERING.md`, `docs/REPRODUCIBILITY.md` — engineering details
- `scripts/README.md` — scripts and result layout reference

## Reference

If you use the official SACM implementation, cite its paper and repository.
The official baseline code is preserved under
`src/baseline_zoo/models/sacm/`; the proposed method lives under
`src/baseline_zoo/models/dcd/`.
