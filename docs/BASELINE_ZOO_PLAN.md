# Baseline Zoo Plan

Status: draft  
Scope: unify all baselines under `baselines/` into a callable model library  
Target: one shared PyTorch/CUDA/cuDNN environment; one command runs any/all baselines

## 1. Goal

Turn the current collection of vendored upstream repositories into a
**baseline zoo** with these properties:

- `list_baselines()` returns every available method;
- `create_baseline(name)` returns a model adapter that is independently callable;
- all methods use the same data protocol, prediction format and metrics;
- any subset of methods can be trained/evaluated with one command;
- all methods run in one pinned environment (ideally one Docker image);
- upstream implementations remain intact and licensed as upstream code.

Non-goals:

- rewriting every model from scratch;
- forcing every model to share one loss/optimizer/preprocessing;
- guaranteeing official paper numbers when retraining under the 3-shot protocol;
- running 8-GPU prediction by default (sequential is the default).

## 2. Current state

There are 12 model source directories plus one adapter-only method:

```text
unet, cs2net, bcunet, nnwnet, maskvsc,
sam-med2d, sam-octa, cwsam, knowsam, segdino,
sam2-unet, sacm
+ sam_vit_l_unet (adapter-only)
```

Only three methods are wired into the current unified test runner:

```text
unet, sam_vit_l_unet, sacm_official
```

The remaining methods have source code but no unified prediction adapter.

Known blockers found during audit:

| Method | Blocker |
|---|---|
| `bcunet` | `code/core/models.py` contains the invalid class name `BCU-Net`; `warmlearnrate.py` is missing while `main.py` imports it |
| `nnwnet` | only nnU-Net trainer plugins; requires external `nnunetv2`, no train/test entrypoint |
| `cwsam` | vendored mmseg-like stack, YAML configs, generic package names `models`/`datasets`/`utils` |
| `knowsam` | semi-supervised trainer with discriminator and two-stream sampler; not a standard 3-shot baseline |
| `cs2net` | no argparse, hardcoded arguments, requires `visdom`, DRIVE-specific loader |
| `sam-med2d` | requires `apex`; vendors `segment_anything` |
| `sam-octa` | vendors `segment_anything`; LoRA/config spread out |
| `sacm` | vendors `segment_anything`; no requirements file |
| `maskvsc` | additional `kornia`/`opencv`/`skimage` dependencies and multiple entrypoints |
| `segdino` | top-level modules (`blocks.py`, `dpt.py`, `dataset.py`) and dataset-specific defaults |
| `sam2-unet` | vendors the full `sam2` package; relatively self-contained |
| `unet` | hardcoded `./data/imgs` and `./data/masks`; otherwise self-contained |

The main problem is integration, not model quality: package-name collisions,
different config systems, missing dependencies, incomplete upstream code and
different output conventions.

## 3. Design principles

1. **Registry + factory**  
   A model is registered by name and created lazily.

2. **Namespace isolation**  
   Never expose upstream top-level packages such as `models`, `utils` or
   `segment_anything` to the rest of the project. Each baseline lives under a
   unique namespace or runs in a subprocess.

3. **Adapter, not rewrite**  
   Keep upstream code and licenses; write a thin adapter that converts between
   the zoo contract and the upstream model.

4. **Standard I/O boundary**  
   Inside the adapter, preprocessing and forward are model-specific. Outside,
   every model consumes the same image/mask protocol and writes the same
   binary PNG prediction.

5. **Subprocess-first for unsafe imports**  
   When two baselines cannot coexist in one Python interpreter, the adapter
   invokes a model-specific subprocess using the same interpreter/environment.

6. **Config is the source of truth**  
   CLI only overrides explicit values; model defaults live in YAML/dataclasses.

7. **Resumable and observable**  
   Every (method, dataset) job has a status, log and checkpoint path; the runner
   can resume after failure.

## 4. Target layout

```text
src/
└── baseline_zoo/
    ├── __init__.py          # list_baselines / create_baseline
    ├── registry.py          # name -> lazy adapter path/factory
    ├── base.py              # BaselineAdapter protocol
    ├── config.py            # dataclasses + YAML loading/validation
    ├── io.py                # image/mask/prediction contracts
    ├── runner.py            # queue, resume, logs, W&B
    ├── report.py            # final metrics/report
    └── adapters/
        ├── unet.py
        ├── sam_vit_l_unet.py
        ├── sacm.py
        ├── sam2_unet.py
        └── ...

baselines/                   # upstream code stays where it is
├── unet/
├── sacm/
├── sam2-unet/
└── ...

configs/baselines/
├── unet.yaml
├── sacm.yaml
└── ...

results/baselines/
├── <method>/
│   ├── <dataset>/
│   │   ├── predictions/*.png
│   │   └── metrics.csv
│   ├── best_model.pth
│   └── run.log
└── tables/
    ├── all_metrics.csv
    ├── macro_metrics.csv
    └── final_report.md
```

`baselines/` remains the vendor layer. `src/baseline_zoo/` is the library and
execution layer. Do not move upstream repositories until the adapters are
stable.

## 5. Adapter contract

```python
class BaselineAdapter(Protocol):
    name: str
    family: str
    input_size: int
    threshold: float

    def build(self, cfg: dict): ...
    def load_checkpoint(self, path: str): ...
    def preprocess(self, image): ...
    def forward(self, images): ...          # -> logits/probabilities
    def predict(self, image_path: str) -> np.ndarray:
        """Return a binary HxW mask (0/255) at original image size."""
    def train(self, cfg: dict) -> str:       # optional
        """Train and return checkpoint path."""
    def state_dict(self): ...
```

Requirements:

- `predict()` is the only method the evaluator needs;
- output is original-size, single-channel, binary 0/255;
- model-specific threshold defaults are configurable but the comparison default
  is sigmoid 0.5 (logit 0.0);
- `train()` may use a model-specific loss and optimizer;
- adapters must not import unrelated baselines.

## 6. Registry and factory

```python
# registry.py
BASELINE_REGISTRY = {
    "unet":       "baseline_zoo.adapters.unet:UNetAdapter",
    "sam2_unet":  "baseline_zoo.adapters.sam2_unet:SAM2UNetAdapter",
    ...
}
```

```python
from baseline_zoo import list_baselines, create_baseline

print(list_baselines())
model = create_baseline("sam2_unet", config="configs/baselines/sam2_unet.yaml")
mask = model.predict("image.png")
```

Imports are lazy: a missing optional dependency for one model must not prevent
`list_baselines()` or other models from working.

## 7. Config contract

Each model has a YAML file with explicit fields:

```yaml
name: sam2_unet
family: sam
vendor_dir: baselines/sam2-unet
entrypoint: baselines/sam2-unet/test.py   # if subprocess mode
input_size: 1024
normalization: torchvision_to_tensor
threshold: 0.0            # logit; sigmoid 0.5
selection: auto           # model-specific; documented
epochs: 50
batch_size: 1
lr: 0.0003
checkpoint: null
pretrained: checkpoints/sam2_hiera_large.pt
wandb: true
```

Precedence:

```text
CLI explicit value > run config > model YAML default
```

Unknown keys are errors. Every model config records its upstream commit and
license.

## 8. Prediction and metric contract

All methods write:

```text
results/baselines/<method>/<dataset>/predictions/<stem>.png
```

Rules:

- original GT size;
- single channel;
- 0 background / 255 foreground;
- no color maps, no probability PNGs;
- one file per test image;
- prediction metadata records checkpoint hash, threshold, selection and TTA.

Shared evaluator computes:

```text
Dice, IoU, clDice, HD95
```

with the same code for every method. Model-specific evaluators are allowed only
for debugging; the reported table must use the shared evaluator.

## 9. Environment strategy

Target a single environment, enforced by a container:

```text
Python 3.10/3.11
PyTorch 2.x
CUDA 12.x
cuDNN 9
```

Use one image, e.g. `pytorch/pytorch:<torch>-cuda<cuda>-cudnn<cudnn>`, and pin
all Python dependencies in a lock file.

Dependency policy:

- audit every model's requirements against the target stack;
- replace obsolete APIs where possible:
  - `apex` mixed precision -> `torch.amp`;
  - `visdom` -> disabled/logging;
  - old mmcv APIs -> port or vendor;
- install optional dependencies in the same image when compatible;
- if a dependency cannot coexist, the adapter uses a subprocess entrypoint,
  still using the same Python interpreter and environment;
- `python -m baseline_zoo doctor` reports missing packages and broken imports.

Module-name isolation:

- never add all vendor directories to `sys.path` at once;
- prefer package-relative imports inside each vendor directory;
- for model scripts that require a specific working directory, launch them with
  `cwd=baselines/<model>` and a controlled `PYTHONPATH`;
- `segment_anything` copies in SACM/SAM-Med2D/SAM-OCTA must never be imported
  into the same process.

## 10. One-click runner

Commands:

```bash
python -m baseline_zoo list
python -m baseline_zoo doctor

python -m baseline_zoo train \
  --methods unet,sam2_unet,sacm \
  --datasets all --gpu 3 --epochs 50 --resume

python -m baseline_zoo predict \
  --methods all --datasets all --gpu 3 --resume

python -m baseline_zoo evaluate \
  --methods all --datasets all

python -m baseline_zoo report
```

Or a single shell entrypoint:

```bash
bash scripts/run_paper.sh --model all --seeds all --gpus 3 --resume
```

Runner responsibilities:

- build the selected method list;
- create a job queue of `(method, dataset)` pairs;
- sequential by default; optional `--workers`/`--gpus`;
- launch each adapter (direct or subprocess);
- record `pending/running/done/failed` in `jobs.csv`;
- preserve logs per method/dataset;
- resume only unfinished jobs;
- optionally log to W&B per method;
- run shared aggregation and final report.

## 11. Fairness protocol

For the main comparison:

- 18 training images / 6 validation images (3-shot protocol);
- 20 fixed test datasets;
- no TTA;
- threshold = sigmoid 0.5 (logit 0.0);
- no FOV masking;
- original-size predictions;
- shared Dice/IoU/clDice/HD95;
- checkpoint selected by validation F1 (same rule), unless a model cannot train;
- model-specific preprocessing is allowed and documented.

Two run modes:

- `protocol=unified`: retrain every method with the same 3-shot budget;
- `protocol=official`: use the model's official checkpoint/recipe, then evaluate
  with the shared test protocol.

Both modes must be stored separately and clearly labeled in the report.

## 12. Migration plan

### Phase 0 — Contract and audit

- write this plan;
- complete the dependency/blocker matrix;
- choose target torch/CUDA/cuDNN versions;
- define the adapter protocol and config schema.

### Phase 1 — Skeleton and pilots

- implement `src/baseline_zoo/` registry, base, config, IO, runner, report;
- migrate `unet`, `sam_vit_l_unet`, `sacm`;
- add `list`, `doctor`, `predict`, `evaluate` commands;
- verify against existing prediction files.

### Phase 2 — SAM-family baselines

- `sam2-unet`;
- `sam-med2d`;
- `sam-octa`;
- resolve `segment_anything` collisions through subprocess entrypoints;
- add smoke tests for each.

### Phase 3 — remaining supervised baselines

- `maskvsc`;
- `segdino`;
- `cs2net`;
- repair hardcoded paths and remove `visdom` at runtime.

### Phase 4 — hard cases

- `bcunet`: fix syntax error, restore missing `warmlearnrate.py`;
- `nnwnet`: install/wrap nnU-Net or provide a standalone adapter;
- `cwsam`: resolve mmseg/config dependencies;
- `knowsam`: define whether to support semi-supervised training or use a
  fixed supervised variant.

### Phase 5 — full run and report

- 1 dataset × all methods smoke;
- full 20-dataset run;
- final report with mean ± std for all four metrics;
- document which methods used unified training vs official checkpoints.

## 13. Testing and acceptance

Unit tests:

- registry listing and lazy imports;
- config validation and precedence;
- prediction contract (size, dtype, binary values);
- metric consistency.

Per-model smoke test:

1. build adapter;
2. load checkpoint;
3. predict one fixed image;
4. verify output shape and 0/255;
5. compare against the model's direct script on a small sample.

Integration test:

- one method × one dataset;
- all methods × one dataset;
- then all methods × 20 datasets.

A model is not "integrated" until its adapter passes the smoke test and its
prediction can be evaluated by the shared metric code.

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| package-name collisions | subprocess entrypoints + namespace isolation |
| incompatible dependencies | dependency audit, pinning, Docker, local ports |
| broken upstream code | fix in adapter/vendor copy, record patch |
| missing official checkpoints | use unified retraining or mark unavailable |
| training protocol mismatch | separate unified/official modes and report both |
| long full runs | resumable job queue, per-model logs |
| silent numerical drift | fixed-sample smoke tests and prediction hashes |
| CUDA memory | one method at a time, cleanup between jobs |

## 15. Deliverables

1. `src/baseline_zoo/` package with registry, adapters, runner and report.
2. `configs/baselines/<method>.yaml` for every supported method.
3. A dependency/blocker audit table.
4. One adapter per model, with smoke tests.
5. One command for train/predict/evaluate/report.
6. `results/baselines/tables/final_report.md` with all 13 methods.
7. Documentation of protocols, environment and known model limitations.

## 16. Open decisions

- Target torch/CUDA/cuDNN versions.
- Whether the primary comparison uses unified 3-shot retraining or official
  checkpoints.
- Whether `knowsam`/`nnwnet` are in scope as supervised 3-shot baselines or
  reported as reference-only.
- GPU concurrency policy for full runs.
- Storage budget for 13 methods × 20 datasets predictions.

## 17. Immediate next step

Do not start the migration during an active training/evaluation run. After the
current evaluation is complete:

1. freeze the current results and checkpoint;
2. write the dependency matrix;
3. implement the skeleton and migrate the three existing adapters;
4. add `sam2-unet` as the first new method;
5. only then expand to the remaining models.
