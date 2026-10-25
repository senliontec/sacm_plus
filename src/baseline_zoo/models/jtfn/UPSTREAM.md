# JTFN upstream record

## Source

- Repository: `zkl20061823/JTFN`
  (https://github.com/zkl20061823/JTFN)
- HEAD: `5b23b242d37e58698258b05ca1ce75f6f9a03b8c` (2022-02-07).
- Local reference: `3rd/JTFN`.
- License: **the upstream tree has no LICENSE file**.  The code is vendored
  for research/reproduction only; do not redistribute it without checking
  with the authors.

## Copied files

- `models/jtfn.py` — `JTFN` model and `compute_objective` loss
- `models/modules.py` — `GAU`, `FIM` and supporting blocks
- `models/backbone/base.py` — plain 5-stage backbone (`base64`/`base32`)
- `models/backbone/resbase.py` — residual 5-stage backbone
  (`resbase32`/`resbase64`)

`models/__init__.py`, the dataset/split loaders, TensorBoard logger, training
script and boundary-generation CLI are not vendored; the shared baseline
adapter/runtime provides those responsibilities.

## Modifications

- Imports are package-relative (`from .backbone...`, `from .modules...`).
- `models/jtfn.py` is otherwise unchanged: the two-stream mask/boundary
  decoder, GAU gating, FIM interaction, three refinement steps and the
  multi-scale BCE objective are the upstream code.

## Official recipe preserved by the adapter

- Binary curvilinear segmentation (`classes=1`), three-channel RGB input.
- DRIVE configuration: `base64` backbone, `steps=3`, all GAU/FIM/up flags
  enabled, 256×256 random crop, batch 2.
- Normalization: ImageNet mean/std (`[0.485,0.456,0.406]`,
  `[0.229,0.224,0.225]`), as in the DRIVE configuration.
- Boundary targets are generated from each binary mask with the official
  3×3 max-pool operator (`abs(mask - max_pool2d(mask, 3, 1, 1))`, threshold
  0.5); no extra boundary annotations are needed.
- Loss: sum over three refinement steps of the four-scale mask BCE plus the
  four-scale boundary BCE (`JTFN.compute_objective`, multi-layer mode).
- Optimizer: Adam, lr 1e-3, weight decay 5e-4.  The official per-iteration
  StepLR (step 500, gamma 0.9) is approximated by a per-epoch StepLR whose
  epoch step is `round(500 / iterations_per_epoch)`.
- Checkpoint selection: the shared 6-image validation Dice.
- Inference: full-resolution image (padded to a multiple of 32 for shape
  safety and cropped back), sigmoid > 0.5, no test-time augmentation.

## Deviations in this port

- The official repository has separate configs for DRIVE (`base64`, 1000
  epochs), roads (`resbase32`, 500 epochs) and Crack500 (`resbase32`, 2000
  epochs, 360 crop).  The port uses the DRIVE setting as the single shared
  3-shot model; `--jtfn-backbone` and `--jtfn-steps` expose the other
  backbones/refinement counts.
- The official validation uses the dataset test split and selects by F1;
  the port selects by the shared validation Dice to match the other
  baselines.
- The official 1000-epoch DRIVE recipe is kept as the default; users can
  override `--epochs` for a shorter 3-shot run.
- The official arbitrary-angle/color augmentation is simplified to random
  flips, 90° rotations, ±20° in-plane rotations and brightness/contrast
  jitter applied before the 256 crop.  The boundary target is generated
  before cropping, matching the official offline label.
- No official JTFN checkpoints are present; training starts from scratch.
