# SANSA upstream record

## Sources

| Component | Upstream | Commit |
|---|---|---|
| Method / adapter / training loop / loss | `ClaudiaCuttano/SANSA`, `models/sansa/{sansa,adapter,model_utils}.py`, `engine.py`, `main.py`, `util/losses.py`, `inference_fss.py` | `f562b5642940e08a57e3da2608b804b99af3763b` |
| Modified SAM 2 | same repo, `models/sam2/` and `models/sam2/sam2_configs/` | same |

Paper: "SANSA: Unleashing the Hidden Semantics in SAM 2 for Few-Shot
Segmentation", NeurIPS 2025, arXiv:2505.21795.

License: the checkout contains a LICENSE file; the vendored SAM 2 files
carry Meta's Apache-2.0 header.

## Copied and adapted

- `adapter.py`: verbatim AdaptFormer block (pre-LayerNorm down-projection,
  ReLU, zero-initialized up-projection, scalar 0.1).
- `model_utils.py`: verbatim `BackboneOutput` / `DecoderOutput` containers.
- `sam2/`: SANSA's modified SAM 2 (23 files). Imports were rewritten from
  `models.sam2.*` to `baseline_zoo.models.sansa.sam2.*` and from
  `models.sansa.*` to `baseline_zoo.models.sansa.*`; Hydra `_target_`
  strings were rewritten the same way. The package `__init__` is empty (no
  global Hydra state).
- `sam2/sam2_configs/`: Hiera-T, Hiera-B+ and Hiera-L configs with
  vendored targets.
- `model.py`: SANSA's pseudo-video forward reimplemented against the
  vendored classes: preprocess to 1024, one backbone pass over all frames,
  per-sequence memory bank, mask-prompt reference frames, memory-conditioned
  target frames, memory entries and final mask interpolation; plus
  `focal_dice_loss` and a stateless `build_sansa` that sets
  `adaptformer_stages`, `adapt_dim`, disables the object-score branch and
  loads the SAM 2 checkpoint with adapter-aware strictness.

## Invariants kept from the paper / official code

- SAM 2 Hiera-Large is frozen. AdaptFormer adapters are inserted into the
  last two hierarchical stages (`adaptformer_stages=[2, 3]`), with
  bottleneck `int(dim * 0.3)` and scalar 0.1. Only parameters whose names
  contain `adapter` are trainable (≈9.9M).
- Reference images are encoded with a mask prompt (`_use_mask_as_output`);
  target frames are decoded from memory-conditioned features; every frame,
  including targets, is written to the memory bank for subsequent frames.
  The memory bank is reset per pseudo-video.
- Reference frames are processed before targets and do not undergo memory
  attention, so the result is invariant to reference ordering.
- Inputs are resized to 1024 square and ImageNet-normalized (the released
  dataset transforms), and masks are resized to 1024. The released
  `preprocess` only resizes (its normalization lines are commented out).
- Loss: the released `loss_masks` = sigmoid focal (α=0.25, γ=2) + Dice,
  applied to the target frame only for mask prompts.
- Optimizer/schedule: AdamW lr 1e-4, betas (0.9, 0.999), wd 0; cosine
  annealing; gradient clipping. Paper: 5 epochs, batch 32, clip 1.0.

## Paper / code discrepancies and choices

1. Training episode: the paper describes one reference + J=3 targets and
   states BCE+Dice. The released `engine.py` builds `random_shot` supports
   + one query (with `--shots` default 1) and uses focal+Dice. This port
   follows the released engine (one reference + one target, focal+Dice)
   and documents both discrepancies.
2. Gradient clipping: paper 1.0, released CLI default 0.1. The port uses
   the paper value 1.0.
3. Schedule: released cosine is stepped per iteration; the shared runtime
   steps once per epoch.
4. Budget: official 5 strict-few-shot epochs on thousands of images; the
   port uses the shared 800-epoch, 18-image budget with batch 2 (24 GiB
   target GPUs) and validation-Dice selection.
5. Support prompts: the release supports mask/point/box/scribble; the
   protocol supplies masks, so only mask prompts are exposed.
6. No official SANSA adapter weights are required; training starts from
   the official SAM 2 Hiera-Large checkpoint. The port never downloads
   weights.
7. DINOv2 is not used (SANSA has no auxiliary model); only SAM 2 and the
   adapters are needed.

## SAM 2 checkpoint compatibility

The vendored config is the original SAM 2.0 Hiera-Large config. Its
checkpoint `sam2_hiera_large.pt` contains unused object-score parameters
because SANSA disables `pred_obj_scores`; those keys are allowed to be
unexpected. Missing non-adapter keys or any other unexpected key raises an
error. Adapter keys are intentionally absent from the base checkpoint and
are initialized by AdaptFormer (up-projection zero).
