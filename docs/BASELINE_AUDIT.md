# Baseline porting audit

Date: 2026-10-24

Scope: all self-contained models in `src/baseline_zoo/adapters/`
(13 runnable adapters at audit time; the `sam2_unet` appendix was added
later).  Each port was compared against the vendored
reference in `baselines/` and, where applicable, the released entrypoint
(`train.py`, `test.py`, shell scripts, or framework docs).  This audit also
covers the shared runtime, protocol loader, prediction path, metric
collector and report pipeline.

## Result

| Model | Status after audit | Main fixes |
|---|---|---|
| U-Net (`unet`) | OK with documented 3-shot deviations | official batch Dice (eps 1e-6, empty-set branch); logit-space upsample; LR/WD/batch and 1-class/RGB deviations documented |
| CS2Net (`cs2net`) | OK | corrected LambdaLR one-epoch offset to match upstream `adjust_lr(iter=epoch)` |
| BCUNet (`bcunet`) | OK | prediction no longer downloads ImageNet-22k; threshold-before-resize masks; bilinear-logit upsample + argmax; corrected selection/accumulation docs |
| MaskVSC (`maskvsc`) | OK | no confirmed bug; deviations re-checked |
| nnWNet (`nnwnet`) | OK with documented protocol deviation | checkpoint discovery by `.pth`/`.safetensors`; seeded runs; staging/preprocessing regenerated on reuse; resume rejected with a clear error; docs corrected |
| SAM-Med2D (`sam_med2d`) | OK for the controlled vanilla-SAM protocol | full SAM decoder inheritance (`mask_tokens`, hypernetworks, IoU head); decoder-head regression test; docs updated |
| SAM-OCTA (`sam_octa`) | OK | removed pre-model padding; SAM now records the pre-pad size and crops postprocess correctly for non-square images |
| CWSAM (`cwsam`) | OK | bilinear image resize matches `transforms.Resize`; UPSTREAM checkpointing claims corrected |
| KnowSAM (`knowsam`) | OK | fixed protocol/validation leak in unlabeled pool; CORN source root; protocol seeding; CPU-safe import/training path; equal labeled/unlabeled streams; stale staging cleared; `--epochs` wired to iteration budget; docs updated |
| SegDINO (`segdino`) | OK with local-weight requirement | auto-detects a size-matched checkpoint under `checkpoints/` (no env setup); prediction reconstructs the saved backbone size without a gated download; DPT uses the selected backbone’s layer indices; strict checkpoint coverage check; `drop_last=True`; config stored in checkpoint |
| SACM official (`sacm_official`) | OK | gradient clipping 1.0 is now effective (`--clip-grad-norm` → `grad_clip`); bilinear resize; sigmoid-threshold correction documented; UPSTREAM claims corrected |
| SAM ViT-L + U-Net (`sam_vit_l_unet`) | OK | augmentation moved back after the frozen encoder (matching the previous adapter); decoder-only checkpoint format restored; 1024 input guard restored |

## Shared runtime/pipeline fixes

- Training membership is now read from `split_manifest.csv`; stray or
  missing pairs fail instead of silently changing the 18/6 protocol.
- Partial gradient-accumulation groups are rescaled before the optimizer
  step.
- Checkpoints are written atomically; a run that never produces a finite
  best score fails instead of returning a missing checkpoint.
- Prediction restores the checkpoint’s `input_size` and adapter-specific
  config, validates prediction shape before saving, and removes stale
  metadata before a rerun.
- Resume validates the checkpoint selection, truncates history/CSV to the
  checkpoint epoch, and documents that it is weights-only unless
  `--save-optimizer` was used.
- Duplicate seeds/models are deduplicated; `.train_done`/`.eval_done`
  markers now record the checkpoint identity and are invalidated when the
  checkpoint changes.
- Added the missing `fixed` protocol config and forwarded `PRESET` to
  SACM-X evaluation.
- Metric collection accepts 0/1 as well as 0/255 PNGs, includes GT files in
  the cache signature, and refuses to overwrite a complete table with a
  subset unless `--allow-partial-overwrite` is passed.
- Reports select the requested SACM-X preset and fail on missing
  model/seed runs unless `--allow-missing` is passed.
- `run_paper.sh --dry-run` no longer regenerates figures.
- Eval resume now also requires a non-empty eval artifact
  (`paper_metrics.csv` or SACM-X `tables/all_metrics.csv`); a stale
  `.eval_done` can no longer hide deleted predictions/tables.
- Per-dataset metric caches and report tables are written atomically
  (`.tmp` + `os.replace`), so an interrupted write cannot leave a
  truncated CSV/Markdown/LaTeX file.

## What is unified, and what is intentionally model-specific

Unified interface:

- **Data/protocol**: one `split_manifest.csv` membership check, 18 train /
  6 val, the same 20 frozen test roots and image stems for every model.
  Test roots come from a single parser (`scripts/pipeline/protocol_specs.py`);
  `run_baseline_test.read_specs` delegates to it, so baseline and SACM-X
  evaluation resolve the same roots and fail identically on a conflicting
  dataset root.
- **Adapter input contract**: original-resolution PIL image plus boolean
  GT mask; adapters own their official resize/normalization/channel/prompt
  preprocessing.
- **Prediction output**: one boolean mask at the original GT resolution per
  image, written as 0/255 PNG with the same `<dataset>/<stem>.png` layout by
  both the generic and custom pipelines.
- **Metrics**: Dice, IoU, clDice and HD95 all come from `core.metrics`;
  validation selection and paper-metric collection now call the same
  functions with the same 0.5 threshold and empty-mask conventions.
- **Reporting**: prediction tables and macro tables share one schema and
  one report/aggregate path; checkpoint split hashes are recorded and
  checked.

Not unified, by design or documented exception:

- Model-internal input size, normalization, channel order, augmentation and
  prompt type differ because each model follows its official recipe.
- `nnWNet` uses an nnU-Net model-folder checkpoint and EMA pseudo-Dice
  selection; `KnowSAM` uses an SGDL-only checkpoint and its official
  validation; `SAM ViT-L + U-Net` stores a decoder-only state dict. The
  runner handles all three explicitly.
- SAM-Med2D/SAM-OCTA evaluation uses GT box/point prompts (oracle), so they
  are not prompt-free baselines.
- Training epochs, LR, batch size and augmentation remain method-specific
  official recipes; only the data budget and evaluation protocol are shared.

## Documented residual limitations (not silent bugs)

- **SAM-Med2D**: official training recipe (`batch=2`, `mask_num=5`,
  `iter_point=8`, ~642M trainable Adapter) exceeds 11 GiB; use ≥24 GiB or a
  reduced, explicitly non-official recipe. Evaluation uses GT box prompts
  (oracle). The port uses vanilla SAM ViT-L by design for the shared-backbone
  comparison, not `sam-med2d_b.pth`.
- **SAM-OCTA**: evaluation uses GT-derived point prompts (oracle).
- **nnWNet**: nnU-Net planning sees the six validation cases; documented
  soft protocol deviation.
- **KnowSAM**: semi-supervised unlabeled pool from source domains; not a
  purely supervised baseline.
- **SegDINO**: real training needs DINOv3 weights. A size-matched file under
  `checkpoints/` is detected automatically; otherwise set `DINO_CKPT` /
  `--dino-ckpt` or `HF_TOKEN`. Prediction no longer downloads them.
- **U-Net**: the 3-shot recipe uses AdamW/cosine and a higher LR/batch than
  the upstream RMSprop example; these are explicitly listed as protocol
  deviations.
- Full 5-seed × 20-dataset training has not been run; this audit covers
  source-level correctness and one-step/short smoke tests.

## Verification performed

- All 13 adapters import with `CUDA_VISIBLE_DEVICES=''`.
- Full fast test suite: exit 0, one slow test skipped.
- `python -m baseline_zoo doctor`: 0 missing assets/dependencies.
- One-epoch/short smoke training + prediction for U-Net, CS2Net, BCUNet,
  MaskVSC, nnWNet, KnowSAM, SAM-OCTA, CWSAM, SACM official, SAM ViT-L +
  U-Net, SegDINO; SAM-Med2D build/forward/prediction and decoder
  initialization check (full training blocked by GPU memory).
- Non-square SAM-OCTA prediction verified on a 600×800 CrackTree200 image.
- KnowSAM staging verified on all six source domains, including CORN, with
  zero protocol labeled/validation images in the unlabeled pool.

## Addendum (2026-10-25)

`sam2_unet` was added as a self-contained appendix baseline after this
audit (vendored SAM2 Hiera-L + prompt-learn adapters + U-Net decoder,
`tests/test_sam2_unet.py`, Hydra/import isolation checks).  It is outside
the paper-ten audit scope and not part of the main table.

## UCS addendum (2026-10-25)

`ucs` was added as an oracle-prompted reference baseline (vendored UCS SAM
fork, `tests/test_ucs.py`).  The released code derives GT boxes/points and
samples refinement clicks from prediction-vs-GT errors, so it is not
prompt-free.  It uses the official 20-epoch recipe and training-Dice
selection.  HEAD `cff2db1` also refactored away the paper Sparse Adapter/HFC
modules; see `src/baseline_zoo/models/ucs/UPSTREAM.md`.

## SAM-Adapter addendum (2026-10-25)

`sam_adapter` was added as a prompt-free main-table baseline (vendored
SAM-Adapter/EVP ViT-L with the FFT high-pass prompt generator, per-block
prompt injection and `no_mask_embed` decoder; `tests/test_sam_adapter.py`).
It preserves the official 20-epoch AdamW 2e-4/cosine recipe, the
BCEWithLogits+IoU loss and the freeze policy (image encoder frozen except
the prompt generator; mask decoder trained).  Documented deviations: it is
retrained from SAM ViT-L because the authors' Google-Drive weights are not
in the released tree; validation uses the shared 6-image Dice instead of the
official COD metrics; predictions are exported at GT resolution; and the
official batch 2 default can be overridden to batch 1 on ≤11 GiB.  The
executable upstream wrapper applies horizontal flips even though its YAML
sets `augment: false`; the port follows the code.

## TopoLoRA-SAM addendum (2026-10-25)

`topolora` was added as a prompt-free main-table baseline (vendored
TopoLoRA-SAM from `salimkhazem/Seglab` HEAD `7de3a5f`, plus a minimal
standard Meta SAM package; `tests/test_topolora.py`).  It preserves the
official ViT-B, LoRA r=8 on `mlp`/`fc1`/`fc2`, 256-channel conv adapter,
frozen SAM + trainable mask decoder, empty-prompt decode,
BCE+Dice+0.5 clDice, AdamW 1e-4/1e-4/cosine, 50 epochs, batch 1×4 and
gradient clipping 1.0.  Documented deviations: no 16-mixed AMP; the shared
18/6 split and 20-dataset protocol instead of Seglab's HuggingFace split;
GT-resolution export; and no official trained weights (training starts from
SAM ViT-B).  Albumentations 1.3.1 uses `value`/`mask_value` instead of the
upstream `fill`/`fill_mask`; the padding behaviour is identical.

## SAMed addendum (2026-10-25)

`samed` was added as a prompt-free, oracle-free main-table baseline (vendored
SAMed modified SAM + rank-4 LoRA; `tests/test_samed.py`).  It preserves the
official class-token mask decoder, q/v LoRA placement, CE 0.2 + Dice 0.8
loss on 128×128 low-res logits, and AdamW 5e-3/0.1 poly recipe.  The port
uses a binary 2-class head (the official model is 8-class Synapse), direct
512 input (instead of lossy 224→512 test resampling), a 5-epoch warmup and
per-epoch poly decay, batch 4/160 epochs on the 18-image split, and the
shared validation-Dice selection.  The released 8-class SAMed/SAMed_s/
SAMed_h LoRA weights are absent and not shape-compatible; training starts
from SAM ViT-B.  Predictions are exported at original GT resolution.

## JTFN addendum (2026-10-25)

`jtfn` was added as a non-SAM from-scratch baseline (vendored `models/jtfn`,
`modules.py` and both backbones; `tests/test_jtfn.py`).  It preserves the
official DRIVE configuration: base64 backbone, three mask/boundary
refinement steps, four-scale mask BCE + boundary BCE, 256 crop, batch 2,
Adam 1e-3/5e-4 and 1000 epochs.  Boundary targets are generated from the
binary GT masks with the official 3×3 max-pool difference.  Documented
deviations: one shared model instead of per-dataset DRIVE/roads/Crack500
configs, validation-Dice selection instead of test-split F1, per-epoch
StepLR approximating the 500-iteration schedule, simplified augmentation,
32-pixel padding for original-resolution inference, and no official
weights.  The upstream repository has no LICENSE file; see
`src/baseline_zoo/models/jtfn/UPSTREAM.md`.

## DSCNet addendum (2026-10-25)

`dscnet` was added as a non-SAM from-scratch baseline (vendored DSCNet 2D
model, DSConv and soft-BCE loss; `tests/test_dscnet.py`).  It preserves the
official dynamic-snake U-Net, 16 basic channels, kernel 9, 224 crop, batch
1, AdamW 1e-4 with betas (0.9,0.95), 400 epochs and constant LR.  The
NIfTI/MONAI/SimpleITK pipeline is replaced by PIL grayscale I/O, simple
flips/rotations/noise, train-split scalar mean/std and the official
224/112 patch inference.  The upstream blending kernel's 1/0 center is
replaced by a finite `1/(1+d^4)` window and explicit edge coverage; the
initial clamp was also corrected to avoid suppressing edge probabilities.
Documented deviations: validation-Dice selection instead of test-set Dice,
GT-resolution export, and no official weights.  See
`src/baseline_zoo/models/dscnet/UPSTREAM.md`.

## PerSAM addendum (2026-10-25)

`persam` was added as a support-based training-free baseline (vendored
Personalize-SAM modified SAM predictor/model, `tests/test_persam.py`).  It
preserves the official one-shot inference sequence: reference mask
embedding, target feature cosine similarity, top-1 foreground and
least-similar background points, target-guided attention, mask-input
refinement and the box/mask second refinement.  SAM ViT-H is loaded at
prediction; the training stage writes a metadata-only checkpoint through
the new `training_free` runtime path.  The 3-shot protocol is an extension:
each of the three support images is run independently and the binary masks
are majority-voted; unseen/novel datasets use a structurally matched donor
support domain.  The support mask is converted to RGB 0/255 as the official
predictor requires, empty first-pass masks skip refinement, and the
PerSAM-F fine-tuned variant is not used.  No target GT or target prompts
are used; the method is support-based, not prompt-free.

## Matcher addendum (2026-10-25)

`matcher` was added as a support-based training-free baseline (vendored
Matcher, modified SAM fork and DINOv2 ViT-L/14; `tests/test_matcher.py`).
It preserves the official support/target 518 square preprocessing, DINOv2
patch matching, k-means prompt selection, modified SAM mask generation and
official filter defaults (EMD/purity/coverage off, top-k 0.7, 10 merging
masks).  It natively consumes the three project support shots; unseen/novel
datasets use a structurally matched donor domain.  Predictions at 518 are
bilinearly upsampled to the original GT resolution.  The training stage
writes a metadata-only checkpoint through the shared `training_free` path;
SAM ViT-H is present, while DINOv2 ViT-L/14 weights must be downloaded to
`checkpoints/dinov2_vitl14_pretrain.pth` (`--matcher-random-init` is a
code-smoke-only option).  Support DINOv2 features are cached after
`set_reference` to avoid re-encoding the three supports for every target.
No target GT or target prompts are used.
