# Baseline porting audit

Date: 2026-10-24

Scope: all self-contained models in `src/baseline_zoo/adapters/`
(13 runnable adapters).  Each port was compared against the vendored
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
| SegDINO (`segdino`) | OK with gated-weight requirement | prediction reconstructs the saved backbone size without a gated download; DPT uses the selected backbone’s layer indices; strict `--dino-ckpt` coverage check; `drop_last=True`; config stored in checkpoint |
| FoB-SAM (`fob_sam`) | OK; code-faithful, not paper-formula exact | scheduler now steps once per 3000-iteration epoch; support-split fallback warns; 2D augmentation/normalization documented; paper-vs-code differences recorded |
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

## What is unified, and what is intentionally model-specific

Unified interface:

- **Data/protocol**: one `split_manifest.csv` membership check, 18 train /
  6 val, the same 20 frozen test roots and image stems for every model.
  Test roots come from a single parser (`scripts/protocol_specs.py`);
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
- **FoB-SAM**: the released code differs from the CVPR paper in the RAC
  loss, support ring, foreground threshold, training budget and several
  normalization details; the port stays code-faithful. See
  `models/fob_sam/UPSTREAM.md`.
- **nnWNet**: nnU-Net planning sees the six validation cases; documented
  soft protocol deviation.
- **KnowSAM**: semi-supervised unlabeled pool from source domains; not a
  purely supervised baseline.
- **SegDINO**: real training requires gated DINOv3 weights (`HF_TOKEN`,
  `DINO_CKPT`, or `--dino-ckpt`); prediction no longer downloads them.
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
  U-Net, SegDINO, FoB-SAM; SAM-Med2D build/forward/prediction and decoder
  initialization check (full training blocked by GPU memory).
- Non-square SAM-OCTA prediction verified on a 600×800 CrackTree200 image.
- KnowSAM staging verified on all six source domains, including CORN, with
  zero protocol labeled/validation images in the unlabeled pool.
