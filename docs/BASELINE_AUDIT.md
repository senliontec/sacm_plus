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

`samed` is a prompt-free, oracle-free SAMed-inspired baseline (vendored
SAMed modified SAM + rank-4 q/v LoRA; `tests/test_samed.py`).  It preserves
the class-token mask decoder, q/v LoRA placement, frozen image encoder and
trainable prompt/mask decoder.  The binary objective is adapted for the
0.17%-8% curvilinear foreground: class-weighted CE (background 0.1,
foreground 1.0) + foreground-only Dice + soft-clDice (0.1) on 128×128
logits, plus a full-resolution auxiliary CE+Dice term (0.5).  A
zero-initialized full-resolution residual refinement head, supervised by
that auxiliary term, corrects the bilinearly upsampled logits for thin
structures; it is a task adaptation and can be disabled with
`--samed-high-res-head 0`.  Low-foreground images are oversampled by
inverse sqrt foreground ratio.  AdamW uses the batch-4 default 1e-3 (paper
5e-3 was for batch 24), weight decay 0.1, gradient clipping 1.0, 50-epoch
warmup and per-epoch poly decay, batch 4/800 epochs on the 18-image split,
and shared validation-Dice selection.  The released
8-class SAMed/SAMed_s/SAMed_h LoRA weights are absent and not
shape-compatible; training starts from SAM ViT-B.  Evaluation uses argmax
at the input resolution (with the refinement head) followed by nearest
resize to the original GT resolution.

Implementation smoke (seed 0, 60 epochs at 512): the refinement head reaches
0.4345 validation Dice versus 0.4219 with the head disabled; 1024-input
batch-1 training and prediction from a legacy headless checkpoint also run.
These are smoke checks, not final 800-epoch/HPO results.

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

## MedCAGD addendum (2026-10-27)

`medcagd` is a non-SAM binary medical baseline (ECCV 2026, MIT; vendored
context-aware gated decoder, PVTv2-B2 timm encoder, deep supervision and
edge BCE).  The upstream `cfgs`-driven constructor is parameterized.  A
seed-0 reproduction exposed a BCE-only failure on this protocol: training
loss decreased while validation (and training-image) predictions became
all background from epoch 18, giving Dice 0 on every validation image.  The
loss therefore adds a soft Dice term on the main output (weight 1.0,
`--medcagd-dice-weight 0` for the paper's BCE-only objective); with it,
validation Dice reached ~0.50 by epoch 200.  The repeated LR waves are the
upstream `CosineAnnealingWarmRestarts(T_0=5, T_mult=1)` schedule and are
not the cause (constant LR shows the same BCE-only collapse).  The
paper uses 256/128 patches for DRIVE/FIVES and 224 full images for its
large-object benchmarks.  All 20 protocol datasets are thin curvilinear
structures, so the default task adaptation uses the patch recipe for every
dataset: 256 crops for training and overlapping 256 patches (stride 128)
with probability averaging at inference.  `--medcagd-input-policy paper`
reproduces the paper's 224-full/256-patch mapping.  The paper's final
model excludes the released code's raw-image Stage 0 path, so
`use_stage0=False` is the default (`--medcagd-use-stage0 1` reproduces the
code variant).  Documented deviations: batch 8 instead of paper 16,
Albumentations/CutMix replaced by seeded crops/flips/rot90, shared
800-epoch budget with validation-Dice selection, no CuPy evaluator, and
no official trained weights.  The encoder is initialized from a local
`pvt_v2_b2.in1k` file (`checkpoints/pvt_v2_b2.in1k.safetensors`); the port
never calls timm's downloading `pretrained=True` path, so an offline server
fails fast with the exact download URL if the file is missing.  See
`src/baseline_zoo/models/medcagd/UPSTREAM.md`.

## BLO-SAM addendum (2026-10-27)

`blo_sam` is a limited-data SAM baseline (SAM ViT-B with the image encoder
frozen, rank-4 LoRA on mask-decoder self/cross attention and the final
token-to-image attention, trainable no-mask prompt embedding, binary
background+foreground head).  The official bilevel update is preserved: a
fixed seeded 9/9 D1/D2 split stratified across the six pooled domains,
network update on D1, prompt-embedding update on D2, D2-Dice checkpoint
selection, and the effective per-step polynomial LR (the upstream
`OneCycleLR` is constructed but never stepped).  The
released loop rereads only the first D2 batch every step; the port cycles
through all D2 batches.  Other documented deviations: shared 18-image
split, flips/rot90, official 100-epoch budget, and no official BLO-SAM
weights.  The paper reports Dice from 64×64 logits; this protocol exports
a GT-resolution mask by bilinearly upsampling the foreground logit before
thresholding (not argmax at 256 then nearest resize).  The upstream README claims MIT but the checkout has no LICENSE
file; verify before redistribution.  See
`src/baseline_zoo/models/blo_sam/UPSTREAM.md`.

## Conv-LoRA addendum (2026-10-28)

`convlora` is a prompt-free SAM ViT-H baseline (ICLR 2024).  The official
implementation ships inside AutoGluon (`examples/automm/Conv-LoRA` plus
the multimodal SAM model), licensed Apache-2.0.  The port keeps the
official layer math: rank-3 LoRA on the fused qkv linear, top-1 noisy
MoE over eight 3x3 conv experts with bicubic up/down scaling 1..8, the
`1e-2` per-layer load-balancing loss summed and added with weight 1.0,
structure loss, one binary mask token, frozen prompt encoder and constant
no-mask prompt, and the official 1024-square bicubic/nearest transforms
with random horizontal flip.  Two paper/code discrepancies are resolved
in favour of the paper: (1) the paper fully fine-tunes the mask decoder
and its 4.02M trainable count equals the 3.51M decoder plus ~0.51M
Conv-LoRA; the released AutoGluon PEFT optimizer filter (`.*lora_*`)
would freeze the decoder, so the port trains it except the IoU head,
which the official SAM config explicitly freezes; (2) the paper states
Adam with lr/wd 1e-4, while the released semantic-segmentation preset
uses AdamW/wd 1e-4, so the paper's optimizer/values are used.  The
released AutoGluon data pipeline normalizes with ImageNet statistics
(`image_norm: imagenet`), not SAM's pixel mean/std; the port follows that.
The same semantic preset sets `polynomial_decay` (power 1, no warmup,
`end_lr=0`), gradient clipping 1.0 and `lora_alpha=32`; those values are
used, stepped per epoch rather than per iteration.  Layerwise LR decay,
top-3 greedy-soup averaging and early stopping are not reproduced (shared
validation-Dice selection).  The official 30-epoch (100-epoch low-data)
budget is extended to the shared 800 epochs.  `lora_alpha` defaults to 32
(the paper does not state it) and is exposed via `--convlora-alpha`.  The
released processor's `RandomHorizontalFlip(1.0)` is treated as an
upstream quirk; p=0.5 is used per the paper.  No
official Conv-LoRA weights exist, so training starts from the local SAM
ViT-H checkpoint.  See `src/baseline_zoo/models/convlora/UPSTREAM.md`.

## FSSAM addendum (2026-10-28)

`fssam` is the ICML 2025 SAM 2 few-shot segmenter, a support-based
baseline (not prompt-free).  The port vendors FSSAM's modified SAM 2
(Hiera-S/B+/L configs), reuses the self-contained DINOv2 implementation,
and keeps the paper's three modules: DINOv2 pseudo FG/Disc priors,
iterative memory refinement, and support-calibrated memory attention.
Official training freezes the SAM 2 image/prompt encoders, `obj_ptr_proj`,
`mask_downsample` and DINOv2, and optimizes only the memory encoder,
memory attention and mask decoder (AdamW 1e-3, wd 0, poly power 0.9, Dice
loss, 512 square, effective batch 8).  The port extends the official
300/75 epochs to the shared 800-epoch budget and uses validation Dice
instead of the official per/half-epoch mIoU with patience-100 early
stopping.  Because each seen training domain has only three images,
training episodes use at most two non-query supports; evaluation uses the
protocol's support set (own shots for seen domains, donor for
unseen/novel).  The DINOv2 prior uses the final block only (the upstream
requests all intermediate blocks but consumes only the last, so the feature
is identical).  The released code cannot run its SCMA calibration as
published: `self.Weighted_GAP` is never defined and the support-mask tensor
is overwritten inside the shot loop.  The vendored copy supplies the
missing weighted-GAP helper and averages the per-shot support similarities
as described in the paper.  The official main configuration is SAM 2-S +
DINOv2-B; the port defaults to SAM 2-S + locally available DINOv2-L (the
paper's Table 7 variant) and exposes `--fssam-dino-size vitb`.  The port
runs in fp32 (the official trainer uses bf16 autocast) and casts the
memory features that the upstream helper hard-codes to bf16 back to the
model dtype.  No FSSAM-trained weights exist in the released tree;
training starts from the official SAM 2 and DINOv2 checkpoints.  See
`src/baseline_zoo/models/fssam/UPSTREAM.md`.

## SANSA addendum (2026-10-28)

`sansa` is the NeurIPS 2025 semantically aligned SAM 2 few-shot segmenter
(support-based, not prompt-free).  The port vendors SANSA's modified SAM 2
(Hiera-T/B+/L configs) and its AdaptFormer adapter, keeps SAM 2 frozen and
trains only the adapters (≈9.9M) in the last two hierarchical stages with a
0.3× bottleneck, exactly as the official builder.  Reference masks plus the
target are processed as a pseudo-video through the unmodified memory
attention path.  The released training code (`engine.py`) trains one
reference + one target with mask prompts and focal+Dice, although the paper
describes BCE+Dice and a J=3 multi-target objective; the port follows the
release.  The official 5-epoch, effective-batch-32 recipe is extended to the
shared 800-epoch budget with batch 2 on 24 GiB GPUs and validation-Dice
selection.  The released datasets ImageNet-normalize and resize to 1024
before SAM 2 (the `preprocess` helper only resizes); the port matches that.
Evaluation uses the model's native k-shot path with the protocol's three
references (own shots for seen sets, donor for unseen/novel), mask prompts
only.  Documented deviations: released gradient-clip CLI default 0.1 vs the
paper's 1.0 (port uses 1.0); per-epoch instead of per-iteration cosine;
no official adapter weights (training starts from SAM 2 Hiera-Large); no
support prompts other than masks.  See
`src/baseline_zoo/models/sansa/UPSTREAM.md`.

## RSRM addendum (2026-10-28)

`rsrm` is the ECCV 2026 training-free cross-domain FSS baseline
(support-based, DINOv3-only, no SAM).  The port vendors the Meta DINOv3
implementation used by the official repo and keeps SAFR (variance-proportion
layer selection + Fisher-ratio weighting over raw attention features), ASE
(robust support-derived FG/BG thresholds + masked query-to-support
cross-attention) and HPM (global, k-means regional and pixel-level prototype
matching fused by confidence).  Official preprocessing is 480-square resize
with ImageNet normalization; the official test loop argmaxes the two-channel
output at 480, and this port exports that mask nearest-resized to the GT
resolution.  The official `fast_pytorch_kmeans` GPU dependency is replaced
by deterministic scikit-learn k-means++ on CPU; the region-similarity math
is unchanged.  The paper uses Meta DINOv3 ViT-B/16.  The local
`checkpoints/dinov3/pytorch_model.bin` is a timm ViT-S/16 export
(`reg_token`/`gamma_1` keys) and is loaded through timm's own architecture
rather than force-fitted onto the Meta module layout; its `F_attn` is
captured from a `block.attn` hook plus the block's `gamma_1` layer scale
(the earlier `block.norm2` pre-hook wrongly included the residual).
Official Meta ViT-B/16 state dicts are accepted with the vendored
architecture.  No RSRM-specific
weights exist; the training stage writes a metadata-only checkpoint.  No
target GT or target prompts are used.  See
`src/baseline_zoo/models/rsrm/UPSTREAM.md`.

## C2P addendum (2026-10-28)

`c2p` is the ECCV 2026 prompt-free universal segmentation model
(ConvNeXtV2-B, 9 geometric tokens supervised on physical attributes, 9
MLLM-distilled semantic tokens, bidirectional token-image cross-attention,
token-guided dynamic head, geometry-aware inference consensus).  The
released repository is not runnable end-to-end: `train.py` imports a
non-existent `SRICL` class, `evaluate_sem.py`/`train_sem.py` referenced by
the README are absent, and no property-cache or MLLM-embedding generator is
provided.  The full `C2P` network is present, and the official HuggingFace
checkpoint (`yunyundi/c2p`) supplies the trained backbone, both token sets,
and the dynamic head.  This port therefore evaluates the official
checkpoint **zero-shot** as the zoo's prompt-free universal foundation
baseline; it does not consume the shared 18-image split, support images or
target labels.  The official geometry-aware 3-view consensus is
reimplemented from `predict_with_token_tta_fast` (area/centroid consistency
weighting; empty-prediction suppression).  Inputs are 384 square with
ImageNet normalization; the sigmoid output is exported at GT resolution.
The checkpoint contains the ConvNeXtV2-B backbone, so timm is built with
`pretrained=False` and never downloads weights; `--c2p-random-init` is
smoke-only.  Documented limitation: the model was trained on eight medical
datasets and is applied zero-shot to the protocol's mixed medical and
non-medical curvilinear domains, analogous to the paper's unseen-modality
test.  No official weights are bundled.  See
`src/baseline_zoo/models/c2p/UPSTREAM.md`.

## DKA addendum (2026-10-28)

`dka` is the ICLR 2026 Dual-Kernel Adapter segmentation baseline.  The
released repository includes the DKA module (linear down-projection to a
bottleneck, parallel depthwise 51×51 and 5×5 branches, GELU, linear
up-projection, residual) and classification training; its segmentation
experiments use Segmenter-B in MMSegmentation, which is not part of the
release.  The port reconstructs MMSegmentation's Segmenter-B ViT-B/16 host
at the exact module/key level (learned absolute position embedding resized
24×24→32×32, `layers.*.attn.attn.*`, `layers.*.ffn.layers.*`,
`drop_path_rate=0.1` on the FFN branch only, final LayerNorm and a linear
1×1 FCN head), loads the official ImageNet-21k
`vit_base_p16_384_20220308-96dfe169.pth` checkpoint strictly, and uses
MMSeg's `(x-127.5)/127.5` normalization.  DKA is inserted after all 12
blocks (middle dimension 192).  The paper's segmentation recipe supplies
head lr 1e-4, DKA lr 1e-3, batch 1 and 300 epochs; the port uses the shared
800-epoch budget and validation-Dice selection.  No segmentation loss or
augmentation is stated in the paper or release, so the port uses
512-square inputs, horizontal flip only, and cross-entropy + foreground
Dice (Dice added to avoid the all-background degeneracy observed on sparse
curvilinear domains).  The released classification code uses a 1e-5
adapter LR and StepLR and leaves the paper's zero down/up initialization
commented out; the paper's segmentation values (1e-3 adapter LR) and stated
zero initialization are used.  No DKA-trained weights exist; training starts
from the official Segmenter-B checkpoint, and `--dka-random-init` is
smoke-only.  See `src/baseline_zoo/models/dka/UPSTREAM.md`.

## SCRWKV addendum (2026-10-28)

`scrwkv` is the ICML 2026 ultra-compact Structure-Calibrated Vision-RWKV
crack segmenter, a non-SAM from-scratch baseline.  The port vendors the
released SFE (AMCM + GBST/DSCD SCIU blocks), CSHF decoder with DySample
upsamplers and the custom CUDA `bi_wkv` kernel.  The released SFE imports
mmcv/mmcls (`build_norm_layer`, `ModuleList`, `AdaptivePadding`,
`resize_pos_embed`, `BaseBackbone`); those are generic helpers, and the port
replaces them with local torch equivalents without changing the architecture
or forward path for the fixed 512 input (corner padding is zero).  The CUDA
kernel is official and required; the release hardcodes sm_86, while the port
selects the current GPU capability or `SCRWKV_CUDA_ARCH`.  The official
recipe is kept: 512-square cubic resize for images and masks, mask threshold
127, `(x/255-0.5)/0.5`, no augmentation, 0.25 BCE + 0.75 Dice, AdamW
5e-4/0.01, PolyLR to 1e-6, batch 1.  The official 50 epochs are extended to
the shared 800-epoch budget with validation-Dice selection.  Inference
follows the released `test.py` (max-normalize logits, threshold 0.2) and
then nearest-resizes to GT, whereas the release evaluates at 512.  The port
has ~2.04M parameters versus the paper's 1.22M; the released Crack config
and CSHF/DySample modules are preserved, so the difference is in the paper's
reported count/config.  No official weights are required; training starts
from random initialization.  See
`src/baseline_zoo/models/scrwkv/UPSTREAM.md`.

## SAMRefiner addendum (2026-10-27)

`sam_refiner` is a model-agnostic binary-mask post-processor (ICLR 2025,
MIT), vendored with its modified SAM package and exposed through
`scripts/tools/apply_sam_refiner.py`.  It is not a backbone baseline and is
not part of the main table; compare it only as a post-processing ablation.
Defaults now follow the paper (one refinement pass, box margin 0.1,
Gaussian amplitude 15, span 4); the released code defaults (5/0.0/30) are
available explicitly.  Multi-object STM, HQ-SAM and SAMRefiner++ are not
exposed.  FastGeodis is optional, with a SciPy distance-transform fallback;
the upstream 2-D mask slicing bug is fixed in the vendored copy and also
worked around in the wrapper.  See
`src/core/postprocess/sam_refiner/UPSTREAM.md`.

## Post-audit fixes (2026-10-28, second pass)

A second pass against the papers and upstream code found four issues that
the first regression suite did not cover:

1. **FSSAM / SCRWKV multi-GPU.**  FSSAM's vendored SAM 2 helpers call
   `tensor.cuda()` (`sam2_video_predictor.py:1581`, `:1397`, `:1400`), and
   SCRWKV's `RUN_CUDA` calls `w.cuda()` etc. (`SCIU.py:201`).  Index-less
   `.cuda()` always targets the current CUDA device, which defaults to GPU
   0; jobs assigned to `cuda:1`..`cuda:7` failed with a device mismatch.
   `runtime._select_cuda_device()` now calls `torch.cuda.set_device()` in
   `run_training_free`, `run_training` and `run_prediction`, so the
   vendored helpers use the job's GPU.  The behavior is covered by
   `tests/test_baseline_zoo.py::test_select_cuda_device_targets_job_gpu`.
2. **SANSA evaluation.**  `SANSAAdapter.predict_sample` built a 5-D tensor
   (`logits[-1][None, None]`) and passed it to 2-D `F.interpolate`, which
   always raised.  It now keeps the query frame as 4-D (`logits[-1:]`);
   `tests/test_sansa.py` covers the GT-resolution bool-mask contract.
3. **RSRM timm DINOv3 features.**  The timm wrapper captured the input to
   `block.norm2`, i.e. `x + gamma_1*attn(norm1(x))`, while Meta DINOv3's
   `F_attn` is `gamma_1*attn(norm1(x))`.  The wrapper now hooks
   `block.attn`, applies `gamma_1`, then the final norm.  This matters for
   the local timm ViT-S path; the official Meta ViT-B/16 path was already
   correct.  `tests/test_rsrm.py` checks both `F_attn` and `F_sum` against
   a reference.
4. **Conv-LoRA recipe.**  The released example inherits AutoGluon's
   `semantic_segmentation` preset, which sets `lora_alpha=32`,
   `polynomial_decay` (power 1), `warmup_steps=0.0`, `end_lr=0` and
   `lr_decay=0`; the port had used the generic defaults (alpha 8,
   cosine+10% warmup).  Defaults and docs now follow the preset.  Adam
   (paper) is retained instead of the preset's AdamW, and the released
   `RandomHorizontalFlip(1.0)` is treated as a quirk (p=0.5 per paper).
5. **DKA Segmenter-B host.**  The first DKA port used a torchvision
   ViT-B/16-ImageNet-1k host.  It now reconstructs MMSegmentation's
   Segmenter-B ViT-B/16 at the module/key level (`patch_embed.projection`,
   `ln1`, `layers.*.attn.attn.*`, `layers.*.ffn.layers.*`), resizes the
   official checkpoint's 24×24 position embedding to 32×32, applies the
   0.1 FFN-only stochastic depth and final LayerNorm, uses the official
   ImageNet-21k checkpoint and `(x-127.5)/127.5` normalization, and keeps
   a linear 1×1 FCN head.  `load_state_dict(strict=True)` passes with the
   official `segmenter_vit_b16_384.pth`.  Earlier torchvision-host DKA
   checkpoints are incompatible and must be retrained.

Remaining known caveats are asset/host decisions rather than runtime bugs:
FSSAM's default DINOv2-L is the paper ablation (main table needs DINOv2-B),
RSRM's local timm ViT-S is not the paper's Meta ViT-B/16, and the C2P
official checkpoint is not present so strict loading is unverified.  DKA
now uses the exact MMSeg Segmenter-B host layout and official checkpoint;
only its unstated segmentation loss, augmentation, head choice and
single-scale test pipeline remain documented assumptions.

## ACO-MoE-LoRA addendum (2026-10-28)

The released ACO repository (`lky0234/ACO-MoE-LoRA`, commit `996270d`)
provides the ACO/MoE model and its training/test scripts, but no LICENSE and
no trained weights.  The official constructor builds full SAM2 with
`build_sam2("sam2_hiera_l.yaml")`, deletes the prompt/mask decoder, memory
modules and FPN neck, and keeps `image_encoder.trunk`.  The port builds that
Hiera-L trunk directly with the same config and loads only
`image_encoder.trunk.*` from the local checkpoint strictly; this is
equivalent to the used path and avoids vendoring the unused SAM2 runtime.

The port keeps the official frozen Hiera encoder, dynamic LoRA on every
attention `qkv`/`proj` Linear (max rank 32, `alpha/rank`), four ACO-ConvLoRA
decoder blocks with eight gated experts, the ACO rank/alpha/top-k search,
weighted BCE + weighted IoU + gate loss, AdamW 1e-3/1e-4 and cosine decay.
Documented deviations: validation-Dice selection instead of the official
minimum-training-loss rule; batch 4 x 3 accumulation instead of batch 12;
the shared 800-epoch budget instead of 50; the checkpoint stores the
hyperparameters active at the validation-best epoch, not necessarily the
global ACO training-best solution.  The ACO training IoU remains the
pheromone feedback signal, and internal search prints are suppressed so the
shared one-row-per-epoch table stays clean.  Assets: local
`checkpoints/sam2_hiera_large.pt`; `--aco-random-init` is smoke-only.

## SegDINO v2 addendum (2026-10-28)

`segdino_v2` vendors the MICCAI 2026 SegDINO v2 decoder (TPA/SAD: four token
projections, 8/4/2/1 bilinear branches, residual depthwise refinement per
branch and top-down fusion) over a DINOv3-S/16 timm backbone.  The paper is
absent from `parsing/`, so the port was checked against arXiv 2606.17972 and
`3rd/segdino_v2`.  The paper states a frozen encoder, cross-entropy and no
augmentation; the released trainer trains the backbone, uses BCEWithLogits
and applies flips/rotation/ColorJitter.  Defaults follow the paper's frozen
encoder with the released decoder, BCE and augmentation; flags expose
`--segdino-v2-train-backbone` and `--segdino-v2-no-augment`.  Weight decay is
1e-4 per the paper (the released AdamW leaves 0.01).  Input is the official
256 square with ImageNet normalization; logits are bilinearly upsampled to
the exact GT resolution.  The 50-epoch official budget is extended to 800
with validation-Dice selection (the released code already selects validation
Dice).  DINOv3-S weights come from the local `checkpoints/dinov3` file; no
hub/download path is used.  See `src/baseline_zoo/models/segdino_v2/UPSTREAM.md`.

## nnU-Net 2D addendum (2026-10-28)

`nnunet2d` uses the installed stock `nnunetv2` 2D planner, `DefaultPreprocessor`,
`nnUNetTrainer` and `nnUNetPredictor`; no network code is vendored.  The
official recipe is preserved: CE+Dice with deep supervision, SGD Nesterov
1e-2, weight decay 3e-5, PolyLR 0.9, AMP, gradient clip 12, 0.5-overlap
Gaussian sliding-window prediction with mirroring TTA, and EMA pseudo-Dice
checkpoint selection.  `Dataset502_Curvilinear3Shot` stages the shared 24
cases with a fixed 18/6 fold; plans, normalization statistics and
`splits_final.json` are fitted on the 18 train cases only (`--no_pp`), and
the 6 validation cases are preprocessed after planning.  The official
1000-epoch default is replaced by the shared 800-epoch budget; the trainer
still reports its patch-based pseudo-Dice for checkpoint selection and the
shared evaluator computes Dice/IoU/clDice/HD95 on 0/1 masks exported at the
exact GT resolution.  Planner-controlled `--resume`, batch/accumulation,
gradient clip, selection and worker overrides are rejected.  No pretrained
network is used.  See `src/baseline_zoo/models/nnunet2d/UPSTREAM.md`.

## FlowSDF addendum (2026-10-28)

`flowsdf` is the IJCV 2025 image-guided conditional flow-matching segmenter
(`3rd/FlowSDF`, MIT).  The vendored `unet.py`/`rrdb.py`/`nn.py` reproduce the
official velocity UNet exactly: 160,607,745 parameters, 980/980 state-dict
keys and bit-identical outputs after loading the same weights (verified
against `3rd/FlowSDF/models/unet_segdiff.py` on a fixed 64x64 input).  The
official velocity target, Adam lr 1e-4, inf-norm clipping 1.0, EMA update
and `sdf <= 3e-2` rule are preserved.  The released repository depends on an
empty `SimulationHelper` submodule, TensorBoard and `torchdiffeq`; the port
removes all three (the shared runtime owns checkpoints/logging and a local
Euler loop replaces `odeint`).  SDFs are computed on the fly from the GT masks
with SciPy's exact EDT, clipped at delta=15 and divided by delta; the
companion `precompute_sdf.ipynb` is not vendored.  The official loader applies
Vahadane colour normalization and a 500x500 resize outside the released
code; this port uses the raw protocol images.  Final prediction uses
overlapping 128x128 tiles (stride 64) averaged in SDF space, following the
MoNuSeg patch-averaging ablation; per-epoch validation resizes the whole
image to 128 (the official GlaS path) with K=1 so the shared every-epoch
validation is affordable, while the one-time test prediction averages K=4
samples (MMSE).  Checkpoints are `{'model', 'ema'}` so EMA weights are used
for validation/prediction and raw weights resume training.  EMA decay is
0.999 (released config; the paper says 0.9999); batch 4 follows the released
config (the paper states 16); the official 30k-40k epochs are replaced by
the shared 800-epoch budget with validation-Dice selection (official selects
training loss).  No pretrained weights are required.  See
`src/baseline_zoo/models/flowsdf/UPSTREAM.md`.

## A2SAM addendum (2026-10-28)

`a2sam` is the Medical Image Analysis 2026 informativeness-driven SAM
adaptation (MIT for A2SAM; the vendored SAM ViT-B modules are Apache-2.0).
The released repository supplies SAM ViT-B, a small detached
uncertainty/logit fusion network and the CPS selection code, but **not** the
paper's probabilistic U-Net prior or the topological structural prompt
generator, and it expects precomputed `uncertainty`/`logit` `.npy` files.
This port therefore reimplements those missing pieces: a conditional
probabilistic U-Net trained for 100 epochs on the 18 images (Adam 1e-3,
BCE+Dice+1e-5 KL, GroupNorm for batch 1; K=5 prior samples → mean likelihood
and 4×variance uncertainty), skeleton-graph
critical cells with geodesic persistence pruning, a learnable branch gate
with Gaussian rasterization, and a Conv-BN-LeakyReLU fusion of the
structural/likelihood/uncertainty maps.  The prompt is not detached.  The
SAM ViT-B weights load strictly from the local
`checkpoints/sam_vit_b_01ec64.pth` (177 image-encoder, 17 prompt-encoder and
120 mask-decoder tensors).  CPS follows the release: accumulated squared
gradient influence normalized with parameter magnitude, alpha 0.4, top 3.5%
elements trained, norms/position embeddings/IoU head frozen; statistics use
the 18 training images rather than the released validation loader to avoid
validation-label leakage.  The smoke run kept 3.48% of 93.6M SAM elements.
The official 1024-square ImageNet-normalized input, horizontal flip only,
AdamW 2e-4/0.01/cosine to 1e-7, batch 1 and BCE+IoU+0.7 thickness-aware
loss are preserved.  Documented deviations: 100 prior + 700 A2SAM epochs
instead of the official 50; validation-Dice selection; prior maps at 512;
the graph-based DMT/persistence approximation (the paper's exact cell
decomposition is not in the release and the full text is not local); the
thickness term divides by the foreground pixel count rather than all pixels;
and no official A2SAM weights.  The GPU smoke trained both stages, saved the
whole prior+SAM+SPG checkpoint, and exported 16 DRIVE masks at 565×584 with
0/255 values.  See `src/baseline_zoo/models/a2sam/UPSTREAM.md`.

## TPSegformer addendum (2026-10-28)

`tpsegformer` is the CVPR 2026 Hilbert-curve topology-preserving Swin-T
segmenter (the upstream repository has no LICENSE file).  The release is an
MMSegmentation overlay and is not runnable as published: `TPHead.forward`
calls the non-existent `self.cls_seg` (BaseDecodeHead defines `conv_seg`),
the auxiliary FCN head outputs 19 classes while the main head has five, and
`lccm.py` imports the external `hilbertcurve` package plus the absolute
module `hbtrans`.  This port vendors SEM/HLFF/LCCM into pure PyTorch, defines
an explicit 5→1 `conv_seg`, fixes the auxiliary head to one binary channel,
and implements the Hilbert traversal locally; the LCCM math (channel max,
1×1 QKV, curve flattening, Q^T K softmax over queries, V weighting, inverse
curve, `x * sigmoid(attention)`) is unchanged.  Hilbert and row-first
traversals produce identical outputs with shared weights (≤1e-6), as
expected from the attention's permutation equivariance.  The model uses
timm Swin-T (depths 2/2/6/2, heads 3/6/12/24, window 7, drop path 0.3) and
the release's 512 input statistics.  The main CE becomes binary BCE; the
auxiliary Dice 0.5 + Sobel edge-aware 1.0, five-conv FCN head, head lr ×10
and zero weight decay for norm/position parameters are preserved.  No local
Swin-T ImageNet checkpoint is bundled, so training is from scratch unless
`--tpsegformer-backbone-ckpt` is supplied.  Documented deviations: 512-square
resize + light photometric jitter instead of random resize/crop/pad;
per-epoch 5-warmup+poly instead of the per-iteration 1500/160k schedule;
shared 800 epochs and validation-Dice selection; no TTA or Dacl10k sliding
window.  GPU smoke: 4 train / 1 val images, one epoch in 2 s, 34.6M params,
checkpoint saved; prediction exported 16 DRIVE masks in 2 s at 565×584 with
0/255 values.  See `src/baseline_zoo/models/tpsegformer/UPSTREAM.md`.

## S2M-Net addendum (2026-10-28)

`s2mnet` is the ICML 2026 Spectral-Spatial Mixing network.  The official
implementation is TensorFlow/Keras (no LICENSE file); this port translates
the architecture and Morphology-Aware Adaptive Loss into PyTorch without
TensorFlow.  The translated model keeps the 16-channel stem, five strided
stages (24/32/64/80/128), MRF-SE (1×1 expansion ×6, parallel depthwise
3/5/7, concat/fuse, SE-16, projection, residual), SSTM (per-channel FFT with
a learnable KxK filter initialized around 0.25 normalized frequency, K =
min(32, H) per stage, spectral-only stages 1–2, gated dense projection on
stages 3–5, concat/1×1/LayerNorm fusion, residual), the BFP decoder
(boundary map, boundary-attended refinement and soft region/boundary
routing) and the 2× nearest head.  The PyTorch model has 4,768,925
parameters, matching the paper's ~4.7M.  Keras defaults that affect
behavior are matched (Conv2D biases, ELU alpha 1, BatchNorm eps 1e-3 /
momentum 0.01, elementwise dropout, channel-axis LayerNorm and sigmoid
probabilities).  MAL reproduces the five components, morphology modulation
and learnable weights initialized at (1,1,1,.5,.5); the adapter clamps them
to [0.1,10] after every optimizer step.  The official retinal
CLAHE/green/FOV/256-patch pipeline is not applied to the mixed protocol;
training uses RGB [0,1] 352 crops with flips/rot90/light colour jitter and
inference uses 352/176 overlapping tiles with probability averaging.  The
official effective batch 32 and 150 epochs become batch 8 and the shared
800 epochs with validation-Dice selection; the per-iteration cosine +
10-warmup schedule is stepped per epoch.  No official weights are provided.
GPU smoke: 4 train / 1 val at 352, one epoch in 1 s, checkpoint saved;
prediction exported 16 DRIVE masks in 2 s at 565×584 with 0/255 values.  See
`src/baseline_zoo/models/s2mnet/UPSTREAM.md`.

## CS-UNet addendum (2026-10-28)

`csunet` is the ICASSP 2026 Curvilinear Spectral U-Net.  The official code is
a TensorFlow/Keras notebook (no LICENSE file); this port is a PyTorch
rewrite with the notebook's 32-filter U-Net (32/64/128/256, 512×2
bottleneck), its `DynamicSnakeConv2D`, and its `fsf_block` GFR module.  Two
release caveats are explicit: the notebook's “dynamic snake” is a
placeholder (a plain 3×3 conv with GroupNorm(16)+ReLU+Snake, no deformable
offsets), and the paper describes a PACB with batch normalization while the
code uses GroupNorm+ReLU; the port follows the code and documents both.  GFR
is reproduced exactly: concat [decoder, skip], global-max channel squeeze,
reduction 8, degree-5 Gegenbauer activations with learnable softplus α and
sigmoid-bounded inputs, 1×1 compression to half channels, global-max sigmoid
gate on the skip and `skip*gate + decoder`.  Dice loss, SGD 0.01/momentum
0.9, no clipping, RGB /255, 512-square inputs and no augmentation are kept.
The official batch 32 gives `steps_per_epoch=0` on the 18-image split, so the
port uses batch 4; the 100-epoch/val-loss-early-stopping recipe becomes the
shared 800 epochs with validation-Dice selection, and the constant
LR−1e−6 per epoch is stepped per epoch.  The PyTorch model has 7,240,385
parameters.  GPU smoke: 4 train / 1 val at 512, one epoch in 1 s, val Dice
0.187, checkpoint saved; prediction exported 16 DRIVE masks in 1 s at
565×584 with non-empty 0/255 values.  See
`src/baseline_zoo/models/csunet/UPSTREAM.md`.

## LIOT-IterNet addendum (2026-10-28)

`liot` is the IEEE TIP 2022 local-intensity-order representation plus the
release's IterNet (32 base channels, one U-Net and three mini U-Nets,
13,604,360 parameters).  The 4-channel LIOT transform is a vectorised NumPy
translation of `distance_weight_binary_pattern_faster`; it was checked
bit-for-bit against the release's numba loop on random uint8 images and a
DRIVE image (max absolute difference 0).  The adapter trains 128×128 random
crops with CE and Adam 1e-3, batch 8 × 4 accumulation (official effective
batch 32), the release repeat=30 replacement sampler (540 crops/epoch), and
the shared 800-epoch/validation-Dice budget; the optional
post-150-epoch topological loss is omitted because it imports a hard-coded
`/home/sty/...` PersistencePython module and is not part of the LIOT
contribution.  LIOT assumes dark curvilinear structures; the protocol's
polarity was measured from training/support masks and bright-foreground
datasets (CHASEDB1, CORN, DRIVE, FIVES, HRF, LES-AV, OCTA-500, STARE,
DeepGlobe, ROAD, TopoMortar, WIRE) are inverted before LIOT at training and
inference.  Inference is a whole-image forward pass at threshold 0.5 with
exact GT-size export, instead of the release's 128/8 overlap and F1-threshold
sweep.  No official weights are used.  CPU tests cover transform parity,
polarity, model forward/backward, batch preparation, scheduler and
prediction shape.  GPU smoke: 2 train / 1 val images, one epoch in 1 s,
train loss 0.9640, val Dice 0.1596, checkpoint saved; prediction exported
20 DRIVE masks in 2 s at 565×584 with 0/255 values.  See
`src/baseline_zoo/models/liot/UPSTREAM.md`.

## DconnNet addendum (2026-10-28)

`dconnnet` is the CVPR 2023 directional-connectivity segmenter (28,382,912
parameters at `num_class=1`).  The port keeps the official CHASE recipe:
ResNet-34 backbone, 960×960 BGR inputs, `[0,1]*3.2-1.6` normalization,
8-direction connectivity head, `BCE + connectivity BCE + edge + Dice`
(auxiliary map weighted 0.3 during training), Adam 3.8e-3, polynomial LR
0.9 and batch 4.  It uses the shared 800-epoch/validation-Dice budget and
fp32 instead of apex.  Two release defects are fixed: `connectivity_matrix`
no longer squeezes away the batch dimension at batch size 1, and the
edge-loss denominator is clamped.  The loss/voting helpers are device-aware;
`ResNet.forward`'s unused `conv1_1` typo is fixed and all network-download
paths are removed.  The optional ResNet-34 ImageNet state dict is loaded
strictly from `checkpoints/resnet34-333f7ec4.pth`; when absent, training is
from random initialization and marked as a deviation.  Inference follows
sigmoid → 0.5 → reshape → `Bilateral_voting` → max, then nearest export at
the exact GT size.  The upstream repository has no LICENSE file.  CPU tests
cover connectivity targets/voting, batch-1 loss, model forward/backward,
adapter batch/prediction and polynomial LR.  GPU smoke: 2 train / 1 val
images, one epoch at 320 in 1 s, train loss 4.0134, val Dice 0.1036,
checkpoint saved; a 960 batch-4 forward/backward peaks at 6.72 GiB, so the
official batch fits the 24 GiB target GPUs.  Prediction exported 20 DRIVE
masks in 1 s at 565×584 with 0/255 values.  The local ResNet-34 loader was
also checked with a synthetic 218-key state dict.  See
`src/baseline_zoo/models/dconnnet/UPSTREAM.md`.
