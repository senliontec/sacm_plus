# Self-contained baseline zoo

The goal of this layout is to make `baselines/` disposable.  Every model
that is part of the paper reproduction has its runtime code under
`src/baseline_zoo/models/<model>/`, and its official training/inference logic
under `src/baseline_zoo/adapters/<model>.py`.  Deleting `baselines/` must not
change any import, checkpoint or evaluation path.

## Layout

```text
src/baseline_zoo/
  data.py                  # protocol samples + original-resolution masks
  runtime.py               # shared train/val/select/log/W&B/predict loop
  registry.py protocols.py report.py __main__.py
  run_one.py               # python -m baseline_zoo.run_one --model ...
  models/<model>/          # self-contained official code + LICENSE/UPSTREAM
  adapters/<model>.py      # official preprocessing/loss/optimizer/inference
baselines/                 # development reference snapshot; deletable
configs/models/*.yaml      # model metadata + official defaults
```

`UPSTREAM.md` in every model directory records the upstream URL, the files
copied, the license status, and every modification.  Adapters expose the
model-specific deviations in their `deviations` tuple; the shared runtime
prints them at the start of training and stores them in every checkpoint.

## Paper baselines

| Model | Official logic preserved | Main documented deviations |
|---|---|---|
| U-Net | milesial architecture, BCE+Dice | fixed 512 input, 800-epoch shared few-shot budget, flips/rot90, val Dice |
| CS2Net | centre-square crop→512, ±40° rotation, ImageEnhance, MSE+Dice(1e-4), Adam 1e-4/5e-4, poly LR, 1000 epochs, Otsu | val Dice instead of the unreachable/test-leaking official rule; GT-size export |
| BCUNet | BGR [0,255]/INTER_AREA, 3-way CE, Adam 1.5e-4, 50 epochs, no augmentation | batch 1×2 accumulation on 11 GiB; val Dice instead of test IOU; repaired `OutConv` |
| MaskVSC | Fundus UNet, masked-vessel corruption schedule, BCE+Connect_Loss, Adam 2e-4, 200 epochs | val Dice instead of test Dice; GT-size export |
| nnWNet | official W-Net 2D nnU-Net trainer, planner, SGD/AMP/deep supervision, sliding-window+TTA | pooled 18/6 one-fold split; planning sees the 6 val cases; val EMA pseudo-Dice |
| SAM-Med2D | 256 BGR/SAM-stat preprocessing, per-block Adapter, GT box/point + 8 refinement passes, 20*Focal+Dice+IoU, Adam 1e-4, 15 epochs, multimask max-IoU/0.5 | vanilla SAM ViT-L because the official SAM-Med2D checkpoint is absent; val Dice; ≥24 GiB for official batch |
| SAM-OCTA | LoRA rank 4, 1024 longest-side+pad, SAM normalization, p=0.3 augmentations, GT per-component point prompts, 0.8*Dice+0.2*clDice, warmup/0.98 decay, 0.8 threshold | ViT-L base; RGB/grayscale-replicated input; non-square pad; val Dice |
| CWSAM | frequency prompt generator + per-block adapters + mask decoder, ImageNet normalization, no augmentation, AdamW 2e-4/cosine 1e-7, 200 epochs | binary 2-class port; ViT-L (paper backbone); val Dice; activation checkpointing |
| SAM-Adapter | FFT high-pass prompt generator injected into every ViT block, empty sparse + `no_mask_embed` prompt-free decoder, ImageNet normalization, horizontal flip p=0.5, BCE+IoU, AdamW 2e-4/cosine 1e-7, 20 epochs, frozen encoder except prompt generator | retrained from SAM ViT-L (authors' Google-Drive weights unused); val Dice instead of COD metrics; GT-size export; batch 2 official, use `--batch-size 1 --accum-steps 2 --grad-checkpointing` on ≤11 GiB |
| TopoLoRA-SAM | official ViT-B, LoRA r=8 on ViT MLP layers, depthwise-separable conv adapter, frozen SAM + trainable mask decoder, empty-prompt decode, 384 input, BCE+Dice+0.5 clDice, AdamW 1e-4/1e-4/cosine, 50 epochs, batch 1×4, grad clip 1.0 | no mixed precision; shared 18/6 split instead of Seglab's HF split; val Dice; GT-size export; no official trained weights (starts from SAM ViT-B) |
| SAMed | official ViT-B, rank-4 q/v LoRA, trainable prompt encoder + mask decoder, empty-prompt decode, 512 input, binary 2-class head; class-weighted CE (bg 0.1/fg 1.0) + foreground Dice + soft-clDice (0.1) on 128×128, plus a full-resolution aux CE+Dice (0.5) and zero-init residual refinement head; inverse-sqrt foreground oversampling; AdamW 1e-3/0.1/poly (paper 5e-3 was batch 24), grad clip 1.0, 800 epochs, batch 4 | task adaptation for 0.17%-8% curvilinear foreground; official batch 24/epoch-159 selection adapted to the 18-image split; val Dice; argmax+nearest GT export; optional `--input-size 1024` high-res; released 8-class LoRA weights unused |
| JTFN | official DRIVE config: base64 5-stage backbone, three mask/boundary refinements, 256 crop, batch 2, Adam 1e-3/5e-4, mask BCE + boundary BCE over four scales, StepLR 500/0.9, 1000 epochs | one shared model instead of per-dataset DRIVE/roads/Crack500 configs; val Dice instead of test-split F1; per-epoch StepLR approximation; simplified augmentation; GT-size export; no official weights |
| DSCNet | official 2D DSCNet: dynamic-snake U-Net, 16 basic channels, kernel 9, grayscale 224 crop, batch 1, soft BCE, AdamW 1e-4 betas (0.9,0.95), 400 epochs, constant LR, 224/112 patch inference | NIfTI/MONAI/SimpleITK pipeline replaced by PIL + simple augmentation; train-split mean/std; val Dice; finite quartic blending kernel; GT-size export; no official weights |
| MedCAGD | official PVTv2-B2 encoder, context-aware gated decoder, normalized deep supervision (1.0/0.4/0.3/0.2), edge BCE; binary head, AdamW 1e-4/1e-4, >300 epochs; paper uses 224 full images and 256/128 patches for DRIVE/FIVES; final model excludes the code's raw-image Stage 0 | shared 18-image 3-shot split; default task adaptation uses 256/128 patches for every dataset because all protocol domains are thin structures (`--medcagd-input-policy paper` restores 224 full images outside DRIVE/FIVES); Stage 0 disabled by default; batch 8 and 800 epochs instead of paper batch 16; Albumentations/CutMix replaced by seeded crops/flips/rot90; cosine warm-restart; no official weights |
| BLO-SAM | official decoder-only LoRA SAM ViT-B (image encoder frozen), no-mask prompt embedding + mask decoder, rank 4, 256 input, binary background+foreground head, AdamW 5e-3/0.1 + prompt optimizer 5e-3/0.1, effective per-step poly LR, 100 epochs; fixed disjoint D1/D2 halves and D2-Dice selection | fixed seeded 9/9 D1/D2 split on the 18 images, stratified across the six pooled domains; cycles all D2 batches (released loop rereads only the first); flips/rot90; no official BLO-SAM weights; starts from SAM ViT-B |
| PerSAM | training-free PerSAM (Personalize-SAM) with SAM ViT-H: support mask → object embedding, target-guided points, two point/mask/box refinement stages; no target GT or prompts | official method is one-shot; 3-shot support is majority-voted; support donor for unseen/novel domains; training-free metadata checkpoint; original-resolution output |
| Matcher | training-free DINOv2 ViT-L/14 + SAM ViT-H patch matching; support-based, native n-shot; 518 square inputs, modified SAM mask generator, official filters | 518 output upsampled to GT; shared 3-shot supports / donor domains; support-feature cache; DINOv2 weights required; training-free metadata checkpoint |
| KnowSAM | official semi-supervised Trainer/weak-strong transforms/two-stream sampler/SGDL validation | 3+3 batch; source-domain unlabeled pool; ViT-B as upstream; GT-size export |
| SegDINO | official DPT decoder, 256/ImageNet, BCEWithLogits, AdamW 1e-4/1e-4, 50 epochs, batch 4, no augmentation | timm DINOv3 instead of local hub repo; val Dice; GT-size export |
| SACM (official) | zero-prompt ViT-L + adapters, AdamW 3e-4/0.01, cosine 1e-6, BCE 0.7+Dice 0.3, 60 epochs, grad clip 1.0 | val Dice; GT-size export; activation checkpointing |

The shared 3-shot project budget in `configs/models/` is now **800 epochs**
for every baseline whose previous budget was 200 or 400, including DSCNet's
official 400-epoch recipe. The epoch numbers in the table above remain the
upstream recipe values where they describe official logic. Exceptions to
the 800-epoch project budget are CS2Net/JTFN (1000), BLO-SAM (official
100), KnowSAM (50,000 iterations), and training-free Matcher/PerSAM (0).
SACM-X is the proposed method, not a baseline, and is unchanged.

`sam_adapter` is a prompt-free main-table baseline: the official FFT
high-pass prompt generator is injected into every ViT block, and the mask
decoder receives an empty sparse prompt plus `no_mask_embed`.  It is
retrained from SAM ViT-L because the authors' trained weights are not in the
released tree.

`topolora` is a prompt-free main-table baseline from Seglab: official
SAM ViT-B with LoRA on the ViT MLP layers, a depthwise-separable conv
adapter, a trainable mask decoder and the empty-prompt decode.  The paper
table records its ViT-B backbone; the port preserves the official 384-input,
50-epoch, batch-1×4 recipe.

`samed` is a prompt-free, oracle-free SAMed-inspired baseline: official
ViT-B with rank-4 q/v LoRA, a trainable prompt encoder and mask decoder.
It is a fully supervised per-dataset segmenter (not a few-shot matcher), so
the shared 18-image train split is used for a binary fine-tune.  The binary
objective is adapted to the curvilinear foreground: class-weighted CE
(background 0.1/foreground 1.0), foreground-only Dice + soft-clDice
(weight 0.1) on the 128×128 logits, and a full-resolution auxiliary CE+Dice
term (weight 0.5).  A zero-initialized residual refinement head uses the
input image to correct the bilinearly upsampled logits; it is a task
adaptation for thin structures and can be disabled with
`--samed-high-res-head 0`.  Low-foreground images are oversampled by
inverse sqrt foreground ratio.  The batch-4 default LR is 1e-3 (paper 5e-3
was for batch 24), with gradient clipping 1.0; evaluation uses argmax at the
input resolution + nearest resize.  `--input-size 1024` is available for a
further high-resolution run, but the default 512 keeps the shared budget.

`jtfn` is a non-SAM, from-scratch curvilinear baseline: the official DRIVE
base64 configuration with mask/boundary multi-task refinement and boundary
targets generated from the GT masks.  It uses the shared 18-image split and
is reported in the non-SAM group.

`dscnet` is the non-SAM Dynamic Snake Convolution U-Net: official 2D model
and soft-BCE recipe, with a dependency-free PIL training/patch-inference
port.  It is also reported in the non-SAM group.

`persam` is a **support-based** SAM baseline, not prompt-free: one support
image/mask produces target-guided point/box prompts for a frozen SAM ViT-H.
The official method is one-shot; the 3-shot protocol runs the three supports
independently and majority-votes.  It is training-free and writes a
metadata-only checkpoint.

`matcher` is also **support-based** and training-free: DINOv2 ViT-L patch
matching between the support masks and target produces SAM point prompts,
and the modified SAM mask generator merges candidate masks.  It natively
accepts multiple supports, so the three project shots are passed directly;
it writes a metadata-only checkpoint.  Its DINOv2 ViT-L checkpoint is not
bundled.

Two self-contained probes are separate from the original paper-ten audit:
`sam_vit_l_unet` is self-contained (frozen SAM ViT-L + decoder), and
`sam2_unet` is a self-contained appendix baseline (frozen SAM2 Hiera-L +
prompt-learn adapters + U-Net decoder).  `sam2_unet` requires
`checkpoints/sam2_hiera_large.pt` for real training; `--random-init` is a
code-smoke-only option.

`ucs` is an **oracle-prompted reference baseline**: the released code derives
boxes/points from the GT mask and samples refinement clicks from
prediction-vs-GT errors.  It is not prompt-free.  The port keeps the official
20-epoch, batch-1, `iter_point=8` recipe and selects checkpoints by training
Dice; it is not part of the prompt-free main table.

## Assets

| Asset | Status / action |
|---|---|
| SAM ViT-L | `checkpoints/sam_vit_l_0b3195.pth` (present) |
| SAM ViT-B | `checkpoints/sam_vit_b_01ec64.pth` (present; KnowSAM, TopoLoRA-SAM) |
| DINOv2 ViT-L/14 | missing; required for `matcher`; `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth` |
| SAM ViT-H | `checkpoints/sam_vit_h_4b8939.pth` (PerSAM/Matcher; 2.4 GiB) |
| SAM2 Hiera-L | missing; required for real `sam2_unet` training (`--random-init` is smoke-only); `https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt` |
| DINOv3 | auto-detected under `checkpoints/`; otherwise set `DINO_CKPT` / `--dino-ckpt` or `HF_TOKEN` |
| nnU-Net | `nnunetv2==2.8.1` (installed via pip) |
| KnowSAM metrics | `medpy==0.4.0`, `hausdorff` |
| Matcher EMD | `POT` (`pip install POT`; already in the `topology` extra) |
| SAM-Med2D/SAM-OCTA/KnowSAM transforms | `albumentations==1.3.1` (official API) |

## Commands

The unified entry points are `python -m baseline_zoo` and
`python -m baseline_zoo.run_one`.  Shell wrappers export `PYTHONPATH=src`.

```bash
python -m baseline_zoo list-models
python -m baseline_zoo doctor
python -m baseline_zoo plan  --model all --seeds all --gpus 0,1
python -m baseline_zoo train --model all --seeds all --gpus 0,1,2,3
python -m baseline_zoo eval  --model all --seeds all --gpus 0,1,2,3
python -m baseline_zoo report --model all --seeds all
```

Per-model smoke example:

```bash
python -m baseline_zoo.run_one --model unet --stage train \
  --data-root data/sacm_3shot_seed0 --workdir /tmp/unet \
  --epochs 1 --limit-train 4 --limit-val 2 --device cuda:0
```

`--limit-train/--limit-val` are debug-only and must not be used for reported
results.  SAM-Med2D official training needs a ≥24 GiB GPU due to its ~646M
trainable Adapter parameters; prediction fits in 11 GiB.

SAM2-UNet code-only smoke (no SAM2 weights; slow but runs on CPU):

```bash
python -m baseline_zoo.run_one --model sam2_unet --stage train \
  --data-root data/sacm_3shot_seed0 --workdir /tmp/sam2_unet \
  --random-init --epochs 1 --limit-train 1 --limit-val 1 --device cpu
```

For real training, put `sam2_hiera_large.pt` under `checkpoints/` first.

SAM-Adapter uses the official ViT-L, so its official batch 2 needs more than
11 GiB.  A batch-1 smoke with block checkpointing fits a 10.6 GiB card:

```bash
python -m baseline_zoo.run_one --model sam_adapter --stage train \
  --data-root data/sacm_3shot_seed0 --workdir /tmp/sam_adapter \
  --epochs 1 --limit-train 1 --limit-val 1 \
  --batch-size 1 --accum-steps 2 --grad-checkpointing --device cuda:0
```

TopoLoRA-SAM uses the official ViT-B and 384 input; its official batch is
already 1 with four accumulation steps:

```bash
python -m baseline_zoo.run_one --model topolora --stage train \
  --data-root data/sacm_3shot_seed0 --workdir /tmp/topolora \
  --epochs 1 --limit-train 4 --limit-val 1 \
  --accum-steps 4 --grad-clip 1.0 --device cuda:0
```

## Verification

`tests/test_self_contained.py` fails if any file under `src/` imports
`baselines` or inserts it into `sys.path`, and imports every registered
adapter from a temporary working directory.  The shared test suite
(`python -m pytest`) must stay green before `baselines/` is removed.
