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
| MedCAGD | official PVTv2-B2 encoder, context-aware gated decoder, normalized deep supervision (1.0/0.4/0.3/0.2), edge BCE; binary head, AdamW 1e-4/1e-4, >300 epochs; paper uses 224 full images and 256/128 patches for DRIVE/FIVES; final model excludes the code's raw-image Stage 0 | shared 18-image 3-shot split; default task adaptation uses 256/128 patches for every dataset because all protocol domains are thin structures (`--medcagd-input-policy paper` restores 224 full images outside DRIVE/FIVES); **soft Dice term on the main output (weight 1.0) because BCE-only collapses to all background on 0.25-3% foreground domains** (`--medcagd-dice-weight 0` restores BCE-only); Stage 0 disabled; batch 8 and 800 epochs; Albumentations/CutMix replaced by seeded crops/flips/rot90; official cosine warm-restart; no official weights. Requires local PVTv2-B2 ImageNet weights at `checkpoints/pvt_v2_b2.in1k.safetensors`; runtime never downloads |
| BLO-SAM | official decoder-only LoRA SAM ViT-B (image encoder frozen), no-mask prompt embedding + mask decoder, rank 4, 256 input, binary background+foreground head, AdamW 5e-3/0.1 + prompt optimizer 5e-3/0.1, effective per-step poly LR, 100 epochs; fixed disjoint D1/D2 halves and D2-Dice selection | fixed seeded 9/9 D1/D2 split on the 18 images, stratified across the six pooled domains; cycles all D2 batches (released loop rereads only the first); flips/rot90; GT export via bilinear foreground-logit upsample (paper reports 64×64 Dice); no official BLO-SAM weights; starts from SAM ViT-B |
| Conv-LoRA | official AutoGluon Conv-LoRA: SAM ViT-H frozen except rank-3 qkv Conv-LoRA with top-1 MoE over 8 multi-scale 3×3 conv experts, constant no-mask prompt, one-token binary decoder, structure loss (weighted BCE+weighted IoU), MoE load-balancing loss (1e-2 per layer, weight 1.0), 1024 bicubic resize, ImageNet normalization, horizontal flip, Adam 1e-4/1e-4, lora_alpha 32, polynomial decay (power 1, no warmup), grad clip 1.0, effective batch 4, 30 epochs | 800-epoch shared budget; mask decoder trained per the paper (released AutoGluon PEFT filter would train only LoRA); IoU head frozen; per-epoch instead of per-step scheduler; no layerwise LR decay/greedy soup; val-Dice selection; no official Conv-LoRA weights; starts from local SAM ViT-H |
| FSSAM | official variable-shot FSSAM (FSSAM5s): DINOv2 pseudo FG/Disc prompts, support memories + query pseudo-memories, iterative memory refinement, support-calibrated memory attention, SAM 2-S memory encoder/memory attention/mask decoder trained, AdamW 1e-3/0, poly power 0.9, Dice loss, 512 square, batch 2×4, 300/75 epochs | support-based 3-shot protocol (seen = own shots, unseen/novel = donor); training episodes use at most 2 non-query supports because each training domain has 3 images; DINOv2-L local instead of official B (paper ablation) with optional `--fssam-dino-size vitb`; fp32 instead of bf16 autocast; per-epoch poly LR; 800-epoch shared budget and val-Dice selection; repaired the released missing `Weighted_GAP` and k-shot SCMA loop; no official FSSAM weights |
| SANSA | official SANSA: frozen SAM 2 Hiera-Large, AdaptFormer adapters (~9.9M) in the last two hierarchical stages (0.3× bottleneck), reference masks + target as a pseudo-video through memory attention, mask prompts, ImageNet-normalized 1024 inputs, AdamW 1e-4/0, cosine, grad clip 1.0, focal+Dice, 5 epochs, batch 32 | support-based 3-shot protocol (seen = own shots, unseen/novel = donor); released engine trains one reference + one target (paper's J=3 multi-target objective is not implemented in the release); 800-epoch shared budget, batch 2 on 24 GiB GPUs, val-Dice selection; focal+Dice per release (paper says BCE+Dice); per-epoch cosine; no official adapters; SAM 2 Hiera-L required |
| RSRM | official training-free RSRM: frozen DINOv3, SAFR layer variance/Fisher re-fusion, ASE robust-threshold support enhancement, HPM global/regional/pixel prototype matching, 480 square ImageNet-normalized, 3-shot supports | support-based protocol (own shots/donor); no training or RSRM weights; official ViT-B/16, local timm ViT-S/16 auto-detected (Meta B state dicts accepted; timm wrapper matches Meta `F_attn`/`F_sum`); fast_pytorch_kmeans replaced by deterministic sklearn k-means++; argmax at 480 exported at GT resolution |
| C2P | official prompt-free universal C2P: ConvNeXtV2-B, 9 geometric + 9 MLLM-distilled semantic tokens, token-image transformer, token-guided dynamic head, geometry-aware 3-view inference consensus, 384 square ImageNet-normalized | zero-shot evaluation of the official 8-medical-dataset checkpoint; does **not** use the shared 18-image split; released repo has no runnable trainer/evaluator, so the network + official TTA are ported; requires `checkpoints/c2p/1.pth`; random-init is smoke-only; no fine-tuning or target labels |
| DKA | official DKA module (down-project, 51×51 + 5×5 depthwise branches, GELU, up-project, residual; zero-init down/up) in all 12 ViT-B/16 blocks; Segmenter linear decoder; 512 ImageNet inputs; head 1e-4 / DKA 1e-3; batch 1; 300 epochs | released repo has DKA + classification only; segmentation uses MMSegmentation Segmenter-B, not released, so the port reconstructs Segmenter-B with torchvision ViT-B/16-ImageNet-1k + linear decoder; no segmentation loss/augmentation/input size stated → CE+foreground Dice, hflip, 512 square; 800 shared epochs, val-Dice selection; no official DKA weights |
| SCRWKV | official SCRWKV Crack model: SFE (AMCM + GBST/DSCD SCIU blocks), CSHF decoder with DySample, 512 cubic resize, 0.5 normalization, no augmentation, 0.25 BCE + 0.75 Dice, AdamW 5e-4/0.01, PolyLR 1e-6, batch 1, 50 epochs | from-scratch training on the shared split; 800-epoch budget and val-Dice selection; mmcv/mmcls build-time deps replaced by local torch equivalents (same architecture); official CUDA bi_wkv kernel vendored, arch configurable (official sm_86); inference uses the official max-normalized 0.2 threshold; no official weights; actual port size ~2.04M vs paper's 1.22M |
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

`convlora` is a prompt-free SAM ViT-H main-table baseline.  Conv-LoRA is
inserted into the fused qkv projection of every ViT block (rank 3,
`lora_alpha` 32 as in AutoGluon's semantic-segmentation preset, top-1 noisy MoE over
eight multi-scale 3×3 convolution experts); the image encoder is otherwise
frozen and the constant no-mask prompt is used.  The paper states that the
mask decoder is fully fine-tuned and its 4.02M trainable-parameter count
(3.51M decoder + 0.51M Conv-LoRA) confirms it; the released AutoGluon PEFT
filter (`.*lora_*`) would freeze the decoder, so this port follows the
paper and keeps the decoder trainable except the IoU head.  The official
AutoGluon pipeline normalizes 1024-square inputs with **ImageNet** statistics
(not SAM's pixel mean/std).  The paper's Adam/1e-4/1e-4 recipe is used with
the semantic preset's polynomial decay (power 1, no warmup), `lora_alpha=32`
and gradient clipping 1.0; the shared runtime steps the schedule per epoch,
and layerwise LR decay/greedy soup are omitted.  The released processor's
`RandomHorizontalFlip(1.0)` is treated as an upstream quirk; the port uses
p=0.5 per the paper.
No official Conv-LoRA weights exist; training starts from the local SAM
ViT-H checkpoint and the 30-epoch official budget is extended to the shared
800-epoch budget.  ViT-H training OOMs on an 11 GiB card (verified with one
image); the 24 GiB target GPUs are required.  The adapter's
`--convlora-model-type vit_b` + `--convlora-checkpoint` selects ViT-B for
fast local smoke tests only.

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

`fssam` is the ICML 2025 SAM 2 few-shot segmenter and is **support-based**
(not prompt-free).  It uses the protocol's support images/masks to build
support memories and DINOv2 pseudo FG/Disc query memories, refines the Disc
memory iteratively (IMR), and calibrates the memory attention with the
support masks (SCMA).  Seen test domains use their own three training shots;
unseen/novel domains use the structurally matched donor set (same rule as
Matcher/PerSAM).  The official trainer optimizes only SAM 2's memory
encoder, memory attention and mask decoder with AdamW 1e-3/0, polynomial
power 0.9 and Dice loss at 512 square with effective batch 8; this port
extends 300/75 epochs to the shared 800-epoch budget and selects by
validation Dice.  Each training domain has only three images, so training
episodes use at most two non-query supports while evaluation uses the
protocol's three supports.  The released code's SCMA path is unusable as
published (missing `Weighted_GAP`, support-mask stack overwritten in the
shot loop); the vendored copy restores the weighted GAP and averages the
per-shot similarities as the paper describes.  The official main model is
SAM 2-S + DINOv2-B; this port defaults to the locally available DINOv2-L
(an official ablation variant) and can use SAM 2-B+/L and DINOv2-B with
explicit flags.  It starts from the official SAM 2/DINOv2 checkpoints; no
FSSAM-trained weights are required.

`sansa` is the NeurIPS 2025 semantically aligned SAM 2 baseline and is also
**support-based**.  Only AdaptFormer adapters (≈9.9M parameters) in the
last two Hiera-Large stages are trained; the rest of SAM 2 stays frozen.
Reference images and masks plus the target form a pseudo-video, and SAM 2's
memory attention propagates the reference concept to the target.  The
released engine trains one reference + one target with focal+Dice (the paper
describes BCE+Dice and a J=3 multi-target objective, but the release does
not implement the latter), AdamW 1e-4/0, cosine and gradient clipping 1.0
(the CLI default is 0.1).  This port extends the official 5 epochs to the
shared 800-epoch budget, uses batch 2 on 24 GiB GPUs instead of the official
effective batch 32, and selects by validation Dice.  Evaluation uses the
model's native k-shot path with the protocol's three references (own shots
for seen domains, donor otherwise).  Inputs are resized to 1024 square and
ImageNet-normalized, matching the released dataset transform (its
`preprocess` never normalizes).  Support prompts are masks only.  Training
starts from the local SAM 2 Hiera-Large checkpoint; no official SANSA
adapters are required.

`rsrm` is a **training-free, support-based** DINOv3 baseline (ECCV 2026).
SAFR selects semantic attention layers per image using variance proportions
and Fisher ratios and re-fuses them; ASE derives robust FG/BG thresholds
from the support predictions and aggregates query information into the
support via masked cross-attention; HPM fuses global, k-means regional and
pixel-level prototype matching.  It uses the protocol's three supports
(own shots for seen domains, donor otherwise), 480-square ImageNet-
normalized inputs, and no target GT/prompts.  The paper uses DINOv3
ViT-B/16; the local `checkpoints/dinov3/pytorch_model.bin` is a timm
ViT-S/16 export, auto-detected and run through timm's own architecture
(official Meta ViT-B/16 state dicts are also accepted).  Regional k-means
uses deterministic scikit-learn k-means++ instead of `fast_pytorch_kmeans`.
Predictions are the official argmax at 480, nearest-resized to the original
GT resolution.  Training writes a metadata-only checkpoint.

`c2p` is the zoo's **prompt-free universal foundation baseline** (ECCV
2026) and is evaluated **zero-shot** with the official HuggingFace
checkpoint trained on eight medical datasets.  It does not consume the
shared 18-image training split and uses no support images, target GT or
prompts; applying it to the protocol's curvilinear and non-medical domains
is a zero-shot generalization test analogous to the paper's unseen-modality
setting.  The released repository has no runnable trainer/evaluator
(`train.py` imports a missing `SRICL`, `evaluate_sem.py` is absent), so the
port vendors the released `C2P` network and reimplements the official
geometry-aware 3-view inference consensus.  The checkpoint contains the
ConvNeXtV2-B backbone and trained geometric/semantic tokens, so no timm
download or MLLM/PubMedBERT text is needed at inference.  Inputs are 384
square with ImageNet normalization; sigmoid probabilities are exported at
the original GT resolution.  The official checkpoint must be placed at
`checkpoints/c2p/1.pth`; `--c2p-random-init` is smoke-only.  Training writes
a metadata-only checkpoint.

`dka` is a non-SAM PEFT segmentation baseline (ICLR 2026).  The released
repository provides the DKA module and classification training, but its
segmentation experiments use Segmenter-B in MMSegmentation, which is not
released.  The port keeps the official DKA module — down-projection to
middle dimension 192, parallel depthwise 51×51 and 5×5 convolutions, GELU,
up-projection, residual, zero-initialized down/up — after all 12 blocks of a
frozen torchvision ViT-B/16, with Segmenter's linear decoder.  The paper's
segmentation recipe provides head lr 1e-4, DKA lr 1e-3, batch 1 and 300
epochs; the port uses the shared 800-epoch budget and validation-Dice
selection.  No segmentation loss, input size, or augmentation is stated, so
the port uses 512-square ImageNet-normalized inputs, horizontal flip only,
and cross-entropy + foreground Dice (the Dice term avoids the all-background
degeneracy on the 0.25-3% foreground protocol domains).  The backbone is
loaded from `checkpoints/vit_b_16_imagenet1k_v1.pth`; no DKA weights exist,
and `--dka-random-init` is smoke-only.

`scrwkv` is the ICML 2026 ultra-compact crack segmenter, a non-SAM
from-scratch baseline and a direct structural competitor for curvilinear
targets.  The port keeps the released SFE (AMCM plus GBST/DSCD SCIU blocks),
the CSHF cross-scale harmonic decoder with DySample upsamplers, and the
official crack recipe: 512-square cubic resize for images and masks, mask
threshold 127, `(x/255 - 0.5) / 0.5` normalization, no augmentation,
`0.25 BCEWithLogits + 0.75 Dice`, AdamW 5e-4/0.01, PolyLR to 1e-6, batch 1.
The 50-epoch official budget is extended to the shared 800 epochs with
validation-Dice selection.  The released code depends on mmcv/mmcls only for
generic containers/norm/padding helpers and a CUDA `bi_wkv` extension; the
port replaces the former with local torch code (same architecture) and
vendors the latter, compiling it at import for the current GPU (official
sm_86 / RTX 3090, override `SCRWKV_CUDA_ARCH`).  Without nvcc the model can
be built but forward fails with a clear error.  Inference follows the
released `test.py`: divide raw logits by the image maximum, threshold at
0.2, then nearest-resize to the original GT.  The port contains ~2.04M
parameters, not the paper's 1.22M; the released Crack config and
CSHF/DySample are kept verbatim, so the difference is in the reported
count/config.  Training starts from random initialization; no SCRWKV weights
are required.

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
| DINOv2 ViT-L/14 | `checkpoints/dinov2_vitl14_pretrain.pth` (present; `matcher`, `fssam`); ViT-B optional for FSSAM's official main config: `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth` |
| SAM ViT-H | `checkpoints/sam_vit_h_4b8939.pth` (PerSAM/Matcher/Conv-LoRA; 2.4 GiB) |
| SAM2 Hiera-L | `checkpoints/sam2_hiera_large.pt` (present; SAM2-UNet and the FSSAM local smoke/ablation config) |
| SAM2 Hiera-S | missing; FSSAM's official default (46M); `https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt` |
| DINOv3 | auto-detected under `checkpoints/`; SegDINO uses DINOv3 through timm; RSRM reads `checkpoints/dinov3/pytorch_model.bin` (present; timm ViT-S/16) or a Meta ViT-B/16 state dict (`https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`) via `--rsrm-dinov3-weights` |
| C2P checkpoint | missing; required for zero-shot `c2p`; HuggingFace `yunyundi/c2p`, file `sem-OL-3dice-v2-mutil-deep_only-norfb-DS-onlyD5-weightdecay-0.3-0.05/1.pth`, place at `checkpoints/c2p/1.pth` |
| ViT-B/16 ImageNet-1k | missing; required for `dka`; `https://download.pytorch.org/models/vit_b_16-c867db91.pth` → `checkpoints/vit_b_16_imagenet1k_v1.pth` |
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
