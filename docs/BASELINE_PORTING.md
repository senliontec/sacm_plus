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
| U-Net | milesial architecture, BCE+Dice | fixed 512 input, 200-epoch few-shot recipe, flips/rot90, val Dice |
| CS2Net | centre-square crop→512, ±40° rotation, ImageEnhance, MSE+Dice(1e-4), Adam 1e-4/5e-4, poly LR, 1000 epochs, Otsu | val Dice instead of the unreachable/test-leaking official rule; GT-size export |
| BCUNet | BGR [0,255]/INTER_AREA, 3-way CE, Adam 1.5e-4, 50 epochs, no augmentation | batch 1×2 accumulation on 11 GiB; val Dice instead of test IOU; repaired `OutConv` |
| MaskVSC | Fundus UNet, masked-vessel corruption schedule, BCE+Connect_Loss, Adam 2e-4, 200 epochs | val Dice instead of test Dice; GT-size export |
| nnWNet | official W-Net 2D nnU-Net trainer, planner, SGD/AMP/deep supervision, sliding-window+TTA | pooled 18/6 one-fold split; planning sees the 6 val cases; val EMA pseudo-Dice |
| SAM-Med2D | 256 BGR/SAM-stat preprocessing, per-block Adapter, GT box/point + 8 refinement passes, 20*Focal+Dice+IoU, Adam 1e-4, 15 epochs, multimask max-IoU/0.5 | vanilla SAM ViT-L because the official SAM-Med2D checkpoint is absent; val Dice; ≥24 GiB for official batch |
| SAM-OCTA | LoRA rank 4, 1024 longest-side+pad, SAM normalization, p=0.3 augmentations, GT per-component point prompts, 0.8*Dice+0.2*clDice, warmup/0.98 decay, 0.8 threshold | ViT-L base; RGB/grayscale-replicated input; non-square pad; val Dice |
| CWSAM | frequency prompt generator + per-block adapters + mask decoder, ImageNet normalization, no augmentation, AdamW 2e-4/cosine 1e-7, 200 epochs | binary 2-class port; ViT-L (paper backbone); val Dice; activation checkpointing |
| KnowSAM | official semi-supervised Trainer/weak-strong transforms/two-stream sampler/SGDL validation | 3+3 batch; source-domain unlabeled pool; ViT-B as upstream; GT-size export |
| SegDINO | official DPT decoder, 256/ImageNet, BCEWithLogits, AdamW 1e-4/1e-4, 50 epochs, batch 4, no augmentation | timm DINOv3 instead of local hub repo; val Dice; GT-size export |
| FoB-SAM | official ResNet-101 background-prompt localizer, contour/Gaussian keypoint prompts, SPR refinement, frozen SAM point-prompt inference, 39001 iterations (13×3000), Adam 1e-4, MultiStepLR 0.98 | 2D GT support episodes instead of 3D/supervoxels; same-domain support for seen, donor support for unseen; ViT-H required; val Dice; GT-size export |
| SACM (official) | zero-prompt ViT-L + adapters, AdamW 3e-4/0.01, cosine 1e-6, BCE 0.7+Dice 0.3, 60 epochs, grad clip 1.0 | val Dice; GT-size export; activation checkpointing |

Two extensions are intentionally separate from the paper ten:
`sam_vit_l_unet` is self-contained (frozen SAM ViT-L + decoder), while
`sam2_unet` is `reference_only` because its SAM2 Hiera checkpoint and legacy
adapter are outside the paper zoo.

## Assets

| Asset | Status / action |
|---|---|
| SAM ViT-L | `checkpoints/sam_vit_l_0b3195.pth` (present) |
| SAM ViT-B | `checkpoints/sam_vit_b_01ec64.pth` (present; KnowSAM) |
| SAM ViT-H | `checkpoints/sam_vit_h_4b8939.pth` (FoB-SAM; 2.4 GiB) |
| ResNet-101 COCO | `checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth` (FoB-SAM) |
| SAM2 Hiera-L | missing; only needed by the reference-only SAM2-UNet |
| DINOv3 | auto-detected under `checkpoints/`; otherwise set `DINO_CKPT` / `--dino-ckpt` or `HF_TOKEN` |
| nnU-Net | `nnunetv2==2.8.1` (installed via pip) |
| KnowSAM metrics | `medpy==0.4.0`, `hausdorff` |
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

FoB-SAM additionally needs the official SAM ViT-H and DeepLabV3 COCO
ResNet-101 checkpoints:

```bash
curl -L -o checkpoints/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
curl -L -o checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth \
  https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth
python -m baseline_zoo.run_one --model fob_sam --stage train \
  --data-root data/sacm_3shot_seed0 --workdir results/paper/fob_sam/seed_0 \
  --epochs 13 --seed 0 --device cuda:0 --use_wandb
```

FoB's official 39001 iterations are represented as 13 epochs × 3000
episodes; `--fob-shots 1` is the official episode size (set `3` for a
3-shot support variant).  Unseen datasets use a structurally matched donor
support set; override with `--fob-support-dataset DRIVE` (or
`CrackTree200`).

## Verification

`tests/test_self_contained.py` fails if any file under `src/` imports
`baselines` or inserts it into `sys.path`, and imports every registered
adapter from a temporary working directory.  The shared test suite
(`python -m pytest`) must stay green before `baselines/` is removed.
