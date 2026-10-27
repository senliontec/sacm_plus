# MedCAGD upstream record

## Source

- Repository: `saadwazir/MedCAGD`
  (https://github.com/saadwazir/MedCAGD)
- Local reference: `3rd/MedCAGD`
- HEAD: `fe2cbfc3656e72e033cfeb2bf4af162ded4656d0` (2026-10-05)
- License: MIT (copyright 2026 Saad Wazir); copied as `LICENSE`.
- Paper: "MedCAGD: Context-Aware Gated Decoder for Efficient Medical Image
  Segmentation", ECCV 2026. This is not the S2M-Net paper.

## What was preserved

- Five-stage encoder-decoder with PVTv2-B2 timm encoder.
- Context-aware gated decoder blocks, ECABlock, non-local refinement and
  the official deep-supervision outputs.
- Binary `BCEWithLogitsLoss` with normalized deep-supervision weights
  `[1.0, 0.4, 0.3, 0.2]`.
- Edge supervision using the upstream dilation/erosion edge band and
  `BCEWithLogitsLoss`.

## Deviations

1. The upstream `model.py` imports the global `cfgs` module for encoder
   name, image size and class count. This copy exposes those as constructor
   arguments; there is no global config dependency.
2. Upstream training uses pre-patched DRIVE 256x256 directories, 8 epochs
   per `xrun` iteration and up to 16 iterations. This port trains on the
   shared 18-image 3-shot split with the project epoch budget and
   validation-Dice checkpoint selection. The paper uses 256x256 crops for
   DRIVE/FIVES and 224x224 full images for its large-object benchmarks
   (CellSeg uses 384/192 patches). All 20 protocol datasets are thin
   curvilinear structures, so the default task adaptation applies the
   DRIVE/FIVES patch recipe to all datasets; `--medcagd-input-policy paper`
   reproduces the paper dataset mapping.
3. The default inference tiles full-resolution protocol images with
   overlapping 256 patches (default stride 128) and averages probabilities.
   In paper mode, DRIVE/FIVES are tiled the same way while other datasets
   are resized to 224x224 and the prediction is mapped back to GT
   resolution. The upstream pre-patched test directory,
   `patch.py`/`unpatch.py` and CuPy evaluator are not used.
4. The paper's final model excludes the upstream raw-image Stage 0 decoder
   path, although the released code builds it. The port defaults to
   `use_stage0=False`; `--medcagd-use-stage0 1` reproduces the released code
   variant.
5. Albumentations/CutMix augmentation is replaced by seeded random crops,
   foreground-biased crops, flips, 90-degree rotations and mild photometric
   jitter. This keeps the baseline dependency-free and reproducible.
6. Upstream reads images with OpenCV (BGR) and scales to [0,1] without
   ImageNet mean/std normalization. This port uses PIL RGB with the same
   [0,1] scaling; the encoder is ImageNet-pretrained on RGB.
7. The upstream xrun scheduler/resume orchestration is replaced by the
   shared runtime and a cosine warm-restart scheduler.
8. No official trained checkpoint is present. Training loads the timm
   ImageNet-1k encoder weights from a local file
   (`checkpoints/pvt_v2_b2.in1k.safetensors` by default, or
   `--medcagd-encoder-weights`). The upstream `pretrained=True` path can
   contact Hugging Face; this port builds with `pretrained=False` and never
   accesses the network. Download `model.safetensors` (101 MB) from
   `https://huggingface.co/timm/pvt_v2_b2.in1k` on a networked machine and
   rename it to `checkpoints/pvt_v2_b2.in1k.safetensors` (the descriptive
   name avoids confusion with other backbones). The loader converts timm's
   classification keys
   (`stages.0.*`, `head.*`) to the features-only wrapper keys
   (`stages_0.*`). Prediction builds the same architecture and loads the
   zoo checkpoint.

The code under `3rd/MedCAGD` is not imported or modified at runtime.
