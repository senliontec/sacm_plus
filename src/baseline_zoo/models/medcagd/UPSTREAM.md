# MedCAGD upstream record

## Source

- Repository: `saadwazir/MedCAGD`
  (https://github.com/saadwazir/MedCAGD)
- Local reference: `3rd/MedCAGD`
- HEAD: `fe2cbfc3656e72e033cfeb2bf4af162ded4656d0` (2026-10-05)
- License: MIT (copyright 2026 Saad Wazir); copied as `LICENSE`.
- Paper: "MedCAGD: Context-Aware Gated Decoder for Robust Medical Image
  Segmentation", ICML 2026. This is not the S2M-Net paper.

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
   validation-Dice checkpoint selection.
3. Inference tiles full-resolution protocol images with overlapping 256
   patches (default stride 128) and averages probabilities. The upstream
   pre-patched test directory, `patch.py`/`unpatch.py` and CuPy evaluator
   are not used.
4. Albumentations/CutMix augmentation is replaced by seeded random crops,
   foreground-biased crops, flips, 90-degree rotations and mild photometric
   jitter. This keeps the baseline dependency-free and reproducible.
5. The upstream xrun scheduler/resume orchestration is replaced by the
   shared runtime and a cosine warm-restart scheduler.
6. No official trained checkpoint is present. Training uses ImageNet
   initialization through timm; prediction builds the same architecture and
   loads the zoo checkpoint.

The code under `3rd/MedCAGD` is not imported or modified at runtime.
