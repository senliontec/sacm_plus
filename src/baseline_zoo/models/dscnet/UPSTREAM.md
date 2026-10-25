# DSCNet 2D upstream record

## Source

- Repository: `YaoleiQi/DSCNet`
  (https://github.com/YaoleiQi/DSCNet)
- HEAD: `0e35b2b6ea748a5e8ce7557ea3e580b9951471d6` (2026-05-30).
- Local reference:
  `3rd/DSCNet/DSCNet_2D_opensource/Code/DRIVE/DSCNet`.
- License: MIT (Copyright (c) 2025 Yaolei Qi), copied as `LICENSE`.

## Copied files

- `S3_DSCNet.py` → `model.py` — `EncoderConv`, `DecoderConv`, `DSCNet`
- `S3_DSConv.py` → `dsconv.py` — `DSConv` and the `DSC` deformation core
- `S3_Loss.py` → `losses.py` — `cross_loss` soft BCE
- `LICENSE`

The NIfTI/MONAI/SimpleITK data loaders, `S1` mean/std and txt generators,
`S2` split generator, `S0_Main.py`, TensorBoard plumbing and the 3D/NPU/Pro
variants are not vendored.

## Modifications

- The DSConv import is package-relative.
- `DSConv.forward` passes the input tensor's device to `DSC` instead of the
  constructor-time device.  The upstream code stored a fixed device, which
  breaks CPU smoke tests and CPU/GPU model moves; the deformation math is
  unchanged.
- The upstream inference code's blending kernel
  `1 / ((i - 112)^4 + (j - 112)^4)` is singular at the patch center (1/0).
  The adapter uses a finite `1 / (1 + d^4)` quartic window, normalizes it,
  and handles the last patch coverage explicitly.
- The initial implementation used `norm.clamp_min(1e-6)`, which suppressed
  probabilities near low-weight image edges; it is now `1e-12`.

## Official recipe preserved by the adapter

- DSCNet 2D dynamic-snake U-Net: four encoder/decoder levels, 16 basic
  channels, kernel size 9, `extend_scope=1`, offsets enabled, `dim=1`.
- Grayscale input, 224×224 random crop, batch 1.
- Loss: official `cross_loss` soft BCE on the sigmoid output.
- Optimizer: AdamW, lr 1e-4, betas (0.9, 0.95), PyTorch default weight
  decay 0.01.  The upstream `ReduceLROnPlateau` is constructed but never
  stepped; the port keeps a constant learning rate.
- 400 epochs.
- Inference: 224×224 sliding patches with stride 112, quartic center
  weighting, average probabilities, crop to the original image, threshold
  0.5.

## Deviations in this port

- The official DRIVE pipeline uses SimpleITK NIfTI I/O, MONAI augmentation
  (`Orientation`, `RandAffine`, `Rand2DElastic`, intensity clamp, noise,
  smoothing) and a test-set mean/std.  The port uses PIL grayscale I/O,
  simple flips/90° rotations/±20° rotations/noise, and a scalar mean/std
  computed from the shared 18-image train split; MONAI and SimpleITK are not
  dependencies.
- The official checkpoint selection predicts the test set from epoch 200
  onward and selects by test Dice; the port selects on the shared 6-image
  validation Dice.  The official last-epoch weights are not used.
- Predictions are exported at original GT resolution rather than NIfTI
  probability maps.
- No official trained weights are present in the repository; training
  starts from scratch.
- No test-time augmentation.
- The 3D, NPU and Pro variants are not used.  The Pro variant is faster but
  changes parameter names/offset-loop bounds and has no training entrypoint;
  the original 2D pair is kept for fidelity.
