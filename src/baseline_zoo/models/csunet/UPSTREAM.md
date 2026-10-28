# CS-UNet upstream record

## Sources

| Component | Upstream | Version |
|---|---|---|
| Paper | "Curvilinear Spectral U-Net: A Framework for Structure-Aware Road Extraction from Very High Resolution Imagery", ICASSP 2026 | 2026 |
| Code | `Jayku88/CSUNet` (`3rd/CSUNet/CSUNet.ipynb`, TensorFlow/Keras notebook) | default branch |
| Paper parsing | `parsing/[2026 icassp] Curvilinear Spectral U-Net ...` | local |

License: the upstream repository has **no LICENSE file**.  The port is a
PyTorch translation of the notebook; verify redistribution rights before
publishing.

## What is translated

- `layers.py`: Gegenbauer recurrence and `GegenbauerNeuralBlock` (learnable
  softplus alpha, sigmoid-bounded input, degree 5), fixed-alpha Snake
  activation, `ConvGNReLU` (3x3 conv, GroupNorm(16), ReLU),
  `PeriodicConvBlock` (the notebook's `DynamicSnakeConv2D`), `SENet` (global
  max squeeze, reduction 8, Dense -> ReLU -> Gegenbauer -> Dense -> sigmoid
  -> Gegenbauer, channel multiply) and `FSFBlock`
  (`skip * sigmoid(globalmax(compress(SE([decoder, skip])))) + decoder`).
- `model.py`: 32-filter U-Net: encoder 32/64/128/256, 512x2 bottleneck,
  decoder stages 256/128/64/32 with ConvTranspose -> FSF -> Conv+ReLU ->
  PeriodicConvBlock, and a 1x1 sigmoid output.
- `loss.py`: segmentation_models-style Dice with smooth 1.0 and per-image
  mean.

## Release caveats

The notebook's `DynamicSnakeConv2D` is explicitly a **placeholder**:

```python
# Placeholder deformable conv: simply apply conv for now
deformed = x
```

so there are no deformable offsets or dynamic snake kernels.  The paper
describes a PACB with a 2D conv + batch normalization + Snake activation,
while the notebook uses Conv (no bias) + **GroupNorm(16)** + **ReLU** +
Snake.  This port follows the released notebook (the plan's specification)
and documents both mismatches.  The paper's GFR is called `fsf_block` in the
code; the port implements the code's exact gating direction
(`skip * gate + decoder`).

## Official recipe kept

- 512x512 RGB, rescale 1/255 only (no ImageNet normalization), no
  augmentation.
- Dice loss, SGD lr 0.01, momentum 0.9, no gradient clipping, 100 epochs,
  constant learning rate minus 1e-6 per epoch, model checkpoint on validation
  loss with patience-10 early stopping.

## Protocol adaptations / assumptions

- The official batch 32 cannot run on the 18-image split
  (`steps_per_epoch = 18 // 32 = 0`); this port uses batch 4.  The official
  effective batch is thereby reduced and documented.
- The shared 800-epoch budget and validation-Dice selection replace the
  official 100 epochs, val-loss early stopping and Keras checkpointing.
- The LR decrement is applied per epoch (shared scheduler semantics).
- Predictions are bilinearly resized from the 512 probability map to the
  exact GT resolution and thresholded at 0.5.
- No official weights; random initialization.
- The paper targets road extraction; this is the curvilinear-structure task
  adaptation.

## Verification

- CPU tests: Gegenbauer values/alpha positivity, Snake/PACB/FSF shapes,
  model forward/backward and 7,240,385 parameters, adapter data/config,
  SGD momentum/constant-decay schedule, prediction shape.
- GPU smoke: 4 train / 1 val at 512, one epoch in 1 s, val Dice 0.187,
  checkpoint saved.
- Prediction smoke: 16 DRIVE masks exported in 1 s at 565x584 with 0/255
  values (non-empty).
