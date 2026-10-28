# DKA upstream record

## Sources

| Component | Upstream | Commit |
|---|---|---|
| DKA module / training code | `misswayguy/DKA`, `peft_model/adapter/vit_adapter.py`, `experiments/cnn/adapter_tuning_dka.py` | `70f45a2ecaf7aab3ebbf1b175853988c15219318` |

Paper: "Dual-Kernel Adapter: Expanding Spatial Horizons for
Data-Constrained Medical Image Analysis", ICLR 2026.

License: MIT (repository `LICENSE`).

## What was copied / reconstructed

- `DKAModule`: the active DKA adapter from `vit_adapter.py`:
  `adapter_down` (Linear, no bias), parallel depthwise `conv_large`
  (51×51, padding 25, groups=middle_dim) and `conv_small` (5×5, padding 2),
  element-wise sum, GELU, dropout 0.1, `adapter_up` (Linear, no bias),
  residual.  The released file contains many commented experimental
  variants (dilated kernels, extra convolutions); only the active
  dual-kernel path is ported.
- `SegmenterDKA`: a self-contained Segmenter-B reconstruction, because the
  released repository does not include its segmentation code.  It uses
  torchvision `vit_b_16`, inserts DKA after every transformer block, and
  applies Segmenter's linear decoder (1×1 conv to two classes + bilinear
  upsampling) to the patch grid.
- Position embeddings are bicubically interpolated from 224 to the 512
  input grid; torchvision requires `image_size=224` otherwise.

## Invariants kept from the paper

- DKA in all transformer blocks (the released classification entrypoint
  uses `selected_layers=list(range(0, 12))`).
- Middle dimension 192 for segmentation (16 for classification).
- Head learning rate 1e-4, DKA learning rate 1e-3, Adam, batch size 1.
- Frozen pretrained backbone; only DKA and the task head are trained.
- DKA large kernel 51×51, small kernel 5×5 (the paper's selected
  combination), depthwise, no bias.
- Paper B.2 zero initialization of the down-projection and up-projection
  weights/biases; the released code has this commented out, so the port
  follows the paper (adapter is identity at initialization).

## Deviations / reconstruction

1. The paper's segmentation backbone/decoder is Segmenter-B in
   MMSegmentation.  MMSegmentation and its Segmenter config/checkpoint are
   not in the release.  The port reconstructs a Segmenter-B-style model
   with torchvision ViT-B/16 (ImageNet-1k) and Segmenter's linear decoder.
   The exact pretraining source of MMSeg's Segmenter-B and its mask
   transformer option are not stated in the available material.
2. Segmentation input size, augmentation and loss are not specified.  The
   port uses 512-square stretch resize, ImageNet normalization, horizontal
   flip only, and cross-entropy + foreground Dice.  The Dice term is a task
   adaptation for the 0.25-3% foreground protocol domains (the all-
   background degeneracy observed with BCE-only objectives).
3. The released classification code uses a 1e-5 adapter LR and StepLR; the
   paper states 1e-3 for segmentation.  The port uses the paper's values
   and MMSegmentation's default polynomial decay (power 1.0), stepped per
   epoch by the shared runtime.
4. The official 300-epoch segmentation budget is extended to the shared
   800-epoch budget with validation-Dice selection.  Batch size 1 is kept
   (the paper's segmentation batch).
5. No official DKA weights exist; the backbone is loaded from the local
   torchvision ViT-B/16 ImageNet-1k checkpoint.  `--dka-random-init` is a
   code-smoke option only and must not be used for reported results.
