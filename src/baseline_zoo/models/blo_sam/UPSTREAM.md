# BLO-SAM upstream record

## Source

- Repository: `importZL/BLO-SAM`
  (https://github.com/importZL/BLO-SAM)
- Local reference: `3rd/BLO-SAM`
- HEAD: `edf2b7f35c07fe31f764450853d400de084f6bfa`
- README claims MIT, but the checkout has no `LICENSE` file. Verify the
  license with the authors before redistributing this port.
- Paper: Li Zhang, Youwei Liang, Pengtao Xie,
  "BLO-SAM: Bi-level Optimization Based Finetuning of the Segment Anything
  Model for Overfitting-Preventing Semantic Segmentation", 2024.

## What was preserved

- SAM ViT-B image encoder with rank-4 LoRA on q/v attention in every layer.
- LoRA on the mask-decoder transformer self/cross attention and final
  token-to-image attention.
- Trainable `no_mask_embed` prompt embedding and mask-decoder output heads;
  frozen prompt encoder except the no-mask prompt.
- Bilevel update: supervised model step followed by a prompt-embedding step
  on a meta-validation batch.
- Official binary foreground/background output shape (two mask tokens).

## Deviations

1. The official trainer splits the training directory into two halves for
   the two bilevel levels. The zoo runtime does not pass the val loader to
   adapters, so this port uses the previous training batch as the
   meta-validation batch; the first batch uses itself.
2. The official dataset classes/preprocessed folders are replaced by the
   shared 18-image split; binary masks are converted to long labels at
   256 and 64 resolution.
3. The official `num_classes=5` multi-class head becomes a binary
   background+foreground head.
4. Flips and 90-degree rotations are added for the tiny split; the official
   recipe relies on its preprocessed dataset.
5. The shared epoch budget and runtime checkpoint selection are used instead
   of the official `max_epochs=100` / prompt-iteration schedule.
6. The upstream `utils.py` evaluation helpers (SimpleITK/medpy/imageio) are
   not vendored; only the Dice loss used by the prompt optimizer is copied.
7. No official BLO-SAM weights are present. Training starts from
   `sam_vit_b_01ec64.pth`.

The code under `3rd/BLO-SAM` is not imported or modified at runtime.
