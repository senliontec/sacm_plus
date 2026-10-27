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

- SAM ViT-B image encoder frozen; rank-4 LoRA on q/v projections of the
  mask-decoder transformer self-attention, both cross-attention directions
  and the final token-to-image attention (the official
  `sam_lora_mask_decoder.py` module).
- Trainable `no_mask_embed` prompt embedding and mask-decoder output heads;
  frozen prompt encoder except the no-mask prompt.
- Bilevel update: the network (LoRA + decoder) takes a supervised step on
  D1, then the prompt embedding takes a step on D2.  The official loop
  splits the training set once into disjoint halves.
- Official binary foreground/background output shape (two mask tokens).
- Official CE + Dice loss with `dice_param=0.8`, AdamW 5e-3, weight decay
  0.1, rank 4, batch 1, and the effective per-step polynomial LR
  `lr_0 (1 - i / Iter_max)^0.9`, with 100 epochs.
- Checkpoint selection on the D2 foreground Dice computed from the
  low-resolution logits and low-resolution target, as in the official
  `validate`.

## Deviations

1. The official dataset classes/preprocessed folders are replaced by the
   shared 18-image split; binary masks are converted to long labels at
   256 and 64 resolution.  D1/D2 are a seeded, fixed 9/9 split; because
   this protocol pools six domains, the split is stratified so each domain
   appears in both levels.
2. The released loop calls `next(iter(validloader))` every step, so it
   always uses only the first D2 batch; this port cycles through all D2
   batches.  The reported D2 selection Dice uses the full D2 half, as the
   official per-epoch `validate` does, with deterministic resize-only
   preprocessing (no random flips/rotations).
3. The official `num_classes=5` multi-class head becomes a binary
   background+foreground head.
4. Flips and 90-degree rotations are added for the tiny split; the official
   recipe relies on its preprocessed dataset.
5. The official code constructs a `OneCycleLR` for the network optimizer
   but never calls its `step()`; the effective schedule is the manual
   polynomial assignment.  This port implements the effective schedule and
   also uses the shared 100-epoch config (the official `max_epochs=100`).
6. The upstream `utils.py` evaluation helpers (SimpleITK/medpy/imageio) are
   not vendored; only the Dice loss used by the prompt optimizer is copied.
7. No official BLO-SAM weights are present.  Training starts from
   `sam_vit_b_01ec64.pth`.

The code under `3rd/BLO-SAM` is not imported or modified at runtime.
