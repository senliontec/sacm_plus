# SAMed upstream record

## Source

- Repository: `hitachinsk/SAMed`
  (https://github.com/hitachinsk/SAMed)
- HEAD: `e351840481f51d0c7dee50007f29d02197b63d99` (2023-09-02).
- Local reference: `3rd/SAMed`.
- License: MIT (Copyright (c) 2023 Kaidong Zhang), copied as `LICENSE`.
  The SAM-derived modules retain Meta Platforms' source headers.

## Copied files

From `3rd/SAMed/segment_anything/`:

- `build_sam.py`
- `modeling/{common,image_encoder,mask_decoder,prompt_encoder,
  transformer,sam}.py`

From `3rd/SAMed/`:

- `sam_lora_image_encoder.py` — `_LoRA_qkv` and `LoRA_Sam`, reduced to
  construction/reset/forward (the safetensors save/load helpers and their
  unused imports are omitted).
- `utils.py::DiceLoss` — copied into `losses.py`.
- `LICENSE`

The predictor and automatic mask generator (which import cv2/pycocotools)
are intentionally omitted, and `segment_anything/__init__.py` is reduced to
the builders and `Sam` export.

## Modifications

- Removed unused `icecream` imports from the vendored SAM files.
- `build_sam.py` loads the base checkpoint with
  `torch.load(..., map_location="cpu", weights_only=False)`.
- `LoRA_Sam` is a minimal self-contained wrapper; the official save/load
  helpers reference module-level globals and are not used by this runtime.
- The Lightning/Synapse training loop, H5/NPZ dataset class, per-class
  metrics and volume test helper are replaced by the shared baseline
  adapter/runtime.  The model graph, binary class-token decoder, LoRA
  placement and prompt-free decoder are preserved; the loss is adapted for
  the tiny curvilinear foreground (see below).
- A small zero-initialized full-resolution residual refinement head
  (`HighResRefinementHead`) reads the input image and corrects the
  bilinearly upsampled low-resolution logits.  It starts as an exact no-op
  and is supervised by the auxiliary full-resolution term.  This is a
  task-specific adaptation for thin structures, not part of the released
  SAMed architecture; set `--samed-high-res-head 0` to disable it.

## Official recipe elements preserved by the adapter

- Base SAM ViT-B, image size 512, binary head: the mask decoder has
  background + foreground tokens (SAMed's `num_classes=1`), no prompt is
  passed to the prompt encoder, and the learned `no_mask_embed` dense prompt
  is used.  This is prompt-free and oracle-free.
- LoRA rank 4 on the q and v projections of every image-encoder block; the
  image encoder is frozen.  The prompt encoder and mask decoder are
  trainable.
- Input preprocessing: RGB [0,1], direct resize to 512, SAM pixel mean 0 /
  std 1, bottom/right padding to square.  The official 2D test branch also
  resizes directly to 512; its 224→512 resampling belongs to the 3D volume
  path.
- Loss: SAMed's class-wise CE+Dice idea is retained, but adapted to the
  binary curvilinear task: class-weighted CE (background 0.1, foreground
  1.0) plus foreground-only Dice on the 128×128 low-resolution logits,
  a soft-clDice topology term (weight 0.1), and an auxiliary
  full-resolution (512) CE+Dice term with weight 0.5.  The official
  equal-weight 2-class CE+Dice is not used for the adapted baseline.
- Optimizer: AdamW, weight decay 0.1; paper lr 5e-3 was tuned for batch 24,
  so the batch-4 default is 1e-3.  The official polynomial decay
  `(1 - epoch/total)^0.9` is applied per epoch with the 250-iteration
  warmup converted to 50 epochs at batch 4.  Gradient clipping 1.0 is added
  for stability.
- Augmentation: official random 90° rotations + axis flips, and ±20°
  in-plane rotation; grayscale is replicated to three channels.
- Inference: argmax over the two full-resolution refined logits at 512,
  followed by nearest-neighbour resize to the original GT resolution
  (official 2D evaluator semantics).  With the refinement head disabled,
  this is the official bilinear-upsampled low-resolution output.

## Deviations in this port

- SAMed's official training is per-dataset full supervision (Synapse: 2211
  slices, batch 24, 200 epochs with a stop at epoch 159).  The 3-shot
  protocol has 18 training images across six seen domains, so this port uses
  batch 4, 800 epochs and the shared 6-image validation Dice for checkpoint
  selection.  The paper lr 5e-3 is kept as `paper_lr`; the adapted batch-4
  default is 1e-3.
- The official 250-iteration warmup is preserved in iteration units; with
  the 18-image loader and batch 4 it spans about 50 epochs.
- The binary objective is adapted to the curvilinear foreground because the
  official equal-weight background/foreground CE+Dice collapses to an
  all-background solution at 0.17%-8% foreground.  Foreground-only Dice,
  class-weighted CE, soft-clDice, 512-resolution auxiliary supervision and
  the full-resolution residual refinement head are the task-specific
  adaptations; they are not claimed as SAMed's Synapse recipe.
- Training uses a WeightedRandomSampler with inverse-sqrt foreground
  weights, so low-foreground images receive more updates.  Validation is
  unchanged.  Set ``--samed-oversample 0`` to disable it.
- ``--input-size 1024`` (low-res 256) is supported as a true higher-
  resolution option for thin structures; the default remains 512 to keep
  the shared project budget and memory.
- The official root SAMed has no mixed precision; this port is also fp32.
- The released SAMed/SAMed_s/SAMed_h LoRA checkpoints are 8-class and are
  not present in the repository; they are not shape-compatible with the
  binary head.  Training starts from the official SAM ViT-B checkpoint.
- The official CT intensity window is not used; project images are loaded as
  RGB/grayscale and resized directly.  Grayscale is replicated to RGB.
- Predictions use argmax + nearest resize, not the official 8-class
  per-volume evaluation; the shared Dice/IoU/clDice/HD95 metrics are used.
- No test-time augmentation.
