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
  placement and loss are preserved.

## Official recipe preserved by the adapter

- Base SAM ViT-B, image size 512, binary head: the mask decoder has
  background + foreground tokens (SAMed's `num_classes=1`), no prompt is
  passed to the prompt encoder, and the learned `no_mask_embed` dense prompt
  is used.  This is prompt-free and oracle-free.
- LoRA rank 4 on the q and v projections of every image-encoder block; the
  image encoder is frozen.  The prompt encoder and mask decoder are
  trainable.
- Input preprocessing: RGB [0,1], direct resize to 512 (the official test
  path's lossy 224→512 resampling is not used), SAM pixel mean 0 / std 1,
  bottom/right padding to square.
- Loss: `0.2 * CrossEntropy(low_res_logits) + 0.8 * DiceLoss(2)` on the
  128×128 low-resolution logits.
- Optimizer: AdamW, lr 5e-3, weight decay 0.1.  The official polynomial
  decay `(1 - iter/max_iter)^0.9` is applied per epoch with a short warmup.
- Augmentation: official random 90° rotations + axis flips, and ±20°
  in-plane rotation; grayscale is replicated to three channels.
- Inference: softmax over the two decoder channels, foreground probability
  > 0.5, upsampled to the original GT resolution.

## Deviations in this port

- SAMed's official training is per-dataset full supervision (Synapse: 2211
  slices, batch 24, 200 epochs with a stop at epoch 159).  The 3-shot
  protocol has 18 training images across six seen domains, so this port uses
  batch 4, 160 epochs and the shared 6-image validation Dice for checkpoint
  selection.  The official lr 5e-3 / wd 0.1 are kept; linear LR scaling for
  the smaller batch is left to the user.
- The official 250-iteration warmup is replaced by a 5-epoch warmup because
  an 18-image epoch has only a few iterations.
- The official root SAMed has no mixed precision; this port is also fp32.
- The released SAMed/SAMed_s/SAMed_h LoRA checkpoints are 8-class and are
  not present in the repository; they are not shape-compatible with the
  binary head.  Training starts from the official SAM ViT-B checkpoint.
- The binary head is a protocol adaptation; the official Synapse result is
  8-class.  The released per-class evaluation, HD95 and volume inference are
  replaced by the shared image-level Dice/IoU/clDice/HD95 metrics.
- The official CT intensity window is not used; project images are loaded as
  RGB/grayscale and resized directly.  Grayscale is replicated to RGB.
- Predictions are exported at original GT resolution rather than the
  official 512 argmax + zoom.
- No test-time augmentation.
