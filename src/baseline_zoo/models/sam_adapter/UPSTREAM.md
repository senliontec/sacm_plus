# SAM-Adapter upstream record

## Source

- Repository: `tianrun-chen/SAM-Adapter-PyTorch`
- URL: https://github.com/tianrun-chen/SAM-Adapter-PyTorch
- Paper: *SAM Fails to Segment Anything? — SAM-Adapter: Adapting SAM in
  Underperformed Scenes: Camouflage, Shadow, and More* (ICCV 2023),
  arXiv:2304.09148.
- Local reference snapshot: `3rd/SAM-Adapter-PyTorch` (read-only third-party
  tree).  The snapshot does not expose a usable commit hash.
- License: the upstream repository ships an MIT `LICENSE` (Copyright (c)
  2023 tianrun-chen), copied here as `LICENSE`.  The vendored SAM-derived
  modules retain Meta Platforms' copyright headers and originate from
  `facebookresearch/segment-anything`.

## Copied files

From `models/mmseg/models/sam/`:

- `common.py` — `MLPBlock`, `LayerNorm2d`
- `transformer.py` — `TwoWayTransformer`
- `mask_decoder.py` — `MaskDecoder`
- `image_encoder.py` — `ImageEncoderViT`, FFT high-pass `PromptGenerator`,
  patch embedding and ViT blocks

From `models/`:

- `iou_loss.py` — the official `IOU` loss
- `LICENSE`

The upstream `models/sam.py` was not copied verbatim.  `model.py` is a reduced
port that keeps the official `SAM.__init__` architecture, `PositionEmbeddingRandom`,
`get_dense_pe`, prompt-free `infer`, and `postprocess_masks`.  The upstream
`forward`, `set_input`, `backward_G`, `optimize_parameters`, logging and
`@register('sam')` plumbing are removed because the shared baseline runtime
owns training and checkpointing.

## Modifications

- All imports are package-relative; there is no `models` package, mmseg,
  mmcv, DDP or `baseline/3rd` import.
- `model.py` removes the dead `if 'evp' in encoder_mode['name']` branch
  (`self.encoder` does not exist upstream; the official ViT-L config uses
  `name: sam`, so the branch never ran).
- `SAM.infer` derives the device/dtype from the module and keeps the
  upstream single prompt set (`prompt_batch = 1`) broadcast over the image
  batch.  The official code hardcodes `bs = 1` in the same way.
- No parameter names, tensor shapes, FFT/prompt computations, attention
  blocks, decoder or loss formulas were changed.
- `image_encoder.py` keeps the upstream hardcoded prompt settings
  (`scale_factor=32`, `prompt_type='highpass'`, `input_type='fft'`,
  `freq_nums=0.25`, `tuning_stage=1234`).  The YAML keys in
  `encoder_mode` do not override them upstream either.
- `image_encoder.py` gains an optional `grad_checkpointing` flag (default
  `False`) used by `--grad-checkpointing`.  When enabled it wraps each ViT
  block in `torch.utils.checkpoint`; this changes activation memory only,
  not the computation.  It is off by default, matching the upstream code.

## Official recipe preserved by the adapter

- Input: RGB, direct resize to 1024×1024, ImageNet normalization
  (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`); mask resized with
  nearest-neighbour.
- Training augmentation: random horizontal flip with p=0.5.  The upstream
  `TrainDataset` applies this flip unconditionally even though the config
  says `augment: false`; the port follows the executable code.
- Model: ViT-L (`embed_dim=1024`, depth 24, 16 heads, patch 16, four global
  attention blocks), prompt dimension 256, mask decoder transformer depth 2.
- Initialization: raw `sam_vit_l_0b3195.pth` loaded with `strict=False`.
  The FFT prompt generator and the random positional encoding are newly
  initialized; the prompt-encoder keys in the SAM checkpoint are unused.
- Trainable parameters: all `image_encoder` parameters except
  `prompt_generator` are frozen; the prompt generator, mask decoder,
  `pe_layer` and `no_mask_embed` are trained.  This is the upstream freeze
  policy.
- Loss: `BCEWithLogitsLoss()` plus `1 - IoU` (`iou_loss.IOU`), both averaged
  over the batch.
- Optimizer/schedule: AdamW, lr 2e-4, executable weight decay 0.01
  (PyTorch default; upstream passes only lr), 20 epochs,
  `CosineAnnealingLR(T_max=20, eta_min=1e-7)`, official batch size 2.
- Inference: prompt-free.  Empty sparse prompt plus the learned
  `no_mask_embed` dense prompt; sigmoid threshold 0.5.

## Weights

The upstream README links trained SAM-Adapter checkpoints from Google Drive,
but they are not part of the vendored tree and were not used.  Training starts
from the official SAM ViT-L checkpoint.  The port is therefore a retrained
baseline, not a reproduction of the authors' released weights.

## Dataset / metric differences

The upstream code is written for COD datasets and reports S-measure, E-measure,
weighted F-measure and MAE.  This port consumes the shared 18/6 split and
exports predictions at original GT resolution; the shared runtime selects the
checkpoint by the 6-image validation Dice and reports Dice/IoU/clDice/HD95.
