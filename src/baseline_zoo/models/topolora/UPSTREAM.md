# TopoLoRA-SAM upstream record

## Sources

- TopoLoRA-SAM implementation: `salimkhazem/Seglab`
  (https://github.com/salimkhazem/Seglab), HEAD
  `7de3a5fbad992363e690190eb82dffbe8a3c1a0b` (2026-01-02).
  Local reference: `3rd/Seglab`.  License: MIT (copied as `LICENSE`).
- SAM: Meta `facebookresearch/segment-anything`, HEAD
  `dca509fe793f601edb92606367a655c15ac00fdf`.
  Local reference: `3rd/segment-anything`.  License: Apache-2.0 (copied as
  `segment_anything/LICENSE`).

## Copied files

From `3rd/Seglab/seglab/models/sam_peft/`:

- `lora.py` — `LoRALinear`, `inject_lora` (verbatim)
- `adapters.py` — `ConvAdapter` (verbatim)
- `topo_losses.py` — `dice_loss`, soft skeletonization, `cldice_loss`,
  `boundary_loss` (verbatim as `topo_losses.py`)

From `3rd/Seglab/seglab/data/transforms.py`:

- `transforms.py` — the official Albumentations train/eval pipelines,
  copied with the same defaults (`RandomScale` 0.8–1.2, pad, 384 crop,
  horizontal/vertical flips, ColorJitter, ImageNet normalize).

From `3rd/segment-anything/segment_anything/`:

- `build_sam.py`, `modeling/{common,image_encoder,prompt_encoder,
  mask_decoder,transformer,sam}.py`
- `LICENSE`

The Meta SAM predictor, automatic mask generator, ONNX helper and transform
utilities are omitted.  `segment_anything/__init__.py` is a minimal export
file (the upstream one imports the omitted predictor/AMG modules).

## Modifications

- `build_sam.py` loads the checkpoint with
  `torch.load(..., map_location="cpu", weights_only=False)`.  This is a
  loading-compatibility change only.
- The upstream `SAMPEFTNet` in `models/sam_peft/model.py` inherits the
  PyTorch Lightning `LitBinarySeg` module and registers itself through
  `seglab.utils.registry`.  `model.py` is a plain `nn.Module` with the same
  forward computation and the same LoRA/adapter/freeze policy; it removes
  Lightning, artifact logging, and the registry import.
- The standard SAM implementation is vendored (the upstream Seglab code
  imports the `segment-anything` pip package).  The model graph and
  parameter names are Meta SAM's.
- No official TopoLoRA-SAM trained weights are present in the Seglab
  repository; training starts from the official SAM ViT-B checkpoint.

## Official recipe preserved by the adapter

- SAM ViT-B (`vit_b`), prompt-free: `prompt_encoder(points=None, boxes=None,
  masks=None)` supplies an empty sparse prompt and the learned
  `no_mask_embed` dense prompt.
- Input: ImageNet-normalized RGB, official 384×384 train/eval transforms.
  The model inverts ImageNet normalization to [0,255], converts to SAM's
  pixel statistics, resizes to 1024, and returns logits at 384.
- LoRA: enabled, rank 8, alpha 16, dropout 0, target keywords
  `mlp`/`fc1`/`fc2` (standard SAM's `mlp.lin1/lin2` layers are wrapped).
  Base SAM parameters are frozen; LoRA parameters are trained.
- Adapter: `ConvAdapter(channels=256, hidden_dim=256, kernel_size=3)`,
  depthwise-separable conv with BatchNorm and a residual.
- Mask decoder: trainable.
- Loss: BCE 1.0 + soft Dice 1.0 + clDice 0.5; boundary loss is disabled.
- Optimizer/schedule: AdamW, lr 1e-4, weight decay 1e-4, cosine,
  50 epochs, batch size 1 with 4 gradient-accumulation steps, gradient
  clipping 1.0.  Upstream uses 16-bit mixed precision.
- Checkpoint selection: val Dice (upstream `ModelCheckpoint` monitors
  `val/dice`, mode max).
- Inference threshold: sigmoid > 0.5.

## Deviations in this port

- The shared baseline runtime is plain PyTorch; the official Lightning
  trainer, CSV/W&B loggers, artifact figures and per-image CSV are not
  reproduced.  The model, loss, optimizer, schedule and accumulation are
  preserved.
- No mixed precision (the upstream `precision: 16-mixed` is not used).
  This changes speed and numerical details, not the model graph; a larger
  GPU or an explicit AMP extension would be needed for exact throughput.
- The port uses the shared 18/6 split and the project 20-dataset test
  protocol, not Seglab's HuggingFace train/validation/test split.
- Predictions are exported at original GT resolution; the upstream
  evaluator compares at 384×384.
- Albumentations 1.3.1 has no per-Compose seed; the adapter seeds the
  global Python/NumPy RNG from the shared per-epoch RNG before each train
  batch, making the official transform sequence reproducible.
- Albumentations 1.3.1 spells constant padding as `value`/`mask_value`
  instead of the upstream code's `fill`/`fill_mask`; the port uses the
  1.3.1 names with the same zero padding.
- Training uses the official ViT-B backbone, while other SAM baselines in
  this zoo use ViT-L; the backbone is documented in the paper table.  A
  ViT-L variant can be built with `sam_type='vit_l'` but is not the
  official TopoLoRA-SAM configuration.
