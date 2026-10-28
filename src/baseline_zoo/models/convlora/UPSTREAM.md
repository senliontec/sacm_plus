# Conv-LoRA upstream record

## Sources

| Component | Upstream | Commit |
|---|---|---|
| Method / layers / loss / recipe | `autogluon/autogluon`, `examples/automm/Conv-LoRA` and `multimodal/src/autogluon/multimodal/models/{adaptation_layers,sam}.py`, `models/custom_hf_models/modeling_sam_for_conv_lora.py`, `optim/losses/structure_loss.py`, `configs/model/default.yaml` | `9ad8e9a9d8028f87d6b1e8572191389351f9a351` |
| SAM backbone | `facebookresearch/segment-anything` (`segment_anything/`) | `dca509fe793f601edb92606367a655c15ac00fdf` |

Paper: "Convolution Meets LoRA: Parameter Efficient Finetuning for Segment
Anything Model", ICLR 2024, arXiv:2401.17868.

License: AutoGluon is Apache-2.0 (`LICENSE_AUTOGLUON`); Meta SAM is
Apache-2.0 (`LICENSE_SAM`). The LoRA base class originates from Microsoft
loralib (MIT) via AutoGluon.

## Copied and adapted

- `layers.py`: `ConvLoRALinear`, `MoEGate`, `SparseDispatcher`, `LoRALayer`
  copied from AutoGluon `adaptation_layers.py`. Only dead imports were
  removed and names/types made explicit. The module returns
  `(output, moe_loss)` and records `last_moe_loss` so the binary head can
  aggregate the per-layer MoE losses.
- `losses.py`: `StructureLoss`, copied from AutoGluon
  `optim/losses/structure_loss.py` (weighted BCE + weighted IoU, weight
  `1 + 5*|avg_pool(gt,31) - gt|`). Only reduction/labels were made explicit.
- `model.py`: self-contained assembly using the vendored Meta SAM. It
  injects `ConvLoRALinear` into every ViT block's `qkv`, reduces the mask
  decoder to one binary token, and implements the prompt-free forward.
- `segment_anything/`: self-contained Apache-2.0 Meta SAM copy. Only
  `build_sam`, the modeling modules and `utils/transforms` are kept;
  predictors, ONNX export and the automatic mask generator are omitted.
  `image_encoder.Attention.forward` accepts the tuple returned by
  `ConvLoRALinear` (`qkv = qkv[0]`). No other behavioral change.
- `__init__.py`: exports `StructureLoss`, `ConvLoRASAM`, `build_convlora`.
- `LICENSE_SAM`, `LICENSE_AUTOGLUON`: verbatim upstream licenses.

## Invariants kept from the paper / official code

- Conv-LoRA is applied to the **query, key and value** projection (the
  fused `qkv` linear) of every image-encoder self-attention block.
- Rank `r=3` and `lora_alpha=8` (AutoGluon `optim.lora.alpha` default;
  the released `run_semantic_segmentation.py` does not override it).
  Scaling is `alpha/r`. The value is exposed via `--convlora-alpha`.
- `topk=1` noisy MoE gate over 8 convolutional experts with upsampling
  ratios 1..8; each expert is `interpolate(bicubic) -> 3x3 Conv+GELU ->
  interpolate(bicubic)`; expert outputs are combined through
  `SparseDispatcher` (softmax/exponential combine).
- LoRA A is Kaiming-initialized, B is zero; the pretrained qkv weight is
  frozen. LoRA scaling is `alpha/r`.
- MoE load-balancing loss: `cv_squared(importance) + cv_squared(load)` with
  coefficient `1e-2` inside every gate, summed over all layers and added to
  the main loss with weight 1.0 (paper: 1.0 for binary segmentation).
- Prompt encoder frozen; constant empty prompt (SAM `no_mask_embed`) is the
  only dense prompt; sparse prompt is the empty sequence.
- Mask decoder reduced to one binary mask token; first hypernetwork MLP and
  the mask token are copied from the standard SAM weights. Paper: full
  fine-tuning of the mask decoder. The official AutoGluon SAM config
  freezes `mask_decoder.iou_prediction_head`, so that head stays frozen.
- Structure loss (weighted BCE + weighted IoU), batch size 4 (1 x 4
  accumulation), 30 epochs (100 in the paper's low-data study), random
  horizontal flip, images and masks resized to 1024 square.
- Official preprocessing: `resize_to_square` bicubic for images, nearest
  for masks, `ToTensor` (0-1) and **ImageNet** normalization
  (`configs/model/default.yaml: image_norm: imagenet`) before the SAM
  encoder. Meta SAM pixel mean/std are not applied.
- Official optimizer/schedule in the released example: AutoGluon defaults
  `adamw`, cosine decay, `warmup_steps=0.1` of the total steps, gradient
  clipping 1.0, layerwise LR decay 0.9, top-3 greedy-soup checkpoint
  averaging. The paper states Adam, lr 1e-4 and weight decay 1e-4.

## Deviations / decisions

1. **Mask decoder training.** The paper fine-tunes the decoder and reports
   4.02M trainable parameters (3.51M decoder + ~0.51M Conv-LoRA). The
   released AutoGluon PEFT path computes trainable names as `.*lora_*.`
   only (`get_peft_param_names`), which would freeze the decoder despite
   the SAM config listing only the IoU head/prompt encoder as
   `frozen_layers`. This port follows the paper: the decoder is trainable
   except `iou_prediction_head`.
2. **Optimizer.** Paper: Adam, lr 1e-4, weight decay 1e-4 (implemented).
   The released AutoGluon defaults are AdamW with weight decay 1e-3.
3. **Schedule.** AutoGluon cosine decay with 10% warmup and gradient
   clipping 1.0 is implemented, but the shared runtime steps the scheduler
   once per epoch instead of per iteration. Layerwise LR decay 0.9 and
   top-k greedy soup are not reproduced; checkpoint selection is the
   project's validation Dice.
4. **Budget.** The official 30-epoch recipe is extended to the shared
   800-epoch 3-shot budget; the effective batch 4 is kept via batch 1 and
   four accumulation steps.
5. **alpha.** The released example script sets `rank=3` and `expert_num=8`
   but leaves AutoGluon's `lora_alpha` default at 8. The paper does not
   state alpha. This port defaults to 8 and records the value in the
   checkpoint config; `--convlora-alpha` is available for HPO.
6. **SAM weights.** The official example loads `facebook/sam-vit-huge`
   through HuggingFace. This port loads the equivalent official Meta SAM
   ViT-H checkpoint locally (`checkpoints/sam_vit_h_4b8939.pth`), avoiding
   the offline server's missing `transformers`/network. No Conv-LoRA
   trained weights exist in the upstream tree; training starts from SAM.
7. **Task adaptation.** The official model is multi-class semantic
   segmentation with a Mask2Former-style classification branch. The problem
   here is binary, prompt-free curvilinear segmentation, so the port keeps
   the single binary mask token and the structure loss. This is the same
   binary head used by the official `num_classes=1` path.
8. **Datasets.** The shared 18-image 3-shot split replaces the AutoGluon
   CSV manifests. No test labels are used.
9. **Memory.** A one-image ViT-H training step OOMs on an 11 GiB card
   because activations for every ViT-H block are retained for the LoRA
   backward pass. The target 24 GiB GPUs fit; `vit_b` is available for
   local smoke tests only.
