# ACO-MoE-LoRA upstream record

## Sources

| Component | Upstream | Commit / version |
|---|---|---|
| ACO model, ACO/MoE modules, training loss | `lky0234/ACO-MoE-LoRA`, `ACOMoELoRA.py`, `train.py` | `996270d` (local clone) |
| SAM2 Hiera-L trunk | Meta SAM2, `sam2/modeling/backbones/hieradet.py`, `utils.py`, `sam2_utils.py` | official SAM2; Apache-2.0 |
| SAM2 Hiera-L weights | `sam2_hiera_large.pt` | local `checkpoints/` |

Paper: "ACO-MoE-LoRA: Evolving-while-Training for Adapting Segment Anything
Model 2 to Specialized Domains", ICML 2026.

License: the released ACO repository has **no LICENSE file**; record this and
do not redistribute that code without checking with the authors. The vendored
Meta SAM2 files are Apache-2.0 (`LICENSE_SAM2`).

## What was copied / reconstructed

- `model.py`: the official `ACOMoELoRA.py` model, including
  `ACOHyperparamOptimizer`, `MultiStartACOOptimizer`, `DynamicLoRALayer`,
  `DynamicLoRAWrapper`, `ACOMoeGate`, `AdaptiveExpert`, `ACOConvLoRA`,
  `ACOMoELoRA` and `AdaptiveTrainingScheduler`, with only the backbone
  construction changed (below).
- `backbone/`: the SAM2 Hiera backbone (`hiera.py`), window/patch utilities
  (`utils.py`) and the `DropPath`/`MLP` helpers (`sam2_utils.py`) from Meta
  SAM2. Package-relative imports only.
- Not copied: the SAM2 prompt/mask decoder, memory encoder/attention, FPN
  neck, predictors, automatic mask generator, `_C.pyd` and the ACO
  `train.py`/`test.py`/`eval.py` scripts.

## Backbone reconstruction

The official ACO constructor builds the **full SAM2 model** with
`build_sam2("sam2_hiera_l.yaml", checkpoint)`, deletes
`sam_mask_decoder`, `sam_prompt_encoder`, `memory_encoder`,
`memory_attention`, `mask_downsample`, `obj_ptr_tpos_proj`, `obj_ptr_proj`
and `image_encoder.neck`, then keeps `image_encoder.trunk`.

This port builds the same Hiera-L trunk directly:

- `embed_dim=144`, `num_heads=2`, `stages=(2,6,36,4)`,
  `global_att_blocks=(23,33,43)`,
  `window_pos_embed_bkg_spatial_size=(7,7)`,
  `window_spec=(8,4,16,8)`;
- loads only `image_encoder.trunk.*` keys from the SAM2 checkpoint, strictly.

The used modules and weights are identical to the official model after its
deletion step; the unused SAM2 runtime is not vendored. At prediction the
full model state is in the project checkpoint, so the SAM2 file is not needed.

## Invariants kept

- SAM2 Hiera-L encoder frozen; dynamic LoRA on every `qkv`/`proj` attention
  Linear (max rank 32, scaling `alpha/rank`).
- Decoder: four `ACOConvLoRA(144/288/576/1152 -> 64)`, U-Net
  upsample/concat `up4,up1,up2,up3`, 1x1 head, x2 bilinear to 352.
- ACOConvLoRA: base 3x3 conv, LoRA A/B, 8 experts (1x1->32, 3x3->32, 1x1),
  top-k gate; pheromone log-prior and noisy gating during training;
  deterministic gating at inference.
- ACO search: rank {4,8,16,32}, alpha {4,8,16,32,64}, top-k {1,2,3,4},
  20 ants, evaporation 0.3, diversification interval 50, mutation 0.2,
  argmax after 98% progress.
- Loss: official weighted BCE + weighted IoU (`structure_loss`) plus the
  gate/MoE auxiliary loss (`1.5*CV^2 - 0.1*entropy`), with the official
  adaptive ACO-loss weight.
- Input: 352 square, ImageNet mean/std, random horizontal/vertical flips,
  masks resized BICUBIC and kept soft.
- Optimizer: AdamW lr 1e-3, weight decay 1e-4; cosine to 1e-6; gradient clip
  1.0; no AMP.
- Inference: bilinear logits to GT resolution, sigmoid > 0.5, no TTA.

## Deviations

1. Validation-Dice checkpoint selection instead of the official minimum
   training main loss. The ACO pheromone feedback still uses training IoU.
2. Official batch 12 -> batch 4 x 3 accumulation (effective 12). The local
   smoke uses batch 1 when memory is tight.
3. The official 50-epoch budget is extended to the shared 800-epoch budget;
   the ACO `total_epochs` follows the actual budget so its search schedule is
   consistent.
4. The checkpoint's `aco_hyperparams` are the hyperparameters active when the
   validation-best checkpoint was written, not necessarily the global ACO
   training-best solution.
5. ACO/SAM2 internal search prints are suppressed in the adapter so the
   shared one-row-per-epoch terminal table stays clean.
6. The ACO repo has no license; see above.

## Assets

- `checkpoints/sam2_hiera_large.pt` (876 MB). No ACO-trained weights exist;
  training starts from the frozen SAM2 Hiera-L trunk.
- `--aco-random-init` is a code-smoke option only and must not be used for
  reported results.

## Verification

- `tests/test_aco_moe_lora.py` checks the registry, loss, preprocessing,
  LoRA/MoE modules and config round-trip; the slow test loads the real
  Hiera-L trunk and runs a 64x64 forward.
- One-epoch GPU smoke and a one-image prediction (0/255 PNG at GT resolution)
  were run locally.
