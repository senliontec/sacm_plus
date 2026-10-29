# FSSAM upstream record

## Sources

| Component | Upstream | Commit |
|---|---|---|
| Method / training / priors / IMR | `Sam1224/FSSAM`, `model/FSSAM.py` and `model/FSSAM5s.py`, `train.py`, `util/transform_new.py`, `util/dataset.py` | `9318e28e24d08534aef9facde31dfe728746cc5d` |
| Modified SAM 2 (FSSAM additions) | same repo, `sam2/` and `sam2_configs/` | same |
| DINOv2 implementation | `facebookresearch/dinov2`, vendored in `baseline_zoo.models.matcher.dinov2` (Apache-2.0) | see Matcher's UPSTREAM |

Paper: "Unlocking the Power of SAM 2 for Few-Shot Segmentation",
ICML 2025, arXiv:2505.14100.

License: FSSAM's repository contains an MIT license (`LICENSE`); the
vendored SAM 2 files carry Meta's Apache-2.0 header. The DINOv2 package is
Apache-2.0.

## Copied and adapted

- `sam2/`: the complete FSSAM-modified SAM 2 runtime (25 files). Imports
  were rewritten from `sam2.*` to
  `baseline_zoo.models.fssam.sam2.*`; the Hydra config's `_target_`
  strings were rewritten the same way. `sam2/__init__.py` no longer
  registers a global Hydra config module.
- `sam2_configs/`: Hiera-S, Hiera-B+ and Hiera-L YAML configs with the
  vendored `_target_` paths.
- `sam2/build_sam.py`: replaced Hydra-global `compose` with direct OmegaConf
  loading and `hydra.utils.instantiate`; `build_sam2_video_predictor` sets
  the `SAM2VideoPredictor` target programmatically and keeps the upstream
  `binarize_mask_from_pts_for_mem_enc`/`fill_hole_area` defaults. Checkpoint
  loading is strict for missing/unexpected keys and uses
  `weights_only=False` (SAM2 checkpoints are dicts).
- `model.py`: FSSAM5s' forward reimplemented with explicit shapes:
  `Weighted_GAP`, `weighted_dice_loss`, `generate_prior_proto`,
  `generate_prior_mem`, `mem_refine`, the DINOv2 prior path, the SAM 2
  memory pipeline, IMR and SCMA. The DINOv2 package is reused from the
  self-contained Matcher port; the SAM 2 and DINOv2 weights are local.
- `LICENSE`: verbatim FSSAM license.

## Invariants kept from the paper / official code

- DINOv2 input: 560 square, bilinear (`F.interpolate(..., mode='bilinear',
  align_corners=True)`, matching the release), `(x/255 - ImageNet mean)/ImageNet std`
  (the code scales mean/std by 255 and operates on 0-255 tensors).
- Pseudo Prompt Generator: support FG/BG prototypes by weighted GAP,
  max-over-channels cosine similarity, per-image min-max normalization,
  `Disc = ReLU(FG - BG)`, per-support min-max after the subtraction
  (FSSAM5s variant for k-shot), then averaged over supports.
- Support memory from the GT support mask; pseudo query FG memory from
  `P_FG`; pseudo query Disc memory from `P_Disc`.
- IMR: query-Disc and query-FG memory similarities, FG-memory minus
  support-memory gating (`ReLU(A_QQ + A_QS - 1)`), weighted update of both
  the Disc memory and the Disc prior, repeated `num_refine=3` times. The
  support/query similarity is computed once and reused (the released code
  caches it in `prior_mem_qry_sup_fg`).
- SCMA: support memories calibrate the query-memory attention bias
  (`Norm(A_QQ) + Norm(A_SQ) - 1`, negative entries scaled by 10), support
  keys/values are averaged over shots, query memory plus object-pointer
  tokens are kept; mask decoder produces the final logits.
- Training: SAM 2 image encoder, prompt encoder, `obj_ptr_proj`,
  `mask_downsample` and DINOv2 frozen; memory encoder, memory attention
  and mask decoder optimized with AdamW lr 1e-3, wd 0, poly power 0.9, no
  warmup. Dice loss only. 512-square random scale (0.9-1.1), ±10° rotation
  (p=0.5), Gaussian blur 5 (p=0.5), horizontal flip (p=0.5), random
  512 crop with mean/255 padding and mask ignore 255; effective batch 8;
  300 epochs (PASCAL) / 75 (COCO), early stopping patience 100.
- Evaluation: plain 512-square resize, logits bilinearly resized back to
  the original GT size, sigmoid > 0.5.

## Repairs required to run the released code

1. `sam2/modeling/sam/transformer.py` calls `self.Weighted_GAP` in the SCMA
   branch but never defines it. The identical helper exists as a module
   function in `model/FSSAM.py`; this port adds it as a static method on
   `RoPEAttention`.
2. The same loop overwrites `sup_mask` with the current shot's mask, which
   makes the calibration tensor invalid for `shot > 1`. The port keeps the
   stacked `[B*S, 1, h, w]` mask tensor and indexes it per shot, matching
   the paper's k-shot averaging of `A_QS`.
3. The released FSSAM5s passes only `s_mask[:, 0]` to SCMA. This port
   passes all support masks and averages their normalized similarities.
4. `add_new_mask_batch` hard-casts memory features to bfloat16 (the
   official trainer runs under global bf16 autocast). In fp32 mode the
   port casts the memory features/positional encodings/predictions back to
   the model dtype.

## Deviations

1. Support-based adaptation: the shared 18-image split has three images
   per seen domain. Training episodes use at most two supports after
   excluding the query image; evaluation uses the protocol's three supports
   (own domain for seen test sets, donor domain otherwise).
2. Backbones: official main is SAM 2-S + DINOv2-B. Default here is SAM 2-S
   + local DINOv2-L (the paper's ablation variant); Hiera-B+/L and
   DINOv2-B are selectable. `--fssam-dino-size vitb` selects
   `checkpoints/dinov2_vitb14_pretrain.pth` automatically unless
   `--fssam-dino-weights` is supplied. The checkpoint must match the
   selected size.
3. fp32 instead of bf16 autocast (hardware-independent; the 3090 server
   could use bf16, but the shared runtime does not own an autocast loop).
4. Per-epoch poly LR (the shared runtime steps once per epoch). The
   official 300/75-epoch budgets are extended to the shared 800-epoch
   budget; checkpoint selection is validation Dice, not the official
   per/half-epoch mIoU with patience-100 early stopping.
5. The DINOv2 prior path requests only the final block; the upstream
   requests all blocks and keeps the last, so the consumed feature is
   identical.
6. No FSSAM-trained weights are required; training starts from the
   official SAM 2 and DINOv2 checkpoints. The port never downloads weights.
7. Images and supports are resized to 512 square (stretch) as in the
   official validation transform; predictions are resized to the original
   GT resolution.
