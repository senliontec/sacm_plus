# FoB-SAM (CVPR 2026)

- Upstream: https://github.com/primebo1/FoB_SAM
- Vendored snapshot: `3rd/FoB_SAM`, HEAD
  `40d0e23d75f81f07bb7f34cb641db51ef490d025`.
- Paper: *Focus on the Background: Exploring SAM's Potential in Few-shot
  Medical Image Segmentation with Background-centric Prompting*, CVPR 2026
- License: Apache License 2.0 (copied as `LICENSE`)
- Files copied:
  - `models/FoB.py` -> `model.py`
  - `models/encoder.py` -> `encoder.py`
  - `SAM.py` -> `sam.py`
  - `dataloaders/image_transforms.py` -> `transforms.py`
- Runtime consumer: `baseline_zoo.adapters.fob_sam.FoBSAMAdapter`

## Model logic preserved

- `FewShotSeg` / `IDR` / `SPG` / `SPR` / `Head` / `PromptMatching` /
  `MaskedAttention` / `JointsMSELoss` are copied verbatim, including the
  official loss composition (`heatmap_loss*1000 + L2/10000 + rac_loss +
  foreground_loss`).
- Background prompt construction: contour sampling with the official
  21/15-pixel ring dilation, Gaussian keypoint heatmaps (sigma=4),
  masked-average prototypes, query attention suppression, and
  structure-guided prompt refinement (`SPR`, 3 iterations).
- Frozen SAM point-prompt inference is preserved.  SAM modeling code is
  reused from the already-vendored `baseline_zoo.models.sam_octa`
  Segment Anything copy (official Meta code), not duplicated.

## Deviations required by this repository

- The official model hard-codes CUDA in `FewShotSeg.__init__`,
  `MaskedAttention.__init__` and the NLL weight.  This port registers
  `sin_pos` and the NLL weight as buffers and makes the model
  device-agnostic so CPU smoke tests and `.to(device)` work.
- Absolute checkpoint paths are replaced by
  `--resnet-checkpoint` / `--sam-checkpoint`.
- `info_nce` is an optional dependency.  The official forward never
  calls `self.InfoNCE`; a dependency-free placeholder is provided when
  the package is absent.
- The official training is 3D-volume/episodic with supervoxel pseudo
  classes.  The 20-dataset adapter uses 2D support/query pairs with GT
  support masks (`use_gt=True` equivalent), 1-shot training episodes and
  same-domain validation support.
- For unseen/novel test datasets (no labeled support in the 3-shot
  split), the adapter uses a structurally matched donor support set:
  crack-like datasets -> `CrackTree200`, vessel-like datasets -> `DRIVE`;
  override with `--fob-support-dataset`.
- Prompt predictions and SAM masks are produced at 256x256 and resized
  to the original GT resolution for the shared metrics.
- The official test passes `query_labels` into `forward`; it is used only
  for the empty-foreground guard.  The adapter passes an all-ones dummy
  at evaluation, so test GT never influences prompts or masks.
- SAM `best_pred_idx=0` (the official non-ISIC setting); expose
  `--fob-best-pred-idx` for completeness.

## Paper vs released-code discrepancies (kept code-faithful)

The CVPR 2026 paper and the released code differ in several method details.
This port follows the released code; these are documented here so results
are not mislabeled as a literal paper-formula reproduction:

- RAC loss: paper Eq.16 uses InfoNCE with temperature 0.1 and an
  outermost-foreground positive; the released `forward` uses
  `clamp(0.5 + cosine, min=0)` over the full foreground prototype and never
  calls the instantiated `InfoNCE`.
- Support prompt ring: paper Eq.1 specifies dilation 15 minus 13
  (`r=15`, `epsilon=2`); `get_ring()` in the released code uses dilation
  21 minus 15.
- Inference foreground threshold: paper §3.7 says 0.9; the released code
  uses 0.96.
- Training budget: paper says 36K iterations with 0.95 decay every 1K; the
  released scripts use 39001 iterations with 0.98 decay every 3K.
- Heatmap loss: paper Eq.17 normalizes the sum of coarse+refined MSE by
  `N_p H W`; the released `JointsMSELoss` call applies an extra leading
  dimension to the head output, effectively dividing the head term by 1
  instead of `N_p`, and uses a 0.5 factor.
- GCN: paper Eq.11 uses `D^{-1/2} A D^{-1/2} Q W_g`; the released `SPG`
  row-normalizes A and applies a sigmoid channel gate.
- `L_fore` is weighted NLL in code (`[0.1, 1.0]`) rather than the paper's
  unweighted BCE; `L_corr` uses `nn.MSELoss` mean rather than `1/N_p` sum.

The adapter's 20-dataset support/donor policy and 2D GT episodes are
additional protocol adaptations, not paper/code details.
