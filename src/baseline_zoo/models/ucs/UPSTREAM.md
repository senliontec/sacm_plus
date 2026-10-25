# UCS (Universal Curvilinear Structure)

- Upstream: https://github.com/kylechuuuuu/UCS
- Paper: "UCS: A Universal Model for Curvilinear Structure Segmentation"
  (arXiv:2504.04034)
- Vendored snapshot: `3rd/UCS`, git HEAD `cff2db1`
- License: no LICENSE file in the upstream snapshot; do not redistribute
  without checking with the authors.

## Role

Appendix / oracle-prompted reference baseline.  The released inference path
derives boxes and points from the **GT mask** and samples refinement clicks by
comparing predictions to GT.  It is therefore **not prompt-free**, despite the
paper's no-interaction claim.  The FFT high-pass `PromptGenerator` is an
encoder-side feature prompt, not an automatic SAM point/box generator.

## Files copied

| Upstream | Destination |
|---|---|
| `segment_anything/__init__.py` | `segment_anything/__init__.py` (predictor/AMG imports removed) |
| `segment_anything/build_sam.py` | `segment_anything/build_sam.py` (`weights_only=False`) |
| `segment_anything/modeling/{common,image_encoder,mask_decoder,prompt_encoder,sam_model,transformer}.py` | same paths (package-relative imports preserved) |
| `segment_anything/modeling/__init__.py` | same |
| `utils.py` | `utils.py` (prompt/loss/transform helpers) |

Omitted: `predictor.py`, `predictor_sammed.py`, `automatic_mask_generator.py`,
`utils/{amg,onnx,transforms}.py`, `sam.py` (unused broken wrapper),
`DataLoader.py`, `metrics.py`, `metric.py`, `PCA.py`, `train.py`, `test.py`,
`interface.py`.

## Modifications

1. `build_sam.py`: `torch.load(..., weights_only=False)` for modern PyTorch.
2. `mask_decoder.py`: PCA feature moved to `mask_tokens_out.device` instead of
   hardcoded `.cuda()`; added a clear batch-size-1 guard because `pca_fc`
   expects `num_mask_tokens**2` inputs.
3. `segment_anything/__init__.py`: removed imports of the incompatible
   predictor and automatic mask generator.
4. The adapter bypasses the UCS JSON dataset loaders and uses
   `baseline_zoo.data.Sample` plus the shared split manifests.

## Important code/paper discrepancies

- **Prompt semantics:** official test.py uses GT boxes/points and GT-guided
  refinement clicks; it is oracle-prompted. The paper's "automatic prompt
  generation" is not implemented in the released test path.
- **Architecture drift:** HEAD `cff2db1` refactored the paper's heavy Sparse
  Adapter into a lightweight dense Linear-GELU adapter on the four global
  blocks, and commit `06ec17f` removed the HQ-Feature Compression branch.  The
  paper-faithful modules are in git history
  (`cff2db1^:segment_anything/modeling/image_encoder.py`,
  `06ec17f^:segment_anything/modeling/mask_decoder.py`).  This port uses HEAD
  (the current official code) and documents the difference.
- **PCA/GFC:** the remaining GFC branch runs sklearn PCA on CPU inside every
  decoder forward; it is detached, non-differentiable, and requires batch 1.
- **Selection:** upstream selects the best checkpoint by training Dice and has
  no validation loop; the adapter overrides `selection_score` to match that.
- **No seeds:** upstream does not seed RNGs; the shared runtime seeds the
  adapter, including prompt sampling.

## Checkpoint

Training starts from `checkpoints/sam_vit_l_0b3195.pth`.  No official trained
UCS weights were found.  The adapter stores the full UCS state dict and the SAM
checkpoint path/SHA in the adapter config.

## Runtime consumer

`baseline_zoo.adapters.ucs.UCSAdapter`
