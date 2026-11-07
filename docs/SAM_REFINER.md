# SAMRefiner post-processing

SAMRefiner (ICLR 2025, MIT) is a model-agnostic binary-mask refiner.  It is
vendored under `src/core/postprocess/sam_refiner/`; the upstream tree under
`3rd/SAMRefiner` is never imported at runtime.

## Requirements

- A local SAM checkpoint.  The example uses ViT-H, but ViT-B works for smoke
  tests via `--model-type vit_b`.
- Optional: `FastGeodis` for the official geodesic prompts.  Without it the
  code falls back to SciPy's Euclidean distance transform.

## Apply to a prediction tree

The one-click wrapper `scripts/tools/apply_sam_refiner.py` was removed in the
SACM-STC cleanup.  The vendored refiner under `src/core/postprocess/sam_refiner/`
remains available programmatically; write a small wrapper if the post-hoc
ablation is needed again.

## Evaluation protocol

- Treat this as a post-processing stage, not a backbone baseline.
- Compare `base` vs `base + SAMRefiner` on the same frozen test set.
- Do not stack SAMRefiner with Deep Closing or SegRefiner without a
  factored ablation.
- Select thresholds/iterations on validation, never on test.

## Caveats

- SAMRefiner can change boundary placement and may add/remove regions; it is
  not guaranteed to improve Dice.
- The wrapper refines one binary mask per call; the upstream multi-object
  `split-then-merge` pipeline, HQ-SAM path and SAMRefiner++ IoU adapter are
  not exposed.  They are separate components in the paper/upstream tree.
- One refinement pass is the paper default; cascaded refinement is an
  appendix experiment and may or may not help this task.
- `FastGeodis` is optional; the SciPy fallback uses the Euclidean distance
  transform, which matches the official `lamb=0.0` prompt-distance setting.
- No SAM weights are bundled.
