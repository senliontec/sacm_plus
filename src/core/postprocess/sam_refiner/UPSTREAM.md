# SAMRefiner upstream record

## Source

- Repository: `linyq2117/SAMRefiner`
  (https://github.com/linyq2117/SAMRefiner)
- Local reference: `3rd/SAMRefiner`
- HEAD: `4bb7f95738c7a2e805f189210fb09e02a2197557`
- License: MIT (copyright 2025 linyq2117); copied as `LICENSE`.
  The vendored `segment_anything` package carries Meta's Apache-2.0 license.
- Paper: Yuqi Lin et al., "SAMRefiner: Taming Segment Anything Model for
  Universal Mask Refinement", ICLR 2025.

## What was preserved

- `sam_refiner.py` noise-tolerant prompting: distance-guided positive
  points, optional negative points, elastic boxes, Gaussian mask prompts,
  iterative refinement and multimask IoU selection.
- The modified SAM package with `forward_with_image_embeddings`.
- Binary 0/1 coarse-mask input and PNG output.

## Deviations

1. `FastGeodis` is optional. If it is absent, `utils._generalised_geodesic2d`
   uses SciPy's Euclidean distance transform. The official
   `extract_points` calls FastGeodis with `lamb=0.0` (Euclidean distance), so
   the fallback matches the metric actually used by SAMRefiner; install
   `FastGeodis` only to reproduce its exact update scheme / other lambdas.
2. The wrapper exposes one binary mask per call; multi-object `split-then-
   merge` (STM), HQ-SAM and the SAMRefiner++ IoU-adapter are not exposed by
   the CLI.  These are separate paper components, not needed for universal
   binary-mask refinement.
3. The CLI resolves source images from the protocol manifest, because the
   prediction tree contains masks only.
4. No SAM weights are bundled. The user must pass a local standard SAM
   (ViT-H/L/B) checkpoint; CPU execution is supported but slow.
5. The paper uses a single refinement pass with box margin 0.1, Gaussian
   amplitude 15 and span 4.  The released `sam_refiner.py` defaults are 5
   passes, margin 0.0 and amplitude 30.  The wrapper/CLI default to the
   paper values; pass `--iterations 5 --margin 0.0 --strength 30` to
   reproduce the released-code default.  Iterative cascades are an appendix
   experiment, not the paper's main setting.
6. Upstream's 2-D mask branch uses `coarse_masks[None: ,]`, which is a
   no-op slice rather than a new batch axis.  The vendored copy changes it
   to `coarse_masks[None, ...]`; the wrapper also passes an explicit
   `[1,H,W]` array.

The code under `3rd/SAMRefiner` is not imported or modified at runtime.
