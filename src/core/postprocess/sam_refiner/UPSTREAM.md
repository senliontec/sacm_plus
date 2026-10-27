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
   uses SciPy's Euclidean distance transform. This is not the exact geodesic
   distance, but it preserves the interior-point selection used by the
   prompt generator; install `FastGeodis` for the official behavior.
2. The wrapper exposes one binary mask per call; multi-object `split-then-
   merge` and HQ-SAM are not exposed by the CLI.
3. The CLI resolves source images from the protocol manifest, because the
   prediction tree contains masks only.
4. No SAM weights are bundled. The user must pass a local SAM/HQ-SAM
   checkpoint; CPU execution is supported but slow.
5. Upstream's 2-D mask branch uses `coarse_masks[None: ,]`, which is a
   no-op slice rather than a new batch axis. The wrapper passes an explicit
   `[1,H,W]` array to `sam_refiner`; the vendored upstream file is kept
   verbatim.

The code under `3rd/SAMRefiner` is not imported or modified at runtime.
