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

The CLI needs the source images because SAMRefiner uses them for prompts.
It resolves them from the protocol manifest by `(dataset, stem)`:

```bash
PYTHONPATH=src python scripts/tools/apply_sam_refiner.py \
  --input results/paper/unet/seed_0/predictions \
  --output results/paper/unet/seed_0/predictions_samrefiner \
  --manifest data/sacm_3shot_seed0/protocol_manifest.csv \
  --checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --device cuda:0
```

Use `--limit 5` for a first smoke test, `--overwrite` to replace existing
outputs, and `--iterations` to control refinement passes.

For a single mask, pass `--image path/to/source.png` and point
`--input`/`--output` at files.

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
  `split-then-merge` pipeline is not exposed.
- `FastGeodis` is optional; the SciPy fallback is slower/approximate.
- No SAM weights are bundled.
