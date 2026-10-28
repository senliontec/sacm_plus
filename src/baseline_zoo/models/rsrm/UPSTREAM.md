# RSRM upstream record

## Sources

| Component | Upstream | Commit |
|---|---|---|
| Method / SAFR / ASE / HPM | `Sparkling-Water/RSRM`, `model/rsrm.py`, `test.py`, `data/dataset.py` | `4173c9fd38b4156b9d9200dd6ef941d88eca91bd` |
| DINOv3 implementation | same repo's `dinov3/dinov3` subset (`layers/`, `models/`, `utils/`), from `facebookresearch/dinov3` | same |

Paper: "Training-free Cross-domain Few-shot Segmentation via Robust Semantic
Representation and Matching", ECCV 2026.

License: the RSRM checkout has no LICENSE file; the DINOv3 subset carries
Meta's DINOv3 License (`LICENSE_DINOV3`). Verify redistribution terms with
the authors before releasing.

## Copied and adapted

- `dinov3/layers`, `dinov3/models`, `dinov3/utils`: the minimal DINOv3
  runtime required for the ViT backbone. Internal imports were rewritten
  from `dinov3.*` to `baseline_zoo.models.rsrm.dinov3.*`. Training/eval
  harnesses and third-party packages were omitted.
- `model.py`: RSRM's forward and all helper methods reimplemented with the
  same equations: SAFR variance proportions/Fisher ratios/re-fusion, robust
  threshold search, prior-mask generation, masked support enhancement,
  multi-scale prototype matching, and region similarity. `build_rsrm`
  freezes the backbone and auto-detects the DINOv3 checkpoint format.

## Invariants kept from the official code

- Frozen DINOv3 backbone; zero trainable parameters.
- Multi-layer features via `get_intermediate_layers(..., n=range(12))`:
  block outputs (`F_sum`) and post-attention residual tensors (`F_attn`).
- SAFR: top `round(C/200)` highest-variance channels of the final feature
  define the variance proportion `r`; layers above the mean `r` are
  position-aware; between adjacent position-aware layers the minimum-`r`
  layer is selected; fusion weights are `FDR / r`, normalized; both support
  and query use the weights derived from the supports; fusion divides each
  layer by its own std before summing.
- ASE: per-support FG/BG prototypes, softmax cosine similarity ×10, the
  best threshold from 21 uniform thresholds, robust threshold = mean of
  thresholds with IoU ≥ 0.75 × best; query prior mask from averaged
  prototypes with the official FG/BG thresholds and top-k fallback; masked
  cross-attention (no projection) restricted to same-region support/query
  pairs, then `(cross + support)/2`.
- HPM: fg scales `[1, 9, -1]`, bg scales `[1, 18, -1]`; `-1` is
  pixel-level max-cosine matching, others are k-means prototypes; per-scale
  confidence = mean normalized similarity on the pseudo mask minus outside,
  divided by the std, softmax over scales; support-average of fg/bg maps;
  final `argmax(bg, fg)` and bilinear resize to the input size.
- Inputs 480 square, ImageNet normalization; support masks resized
  bilinearly and thresholded at 0.5.
- No target GT or target prompts; support-based 3-shot protocol.

## Deviations / notes

1. `fast_pytorch_kmeans` is replaced by scikit-learn `KMeans(init='k-means
   ++', n_init=1, max_iter=100, random_state=seed+call)` on CPU. The
   official GPU k-means is not a dependency of this environment; the
   clustering objective is the same and the port is deterministic.
2. The paper uses DINOv3 ViT-B/16. The local
   `checkpoints/dinov3/pytorch_model.bin` is a timm ViT-S/16 export
   (`reg_token`, `gamma_1/2`, no qkv bias). It is loaded through timm's own
   `vit_small_patch16_dinov3` architecture, and a wrapper captures the
   post-attention residual (pre-hook on `block.norm2`) to reproduce
   `F_attn`. Official Meta ViT-B/16 state dicts take the vendored
   architecture path. The config records which file is used.
3. The official test loop resizes query masks to 480 and computes metrics
   there; this port argmaxes at 480 (unchanged) and nearest-resizes the
   binary mask to the original GT for the shared evaluation contract.
4. The official benchmark samples 1200/2400 episodes from Deepglobe, ISIC,
   Chest X-ray and FSS-1000. The shared protocol instead evaluates the
   frozen 20-dataset split; no TTA is used.
5. No RSRM-trained weights exist; the training stage writes a metadata-only
   checkpoint carrying the DINOv3 provenance and support configuration.
