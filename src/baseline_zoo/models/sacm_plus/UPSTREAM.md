# SACM+ (fixed official SACM)

## Source

- Upstream: `kylechuuuuu/SACM`
- URL: https://github.com/kylechuuuuu/SACM
- Local reference snapshot: `3rd/SACM` (read-only third-party tree)
- Snapshot commit: `986504957e3f41f7fc02e610e2ad9ad27c43300f`
- Derived from the faithful port:
  `src/baseline_zoo/models/sacm_official/segment_anything/`
- Vendored path: `src/baseline_zoo/models/sacm_plus/segment_anything/`
- License: the upstream snapshot has no `LICENSE` file. Do not redistribute
  the vendored copy without obtaining a license from the authors.

## Why this copy exists

`sacm_official` preserves upstream behaviour verbatim (including its bugs)
and remains the main-table SACM baseline.  `sacm_plus` fixes the correctness
defects so that the paper's algorithm can be evaluated as intended.  The two
checkpoints are not interchangeable (`runtime.py` rejects a mismatched
`adapter` id).

## Fixes (all differ from `sacm_official`)

1. **SAM normalization** — the adapter scales `[0,1]` to `[0,255]` and calls
   `Sam.preprocess`.  Upstream fed `[0,1]` straight into the frozen encoder.
2. **Prompt-free anchor** — `prompt_encoder(None, None, None)` supplies the
   learned `no_mask_embed` dense anchor instead of upstream's all-zero map.
3. **Per-block Adapter-E norm** — `adapter_norms[i]` per block (paper Eq. 3)
   instead of one shared `adapter_norm` (`modeling/image_encoder.py`).
4. **Eq. 6 fusion weights** — pooled layer descriptors → `Linear(D→32)` →
   concat → `LayerNorm` → `Linear(L*32→128)` → `GELU` → `Linear(128→L)` →
   softmax.  Upstream applied `Sigmoid` before softmax, compressing all 24
   weights into roughly `[0.039, 0.106]`.  The per-layer low-rank projection
   is a tractable approximation of the paper's `L*D` concatenation.
5. **Eq. 10 descriptor reorder** — Stage-2 `MLP2_j` consumes the descriptor
   ranked at position `j`; the Stage-2 masks are produced directly in ranked
   order and are **not** gathered again.  Upstream applied Stage-2 heads to
   unreordered tokens and gathered outputs, which pairs each descriptor with
   the wrong head.
6. **Per-head IoU prediction** —
   `iou_pred[j] = iou_prediction_head(sorted_tokens[:, j])[:, j]`; slot `j`
   is permanently bound to Stage-2 head `j`.  Upstream used the IoU token
   and never trained the head.
7. **Batch repeat fix** — `divmod(tokens.shape[0], image_batch)`; upstream
   repeated each image `tokens.shape[0]` times (`B*B` for `B > 1`).
8. **multimask semantics** — `multimask_output=False` returns the top-1
   candidate by predicted IoU; `True` returns all four ranked candidates.
   Upstream sliced `[1:]`, dropping the best candidate.
9. **Predictor fix** — `SamPredictor` unpacks the adapter encoder tuple,
   stores `adapter_features`, forwards them to the decoder, and clears them
   in `reset_image`.
10. **Dead code removed** — encoder `adapter_fusion` (about 2.1M unused
    trainable parameters), decoder `gating_mlp`, `adapter_stack`.
11. **Head supervision** (adapter-side) — all four Stage-2 heads, all four
    Stage-1 heads (deep supervision, weight 0.3) and the per-head IoU
    predictions (MSE, weight 1.0) are trained.  Upstream trained only the
    gating-champion Stage-2 head and never trained Stage-1/IoU.
12. **Binary GT masks** — the adapter bilinearly resizes then binarizes at
    0.5; upstream trained on soft resized masks.
13. **Weight decay** — the adapter excludes LayerNorm/bias parameters;
    upstream applied 0.01 to every parameter.

## Preserved upstream behaviour (deliberate)

- `Resize(1024, 1024)` data path (including aspect-ratio distortion), no
  augmentation.
- Adapter and decoder learning rate 3e-4, cosine to 1e-6, gradient clipping
  1.0.
- BCE 0.7 + Dice 0.3, frozen image encoder except adapters, frozen prompt
  encoder, all mask-decoder parameters trainable.
- Shared 800-epoch protocol budget (upstream default 60).
- GT-resolution prediction export.
- Adapter-E input is `LN(block output)` (paper Eq. 3); Fig. 3's
  concatenation is not used, matching upstream.

## Not supported

- ONNX export (`utils/onnx.py`): the rank `argsort`/`gather` and adapter
  features are not exported.
- `SamAutomaticMaskGenerator` is kept for API parity but is not used by the
  baseline; `multimask_output=True` now returns four candidates.

## Runtime consumer

`baseline_zoo.adapters.sacm_plus.SACMPlusAdapter`
