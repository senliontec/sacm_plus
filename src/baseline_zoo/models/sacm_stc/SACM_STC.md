# SACM-STC (Structure-Task Composition)

## Source

- Upstream: `kylechuuuuu/SACM`
- URL: https://github.com/kylechuuuuu/SACM
- Local reference snapshot: `3rd/SACM` (read-only third-party tree)
- Snapshot commit: `986504957e3f41f7fc02e610e2ad9ad27c43300f`
- Derived from the faithful port:
  `src/baseline_zoo/models/sacm_official/segment_anything/`
- Vendored path: `src/baseline_zoo/models/sacm_stc/segment_anything/`
- License: the upstream snapshot has no `LICENSE` file. Do not redistribute
  the vendored copy without obtaining a license from the authors.

## What this method is

SACM-STC extends the SACM prompt-free decoder by assigning the four native
Stage-2 heads to fixed geometric roles: dense mask, centerline, boundary and
normalized distance/width.  Stage-1 keeps dense deep supervision; Stage-2
uses a stable order (no gating reorder).  The **dense head is the default
deployment output**.  The centerline/width branch is an optional structural
ensemble: a post-hoc per-checkpoint calibration on the validation split
chooses between dense alone (g=0) and a fixed blend
`dense + g * (structure - dense)`; g=0 is a legal and often optimal choice.
No learned test-time gate or head selector is required for deployment.
`sacm_official` remains the main-table baseline and is not modified.

Geometry consistency is applied directly to the dense output, not only to
the auxiliary heads: Skeleton Recall, soft-clDice and a soft-contour
boundary loss supervise the dense logits.  The auxiliary heads remain
directly supervised as geometric teachers.  This targets the topology,
connectivity and boundary metrics without adding inference-time parameters.

Both additions start as neutral operations.  Zero-initialized type
embeddings (`stage2_type_emb`) let the four heads specialize without changing
their initial SAM behaviour, and a zero-initialized `stage1_prior` branch
fuses the four coarse Stage-1 masks into the Stage-2 substrate, giving Stage-1
an inference role beyond deep supervision.  Structural and geometry losses
(centerline/boundary/width, dense topology and boundary consistency) are off
for `--typed-structure-warmup-epochs` epochs and then ramp in linearly.  The
learned gate/oracle distillation remains available as an ablation path, but
deployment uses a fixed/calibrated blend or dense alone.

Width is parameterised in pixels as
`0.5 + 13.5 * sigmoid(logit)` and supervised in logit space against the
inverse-sigmoid of the GT distance transform.  The bound matches the largest
composition radius (14 px at the typed resolution), so thick roads and
membranes remain representable.  Supervising the logit keeps a constant
gradient when the head starts far from the target; the earlier
`softplus(normalized_width)` form saturated and permanently killed the width
head.  The composer detaches the four task heads for its own losses, so
composition cannot collapse a head's scale.

The optional structural ensemble uses the bounded width-based structure
prior and a **fixed or validation-calibrated** blend; the learned
global/spatial gate and GT-oracle distillation are retained only as ablation
paths, not as deployment requirements.  The boundary edge residual is not
used in deployment: GT-resolution audits showed it consistently hurt Dice.
The boundary head uses BCE plus a Dice term, because BCE alone on a
one-pixel contour collapses to a near-zero map.

## Calibration

`scripts/calibrate_sacm_stc.py` evaluates every saved checkpoint at GT
resolution with

```
dense + g * (structure - dense),   g in {0, 0.1, 0.2, 0.3, 0.4}
```

It selects the best (checkpoint, g) pair on the validation split and can
write a `calibrated_model.pth` whose `config.adapter_config.calibrated_blend_g`
is consumed by `predict_sample`.  `g=0` means the dense head alone; this is
the most common choice at the best checkpoint, while later/degraded
checkpoints often benefit from `g≈0.4`.  The learned composer LR guard still
applies when the composer is trained: generic scheduler overrides reset all
parameter groups to `--lr`, so the adapter refuses that combination.

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
11. **Head supervision** (adapter-side) — the typed default trains Stage-2
    head 0 with BCE+Dice and heads 1-3 as centerline, boundary and
    normalized distance/width; all four Stage-1 heads keep dense deep
    supervision (weight 0.3).  The legacy non-typed path additionally trains
    per-head IoU predictions (MSE); the typed run sets
    `--iou-loss-weight 0` because no test-time selector consumes them.
    Upstream trained only the gating-champion Stage-2 head and never trained
    Stage-1/IoU.
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

`baseline_zoo.adapters.sacm_stc.SACMSTCAdapter`
