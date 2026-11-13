# SACM-GCD (cleaned canonical recipe)

## Source

- Upstream: `kylechuuuuu/SACM`
- URL: https://github.com/kylechuuuuu/SACM
- Local reference snapshot: `3rd/SACM` (read-only third-party tree)
- Snapshot commit: `986504957e3f41f7fc02e610e2ad9ad27c43300f`
- Derived from the faithful port:
  `src/baseline_zoo/models/sacm_official/segment_anything/`
- Vendored path:
  `src/baseline_zoo/models/sacm_gcd/segment_anything/`

## Canonical recipe (current best)

Training:

1. All four native Stage-2 candidates receive BCE + Dice on the dense mask.
2. All four Stage-1 coarse heads receive dense deep supervision (weight 0.3).
3. Soft-clDice (0.1) and Skeleton Recall (0.1) are applied to all four
   Stage-2 heads.
4. The three auxiliary Stage-2 heads receive role losses:
   head 1 soft-clDice, head 2 boundary-weighted BCE+Dice, head 3
   recall-heavy Tversky.  All heads keep the dense loss.
5. A zero-initialised linear fusion is trained as an auxiliary output
   (`--head-fusion 1 --fusion-loss-weight 1.0`).
6. Training uses the Stage-1 gating order (`--head-select gating`).
7. Optimizer/schedule follow the official recipe: AdamW 3e-4, weight decay
   0.01, cosine to 1e-6, gradient clipping 1.0, BCE 0.7 + Dice 0.3,
   foreground-weighted BCE (pos_weight 2), soft bilinear targets.

Deployment:

- Derive a checkpoint with
  `scripts/pipeline/make_fixed3_checkpoint.py --deployment-head 3`.
  The derived config is `head_select=fixed`, `deployment_head=3`,
  `head_fusion=0`; no weight is retrained.
- Predictions are exported at GT resolution and thresholded after sigmoid
  (> 0.5).

The historical 800-epoch result is
**56.66 / 43.05 / 63.42 / 64.28** (Dice / IoU / clDice / HD95, 20 datasets,
4 seeds), versus the official SACM code at
**55.40 / 41.85 / 60.34 / 67.04**.

## Kept ablations

- `--specialized-heads 0`: native four-head control.
- `--head-fusion 0`: no-fusion training.
- `--cl-dice-weight 0 --skeleton-recall-weight 0
  --specialized-topology-weight 0`: no-topology control.
- `--stage2-loss-scope` / `--topology-head`: dense/structure scope
  factorial.

The canonical 800-epoch pair is in
`configs/ablations/sacm_gcd_ssh_full_ab.yaml`; the lean ablation is in
`configs/ablations/sacm_gcd_fixed3_lean.yaml`; the scope factorial is in
`configs/ablations/sacm_gcd_scope_structure.yaml`.

## Fixes relative to the released official code

These are paper-faithful corrections, not new architecture:

1. SAM normalization before the frozen encoder.
2. Learned `no_mask_embed` prompt anchor instead of an all-zero dense map.
3. Per-block Adapter-E LayerNorm (paper Eq. 3).
4. Eq. 6 fusion as Softmax(FFN(Concat(z_l))) with per-layer projections.
5. Eq. 10 descriptor reorder before Stage-2 heads.
6. Per-head IoU prediction bound to the ranked Stage-2 slot.
7. Batch-repeat fix (`divmod` instead of `B*B`).
8. `multimask_output=False` returns the top-IoU candidate instead of
   dropping the gating champion.
9. All four Stage-1 and Stage-2 heads are supervised; the official code
   trains only the gating champion.
10. LayerNorm/bias parameters are excluded from weight decay.
11. Predictions are thresholded after sigmoid at GT resolution.

## Removed experiments

The following branches were physically removed because they are not part of
the canonical model or its kept ablations:

- STC structural task composition (`stc.py`).
- Typed Stage-2 heads / composer (`typed.py`).
- TopoQA quality head and Pareto arbitration (`topoqa.py`).
- TPD / POMP prompts and the topology-quality selector (`tpd.py`).
- IoU-based selection, `gating_soft`/`mean` selection, stage-1 fusion,
  polarity augmentation, and quality-based refinement.

Historical results for those branches remain under `results/` and are not
part of the released recipe.

## Run

One-click 200-epoch verification:

```bash
bash scripts/run_sacm_gcd_expert_200.sh
```

The script trains four seeds, derives the fixed candidate-3 checkpoint,
evaluates 20 datasets, and summarizes against the official tables.  Use
`DRY_RUN=1` for a preflight check.

## Runtime consumer

`baseline_zoo.adapters.sacm_gcd.SACMGCDAdapter`
