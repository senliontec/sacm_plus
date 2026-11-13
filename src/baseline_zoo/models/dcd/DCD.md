# DCD: Deterministic Candidate Decoding

DCD is the cleaned canonical recipe: train all four Stage-2 candidates with
dense and geometry-consistent losses, then deploy one fixed candidate at
inference (no test-time selector, no fusion).

## Source

- Upstream: `kylechuuuuu/SACM`
- URL: https://github.com/kylechuuuuu/SACM
- Local reference snapshot: `3rd/SACM` (read-only third-party tree)
- Snapshot commit: `986504957e3f41f7fc02e610e2ad9ad27c43300f`
- Derived from the faithful port:
  `src/baseline_zoo/models/sacm/segment_anything/`
- Vendored path:
  `src/baseline_zoo/models/dcd/segment_anything/`

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

The canonical 800-epoch fixed3 result is
**56.66 / 43.05 / 63.42 / 64.28** (Dice / IoU / clDice / HD95, 20 datasets,
4 seeds), versus the official SACM code at
**55.40 / 41.85 / 60.34 / 67.04**.  This is the frozen main result:
`ssh` training with gating + auxiliary fusion, followed by inference-only
fixed candidate 3.

The 200-epoch rerun is
**56.83 / 43.17 / 64.21 / 64.72** against the same official tables
(Dice +1.43, IoU +1.32, clDice +3.87, HD95 -2.32).  Relative to the
canonical 800-epoch run the difference is Dice +0.17, IoU +0.13,
clDice +0.79, HD95 +0.44 (slightly worse), all within seed variance.
This is a budget-reproduction check, not a replacement for the 800-epoch
main result: the cosine schedule depends on the total epoch count, the
selected checkpoint differs, and the two runs are not a controlled
budget ablation.  The one-click runner below reproduces the 800-epoch
main result and trains the component ablation from the same recipe.

## Frozen canonical state

DCD is frozen at the fixed3 800 recipe.  Do not add new experimental
branches or flags to this adapter; new research directions must use a
separate adapter/model.  The only supported switches are the kept
ablations below, used by `configs/ablations/dcd_full_800.yaml` and the
scope factorial.

## Kept ablations

- `--specialized-heads 0`: native four-head control.
- `--head-fusion 0`: no-fusion training.
- `--cl-dice-weight 0 --skeleton-recall-weight 0
  --specialized-topology-weight 0`: no-topology control.
- `--stage2-loss-scope` / `--topology-head`: dense/structure scope
  factorial.

The canonical fixed3 800 pair is in
`configs/ablations/dcd_ssh_full_ab.yaml`; the frozen consolidated runner
config is `configs/ablations/dcd_full_800.yaml` (same `ssh` arm plus the
three component-ablation arms); the older lean ablation is in
`configs/ablations/dcd_fixed3_lean.yaml`; the scope factorial is in
`configs/ablations/dcd_scope_structure.yaml`.

## Fixes relative to the released official code

These are paper-faithful corrections, not new architecture:

1. SAM normalization before the frozen encoder.
2. Learned `no_mask_embed` prompt anchor instead of an all-zero dense map.
3. Per-block Adapter-E LayerNorm (paper Eq. 3).
4. Eq. 6 fusion as Softmax(FFN(Concat(z_l))) with per-layer projections.
5. Eq. 10 descriptor reorder before Stage-2 heads.
6. Per-head IoU prediction bound to the ranked Stage-2 slot.
7. Batch-repeat fix (`divmod` instead of `B*B`).
8. `multimask_output=False` returns the Stage-1 gating champion; the
   untrained IoU head is not used as a selector.
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

One-click canonical fixed3 800 run + component ablation:

```bash
bash scripts/run_dcd_full_800.sh
```

This reproduces the frozen main result and trains the kept ablation.  It
trains four arms for seeds 0-3 (`ssh` canonical, `nofusion`,
`native_control`, `notopo`), derives fixed candidate 3 for every arm,
evaluates all arms on the 20 test datasets, and adds two eval-only
deployment references (DCD gating+fusion and native gating).  The `ssh`
arm is exactly the previous fixed3 800 recipe (gating + fusion training,
inference-only fixed candidate 3).  It writes:

```text
results/runs/dcd_full_800/<arm>/seed_<s>/
results/runs/dcd_full_800/fixed3/
results/ablations/dcd_full_800/summary.{md,csv,json,html}
results/ablations/dcd_full_800/fixed3_vs_sacm/summary.md
```

The 300-epoch dense/structure scope factorial remains a separate study
(`configs/ablations/dcd_scope_structure.yaml`) and is not mixed into this
800-epoch table.

The 200-epoch script is only a fast budget-reproduction check:

```bash
bash scripts/run_dcd_expert_200.sh
```

It trains four seeds, derives fixed candidate 3, evaluates 20 datasets, and
summarizes against the official tables.  Use `DRY_RUN=1` for a preflight
check.

## Runtime consumer

`baseline_zoo.adapters.dcd.DCDAdapter`
