# DCD ablation configs

The frozen canonical DCD main is the fixed3 800 recipe.  Use
`configs/ablations/dcd_full_800.yaml` for the one-click main + component
ablation, or `configs/ablations/dcd_ssh_full_ab.yaml` for the minimal
canonical pair.

`configs/ablations/dcd_scope_structure.yaml` is the single source of
truth for the dense-scope × structure-scope factorial.  The runner does not
contain arm names or hyperparameters; edit the YAML and rerun.

## Why these six arms

The paper claim is **all-candidate structural supervision**, so both dense
and structural scope must be varied independently:

| arm | dense Stage-2 loss | structural loss | purpose |
|---|---|---|---|
| `b0_selected_base` (S0) | deployed slot only | none | clean selected-only base |
| `b1_all_dense_base` (A0) | all four candidates | none | all-candidate dense supervision |
| `ss_selected_dense_selected_structure` (Ss) | deployed slot only | deployed slot only | structure on the deployed candidate |
| `sa_selected_dense_all_structure` (Sa) | deployed slot only | all four candidates | structure scope with selected-dense base |
| `b2_all_dense_selected_structure` (As) | all four candidates | deployed slot only | dense scope with selected structure |
| `b3_all_dense_all_structure` (Aa) | all four candidates | all four candidates | shipped full method |

The two key structural-scope contrasts are **Aa - As** and **Sa - Ss**.
Each holds dense supervision fixed and holds the total structural-loss
budget equal (the topology loss averages over its selected heads), so only
the structure scope changes.  The dense-scope contrasts are **As - Ss** and
**Aa - Sa**.  A 2x2 that omits Ss/Sa cannot answer whether the structure
scope effect depends on the all-dense base.

All arms fix Stage-1 deep supervision at 0.3, soft targets, foreground BCE
weight 2, gating deployment, the 1024 input, and the official optimizer.  The
ablation uses 300 epochs and seeds 0-3 for the fast validation pass; Aa at
300 epochs is the budget control for the main 800-epoch result.  If the
factorial pattern is promising but a key contrast is borderline, add seed 4
for the key arms rather than rerunning the whole matrix.

## Commands

### Run the factorial

The runner audits the existing checkpoints, then launches the 24 short runs.
Each run trains for 300 epochs and gets its 20-dataset evaluation and
candidate diagnostics on `best_model.pth`; the paired analysis is written
after the selected arms finish.

```bash
# server, repo root
bash scripts/run_dcd_ablation.sh --stage audit --gpus 4,5,6,7
bash scripts/run_dcd_ablation.sh --stage all --gpus 4,5,6,7
```

`--resume` is the default; pass `--no-resume` to rerun completed stages.

### Manual stages

Step 0, audit the existing 800-epoch full checkpoints (no training):

```bash
bash scripts/run_dcd_ablation.sh --stage audit --gpus 4,5,6,7
```

If the candidates are nearly identical / oracle regret is negligible, do
not spend the full matrix budget; reconsider the framing first.

Full controlled run (24 short runs = 6 arms × 4 seeds):

```bash
bash scripts/run_dcd_ablation.sh --stage all --gpus 4,5,6,7
```

Useful variants:

```bash
# only the two all-dense structural-scope arms (As/Aa) on four seeds
bash scripts/run_dcd_ablation.sh --stage all --gpus 4,5,6,7 \
  --arms b2_all_dense_selected_structure,b3_all_dense_all_structure --seeds 0 1 2 3

# inspect commands without launching anything
bash scripts/run_dcd_ablation.sh --stage all --dry-run

# re-run completed stages
bash scripts/run_dcd_ablation.sh --stage all --no-resume
```

`--resume` is on by default.  Training/eval stages source
`scripts/wandb_env.sh`; audit/diagnose/analyze do not require W&B.

## Outputs and separation from the main runs

```text
results/runs/dcd/<arm>/seed_<s>/                 # training + eval
results/runs/dcd/<arm>/seed_<s>/eval/tables/     # 20-dataset results
results/ablations/dcd/candidate_diagnostics/<arm>/seed_<s>/
results/ablations/dcd/candidate_audit/seed_<s>/  # Step 0, full runs
results/ablations/dcd/analysis/                  # arm summary + contrasts
results/ablations/dcd/config_snapshot_<stamp>.yaml
results/logs/dcd_ablation/                       # driver/child logs
```

New runs write under `results/runs/dcd/<arm>/`; archived pre-rename
800-epoch results are kept as immutable history and are not reused by new
runs. Ablation runs use distinct arm variants, and W&B uses the
`dcd_scope_structure` group, so they cannot be confused with the canonical
fixed3 800 main row.

Morning checks:

```bash
cat results/ablations/dcd/analysis/ablation_report.md
cat results/ablations/dcd/analysis/arm_summary.csv
find results/ablations/dcd/candidate_diagnostics -name summary.json | wc -l
```

The analysis report contains the arm-wise mean/std, paired seed differences,
seed-level 95% CI and win/loss, a dataset-level paired bootstrap, and
candidate-head diagnostics when present. All Dice/IoU/clDice/HD95 values are
retained; the primary endpoint is pre-specified in the YAML
(`analysis.primary_metric/primary_role`).

### Execution details

- 6 arms × 4 seeds = 24 jobs; each job runs train → 20-dataset eval →
  candidate diagnostics.
- Use `--resume` to skip completed stages; rerun the same command to fill
  missing results.
- The runner writes a timestamped driver log and config snapshot.

## Decision rules

Do not claim the all-candidate structural contribution unless **Aa - As** is
consistent on the deployed slot and, preferably, the candidate diagnostics
show a corresponding improvement.  A positive **Sa - Ss** strengthens the
claim by showing the structure-scope effect is not specific to the all-dense
base; a null Sa - Ss means the all-candidate effect is conditional and must
be phrased that way.  If Aa ~ As, the structural loss may still help
(Ss - S0 or As - A0), but it is not evidence for the all-candidate scope;
adjust the paper claim/name accordingly.  If A0 ~ S0, all-candidate dense
supervision is not supported.  A null component is reported as null, not
rescued by changing the primary endpoint or adding seeds until a p-value
appears.  Four seeds are an exploratory screen; promote only the key arms to
five seeds if the factorial pattern is promising.
