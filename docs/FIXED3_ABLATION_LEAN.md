# fixed3 Lean Ablation — Run Sheet & Paper Table

One page. **4 seeds × 20 datasets × 800 epochs.** Main paper uses one compact
table; per-role/per-dataset and P0 head audit go to supplementary.

## 0. One-click run

```bash
# server, repo root; trains, derives H3, evaluates, writes summary
bash scripts/run_sacm_stc_fixed3_lean.sh
# inspect without launching anything
bash scripts/run_sacm_stc_fixed3_lean.sh --dry-run
```

Outputs:

- `results/ablations/sacm_stc_fixed3_lean/summary.md`
- `results/ablations/sacm_stc_fixed3_lean/summary.csv`
- `results/ablations/sacm_stc_fixed3_lean/summary.html`

Pull the completed ablation back to the local repo:

```bash
# local repo root; enter the server password once if prompted
bash scripts/pull_fixed3_lean_results.sh
bash scripts/pull_fixed3_lean_results.sh --dry-run
```

The pull includes the four new eval tables plus the existing fixed3 eval,
skips predictions/checkpoints by default, and exits non-zero if any seed lacks
a 20-dataset `all_metrics.csv` or the summary is missing.

## 1. Paper table (fill from 4-seed `all_metrics.csv`)

| Method | Dice | IoU | clDice | HD95 | Role in the paper |
|---|---:|---:|---:|---:|---|
| **fixed3** | 56.66 | 43.05 | 63.42 | 64.28 | full method (existing 4-seed eval) |
| **gating + fusion** | — | — | — | — | deployment control; existing SSH checkpoint, eval only |
| **w/o fusion training** | — | — | — | — | fusion auxiliary ablation; derive H3 |
| **native four-head control** | — | — | — | — | system control; eval as-is |
| **w/o topology loss** | — | — | — | — | clDice/skeleton supervision ablation; derive H3 |

Use All-role macro means. Do not add a second per-role ablation table to the
main paper.

## 2. Train the three new arms

Config: `configs/ablations/sacm_stc_fixed3_lean.yaml`

```bash
# server, repo root; 12 jobs on 8 GPUs, round-robin
bash scripts/run_sacm_stc_ablation.sh \
  --config configs/ablations/sacm_stc_fixed3_lean.yaml \
  --stage train \
  --gpus 0,1,2,3,4,5,6,7 \
  --seeds 0 1 2 3
```

Training uses canonical `head-select=gating`; this is intentional. The
`nofusion` and `notopo` rows are deployed with **fixed H3**, not with the
checkpoint's gating selector.

**Do not use `--stage eval` for `nofusion` or `notopo`.** It evaluates the
checkpoint's gating deployment and would not match the fixed3 row.

## 3. Evaluate fixed H3 for nofusion / notopo

For each arm in `nofusion notopo`, each seed 0–3:

```bash
arm=nofusion
seed=0
gpu=0
run_root="results/runs/sacm_stc_fixed3_lean/${arm}/seed_${seed}"

python scripts/pipeline/make_fixed3_checkpoint.py \
  --input "${run_root}/best_model.pth" \
  --output "${run_root}/best_model_fixed3_nofusion.pth" \
  --deployment-head 3

python scripts/pipeline/run_baseline_test.py \
  --method sacm_stc \
  --protocol "data/sacm_3shot_seed${seed}/protocol_manifest.csv" \
  --seed "${seed}" \
  --variant "${arm}_fixed3" \
  --checkpoint "${run_root}/best_model_fixed3_nofusion.pth" \
  --pred-root "${run_root}/predictions_fixed3" \
  --paper-root "${run_root}/eval_fixed3" \
  --device "cuda:${gpu}" \
  --skip-visualize
```

Repeat for `seed=1,2,3` and for `arm=notopo`. Keep the derived checkpoint and
its `eval_fixed3/tables/all_metrics.csv`.

## 4. Evaluate native control and the two existing rows

`native_control` is evaluated as-is (gating, no fusion). It is a **system
control**, not a strict one-factor role-head ablation: fixed3 differs from it
in both specialized heads and training fusion. Use it to bound the recipe,
not to claim a clean role-head effect. A strict role-head control would need
native heads evaluated with fixed H3 in supplementary.

```bash
# repeat for seed 0-3; use the checkpoint's existing config
python scripts/pipeline/run_baseline_test.py \
  --method sacm_stc \
  --protocol data/sacm_3shot_seed0/protocol_manifest.csv \
  --seed 0 \
  --variant native_control \
  --checkpoint results/runs/sacm_stc_fixed3_lean/native_control/seed_0/best_model.pth \
  --pred-root results/runs/sacm_stc_fixed3_lean/native_control/seed_0/predictions \
  --paper-root results/runs/sacm_stc_fixed3_lean/native_control/seed_0/eval \
  --device cuda:0 \
  --skip-visualize
```

- **fixed3**: read the existing `results/runs/sacm_stc_fixed3_800/eval/seed_*/`.
- **gating + fusion**: evaluate the existing SSH checkpoint
  `results/runs/sacm_stc_ssh_full_ab/ssh/seed_*/best_model.pth` as-is; no
  retraining.
- **official**: read the existing `results/runs/sacm_official/default/seed_*/`.
- **P0 H0/H1/H2/H3**: optional supplementary head audit; derive each head with
  `make_fixed3_checkpoint.py --deployment-head 0/1/2/3`. Not part of the main
  paper table.

## 5. Run manifest

| Arm | Training | Evaluation | Seeds | Datasets |
|---|---|---|---|---|
| fixed3 | existing | existing | 0–3 | 20 |
| gating + fusion | none | existing SSH ckpt | 0–3 | 20 |
| nofusion | 800 epochs | derived fixed H3 | 0–3 | 20 |
| native_control | 800 epochs | checkpoint as-is | 0–3 | 20 |
| notopo | 800 epochs | derived fixed H3 | 0–3 | 20 |

## 6. Decision rules

- **nofusion**: keep fusion only if it improves Dice or clDice by a
  pre-specified meaningful margin; otherwise use the simpler no-fusion method.
- **native_control**: if it beats fixed3 on ≥3 metrics, do not claim
  “role heads improve everything”; shrink the claim to topology/clDice.
- **notopo**: if clDice drops by ≥1 pp, topology supervision is a supported
  contribution; if <0.5 pp, report it as neutral.
- Report all four metrics, 4-seed mean ± std, W/T/L, and per-role results in
  supplementary. Do not tune on test or select seeds/datasets.

## 7. Caveats

- p-values are descriptive Wilcoxon over 20 datasets; seeds are not
  independent test images.
- Novel macro is sensitive to TopoMortar's known GT caveat.
- HD95 is lower-better; do not call a small difference “significant” without
  the paired table.
- This sheet uses the server 4-seed scope only. Do not mix in seed0 or
  300-epoch diagnostics.
