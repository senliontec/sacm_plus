# CURRENT_BEST

**Current champion:** BCTD-SAM + AS

Updated: 2026-11-18

| Metric | 20 datasets | 12-dataset subset |
|---|---:|---:|
| Dice | 61.93 | 64.57 |
| IoU | 48.29 | 51.37 |
| clDice | 70.14 | 73.63 |
| HD95 (lower is better) | 59.23 | 57.13 |

4 seeds, 3-shot protocol, dataset-equal macro. No target labels/supports, no
test-time tuning.

## Canonical paths

- Method tree: `src/methods/topostrand/`
- Independent baseline port: `src/baseline_zoo/models/bctd/`
- Adapter: `src/baseline_zoo/adapters/bctd.py`
- Config: `configs/models/bctd.yaml`
- Checkpoints: `results/runs/topostrand/bctd200/seed_{0..3}/best_model_fixed3.pth`
- Report: `results/reports/topostrand_bctd200_4seed_20.md`
- Audit: `results/reports/bctd_baseline_port_audit_2026-11-18.md`
- Visualization: `results/visualizations/bctd_seed0_20dataset_2026-11-18/index.html`

## Standing research contract

`模型迭代.MD` at the repository root is the standing research contract. When the user says "读模型迭代.MD" or "按模型迭代.MD 执行", read it first, then resolve `CURRENT_BEST.json`, then follow its constraints.

## How "current best" is resolved

The JSON file `CURRENT_BEST.json` is the single source of truth. When the user
says "当前最强模型", "当前冠军", "CURRENT_BEST", or "champion", read the JSON
first and use the champion, metrics, code paths, and caveats recorded there.
Do not rely on memory or old model names.

## Promotion rule

A challenger becomes champion only after a 4-seed, 20-dataset run under the
strict 3-shot protocol where all four macro metrics are strictly better than
the current champion's 4-seed macro; at least 3/4 seeds must be not worse on
Dice/IoU/clDice, and HD95 must be not worse in at least 3/4 seeds. Selection
must be validation-only. A 3/4 win is recorded as a challenger, not champion.

## Metric leaders

- Dice / IoU / clDice: BCTD-SAM + AS
- HD95: TopoStrand-SAM base (58.76), but it is not the overall champion.

See `CURRENT_BEST_HISTORY.md` for the change log.
