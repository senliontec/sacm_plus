# CVPR 2027 DCD records

Generated: 2026-11-15 14:42 UTC

This directory contains paper-ready data tables for the **16-dataset working subset**:

`CHASEDB1, CHCrack5k, CREMI, CrackTree200, DCA1, DRIVE, DeepCrack, FIVES, HRF, LES-AV, OCTA-500, ROAD, STARE, TopoMortar, WIRE, XCAD`

Excluded from the original 20: `CORN, TYRE, DeepGlobe, LEAF`.

## Important caveats

- The 16-dataset subset is a **user-specified working subset**, not the full benchmark. Keep the full 20-dataset results in the appendix.
- Main results use **4 seeds × 16 datasets**, dataset-equal macro, and the same protocol/checkpoints as the DCD report.
- Dice/IoU/clDice are percentages; HD95 is pixels, lower is better.
- `±` is the sample std across the 4 training seeds, not across test images.
- p-values in `table4` are descriptive paired Wilcoxon over 16 datasets; they are not corrected for multiple comparisons.
- The DCD ablation has a mixed result: fixed3 improves clDice over gating+fusion, but HD95 is slightly worse; `nofusion` is close to fixed3. Do not describe fixed3 as universally superior.
- No DCD candidate-level audit exists yet; see `table8_candidate_audit_status.md`.

## File index

| File | Content |
|---|---|
| `table0_subset_16.csv/.md` | Included/excluded datasets and roles |
| `table1_main_macro_16.csv/.md` | 11-model macro table with ranks and seed std |
| `table2_per_dataset_16.csv` | Long per-dataset table: all models × 4 metrics × ranks |
| `table2b_dcd_vs_sacm_16.md` | Readable DCD vs SACM vs SAM-OCTA per-dataset table |
| `table3_roles_16.csv/.md` | Seen / Unseen / Novel / All macro |
| `table4_win_loss_vs_sacm_16.csv/.md` | W/T/L and paired Wilcoxon versus SACM |
| `table5_rank_counts_16.csv/.md` | Top-1 / Top-3 counts |
| `table5b_dataset_champions_16.csv` | Per-dataset champion and DCD rank/value |
| `table6_ablation_16.csv/.md` | fixed3 / gating / nofusion / native / notopo / official |
| `table6b_ablation_per_dataset_16.csv` | Per-dataset ablation means |
| `table6c_ablation_paired_deltas_16.csv` | Paired DCD-vs-control deltas and 95% CI over 4 seeds |
| `table7_dataset_stats_16.csv/.md` | Protocol counts, image sizes, foreground ratio |
| `table8_candidate_audit_status.md` | What candidate-level evidence is missing |
| `table9_source_manifest.csv` | Source result paths for reproducibility |
| `validation_report.md` | Independent check of generated numbers against raw tables |

## Main result snapshot

DCD fixed3 macro on the 16 subset: **62.07 / 48.10 / 68.75 / 45.84**.

All-role ranks are recomputed on this subset; use `table1_main_macro_16.md` rather than copying 20-dataset numbers.
