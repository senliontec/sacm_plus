# Validation report

Checked 0 discrepancy/ies against raw per-seed metric tables and protocol manifest.

Status: **PASS**

Validated:
- table1 macro/mean/std/rank
- table2 per-dataset mean/std/rank
- table3 role mean/std
- table4 W/T/L and mean delta
- table5 top1/top3 counts
- table6 ablation mean/std
- table7 train/val/test counts


Additional checks:
- table4 Wilcoxon p-values matched recomputation (displayed as `<0.001` when below 0.001).
- table7 roles, test counts, and image dimensions matched the protocol manifest.

Correction note:
- The first generated `table7` counted only `split == "test"`, but the protocol uses `seen_test`, `unseen_test`, and `novel_test`. The test column was corrected before this validation and now matches the protocol counts (e.g., CHCrack5k 990, TopoMortar 350).
