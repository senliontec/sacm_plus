# DCD ablation on the 16-dataset working subset

Four seeds × 16 datasets; All-role macro. Deltas are relative to fixed3. HD95 lower is better. Rows are the actual lean ablation arms; no p-values are encoded in this table.

| Arm | Dice | IoU | clDice | HD95 |
|---|---|---|---|---|
| DCD fixed3 | 62.07 ± 0.89 (+0.00) | 48.10 ± 0.84 (+0.00) | 68.75 ± 0.86 (+0.00) | 45.84 ± 1.35 (+0.00) |
| gating + fusion | 61.34 ± 0.71 (-0.73) | 47.44 ± 0.60 (-0.67) | 67.33 ± 0.76 (-1.42) | 45.75 ± 2.22 (-0.09) |
| w/o fusion training | 62.05 ± 0.53 (-0.01) | 48.09 ± 0.53 (-0.02) | 68.55 ± 0.67 (-0.20) | 45.55 ± 1.81 (-0.28) |
| native four-head control | 61.20 ± 0.70 (-0.86) | 47.29 ± 0.66 (-0.82) | 67.26 ± 1.78 (-1.49) | 46.82 ± 2.80 (+0.98) |
| w/o topology loss | 62.12 ± 1.41 (+0.05) | 48.32 ± 1.34 (+0.22) | 68.47 ± 1.40 (-0.28) | 45.88 ± 1.84 (+0.05) |
| SACM official | 60.11 ± 0.82 (-1.96) | 46.32 ± 0.83 (-1.79) | 65.00 ± 1.76 (-3.75) | 50.07 ± 3.85 (+4.23) |
