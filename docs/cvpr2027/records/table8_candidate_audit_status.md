# Candidate-level audit status

No DCD candidate-level audit table is included here because the repository does **not** contain a DCD fixed3 candidate audit with all 16 datasets × 4 seeds.

What exists but is not suitable:

- `results/ablations/sacm_stc_topoqa_*` candidate audits: validation-only, 6 images, and from TopoQA/SSH checkpoints, not DCD fixed3.
- `candidate_contrast_summary.csv` in `results/ablations/sacm_stc*/analysis/`: header only, no rows.

Therefore these tables are intentionally absent:

- candidate-wise Dice/IoU/clDice/HD95;
- candidate pairwise disagreement;
- gating candidate selection frequency;
- candidate oracle gap.

If the paper needs a candidate-level figure, run the DCD fixed3 candidate audit on the 16 test datasets and 4 seeds first. Until then, do not put candidate-level numbers in the paper.
