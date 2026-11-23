# WPRF-SAM baseline package

Status: **available baseline**, prompt-free supervised group.  It is not a
proposed method and it did not pass the champion promotion gate.

## What this package owns

`baseline_zoo.models.wprf_sam` physically owns:

- its SAM ViT-L code (`segment_anything/`),
- the copied GFA/BCTD four-candidate scaffold (`base.py`, `adapter.py`,
  `tcsu.py`, `boundary_loss.py`, `topology_losses.py`, `decoding.py`,
  `adaptive_skel.py`, `fusion.py`, `cape.py`, `atc.py`, `registry.py`),
- the WPRF affinity head, support graph, Max-Min DP and losses
  (`wprf.py`),
- the runtime adapter (`wprf_adapter.py`).

It imports only the shared `baseline_zoo.data` / `baseline_zoo.runtime`
harness.  It does **not** import `methods.*`, `baseline_zoo.models.cosna`,
or any other model package at runtime.

## Upstream and provenance

Method: "Widest-Path Reachability Fields for Connectivity-Preserving
Slender Structure Segmentation" (2026).  The paper's code link
`github.com/mituan-ai/WPRF` returns 404 and GitHub search finds no
repository, so this is an independent implementation from the paper.  The
underlying four-candidate SAM scaffold is the SACM/BCTD family, physically
copied (not imported).

The experimental source package is `src/methods/wprf_sam/`.  The baseline
was produced by a physical copy; only import paths and docstrings changed
(see audit below).

## Port audit

- `diff -rq` against `src/methods/wprf_sam/`: only four files differ.
  - `__init__.py`: package docstring.
  - `adapter.py`: one docstring path (`baseline_zoo.models.cosna` ->
    `wprf_sam`).
  - `base.py`: the same docstring path.
  - `wprf_adapter.py`: import header (`methods.wprf_sam.*` ->
    `baseline_zoo.models.wprf_sam.*`; removed the `sys.path` bootstrap).
- `wprf.py`, `segment_anything/`, `tcsu.py`, `boundary_loss.py`,
  `topology_losses.py`, `decoding.py`, `adaptive_skel.py`, `fusion.py`,
  `cape.py`, `atc.py`, `registry.py`: byte-identical.
- Inference parity: the seed-0 WPRF v3 checkpoint
  (`results/runs_local/wprf_v3_s0_200/best_model.pth`) was run through the
  baseline `run_one --stage predict` on DRIVE; all 16 exported masks are
  byte-identical to the experimental-package predictions.
- Tests: `tests/test_wprf_sam_baseline.py` (registry/independence, parser
  defaults, affinity-head shape, batch coverage, Max-Min propagation)
  pass.
- Smoke: `baseline_zoo.run_one --model wprf_sam --stage train` with 2
  images / 1 val image builds from the official SAM ViT-L checkpoint and
  runs end to end.
- `WPRFLoss` averages the edge and reachability terms over the batch
  dimension; the canonical recipe remains batch=1 (the local 11 GiB card
  cannot fit ViT-L batch 2 at 1024, but the loss is unit-tested at B=2).

## Canonical recipe (V3)

The config `configs/models/wprf_sam.yaml` is the canonical baseline
recipe:

- scaffold: copied GFA/BCTD four-candidate decoder, gating training,
  fixed slot-3 deployment, RCD/AS inference, no TCSU;
- WPRF: `--wprf-edge-weight 0.0 --wprf-reach-weight 0.5
  --wprf-warmup-epochs 10 --wprf-stride 8 --wprf-sources 16`;
- 200 epochs, the shared structure-preserving augmentation and copy-paste.

The class defaults are the paper's `edge 1.0 / reach 1.0 / warmup 0`;
the config is the selected baseline recipe.  The 0.5/0.5 and 1.0/1.0
ablations are in the round report.

## Results (4 seeds, dataset-equal macro)

| benchmark | Dice | IoU | clDice | HD95 |
|---|---:|---:|---:|---:|
| 20 datasets | 61.949 | 48.331 | 69.673 | 59.723 |
| fixed 12 subset | 64.120 | 50.875 | 73.110 | 57.706 |

For comparison, CoSNA is 62.09 / 48.42 / 70.21 / 59.07 (20) and
64.61 / 51.40 / 73.63 / 56.99 (12).  WPRF-SAM does not beat the champion
on all four and was **not promoted**.  The seed-0 4/4 was seed luck; see
`results/reports/wprf_sam_final_2026-11-23.md`.

Known failure: TopoMortar raw Stage-2 probabilities collapse on roughly a
quarter of images (reach-only mitigates but does not fix it).  Full
per-dataset and ablation data are in the round report and
`results/reports/iccv_model_search_2026-11-21.md`.

## Usage

```bash
# train one seed
python -m baseline_zoo.run_one --model wprf_sam --stage train \
  --data-root data/sacm_3shot_seed0 --workdir results/runs/wprf_sam/default/seed_0 \
  --epochs 200 --seed 0 --device cuda:0 --use_wandb true

# predict / evaluate
python scripts/pipeline/run_baseline_test.py --method wprf_sam \
  --protocol data/sacm_3shot_seed0/protocol_manifest.csv --seed 0 \
  --checkpoint results/runs/wprf_sam/default/seed_0/best_model.pth \
  --data-root data/sacm_3shot_seed0 \
  --pred-root results/runs/wprf_sam/default/seed_0/predictions \
  --paper-root results/runs/wprf_sam/default/seed_0/eval \
  --device cuda:0 --use_wandb true --skip-visualize

# one click (seeds 0-3 train then eval)
bash scripts/run_wprf_sam_baseline_seeds03_200.sh
```

## Caveats

- No official WPRF code; the implementation follows the paper with the
  documented deviations (stride 8 at 1024, 16 sources, no
  bottleneck-aware segmentation term, positive-bias init/BCE clamping).
- The V3 recipe was selected after a seed-0 test-failure diagnosis; its
  4-seed result is not a clean pre-registered test (disclosed in the
  report).
- WPRF pair sampling uses an unseeded NumPy RNG (as in the experimental
  source), so exact training reruns are not bit-identical; the recorded
  4-seed numbers were produced with this behavior.
- All inference constants (RCD tile/stride, AS, close2, threshold 0.5)
  are inherited from the GFA control; no TTA is used.
- The canonical checkpoint is the official SAM ViT-L initialization plus
  source-only training on the 18-image 3-shot split.
