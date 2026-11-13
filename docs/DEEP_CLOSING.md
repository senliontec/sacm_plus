# Deep Closing post-processing

Deep Closing is integrated as an **optional post-processing refiner**, not
as a segmentation backbone.  It is applied after a model has produced a
binary mask and is particularly useful for broken thin tubular structures.

Paper: Qian Wu, Yufei Chen, Wei Liu, Xiaodong Yue, Xiahai Zhuang,
"Deep Closing: Enhancing Topological Connectivity in Medical Tubular
Segmentation", IEEE TMI, 2024.  Upstream code: `3rd/DeepClosing` (MIT).
Implementation/provenance: `src/core/postprocess/UPSTREAM.md`.

## Method kept from the paper

1. Train a MIM U-Net shape prior on **training masks only**.  Training uses
   random block masking, Recurrent TubeScaler augmentation and the
   topology-aware reconstruction loss.
2. At inference, `Deep Dilation` applies the shape prior to the predicted
   mask and thresholds it (default 0.2).
3. `Simple Component Erosion` labels the positive residue
   `dilated - prediction`, erodes simple points recursively, and keeps only
   non-simple components.  Original foreground is never removed.
4. The result is reported as `<method> + Deep Closing`.  It is not a new
   baseline architecture.

## Historical status

The Deep Closing training and post-processing scripts were removed in the
SACM-GCD cleanup.  Deep Closing is not part of the deployed SACM-GCD model.  The
design description above is retained as a historical note; no runnable
command is provided.

## Evaluation protocol

- Train shape prior on `train` masks only; use `val` for checkpoint
  selection; never train on `test`.
- Apply the same shape-prior checkpoint to base and post-processed runs;
  compare identical test sets, seeds, checkpoint and threshold.
- Report Dice/IoU **and** topology metrics (clDice, HD95, Betti, connected
  components).  Deep Closing mainly adds recall/connectivity and can
  slightly reduce precision; do not select it on test.
- Treat it as an ablation: `base` vs `base + Deep Closing`.  Apply it to
  all compared methods if it is used as a common post-processing stage;
  otherwise label it as part of the proposed method.

## Caveats

- The shape prior is trained from 18 seen-domain masks; generalization to
  unseen/novel domains is empirical and must be measured.
- It never removes original foreground, so it can add false positives in
  regions the shape prior over-dilates.  Validate the dilation threshold
  (0.2 default) and report precision/recall.
- The port uses a 512 square working resolution and nearest resizing by
  default; the official implementation uses 224 crops and sliding windows.
  See `UPSTREAM.md`.
- The upstream repository is incomplete and releases no weights; our
  checkpoint is trained from scratch on the protocol masks.
