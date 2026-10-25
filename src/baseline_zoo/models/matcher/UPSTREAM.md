# Matcher upstream record

## Source

- Repository: `aim-uofa/Matcher`
  (https://github.com/aim-uofa/Matcher)
- HEAD: `d59b6c761f427d52b1ba00132a6b3488f169d039` (2025-12-03).
- Local reference: `3rd/Matcher`.
- License: 2-Clause BSD (Copyright (c) 2023 Advanced Intelligent
  Machines (AIM)), copied as `LICENSE`.

## Copied files

- `matcher/Matcher.py` — `Matcher`, `RobustPromptSampler`, official builder
- `matcher/k_means.py` — k-means++ prompt clustering
- `segment_anything/` — Matcher's modified SAM fork, including the
  automatic mask generator with selected point/box prompts
- `dinov2/` — the DINOv2 model/layer/data subset, including the ViT-L/14
  builder and ImageNet transform
- `LICENSE`

The Matcher dataset/CLI/evaluation/visualization packages and the
Semantic-SAM path are not vendored.  The method itself is inference-only.

## Modifications

- `Matcher.py` imports are package-relative for the SAM fork, k-means and
  DINOv2; DINOv2's internal absolute imports are rewritten to
  `baseline_zoo.models.matcher.dinov2...`.
- The SAM fork's `predictor.py` absolute `segment_anything.modeling` import
  is made package-relative, and its checkpoint loader uses
  `torch.load(..., map_location="cpu", weights_only=False)`.
- The upstream `Matcher.extract_img_feats` re-encoded all support images
  for every target.  The port caches normalized support features after
  `set_reference`; this is a throughput change only (support features are
  deterministic in eval mode).
- The removed NumPy alias `np.int` in `RobustPromptSampler.is_in_mask` is
  replaced by `int` (the upstream pins NumPy 1.22; the project uses
  NumPy 2.x).
- The Matcher CLI, COCO/FSS dataset classes, evaluator, visualizer and
  Semantic-SAM branch are replaced by the shared baseline adapter/runtime.

## Official recipe preserved by the adapter

- DINOv2 ViT-L/14 (`img_size=518`, patch 14, ImageNet normalization) and
  SAM ViT-H (`sam_vit_h_4b8939.pth`).
- Support/target images are square-resized to 518; support masks are
  nearest-resized and average-pooled by the DINOv2 patch size.
- DINOv2 patch matching, k-means++ (`num_centers=8`), point sampling range
  `(4,6)`, 30 sampling iterations.
- Official `SamAutomaticMaskGenerator` settings: `points_per_side=64`,
  `points_per_batch=64`, `pred_iou_thresh=0.88`,
  `stability_score_thresh=0.95`, `sel_output_layer=3`, dense/mm output off.
- Official score filters: EMD/purity/coverage off, `topk_scores_threshold`
  0.7, `num_merging_mask=10`, alpha 1.0, beta 0.0, exp 0.0.
- Matcher natively accepts n-shot supports; the three project train shots
  are passed for seen datasets; unseen/novel datasets use a structurally
  matched donor domain.
- No optimizer, loss or method-specific checkpoint; training-free.

## Deviations in this port

- The official output is 518×518; the port bilinearly upsamples the binary
  merged mask to the original GT resolution.
- The official COCO/FSS target classes and dataset-specific filter tuning
  are not used; the official default filters are kept for all 20 datasets.
- DINOv2 ViT-L/14 weights are not in the local checkpoint directory.
  `--matcher-random-init` is a code-smoke-only option; reported results
  require the public DINOv2 checkpoint.
- SAM ViT-H is present locally.  The Matcher code does not have released
  method weights.
- Target GT is never used: matching points come from the support masks and
  DINOv2 features.  The method is support-based, not prompt-free.
- No test-time augmentation.
