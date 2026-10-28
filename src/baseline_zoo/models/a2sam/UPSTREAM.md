# A2SAM upstream record

## Sources

| Component | Upstream | Version |
|---|---|---|
| Paper | "Informativeness-driven active adaptation of SAM: Structural prompts and contrastive parameter selection for medical tubular segmentation", Medical Image Analysis 2026, DOI 10.1016/j.media.2026.104285 | 2026 |
| Official code | `YongChen-Exact/A2SAM` | `/home/dev/mywork/project/segment/3rd/A2SAM`, commit `2d0574a` |
| SAM ViT-B | Meta `segment-anything` modules vendored inside A2SAM | ViT-B, local checkpoint |
| Unrelated same-name repo | NeurIPS 2024 "Flaws can be Applause" | `/home/dev/mywork/project/segment/3rd/a2sam`, not used |

Licenses: A2SAM is MIT; the SAM modules are Meta's Apache-2.0
implementation. The SAM ViT-B checkpoint is local and offline. `cripser`
(LGPL-3.0) or `gudhi` (MIT) is used as an installed optional topology
dependency; neither is vendored. Do not import from `3rd/` at runtime.

The full source-of-truth ledger is `docs/A2SAM_FIDELITY.md`.

## Official components (reproduced)

- `sam/{common,image_encoder,prompt_encoder,mask_decoder,transformer}.py`:
  copied verbatim from the A2SAM release (`tuning_mode='cps'`).
- `model.load_sam_checkpoint`: official allowlist. It loads all
  image/prompt/mask-decoder keys except `mask_tokens`,
  `output_hypernetworks_mlps` and `iou_prediction_head`. Expected counts are
  177 / 17 / 89 loaded and 31 tensors / 692,868 parameters left random.
- `prompt.py`: exact two-branch `PromptGenerator` layout
  (1→16→16 and 1→16→32, concat 48→48→1, BN+LReLU(0.2)).
- `cps.py`: official freeze semantics — grad² and |w| globally min-max
  normalized, alpha 0.4, freeze the lowest 96.5%, exclude the IoU head from
  scoring/masking, force image-encoder norm/pos and 1-D neck, zero gradients
  without `requires_grad=False`.
- `loss.py`: IOU (with a zero-union safety guard). Paper thickness loss is
  the default; `mode='release'` reproduces the release reciprocal-weight /
  adjacent-product / all-pixel denominator behavior.

## Paper-level reconstruction (not in the release)

The release has no U-Net, prior, DMT/Morse complex, persistence sampling,
uncertainty generator, or full structural prompt generator. It consumes
precomputed `uncertainty/*.npy` and `logit/*.npy` and detaches its fusion.

This port reconstructs:

- `unet.py`: coarse U-Net, ~4.24M parameters, GroupNorm for batch 1.  It
  uses BCE with foreground `pos_weight=10` by default because the shared
  3-shot split is strongly imbalanced; set `--a2sam-unet-pos-weight 1` for
  the paper plain-BCE variant.
- `prior.py`: global scalar `xi ~ N(mu, sigma^2)` prior conditioned on the
  image and coarse probability; ~2.75M parameters. The predicted mean is
  constrained to `[0,1]`, its variance has a `min_sigma=0.05` floor, and the
  training target is the Dice-optimal persistence threshold from training GT
  (no val/test labels).
- `topology/persistence.py`: exact 2D superlevel-set H0 persistence via
  `cripser` (default `auto`; if cripser is absent, `auto` refuses rather
  than silently changing connectivity). `gudhi` is an explicit MIT cubical
  backend (diagonal coface connectivity, so its H0 bars can differ from the
  4-connected cripser/join-tree convention); `join_tree` is an explicit
  pure-Python 4-connected backend for verification/small inputs.
  `topology/morse.py` reconstructs branch basins with a deterministic
  watershed; `topology/fms.py` implements `F_MS` and `u=Var(y_z)`.
  Persistence values are exact under the selected connectivity; the DMT
  gradient field and stable manifolds are not bit-exact.
- `model.py`: U-Net → prior → `F_MS`/uncertainty → official prompt fusion →
  SAM decoder.

Parameter budgets: SAM 93.735M, U-Net 4.239M, prior 2.752M, fusion 0.022M;
total 100.749M versus the paper's 100.79M.

## Default modes

| Component | Default | Release/ablation mode |
|---|---|---|
| SAM init | official allowlist | same |
| Prompt inputs | release style `(1-u, coarse_logit)` | paper `(u, y_c)` via `--a2sam-fusion-inputs paper` |
| Prompt gradient | trainable | `--a2sam-prompt-detach` |
| CPS | official freeze/score | paper score via `--a2sam-cps-mode paper` |
| CPS statistics | 18 training images | release val-loader behavior is not reproduced (label leakage) |
| TA loss | paper Eq. 24–30 | `--a2sam-ta-mode release` |
| Optimizer | AdamW 2e-4 / 0.01 | release Adam via `--a2sam-optimizer adam` or `--a2sam-release-parity` |
| Budget | 800 shared epochs: 100 U-Net + 100 prior + 600 A2SAM, batch 1 | n/a |

The `--a2sam-release-parity` aggregate switch sets release inputs, detach,
release TA, official CPS and Adam for parity/ablation tests.

## Protocol adaptations

- 18 train / 6 validation images. The U-Net uses foreground-weighted BCE
  (pos_weight 10); plain BCE collapsed to all-background in the local
  100-epoch diagnostic. The three stages share the 800-epoch
  project budget; the paper's A2SAM fine-tuning is 50/100 epochs.
- Validation-Dice checkpoint selection; only stage-3 epochs are eligible, so
  a random-SAM/prior-only checkpoint cannot become `best_model.pth`.
- CPS statistics use training images only (paper Algorithm 1), not the
  release validation loader.
- U-Net/prior resolution is 1024 because the release resizes its precomputed
  maps to 1024; there is no separate 512 path in this port.
- Deterministic per-sample generators seed prior/xi sampling for validation
  and prediction. Training caches are rebuilt deterministically.
- There are no official A2SAM weights; only the SAM ViT-B checkpoint is
  pretrained. No official end-to-end parity is claimed.

## Verification

- `tests/test_a2sam.py`: SAM allowlist counts/random keys, PromptGenerator
  parity against the release reference (when available), CPS freeze
  semantics, paper/release loss modes, synthetic persistence/Morse/F_MS,
  parameter budgets, deterministic xi sampling, adapter stage/config/data
  pipeline, and a real SAM checkpoint slow test.
- `tests/test_baseline_zoo.py`, `tests/test_self_contained.py`,
  `tests/test_configs.py`, `tests/test_runtime_fixes.py`.
- GPU smoke and prior-Dice/uncertainty triage are required before a 5-seed
  campaign.
