# A2SAM fidelity ledger

This file records what is official, what is reconstructed from the paper, and
what is a project-protocol adaptation. It is the source of truth for A2SAM
claims in the paper tables and appendices.

## 1. Provenance

| Item | Value |
|---|---|
| Paper | [2026 MIA] *Informativeness-driven active adaptation of SAM: Structural prompts and contrastive parameter selection for medical tubular segmentation* |
| DOI | `10.1016/j.media.2026.104285` |
| Official code (this port) | `/home/dev/mywork/project/segment/3rd/A2SAM` |
| Official code commit | `2d0574affcbf51801af3f9e214671140778ee583` (`2d0574a`, 2026-08-01) |
| Official code remote | `git@github.com:YongChen-Exact/A2SAM.git` |
| Unrelated same-name repository | `/home/dev/mywork/project/segment/3rd/a2sam` |
| Unrelated repository identity | NeurIPS 2024, *Flaws can be Applause: Unleashing Potential of Segmenting Ambiguous Objects in SAM*; commit `b658ad8`; **not used by `models/a2sam`** |
| SAM ViT-B checkpoint | `checkpoints/sam_vit_b_01ec64.pth` |
| SAM ViT-B SHA-256 | `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` |

Official release file hashes:

```text
445ffe0d9b0827c81a17a5608fb46b73a8e2bf952d1bcd159c98d06a3552b0d4  train.py
ddc0a02a855bd7e2e586798a216b4a79e1eb3f9136299111a086319f52049617  test.py
121333da6cb4e722b7f3950d47edf89f288d0dc4d0edef6dd184925f2af09532  models/sam.py
86d06a4669034f45df541cfab2b15352c20004d2cbcc482faf2755e53903bcf9  models/iou_ta_loss.py
8da0a821ea6a8bd43e5c71ba21163d7c990836898f87be9b97685a7a58524642  utils_/pruning.py
fd84b45e5cb3a5ffef793f06f0de878bfe8d30565da6251c60bd8cfda28c4ee1  models/mmseg/models/a2sam/prompt_discriminator.py
fe6846f1badcdbb22fe80b0bbed84767637a5cdc22a16b10376b2c125ea47b83  configs/cod-sam-vit-b-drive.yaml
```

## 2. What the official release actually contains

Present:

- SAM ViT-B image encoder, prompt encoder, mask decoder, two-way transformer.
- `PromptGenerator`: two Conv-BN-LeakyReLU branches, `(1 - unc, coarse_logit)`
  -> 16/32 channels -> concat 48 -> 1 channel.
- CPS: squared-gradient influence, `|w|` magnitude, global min-max
  normalization, `alpha=0.4`, freeze lowest 96.5%.
- Thickness-aware loss and IOU loss.
- Configs for DRIVE/FIVES/OCTA-500/XCAD/2PFM.

Absent:

- Lightweight U-Net for coarse likelihood/mask.
- Prior network and `xi ~ N(mu, sigma^2)` persistence-threshold sampling.
- DMT/Morse-complex construction.
- Persistent-homology branch pruning.
- Structural uncertainty map generation.
- Full structural prompt generator (only the final conv fusion exists).
- Official A2SAM trained weights.

Release defects we intentionally do not reproduce:

- Missing `models/__init__.py` and unregistered `validation` wrapper.
- Undefined `self.discriminator`; `self.prompt_generator` created but unused.
- Prompt fusion detached from the loss.
- `test.py` builds a different image-encoder `tuning_mode` than training.
- Validation aggregates channels instead of images.
- Heavy checkpoint loading is silent/partial.
- CLI defaults to Adam and overwrites YAML AdamW.
- Hardcoded GPU 2 and hardcoded W&B key.
- Test metric code and IOU zero-union edge cases.

## 3. Source-of-truth per component

| Component | Default mode | Release mode | Status |
|---|---|---|---|
| SAM modules | vendored byte-identical | same | official |
| SAM checkpoint load | official allowlist (exclude `mask_tokens`, `output_hypernetworks_mlps`, `iou_prediction_head`) | same | official |
| Dense SAM PE | loaded `prompt_encoder.pe_layer` | release separate random PE not reproduced | official/SAM repair |
| Prompt fusion architecture | official two-branch widths | same | official |
| Prompt fusion gradient | trainable, no detach | optional detach | paper repair |
| Prompt fusion inputs | `(1 - u, coarse_logit)` | same | official |
| Prompt fusion inputs (paper mode) | `(u, y_c)` | n/a | paper ablation |
| CPS formula | official normalized influence/magnitude, alpha 0.4, freeze 96.5% | same | official |
| CPS statistics | 18 training images only | release val-loader behavior not reproduced (label leakage) | protocol repair |
| CPS mask scope | official: exclude IoU head; freeze image-encoder norm/pos/1-D neck; zero grads, no `requires_grad=False` | same | official |
| U-Net | lightweight BCE-only, 4.24M | n/a | paper reconstruction |
| Prior network | global scalar `xi ~ N(mu, sigma^2)`, 2.75M | n/a | paper reconstruction |
| DMT/Morse | exact H0 persistence (cripser auto-preferring; gudhi fallback) + join-tree verification/watershed reconstruction | n/a | persistence exact; DMT field reconstructed |
| Persistence sampling | `Z=5`, `u = Var(y_z)` | n/a | paper reconstruction |
| Thickness loss | paper Eq. 24-30 (log W, foreground denominator) | optional official reciprocal/all-pixel mode | paper default |
| Optimizer | AdamW 2e-4, wd 0.01, cosine to 1e-7 | optional official CLI Adam | paper/config default |
| Epochs | 800 shared (100 U-Net + 100 prior + 600 A2SAM) | n/a | project protocol |
| Selection | validation Dice on 6-image val split | n/a | project protocol |

## 4. Verification snapshot (2026-10-28)

- Parameter counts: SAM 93.735M + U-Net 4.239M + prior 2.752M + fusion
  0.022M = 100.749M (paper 100.79M).
- Official SAM allowlist: 177 image / 17 prompt / 89 mask-decoder tensors
  loaded; 31 tensors / 692,868 parameters random.
- Three-stage GPU smoke (1 U-Net + 1 prior + 1 A2SAM epoch, 1 train/1 val):
  completed; CPS kept 3.48% of 93.6M SAM elements; stage-3 checkpoint saved.
  That one-epoch U-Net produced no foreground branch (U-Net Dice 0, branches
  0, u 0), so the smoke proves wiring only, not prior/SPG quality.
- Prediction smoke: 16 DRIVE masks at 565x584, 0/255 values; two independent
  runs produced identical SHA-256 sets.
- These are smoke/integration checks, not a quality or paper-parity claim.

## 5. Fidelity boundaries

- `cripser` computes exact 0D superlevel-set persistence. The discrete Morse
  gradient field and stable manifolds are not available in the local release;
  we reconstruct branch regions with a deterministic join tree + watershed.
  Therefore the SPG is **paper-level reconstruction**, not an exact DMT port.
- The paper does not specify the U-Net/prior architectures, the prior training
  target, or the exact `F_MS` reconstruction. Our choices must be documented
  with each run and validated with U-Net Dice / uncertainty diagnostics.
- There are no official A2SAM weights, so no end-to-end numerical parity with
  the published results is possible.
- The main table uses the defaults above. Release/paper switches are for
  ablations and parity tests only; they must not be mixed silently.
