# CVPR Story and Iteration Policy

Status: **waiting for the 8-GPU screen results**. No architecture
changes until the screen is complete. This file records the intended
logic so later iterations cannot silently drift into "we tried many
modules until the numbers moved".

Related registry: [`docs/SACM_X_VARIANTS.md`](SACM_X_VARIANTS.md).

---

## 1. One-sentence thesis

> **Global adaptation teaches SAM what to look for; it does not fix
> where its remaining structural errors are. We prevent
> topology-destructive predictions during training and repair residual
> structural errors at inference.**

Short version for the introduction:

```text
Representation adaptation → error prevention → error repair
```

## 2. Logical chain

Every claim must have a measurement. If a premise fails, narrow the
story; do not force the original narrative.

| # | Premise | Mechanism / evidence | Status |
|---|---|---|---|
| P1 | In 3-shot, global adaptation saturates | adapter capacity sweep; EVP/ConvLoRA/hybrid vs baseline | pending |
| P2 | Remaining errors are local and structural | error decomposition: fragmentation, missed thin branch, spurious component, boundary | pending |
| P3 | Error support is predictable at inference | gate AUROC (Seen/Unseen), gate heatmap, positive rate | pending |
| P4 | Local residual repair improves structure | ResMoE ablation on Dice/clDice/components/Betti | pending |
| P5 | Prevention and repair are complementary | SCNP-only / ResMoE-only / SCNP+ResMoE ablation | pending |

If P1 fails, the story is "global + local adaptation", not "global
adaptation saturates".

If P3 fails, the repair mechanism has no basis; do not keep ResMoE as
the hero.

If P4 fails, stop adding repair modules.

If P5 fails, choose the stronger single mechanism; do not claim
"prevent + repair".

## 3. Three layers, with ownership

| Layer | Current implementation | Ownership |
|---|---|---|
| Representation / global adaptation | SACM adapters, FusionV2, C2F, multi-depth, 4 heads, fixed head 2 | Inherited from SACM/our backbone; **not a novelty claim** |
| Training-time topology control | Skeleton Recall + centerline/boundary/width aux; optional SCNP (`full_fixed2_scnp`) | Skeleton/aux integration is ours; **SCNP is existing CVPR 2026 work, cite it** |
| Inference-time structural repair | `TopoResidualMoE` (`full_fixed2_resmoe`, `_dg`) on head 2 | **Main method novelty**: head-disagreement gate, topology-aware target, deterministic experts, no target labels/test-time optimization |

Important: the first screen tests these **separately**:

```text
B          = full_fixed2_aux
B+SCNP     = full_fixed2_scnp
B+Repair   = full_fixed2_resmoe
B+Hybrid   = full_fixed2_lora_evp
```

There is currently **no combined SCNP+ResMoE preset**. Add one only if
both individual mechanisms show a positive signal; then the combined
run is the evidence for "prevent + repair".

## 4. Story variants (decide from results)

| Result pattern | Story | Hero |
|---|---|---|
| ResMoE improves Seen and Unseen; gate transfers | "Learning Where to Repair" | inference-time repair |
| SCNP-only improves topology; ResMoE flat | "Prevention" / topology-aware training | training-time regularization |
| SCNP and ResMoE individually positive and combined positive | "Prevent and Repair" | two-stage error control |
| Hybrid only improves | "Global + local adaptation" | encoder capacity; repair becomes ablation |
| Nothing beats baseline | Do not add modules; pivot to XCAD/OCTA audit and systematic 3-shot analysis | analysis/benchmark |

Do not expand the thesis to cover a mechanism that has no positive
ablation.

## 5. Iteration loop

Each round:

```text
observation → diagnosis → hypothesis → one minimal change → validation
→ keep/revert → record
```

Rules:

1. **Test set is opened once, at the end.** All iteration is on
   validation/support.
2. **One variable at a time.** A run must be explainable by one
   hypothesis.
3. **Macro + direction consistency.** A gain on one dataset or one seed
   is not a conclusion.
4. **Negative results are recorded.** A failed gate/loss/target is
   analysis material, not something to delete.
5. **The thesis changes only when evidence changes it.** Implementation
   can change freely; the claim cannot be retrofitted.
6. **Maximum two architecture-level iteration rounds.** After that,
   freeze the method and spend the remaining time on seeds, evaluation,
   XCAD/OCTA, and writing.

## 6. Allowed vs forbidden iterations

Allowed:

- gate input: head disagreement / centerline / image high-pass;
- gate target: raw error vs topology-aware;
- expert transforms: opening / closing / threshold;
- J, hidden width, warm-up/ramp;
- residual loss weights and Skeleton Recall weight;
- SCNP window / weight / warm-up;
- TTA / EMA / model soup;
- 300 vs 800 epochs if validation is still rising;
- learned gate vs disagreement gate.

Forbidden:

- threshold or head selection tuned on the 20 test sets;
- per-dataset special cases to inflate the macro;
- changing evaluation geometry in the middle of a comparison;
- adding a module without a failure mode and an ablation;
- silently dropping negative runs;
- changing the thesis after seeing the test numbers.

## 7. Success / kill criteria

A candidate enters the final method only if, relative to the 300-epoch
fixed2 baseline:

- >= +0.5 Dice **or** >= +0.5 clDice; and
- no Unseen regression; and
- component/Betti improvement if Dice is flat; and
- direction consistent across seeds; and
- acceptable inference cost.

If not met after two rounds: stop architecture work. Pivot to:

- XCAD / OCTA audit (preprocessing, head selection, threshold on
  val/source only);
- TTA / EMA / model soup;
- 5-seed stability;
- error decomposition and systematic 3-shot analysis.

## 8. Paper positioning

### Inherited vs ours

- SACM adapters / FusionV2 / C2F / multi-depth: inherited; cite SACM.
- Skeleton Recall / aux heads: existing losses, our integration.
- SCNP: existing CVPR 2026; cite; optional training regularizer.
- TopoResidualMoE: our adaptation; cite FineSAM++ as inspiration and
  state the deltas.

### Delta vs FineSAM++

| FineSAM++ | Our setting |
|---|---|
| Full-data, per-dataset | 18-image 3-shot |
| Same-domain train/test | 20 datasets, Seen/Unseen |
| Single coarse mask | prompt-free 4-head decoder; fixed head 2 |
| Gate from same-domain error labels | head-disagreement / topology-aware gate; no target-domain error labels |
| Random degradation, clean test | deterministic train/test transforms |
| MSE residual | Dice/BCE + Skeleton Recall |
| Pixel error target | missed skeleton / spurious component / boundary target |

### Contributions (three, not five)

1. **Insight**: in 3-shot prompt-free curvilinear segmentation, global
   adaptation saturates and residual errors are local/structural.
2. **Method**: a prevent+repair framework; the core is a label-free
   disagreement gate and topology-aware deterministic residual experts.
3. **Evidence**: 20-dataset 3-shot evaluation, Seen/Unseen, 5 seeds,
   Dice/clDice + component/Betti, ablations and cost.

### Title candidates

- **Learning Where to Repair: Topology-Aware Residual Experts for
  Few-Shot Curvilinear Segmentation**
- **From Fragments to Topology: Local Repair for Prompt-Free SAM in
  Few-Shot Curvilinear Segmentation**
- **Prevent and Repair: Error-Aware Topology for Few-Shot Curvilinear
  Segmentation** (only if SCNP+ResMoE are complementary)

Do not use "SACM-X + FineSAM++" or a benchmark-report title.

## 9. Current state

- Code/provenance: `code_fingerprint` implemented; 8-GPU screen
  requires a clean tree by default (`ALLOW_DIRTY=1` for exceptions).
- Screen: not yet run; no current 300-epoch result for SCNP/ResMoE/
  hybrid.
- Legacy anchors (do not mix): old fixed2 800-epoch All Dice ~59.16
  (source artifact not in the current tree); `full_v2_skel_aux` local
  800-epoch best validation F1 ~0.713 at epoch 286, val Dice ~0.669.
- 8-GPU allocation and canonical names: see
  [`docs/SACM_X_VARIANTS.md`](SACM_X_VARIANTS.md).

## 10. Experiment log template

For every iteration, record:

```text
date:
hypothesis:
change:
preset:
code_fingerprint:
seed(s):
budget:
val Dice / clDice / components:
decision: keep / revert
reason:
```

The results table is filled only after the screen completes. Until
then, this file is a plan, not a claim.

---

## 11. Hidden assumptions and failure modes

The story is internally coherent, but it is **not yet empirically
proven**. These are the places where it can break.

### 11.1 "Global adaptation saturates" may be false

If EVP/ConvLoRA/hybrid still improve substantially, the premise is
wrong. The safe fallback story is "global + local adaptation", not
"global adaptation saturates".

### 11.2 Residual errors may not be local/structural

Some errors are domain shift, contrast, annotation noise, class
imbalance, or resolution. A local residual expert cannot fix those.
Need an error taxonomy and a measured share of local structural errors.

### 11.3 The gate may not transfer to Unseen

Head disagreement can be low when the four heads are correlated, or it
can reflect head instability rather than true error. Learned gates can
memorize training errors. Without gate AUROC on Seen/Unseen, the repair
premise is unsupported.

### 11.4 Prevention may already remove the residual

Skeleton Recall + SCNP may fix the exact errors ResMoE is meant to
repair. The ablation may show:
- prevention helps, repair adds nothing;
- repair helps only when prevention is off;
- both help but not complementarily.

If prevention dominates, the "prevent + repair" story collapses to
"prevention".

### 11.5 Joint training breaks the clean separation

The coarse head and repair module are trained jointly. "Repair" can
change the coarse model, so the claimed separation between
representation, prevention, and repair is not mechanically true.
Need a frozen-coarse vs joint-training ablation.

### 11.6 Gains may come from parameters/compute, not repair

A parameter-matched single expert, a single residual conv, or a global
adapter with the same budget could match ResMoE. Need
parameter/compute-matched controls; otherwise the gain is not
attributable to the gate-and-expert design.

### 11.7 One expert may be enough

If J=1 matches J=4, the MoE framing is unnecessary. The honest method
is a gated residual refiner, not a mixture-of-experts. Test J=1 before
using "MoE" in the title.

### 11.8 Cheap alternatives may match the repair

TTA, EMA, model soup, SCNP, threshold/head calibration, or simple
post-processing may recover the same gains at lower complexity. The
paper must compare against these before claiming repair is necessary.

### 11.9 Validation overfitting

There are only six validation images. Iterating gate/target/loss
hyperparameters on them can overfit. Use a small pre-registered grid,
prefer robust defaults, and require 5-seed consistency. Never tune on
test.

### 11.10 Metric mismatch

Dice, clDice, HD95, components and Betti can move in opposite
directions. Define the primary claim before seeing the test results:
- "Dice improvement", or
- "topology at matched Dice", or
- "component/Betti reduction".
Do not switch the primary metric after the fact.

### 11.11 Dataset heterogeneity

A single gate/expert may help cracks but hurt retinal vessels, or vice
versa. Macro averages can hide per-dataset regressions. Report
per-dataset wins/losses and the Seen/Unseen split.

### 11.12 Novelty is still at risk

The chain can be coherent and still be judged "SACM + SCNP +
FineSAM++". The novelty must be one of:
- label-free disagreement gate for cross-domain repair;
- topology-aware repair target;
- deterministic expert specialization;
- the 3-shot cross-domain study itself.
Pick one hero; do not list all modules as equal contributions.

### 11.13 The story may be too modular

Three layers are easy to explain but can look like a pipeline paper.
The introduction should lead with one falsifiable claim, not with
"representation → prevention → repair" as three independent modules.

## 12. Falsification tests

Before writing the paper, run these tests and record the answers:

| Premise | Falsification test | If it fails |
|---|---|---|
| P1 saturation | adapter capacity sweep on val | change thesis to global+local |
| P2 local errors | error-type decomposition | repair is the wrong tool |
| P3 gate transfer | gate AUROC Seen/Unseen | drop learned gate; try DG or stop |
| P4 repair gain | ResMoE vs parameter-matched refiner | simplify to single expert or drop |
| P5 complementarity | SCNP-only / Repair-only / both | choose one mechanism |
| cheap alternatives | TTA/EMA/SCNP vs repair | repair must beat them or be dropped |
| attribution | frozen coarse vs joint training | rewrite the separation claim |
| robustness | per-dataset wins/losses, seeds | narrow claims or stop |

A good CVPR story is not "the logic sounds coherent". It is:
**every premise has a test, every test has a result, and the story is
whatever the results support.**

---

## 13. Backup error-localization designs (if the learned gate fails)

The learned gate can fail for three reasons: training errors are too
sparse in 3-shot, the gate memorizes the 18 training images, or head
disagreement is weak/correlated. Do not respond by adding another
learned head. Prefer signals that are available at inference and are
domain-agnostic.

Priority order:

### 13.1 Structural discrepancy gate (recommended first backup)

Replace the learned gate with a **training-free structural suspicion
map**. It uses the coarse mask, the structural aux heads, and simple
morphology:

- skeleton endpoints / branchpoints of the coarse mask;
- small connected components and short skeleton branches;
- gaps between components / endpoints close in space but disconnected;
- centerline aux high where coarse mask is low (missed thin branch);
- coarse mask high where centerline aux is low (spurious region).

The gate is a dilated union of these suspicious pixels. No target-domain
labels, no learned gate, and no extra forward pass. It directly targets
fragmentation and spurious components, which are the stated failure
modes.

Risk: thresholds and morphology can be noisy; tune on validation only.

### 13.2 TTA-disagreement gate

Run the coarse model on the test image and 2-4 geometric augmentations
(identity, horizontal flip, rot90). Invert the transforms and compute
per-pixel variance/entropy. Use it as the gate.

- Strong, label-free uncertainty signal.
- Works on Unseen by construction.
- Cost: 2-4x inference; report it.
- May highlight boundaries rather than all structural errors.

### 13.3 Teacher-student gate distillation

Use the structural/TTA signal as a **teacher** on training images, and
distill it into a lightweight gate **student**. At test, run only the
student, so inference stays cheap.

This is the best of both worlds if TTA is too expensive, and it gives a
stronger story than "we learned a gate from 18 error masks".

### 13.4 Synthetic error mining / cross-fitting

Keep the learned gate but fix its supervision:

- corrupt the coarse mask (gap carving, thinning, threshold jitter,
  false-positive blobs) and train the gate to predict the corruption
  error;
- or use 3-fold cross-fitting: train coarse models on 12 images,
  generate out-of-fold errors on the held-out 6, train the gate on
  those errors.

Cross-fitting is the principled fix but costs extra coarse-model
training.

### 13.5 Cheap baselines that may already be enough

Before building a complex gate, test these on validation:

- pixel-wise median / mean of the four candidate heads;
- best head per pixel by local confidence;
- seed ensemble / model soup;
- entropy/margin thresholding.

If one of these matches the repair gain, the paper does not need a
learned gate; it needs a simpler and more honest baseline.

### 13.6 Diagnostic before choosing a backup

Using an existing trained checkpoint, on validation (and an Unseen
diagnostic split), compute AUROC of each candidate signal against the
hard error mask `|p-y|>0.5`:

```text
learned gate
head disagreement
entropy / margin
TTA disagreement
structural suspicion
centerline-mask mismatch
```

Only a signal with AUROC clearly above chance (target >= 0.75) is worth
turning into a gate. If no signal works, stop the repair line and pivot
to SCNP/topology analysis or XCAD/OCTA.

The backup is not "another module"; it is a better **error-localization
principle**: measure structural inconsistency at inference instead of
learning a training-error pattern from 18 images.
