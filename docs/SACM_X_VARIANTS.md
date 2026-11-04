# SACM-X Variants: Single Source of Truth

Last updated: 2026-11-04

This file is the canonical registry for SACM-X experiments.  The
`preset` column is the exact key in
`src/models/sacm_x/configs.py`; if a name is not in this table, it is not a
current SACM-X variant.

**Status meanings**

| Status | Meaning |
|---|---|
| `MAIN` | current baseline / final comparison |
| `CANDIDATE` | current 8-GPU screen |
| `FALLBACK` | only if the primary candidate fails |
| `CONTROL` | attribution only; not the final method |
| `ABLATION` | single-switch / historical check |
| `LEGACY` | historical result; do not run for current comparisons |
| `INTERNAL` | no-op / debugging preset |

**Global rules**

1. Results from different epoch budgets are not comparable.  Current
   screen budget is **300 epochs**; historical runs used **800**.  Never
   mix them in one table without a `budget` column.
2. Results from different evaluation geometry are not comparable
   (`aspect_pad` vs `resize`; `EMPTY_GUARD` on/off).
3. Fixed2 rows use `VAL_SELECTION=fixed`, `VAL_FIXED_HEAD=2`,
   `fixed_supervision_head=2`.
4. A single-seed variant result is a screen, not a conclusion.  A final
   claim needs 5 seeds.
5. If a number is not in a run's `metrics_history.json` / `eval/tables`,
   it does not exist for this project.
6. Do not compare to the SACM paper's clDice numbers without re-checking
   the metric definition; use this project's verified metrics.

---

## 1. Current mainline (read this first)

The current SACM-X baseline is `full_fixed2_aux`.  Everything below is
defined relative to it.

| Alias | Preset | Parent | Key change | Purpose / status |
|---|---|---|---|---|
| **B** | `full_fixed2_aux` | `full` | fixed2 decoder + structural aux + fixed head 2 + Skeleton Recall@512 | **MAIN baseline** |
| **B+SCNP** | `full_fixed2_scnp` | `full_fixed2_aux` | `scnp=True`; training-only, inference unchanged | **CANDIDATE** |
| **B+Repair** | `full_fixed2_resmoe` | `full_fixed2_aux` | learned gate + 4 deterministic residual experts on head 2 | **CANDIDATE** |
| **B+Repair-DG** | `full_fixed2_resmoe_dg` | `full_fixed2_resmoe` | gate = four-head disagreement; no learned gate | **FALLBACK** |
| **B+Hybrid** | `full_fixed2_lora_evp` | `full_fixed2_aux` | ConvLoRA qkv + EVP prompts; keeps SACM external adapter + FusionV2 | **CANDIDATE** |
| **B+EVP** | `full_fixed2_evp` | `full_fixed2_aux` | EVP only; keeps internal/external adapters | CONTROL |
| **B+ConvLoRA** | `full_fixed2_convlora` | `full_fixed2_aux` | ConvLoRA only; keeps internal/external adapters | CONTROL |
| **B+Replace** | `full_fixed2_peft_lora_evp` | `full_fixed2_lora_evp` | remove SACM adapters/FusionV2; ConvLoRA + EVP only | CONTROL |
| **B+HR** | `full_fixed2_hr_aux` | `full_fixed2_aux` | DetailStem + 1024 image-guided refine | OPTIONAL control |

Common fixed2 settings:

- 4 Stage-2 heads; deployment head = index 2 (ranked order)
- `fixed_head_loss_weight=0.5`, `fixed_supervision_head=2`
- `high_res_output=True`, `high_res_size=512`
- `high_res_refine=False` except `full_fixed2_hr_aux`
- `topology_loss=skeleton_recall`, `topology_resolution=512`
- `tversky_weight=0.3`, `tversky_alpha=0.3`, `tversky_beta=0.7`
- structural aux: centerline 0.2, boundary 0.1, width 0.05, area 0.05

**Result roots**

- Normal run: `results/runs/sacm_x/<preset>/seed_<seed>/`
- Current 8-GPU screen: `results/runs/cvpr8gpu_screen/sacm_x/<preset>/seed_<seed>/`
- Logs: `results/ablation_logs/cvpr8gpu_*`

**Budget**

- 300 epochs: current screen (baseline seeds 0-4 + one seed of each
  candidate).
- 800 epochs: legacy/full budget; only for a winner whose validation
  curve is still rising at 300, and never mixed with 300-epoch rows.

---

## 2. Current 8-GPU allocation

`scripts/sacm_x/run_cvpr_8gpu_screen.sh` runs exactly eight jobs:

| GPU | Preset | Seed | Role |
|---|---|---|---|
| 0 | `full_fixed2_aux` | 0 | baseline |
| 1 | `full_fixed2_aux` | 1 | baseline |
| 2 | `full_fixed2_aux` | 2 | baseline |
| 3 | `full_fixed2_aux` | 3 | baseline |
| 4 | `full_fixed2_aux` | 4 | baseline |
| 5 | `full_fixed2_scnp` | 0 | training-only topology prevention |
| 6 | `full_fixed2_resmoe` | 0 | local structural repair |
| 7 | `full_fixed2_lora_evp` | 0 | encoder-side hybrid |

The script forces `SCNP=false` for every job; the SCNP wrapper forces
`SCNP=true`, the ResMoE wrappers force `SCNP=false`, so attribution is
clean.  `full_fixed2_resmoe_dg` is not in the first wave.

Second wave (only after the screen):

- seeds 1-4 of the winning candidate;
- optional single-component controls (`full_fixed2_evp`,
  `full_fixed2_convlora`);
- optional `full_fixed2_resmoe_dg` if the learned gate fails on Unseen.

---

## 3. SACM historical lineage

These are historical or ablation presets, not current candidates.
`full_fixed2_aux` is the current baseline; `full` is its ancestor.

| Preset | Status | Parent | Key meaning |
|---|---|---|---|
| `sacm` | LEGACY | — | bug-fixed official SACM: adapters/C2F/FusionV2/multi-depth off; WTA Stage-2 |
| `stage1` | LEGACY | `sacm` | +C2F + deep supervision + clDice + IoU |
| `stage2` | LEGACY | `stage1` | +geometric internal/external adapters |
| `full` | LEGACY | `stage2` | +FusionV2 + multi-depth; 4 heads; soft supervision |
| `full_wta` | ABLATION | `full` | winner-take-all Stage-2 supervision |
| `full_allhead` | ABLATION | `full` | mean over all Stage-2 heads |
| `full_soft` | ABLATION | `full` | explicit soft Stage-2 supervision |
| `full_scnp` | LEGACY | `full` | old `full` + SCNP; not fixed2 |
| `full_typed` | LEGACY | `full` | typed Stage-2 slots (thin/blob/boundary/recall) |
| `full_typed_topo` | LEGACY | `full_typed` | typed + Betti matching loss |

`full_scnp` is **not** the fixed2 SCNP candidate.  Use
`full_fixed2_scnp` for the current comparison.

---

## 4. PEPA / typed exploratory branch

All of these are LEGACY.  They were explored before the project returned
to fixed2; they are not current candidates.

| Preset | Status | Key meaning |
|---|---|---|
| `full_pepa` | LEGACY | TCSU snake upsampling + TADT adaptive threshold |
| `full_pepa_topo` | LEGACY | + Topograph loss |
| `full_pepa_cape` | LEGACY | + CAPE topology loss |
| `full_pepa_skel` | LEGACY | + Skeleton Recall loss |
| `full_pepa_topo_cape` | LEGACY | + Topograph + CAPE |
| `full_pepa_rank` | LEGACY | + learned candidate ranker |
| `full_pepa_all` | LEGACY | + ranker + topology terms |
| `full_pepa_hr` | LEGACY | + DetailStem + 512 output + 1024 refine |
| `full_pepa_all_hr` | LEGACY | `full_pepa_all` + high-resolution path |

---

## 5. v2 / v3 branch

These are historical architecture attempts.  They did not become the
current baseline.

| Preset | Status | Key meaning |
|---|---|---|
| `full_v2` | LEGACY | external adapters only; single head; DetailStem; 512/1024 refine; Topograph+Skeleton |
| `full_v2_skel` | LEGACY | v2 + fixed Skeleton Recall@512; TADT off |
| `full_v2_skel_aux` | LEGACY | v2 + structural centerline/boundary/width aux |
| `full_v3` | LEGACY | dense structural decoder; no prompt decoder/candidate heads |

`full_v2_skel_aux` is the closest historical predecessor to fixed2, but
it is **not** the current baseline.

---

## 6. Single-switch / layer ablations

These are ABLATION presets.  They are meaningful only against their
parent (`full`), not against fixed2.

| Preset | Status | Parent | Switch |
|---|---|---|---|
| `full_dynamic_strip` | ABLATION | `full` | dynamic strip adapters |
| `no_geo_i` | ABLATION | `full` | remove internal geometric adapter |
| `no_geo_e` | ABLATION | `full` | remove external geometric adapter |
| `no_c2f` | ABLATION | `full` | remove coarse-to-fine refiner |
| `no_fusion_v2` | ABLATION | `full` | remove FusionV2 |
| `no_multi_depth` | ABLATION | `full` | remove multi-depth path |
| `no_cl` | ABLATION | `full` | clDice weight = 0 |
| `no_ds` | ABLATION | `full` | deep supervision = 0 |
| `no_iou` | ABLATION | `full` | IoU head loss = 0 |
| `geo_e_shallow` | ABLATION | `full` | external adapters on shallow window layers only |
| `geo_e_deep` | ABLATION | `full` | external adapters on deep window layers only |

`none` is INTERNAL: keep CLI values, apply no preset.

---

## 7. Complete preset inventory

44 presets exist in `configs.py`.  This table is intentionally terse; use
the sections above for meaning.

| Preset | Status | Family |
|---|---|---|
| `sacm` | LEGACY | SACM lineage |
| `stage1` | LEGACY | SACM lineage |
| `stage2` | LEGACY | SACM lineage |
| `full` | LEGACY | SACM lineage |
| `full_wta` | ABLATION | Stage-2 supervision |
| `full_allhead` | ABLATION | Stage-2 supervision |
| `full_soft` | ABLATION | Stage-2 supervision |
| `full_scnp` | LEGACY | old full + SCNP |
| `full_typed` | LEGACY | typed heads |
| `full_typed_topo` | LEGACY | typed heads |
| `full_pepa` | LEGACY | PEPA |
| `full_pepa_topo` | LEGACY | PEPA |
| `full_pepa_cape` | LEGACY | PEPA |
| `full_pepa_skel` | LEGACY | PEPA |
| `full_pepa_topo_cape` | LEGACY | PEPA |
| `full_pepa_rank` | LEGACY | PEPA |
| `full_pepa_all` | LEGACY | PEPA |
| `full_pepa_hr` | LEGACY | PEPA |
| `full_pepa_all_hr` | LEGACY | PEPA |
| `full_v2` | LEGACY | v2 |
| `full_v2_skel` | LEGACY | v2 |
| `full_v2_skel_aux` | LEGACY | v2 |
| `full_v3` | LEGACY | v3 |
| `full_dynamic_strip` | ABLATION | single switch |
| `no_geo_i` | ABLATION | single switch |
| `no_geo_e` | ABLATION | single switch |
| `no_c2f` | ABLATION | single switch |
| `no_fusion_v2` | ABLATION | single switch |
| `no_multi_depth` | ABLATION | single switch |
| `no_cl` | ABLATION | single switch |
| `no_ds` | ABLATION | single switch |
| `no_iou` | ABLATION | single switch |
| `geo_e_shallow` | ABLATION | layer placement |
| `geo_e_deep` | ABLATION | layer placement |
| `full_fixed2_aux` | MAIN | fixed2 |
| `full_fixed2_scnp` | CANDIDATE | fixed2 |
| `full_fixed2_resmoe` | CANDIDATE | fixed2 |
| `full_fixed2_resmoe_dg` | FALLBACK | fixed2 |
| `full_fixed2_lora_evp` | CANDIDATE | fixed2 + encoder PEFT |
| `full_fixed2_evp` | CONTROL | fixed2 + encoder PEFT |
| `full_fixed2_convlora` | CONTROL | fixed2 + encoder PEFT |
| `full_fixed2_peft_lora_evp` | CONTROL | fixed2 replacement |
| `full_fixed2_hr_aux` | OPTIONAL | fixed2 high-res |
| `none` | INTERNAL | no preset |

---

## 8. Historical results: what may and may not be quoted

Only quote a result together with:

- preset,
- seed,
- epoch budget,
- evaluation mode (`aspect_pad`/`resize`, empty guard),
- source path.

Known historical anchors (do **not** mix with current 300-epoch results):

- Old `full_fixed2` 800-epoch All Dice ~59.16 (legacy result).
- `full_v2_skel_aux` local 800-epoch run: best validation score ~0.713
  at epoch 286 (see its `metrics_history.json`; not a 20-test result).
- New `full_fixed2_scnp` / `full_fixed2_resmoe` /
  `full_fixed2_resmoe_dg` / `full_fixed2_lora_evp` have **no completed
  current result yet**.

If a path is under `results/paper`, `results/runs_v3test*`,
`results/runs/night_final`, or another ad-hoc root, it is legacy.

---

## 9. Naming guardrails

Use these exact aliases in discussion:

- `B` = `full_fixed2_aux` (never “fixed2” alone without context)
- `B+SCNP` = `full_fixed2_scnp` (not `full_scnp`)
- `B+Repair` = `full_fixed2_resmoe` (learned gate)
- `B+Repair-DG` = `full_fixed2_resmoe_dg` (disagreement gate)
- `B+Hybrid` = `full_fixed2_lora_evp` (not replacement)
- `B+EVP` = `full_fixed2_evp`
- `B+ConvLoRA` = `full_fixed2_convlora`
- `B+Replace` = `full_fixed2_peft_lora_evp`
- `B+HR` = `full_fixed2_hr_aux`

Rules:

- “ResMoE” alone is ambiguous: always say **learned** or **DG**.
- “SCNP” alone is ambiguous: always say **old full_scnp** or
  **fixed2_scnp**.
- “Hybrid” means `full_fixed2_lora_evp`; “replacement” means
  `full_fixed2_peft_lora_evp`.
- “v2/v3” always means the legacy v2/v3 branch, not fixed2.
- “full” is the historical SACM-X full preset, not `full_fixed2_aux`.

---

## 10. How to add a new variant

1. Add the preset to `src/models/sacm_x/configs.py` with a new explicit
   key; inherit from the closest canonical parent.
2. Add a one-click wrapper in `scripts/sacm_x/`.
3. Add a row to this file with alias, preset, parent, status, key switch,
   budget, and result root.
4. Add a preset field test in `tests/`.
5. Add it to the 8-GPU screen only if it is a current CANDIDATE; do not
   expand the screen with legacy presets.
6. Never reuse a historical name with different semantics.
