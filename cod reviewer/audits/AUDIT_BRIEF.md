# All-model audit brief (provisional)

**A0 is unverified**: no calibration pack is frozen. Every report produced under
this brief is a provisional manual pilot. Do not claim G0–G6 all pass and do not
put any model in the main table based on this audit alone.

## Scope

Audit every adapter in `src/baseline_zoo/adapters/registry.py` (22 models).
`src/baseline_zoo/README.MD` maps each adapter to its `3rd/` code directory and
`parsing/` paper directory (if available). `src/baseline_zoo/README.MD` and
`cod reviewer/audits/all_models_inventory.json` are read-only inputs.

## Per-model output

Create `cod reviewer/audits/<adapter_id>/README.md` with:

1. **Header**: official repo/commit, port files, project commit, date.
2. **01 sensitivity inventory**: stable `S-<id>-nn` IDs, official `file:line`,
   expected behavior, failure symptom.
3. **02 discrepancies**: `D-<id>-nn` rows; verdict `differ|equivalent|
   intentional_deviation|unverified`; category; gate; documented/undocumented;
   exact evidence; numeric impact.
4. **01→02 closure**: every S-id has one disposition
   (`covered_by_discrepancy|preserved|not_applicable|unverified`); no blanks.
5. **03 numeric evidence**: commands actually run and their results. If no
   same-state numeric probe was run, write `G2=unverified` explicitly.
6. **04 training recipe**: official vs port table (epochs, lr, batch, loss,
   optimizer, scheduler, transforms, data, workers/GPU, selection).
7. **05 eval protocol**: eval transform, checkpoint, postprocess, threshold,
   metric, export, resolution.
8. **07 provisional gates**: G0–G6a/b/c, `main_table=blocked`,
   `claim_label=...`, and a short list of what is still missing.

## Hard rules

- Use only local `3rd/`, `parsing/`, port, docs, tests. **Do not use web search.**
- Do not edit `src/`, `configs/`, `3rd/`, or `parsing/`; only write under
  `cod reviewer/audits/`.
- Do not install packages or download weights.
- Every claim needs `file:line` + 3–5 lines of source, or a reproducible command
  result. No “整体一致/基本符合/应该相同”.
- Same-weights numeric probe required for G2. If official code cannot be
  constructed or weights cannot be shared, mark `unverified`, not equivalent.
- No official checkpoint → `G6b=N/A`; no full training/eval → `G6c=unverified`;
  no calibration → `A0=unverified`.
- Training-free methods: no loss/grad; compare fixed support/target mask/prompt
  or mark unverified.
- Keep reports concise (roughly 80–150 lines each); one report per model.

## Status vocabulary

`pass | fail | unverified | N/A`

Default for models without numeric probes:

```text
A0=unverified
G0=pass|unverified
G1=unverified|pass (only with state_dict/shape evidence)
G2=unverified
G3=unverified
G4=pass only with explicit evidence, otherwise unverified
G5=unverified
G6a=unverified
G6b=N/A|unverified
G6c=unverified
main_table=blocked
claim_label=none|path_verified|checkpoint_verified
```

Return a final one-line status for every assigned model:
`<id>: G0=... G1=... G2=... G3=... G4=... G5=... G6a=... G6b=... G6c=...`.
