# A0 calibration audit output contract

## Input

Audit only the case directory you are given (`cases/<CASE_ID>/`). Do not read
`calibration/gold/`, and do not use knowledge of other cases.

Each case provides:

- `case.json` — case id, profile, visible files, task;
- `official.py` / `port.py` — synthetic official and port snapshots;
- `context.md` — profile and case context;
- optional `deviations.md` and `assets.json`.

## Output

Write one JSON file per case, named `<CASE_ID>.json`:

```json
{
  "case_id": "CAL-02",
  "sensitivity_keys": ["eval_resize_interpolation"],
  "closure": {
    "eval_resize_interpolation": "covered_by_discrepancy"
  },
  "discrepancies": [
    {
      "key": "eval_resize_interpolation",
      "verdict": "differ",
      "category": "official_recipe",
      "impact": "numeric",
      "gate": "G4",
      "undocumented": true,
      "evidence": [
        {
          "file": "official.py",
          "line": 8,
          "snippet": "mode='bilinear'",
          "sha256": "<sha256 of official.py>"
        },
        {
          "file": "port.py",
          "line": 7,
          "snippet": "mode='nearest'",
          "sha256": "<sha256 of port.py>"
        }
      ]
    }
  ],
  "claims": {
    "main_table": "blocked",
    "claim_label": "none",
    "official_metric_match": "unverified"
  }
}
```

## Gate mapping (protocol context)

Use the same G0–G5 semantics as the main audit protocol:

- **G0 source**: official repo/commit/files/dependencies and license/weights
  provenance are traceable.
- **G1 structure**: key mapping, shapes, parameters, buffers and initialization
  are accounted for.
- **G2 numeric**: same weights and fixed input/seed; forward/intermediate/loss/
  gradient/optimizer-step parity within tolerance.
- **G3 training**: training recipe and runtime semantics, including optimizer,
  scheduler, accumulation, clipping, EMA, seed, **autograd context and memory
  behavior**; no undisclosed training difference.
- **G4 inference**: eval transform, checkpoint selection, postprocess, threshold,
  metric and export; no unexplained eval difference.
- **G5 protocol**: 18/6/support/donor/test-root/manifest/GT-export roles;
  **support or donor must not come from test/val**; all protocol adaptations
  must be disclosed.

G6a/G6b/G6c are separate path/checkpoint/table-metric gates; they are not
substitutes for G0–G5.

### Field rules

- `sensitivity_keys`: stable snake_case names for the official-sensitive
  points you inspected. Put at least one key in a positive case; do not list a
  key you did not inspect.
- `closure`: every key in `sensitivity_keys` must have a disposition:
  `preserved`, `covered_by_discrepancy`, `missing_asset`, or `unverified`.
- `discrepancies[].key`: stable snake_case identifier. The scorer also matches
  by evidence, so a different but reasonable key name is not automatically a
  miss.
- `verdict`: one of `differ`, `equivalent`, `intentional_deviation`,
  `unverified`.
- `category`: one of `official_recipe`, `protocol_adaptation`, `bug_fix`,
  `wrapper_change`, `missing_asset`, `unverified`.
- `impact`: one of `numeric`, `training`, `eval`, `protocol`, `none`.
- `gate`: one of `G0`–`G5`, `G6a`, `G6b`, `G6c`.
- `undocumented`: true when the difference is a real deviation without an
  explicit disclosure in the case files.
- `evidence`: at least one entry per discrepancy. `file` is relative to the
  case directory. `sha256` must be the full-file SHA-256. `snippet` must be an
  exact substring of the file. `line` must point to the first line of the
  snippet, within ±2 lines.
- `claims.main_table` must be `blocked` for A0 cases: no G6c run exists.
  `claim_label` must not be `table_metric_reproduced`.
  `official_metric_match` must not be `verified` unless the case explicitly
  supplies a matched official metric, which none do.

## Negative controls

For a negative/refactor case, do not report a semantic discrepancy. If you
report an equivalent wrapper change, use verdict `equivalent` and cite both
sides; it is treated as neutral only for the recognized wrapper keys. A
`differ`/`unverified` finding on a negative case is a false positive.

## Grading summary

The scorer computes sensitivity recall, discrepancy recall, precision,
evidence validity, closure, gate accuracy, overclaim count, and dual-audit
consistency. See `00_auditor_calibration.md` for thresholds and the report
schema. A high-severity miss, invalid evidence, closure < 1, overclaim, or any
metric below threshold fails A0.

This package is synthetic. It is not a real baseline result and must not be
quoted as one.
