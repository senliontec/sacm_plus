# A0 auditor calibration pack

This directory contains a frozen, synthetic A0 calibration pack. It is not a
baseline audit and must not be used as a paper result.

## Status

- Pack version: `a0-cal-2026-10-26-v3`
- Case count: 12
- High-severity cases: 8
- Negative controls: 2
- Ambiguous cases: 1
- Missing-asset cases: 1
- A0 status: **pass** (dual audit, 2026-10-26).
- Pack SHA-256: `df593177d7213f6f1452f61062a41a67539acaef3009a0c43feb9e5905cedd1a`
- Report: `cod reviewer/calibration/audits/v3/calibration_report.json`
- v1/v2 failed for pack-design ambiguity and are retained under `audits/v1/`
  and `audits/v2/`; v3 is the qualified pack.

## Layout

```
cases/CAL-01/ ... CAL-12/   # input visible to the auditor
gold/CAL-01.json ...        # hidden expected result; scorer only
AUDIT_OUTPUT.md             # output contract
score_calibration.py        # scorer
pack_manifest.json          # per-file sha256 + pack sha256
SHA256SUMS                  # plain sha256 list
```

The auditor receives only a copy of `cases/` and `AUDIT_OUTPUT.md`. Do not give
the auditor `gold/`, `pack_manifest.json`, or this README.

## Freeze and hash

`pack_manifest.json` hashes every file under `cases/` and `gold/`, plus
`AUDIT_OUTPUT.md`. `SHA256SUMS` is a plain-text digest list for all pack files.
A pack whose hashes do not match must not be used for A0; create a new pack
version instead of editing a frozen case.

## Run the dual audit

1. Copy `cases/` to an isolated directory and give each auditor only that copy
   plus `AUDIT_OUTPUT.md`.
2. Each auditor writes `<CASE_ID>.json` for all 12 cases into a run directory,
   e.g. `audits/run-a/` and `audits/run-b/`.
3. Score:

```bash
python 'cod reviewer/calibration/score_calibration.py' \
  --audit-dir 'cod reviewer/calibration/audits/run-a' \
  --audit-dir 'cod reviewer/calibration/audits/run-b' \
  --out 'cod reviewer/calibration/calibration_report.json'
```

The scorer verifies the pack manifest, evidence hashes/snippets, closure,
overclaim rules, and the A0 thresholds in
`../00_auditor_calibration.md`.

## What this pack can and cannot establish

It measures recall on planted synthetic differences and false-positive control
on refactors. It does not prove that unplanted real-baseline differences will
be found. It is a qualification gate, not evidence that any baseline is
faithful or that any paper metric is reproducible.
