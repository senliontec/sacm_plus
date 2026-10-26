# A0 v1 calibration attempt

- Pack: `a0-cal-2026-10-26-v1`
- Report: `calibration_report.json`
- Result: **fail** (`a0_pass=false`, consistency 0.833)

Both auditors used the v1 cases. Auditor B had full recall but gate accuracy
0.80 because the input did not include the G0–G5 semantics. Auditor A also
classified the ambiguous CAL-05 learned-prompt case as `unverified` (0.875
high recall). These are pack-design gaps, not proof that the auditors are
unqualified.

v2 fixes the pack, not the gold answer:

- adds the G0–G5 gate mapping to `AUDIT_OUTPUT.md`;
- makes CAL-05's learned prompts explicit parameters instead of undefined
  attributes.

v1 outputs are retained only as calibration history.
