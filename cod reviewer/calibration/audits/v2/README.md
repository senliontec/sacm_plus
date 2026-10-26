# A0 v2 calibration attempt

- Pack: `a0-cal-2026-10-26-v2`
- Report: `calibration_report.json`
- Result: **fail** (`a0_pass=false`, consistency 0.917)

Both auditors missed the high-severity CAL-07 class-head case as
`intentional_deviation`, because the v2 context explicitly said the port
targeted the project's binary protocol. That sentence effectively disclosed
the adaptation and made the change documented by construction. Auditor A also
chose G2 for the CAL-03 loss-weight change (G3 expected); with CAL-07 corrected,
that single gate disagreement is within the 0.9 gate threshold.

v3 removes that disclosure from CAL-07 and keeps the gold verdict unchanged
(`differ`/`unverified`, undocumented, G1/G2). No gold answers were relaxed.
