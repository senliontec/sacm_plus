# A0 v3 calibration result

- Pack: `a0-cal-2026-10-26-v3`
- Pack SHA-256: `df593177d7213f6f1452f61062a41a67539acaef3009a0c43feb9e5905cedd1a`
- Report: `calibration_report.json`
- Result: **pass** (`a0_pass=true`)

| auditor | high recall | all recall | precision | evidence | closure | gate | overclaim |
|---|---|---|---|---|---|---|---|
| run-a | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |
| run-b | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9 | 0 |

Dual-audit consistency: 0.917. Both auditors passed the A0 thresholds.
The only gate disagreement is CAL-03: run-a chose G3, run-b chose G2 (G3 is
the gold gate); run-b still passes the aggregate 0.9 threshold.

Scope: this qualifies the auditor/prompt pair that produced v3, not the 22
pre-A0 model reports. Those reports must be re-audited under this qualification.
