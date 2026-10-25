# A0：审计者校准与资格门禁

> 当前状态：仓库只有本协议和案例矩阵；实际的 `cases/`、`gold/` 尚未冻结，A0 还不能运行。
> 在 A0 完成前，任何真实 baseline 审计一律按 `blocked` 处理。

A0 不是 baseline 门禁，而是**审计者资格门禁**。A0 未通过时，审计者产出的任何
G0–G6a/b/c 结论都无效。它解决的是“20 份报告的召回率未知”的问题，而不是替代
逐 baseline 的证据。

---

## 1. 要校准什么

| 能力 | 失败模式 | 校准方式 |
|---|---|---|
| 敏感点召回 | 漏掉 off-by-one、scheduler step 时机、class head、prompt 语义 | 01 的 S-id 是否覆盖 gold sensitivity |
| 差异召回 | 看到了敏感点但没在 02 输出差异 | gold discrepancy 是否被 02 命中 |
| 误报控制 | 把重命名/import 顺序/等价表达式当差异 | 负控案例必须不报 |
| 证据有效性 | 编造行号、引用不存在片段、用注释当证据 | 逐条核对文件 hash + 行号 + 片段 |
| 门禁正确性 | 把 G6a/G6b 当 G6c、把补充表当主表 | 与 gold gate 对比 |
| 闭环 | 01 的 S-id 在 02 中没有 disposition | closure 必须 100% |
| 过度声明 | 没有等价断言却写 `equivalent`，或没跑 G6c 却写“指标复现” | overclaim 计数必须为 0 |
| 一致性 | 同一案例两次审计结论不同 | 双审计一致率 |

---

## 2. Calibration pack 的构造

- **冻结 + hash**：pack 目录、每个案例、gold 文件都要有 sha256；校准开始后不得修改。
- **审计者看不到 gold**：`cases/` 只给官方/移植快照、profile、资产；`gold/` 只给 scorer。
- **案例类型**：
  - `negative`：语义等价的纯重构，必须不报（或只报 `wrapper_change` + 等价断言）。
  - `positive`：单点植入偏差，必须命中；一次只植入一个，避免归因混乱。
  - `ambiguous`：官方 bug、缺资产、类数改变；必须用 `intentional_deviation` /
    `missing_asset` / `unverified`，不得猜。
- **覆盖矩阵**：resize/normalize、loss 权重、optimizer/scheduler、prompt 语义、
  class head/输出空间、support 泄漏、training-free 阈值/投票、checkpoint 加载、
  metric/导出、`no_grad`/detach。
- **不得泄漏**：真实 baseline 的 gold 不能当 calibration；calibration 案例不得直接
  当作真实审计结果；如复用，必须轮换并重新 hash。
- **难度**：至少 8 个案例，其中 ≥2 个负控、≥4 个 high 正控、≥1 个歧义、≥1 个缺资产。
  案例太容易会高估召回；至少保留 2 个“只有读代码才能发现”的 subtle case。

---

## 3. 最小案例矩阵

| id | kind | 植入变化 | 期望 verdict | 期望 category | 期望 gate | 考什么 |
|---|---|---|---|---|---|---|
| CAL-01 | negative | 变量重命名/import 重排 | 不报 | `wrapper_change` | G1/G2 | 误报控制 |
| CAL-02 | positive high | eval resize 插值 bilinear→nearest | `differ` | `official_recipe` | G4 | 预处理/导出 |
| CAL-03 | positive high | loss 权重 0.5→1.0 | `differ` | `official_recipe` | G3 | loss 配方 |
| CAL-04 | positive high | scheduler 每 epoch step→每 iter step | `differ` | `official_recipe` | G3 | step 时机 |
| CAL-05 | positive high | 去掉空 prompt / `no_mask_embed` | `differ` | `official_recipe` | G1/G2 | prompt 语义 |
| CAL-06 | positive high | support 取自 test/val | `differ` | `protocol_adaptation` | G5 | 泄漏 |
| CAL-07 | positive high | 8 类 head→2 类且未用官方同配置 | `unverified`/`differ` | `protocol_adaptation` | G1/G2 | 类数/输出空间 |
| CAL-08 | positive high | training-free 阈值/投票规则改动 | `differ` | `official_recipe` | G4 | 推理/聚合 |
| CAL-09 | ambiguous | 官方 bug 保留 vs 修复 | `intentional_deviation` | `bug_fix` | G4/G6c | 官方 bug 分类 |
| CAL-10 | missing asset | 官方 checkpoint 缺失 | `unverified` | `missing_asset` | G6b | 不猜、不伪造 |
| CAL-11 | negative | `.cuda()`→`.to(device)` | 不报/等价 | `wrapper_change` | G1/G2 | 设备包装 |
| CAL-12 | positive high | support 编码去掉 `torch.no_grad()` | `differ` | `official_recipe` | G2/G3 | 计算图/显存 |

gold 文件必须为每个案例写清：

- `expected_sensitivity_ids`（01 应产出的 S-id）；
- `expected_discrepancies`（D-id、verdict、category、impact、gate、undocumented）；
- `required_evidence`（官方/移植各 3–5 行、文件 sha256、可执行断言）；
- `allowed_unverified`（只有确实缺资产时允许）；
- `forbidden_claims`（例如未跑 G6c 时不得写 `table_metric_reproduced`；G6c pass 时
  也不得自动写 `official_metric_match=verified`）。

---

## 4. 评分

对每个案例，用审计者正常的 `01 → 02 → 07` 输出与 gold 对比：

| 指标 | 定义 | A0 阈值 |
|---|---|---|
| `sensitivity_recall_01` | 01 命中 gold sensitivity / gold sensitivity 总数 | ≥ 0.9 |
| `recall_high_02` | 02 命中 high gold discrepancy / high gold 总数 | = 1.0 |
| `recall_all_02` | 02 命中全部 gold discrepancy / gold 总数 | ≥ 0.9 |
| `precision_02` | 真实差异 / 所有上报差异（负控误报计入分母） | ≥ 0.9 |
| `evidence_validity` | 片段存在、文件 hash 匹配、行号误差 ≤2、断言可执行 | = 1.0 |
| `closure` | 01 S-id 有 disposition / 01 S-id 总数 | = 1.0 |
| `gate_accuracy` | gate 判定与 gold 一致 / 案例数 | ≥ 0.9 |
| `overclaim` | 无断言写 equivalent、无 G6c 写本表指标复现、把 G6a/G6b 当 G6c、把 G6c 当官方数字对齐 | = 0 |
| `consistency` | 双审计结论 Jaccard / 一致率（高风险案例） | ≥ 0.8 |

**A0 pass**：以上全部满足，且 **没有任何 high-severity 漏报**。
**A0 fail**：任一 high 漏报、证据无效、closure < 1、overclaim > 0，或任一阈值未达。
A0 fail 时，审计者的报告不得进入主表或 supplement；修 prompt/模型后必须重跑 A0。

---

## 5. 报告 schema

```json
{
  "auditor": {
    "model": "<str>",
    "prompt_sha256": "<str>",
    "protocol_version": "<str>",
    "run_id": "<str>"
  },
  "pack": {
    "sha256": "<str>",
    "case_count": 12,
    "high_case_count": 8
  },
  "cases": [
    {
      "case_id": "CAL-01",
      "kind": "negative|positive|ambiguous|missing_asset",
      "sensitivity_recall_01": 1.0,
      "recall_02": 1.0,
      "precision_02": 1.0,
      "evidence_validity": 1.0,
      "closure": 1.0,
      "gate_accuracy": 1.0,
      "overclaim": 0,
      "a0_case": "pass|fail"
    }
  ],
  "aggregate": {
    "sensitivity_recall_01": 0.0,
    "recall_high_02": 0.0,
    "recall_all_02": 0.0,
    "precision_02": 0.0,
    "evidence_validity": 0.0,
    "closure": 0.0,
    "gate_accuracy": 0.0,
    "overclaim": 0,
    "consistency": 0.0
  },
  "a0": "pass|fail",
  "missed": ["CAL-..."],
  "false_positives": ["CAL-..."],
  "invalid_evidence": ["CAL-..."],
  "closure_gaps": ["CAL-..."],
  "notes": "<str>"
}
```

报告按 `${auditor_run_id}.json` 保存；真实审计的 context pack 必须引用它
（`auditor.calibration_report`）并记录 `a0`、`recall_*`、`precision`。

---

## 6. 双审计与裁决

- **高风险 baseline**（UCS、Matcher、SAMed、DSCNet、PerSAM、任何有 bug_fix 的）
  必须由两个独立审计者各跑一遍；比较 findings 的并集与交集。
- 任一审计者漏掉 high gold（校准）或 high real finding（双审计）→ 该审计者本轮
  A0 记 fail；不得用另一个审计者的结果掩盖。
- 双审计结论冲突时，由第三审计者/人工裁决；裁决必须写进 ledger，不能只改结论。
- 报告 `consistency` = 两个审计者 findings 的 Jaccard；低于 0.8 时增加抽样人工复核。

---

## 7. 边界

1. A0 只测量**已植入**偏差的召回率，不能证明未植入偏差一定被发现。
2. Calibration pack 不能包含真实 baseline 的 gold；不能边看结果边改案例。
3. A0 pass 不代表 baseline 通过；它只说明该审计者的召回/证据/门禁行为达到最低标准。
4. 每次 prompt、模型、协议变更后，以及每批真实审计前，都要重新跑 A0。
5. 没有 A0 报告的真实审计，按 `blocked` 处理，不得进入主表或 supplement。
