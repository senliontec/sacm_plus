<task>
下面是某个 baseline 的保真差异清单。你的任务不是重新审计代码，而是：
1. 剔除确实不构成数值/行为差异的条目；
2. 按“最可能导致论文表格数值偏差”排序；
3. 对每条给出一个具体、可当天执行的验证动作（命令级，不是“建议检查”）；
4. 标出哪些条目必须用官方 checkpoint / 官方数据 / 短程训练 / 目标 GPU 才能确认；
5. 先检查 A0 与 01→02 closure；前置不满足时直接 blocked；
6. 给出主表/附录门禁结论。
</task>

<rules>
0. 前置：A0=pass 且 SENS_CLOSURE=1.0；否则只输出 <blocked>，不得排序或给出 G0–G6c 结论。
1. 剔除某条目，必须满足其一：
   (a) 原条目 verdict="equivalent"，且原条目给出了可验证的等价断言（具体 shape/边界/广播/插值/随机数消耗），而不是“应该相同”；你只需核对断言是否成立，不得用新的等价论证替代；
   (b) 纯文本差异且原描述已明确不含语义差异；
   (c) expected_missing/expected_extra 且已给出结构原因（类数变化、裁剪、wrapper），
       且不影响 shared weights/forward。
   不满足者一律保留进 ranked，即使你认为它是噪声。
2. 被剔除的条目不得删除，原样放进 <dropped> 并附剔除理由。
3. 不得新增原清单中不存在的差异条目。若认为原清单遗漏明显高风险项，
   单独放 <possible_omissions> 并说明依据（例如共享 runtime 未审计、support/donor 未审计、
   metric 未审计）。01 遗漏必须走 02 的 `new_finding` 回写 01；未映射差异不得进入 ranked。
4. 排序优先级：
   - `differ` + `impact=numeric/protocol` 最前；
   - `intentional_deviation` 单独标注“非移植缺陷”，但若未披露则升级；
   - `unverified` 不得排到“已验证”之前；
   - `equivalent`/纯格式 last。
5. 每条验证动作必须能当天执行，包含：
   - 静态：grep/读取具体行的断言；
   - 数值：固定输入/固定 seed 的官方 vs 移植探针；
   - 短程：固定 batch 的 N-step loss/grad 对比；
   - 全协议：官方 checkpoint 在固定图像上的输出/metric 对比；
   - 需要资产：明确官方权重/数据路径；缺失则写 BLOCKED。
6. 最后给出 gate 结论：
   - 主表：A0 pass + SENS_CLOSURE=1.0 + G0–G5 pass + G6a pass + G6c pass；
     声明官方权重/前向复现时还必须 G6b pass。
   - G6a = 同一预处理 batch + 同一初始化/state_dict（能共享时）+ 固定 seed 的
     N≥20 optimizer steps loss/grad parity；只证明训练路径。
   - G6b = 官方 checkpoint 固定输入矩阵 forward/输出 parity；只证明官方资产/前向。
   - G6c = 全量训练/评测、保存 masks、重算指标、复跑 eval、seed 均值/方差；
     **只有 G6c 能支持“本表指标可复现”**，它不代表官方论文数字一致。
   - `official_metric_match` 是独立字段；只有同一协议/数据/checkpoint 或同口径官方数字
     才可标 `verified`，否则 `unverified`。
   - 禁止裸 `G6=pass`；G6a/G6b/G6c 必须分别输出。G6b 可为 N/A，但不能代替 G6c。
   - G6a/G6b/G6c 各自聚合：任一必需子项 unverified → 该子门禁 unverified；
     任一子项 fail → fail；不得互相抵消。
   - A0 或 closure fail，或 G0–G5 任一 fail/unverified，或 G6a/G6c fail/unverified →
     `main_table=blocked`；G0–G5+G6a 全 pass、仅 G6c unverified → `main_table=supplement_only`。
   - 只有 G6a/G6b 通过而 G6c 未跑：`claim_label=path_verified|checkpoint_verified`，
     不得写 `table_metric_reproduced`。
   - 不允许主表：列出阻断项、未披露 intentional deviation、unverified、closure gaps。
</rules>

<discrepancy_list>
{{02_port_vs_official_audit.md 的完整输出}}
</discrepancy_list>

<context>
baseline / profile / training_mode / prompt_modality / backbone / paper_group /
official_commit / assets / known_deviations / auditor_calibration / sensitivity_inventory /
sensitivity_closure
</context>

<output_format>
<ranked>
  1. [id] [level] [severity] [impact] [verdict] 验证动作（命令级）
     expected_result / failure_signal
  ...
</ranked>
<dropped>
  [id] 剔除理由
  ...
</dropped>
<possible_omissions>
  ...
</possible_omissions>
<gate_status>
  A0=pass|fail calibration_report=<id> recall_high=<...> recall_all=<...> precision=<...>
  SENS_CLOSURE=<0..1> open_items=[...] new_findings=[...] unmapped_findings=[...]
  G0=... G1=... G2=... G3=... G4=... G5=...
  G6a=pass|fail|unverified
  G6b=pass|fail|unverified|N/A
  G6c=pass|fail|unverified
  G6_reason=G6a:...; G6b:...; G6c:...
  claim_label=table_metric_reproduced|path_verified|checkpoint_verified|none
  official_metric_match=verified|unverified|N/A
  main_table=pass|blocked|supplement_only
  blocking_items=[...]
  closure_gaps=[...]
  undisclosed_intentional_deviations=[...]
  unverified=[...]
</gate_status>
前置不满足时只输出：
<blocked reason="A0/closure/asset gap">...</blocked>
</output_format>
