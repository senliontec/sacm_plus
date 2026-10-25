<task>
下面是某个 baseline 的保真差异清单。你的任务不是重新审计代码，而是：
1. 剔除确实不构成数值/行为差异的条目；
2. 按“最可能导致论文表格数值偏差”排序；
3. 对每条给出一个具体、可当天执行的验证动作（命令级，不是“建议检查”）；
4. 标出哪些条目必须用官方 checkpoint / 官方数据 / 短程训练 / 目标 GPU 才能确认；
5. 给出主表/附录门禁结论。
</task>

<rules>
1. 剔除某条目，必须满足其一：
   (a) 原条目 verdict="equivalent"，且原条目给出了可验证的等价断言（具体 shape/边界/广播/插值/随机数消耗），而不是“应该相同”；你只需核对断言是否成立，不得用新的等价论证替代；
   (b) 纯文本差异且原描述已明确不含语义差异；
   (c) expected_missing/expected_extra 且已给出结构原因（类数变化、裁剪、wrapper），
       且不影响 shared weights/forward。
   不满足者一律保留进 ranked，即使你认为它是噪声。
2. 被剔除的条目不得删除，原样放进 <dropped> 并附剔除理由。
3. 不得新增原清单中不存在的差异条目。若认为原清单遗漏明显高风险项，
   单独放 <possible_omissions> 并说明依据（例如共享 runtime 未审计、support/donor 未审计、
   metric 未审计）。
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
   - 主表：G0–G6 是否全部 pass；G6 unverified 即不能进主表；
   - G6 pass = supervised：官方 checkpoint 固定 batch 输出在容差内，或官方代码+
     同一份数据（官方或项目 18/6）+同一初始化/state_dict（能共享时）固定 seed 的 N≥20 optimizer steps loss/grad 在容差内；
     training-free：同 support/target + 官方权重下 mask/prompt 在容差内；
     短程 parity 只证明训练路径一致，不证明全量训练后的精度；
   - G6 unverified = 缺官方 checkpoint 且无法短程 parity，或缺可用的同一份数据/入口；G6 fail = 有资产但超差；
   - gate 聚合：同一 gate 内任一必需子项 unverified → 该 gate unverified；任一子项 fail → fail；
   - G0–G5 任一 fail/unverified → `main_table=blocked`；
     G0–G5 全 pass 且仅 G6 unverified → `main_table=supplement_only`；
   - 不允许主表：列出阻断项、未披露 intentional deviation、unverified。
</rules>

<discrepancy_list>
{{02_port_vs_official_audit.md 的完整输出}}
</discrepancy_list>

<context>
baseline / profile / training_mode / prompt_modality / backbone / paper_group /
official_commit / assets / known_deviations
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
  G0=... G1=... G2=... G3=... G4=... G5=... G6=...
  G6_reason=...
  main_table=pass|blocked|supplement_only
  blocking_items=[...]
  undisclosed_intentional_deviations=[...]
  unverified=[...]
</gate_status>
</output_format>
