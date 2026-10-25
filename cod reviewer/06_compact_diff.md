<task>
对比官方实现与移植实现，只输出会导致 L1–L5 数值或行为差异的条目。
不评价质量，不输出总结，不提出修复建议。
</task>

<rules>
0. 前置：A0=pass 且 SENS_CLOSURE=1.0；否则只输出 `BLOCKED:` 与原因，不输出 diff 条目。
1. 每条格式：
   `[level] [high|med|low] [numeric|training|eval|protocol|none] [differ|intentional_deviation|unverified] [category=official_recipe|protocol_adaptation|bug_fix|wrapper_change|missing_asset|unverified] 官方:行号 -> 移植:行号 | 差异 | 验证方法`
2. 纯格式差异不写；重命名改变绑定/作用域时必须写。
3. 必须覆盖：state_dict 映射、forward、loss/grad、optimizer/scheduler、
   transform/normalize、prompt/support、checkpoint、postprocess、metric、18/6 协议。
4. 不要求 state_dict 键集合相等；expected_missing/expected_extra 要说明原因，
   unexpected_missing/extra 才是风险。
5. 官方 bug/泄漏：verdict=`intentional_deviation`，category=`bug_fix`，说明保留/修复方向；不要发明 `official_bug` 这个 verdict。
6. `training_free_support_based_sam` 条目不得用 loss/grad 缺失判 fail；改比 mask/prompt/
   support 选择/确定性。`supervised_support_based_sam` 仍有 loss/grad。
7. 无法确认的写 `UNVERIFIED:`，列出缺哪个文件/权重/数据。
8. 末尾必须追加：
   `COVERAGE: <检查过的维度列表，逗号分隔>`
   `A0=... CLOSURE=...`
   `GATES: G0=... G1=... G2=... G3=... G4=... G5=... G6a=... G6b=... G6c=...`
   `CLAIM_LABEL: table_metric_reproduced|path_verified|checkpoint_verified|none`
   `OFFICIAL_METRIC_MATCH: verified|unverified|N/A`
9. 禁止裸 `G6=...`；必须分别写 G6a/G6b/G6c；只有 G6c 能支持 `table_metric_reproduced`；
   它不代表官方论文数字一致，官方对齐单独写 `OFFICIAL_METRIC_MATCH`。
10. high 条目为 0 时追加：
   `ZERO_HIGH_JUSTIFICATION: <为什么确实只有低风险差异，举证>`
11. 不得新增原清单中不存在的差异；遗漏项放 `POSSIBLE_OMISSIONS:`；
    01 遗漏必须回写 01 并重跑，不得直接塞进 compact diff。
</rules>

<official_files>
  <!-- 必须粘贴全文；只给路径视为不存在 -->
</official_files>
<ported_files>
  <!-- 必须粘贴全文；只给路径视为不存在 -->
</ported_files>
<shared_runtime>
  <!-- runtime.py / data.py / support.py / protocols.py 的相关段落全文 -->
</shared_runtime>
<profile><!-- supervised_prompt_free_sam / supervised_oracle_prompted_sam / supervised_support_based_sam / training_free_support_based_sam / semi_supervised_prompt_free_sam / supervised_standard_non_sam / probe_standard_sam --></profile>

<output_format>
逐行 diff 条目
UNVERIFIED: <...>
POSSIBLE_OMISSIONS: <...>
COVERAGE: <...>
A0=pass|fail CLOSURE=<0..1>
GATES: G0=... G1=... G2=... G3=... G4=... G5=... G6a=... G6b=... G6c=...
CLAIM_LABEL: table_metric_reproduced|path_verified|checkpoint_verified|none
OFFICIAL_METRIC_MATCH: verified|unverified|N/A
[ZERO_HIGH_JUSTIFICATION: ...]
</output_format>
