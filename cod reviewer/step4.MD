<task>
对比下面两份代码，只输出会导致数值或行为差异的条目。不评价质量，不输出总结。
</task>

<rules>
1. 每条格式：`[high|med|low] [numeric|training|eval|none] 官方:行号 -> 移植:行号 | 差异 | 验证方法`
2. 纯格式差异不写。
3. 结束单独一行：`UNVERIFIED: <无法确认项，逗号分隔>`。
4. 无论 high 条目多少，末尾都必须追加一行：`COVERAGE: <你检查过的维度列表，逗号分隔>`。
5. high 条目为 0 时，追加一行：`ZERO_HIGH_JUSTIFICATION: <举证>`。
6. 不猜测未提供文件的内容；缺失引用列入 UNVERIFIED。
</rules>

<official_files>
  <file path="..." role="model_def">...全文...</file>
</official_files>
<ported_files>
  <file path="..." role="model_def">...全文...</file>
</ported_files>