<task>
下面是某个 baseline 的移植差异清单。你的任务不是重新审计代码，而是：
1. 剔除其中确实不构成数值/行为差异的条目
2. 把剩余条目按"最可能导致论文表格数值偏差"排序
3. 对每条给出一个具体、可当天执行的验证动作（命令级别，不是"建议检查"）
4. 标出哪些条目必须用官方 checkpoint 加载 + 短程训练才能确认，无法靠读代码确认
</task>

<rules>
1. 剔除某条目，必须满足其一：
   (a) 该条目在原清单中 verdict="equivalent" 且你能基于描述本身指出等价论证的漏洞；
   (b) 该条目属于纯文本差异且原清单描述中已明确不含语义差异。
   不满足者一律保留进 ranked，即使你认为它是噪声。
2. 被剔除的条目不得删除，原样放进 <dropped> 并附剔除理由。
3. 不得新增原清单中不存在的差异条目。若认为原清单遗漏明显高风险项，单独放 <possible_omissions> 并说明依据。
4. 排序时优先把 verdict="differ" 且 impact="numeric" 的条目靠前；verdict="intentional_deviation" 单独标注"非移植缺陷"。
</rules>

<discrepancy_list>{{Prompt A 的完整输出}}</discrepancy_list>

<output_format>
<ranked>
  1. [id] [严重度] [影响类别] 验证动作（命令级）
  ...
</ranked>
<dropped>
  [id] 剔除理由
  ...
</dropped>
<possible_omissions></possible_omissions>