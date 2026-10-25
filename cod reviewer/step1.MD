<task>
只读官方实现，不要看任何移植代码。输出一份"移植敏感点清单"：如果要把这个模块搬到另一个项目，哪些地方必须逐字保留，否则会改变数值或训练行为。
</task>

<rules>
1. 每条必须绑定官方文件的行号 + 3–5 行原文片段。
2. 只列数值/行为敏感点，不列架构概述、不列设计优点。
3. 每条给出：如果被改掉，最可能表现为什么现象（loss 不收敛 / 指标低几个点 / 只在 eval 出错 / 静默加载权重失败等）。
4. 同时列出"官方代码中看起来奇怪但必须原样保留"的写法（冗余、弃用 API、可疑默认值）。
5. 若发现未提供的外部引用，列入 <unverified>，不猜测内容。
6. 本清单供用户与 Prompt A 输出人工取并集，你不需要对照任何其他清单。
</rules>

<official_files>
  <file path="..." role="model_def">...全文...</file>
  <file path="..." role="dependency">...全文...</file>
</official_files>

<output_format>
<sensitivity_inventory>
  <item ref="文件:行号 | 原文片段" category="" must_preserve="" failure_symptom="" confidence="high|medium|low" />
</sensitivity_inventory>
<oddities>看起来奇怪但必须原样保留的写法</oddities>
<unverified></unverified>
</output_format>