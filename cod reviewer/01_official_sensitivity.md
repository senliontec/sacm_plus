<task>
只读官方实现，不看任何移植代码。输出一份“移植敏感点清单”：
如果要把这个模块搬到另一个项目，哪些地方必须逐字保留，否则会改变 L1–L5 行为或 L6 复现。
</task>

<context>
使用 context pack 中的官方全文。缺外部引用时写入 <unverified>，不猜测内容。
</context>

<rules>
1. 每条绑定官方文件:行号 + 3–5 行原文片段。
2. 只列数值/行为敏感点，不列架构概述、不列设计优点。
3. 每条给出：如果被改掉，最可能表现为什么现象
   （loss 不收敛 / 指标低几个点 / 只在 eval 出错 / 静默加载权重失败 /
   只在某类数据上错 / 只有 GPU 上错 / 只在奇数尺寸错 等）。
4. 同时列出“官方看起来奇怪但必须原样保留”的写法（冗余、弃用 API、可疑默认值、泄漏）。
5. 区分“必须保留”和“官方 bug，移植时应显式决定保留还是修复”。
6. 未提供的外部引用列入 <unverified>，说明缺哪个文件、缺它会掩盖哪类差异。
</rules>

<categories>
- 权重加载：路径、strict、missing/unexpected、pos_embed/rel_pos 插值、class head shape、
  LoRA/Adapter 注入前后、冻结时机
- 初始化与 RNG：初始化函数参数、调用顺序、dropout/droppath、随机采样
- 预处理：颜色顺序、 resize 插值、padding 值/位置、normalize 来源、mask 0/1 vs 0/255
- 训练 transform：完整有序列表、随机策略、几何/颜色增强、crop/pad
- loss：BCE/CE/Dice/clDice、reduction、激活前/后、类别权重、ignore_index
- optimizer/scheduler：param groups、wd 排除、betas/momentum、warmup、poly、step 时机
- prompt 语义：GT box/point、自动 point、support mask、空 prompt、no_mask_embed
- 推理：eval transform、checkpoint/EMA、argmax/threshold、GT 分辨率、TTA
- metric：Dice/IoU/clDice/HD95、空 mask、逐图/逐数据集、跨卡
- 数据/协议：split、18/6、support/donor、manifest、DataLoader
- 非确定性：k-means/AMG/PCA/DSConv/CUDA 算子
</categories>

<output_format>
<sensitivity_inventory>
  <item ref="文件:行号 | 3–5 行原文" category="" must_preserve=""
        failure_symptom="" confidence="high|medium|low"
        official_bug="true|false" />
  ...
</sensitivity_inventory>
<oddities>
  看起来奇怪但必须原样保留，或必须显式决定是否修复的写法
</oddities>
<unverified>
  缺什么外部文件/权重/数据；缺它会掩盖什么
</unverified>
</output_format>
