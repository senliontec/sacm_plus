<task>
从官方仓库提取以下训练协议项，逐项给出官方出处（文件:行号 + 3–5 行原文）与移植版对应位置，判定 一致 / 不一致 / 无法确认。
</task>

<items>
1. 训练 transform 的完整有序列表，每项参数（插值方式、padding 值、crop 尺寸与随机策略、flip 概率）
2. Normalize 的 mean/std 来源（ImageNet / 数据集特定 / 其他）
3. ignore_index 的取值，以及在 transform、loss、训练循环三处是否一致
4. num_classes 来源（config 硬编码 / dataset 统计 / checkpoint 推断）
5. batch size 与 total iterations 实际生效值；lr 是否按 linear scaling rule 缩放
6. optimizer 的 param groups、weight decay 是否排除 BN 与 bias、momentum/betas
7. scheduler 类型、warmup 步数与方式、总 iter/epoch、poly power
8. 梯度裁剪阈值与施加位置
9. EMA 是否使用、decay、更新时机
10. 随机种子设置位置与范围（是否固定 cudnn benchmark/deterministic）
11. 预训练权重加载：路径、是否 strict、加载时机（是否在分布式包装前后）、是否覆盖已初始化层
12. 是否使用 SyncBN / DDP 下 loss 是否跨卡平均
13. 数据加载器：num_workers、persistent_workers、sampler 类型、drop_last
</items>

<rules>
禁止"大致相同"。每项必须给出两侧具体行号 + 原文片段，或明确写"移植版无对应代码"。无法确认写"无法确认"并说明缺哪个文件。
</rules>

<output_format>
<table>| 项 | 官方出处 | 移植版出处 | 判定 | 差异说明 | 影响类别（loss曲线/mIoU绝对值/仅多卡/无） |</table>
<critical>按"最可能导致论文表格数值偏差"从大到小排序，列出高风险项</critical>
<unverified></unverified>