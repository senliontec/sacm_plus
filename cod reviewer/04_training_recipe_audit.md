<task>
从官方仓库提取训练协议，逐项给出官方出处（文件:行号 + 3–5 行原文）与移植版对应位置。
判定必须使用以下分类，禁止“大致相同”：
- `same`：行为一致且有证据
- `official_recipe`：官方配方，移植必须保留；不能保留时必须归为 protocol_adaptation 或 bug_fix
- `protocol_adaptation`：为 18/6 3-shot / 20 数据集 / 二值类数 / 共享 runtime 做的有意改动
- `bug_fix`：官方代码有 bug/泄漏/弃用 API，移植显式修复
- `wrapper_change`：只改包装/命名/设备，需用 L2 证明不改变数值
- `unverified`：缺文件/权重/数据，无法确认

必须同时审计**共享 runtime**：只比 model 文件不算完成。
</task>

<items>
1. 数据来源与划分：18 train / 6 val / 20 test；manifest hash；是否有 val/test 泄漏。
2. support / donor：seen 数据集是否用自身 3 shot；unseen/novel 的 donor 映射；
   support mask 是否非空；support 图像是否参与训练。
3. 训练 transform 的完整有序列表：resize/crop/pad 尺寸与随机策略、插值方式、
   padding 值、flip/rotation/color/noise 概率。
4. Normalize 的 mean/std 来源（ImageNet / 数据集 / 训练集统计 / 固定值），
   是否使用 test-set 统计（官方泄漏必须标出）。
5. ignore_index 的取值，以及在 transform、loss、训练循环三处是否一致。
6. num_classes 来源（config 硬编码 / dataset 统计 / checkpoint 推断 / 二值适配）。
7. batch size、梯度累积、实际 effective batch、每 epoch 迭代数、总迭代数。
8. loss：各项权重、reduction、激活前/后、类别权重、标签平滑、low-res vs full-res 监督。
9. optimizer：param groups、lr、weight decay、是否排除 BN/bias、betas/momentum/eps。
10. scheduler：类型、warmup 步数与方式、总 epoch/iter、poly power、
    **step 的时机**（每 iteration / 每 epoch / 每 accumulation group）。
11. 梯度裁剪：阈值、施加位置、是否在 accumulation 之后。
12. EMA：是否使用、decay、更新时机、是否用于评测。
13. 随机种子与确定性：Python/NumPy/Torch/cuDNN/Albumentations；
    是否有非确定性采样（k-means/AMG/PCA/DSConv）。
14. 预训练权重加载：路径、strict、missing/unexpected 处理、pos_embed/rel_pos 插值、
    加载时机（分布式包装前后）、是否覆盖已初始化层、是否冻结。
15. SyncBN / DDP：loss 是否跨卡平均、BN 是否同步、是否影响有效 batch。
16. DataLoader：num_workers、persistent_workers、sampler、shuffle、drop_last、
    collate_fn。
17. checkpoint 选择：val Dice / train Dice / 官方 test 指标 / 固定 epoch / 最后 epoch；
    是否用 EMA；是否与论文表格口径一致。
18. training-free：无 optimizer/loss 时明确写 N/A，并列出 support 选择、
    prompt 生成、聚合方式、确定性重复。
18b. supervised support-based（如 FoB-SAM）：虽然用 support/episode，仍有
    loss/grad/optimizer；必须按 supervised 全量检查，不能因为“support-based”跳过训练项。
19. oracle-prompted：GT box/point 如何生成、refinement 是否使用 GT、
    阈值/后处理是否依赖 GT。
20. 官方 bug/泄漏：例如 test-set mean/std、test split 选 checkpoint、空 mask 指标；
    保留还是修复，必须显式分类。
</items>

<rules>
1. 每项必须给两侧具体行号 + 原文片段，或明确写“移植版无对应代码”。
2. 无法确认写“无法确认”并说明缺哪个文件/权重/数据。
3. 不得把 adapter 的注释当作证据；以代码实际行为为准。
4. 必须给出有效值，不只给配置值：
   例如 config 写 batch=24，但 18 张图/epoch 时 effective batch 实际是多少；
   scheduler 每 epoch step 时，poly 的“iteration”实际是多少。
5. 区分“官方 per-dataset 配方”和“项目 3-shot 协议”：
   前者不能直接套到后者；套用后的差异必须进入 protocol_adaptation。
6. 对论文表格影响最大的项排在最前，并给出机制（loss 曲线/绝对 Dice/HD95/仅多卡/无）。
</rules>

<output_format>
<table>
| 项 | 官方出处（行号+原文） | 移植版出处（行号+原文或“无对应代码”） | 判定 | 有效值 | 差异说明 | 影响类别 | 是否已在 deviations 披露 |
</table>

<critical>
按“最可能导致论文表格数值偏差”从大到小排序，列出高风险项。
每项必须落到一个可执行验证：官方 checkpoint 对比 / 固定 batch 短程训练 /
固定 seed 两次运行 / 官方 test 入口。
</critical>

<unverified>
缺什么信息、需要补什么文件/权重/数据、缺它会掩盖哪类差异。
</unverified>
