<task>
从官方仓库提取评测/推理协议，逐项给出官方出处（文件:行号 + 3–5 行原文）与移植版对应位置。
判定使用：`same` / `official_recipe` / `protocol_adaptation` / `bug_fix` / `wrapper_change` / `unverified`。
必须覆盖共享 runtime 的 metric 与导出路径，不只比模型 forward。
</task>

<items>
1. 验证/测试 transform 的完整有序列表：是否 resize、center crop、pad、插值方式、
   mask 插值、颜色转换。
2. 推理是否复用训练 dataset 的 transform；test 脚本是否悄悄改了 normalize 或 resize。
3. 评测输入分辨率：固定尺寸 / 长边 / 原图；是否多尺度；是否 TTA；翻转增强细节。
4. metric 定义（本项目）：Dice / IoU / clDice / HD95 的实现来源、阈值、空 mask 约定、
   逐图先算再平均还是逐数据集平均、是否跨卡 all_reduce。
   官方若是 mIoU/逐类 Dice/HD95，必须说明映射关系。
5. eval batch size 与训练不同时的 BN 行为：是否强制 eval、是否用 running stats、
   是否有 SyncBN。
6. 权重加载：路径、strict、是否加载 EMA 权重、是否加载官方 LoRA/Adapter 权重、
   缺失时是否静默随机初始化。
7. 输出后处理：argmax 轴 / sigmoid 阈值、是否 resize 回原图、crop/pad 还原、
   0/1 vs 0/255、保存格式、是否保存概率图。
8. support-based：support 选择、donor 映射、n-shot 聚合（投票/平均/原生 n-shot）、
   support mask 语义、target prompt 是否来自 GT；**support 是否来自 test/val（泄漏）**。
9. oracle-prompted：GT box/point 生成、refinement 是否使用 GT、是否在评测时使用 GT
   做早停/选 mask；必须在论文中标为 prompted。
10. 评测脚本默认命令与 README 声称的命令是否一致；默认路径/权重是否是官方值。
11. 数据根与 split manifest：是否同一个 20 数据集根、同一 manifest hash、
   是否误用 test 做 val。
12. 官方评测脚本是否有 test-set 选 checkpoint / test-set mean/std / 空 mask
    返回 1 等泄漏或不一致；保留还是修复，必须分类。
</items>

<rules>
1. 每项必须给两侧具体行号 + 原文片段，或明确写“移植版无对应代码”。
2. 无法确认写“无法确认”并说明缺哪个文件/权重/数据。
3. 不得用“代码看起来一样”代替 metric 定义对比；必须给出 threshold、reduction、
   empty-mask、分辨率、axis。
4. 必须说明该差异影响的是主表哪个指标（Dice/IoU/clDice/HD95/Seen/Unseen/Novel）。
5. 官方若在 512/518/1024 上评测、移植在 GT 分辨率导出，必须列为
   protocol_adaptation，并说明上采样/下采样方式。
</rules>

<output_format>
<table>
| 项 | 官方出处（行号+原文） | 移植版出处（行号+原文或“无对应代码”） | 判定 | 差异说明 | 影响指标/角色 | 是否已披露 |
</table>

<critical>
按“最可能导致论文表格数值偏差”从大到小排序。每项给出一个当天可执行验证：
- 官方固定 pred/gt 的 metric 单测；
- 官方 checkpoint 在固定图像上的输出对比；
- 移植导出 mask 的 shape/dtype/unique/阈值检查；
- 官方 test 入口与移植 eval 的同一图像对比。
</critical>

<unverified>
缺什么信息、需要补什么文件/权重/数据、缺它会掩盖哪类差异。
</unverified>
