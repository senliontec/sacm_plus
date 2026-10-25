<task>
从官方仓库提取以下评测/推理协议项，逐项给出官方出处（文件:行号 + 3–5 行原文）与移植版对应位置，判定 一致 / 不一致 / 无法确认。
</task>

<items>
1. 验证/测试 transform 的有序列表，是否 resize、是否 center crop、插值方式
2. 推理是否复用训练 dataset 的 transform；test 脚本是否悄悄改了 normalize 或 resize
3. 评测输入分辨率、是否多尺度、是否 TTA、翻转增强细节
4. mIoU 计算方式：逐类平均还是逐像素平均、背景类是否计入、混淆矩阵累加方式、是否跨卡 all_reduce
5. eval 时 batch size 与训练不同时的 BN 行为（是否强制 eval 模式、是否用 running stats）
6. 权重加载路径与 strict 设置；是否加载 EMA 权重而非训练权重
7. 输出后处理：argmax 轴、是否 resize 回原图、边界处理
8. 评测脚本的默认命令与 README 声称的命令是否一致
</items>

<rules>
禁止"大致相同"。每项必须给出两侧具体行号 + 原文片段，或明确写"移植版无对应代码"。无法确认写"无法确认"并说明缺哪个文件。
</rules>

<output_format>
<table>| 项 | 官方出处 | 移植版出处 | 判定 | 差异说明 | 影响类别 |</table>
<critical>按"最可能导致论文表格数值偏差"从大到小排序</critical>
<unverified></unverified>