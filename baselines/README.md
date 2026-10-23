# SACM 论文 Baseline 代码

本目录用于放置 SACM 官方论文中对比的 baseline 源码。**一个模型一个子目录**，每个子目录保留上游仓库自己的目录结构和许可证。

官方 SACM 源码不放在这里，仍位于：

```text
/home/dev/mywork/project/segment/3rd/SACM
```

## 目录

| 子目录 | 论文中的名称 | 类型 | 代码状态 |
|---|---|---|---|
| `unet/` | U-Net | 通用分割 | 待放入 |
| `cs2net/` | CS²Net | 曲线结构分割 | 待放入 |
| `bcunet/` | BCUNet | 曲线结构分割 | 待放入 |
| `maskvsc/` | MaskVSC | 曲线结构分割 | 待放入 |
| `nnwnet/` | nnWNet | 曲线结构分割 | 待放入 |
| `sam-med2d/` | SAM-Med2D | 医学 SAM baseline | 待放入 |
| `sam-octa/` | SAM-OCTA | OCTA SAM baseline | 待放入 |
| `cwsam/` | CWSAM | SAM baseline | 待放入 |
| `knowsam/` | KnowSAM | SAM baseline | 待放入 |
| `segdino/` | SegDINO | SAM/DINO baseline | 待放入 |

`.gitkeep` 只用于让 Git 跟踪空目录，放入代码后可保留或删除。

## 放置代码的规则

1. 不修改上游算法的内部实现；需要适配时，把 wrapper 放到 `sacm_plus/scripts/baselines/`。
2. 每个子目录直接放该方法的完整源码，保留其 `README`、`requirements.txt`、`LICENSE`。
3. 预训练权重不要提交到 Git。本仓库顶层 `.gitignore` 已忽略 `*.pth`、`checkpoints/`。
4. 每个方法至少记录：
   - 上游仓库 URL 与 commit；
   - 训练入口和推理入口；
   - 依赖环境；
   - 是否有 3-shot 训练能力；
   - 是否需要 point/box prompt；
   - 输出的 mask 格式与尺寸。
5. 如果该方法没有公开代码或无法适配，保留目录并在本文件状态列标注 `不可用`，不要用官方论文数字冒充我们协议下的结果。

## 统一数据协议

所有 baseline 使用同一份冻结 split：

```text
训练：data/sacm_3shot/train/     # 18 张（6 域 x 3-shot，含 CORN）
验证：data/sacm_3shot/val/       # 6 张（含 CORN 官方 val）
Seen：data/sacm_3shot/test/<dataset>/   # 5 个留出域 + CORN 官方 test
Unseen：datasets/unseen/<dataset>/
Novel：datasets/novel/<dataset>/
```

最终 20 个数据集：

```text
Seen (6):   DRIVE, CHASEDB1, DCA1, CREMI, CrackTree200, CORN
Unseen (8): FIVES, STARE, XCAD, DeepCrack, HRF, LES-AV, CHCrack5k, OCTA-500
Novel (6):  ROAD, LEAF, TYRE, WIRE, DeepGlobe, TopoMortar
```

DeepGlobe 使用有 GT 的 train split 中固定 seed 42 的 500 张子集；
其官方 valid/test 没有 GT。CORN 使用作者提供的官方 batch-disjoint
split：train 1176 / val 188 / test 152。LEAF 使用作者提供的官方 31 张
Novel 集；其 mask 细碎，拓扑指标需在论文中注明标注 caveat。

禁止使用 test 进行调参或选择 checkpoint；checkpoint 只能由 val 选择。

## 统一预测输出

所有方法最终都要输出同一种预测，供统一评测器读取：

```text
results/predictions/<method>/<dataset>/<stem>.png
```

要求：

- 单通道 PNG；
- 0/255 二值；
- 文件名与 GT 的 stem 完全一致；
- 尺寸恢复到原始 GT 尺寸；
- 不在评测器中再次阈值化。

## 统一评测指标

首轮统一计算：

```text
Dice
IoU
clDice
HD95
```

按数据集分别统计，再报告 Seen / Unseen / Novel 的 macro average。具体实现和结果表在 baseline 适配完成后接入 `scripts/`，本目录只存放第三方代码。
