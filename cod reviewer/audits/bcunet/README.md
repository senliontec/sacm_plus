# BCU-Net 移植审计（手工 pilot）

- 官方仓库：`3rd/BCU-Net`
- 官方 commit：`008ff9292097791a3f12c2776aea398f71b7c996`
- 移植版本：`src/baseline_zoo/models/bcunet/` + `src/baseline_zoo/adapters/bcunet.py`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`
- 审计日期：2026-10-25

> **A0 状态：unverified。** 严格 07 输出 `<blocked>`；本文只是 provisional 手工 pilot。
> 官方模型本地不可构造，**G1/G2/G4 均为 unverified，不能按“复制模型”对待**。

---

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 / 失败症状 |
|---|---|---|---|
| S-bcu-01 | `BCU-Net` 类/整体结构 | `code/core/models.py:9-64` | 官方文件应可 import；当前 `class BCU-Net` 是语法错误 |
| S-bcu-02 | `OutConv` 上采样/1×1/激活 | `code/core/unet_parts.py:95-108` | 官方引用未定义 `upconv2x2`；构造即 NameError |
| S-bcu-03 | 三输出 `out1,out2,out` 与 `0.4/0.6` 融合 | `code/core/models.py:65-98` | 输出张量语义必须一致；否则 loss/inference 不可比 |
| S-bcu-04 | 入口 `from core.models import UNet`、`warmlearnrate` | `code/main.py:7,19`; `code/test.py:19` | 官方入口当前不可运行；移植需替代入口 |
| S-bcu-05 | 预训练 ConvNeXt-22k 权重 | `code/core/convnext.py:172-179,211-223` | 默认从 `dl.fbaipublicfiles.com` 下载；本地无资产 |
| S-bcu-06 | loss：三路 `NLLLoss2d(log(softmax+EPS))` | `code/main.py:107-153` | 三输出都参与监督；移植需等价 |
| S-bcu-07 | Adam lr 1.5e-4、50 epochs、batch 2、512 | `code/main.py:28-33,66-69,107-110` | 官方配方 |
| S-bcu-08 | lr 调度/多卡 `DataParallel` | `code/main.py:86-89,128-130` | `ioutrain.py` 另有每 10 epoch /10 的 lr；官方路径无通用调度器 |
| S-bcu-09 | 预处理：cv2 BGR、无归一化、mask≥128、INTER_AREA | `code/core/utils.py:41-57` | 输入尺度 0–255；移植必须保持 |
| S-bcu-10 | 无训练增强 | `code/core/utils.py:41-57`; `code/main.py:136-153` | 官方无随机增强 |
| S-bcu-11 | 数据：train/test 文件夹；无 val；每 10 epoch 存盘 | `code/main.py:96-105,168-170` | 官方无 val 选择 |
| S-bcu-12 | inference：fused→log-softmax→bilinear→argmax | `code/test.py:107-128` | 上采样在 argmax 前；否则边界不同 |
| S-bcu-13 | 导出：512 可视化/原始 GT 指标 | `code/test.py:109-128` | 官方保存 512 PNG；项目统一 GT bool |
| S-bcu-14 | 指标：confusion matrix / Dice / IoU / AUC | `code/test.py:129-151` | 官方指标口径 |
| S-bcu-15 | 来源/许可证/文件清单 | `code/`; `src/baseline_zoo/models/bcunet/UPSTREAM.md` | URL 未记录、无 LICENSE；G0 有缺口 |

### 关键原文证据

```text
# 3rd/BCU-Net/code/core/models.py:8-11
class BCU-Net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
```
`ast.parse` 结果：`SyntaxError: line 9 offset 10: invalid syntax`（连字符类名）。

```text
# 3rd/BCU-Net/code/core/unet_parts.py:95-108
self.up = nn.Sequential(
    upconv2x2(in_channels, out_channels,mode=""),
    conv1x1(in_channels, out_channels),
)
...
outlayer = nn.Sigmoid()
return outlayer(x)
```
`upconv2x2` 全文件未定义；构造 `OutConv(32,2)` 得到
`NameError: name 'upconv2x2' is not defined`。

```text
# 3rd/BCU-Net/code/core/models.py:65-98
out1 = self.out(x1_9)          # OutConv 内含 Sigmoid → 概率
out2 = self.cls_seg(out)       # 1x1 Conv → logits
out = out1*0.4+out2*0.6        # 概率与 logits 混合
return out1,out2,out
```

---

## 02. 移植对照与差异

| id | verdict | category | gate | 官方行为 | 移植行为 | 影响 | 披露 | S |
|---|---|---|---|---|---|---|---|---|
| D-bcu-01 | `unverified` | `missing_asset` | G1 | 官方文件有语法错误，类名不可用 | 类改名 `BCUNet`、相对 import、构造参数增加 | 无同 state 可比；不能称结构一致 | 部分 | 01 |
| D-bcu-02 | `intentional_deviation` | `bug_fix` | G1/G2/G4 | `OutConv` 调未定义 `upconv2x2`，并加 Sigmoid | 改为 1×1 Conv，无 Sigmoid | 输出语义改变；移植可构造 | adapter docstring, UPSTREAM | 02,03 |
| D-bcu-03 | `unverified` | `missing_asset` | G0/G4 | `main.py/test.py` import 不存在的 `UNet` 和 `warmlearnrate` | adapter 自建训练/推理入口 | 官方路径不可执行 | adapter docstring | 04 |
| D-bcu-04 | `unverified` | `missing_asset` | G1/G2 | `convnext_base(pretrained=True)` 远程下载 | 训练默认 pretrained；权重不在本地 | 无法做官方同权重探针 | **未写权重来源** | 05 |
| D-bcu-05 | **`differ`** | `bug_fix` | G2/G4 | `out1` 是概率，`out` 混合概率与 logits | 三路均为 raw logits，`fused=0.4*out1+0.6*out2` | 融合值域/梯度不同 | UPSTREAM 提到修复 | 03 |
| D-bcu-06 | `unverified` | `wrapper_change` | G2 | `NLLLoss2d(log(softmax+EPS))` | `CrossEntropyLoss` 三路 | 数学近似 CE（EPS），但输入语义受 D-05 影响 | 未逐项说明 | 06 |
| D-bcu-07 | `intentional_deviation` | `official_recipe` | G3 | Adam 1.5e-4、batch2、单进程、50 epoch | batch1×accum2、单卡、50 epoch、无 scheduler | 梯度求和近似，BN 统计不同 | deviations #1,#5 | 07,08 |
| D-bcu-08 | `equivalent` | `wrapper_change` | G3/G4 | cv2 BGR 0–255、无归一化、mask≥128、INTER_AREA | 同（`adapters/bcunet.py:73-89`） | 预处理保持 | deviations #3 | 09 |
| D-bcu-09 | `intentional_deviation` | `protocol_adaptation` | G4/G5 | train/test 文件夹、无 val、每 10 epoch 存盘 | 18/6 协议、val Dice、每 epoch best | 选择/数据不同 | deviations #2 | 11 |
| D-bcu-10 | `equivalent` | `wrapper_change` | G4 | `log(softmax)` 后 bilinear，再 argmax | raw logits bilinear，再 argmax | 逐像素 logsumexp 常数不改变 argmax；官方路径不可执行 | 注释不精确 | 12 |
| D-bcu-11 | `intentional_deviation` | `protocol_adaptation` | G4/G5 | 官方 512 PNG/原始 GT 指标 | bool mask@GT + 项目指标 | 导出/指标口径不同 | deviations/协议文档 | 13,14 |
| D-bcu-12 | `differ` | `wrapper_change` | G3 | batch2 单进程、多卡 DataParallel | batch1×accum2、DataLoader workers0、单卡 | 数据顺序/BN/梯度累加不同 | **DataParallel 未写** | 07,08,11 |
| D-bcu-13 | `differ` | `wrapper_change` | G0 | 上游 URL 未记录、无 LICENSE | UPSTREAM 同样未记录；注明需补 | 许可证/身份不可核验 | UPSTREAM | 15 |
| D-bcu-14 | `unverified` | `missing_asset` | G2/G6a | 官方模型不可构造 | 无同 state 探针 | 无 loss/grad/forward 数值证据 | — | 01,02,03 |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-bcu-01 | `covered_by_discrepancy` | D-01, D-14 |
| S-bcu-02 | `covered_by_discrepancy` | D-02, D-14 |
| S-bcu-03 | `covered_by_discrepancy` | D-02, D-05, D-14 |
| S-bcu-04 | `covered_by_discrepancy` | D-03 |
| S-bcu-05 | `covered_by_discrepancy` | D-04 |
| S-bcu-06 | `covered_by_discrepancy` | D-06 |
| S-bcu-07 | `covered_by_discrepancy` | D-07, D-12 |
| S-bcu-08 | `covered_by_discrepancy` | D-07, D-12 |
| S-bcu-09 | `preserved` | D-08 |
| S-bcu-10 | `preserved` | 无增强的官方与移植代码 |
| S-bcu-11 | `covered_by_discrepancy` | D-09, D-12 |
| S-bcu-12 | `covered_by_discrepancy` | D-10 |
| S-bcu-13 | `covered_by_discrepancy` | D-11 |
| S-bcu-14 | `covered_by_discrepancy` | D-11 |
| S-bcu-15 | `covered_by_discrepancy` | D-13 |

closure = 15/15，但没有一条数值等价断言；closure 只表示差异已定位，不代表保真。

---

## 03. 数值证据

命令（只读本地；未下载、未训练）：

```bash
cd /home/dev/mywork/project/segment/sacm_plus
/home/dev/anaconda3/envs/tsai/bin/python 'cod reviewer/audits/_shared/03_same_state_probes.py'
```

结果见 `03_numeric_probe.json`：

```text
official_model_ast_parse:  SyntaxError: line 9 offset 10: invalid syntax
outconv_construct:         NameError: name 'upconv2x2' is not defined
outconv_first_line:        upconv2x2(in_channels, out_channels,mode=""),
official_model_construct:  NOT ATTEMPTED: pretrained ConvNeXt would download; OutConv already fails
numeric_parity:            NOT RUN
```

结论：**G2=unverified**。没有官方可构造模型，不能把移植实现称为“官方等价”；
也不能用移植自己的 forward 充当官方数值证据。

---

## 04. 训练配置审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| epochs | 50（`main.py:28-30`） | 50 | same |
| batch | 2（`main.py:66-68`） | 1×accum2（deviations #1） | 近似，BN 不同 |
| input | 512（`main.py:69-71`） | 512 | same |
| optimizer | Adam 1.5e-4（`main.py:107-110`） | Adam 1.5e-4 | same |
| scheduler | 无通用 scheduler；`ioutrain` 另有 /10 | 无 | 基本 same（路径不同） |
| loss | 三路 NLL(log(softmax+EPS)) | 三路 CrossEntropy | 目标近似，输入语义受 D-05 影响 |
| transforms | BGR 0–255、mask≥128、INTER_AREA | 同 | preserved |
| train data | 官方 train/test 文件夹 | 18/6 3-shot | 协议适配 |
| workers/GPU | 手动 batch、DataParallel 多卡 | DataLoader workers0、单卡 | **未完全披露** |
| selection | 每 10 epoch 存盘 | 6 图 val Dice | 协议适配 |

G3 不能 pass：官方入口不可运行、预训练权重来源/许可未闭合、DataParallel/BN 差异未完整披露。

---

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | BGR、512、无归一化（`utils.py:41-57`） | 同（`adapters/bcunet.py:109-117`） | preserved |
| checkpoint | 官方每 10 epoch 权重 | port val-Dice best | 协议适配 |
| postprocess | `out=log(softmax(out)+EPS)`→bilinear→argmax（`test.py:116-127`） | raw logits→bilinear→argmax | argmax 等价，但官方依赖缺失 |
| threshold | argmax（无 0.5） | argmax | same |
| metric | confusion matrix/Dice/IoU/AUC | 项目 Dice/IoU/clDice/HD95 | 口径不同 |
| export | 512 可视化 PNG | bool mask@GT | 协议适配 |
| 分辨率 | 原图指标 + 512 保存 | GT | 协议适配 |

G4 不能 pass：官方推理代码不可执行（`UNet`/`warmlearnrate` 缺失），移植路径虽自洽，但没有
同权重/同输出证据。

---

## 07. 门禁结论（provisional）

```text
A0 = unverified            # 无 calibration；正式 07 = <blocked>
G0 = pass                  # 本地 commit/文件可追溯；URL 与 LICENSE 缺失
G1 = unverified            # 官方模型语法错误且不可构造；只有复制文件静态对照
G2 = unverified            # 无同 state 探针；OutConv/融合语义已不同
G3 = unverified            # 官方入口/权重不可运行；DataParallel/BN 差异未闭合
G4 = unverified            # 官方 test.py 不可 import；无同 checkpoint 输出
G5 = pass                  # 18/6、混合域、val 选择均已写入 deviations
G6a = unverified           # 无 20-step 同状态路径
G6b = N/A                  # 无官方 checkpoint
G6c = unverified           # 无全量训练/评测
main_table = blocked
claim_label = none
official_metric_match = unverified
```

缺失/阻断项：上游 URL 与 LICENSE；可解析的官方模型或明确的上游修复 commit；ConvNeXt-22k
权重来源；官方 `UNet`/`warmlearnrate` 入口；OutConv/融合语义的官方裁决；同 state 数值探针；
G6a/G6c；A0。

## 最终判断

- 只完成了静态文件对照和“官方为什么不能跑”的复现，**没有任何数值等价证据**。
- 移植的 `OutConv` 修复可运行，但把 `out1` 从概率改成 logits，改变了官方融合定义；
  在官方语义未裁决前不能称等价。
- 当前只能 `G0=pass`（来源/许可有 caveat）与 `G5=pass`，其余关键门禁 unverified；不能进入主表。
