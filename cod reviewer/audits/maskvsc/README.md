# MaskVSC 移植审计（手工 pilot）

> **修复后（2026-10-26）：** 本报告表格是修复前审计证据；当前代码状态和回归证据见
> [`../POST_FIX.md`](../POST_FIX.md)。A0/G6c 仍未完成，gate 结论不变。

- 官方仓库：`3rd/MaskVSC`
- 官方 commit：`5e60565b5ec63cb91295e06b9b502d6350e89d76`
- 移植版本：`src/baseline_zoo/models/maskvsc/` + `src/baseline_zoo/adapters/maskvsc.py`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`
- 审计日期：2026-10-25

> **A0 状态：unverified。** 严格 07 = `<blocked>`；本文是 provisional 手工 pilot。
> 模型/loss 有同 state 数值证据；数据管线的 RNG 差异没有数值证据，G3 不能 pass。

---

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 / 失败症状 |
|---|---|---|---|
| S-mv-01 | Fundus `UNet` 结构 | `model/UNet.py:1-...` | 复制逐字一致；否则 state_dict 不兼容 |
| S-mv-02 | `initialize_weights` 初始化 | `lib/Loss.py:1-17` | 初始化一致；否则同 seed 权重不同 |
| S-mv-03 | loss=`BCELoss + Connect_Loss` | `train.py:34-36,141-146`; `lib/Loss.py:20-46` | 数值/梯度一致 |
| S-mv-04 | Adam lr 2e-4, betas (.5,.999) | `train.py:44-45` | 优化器一致 |
| S-mv-05 | mask ratio 随 epoch：两端 0、峰值 0.4 | `train.py:100-110` | epoch 映射必须对齐 |
| S-mv-06 | flip 先于 resize；PIL 默认 resize 512 | `lib/datasets.py:26-32` | 同 RNG 下图像/mask 一致 |
| S-mv-07 | 血管腐蚀：skeleton、删除分支、膨胀、3 路高斯 | `lib/datasets.py:34-67` | 同 RNG 下结果一致 |
| S-mv-08 | ratio=0 仍进入腐蚀分支并消耗随机数 | `lib/datasets.py:34-67` | 跳过会改变后续 RNG 流 |
| S-mv-09 | 标签阈值：max=255→≥128；max=1→≥0.5 | `lib/datasets.py:73-79` | 二值标签一致 |
| S-mv-10 | 数据：Train/Test 文件夹 | `train.py:26-27`; `lib/datasets.py:14-22` | 官方 test 选择；移植 18/6 |
| S-mv-11 | DataLoader batch4、workers8、shuffle | `train.py:112-117` | 训练动力学；否则顺序不同 |
| S-mv-12 | seed 1234 + cuDNN deterministic | `train.py:170-176` | 官方随机性设置 |
| S-mv-13 | eval：PIL resize 512、ToTensor、min-max、≥0.5 | `train.py:54-90`; `lib/EvalMetric.py:9-11` | 阈值/归一化一致 |
| S-mv-14 | 每 epoch 在 test 上算 Dice；仅 `Dice>best && >0.5` 保存 | `train.py:156-162` | 官方存在 test 选择；移植改 val |
| S-mv-15 | 导出 512 mask；官方 clDice 指标 | `test.py:49-65`; `lib/EvalMetric.py` | 分辨率/指标口径 |
| S-mv-16 | 来源/许可证 | `src/baseline_zoo/models/maskvsc/UPSTREAM.md` | URL 有；LICENSE 文件缺失 |
| S-mv-17 | 复制文件差异 | `models/maskvsc/{unet,loss,transforms}.py` | UNet diff 空；loss 仅 device |

### 关键原文证据

```text
# 3rd/MaskVSC/lib/datasets.py:26-32
if (np.random.random() < 0.5):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    lab = lab.transpose(Image.FLIP_LEFT_RIGHT)
img = np.array(img.resize([512,512]))
lab = np.array(lab.resize([512,512]))
```

```text
# 3rd/MaskVSC/lib/datasets.py:34-67（ratio=0 也不 return）
sk = morphology.thin(lab).astype(np.uint8)
...
del_list = random.sample(range(0,length), int(self.mask_ratio*length))
...
gauss_r = np.random.normal(0,20,(lab.shape[0],lab.shape[1]))
gauss_g = np.random.normal(0,20,(lab.shape[0],lab.shape[1]))
gauss_b = np.random.normal(0,20,(lab.shape[0],lab.shape[1]))
```

```text
# src/baseline_zoo/models/maskvsc/transforms.py:18-24
if ratio <= 0:
    return image
skeleton = morphology.thin(label).astype(np.uint8)
if skeleton.sum() == 0:
    return image
```
移植在 ratio=0 或空骨架时提前返回，官方会继续消耗 `random.sample` 与三次
`np.random.normal`。这是可验证的随机数消耗差异。

---

## 02. 移植对照与差异

| id | verdict | category | gate | 官方行为 | 移植行为 | 影响 | 披露 | S |
|---|---|---|---|---|---|---|---|---|
| D-mv-01 | `equivalent` | `wrapper_change` | G1/G2 | UNet / loss 文件 | UNet diff 空；loss 仅 `.cuda()`→`img.device` | forward/loss/grad 探针通过 | UPSTREAM | 01,02,03,17 |
| D-mv-02 | **`differ`** | `bug` | G3 | ratio=0/空骨架仍执行腐蚀、消耗 RNG | 提前 return，不消耗 RNG | 后续 epoch 的增强/采样流不同 | docstring 只写空骨架；ratio=0 未写 | 07,08 |
| D-mv-03 | `differ` | `wrapper_change` | G3 | flip 用 `np.random.random()`，corruption 用全局 `random` | flip 用 `Random(seed+epoch)`，corruption 仍用全局 `random` | 同 seed 下变换顺序/值不同 | **未写** | 06,07,12 |
| D-mv-04 | `intentional_deviation` | `protocol_adaptation` | G3 | seed 1234 + cuDNN deterministic | 协议 seed 0–4；runtime 固定 seed | 无法逐位复现官方轨迹 | deviations #4 | 12 |
| D-mv-05 | `intentional_deviation` | `protocol_adaptation` | G5 | Train/Test 文件夹 | 18/6 3-shot；灰度→RGB | 数据/域不同 | deviations #1 | 10 |
| D-mv-06 | `intentional_deviation` | `protocol_adaptation` | G4/G5 | test 每 epoch、Dice>0.5 门限 | 6 图 val Dice | 选择语义不同，移植更符合协议 | deviations #2 | 14 |
| D-mv-07 | `differ` | `wrapper_change` | G3 | DataLoader workers8 | workers0（runtime 默认） | 数据装载/RNG 顺序不同 | **未写** | 11 |
| D-mv-08 | `equivalent` | `wrapper_change` | G2 | Connect_Loss kernel `.cuda()` | `device=img.device` | GPU 上数值同为 0 diff | deviations #5 | 03 |
| D-mv-09 | `equivalent` | `wrapper_change` | G4 | min-max 后 `>=0.5` | 同（`adapters/maskvsc.py:126-130`） | 阈值语义一致 | 代码 | 13 |
| D-mv-10 | `intentional_deviation` | `protocol_adaptation` | G4/G5 | 512 mask 输出；官方指标 | GT bool + 项目指标 | 分辨率/指标不同 | deviations #3 | 15 |
| D-mv-11 | `equivalent` | `wrapper_change` | G3 | epoch ratio 以 0 为起点 | 1-based epoch 映射 `(epoch-1)/epochs` | 每个 epoch 的 ratio 对齐 | 代码 docstring | 05 |
| D-mv-12 | `differ` | `missing_asset` | G0 | 官方 README 链接 LICENSE | 快照无 LICENSE，UPSTREAM 注明待补 | 再分发合规缺口 | UPSTREAM | 16 |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-mv-01 | `preserved` | D-01 |
| S-mv-02 | `preserved` | D-01 |
| S-mv-03 | `preserved` | D-01, D-08 |
| S-mv-04 | `preserved` | adapter `make_optimizer` |
| S-mv-05 | `preserved` | D-11 |
| S-mv-06 | `covered_by_discrepancy` | D-03 |
| S-mv-07 | `covered_by_discrepancy` | D-02, D-03 |
| S-mv-08 | `covered_by_discrepancy` | D-02 |
| S-mv-09 | `preserved` | adapter 标签阈值代码 |
| S-mv-10 | `covered_by_discrepancy` | D-05 |
| S-mv-11 | `covered_by_discrepancy` | D-07 |
| S-mv-12 | `covered_by_discrepancy` | D-03, D-04 |
| S-mv-13 | `preserved` | D-09 |
| S-mv-14 | `covered_by_discrepancy` | D-06 |
| S-mv-15 | `covered_by_discrepancy` | D-10 |
| S-mv-16 | `covered_by_discrepancy` | D-12 |
| S-mv-17 | `preserved` | D-01 |

closure = 17/17。D-mv-02/D-mv-03/D-mv-07 是 G3 的主要未闭合项。

---

## 03. 数值证据

命令（在 `sacm_plus/` 下；无下载、无训练）：

```bash
/home/dev/anaconda3/envs/tsai/bin/python 'cod reviewer/audits/_shared/03_same_state_probes.py'
```

结果见 `03_numeric_probe.json`：

```text
state keys/shapes equal:        True / True
params official/port:           7,764,065 / 7,764,065
buffers official/port:          3,978 / 3,978
official loads port state:      True
forward maxdiff (B2,64x64):     0.0
loss official/port:             0.8885400891304016 / 0.8885400891304016
loss absdiff:                   0.0
grad maxdiff:                   4.470348358154297e-08
missing grads:                  []
```

结论：模型/损失/梯度路径有精确数值证据，`G2=pass`。但这是模型级探针，不覆盖
`ImageDataset_Fundus` 的 flip/corruption RNG，也不覆盖 DataLoader worker 顺序；
因此 `G3=unverified`，不能用 G2 替代。

---

## 04. 训练配置审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| epochs | 200（`train.py:24`） | 200 | same |
| batch | 4（`train.py:22`） | 4 | same |
| input | 512 PIL resize | 512 PIL resize | same |
| optimizer | Adam 2e-4, betas (.5,.999)（`:44-45`） | 同 | same |
| scheduler | 无 | 无 | same |
| loss | BCE + Connect_Loss（`:34-36,141-146`） | 同 | same |
| mask schedule | epoch/200，峰值 0.4（`:100-110`） | 1-based 映射 | same |
| transforms | flip + resize + corruption；np/global random | flip 用 `Random(seed+epoch)`，corruption 用 global random/early-return | **RNG 流不同** |
| train data | Train 文件夹 | 18/6 3-shot，RGB | 协议适配 |
| workers/GPU | 8 / 单卡 | 0 / 单卡 | **worker 未披露** |
| seed | 1234 + cuDNN deterministic | 协议 seed | 协议适配 |
| selection | test Dice，>0.5 门限 | 6 图 val Dice | 协议适配 |

G3 不能 pass：RNG/worker 差异会改变每个 epoch 的输入，而 loss 本身已精确一致。

---

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | PIL resize 512、ToTensor（`train.py:54-68`） | 同（`adapters/maskvsc.py:113-120`） | preserved |
| checkpoint | 官方 test-Dice best（`:156-162`） | val-Dice best | 协议适配 |
| postprocess | min-max、`>=0.5`（`lib/EvalMetric.py:9-11`） | 同 | preserved |
| threshold | 0.5 | 0.5 | same |
| metric | 官方 clDice 等 @512 | 项目 Dice/IoU/clDice/HD95 @GT | 口径不同 |
| export | 512 PNG（`test.py:59-65`） | bool mask@GT | 协议适配 |
| 分辨率 | 512 | GT | 协议适配 |

推理模型/后处理有代码和 loss 证据；G4 可 pass，但导出/指标是协议适配。

---

## 07. 门禁结论（provisional）

```text
A0 = unverified            # 无 calibration；正式 07 = <blocked>
G0 = pass                  # URL/commit/文件；LICENSE 缺失已记录
G1 = pass                  # UNet diff 空；state/shape/buffer 完全一致
G2 = pass                  # forward/loss/grad 探针通过
G3 = unverified            # corruption RNG 提前返回、flip/worker RNG 未闭合
G4 = pass                  # min-max/0.5/GT 恢复有代码证据
G5 = pass                  # 18/6、混合域、test→val 选择均披露
G6a = unverified           # 无 20-step 同状态路径
G6b = N/A                  # 无官方 checkpoint
G6c = unverified           # 无全量训练/评测
main_table = blocked
claim_label = none
official_metric_match = unverified
```

缺失/阻断项：把 ratio=0/空骨架的 RNG 消耗、flip 的 RNG 源、workers 差异写入 deviations；
对数据管线做同 seed 张量探针（当前没有）；补 G6a/G6c；LICENSE；A0。

## 最终判断

- UNet/loss 实现层面**精确一致**，有同 state 数值证据（loss diff=0，grad 4.5e-8）。
- 完整训练**不是官方逐位复现**：corruption 在 ratio=0 时提前返回、flip 换 RNG、
  workers 8→0；这些都会改变数据顺序与随机流，且未完整披露。
- 当前 `G2=pass`，但只能说明模型/损失；`G3=unverified`，不能称官方训练复现。
