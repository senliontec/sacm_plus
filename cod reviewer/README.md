# 官方实现 ↔ 移植实现 保真审计协议

本目录不是“代码质量 review”，而是一套**可执行、可举证、可门禁**的保真审计协议。
目标：最大限度保证移植模型在论文表格关心的行为上与官方实现一致，并把所有
无法避免的差异显式记录、分类、披露，而不是靠“看起来一样”。

---

## 1. “一样”到底指什么

“一样”不是源码逐字相同，也不是只看 forward 数值。按以下七个保真级别审计：

| 级别 | 名称 | 核心问题 | 典型证据 |
|---|---|---|---|
| L0 | 来源与依赖 | 官方 commit、文件、权重、torch/CUDA/库版本是否明确 | commit、LICENSE、文件清单、`pip freeze` |
| L1 | 结构 | 模块图、state_dict 映射、shape、参数量、buffer、初始化/RNG | named_parameters/named_buffers 表、映射表 |
| L2 | 数值 | 固定输入下 forward、中间层、loss、梯度、一次 optimizer step | 张量 dump、max_abs_diff、allclose |
| L3 | 训练配方 | transform、normalize、loss、optimizer、scheduler、clip、EMA、seed、DataLoader、有效 batch | 配置 + 训练循环逐项表 |
| L4 | 推理协议 | eval transform、checkpoint 加载、后处理、阈值、metric、输出分辨率 | 推理脚本 + metric 定义 |
| L5 | 项目协议 | 18/6 3-shot、support/donor、20 数据集、GT 分辨率导出、表格口径 | protocol/manifest + adapter |
| L6 | 经验复现 | 官方权重/官方数据/短程训练/全量训练是否复现 | 官方 checkpoint 对比、短程 loss 曲线、最终表格 |

**门禁规则**：
- 进入主表：G0–G6 全部通过；所有 intentional deviation 必须进入 deviation ledger
  并在论文/supplement 披露。
- G0–G5 任一 fail/unverified：不得进主表；G6 因缺官方权重/数据/入口无法执行：
  标为 `unverified`，只能进 supplement 或 reference group，不得声称“与官方一致”。
- 任何“整体一致/基本符合/应该相同”都不是证据；没有可验证断言就写 `unverified`。

**G0–G6 的通过标准**：

| gate | pass | fail | unverified |
|---|---|---|---|
| G0 来源 | 官方 repo/commit/文件/依赖版本齐全，来源可追溯；LICENSE 与权重来源已记录（有则附 LICENSE/sha256，缺失则明确记录并披露） | 缺关键源码或来源无法确认 | 源码、LICENSE 或权重来源无法确认 |
| G1 结构 | key mapping 完整；无 unexpected missing/extra；shared shape/param/buffer 相同；init/RNG 差异已解释；若加载官方权重则 mapped max diff=0 | 存在未解释的 missing/extra/shape/初始化差异 | 官方结构文件缺失 |
| G2 数值 | 同一权重、固定输入/seed 下 forward/中间层/loss/grad/optimizer step 在容差内（training-free 的 L2 子项按 §2 替换为 mask/prompt/support/确定性） | 超差，或无法加载相同权重 | 缺可共享的同一权重/无法构造官方模块 |
| G3 训练 | 所有训练项有有效值；无未披露差异；scheduler/accumulation/clip/EMA/seed 已审（training-free 训练项 N/A，但 support/数据/确定性仍审） | 存在未披露或未解释的训练差异 | 官方训练代码/配方缺失 |
| G4 推理 | eval transform/checkpoint/postprocess/threshold/metric/export 一致，或有已披露的 protocol_adaptation | 存在未解释的 eval/metric 差异或泄漏 | 官方 eval/导出定义无法确认 |
| G5 协议 | 18/6/support/donor/20 roots/manifest hash/GT export/roles 完整（support/donor 对非 support-based 为 N/A；support 不得来自 test/val）；所有 adaptation 已披露 | 协议不一致且未披露 | 项目 manifest/split/support 不可核对 |
| G6 经验 | supervised：官方 checkpoint 固定 batch 输出在容差内，或官方代码+同一份数据（官方或项目 18/6）+同一初始化/state_dict（能共享时）固定 seed 的 N≥20 个 optimizer steps 的 loss/grad 在容差内；training-free：同 support/target + 官方权重下 mask/prompt 在容差内 | 有资产但超差 | 缺官方 checkpoint 且无法短程 parity，或缺可用的同一份数据/入口 |

**G6 “attempted” 的定义**：至少完成以下之一，并保存证据：
1. 官方 checkpoint 加载到官方与移植两侧，固定输入/seed 比较输出；
2. 官方代码 + 同一份数据（官方数据或项目 18/6 split）+ 同一初始化/state_dict（能共享时），固定 seed，至少 20 个 optimizer steps，比较 loss 与梯度；
3. training-free：固定同一 support/target，用官方权重跑官方与移植，比较 mask IoU/prompt；
4. 官方入口不可运行：写清缺什么、尝试到哪一步，标 `unverified`。
“只读了官方 train.py”不算 attempt。
不能共享同一初始化/state_dict 时，先用官方同一 num_classes/输出空间配置；
仍不能共享则 G6 标 `unverified`。

短程 N≥20-step parity 只证明训练路径/数值一致，**不证明全量训练后的精度**；
主表数值仍需按项目协议全量训练和评测。

**gate 聚合**：同一 gate 内任一必需子项为 `unverified`，该 gate 记为 `unverified`；
任一子项为 `fail`，该 gate 记为 `fail`；不得用其他子项 pass 抵消。

G0↔L0、G1↔L1、G2↔L2、G3↔L3、G4↔L4、G5↔L5、G6↔L6；门禁按 G 编号，审计条目按 L 编号。

**G2 的“同一权重”**：官方 checkpoint、官方 checkpoint 映射、或复制到两侧的同一
state_dict 均可；`weights.match=full|partial|none` 必须记录来源和 sha256
（`full`=共享键全映射，`partial`=只映射共享模块，`none`=无可共享权重）。用同一随机
state_dict 只证明实现等价，不证明官方权重/训练结果复现；后者归 G6，G6 仍需官方
checkpoint 或短程 loss/grad parity。

---

## 2. 方法画像（method profile）

不同 baseline 不能用同一套探针。**不要把“训练模式”“prompt 语义”“backbone”
混成一个 profile**；先按三个正交属性分类，再映射到下面的复合 profile：

| 属性 | 取值 |
|---|---|
| training mode | `supervised` / `training_free` / `semi_supervised` |
| prompt modality | `prompt_free` / `oracle_prompted` / `support_based` / `standard` |
| backbone | `sam` / `non_sam` / `hybrid` |

| 复合 profile | 适用方法 | 必需检查 | 不适用/需特别处理 |
|---|---|---|---|
| `supervised_prompt_free_sam` | SAMed、CWSAM、SACM official、SAM-Adapter、TopoLoRA-SAM | L1–L6；loss/grad；空 prompt/`no_mask_embed` 路径 | — |
| `supervised_oracle_prompted_sam` | UCS、SAM-Med2D、SAM-OCTA | L1–L6；**GT box/point 生成与 refinement 单独审计** | 不得标为 prompt-free |
| `supervised_support_based_sam` | FoB-SAM | L1–L6；support/episode + loss/grad | 不能因为 support-based 就跳过 loss/grad |
| `training_free_support_based_sam` | PerSAM、Matcher | L1–L6；固定 support/target 的 mask/prompt/support 选择与确定性重复 | 无 loss/grad；不能跳过 L2 |
| `semi_supervised_prompt_free_sam` | KnowSAM | L1–L6；unlabeled pool 是否泄漏；prompt 语义需核实 | 若实际是 support/prompt 方法，改用对应 profile |
| `supervised_standard_non_sam` | U-Net、CS2Net、BCUNet、MaskVSC、nnWNet、JTFN、DSCNet | L1–L6；loss/grad；boundary/DSConv/patch 推理 | — |
| `probe_standard_sam` | sam_vit_l_unet、sam2_unet | L1–L5（训练时含 loss/grad）；不做 L6 | 不进主表；只作附录/探针 |

**任何 profile 都不能跳过 L0、L1、L4、L5。**
L2 对有 forward 的方法必查；training-free 把 L2 的 loss/grad 部分替换为
mask/prompt/support 选择与确定性重复。L3 的训练项对 training-free 标 N/A，
但数据来源、support/donor、checkpoint、确定性仍要审。

---

## 3. 每次审计必须提供的 context pack

只给路径不给全文视为不存在。每个 baseline 的 context pack 至少包含：

```text
<baseline>            # 例如 samed / matcher
<official_commit>     # 官方 repo + commit
<profile>             # README 第 2 节的复合 profile
<training_mode>       # supervised / training_free / semi_supervised
<prompt_modality>     # prompt_free / oracle_prompted / support_based / standard
<backbone>            # sam / non_sam / hybrid
<paper_group>         # main / supplement / reference / probe
<env>                 # torch / CUDA / cuDNN / numpy / timm / albumentations / sklearn 等
<official_files>      # 逐文件全文：model、loss、transforms、train、eval、config、weights loader、关键依赖
<ported_files>        # 逐文件全文：vendored model、adapter、config、tests
<shared_runtime>      # runtime.py、data.py、support.py、protocols.py 的相关段落全文
<known_deviations>    # adapter.deviations + 文档中已声明的偏差
<assets>              # 官方权重、SAM/DINOv2 等外部权重、是否本地存在
<determinism>         # 随机源、CUDA 算子、k-means/AMG/PCA 等非确定性
```

缺少官方文件、官方权重或可运行入口时，无法执行的检查必须进入 `<unverified>`，不得猜测内容；
用同一 state_dict 能做的实现等价性检查另行标注，不得与“官方权重复现”混为一谈。

---

## 4. 偏差分类（deviation ledger）

每条差异必须落成一行，禁止只写“有差异”：

| 字段 | 说明 |
|---|---|
| `id` | 唯一编号 |
| `level` | L0–L6 |
| `fidelity_gate` | 阻断的 G0–G6（如 G2/G4/G6） |
| `category` | `official_recipe` / `protocol_adaptation` / `bug_fix` / `wrapper_change` / `missing_asset` / `unverified` |
| `official_ref` | 官方文件:行号（可与 behavior 合并，CSV 单列） |
| `official_behavior` | 官方实际行为 |
| `ported_ref` | 移植文件:行号 |
| `ported_behavior` | 移植实际行为 |
| `evidence_official` | 官方 3–5 行原文，或可执行断言 |
| `evidence_ported` | 移植 3–5 行原文，或可执行断言 |
| `numeric_impact` | 对 forward/loss/grad 的影响方向与量级 |
| `paper_impact` | 对主表/附录/某个数据集的影响 |
| `disclosure` | 论文哪一节、哪个 footnote、哪个 supplement 表 |
| `verification` | 当天可执行的验证命令/测试 |
| `undocumented` | true/false；true 直接阻断主表 |
| `blocking` | true/false；是否阻断主表 |

字段内避免裸 `|`；`deviation_ledger.csv` 汇总时必须按 16 列校验，列数不符视为 ledger 缺陷。

**判定规则**：
- `official_recipe`：官方配方，必须保留；不能保留时必须归为 `protocol_adaptation`
  或 `bug_fix`，不得静默改。
- `protocol_adaptation`：为了 18/6 3-shot / 20 数据集 / 二值类数 / GT 分辨率导出 /
  共享 runtime 而改；必须披露。
- `bug_fix`：官方代码有 bug/泄漏/弃用 API；必须说明官方原写法为何可疑，以及修复方向。
- `wrapper_change`：只改包装/命名/设备，不应改变数值；必须用 L2 证明。
- `missing_asset`：官方权重/数据缺失；不能声称复现；G2 可用同一 state_dict、
  G6 可用短程 parity 另行判定，判不了才标 `unverified`。
- `unverified`：信息不足；写清缺什么。

**未出现在 ledger 的差异 = 移植缺陷。**
adapter 的 `deviations` 必须与 ledger 逐条对账；文档写了但代码没做、或代码做了但文档没写，都算缺陷。

---

## 5. 文件与执行顺序

| 文件 | 作用 |
|---|---|
| `01_official_sensitivity.md` | 只看官方：必须逐字保留的敏感点 |
| `02_port_vs_official_audit.md` | 主审计：差异清单 + deviation ledger + gates |
| `03_numeric_harness.md` | 自动数值探针脚本规格 |
| `04_training_recipe_audit.md` | 训练协议逐项表 |
| `05_eval_protocol_audit.md` | 评测/推理协议逐项表 |
| `06_compact_diff.md` | 可选：精简 diff 输出，用于快速复核 |
| `07_triage_and_gates.md` | 去噪、排序、生成验证命令、主表 gate |

```text
01_official_sensitivity
  ↓
02_port_vs_official_audit
  ↓
03_numeric_harness
  ↓
04_training_recipe_audit
05_eval_protocol_audit
  ↓
06_compact_diff（可选快速复核）
  ↓
07_triage_and_gates
  ↓
fix → 重新 02/03 复核 → 更新 ledger
```

- 每个 baseline 一个独立 agent / 独立 context pack；禁止把别的 baseline 的结论套过来。
- 高风险 baseline 做第二遍独立审计，比较两份清单的并集。
- `03_numeric_harness` 只生成脚本，不执行；执行后保存 JSON 和原始 tensor dump。
- 任何修复都必须回到 `02_port_vs_official_audit` 重新审计，不能只改代码不更新 ledger。

---

## 6. 不能保证的事情

以下情况不能直接声称“与官方完全一致”；必须按实测把对应 gate 记为 `unverified`，
或在 ledger 中作为 `intentional_deviation` 披露：

1. 官方权重缺失（例如 Matcher 的 DINOv2 ViT-L、SAMed 的 8 类 LoRA 权重）。
   此时 G2 可用同一 state_dict 证明实现等价，但 G6 只能走“官方代码 + 同一数据短程
   loss/grad parity”；两者都无法执行才标 `unverified`。
2. 官方数据/划分不可得：优先把项目 18/6 数据接到官方入口（写 wrapper 并记录）；
   接不上时用同一份项目数据做短程对比，仍无法执行才标 `unverified`。
3. 非确定性算子（Matcher 的 k-means/AMG、DSCNet 的 DSConv、UCS 的 sklearn PCA）：
   重复运行并记录方差；无法稳定复现时对应 gate 标 `unverified`，不得只跑一次就判
   pass 或 fail。
4. CUDA/cuDNN/库版本不同导致的浮点差异：超出容差时对应 gate 标 `unverified`；
   不得用 CPU 结果代替目标 GPU。
5. 官方代码本身有 bug/泄漏；移植版修复后数值本来就会不同：必须作为 `bug_fix` /
   `intentional_deviation` 披露；G6 若无法与官方原行为对齐则标 `unverified`。
6. 只在 CPU 上验证、未在目标 GPU 上验证：目标 GPU 的 gate 标 `unverified`。

**最高保证 = 官方与移植在同一环境、同一输入、同一种子、同一权重/数据下，
G0–G6 全部有证据通过（或在 ledger 中解释），并保存官方 checkpoint/短程训练证据。
仅靠读代码不可能给出“完全一样”的结论。**

---

## 7. 配套模板

- `context_pack.yaml`：每个 baseline 复制一份，填入官方/移植全文、共享 runtime、
  资产、确定性设置、已知 deviations、容差和 gate 要求。
- `deviation_ledger.csv`：把 `02_port_vs_official_audit` 的 `<deviation_ledger>`
  落到统一表头，便于跨 baseline 汇总和与 adapter.deviations 对账。

`03_numeric_harness.md` 的脚本输出 JSON，建议按 `${baseline}.json` 保存；
`deviation_ledger.csv` 按 `${baseline}.csv` 保存；原始 tensor dump 放
`${baseline}/tensors/`，便于复核。
