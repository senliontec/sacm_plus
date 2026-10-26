# 官方实现 ↔ 移植实现 保真审计协议

本目录不是“代码质量 review”，而是一套**可执行、可举证、可门禁**的保真审计协议。
目标：最大限度保证移植模型在论文表格关心的行为上与官方实现一致，并把所有
无法避免的差异显式记录、分类、披露，而不是靠“看起来一样”。

---

## 0. 审计者校准（A0）与 01→02 闭环

### 0.1 A0：先校准审计者，再审计 baseline

A0 是**审计者资格门禁**，不是 baseline 门禁。A0 不是 `pass`，任何 G0–G6a/b/c 结论都无效。

- 使用冻结、带 hash 的 calibration pack；审计者看不到 gold；scorer 独立。
- 案例必须包含：负控（语义等价重构，不得误报）、正控（单点植入偏差，必须发现）、
  歧义控（官方 bug/缺资产，必须 `intentional_deviation`/`unverified`，不得猜）。
- 覆盖至少：resize/normalize、loss 权重、scheduler step 时机、prompt 语义、
  类数/输出空间、support 泄漏、training-free 阈值/投票、checkpoint 加载、metric/导出。
- 评分：high-severity recall、overall recall、precision、verdict/category/gate 正确率、
  evidence 可核验性、01→02 closure、overclaim、双审计一致性。
- A0 pass 阈值：无 high-severity 漏报；overall recall ≥ 0.9；precision ≥ 0.9；
  evidence validity = 1.0；closure = 1.0；overclaim = 0；双审计一致性 ≥ 0.8。
- A0 fail：该审计者的报告不得进入主表/supplement；修 prompt/模型后重跑。
- 每次 prompt/模型/协议变更后、以及每批真实审计前重新校准；高风险 baseline 双审计并取并集。
- **A0 只给出已植入偏差的召回下界，不证明未植入偏差一定被发现。**

详见 `00_auditor_calibration.md`。

### 0.2 01→02 闭环（SENS_CLOSURE）

01 的每条敏感点必须有稳定 ID（`S-<baseline>-nn`），02 必须逐条回流：

| 02 disposition | 含义 |
|---|---|
| `covered_by_discrepancy` | 对应 D-id 的差异条目 |
| `preserved` | 官方行为逐字保留，附可验证断言 |
| `not_applicable` | 该 profile 下不适用，附理由 |
| `unverified` | 缺文件/权重/数据，附缺失项 |
| `new_finding` | **仅用于** 02 新增的 `S-NEW-nn` 占位行；必须回写 01 并重跑 01→02 |

- 对 01 原有的 S-id，disposition 只能是 `covered_by_discrepancy` / `preserved` /
  `not_applicable` / `unverified`；`new_finding` 只出现在 02 新增的 `S-NEW-*` 占位行。
- 02 输出 `<sensitivity_closure>`，每个 S-id 恰好一行；不得空白、不得用“整体一致”代替。
- `SENS_CLOSURE=1.0` 才能进入 03/07；任何 open item 或残留 `new_finding` 直接 `blocked`。
- 02 的每个 D-id 必须带 `sensitivity_ids`；没有对应 S-id 的必须标 `new_sensitivity=true`，
  并触发 01 更新；否则视为未映射差异，阻断。
- 01 遗漏率（02 new findings / 01 items）记入报告，作为审计者召回率的旁证。

### 0.3 与 G0–G6a/b/c 的关系

A0 与 SENS_CLOSURE 是**前置门禁**：A0 != pass 或 closure < 1.0 时，不输出 G0–G6a/b/c
结论，直接 `blocked`。它们不替代 G0–G6a/b/c，只保证“审计过程和审计者的召回”可被检查。

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
- 前置：A0=pass 且 SENS_CLOSURE=1.0；否则 blocked，不输出 G0–G6a/b/c 结论。
- 进入主表：G0–G5 + G6a + G6c 全部 pass；G6b 仅在“声称官方权重/前向复现”时必须 pass；
  所有 intentional deviation 必须进入 deviation ledger 并在论文/supplement 披露。
- A0/G0–G5/G6a/G6c 任一 fail/unverified：不得进主表；G6c unverified：只能
  supplement/reference，不得声称“指标可复现”。
- 只有 G6a/G6b 通过、G6c 未跑：只能标 `path_verified` / `checkpoint_verified`，
  不得标 `table_metric_reproduced`。
- **官方论文数字对齐（`official_metric_match`）不是 G6 子门禁**：只有同一协议、同一数据、
  同一 checkpoint（或官方报告数字按同口径可比）时才可标 `verified`，否则 `unverified`；
  G6c 只证明“本表数字可由冻结产物重算”，不由它推出官方数字一致。
- 任何“整体一致/基本符合/应该相同”都不是证据；没有可验证断言就写 `unverified`。

**G0–G6 的通过标准（G6 拆成 a/b/c）**：

| gate | pass | fail | unverified |
|---|---|---|---|
| G0 来源 | 官方 repo/commit/文件/依赖版本齐全，来源可追溯；LICENSE 与权重来源已记录（有则附 LICENSE/sha256，缺失则明确记录并披露） | 缺关键源码或来源无法确认 | 源码、LICENSE 或权重来源无法确认 |
| G1 结构 | key mapping 完整；无 unexpected missing/extra；shared shape/param/buffer 相同；init/RNG 差异已解释；若加载官方权重则 mapped max diff=0 | 存在未解释的 missing/extra/shape/初始化差异 | 官方结构文件缺失 |
| G2 数值 | 同一权重、固定输入/seed 下 forward/中间层/loss/grad/optimizer step 在容差内（training-free 的 L2 子项按 §2 替换为 mask/prompt/support/确定性） | 超差，或无法加载相同权重 | 缺可共享的同一权重/无法构造官方模块 |
| G3 训练 | 所有训练项有有效值；无未披露差异；scheduler/accumulation/clip/EMA/seed 已审（training-free 训练项 N/A，但 support/数据/确定性仍审） | 存在未披露或未解释的训练差异 | 官方训练代码/配方缺失 |
| G4 推理 | eval transform/checkpoint/postprocess/threshold/metric/export 一致，或有已披露的 protocol_adaptation | 存在未解释的 eval/metric 差异或泄漏 | 官方 eval/导出定义无法确认 |
| G5 协议 | 18/6/support/donor/20 roots/manifest hash/GT export/roles 完整（support/donor 对非 support-based 为 N/A；support 不得来自 test/val）；所有 adaptation 已披露 | 协议不一致且未披露 | 项目 manifest/split/support 不可核对 |
| G6a 路径 parity | supervised：同一预处理 batch（transform 输出 hash 相同）、同一初始化/state_dict（能共享时）、固定 seed，N≥20 optimizer steps 的 loss/grad/optimizer step 在容差内；training-free：同 support/target + 官方权重下 mask/prompt/support 选择在容差内，重复≥3次 | 有资产但超差 | 缺可共享的同一初始化/state_dict、官方权重或可运行入口 |
| G6b 官方资产 parity | 官方 checkpoint 加载到官方与移植两侧（类数不一致时用官方同一 num_classes 配置），固定输入矩阵（batch 1/2、奇数尺寸、空 mask、train/eval）forward/输出在容差内；无官方 checkpoint 则 N/A | 有官方 checkpoint 但输出超差 | 无官方 checkpoint，或 class head 无法对齐 |
| G6c 表格指标溯源 | 按冻结的项目协议完成全量训练+评测（表格要求的 seeds/datasets），保存 masks/metrics/config/manifest hash；从保存 masks 重算指标与表格值一致（≤1e-6）；同 checkpoint 复跑 eval 一致；报告 seed 均值/方差。**只证明本表数字可由冻结产物重算，不代表官方论文数字一致** | 全量运行完成但本表指标不能重算或超差 | 未完成全量运行/缺评测产物 |

**G6a/G6b/G6c 的 “attempted” 定义**：
- G6a：同一预处理 batch（transform 输出 hash 相同）+ 同一初始化/state_dict（能共享时）+ 固定 seed，
  至少 20 个 optimizer steps，比较 loss/grad/optimizer step；training-free 比较 mask/prompt/support
  选择，重复≥3。
- G6b：官方 checkpoint 加载到两侧，固定输入矩阵比较 forward/输出；类数不一致先用官方同一
  `num_classes` 配置；无官方 checkpoint 则 `N/A`。
- G6c：完成表格要求的全量训练/评测，保存 masks/metrics/config/manifest hash，并从 masks 重算指标。
- 缺资产/入口无法执行：写清缺什么、尝试到哪一步，对应子门禁标 `unverified`；
  “只读了官方 train.py”不算 attempt。
不能共享同一初始化/state_dict 时，先用官方同一 num_classes/输出空间配置；
仍不能共享则 G6a 标 `unverified`。

**G6 语义边界**：
- G6a 只证明训练/推理路径一致；G6b 只证明官方权重/前向一致；
  **只有 G6c 证明表格指标可复现**。
- 主表要求 G6a + G6c；G6b 仅在“声称官方权重/前向复现”时必须，且不能替代 G6c。
- 禁止写裸 `G6=pass`；必须分别写 `G6a/G6b/G6c`。G6b 可以 N/A，但不能用 N/A 代替 G6c。
- 只有 G6a/G6b 通过而 G6c 未跑：只能标 `path_verified` / `checkpoint_verified`。

G6a 的短程 parity 只证明训练路径/数值一致，**不证明全量训练后的指标**；
G6c 必须完成全量训练和评测，不能用 20-step parity 代替。

**子门禁聚合**：G6a/G6b/G6c 各自独立聚合；任一必需子项 `unverified` → 该子门禁
`unverified`；任一子项 `fail` → 该子门禁 `fail`。不得用 G6a/G6b 的 pass 抵消 G6c 的 fail/unverified。

G0↔L0、G1↔L1、G2↔L2、G3↔L3、G4↔L4、G5↔L5、G6a/G6b/G6c↔L6；
门禁按 G 编号，审计条目按 L 编号。

**G2 的“同一权重”**：官方 checkpoint、官方 checkpoint 映射、或复制到两侧的同一
state_dict 均可；`weights.match=full|partial|none` 必须记录来源和 sha256
（`full`=共享键全映射，`partial`=只映射共享模块，`none`=无可共享权重）。用同一随机
state_dict 只证明实现等价，不证明官方权重/训练结果复现；后者归 G6a/G6c。
G6a 需官方 checkpoint 或短程 loss/grad parity；G6c 需全量训练/评测。

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
<auditor>             # calibration_report.json + A0 + recall/precision/pack hash
<sensitivity_inventory> # 01 输出全文 + sha256 + item count
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
| `fidelity_gate` | 阻断的 G0–G6c（如 G2/G4/G6a/G6c） |
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
  G6a 可用短程 parity 另行判定；G6b 可为 N/A；G6c 判不了则标 `unverified`。
- `unverified`：信息不足；写清缺什么。

**未出现在 ledger 的差异 = 移植缺陷。**
每个 D-id 必须能追溯到 01 的 S-id；追溯关系以 02 的 `<sensitivity_closure>` 为准。
adapter 的 `deviations` 必须与 ledger 逐条对账；文档写了但代码没做、或代码做了但文档没写，都算缺陷。

---

## 5. 文件与执行顺序

| 文件 | 作用 |
|---|---|
| `00_auditor_calibration.md` | A0：校准审计者、评分、双审计与资格门禁 |
| `01_official_sensitivity.md` | 只看官方：必须逐字保留的敏感点 |
| `02_port_vs_official_audit.md` | 主审计：差异清单 + deviation ledger + gates |
| `03_numeric_harness.md` | 自动数值探针脚本规格 |
| `04_training_recipe_audit.md` | 训练协议逐项表 |
| `05_eval_protocol_audit.md` | 评测/推理协议逐项表 |
| `06_compact_diff.md` | 可选：精简 diff 输出，用于快速复核 |
| `07_triage_and_gates.md` | 去噪、排序、生成验证命令、主表 gate |

```text
00_auditor_calibration（A0；未 pass 不得继续）
  ↓
01_official_sensitivity（S-id）
  ↓
02_port_vs_official_audit + sensitivity_closure
  ↓ SENS_CLOSURE=1.0
03_numeric_harness
04_training_recipe_audit
05_eval_protocol_audit
  ↓
06_compact_diff（可选快速复核）
  ↓
07_triage_and_gates
  ↓
fix → 重新 02/03 复核 → 更新 ledger 与 closure
```

- 每个 baseline 一个独立 agent / 独立 context pack；禁止把别的 baseline 的结论套过来。
- A0 未 pass 的 agent 不得产出真实审计结论；高风险 baseline 做第二遍独立审计，
  比较两份清单的并集，并记录一致性。
- `03_numeric_harness` 只生成脚本，不执行；执行后保存 JSON 和原始 tensor dump。
- 任何修复都必须回到 `02_port_vs_official_audit` 重新审计，不能只改代码不更新 ledger。

---

## 6. 不能保证的事情

以下情况不能直接声称“与官方完全一致”；必须按实测把对应 gate 记为 `unverified`，
或在 ledger 中作为 `intentional_deviation` 披露：

1. 官方权重缺失（例如 Matcher 的 DINOv2 ViT-L、SAMed 的 8 类 LoRA 权重）。
   此时 G2 可用同一 state_dict 证明实现等价，G6a 可走“官方代码 + 同一预处理 batch 短程
   parity”；G6b 为 N/A；G6c 仍需全量运行，无法执行才标 `unverified`。
2. 官方数据/划分不可得：优先把项目 18/6 数据接到官方入口（写 wrapper 并记录）；
   接不上时用同一份项目数据做 G6a 短程对比；G6c 仍必须全量运行，否则标 `unverified`。
3. 非确定性算子（Matcher 的 k-means/AMG、DSCNet 的 DSConv、UCS 的 sklearn PCA）：
   重复运行并记录方差；无法稳定复现时对应 gate 标 `unverified`，不得只跑一次就判
   pass 或 fail。
4. CUDA/cuDNN/库版本不同导致的浮点差异：超出容差时对应 gate 标 `unverified`；
   不得用 CPU 结果代替目标 GPU。
5. 官方代码本身有 bug/泄漏；移植版修复后数值本来就会不同：必须作为 `bug_fix` /
   `intentional_deviation` 披露；G6b/G6c 若无法与官方原行为对齐则标 `unverified`。
6. 只在 CPU 上验证、未在目标 GPU 上验证：目标 GPU 的 gate 标 `unverified`。

**最高保证 = A0 pass、SENS_CLOSURE=1.0，且官方与移植在同一环境、同一输入、
同一种子、同一权重/数据下，G0–G5 + G6a + G6c 有证据通过（声明官方权重复现时
再加 G6b），并保存官方 checkpoint、短程 parity 与全量运行证据。
仅靠读代码不可能给出“完全一样”的结论；G6a/G6b pass 也不等于指标可复现。**

---

## 7. 配套模板

- `00_auditor_calibration.md`：A0 校准协议、案例矩阵、评分与阈值。
- `context_pack.yaml`：每个 baseline 复制一份，填入官方/移植全文、共享 runtime、
  A0 报告、01 inventory、资产、确定性设置、已知 deviations、容差和 gate 要求。
- `deviation_ledger.csv`：把 `02_port_vs_official_audit` 的 `<deviation_ledger>`
  落到统一表头，便于跨 baseline 汇总和与 adapter.deviations 对账。

`03_numeric_harness.md` 的脚本输出 JSON，建议按 `${baseline}.json` 保存；
`deviation_ledger.csv` 按 `${baseline}.csv` 保存；原始 tensor dump 放
`${baseline}/tensors/`，便于复核。

---

## 8. Pilot audit outputs（provisional）

- `audits/ALL_MODELS.md`：22 个模型的总表与最高优先级缺陷。
- `audits/all_models_status.csv`：机器可读状态表。
- `audits/<model>/README.md`：每个模型的 01/02/closure/03/04/05/07 pilot 报告。

当前 **A0=unverified**，所有报告都是 provisional，`main_table=blocked`；
没有 G6c 全量训练/评测，也没有任何模型被标为主表通过。
