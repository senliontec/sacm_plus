# 2025–2026 SAM / 曲线结构分割相关工作扫描

> **快照日期**：2026-10-18
> **目的**：为 CVPR 2027 投稿做 Related Work 与新颖性定位
> **范围**：本地 `parsing/` 中 102 篇 2025–2026 论文（读 `full.md`）+ CVF / AAAI / arXiv / OpenAlex / 期刊检索
> **证据等级**：`FT` = 全文；`A` = 摘要/元数据；`S` = 搜索摘要。表中 `†` = A/S；无标 = 本次读到全文或本地 `full.md`。
> **注意**：2026 venue / arXiv 编号可能变化，投稿前必须核对 camera-ready。本文的“未发现”是检索范围内的结论，不是“不存在”的证明。

**缩写**

- `PF0`：严格 prompt-free（推理时无 prompt encoder 输入）
- `PFa`：自动生成 prompt，无人工输入（不等于严格 PF0）
- `PB`：point / box prompt
- `H / M / L`：对我们新颖性的威胁等级（High / Medium / Low）

---

## 1. 直接竞品（prompt-free SAM + 曲线结构）

| 方法 | SAM 角色 | 核心机制 | 拓扑 / 选择 | 威胁 | 差异 / 代码 |
|---|---|---|---|---|---|
| **SACM** CVPR'26 | PF0；frozen ViT-L | 通用 Adapter-I/E + 双阶段解码 + IoU predictor | 无拓扑 loss；选择 = IoU | **H（直接基线）** | adapter 非几何、选择非拓扑；官方 Stage-1 无梯度、3/4 Stage-2 heads 无主损失（我们对官方代码的分析） · [code](https://github.com/kylechuuuuu/SACM) |
| **CaPro** AAAI'26 | PFa；training-free SAM ViT-B | L-system 合成 + oriented boxes + VQVAE 过滤 | 无 | M（不同路线） | 单张无标注图；非严格 PF0；无 adapter/拓扑 · [code](https://github.com/xmed-lab/CaPro) |
| **FineSAM++** NeurIPS'25 | PF0；LoRA + 默认 prompt + 不确定性门控残差专家 | 单 mask 血管精修 | 拓扑仅评测（clDice/连通性） | **H/M（最接近 PF0 血管）** | 无多候选选择；走“残差修正”而非“候选选择” |
| **TopoLoRA-SAM** 2026† | PF0；LoRA FFN + 空间卷积 adapter | 单 mask；可选 clDice | clDice loss；无选择 | M/H（单输出 clDice 基线） | 无 window/strip；无候选选择 · [arXiv](https://arxiv.org/abs/2601.02273) |
| **CoroSAM** BIBM'25† | PF0；空间-频率 + 方向 adapter | 冠脉 X-ray；单 mask | 方向先验；无拓扑 | **H/M（几何 PF0 最近）** | 无多候选/拓扑选择 · [code](https://github.com/HITZhengGW/CoroSAM) |
| **MSA-SAM** JSTARS'25† | PF0；Pyramid + Window multiscale adapter（7/28） | 遥感语义分割 | 无 | **H（window adapter 已占位）** | 各向同性、非曲线、无选择 · [code](https://github.com/mint0126/Mult-scale-SAM) |
| **Topology-Constrained SAM2-UNet** Electronics'26† | PF0；frozen SAM2 + decoder 条带卷积 | 方向解耦条带 + 边界方向 loss；电力线 | 拓扑 loss；单输出 | M/H（strip + topology 已组合） | decoder 侧、非 ViT adapter、无选择 · [DOI](https://doi.org/10.3390/electronics15184330) |
| **SnakeSAM** Array'26† | 拓扑保持基础模型 | 动态 snake adapter + path-field prompt + 断点修正 | 拓扑驱动迭代 prompt | M（替代路线） | 修正而非候选选择 · [DOI](https://doi.org/10.1016/j.array.2026.101221) |
| **TopoSAM** EAAI'26† | SAM + 拓扑蛇形卷积分支 | 裂纹分割 | 拓扑架构；单 mask | M（拓扑 + SAM） | 非通用 CSS、无选择 · [DOI](https://doi.org/10.1016/j.engappai.2025.113688) |
| **A2SAM** MedIA'26† | PFa（自动结构 prompt） | 结构 prompt + 对比参数选择 + 厚度 loss | 拓扑 prompt/loss | M（参数选择 ≠ mask 选择） | 无多候选选择 · [code](https://github.com/YongChen-Exact/A2SAM) |

---

## 2. 曲线/管状方法 + 拓扑组件

| 方法 | SAM 角色 | 核心机制 | 拓扑 / 选择 | 威胁 | 差异 / 代码 |
|---|---|---|---|---|---|
| **HarmonySeg** ICCV'25 | 非 SAM | 深浅特征融合 + Growth-Suppression loss | 拓扑 loss；clDice | M（强管状基线） | 无 SAM/选择 |
| **TopoTTA** ICCV'25 | 非 SAM | 拓扑元差分卷积 + pseudo-break 对齐 | 拓扑 TTA；clDice/Betti | M（拓扑推理先例） | 无 SAM/选择；+31.81 clDice |
| **DeformCL** CVPR'25 | 非 SAM | 可变形中心线图 | 连通性内在；Betti | M（中心线表示） | [code](https://github.com/barry664/DeformCL) |
| **vesselFM** CVPR'25 | 非 SAM 3D | 3D 血管基础模型 | clDice 指标 | M（基础模型基线） | 无 adapter/选择 |
| **VesMamba** CVPR'26 | 非 SAM 3D | Mamba 结构感知 + 尺度滤波 | clDice/HD95 | L/M | [code](https://github.com/Lzpbright/VesMamba) |
| **SATLoss** ICCV'25 | 非 SAM | 空间感知持久特征匹配 | 拓扑 loss | **H（组件）** | 可直接监督候选拓扑 · [code](https://github.com/bw05025/SATLoss) |
| **MATCH** NeurIPS'25 | 非 SAM | 多面拓扑一致性匹配 | 拓扑匹配 | **H（组件）** | 候选打分/匹配先例 · [code](https://github.com/Melon-Xu/MATCH) |
| **Topograph** ICLR'25 | 非 SAM | 组件图严格拓扑 loss | 严格拓扑 | **H（组件）** | 已在 `SACM-X` 中 · [code](https://github.com/AlexanderHBerger/Topograph) |
| **SCNP** CVPR'26 | 非 SAM | 邻居像素惩罚 | 拓扑 loss | M（低成本 loss） | 可做候选打分 · [project](https://jmlipman.github.io/SCNP-SameClassNeighborPenalization) |
| **TopoFuse** ECCV'26 | SAM ViT-B tri-planar | 持久同调投影 + 拓扑先验头 | 拓扑推理；非选择 | H/M（拓扑推理先例） | 投影修正，不是候选选择 · [code](https://github.com/rohitsalla/TopoFuse) |

---

## 3. 选择/排序先例（最影响新颖性论证）

| 方法 | 设定 | 核心机制 | 选择方式 | 威胁 | 差异 |
|---|---|---|---|---|---|
| **P3-SAM** ECCV'26 | 3D part SAM | 多 head + **trained IoU predictor** + NMS | IoU 回归 | **H（可学习选择已存在）** | 无拓扑；3D part · [arXiv](https://arxiv.org/abs/2509.06784) |
| **SEQSAM** ISBI'25 | SAM decoder | 回归重提示 + **Hungarian 全假设监督** | 无推理选择 | **H（全候选监督已存在）** | 无拓扑；多标注歧义 · [code](https://github.com/BenjaminTowle/SeqSAM) |
| **Topology-Aware Query Selection** arXiv'26 | Mask2Former | query 图 + 结构化子集选择（MILP） | 关系拓扑 | **H（措辞撞车）** | 是 query 图拓扑，不是像素/管状拓扑 · [code](https://github.com/yqjyzzz/structured-instance-set-selection) |
| **SphereTrust** arXiv'26 | SAM/SAM3 候选 | 冻结 DINOv2 特征训练-free 排序 | 外观/几何分数 | M | 无拓扑；可做训练-free 基线 · [arXiv](https://arxiv.org/abs/2609.30080) |
| **DSS** CVPR'26 | SAM 候选池 | MLLM 语义重排 | MLLM | M | 选择瓶颈已被认识；重、无结构拓扑 · [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Discover_Segment_and_Select_A_Progressive_Mechanism_for_Zero-shot_Camouflaged_CVPR_2026_paper.html) |
| **PR-MaGIC** CVPR'26 | SAM in-context | 5 候选 + top-1 相似度 | 相似度 | M | 无拓扑 · [project](https://postech-minjaelee.github.io/PR-MaGIC/) |
| **JAM** BIBM'25 | SAM2 | prototype similarity-coverage 选择 | 原型相似度 | M | 训练-free；无拓扑 · [DOI](https://doi.org/10.1109/BIBM66473.2025.11356921) |
| **SAMIX** CVPR'26 | SAM2 | semantic adapter + RL **参考图**选择 | 参考选择 | M | 不是输出 mask 选择 |
| **Topological Voting** 2022 | 非学习 | 拓扑投票 | 拓扑 | M（概念先例） | 不能声称“first topology-aware selection” · [DOI](https://doi.org/10.3390/jimaging8020016) |

其他相关但不必进主表：MaskSAM（ICCV'25，auto-prompt + 3D adapter）、IP-SAM（ECCV'26）、M4-SAM（CVPR'26）、Focus on Background（CVPR'26）、PF-SAM（TCSVT'26）、SA-SAM2（2026）、HydroSAM（2026）、SAM4EM（CVPRW'25）、VesselSAM（2025）、SAM2LoRA（2025）、Progressive Curriculum（2025，ATRL 拓扑 loss）、Dynamic Exploration（2026，候选图扩展）、Fast χ Euler（TMI'25）、TF-SSD / iFAN / MARS（训练-free 候选择优）。

---

## 4. 扫描结论：什么被占了，空位在哪里

### 4.1 已经被占住的

1. **PF0 + 通用 adapter + 多候选/IoU 选择**：SACM。
2. **PF0 + 几何/方向 adapter**：CoroSAM（方向）、FineSAM++（LoRA + 专家）、TopoLoRA-SAM（LoRA + clDice）、A2SAM（结构 prompt + 厚度）。
3. **window adapter**：MSA-SAM（各向同性多尺度 window）；**strip conv**：Topology-Constrained SAM2-UNet（decoder 侧）、DSC-AMP、Strip R-CNN（非 SAM）。
4. **可学习候选选择**：SAM 自身 IoU head、P3-SAM、SACM；**全候选监督**：SEQSAM；**语义/训练-free 选择**：DSS、SAMIX、JAM、SphereTrust、PR-MaGIC。
5. **“topology-aware selection”这个说法**：Topology-Aware Query Selection（关系图拓扑）；**拓扑投票**：2022 年已有非学习版本。
6. **拓扑感知推理**：TopoFuse（PH 投影）、TopoTTA（TTA）、HybridTopo。

### 4.2 仍然空着的交集（本次检索范围内未发现）

> **严格 PF0 + frozen SAM ViT 内的各向异性/方向性 window/strip adapter + 以像素/管状拓扑为选择目标的可训练候选选择 + 全候选拓扑监督 + 可微 top-k + 曲线结构上的 selection regret 分析。**

注意：前两条腿各自都有人做了（SACM/CoroSAM/MSA-SAM；P3-SAM/SEQSAM），所以**不能把“prompt-free”“多候选选择”“window adapter”“拓扑选择”任何一个单独当卖点**。能立住的是三者的具体交集和选择准则。

### 4.3 最强威胁排序

- **H**：SACM（直接基线）、P3-SAM（trained IoU 选择）、SEQSAM（全候选监督）、FineSAM++（PF0 血管）、CoroSAM（几何 PF0）、MSA-SAM（window adapter）。
- **H（措辞）**：Topology-Aware Query Selection——必须在 Related Work 里明确“关系图拓扑 vs 像素/管状拓扑”。
- **H（组件）**：SATLoss、MATCH、Topograph、TopoFuse——分别是拓扑监督、匹配、严格 loss、拓扑推理的最强先例。
- **M**：PR-MaGIC、DSS、SphereTrust、JAM、Topological Voting——证明“选择”已被广泛研究，但都不是拓扑目标。

---

## 5. 建议的论文定位

**一句话（可直接用）：**

> Unlike SACM, P3-SAM, and SEQSAM, which train or select candidates with IoU/quality scores or Hungarian matching, we make **structural topology the explicit selection objective**: every candidate receives topology-aware supervision, and a differentiable top-k selector is trained end-to-end, so the decoder prefers connected, branch-faithful masks even when IoU prefers a broken one.

**不要声称：**

- first prompt-free SAM for curvilinear segmentation（SACM/CoroSAM/TopoLoRA 已占）
- first multi-candidate selection（SAM/P3-SAM/SACM 已占）
- first window adapter / strip convolution（MSA-SAM / DSC-AMP / Strip R-CNN 已占）
- first topology-aware selection（Topological Voting 2022、Topology-Aware Query Selection 2026 已占措辞）
- first clDice / topology loss（更早）

**必须正面对比：**

- 同一候选集上的选择器：SACM IoU、SAM predicted IoU、confidence、post-hoc clDice/Betti rerank、SphereTrust/DSS/ACES（training-free）、**oracle selector**。
- 单 mask clDice 基线：TopoLoRA-SAM、Topology-Constrained SAM2-UNet。
- 可学习选择先例：P3-SAM、SEQSAM。
- 拓扑组件：SATLoss、MATCH、Topograph、SCNP。

---

## 6. 最小证据包（CVPR 审稿人一定会要）

1. **同一候选集上的选择器对决**：SACM IoU、SAM predicted IoU、confidence、post-hoc clDice/Betti rerank、SphereTrust/DSS/ACES（training-free）、oracle selector。
2. **消融**：WTA vs 全候选监督；final-only vs per-candidate 拓扑监督；IoU vs clDice vs Betti vs PH 目标；argmax vs Gumbel vs straight-through top-k；top-1 vs top-k；多样性/负载均衡。
3. **指标**：Dice/IoU、clDice、Betti-0/1、HD95、component count / branch recall、**selection regret / oracle gap / 拓扑分数校准**。
4. **数据**：12 数据集 3-shot + 跨域 + 3 seeds；并和单 mask clDice 基线正面对比。

**失败分析必须包含**：Dice/IoU 把“断开的 mask”排在“连通 mask”之前的案例；拓扑选择翻转决策；以及翻转后的指标代价。

---

## 7. 与 `SACM-X` 的对应关系

**已有资产：**

- GeoAdapter-I / AdapterEGlobal（window 层）；
- FusionV2、闭环 C2F、MultiDepthPath；
- 深监督、IoU 监督、soft clDice、topograph 等 14 种拓扑损失；
- 16 + 4 指标、3-shot 数据构建、DDP 分布式验证、官方数值对拍。

**被威胁的点：**

- window adapter：MSA-SAM；
- strip conv：Topology-Constrained SAM2-UNet、DSC-AMP、Strip R-CNN；
- prompt-free 血管适配：FineSAM++、CoroSAM、TopoLoRA-SAM；
- 多候选/可学习选择：SACM、P3-SAM、SEQSAM。

**真正的空位：**

- 像素/管状拓扑作为**选择目标**（不是只做 loss/评测）；
- 所有候选的拓扑监督；
- 可微 / straight-through top-k 选择器；
- 曲线结构上的 selection regret / IoU–拓扑分歧分析；
- 与冻结 SAM ViT 内各向异性 window/strip adapter 的组合。

**必须补的实验：**

- Stage-2 全候选监督 / straight-through 选择（修复当前 WTA 残留）；
- 拓扑感知选择头（IoU + clDice/Betti 目标）；
- 选择器 head-to-head 与 oracle gap；
- SACM 基线保真（`--selection gating --pred_threshold 0.5 --no_augment` + 统一 LR/epoch）。

---

## 8. 待核对 / TODO

- [ ] 核对所有 2026 venue、DOI、arXiv 编号与 camera-ready（尤其标记 `†` 的条目）。
- [ ] 复核 FineSAM++ / SnakeSAM / TopoSAM / A2SAM / SA-SAM2 的全文细节（本表部分为摘要级）。
- [ ] 确认 Topology-Aware Query Selection 的“topology”定义，写清与像素/管状拓扑的区别。
- [ ] 运行 SACM 基线保真实验，确保主表对比有效。
- [ ] 实现并 smoke test：Stage-2 全候选监督 + topology-aware selector。
- [ ] 跑选择器 head-to-head（SACM/SAM/confidence/post-hoc/oracle）。
- [ ] 按“被占位/空位”框架重写 Related Work。

---

## 9. 重要修正（2026-10-18）：SACM 的定位是 few-shot 泛化，不是绝对 SOTA

SACM 的卖点是 **18 张图（6 数据集 × 3-shot）+ 跨域泛化**，不是所有算法中的绝对最优。因此后续论文与实验按以下口径调整：

- **主张**：限定为“同一 18 图 3-shot 预算下的 few-shot / cross-domain 最优 + 拓扑保持”，不要写 absolute SOTA / 全面超过全监督方法。
- **主表**：按 Base-Seen / Base-Unseen / Novel 分组；SACM（同协议）、SAM zero-shot、本方法变体为主对比；全监督方法（HarmonySeg、DeformCL、vesselFM、VesMamba、nnU-Net 等）放在“参考上界”栏，明确标注不是同预算对手。
- **其它 few-shot/迁移方法**：CaPro（1 张无标注图，training-free）、FineSAM++、TopoLoRA-SAM 等按各自预算标注，避免 apples-to-oranges。
- **指标**：主看 clDice / HD95 / Betti + Dice/IoU，并报告 Seen / Unseen / Novel 的分层结果；选择故事的核心证据是 **Unseen/Novel 上的 selection regret / oracle gap 是否缩小**。
- **故事**：从“绝对精度更高”改为“**域偏移下的选择鲁棒性**”——IoU 校准对域敏感，拓扑感知选择应主要在 Unseen/Novel 上获益。
- **Pilot**：至少 2 个 Seen（DRIVE、CHASEDB1）+ 2 个 Unseen/Novel（WIRE、ROAD）；决策门看跨域 clDice/HD95 与 selection regret，不只看 in-domain Dice。
- **协议风险**：官方 3-shot split 未公开时，用多个 seed 报告 mean±std，并在论文中说明 split/val 的差异；不要声称逐位复现。
- **代码改动不变**：拓扑选择头 + 全候选/ST 训练 + 诊断脚本；诊断脚本按 Seen/Unseen/Novel 分层。

---

## 10. 全语料补充（2026-10-20）：250 篇 parsing 中的 pre-2025 基础工作

`parsing/` 已扩展到 250 篇（与 `pdf/` 一致）。除前述 2025–2026 工作外，全语料扫描发现以下更早的、必须引用的基础工作：

- **TopoSeg: Topology-Aware Nuclear Instance Segmentation**（ICCV 2023）——提出 **adaptive topology-aware selection (ATS)**，按 Betti error 选择**图像/图像块**参与拓扑 loss 计算，不是选择输出 mask 候选。它占用了“topology-aware selection”这个措辞，但对象是训练样本/区域；我们的主张必须明确写成 **inference-time mask-candidate selection**。
- **Dynamic Snake Convolution Based on Topological Geometric Constraints for Tubular Structure Segmentation**（ICCV 2023）——动态 snake 卷积，把管状几何先验做进卷积核；是与我们的 strip/window adapter 最接近的早期工作之一，必须对比。
- **DTU-Net: Learning Topological Similarity for Curvilinear Structure Segmentation**（IPMI 2023）——曲线结构的拓扑相似性学习；作为曲线拓扑基线引用。
- **Topology-Aware Uncertainty for Image Segmentation**（NeurIPS 2023）——拓扑与不确定性结合；与“不确定性/校准驱动的候选选择”相关。
- **Skeleton Recall Loss**（ECCV 2024）——骨架召回连通性 loss；已在 `3rd/Skeleton-Recall`，但尚未接入 `SACM-X`。
- **Topologically Faithful Image Segmentation via Induced Matching of Persistence Barcodes**（ICML 2023）——Betti matching 的原始论文；已通过 `Betti-matching` 接入。
- **Structure-aware Image Segmentation with Homotopy Warping**（NeurIPS 2022）——homotopy warping loss；已通过 `Warping` 接入。

**对定位的影响**：这些工作说明“拓扑 + 选择/几何先验”本身不新；真正需要精确表述的是：
1. 选择对象是 **prompt-free SAM 解码器输出的多个 mask 候选**，不是训练图像/区域；
2. 选择发生在 **inference-time**，且目标是 **few-shot / cross-domain** 下的拓扑可靠性；
3. 选择信号是 **像素/管状拓扑**（clDice/Betti/持久同调），不是关系图拓扑或语义相似度。

同时，250 篇全语料扫描没有再发现 2025–2026 中比前述表格更直接的威胁；新增的主要是这些更早的基础工作。

---

## 主要来源

- [SACM, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_Dual-level_Adapter_Boosting_Prompt-free_Curvilinear_Structure_Segmentation_CVPR_2026_paper.html#1) · [code](https://github.com/kylechuuuuu/SACM)
- [CaPro, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37315) · [code](https://github.com/xmed-lab/CaPro)
- [FineSAM++, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/97a0f7d7fc2c26732943860241af1c7a-Abstract-Conference.html)
- [TopoLoRA-SAM, arXiv 2601.02273](https://arxiv.org/abs/2601.02273)
- [CoroSAM, IEEE BIBM 2025](https://ieeexplore.ieee.org/document/11355979) · [code](https://github.com/HITZhengGW/CoroSAM)
- [MSA-SAM, JSTARS 2025](https://doi.org/10.1109/JSTARS.2025.3525801) · [code](https://github.com/mint0126/Mult-scale-SAM)
- [Topology-Constrained SAM2-UNet, Electronics 2026](https://doi.org/10.3390/electronics15184330)
- [SnakeSAM, Array 2026](https://doi.org/10.1016/j.array.2026.101221)
- [TopoSAM, EAAI 2026](https://doi.org/10.1016/j.engappai.2025.113688)
- [A2SAM, Medical Image Analysis 2026](https://doi.org/10.1016/j.media.2026.104285)
- [P3-SAM, ECCV 2026](https://arxiv.org/abs/2509.06784)
- [SEQSAM, ISBI 2025](https://arxiv.org/abs/2503.09797) · [code](https://github.com/BenjaminTowle/SeqSAM)
- [Topology-Aware Query Selection, arXiv 2608.11607](https://arxiv.org/abs/2608.11607)
- [SphereTrust, arXiv 2609.30080](https://arxiv.org/abs/2609.30080)
- [DSS, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Discover_Segment_and_Select_A_Progressive_Mechanism_for_Zero-shot_Camouflaged_CVPR_2026_paper.html)
- [PR-MaGIC, CVPR 2026](https://postech-minjaelee.github.io/PR-MaGIC/)
- [JAM, IEEE BIBM 2025](https://doi.org/10.1109/BIBM66473.2025.11356921)
- [TopoFuse, ECCV 2026](https://arxiv.org/abs/2609.29717) · [code](https://github.com/rohitsalla/TopoFuse)
- [SATLoss, ICCV 2025](https://github.com/bw05025/SATLoss)
- [MATCH, NeurIPS 2025](https://github.com/Melon-Xu/MATCH)
- [Topograph, ICLR 2025](https://github.com/AlexanderHBerger/Topograph)
- [SCNP, CVPR 2026](https://jmlipman.github.io/SCNP-SameClassNeighborPenalization)
- [Topological Voting, J. Imaging 2022](https://doi.org/10.3390/jimaging8020016)
