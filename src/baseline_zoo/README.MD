# Baseline zoo

本目录中的每个 baseline 都是自包含的：模型代码在 `src/baseline_zoo/models/<adapter_id>/`，
官方训练/推理逻辑在 `src/baseline_zoo/adapters/<adapter_id>.py`。`3rd/` 只作为官方代码
参考树，运行时不得 import 或 patch。

## 路径约定

- `3rd/...`、`parsing/...` 均相对于 `/home/dev/mywork/project/segment/`。
- `parsing/<论文目录>/` 内的 `full.md` 是解析文本，`*_origin.pdf` 是原 PDF，
  `layout.json`、`content_list*.json` 和 `images/` 是版面/图片解析结果。
- **未收录** 表示 `parsing/` 中没有对应论文目录；不是链接失效。
- 同一篇论文在 `parsing/` 中可能存在大小写不同的重复目录，表中列主目录，并在备注中标出重复项。
- 当前覆盖：42 个 adapter（41 个官方/参考 baseline + 1 个项目扩展
  TopoWeave）中，18 个在 `parsing/` 有论文目录，其余标记为“未收录”。

## Baseline ↔ 3rd 代码 ↔ parsing 论文

### 非 SAM 方法

| Adapter id | 模型 | 3rd 代码目录 | parsing 论文目录 | 备注 |
|---|---|---|---|---|
| `unet` | U-Net | `3rd/Pytorch-UNet` | 未收录 | milesial PyTorch-UNet；U-Net 原论文目录未收录 |
| `cs2net` | CS2Net-2D（历史 id） | `3rd/CS-Net` | 未收录 | 官方 2D 分支 `model/csnet.py` + `train.py`；`model/csnet_3d.py` 是独立 3D 扩展，不用于 2D 数据 |
| `bcunet` | BCUNet | `3rd/BCU-Net` | `parsing/[2023 cbm] BCU-Net: Bridging ConvNeXt and U-Net for medical image segmentation` | 无 LICENSE；按论文实现 MRL + 0.5/0.5 融合 + bilinear + 18× 离线增强 + 100 epochs；release CE/0.4-0.6/无增强作为消融；OutConv 1×1 logits 重建 |
| `maskvsc` | MaskVSC | `3rd/MaskVSC` | `parsing/[2025 tmi] Masked Vascular Structure Segmentation and Completion in Retinal Images` | 视网膜血管；完整两阶段（UNet masked-vessel + ViG completion）；官方 patch16 token-fold 解码（ViG 23.84M，总 31.61M）；主表用 release L_C/flip-only/点积 KNN；paper L_C/±30°旋转/欧氏 KNN 为显式消融（paper L_C 在稀疏域过强且跳过 3×3 opening，需用 `--maskvsc-connect-cap` 重标定）；patch32 仅 `coherent` 消融 |
| `nnwnet` | nnWNet | `3rd/nnWNet` | `parsing/[2025 cvpr] nnWNet: Rethinking the Use of Transformers in Biomedical Image Segmentation and Calling for a Unified Evaluation Benchmark` | 官方 nnU-Net trainer/plugin；WNet 7.04M；项目协议 800/18-6；每 epoch 全图共享验证并按 val Dice 选模（interval 默认 1），W&B schema 统一；论文 2.3.1/200/5-fold/ASD 为文档化偏差 |
| `nnunet2d` | nnU-Net 2D | `3rd/nnUNet` | 未收录 | 原生 nnunetv2 2D planner/trainer/predictor |
| `flowsdf` | FlowSDF | `3rd/FlowSDF` | `parsing/[2025 ijcv] FlowSDF: Flow Matching for Medical Image Segmentation Using Distance Transforms` | 条件流匹配 + SDF；128 crop，Euler ODE；MIT |
| `tpsegformer` | TPSegformer | `3rd/TPSegformer` | `parsing/[2026 cvpr] Hilbert Curve-Based Attention Enabling Topology-Preserving Image Tensor Representation for Semantic Segmentation Network` | Swin-T + SEM/HLFF/LCCM；MMSeg overlay 已改为纯 PyTorch；官方 Swin-T 权重自动加载并做 key 映射；上游无 LICENSE |
| `s2mnet` | S2M-Net | `3rd/S2M-Net-Spectral-Spatial-Mixing-for-Medical-Segmentation` | `parsing/[2026 icml] S2M-Net: Spectral-Spatial Mixing with Morphology-Aware Adaptive Loss for Medical Image Segmentation` | TF/Keras 重写为 PyTorch；MRF-SE + SSTM + BFP + MAL；上游无 LICENSE |
| `csunet` | CS-UNet | `3rd/CSUNet` | `parsing/[2026 icassp] Curvilinear Spectral U-Net: A Framework for Structure-Aware Road Extraction from Very High Resolution Imagery` | TF notebook 重写为 PyTorch；GFR/FSF + Snake；DynamicSnake 为占位实现；上游无 LICENSE |
| `jtfn` | JTFN | `3rd/JTFN` | `parsing/[2021 iccv] Joint Topology-Preserving and Feature-Refinement Network for Curvilinear Structure Segmentation` | parsing 另有小写重复目录；上游无 LICENSE |
| `liot` | LIOT-IterNet | `3rd/LIOT` | `parsing/[2022 tip] Local Intensity Order Transformation for Robust Curvilinear Object Segmentation` | 4 通道 LIOT + IterNet；128 crop、CE；亮/暗前景极性映射；MIT |
| `dconnnet` | DconnNet | `3rd/DconnNet` | `parsing/[2023 cvpr] Directional connectivity-based segmentation of medical images` | 8 方向连通性；960 BGR；ResNet-34 权重需本地准备；上游无 LICENSE |
| `dscnet` | DSCNet | `3rd/DSCNet` | `parsing/[2023 iccv] Dynamic Snake Convolution Based on Topological Geometric Constraints for Tubular Structure Segmentation` | parsing 另有 2 个大小写变体重复目录；blending kernel 已修复 |
| `segdino` | SegDINO | `3rd/segdino` | 未收录 | DPT + DINOv3；权重需本地准备；默认冻结 backbone |
| `segdino_v2` | SegDINO v2 | `3rd/segdino_v2` | `parsing/` 未收录（arXiv 2606.17972） | TPA/SAD decoder；DINOv3-S 本地权重 |

### SAM 监督方法

| Adapter id | 模型 | 3rd 代码目录 | parsing 论文目录 | 备注 |
|---|---|---|---|---|
| `aco_moe_lora` | ACO-MoE-LoRA | `3rd/ACO-MoE-LoRA` | `parsing/[2026 icml] ACO-MoE-LoRA: Evolving-while-Training for Adapting Segment Anything Model 2 to Specialized Domains` | SAM2 Hiera-L + dynamic LoRA + ACO/MoE decoder；上游无 LICENSE |
| `a2sam` | A2SAM (MIA 2026) | `3rd/A2SAM` | `parsing/[2026 MIA] Informativeness-driven active adaptation of SAM: Structural prompts and contrastive parameter selection for medical tubular segmentation` | 官方 SAM/prompt fusion/CPS/TA + 论文级 U-Net/prior + cripser persistence/Morse 重建；见 `docs/A2SAM_FIDELITY.md`；勿与 `3rd/a2sam` (NeurIPS 2024) 混淆 |
| `sam_med2d` | SAM-Med2D | `3rd/SAM-Med2D` | 未收录 | 默认官方 ViT-B `sam-med2d_b.pth`；ViT-L 仅作显式对照 |
| `sam_octa` | SAM-OCTA | `3rd/SAM-OCTA` | 未收录 | 上游 URL 未记录；另有 `3rd/sam-octa2` 参考；默认 ViT-L，可选 H |
| `cwsam` | CWSAM | `3rd/CWSAM` | 未收录 | ClassWise-SAM-Adapter；默认官方 ViT-B，ViT-L 可选、二值 2-class 适配 |
| `sam_adapter` | SAM-Adapter | `3rd/SAM-Adapter-PyTorch` | `parsing/[2023 iccvw] SAM-Adapter: Adapting Segment Anything in Underperformed Scenes` | 无作者训练权重，从 SAM ViT-L 训练 |
| `topolora` | TopoLoRA-SAM | `3rd/Seglab` | `parsing/[2026 icpr] TopoLoRA-SAM: Topology-Aware Parameter-Efficient Adaptation of Foundation Segmenters for Thin-Structure and Cross-Domain Binary Semantic Segmentation` | 官方仓库是 Seglab，不是独立 TopoLoRA 目录；实现位于 `seglab/models/sam_peft/`；parsing 另有 arxiv 重复目录 |
| `samed` | SAMed | `3rd/SAMed` | `parsing/[2023 arxiv] Customized Segment Anything Model for Medical Image Segmentation` | 官方 8 类权重未用于二值头；从 SAM ViT-B 训练 |
| `sacm_official` | SACM (official) | `3rd/SACM` | `parsing/[2026 cvpr] Dual-level Adapter Boosting Prompt-free Curvilinear Structure Segmentation` | 官方 SACM baseline，不是 SACM-X |

### Training-free / semi-supervised

| Adapter id | 模型 | 3rd 代码目录 | parsing 论文目录 | 备注 |
|---|---|---|---|---|
| `persam` | PerSAM | `3rd/Personalize-SAM` | `parsing/[2024 lclr] PERSONALIZE SEGMENT ANYTHING MODEL WITH ONE SHOT` | 官方 one-shot；项目 3-shot 采用多数投票；目录名 `lclr` 为原目录拼写 |
| `matcher` | Matcher | `3rd/Matcher` | `parsing/[2024 iclr] Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching` | DINOv2 ViT-L/14 权重缺失 |
| `knowsam` | KnowSAM | `3rd/KnowSAM` | 未收录 | 半监督；unlabeled pool 已做泄漏审计 |
| `kprism` | K-Prism | `3rd/K-Prism` | 未收录 | ICLR 2026；仅 prompt-free semantic 分支，zero dense prompt；无官方权重；512 输入；MIT |

### 参考 / 探针 / 扩展

| Adapter id | 模型 | 3rd 代码目录 | parsing 论文目录 | 备注 |
|---|---|---|---|---|
| `sam2_unet` | SAM2-UNet | `3rd/SAM2-UNet` | `parsing/[2024 arxiv] SAM2-UNet: Segment Anything 2 Makes Strong Encoder for Natural and Medical Image Segmentation` | 附录扩展 baseline；需 SAM2 Hiera-L 权重 |
| `sam_vit_l_unet` | SAM ViT-L + U-Net | `3rd/segment-anything` | `parsing/[2023 iccv] Segment Anything` | 本项目扩展，不是论文 baseline；3rd 仅提供 SAM 基座，U-Net decoder 为自包含实现 |
| `topoweave` | TopoWeave | 无（新方法） | 未收录 | 本项目扩展，由 MaskVSC 的设计教训衍生；单阶段 GroupNorm ResUNet + 门控双输入图 refiner + width-aware TopoMask + bounded soft-clDice；不是官方 baseline，无官方权重/parity；见 `models/topoweave/UPSTREAM.md` |
| `ucs` | UCS | `3rd/UCS` | `parsing/[2023 iccv] UCS: A Universal Model for Curvilinear Structure Segmentation` | oracle-prompted reference，不是 prompt-free |

## 未集成 / 已排除（不属于 registry 的方法）

以下方法的 `3rd/` 代码和 `parsing/` 论文都在本地，但未进入 `baseline_zoo` 的 adapter 列表，
不计入上表 42 个 adapter。

| 方法 | 3rd 代码目录 | parsing 论文目录 | 状态 |
|---|---|---|---|
| CoroSAM | `3rd/CoroSAM` | `parsing/[2026] CoroSAM: adaptation of the Segment Anything Model for interactive segmentation in Coronary angiograms` | 已排除：代码不完整，不作为定量 baseline |
| MedSAM | `3rd/MedSAM` | `parsing/[2024 nature communication] Segment anything in medical images` | 未集成 |
| MSA-SAM | `3rd/Mult-scale-SAM` | `parsing/[2025 JSTARS] Multiscale Adapter Based on SAM for Remote Sensing Semantic Segmentation` | 未集成 |
| SAMIX | `3rd/SAMIX` | `parsing/[2026 cvpr] SAMIX: Reinforcing SAM2 with Semantic Adapter and Reference Selecting Policy for Mix-Supervised Segmentation` | 未集成 |

## 快速核对

```bash
cd /home/dev/mywork/project/segment
ls 3rd/Pytorch-UNet 3rd/JTFN 3rd/Seglab 3rd/UCS
ls "parsing/[2021 iccv] Joint Topology-Preserving and Feature-Refinement Network for Curvilinear Structure Segmentation"
ls "parsing/[2023 iccv] UCS: A Universal Model for Curvilinear Structure Segmentation"
```
