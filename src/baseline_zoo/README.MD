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

## Baseline ↔ 3rd 代码 ↔ parsing 论文

### 非 SAM 方法

| Adapter id | 模型 | 3rd 代码目录 | parsing 论文目录 | 备注 |
|---|---|---|---|---|
| `unet` | U-Net | `3rd/Pytorch-UNet` | 未收录 | milesial PyTorch-UNet；U-Net 原论文目录未收录 |
| `cs2net` | CS2Net | `3rd/CS-Net` | 未收录 | 官方 CS-Net 代码 |
| `bcunet` | BCUNet | `3rd/BCU-Net` | 未收录 | 无 LICENSE；`OutConv` 已修复 |
| `maskvsc` | MaskVSC | `3rd/MaskVSC` | 未收录 | 视网膜血管分割 |
| `nnwnet` | nnWNet | `3rd/nnWNet` | 未收录 | 官方 nnU-Net trainer/plugin |
| `jtfn` | JTFN | `3rd/JTFN` | `parsing/[2021 iccv] Joint Topology-Preserving and Feature-Refinement Network for Curvilinear Structure Segmentation` | parsing 另有小写重复目录；上游无 LICENSE |
| `dscnet` | DSCNet | `3rd/DSCNet` | `parsing/[2023 iccv] Dynamic Snake Convolution Based on Topological Geometric Constraints for Tubular Structure Segmentation` | parsing 另有 2 个大小写变体重复目录；blending kernel 已修复 |
| `segdino` | SegDINO | `3rd/segdino` | 未收录 | DPT + DINOv3；权重需本地准备 |

### SAM 监督方法

| Adapter id | 模型 | 3rd 代码目录 | parsing 论文目录 | 备注 |
|---|---|---|---|---|
| `sam_med2d` | SAM-Med2D | `3rd/SAM-Med2D` | 未收录 | 官方 SAM-Med2D checkpoint 缺失；当前为 ViT-L 二值适配 |
| `sam_octa` | SAM-OCTA | `3rd/SAM-OCTA` | 未收录 | 上游 URL 未记录；另有 `3rd/sam-octa2` 参考 |
| `cwsam` | CWSAM | `3rd/CWSAM` | 未收录 | ClassWise-SAM-Adapter；ViT-L、二值 2-class 适配 |
| `sam_adapter` | SAM-Adapter | `3rd/SAM-Adapter-PyTorch` | `parsing/[2023 iccvw] SAM-Adapter: Adapting Segment Anything in Underperformed Scenes` | 无作者训练权重，从 SAM ViT-L 训练 |
| `topolora` | TopoLoRA-SAM | `3rd/Seglab` | `parsing/[2026 icpr] TopoLoRA-SAM: Topology-Aware Parameter-Efficient Adaptation of Foundation Segmenters for Thin-Structure and Cross-Domain Binary Semantic Segmentation` | 官方仓库是 Seglab，不是独立 TopoLoRA 目录；实现位于 `seglab/models/sam_peft/`；parsing 另有 arxiv 重复目录 |
| `samed` | SAMed | `3rd/SAMed` | `parsing/[2023 arxiv] Customized Segment Anything Model for Medical Image Segmentation` | 官方 8 类权重未用于二值头；从 SAM ViT-B 训练 |
| `fob_sam` | FoB-SAM | `3rd/FoB_SAM` | `parsing/[2026 cvpr] Focus on Background: Exploring SAM's Potential in Few-shot Medical Image Segmentation with Background-centric Prompting` | parsing 另有 “Focus on the Background” 重复目录 |
| `sacm_official` | SACM (official) | `3rd/SACM` | `parsing/[2026 cvpr] Dual-level Adapter Boosting Prompt-free Curvilinear Structure Segmentation` | 官方 SACM baseline，不是 SACM-X |

### Training-free / semi-supervised

| Adapter id | 模型 | 3rd 代码目录 | parsing 论文目录 | 备注 |
|---|---|---|---|---|
| `persam` | PerSAM | `3rd/Personalize-SAM` | `parsing/[2024 lclr] PERSONALIZE SEGMENT ANYTHING MODEL WITH ONE SHOT` | 官方 one-shot；项目 3-shot 采用多数投票；目录名 `lclr` 为原目录拼写 |
| `matcher` | Matcher | `3rd/Matcher` | `parsing/[2024 iclr] Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching` | DINOv2 ViT-L/14 权重缺失 |
| `knowsam` | KnowSAM | `3rd/KnowSAM` | 未收录 | 半监督；unlabeled pool 已做泄漏审计 |

### 参考 / 探针 / 扩展

| Adapter id | 模型 | 3rd 代码目录 | parsing 论文目录 | 备注 |
|---|---|---|---|---|
| `sam2_unet` | SAM2-UNet | `3rd/SAM2-UNet` | `parsing/[2024 arxiv] SAM2-UNet: Segment Anything 2 Makes Strong Encoder for Natural and Medical Image Segmentation` | 附录扩展 baseline；需 SAM2 Hiera-L 权重 |
| `sam_vit_l_unet` | SAM ViT-L + U-Net | `3rd/segment-anything` | `parsing/[2023 iccv] Segment Anything` | 本项目扩展，不是论文 baseline；3rd 仅提供 SAM 基座，U-Net decoder 为自包含实现 |
| `ucs` | UCS | `3rd/UCS` | `parsing/[2023 iccv] UCS: A Universal Model for Curvilinear Structure Segmentation` | oracle-prompted reference，不是 prompt-free |

## 快速核对

```bash
cd /home/dev/mywork/project/segment
ls 3rd/Pytorch-UNet 3rd/JTFN 3rd/Seglab 3rd/UCS
ls "parsing/[2021 iccv] Joint Topology-Preserving and Feature-Refinement Network for Curvilinear Structure Segmentation"
ls "parsing/[2023 iccv] UCS: A Universal Model for Curvilinear Structure Segmentation"
```
