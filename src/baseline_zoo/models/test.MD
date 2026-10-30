# Baseline model -> 3rd source directory

unet cs2net bcunet maskvsc nnwnet sam_med2d  sam_octa cwsam  knowsam  segdino sacm_official

# 可以参考 ICLR ICML NIPS ACL 等顶会中对 SAM 做高效微调的方法

`3rd` root: `/home/dev/mywork/project/segment/3rd`

From this file's directory: `../../../../3rd`

`3rd` is a read-only upstream reference. Do not import from it at runtime.

```text
.
├── a2sam                 -> 3rd/A2SAM 
├── aco_moe_lora          -> 3rd/ACO-MoE-LoRA
├── bcunet                -> 3rd/BCU-Net
├── blo_sam               -> 3rd/BLO-SAM
├── c2p                   -> 3rd/Concept-to-Pixel
├── convlora              -> 3rd/autogluon/examples/automm/Conv-LoRA [牛逼]
├── cs2net                -> 3rd/CS-Net
├── csunet                -> 3rd/CSUNet
├── cwsam                 -> 3rd/CWSAM
├── dconnnet              -> 3rd/DconnNet
├── dka                   -> 3rd/DKA
├── dscnet                -> 3rd/DSCNet
├── flowsdf               -> 3rd/FlowSDF
├── fssam                 -> 3rd/FSSAM
├── jtfn                  -> 3rd/JTFN
├── knowsam               -> 3rd/KnowSAM
├── kprism                -> 3rd/K-Prism
├── liot                  -> 3rd/LIOT
├── maskvsc               -> 3rd/MaskVSC
├── matcher               -> 3rd/Matcher
├── medcagd               -> 3rd/MedCAGD
├── nnunet2d              -> 3rd/nnUNet
├── nnwnet                -> 3rd/nnWNet
├── persam                -> 3rd/Personalize-SAM
├── rsrm                  -> 3rd/RSRM
├── s2mnet                -> 3rd/S2M-Net-Spectral-Spatial-Mixing-for-Medical-Segmentation
├── sacm_official         -> 3rd/SACM
├── sam2_unet             -> 3rd/SAM2-UNet
├── sam_adapter           -> 3rd/SAM-Adapter-PyTorch
├── samed                 -> 3rd/SAMed
├── sam_med2d             -> 3rd/SAM-Med2D
├── sam_octa              -> 3rd/SAM-OCTA
├── sansa                 -> 3rd/SANSA
├── scrwkv                -> 3rd/SCRWKV
├── segdino               -> 3rd/segdino
├── segdino_v2            -> 3rd/segdino_v2
├── topolora              -> 3rd/Seglab
├── tpsegformer           -> 3rd/TPSegformer
├── ucs                   -> 3rd/UCS
└── unet                  -> 3rd/Pytorch-UNet
```

## Multi-directory / special mappings

- `convlora`: Conv-LoRA code is inside AutoGluon.
  - `3rd/autogluon/examples/automm/Conv-LoRA`
  - `3rd/autogluon/multimodal/src/autogluon/multimodal/models`
  - `3rd/autogluon/multimodal/src/autogluon/multimodal/optim/losses`
  - `3rd/autogluon/multimodal/src/autogluon/multimodal/configs/model`
- `dka`: `3rd/DKA` contains the DKA adapter. The Segmenter-B host was
  reconstructed from MMSegmentation; there is no local MMSeg checkout.
- `dscnet`: exact upstream code is
  `3rd/DSCNet/DSCNet_2D_opensource/Code/DRIVE/DSCNet`.
- `nnunet2d`: `3rd/nnUNet` is the upstream reference; the runtime uses the
  installed `nnunetv2` package, not this tree.
- `topolora`: primary source is `3rd/Seglab`; SAM reference is
  `3rd/segment-anything`.
- `bcunet`: paper is
  `parsing/[2023 cbm] BCU-Net: Bridging ConvNeXt and U-Net for medical image
  segmentation`; official code is `3rd/BCU-Net`. The paper profile uses MRL,
  average 0.5/0.5 fusion, bilinear U-Net and 18x augmentation; the release
  main.py CE/0.4-0.6/no-aug profile is kept as an ablation. See
  `models/bcunet/UPSTREAM.md` for the paper-code conflicts.
- `maskvsc`: upstream is two-stage. Stage 1 is
  `3rd/MaskVSC/train.py` + `test.py` (UNet, masked-vessel corruption,
  Connect_Loss); stage 2 is `3rd/MaskVSC/train_ViG.py` + `test_ViG.py` +
  `model/gcn.py` (ViG completion on the frozen stage-1 prediction). The
  forward inference path is `ViG(Seg(image))`. The paper is
  `parsing/[2025 tmi] Masked Vascular Structure Segmentation and Completion
  in Retinal Images`. The default stage-2 uses the released patch16
  token-fold decode and soft GCN labels (ViG 23.84M, full model 31.61M).
  The main baseline uses release L_C, flip-only augmentation and the
  dot-product ViG graph. Paper L_C (N/foreground scale, no 3x3 opening),
  ±30° rotation and Euclidean KNN are explicit ablations; paper L_C gives
  sparse masks 10-100x BCE weight and collapsed CORN/CrackTree/DCA1 in the
  pooled 3-shot run, so it needs recalibration via
  `--maskvsc-connect-cap` before use. A 100-epoch seed-0 stage-1 A/B gives
  best val Dice 0.4612 (release L_C, no rotation) and 0.4726
  (release L_C, rotation 30) versus 0.1133/0.1636 for paper L_C. The main
  baseline therefore keeps release L_C and flip-only augmentation. The
  earlier patch32 token-grid reconstruction is only `coherent`. See
  `models/maskvsc/UPSTREAM.md`.
- `nnwnet`: upstream is `3rd/nnWNet`; the paper is
  `parsing/[2025 cvpr] nnWNet: Rethinking the Use of Transformers in
  Biomedical Image Segmentation and Calling for a Unified Evaluation
  Benchmark`. WNet matches the paper (7.04M parameters); the project uses
  nnU-Net 2.8.1, 800 epochs, 18/6 one-fold and shared metrics, while the
  paper uses 2.3.1, 200 epochs, 5-fold and Jaccard/Dice/95HD/ASD. These are
  documented deviations; every epoch (default interval 1) it runs full-image shared
  validation (Dice/IoU/clDice/HD95 + per-dataset) and selects by val Dice,
  while patch pseudo-Dice is logged only under `nnunet/*`. See
  `models/nnwnet/UPSTREAM.md`.
- `kprism`: upstream is the whole `3rd/K-Prism` tree (commit `78c3b1b`,
  MIT). The port copies `kprism/modeling/{backbone,pixel_fuser,meta_arch,
  point_encoder,transformer_decoder,loss.py}`; the in-context
  `modeling/task_encoder` is not used because only the prompt-free semantic
  branch (`click_mode='0'`, zero dense prompt) is ported. There is no
  `parsing/` paper directory and no released K-Prism checkpoint; the README
  plus released code are the reference. Runtime code is
  `src/baseline_zoo/models/kprism`; provenance is in its `UPSTREAM.md`.

## A2SAM naming warning

There are two different upstream projects whose names differ only by case:

| 3rd directory | Paper | Task | Used by our baseline? |
|---|---|---|---|
| `3rd/A2SAM` | MIA 2026, "Informativeness-driven active adaptation of SAM" | Medical tubular/vessel segmentation | Yes: `models/a2sam` |
| `3rd/a2sam` | NeurIPS 2024, "Flaws can be Applause: Unleashing Potential of Segmenting Ambiguous Objects in SAM" | Ambiguous-object segmentation (LIDC / Sim10k) | No |

`models/a2sam` maps to **uppercase** `3rd/A2SAM`, not lowercase `3rd/a2sam`.
The repaired provenance/source-of-truth record is `docs/A2SAM_FIDELITY.md`.
