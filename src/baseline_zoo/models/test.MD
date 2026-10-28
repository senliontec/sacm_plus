# Baseline model -> 3rd source directory

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
├── convlora              -> 3rd/autogluon/examples/automm/Conv-LoRA
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
├── sam_vit_l_unet        -> (no 3rd; project extension)
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
- `sam_vit_l_unet`: no `3rd` source. This is a project extension using
  `src/models/sam_network` and the local SAM ViT-L checkpoint.

## A2SAM naming warning

There are two different upstream projects whose names differ only by case:

| 3rd directory | Paper | Task | Used by our baseline? |
|---|---|---|---|
| `3rd/A2SAM` | MIA 2026, "Informativeness-driven active adaptation of SAM" | Medical tubular/vessel segmentation | Yes: `models/a2sam` |
| `3rd/a2sam` | NeurIPS 2024, "Flaws can be Applause: Unleashing Potential of Segmenting Ambiguous Objects in SAM" | Ambiguous-object segmentation (LIDC / Sim10k) | No |

`models/a2sam` maps to **uppercase** `3rd/A2SAM`, not lowercase `3rd/a2sam`.
The repaired provenance/source-of-truth record is `docs/A2SAM_FIDELITY.md`.
