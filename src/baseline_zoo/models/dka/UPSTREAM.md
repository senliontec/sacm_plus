# DKA upstream record

## Sources

| Component | Upstream | Commit / version |
|---|---|---|
| DKA module / training code | `misswayguy/DKA`, `peft_model/adapter/vit_adapter.py`, `experiments/cnn/adapter_tuning_dka.py` | `70f45a2ecaf7aab3ebbf1b175853988c15219318` |
| Segmenter-B host architecture | OpenMMLab MMSegmentation, `mmseg/models/backbones/vit.py`, `mmseg/models/utils/embed.py`, `mmseg/models/decode_heads/fcn_head.py` | `main` |
| Segmenter-B pretrained checkpoint | `vit_base_p16_384_20220308-96dfe169.pth` (ImageNet-21k) | 2022-03-08 |

Paper: "Dual-Kernel Adapter: Expanding Spatial Horizons for
Data-Constrained Medical Image Analysis", ICLR 2026.

License: DKA is MIT (repository `LICENSE`); MMSegmentation is Apache-2.0.

## What was copied / reconstructed

- `DKAModule`: the active DKA adapter from `vit_adapter.py`:
  `adapter_down` (Linear, no bias), parallel depthwise `conv_large`
  (51×51, padding 25, groups=middle_dim) and `conv_small` (5×5, padding 2),
  element-wise sum, GELU, dropout 0.1, `adapter_up` (Linear, no bias),
  residual.  The released file contains many commented experimental
  variants (dilated kernels, extra convolutions); only the active
  dual-kernel path is ported.
- `SegmenterBackbone` / `SegmenterDKA`: a self-contained reconstruction of
  MMSegmentation's `VisionTransformer` ViT-B/16 Segmenter-B host, because
  the released DKA repository does not include its segmentation code.  It
  reproduces the MMSeg key layout exactly:
  `patch_embed.projection`, `cls_token`, `pos_embed`, `ln1`,
  `layers.*.ln1`, `layers.*.attn.attn.*`, `layers.*.ln2`,
  `layers.*.ffn.layers.*`.  DKA is inserted after every transformer block;
  a linear 1×1 FCN head produces the two-class logits.
- The official checkpoint stores a 384-input (24×24 patch) position
  embedding; it is bicubically resized to 512 (32×32) at load, matching
  MMSeg's `resize_pos_embed`.

## Invariants kept from the paper / MMSeg Segmentation

- DKA in all 12 transformer blocks (the released classification entrypoint
  uses `selected_layers=list(range(0, 12))`).
- Middle dimension 192 for segmentation (16 for classification).
- Head learning rate 1e-4, DKA learning rate 1e-3, Adam, batch size 1.
- Frozen pretrained backbone; only DKA and the task head are trained.
- DKA large kernel 51×51, small kernel 5×5 (the paper's selected
  combination), depthwise, no bias.
- Paper B.2 zero initialization of the down-projection and up-projection
  weights/biases; the released code has this commented out, so the port
  follows the paper (adapter is identity at initialization).
- MMSeg Segmenter-B host: ViT-B/16, 512-square input, learned absolute
  position embedding, `drop_path_rate=0.1` applied to the FFN branch only
  (MMSeg's attention branch has no DropPath), final LayerNorm, class token,
  linear 1×1 FCN head, data normalization `(x-127.5)/127.5` with BGR→RGB
  (the project loader already returns RGB).
- The official ImageNet-21k Segmenter-B checkpoint
  `vit_base_p16_384_20220308-96dfe169.pth` is loaded strictly after the
  position-embedding resize; no key remapping is used.

## Deviations / remaining uncertainty

1. Segmentation augmentation and loss are not specified in the paper.
   The port uses 512-square stretch resize, horizontal flip only, and
   cross-entropy + foreground Dice.  The Dice term is a task adaptation for
   the 0.25-3% foreground protocol domains.
2. The official 300-epoch segmentation budget is extended to the shared
   800-epoch budget with validation-Dice selection.  Batch size 1 is kept
   (the paper's segmentation batch).
3. The standard MMSeg ViT-B Segmenter config uses a mask-transformer head;
   the DKA paper text says "linear head", so this port uses an FCN-style
   linear 1×1 head.  The head has no dropout (MMSeg's FCNHead default is
   0.1, while Segmenter linear configs set 0.0); the authors' exact head
   setting cannot be verified from the released code, which contains no
   segmentation entrypoint.
4. Test-time augmentation/scale is not specified; the port evaluates a
   single 512 view and exports the mask at GT resolution.  MMSeg's slide
   test pipeline is not used.
5. No DKA-trained weights exist; training starts from the official
   Segmenter-B checkpoint.  `--dka-random-init` is a code-smoke option only
   and must not be used for reported results.  Checkpoints produced by the
   earlier torchvision ViT-B/16 reconstruction have a different state-dict
   layout and must be retrained with this host.
6. The official checkpoint is required locally at
   `checkpoints/segmenter_vit_b16_384.pth`.  Download URL:
   `https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_base_p16_384_20220308-96dfe169.pth`.
