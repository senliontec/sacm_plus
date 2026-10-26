# SAM ViT-L + U-Net (project extension)

- Not one of the paper's ten baselines; this is our own SAM+U-Net probe.
- Encoder: the self-contained `models.sam_network.build_sam` implementation
  (`src/models/sam_network`), initialized from SAM ViT-L.
- Decoder: self-contained in `baseline_zoo.models.sam_vit_l_unet`; the
  previous project adapter is not part of the runtime tree.
- Runtime consumer: `baseline_zoo.adapters.sam_vit_l_unet.SAMViTLUNetAdapter`
