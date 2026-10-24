# SAM ViT-L + U-Net (project extension)

- Not one of the paper's ten baselines; this is our own SAM+U-Net probe.
- Encoder: the self-contained `models.sam_network.build_sam` implementation
  (`src/models/sam_network`), initialized from SAM ViT-L.
- Decoder: copied from the previous project adapter
  (`baselines/adapters/sam_vit_l_unet.py`) unchanged.
- Runtime consumer: `baseline_zoo.adapters.sam_vit_l_unet.SAMViTLUNetAdapter`
