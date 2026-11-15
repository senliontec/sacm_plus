# CurviMix-SAM model package

This package is a physical, independent copy of the four-candidate SAM ViT-L
decoder, adapters and topology losses used by CurviMix-SAM.  It imports no
SPA-SAM/DCD/SACM model at runtime.

- Adapter: `baseline_zoo.adapters.curvimix.CurviMixAdapter`
- Architecture: frozen SAM ViT-L + four Stage-2 candidates + fixed candidate 3
  (`head_select=fixed`, `deployment_head=3`, `head_fusion=0`)
- Training: mild topology-preserving augmentation + structure-aware
  copy-paste; donors come from the training split only.
- Inference: fixed3 single pass by default; optional 4-view flip TTA.
- fixed3 helper: `derive_fixed3.py` (checks `adapter=curvimix`).
- Config: `configs/models/curvimix.yaml`.
