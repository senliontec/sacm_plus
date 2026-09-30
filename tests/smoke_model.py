"""Fast end-to-end smoke test: build the full model, run one forward pass
and print the tensor shapes. Run with `make smoke`.

This is the first thing to run on a new machine: it verifies the
architecture switches, the zero-prompt path and the tensor flow without
any dataset or training. The forward protocol lives in model_io.py, so
the smoke test exercises exactly the same code path as train/test.
"""

import torch

from models.sam import build_sam_vit_l
from core.io import get_prompt_embeddings, predict_all


def main():
    sam = build_sam_vit_l(
        checkpoint=None,
        use_adapter=True,
        adapter_dim_ratio=0.1,
        use_geo_i=True,
        use_geo_e=True,
        use_coarse_to_fine=True,
        use_fusion_v2=True,
        use_multi_depth=True,
    )

    x = torch.randn(1, 3, 1024, 1024)
    sparse, dense = get_prompt_embeddings(sam, 1)
    masks, iou_pred, masks_stage1 = predict_all(sam, x, sparse, dense, return_stage1=True)

    print(f"masks: {tuple(masks.shape)} | iou: {tuple(iou_pred.shape)} | stage1: {tuple(masks_stage1.shape)}")
    assert masks.shape == (1, 4, 256, 256)
    assert iou_pred.shape == (1, 4)
    assert masks_stage1.shape == (1, 4, 256, 256)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
