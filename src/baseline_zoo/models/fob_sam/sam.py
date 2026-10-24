"""Frozen SAM prompt-to-mask wrapper from the official FoB implementation.

Vendored from ``baselines/FoB_SAM/SAM.py`` (Apache-2.0).  The only
changes are device handling, an explicit model-type/checkpoint switch for
smoke tests, and a constant-image guard in ``pre_process``.  SAM itself
is the official self-contained Segment Anything copy already used by the
SAM-OCTA baseline, so FoB does not carry a second 2.4 GiB model package.

The official code hard-codes ``best_pred_idx = 1`` for ISIC and ``0`` for
the medical datasets.  This port exposes ``best_pred_idx`` (default 0);
our 20-dataset protocol contains no ISIC images, so 0 is the faithful
choice.
"""

import numpy as np
import torch.nn as nn

from baseline_zoo.models.sam_octa.segment_anything import (
    SamPredictor,
    sam_model_registry,
)


class SAM(nn.Module):
    def __init__(self, sam_pretrained_path, model_type='vit_h',
                 device='cpu', best_pred_idx=0):
        super().__init__()
        self.model_type = model_type
        self.best_pred_idx = int(best_pred_idx)
        self.get_sam(str(sam_pretrained_path) if sam_pretrained_path else None)
        self.to(device)

    def get_sam(self, checkpoint_path):
        print(f'Using model type {self.model_type}')
        self.sam = sam_model_registry[self.model_type](
            checkpoint=checkpoint_path).eval()
        self.predictor = SamPredictor(self.sam)
        self.sam.requires_grad_(False)

    def predict_w_points_bbox(self, sam_input_points, bboxes,
                              sam_neg_input_points, qry_img):
        """Predict a mask from positive/negative point prompts."""
        self.predictor.set_image(qry_img)

        bbox_xyxy = None
        all_points = []
        all_labels = []

        for point in sam_input_points:  # positive points
            assert (qry_img.max() <= 255 and qry_img.min() >= 0
                    and qry_img.dtype == np.uint8)
            if point is not None:
                all_points.append(point)
                all_labels.extend([1] * len(point))

        if sam_neg_input_points is not None:  # negative points
            for neg_point in sam_neg_input_points:
                if neg_point is not None:
                    all_points.append(neg_point)
                    all_labels.extend([0] * len(neg_point))

        if all_points:
            points = np.vstack(all_points)
            point_labels = np.array(all_labels)
        else:
            points = None
            point_labels = None

        mask, score, logits = self.predictor.predict(
            point_coords=points,
            point_labels=point_labels,
            box=bbox_xyxy if bbox_xyxy is not None else None,
            return_logits=False,
            multimask_output=True,
        )
        return [mask[self.best_pred_idx]], [score[self.best_pred_idx]]

    def pre_process(self, image):
        """Convert a CHW float tensor in [0,1]-ish to HWC uint8."""
        image = image.permute(1, 2, 0).cpu().numpy()
        lo, hi = image.min(), image.max()
        # Official code divides by (max - min); a constant image would
        # produce NaNs.  Guard only that degenerate case.
        if hi - lo < 1e-12:
            image = np.zeros_like(image)
        else:
            image = ((image - lo) / (hi - lo) * 255).astype(np.uint8)
        return image

    def forward(self, query_image, pos_point=None, neg_point=None,
                return_logits=False):
        self.image_size = query_image.shape[:2]
        query_image = self.pre_process(query_image)
        mask, score = self.predict_w_points_bbox(
            pos_point, None, neg_point, query_image)
        return mask[0]
