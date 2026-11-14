"""Matched-inference control for the TubeToken experiment.

This adapter is *not* a new method.  It loads the canonical DCD fixed3
checkpoint and applies exactly the same inference-time operations that the
TubeToken deployment uses, so the contribution of those operations can be
separated from the contribution of the checkpoint.

Nothing in ``baseline_zoo`` is modified.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baseline_zoo.adapters.dcd import DCDAdapter
from baseline_zoo.data import load_rgb


class DCDReferenceControl(DCDAdapter):
    id = 'dcd_reference_control'
    display_name = 'DCD reference (matched inference)'
    official_source = 'DCD fixed3 checkpoint'
    official_entrypoint = 'DCD fixed3 + optional flip TTA / native tiling'
    deviations = DCDAdapter.deviations + (
        'Inference-only control: optional horizontal-flip averaging and '
        'native-resolution tiling, with no architectural or training change.',
    )

    def __init__(self):
        super().__init__()
        self.tta_flip = False
        self.tile_inference = False
        self.tile_size = 1024
        self.tile_stride = 768
        self.tile_min_size = 3000
        self.final_upsample = True

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument(
            '--tta-flip', action='store_true',
            help='horizontal-flip test-time averaging')
        parser.add_argument(
            '--tile-inference', action='store_true',
            help='native-resolution tiling for large images')
        parser.add_argument('--tile-size', type=int, default=1024)
        parser.add_argument('--tile-stride', type=int, default=768)
        parser.add_argument(
            '--tile-min-size', type=int, default=3000,
            help='tile only when the original long side reaches this size')
        parser.add_argument(
            '--final-upsample', type=int, choices=[0, 1], default=1,
            help='upsample selected logits to 1024 before GT resampling, '
                 'matching the TubeToken deployment path')

    def _read_inference_args(self, args):
        self.tta_flip = bool(getattr(args, 'tta_flip', False))
        self.tile_inference = bool(
            getattr(args, 'tile_inference', False))
        self.tile_size = int(getattr(args, 'tile_size', 1024))
        self.tile_stride = int(getattr(args, 'tile_stride', 768))
        self.tile_min_size = int(
            getattr(args, 'tile_min_size', 3000))
        self.final_upsample = bool(
            int(getattr(args, 'final_upsample', 1)))

    def build_model(self, args):
        model = super().build_model(args)
        self._read_inference_args(args)
        return model

    def _deploy_logits(self, model, images):
        stage2, _ = self._forward(model, images, return_stage1=False)
        if self.head_fusion:
            logits = self._fused_heads(model, stage2)
        else:
            index = self._selected_stage2_index()
            logits = stage2[:, index:index + 1]
        if self.final_upsample:
            logits = F.interpolate(
                logits, size=(1024, 1024), mode='bilinear',
                align_corners=False)
        return logits

    def _predict_tiled(self, model, sample):
        image = load_rgb(sample.image_path)
        width, height = image.size
        if max(width, height) <= self.tile_size:
            return None
        if max(width, height) < self.tile_min_size:
            return None
        array = np.asarray(image, dtype=np.float32).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
        tensor = tensor.to(self.device)
        canvas = torch.zeros(1, height, width, device=self.device)
        weight = torch.zeros(1, height, width, device=self.device)
        size = self.tile_size
        stride = max(1, min(self.tile_stride, size))
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                y1, x1 = min(y + size, height), min(x + size, width)
                if y1 - y < 16 or x1 - x < 16:
                    continue
                patch = torch.zeros(3, size, size, device=self.device)
                patch[:, :y1 - y, :x1 - x] = tensor[:, y:y1, x:x1]
                logits = self._deploy_logits(model, patch.unsqueeze(0))
                logits = F.interpolate(
                    logits, size=(size, size), mode='bilinear',
                    align_corners=False)[0, 0]
                canvas[0, y:y1, x:x1] += logits[:y1 - y, :x1 - x]
                weight[0, y:y1, x:x1] += 1.0
        canvas = canvas / weight.clamp_min(1.0)
        threshold = float(
            getattr(self.args, 'deploy_threshold', 0.5))
        return (torch.sigmoid(canvas)[0].cpu().numpy() > threshold)

    def predict_sample(self, model, prepared, sample):
        if self.tile_inference:
            tiled = self._predict_tiled(model, sample)
            if tiled is not None:
                return tiled
        logits = self._deploy_logits(model, prepared)
        if self.tta_flip:
            flipped = torch.flip(prepared, dims=[3])
            flip_logits = self._deploy_logits(model, flipped)
            logits = 0.5 * (logits + torch.flip(flip_logits, dims=[3]))
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        logits = F.interpolate(
            logits, size=(height, width), mode='bilinear',
            align_corners=False)
        threshold = float(
            getattr(self.args, 'deploy_threshold', 0.5))
        return (torch.sigmoid(logits)[0, 0].cpu().numpy() > threshold)
