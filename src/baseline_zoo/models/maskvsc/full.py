"""Two-stage MaskVSC: masked-vessel segmentation + ViG completion.

Upstream reference:
``3rd/MaskVSC/train.py`` (stage 1) and ``3rd/MaskVSC/train_ViG.py`` /
``model/gcn.py`` (stage 2).  The official inference path is
``GCN_model(Seg_model(image))``; this wrapper keeps both stages in one
``nn.Module`` so the shared runtime can save/load one checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn


class MaskVSCFull(nn.Module):
    """Stage-1 UNet plus stage-2 ViG completion head.

    Stage 1 is trainable only in ``stage1``.  In ``stage2`` it is frozen and
    forced to eval mode (BatchNorm running statistics), matching the official
    two-script workflow while avoiding the upstream missing ``eval()`` call.
    """

    def __init__(self, seg: nn.Module, vig: nn.Module):
        super().__init__()
        self.seg = seg
        self.vig = vig
        self.stage = 'stage1'
        self.set_stage('stage1')

    def set_stage(self, stage: str) -> None:
        if stage not in ('stage1', 'stage2'):
            raise ValueError(f'unknown MaskVSC stage: {stage!r}')
        self.stage = stage
        if stage == 'stage1':
            self.seg.requires_grad_(True)
            self.vig.requires_grad_(False)
        else:
            self.seg.requires_grad_(False)
            self.vig.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.stage == 'stage1':
            self.seg.train()
            self.vig.eval()
        elif mode and self.stage == 'stage2':
            self.seg.eval()
            self.vig.train()
        else:
            self.seg.eval()
            self.vig.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.stage == 'stage1':
            return self.seg(images)
        with torch.no_grad():
            seg_probability = self.seg(images)
        return self.vig(seg_probability)
