"""LoRA wrapper for the SAMed image encoder (MIT, from 3rd/SAMed)."""

from __future__ import annotations

import math
from typing import Optional

import torch.nn as nn

from .segment_anything.modeling import Sam


class _LoRA_qkv(nn.Module):
    """SAM qkv projection with rank-r LoRA on the q and v paths."""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class LoRA_Sam(nn.Module):
    """Official SAMed LoRA wrapper: frozen image encoder, q/v LoRA."""

    def __init__(self, sam_model: Sam, r: int,
                 lora_layer: Optional[list[int]] = None):
        super().__init__()
        if r <= 0:
            raise ValueError('LoRA rank must be positive')
        self.lora_layer = (list(range(len(sam_model.image_encoder.blocks)))
                           if lora_layer is None else list(lora_layer))
        self.w_As: list[nn.Linear] = []
        self.w_Bs: list[nn.Linear] = []

        for parameter in sam_model.image_encoder.parameters():
            parameter.requires_grad = False

        for layer_i, block in enumerate(sam_model.image_encoder.blocks):
            if layer_i not in self.lora_layer:
                continue
            qkv = block.attn.qkv
            dim = qkv.in_features
            w_a_q = nn.Linear(dim, r, bias=False)
            w_b_q = nn.Linear(r, dim, bias=False)
            w_a_v = nn.Linear(dim, r, bias=False)
            w_b_v = nn.Linear(r, dim, bias=False)
            self.w_As.extend([w_a_q, w_a_v])
            self.w_Bs.extend([w_b_q, w_b_v])
            block.attn.qkv = _LoRA_qkv(
                qkv, w_a_q, w_b_q, w_a_v, w_b_v)
        self.reset_parameters()
        self.sam = sam_model

    def reset_parameters(self) -> None:
        for w_a in self.w_As:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_Bs:
            nn.init.zeros_(w_b.weight)

    def forward(self, batched_input, multimask_output, image_size):
        return self.sam(batched_input, multimask_output, image_size)
