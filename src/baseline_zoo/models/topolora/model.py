"""Prompt-free TopoLoRA-SAM model.

Ported from ``3rd/Seglab`` (``seglab/models/sam_peft``, MIT).  The official
model loads SAM, freezes it, injects LoRA into the ViT MLP layers, adds a
depthwise-separable conv adapter on the image embeddings, unfreezes the mask
decoder, and decodes with empty prompts (the prompt encoder's learned
``no_mask_embed`` dense embedding).  The Lightning wrapper is replaced by a
plain ``nn.Module`` so the shared baseline runtime can own training.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .adapters import ConvAdapter
from .lora import inject_lora
from .segment_anything import sam_model_registry

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TopoLoRASAM(nn.Module):
    """SAM ViT + LoRA + conv adapter with the official prompt-free decoder."""

    def __init__(self, sam: nn.Module,
                 adapter: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.sam = sam
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The official datasets feed ImageNet-normalized tensors.  TopoLoRA-SAM
        # inverts that normalization and hands [0, 255] RGB to SAM.
        mean = torch.tensor(
            IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(
            IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x_01 = (x * std) + mean
        x_255 = (x_01.clamp(0, 1) * 255.0).to(dtype=torch.float32)

        img_size = int(getattr(self.sam.image_encoder, 'img_size', 1024))
        if x_255.shape[-2:] != (img_size, img_size):
            x_255 = F.interpolate(
                x_255, size=(img_size, img_size), mode='bilinear',
                align_corners=False)

        x_sam = self.sam.preprocess(x_255)
        embeddings = self.sam.image_encoder(x_sam)
        if self.adapter is not None:
            embeddings = self.adapter(embeddings)

        sparse, dense = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None)
        low_res, _ = self.sam.mask_decoder(
            image_embeddings=embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        return F.interpolate(
            low_res, size=x.shape[-2:], mode='bilinear',
            align_corners=False)


def build_topolora(
    sam_type: str = 'vit_b',
    checkpoint: Optional[str] = None,
    lora_enabled: bool = True,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    lora_targets: Iterable[str] = ('mlp', 'fc1', 'fc2'),
    adapter_enabled: bool = True,
    adapter_hidden: int = 256,
    adapter_kernel: int = 3,
    train_mask_decoder: bool = True,
) -> TopoLoRASAM:
    """Build the official TopoLoRA-SAM graph.

    ``checkpoint`` is the raw SAM checkpoint loaded before LoRA injection.
    Pass ``None`` for prediction: the runtime then loads the complete
    TopoLoRA checkpoint (base SAM + LoRA + adapter) afterwards.
    """
    if sam_type not in sam_model_registry:
        raise ValueError(f'unknown SAM type: {sam_type}')
    sam = sam_model_registry[sam_type](
        checkpoint=str(checkpoint) if checkpoint is not None else None)

    for parameter in sam.parameters():
        parameter.requires_grad = False
    if lora_enabled:
        inject_lora(
            sam.image_encoder,
            target_keywords=lora_targets,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

    adapter = None
    if adapter_enabled:
        adapter = ConvAdapter(
            channels=256, hidden_dim=adapter_hidden,
            kernel_size=adapter_kernel)
    if train_mask_decoder:
        for parameter in sam.mask_decoder.parameters():
            parameter.requires_grad = True
    return TopoLoRASAM(sam, adapter)
