"""Self-contained Conv-LoRA binary semantic segmenter.

Architecture follows the ICLR 2024 Conv-LoRA paper and the official
AutoGluon implementation:

- standard SAM image encoder (ViT-B/L/H), frozen except Conv-LoRA qkv layers;
- Conv-LoRA = low-rank qkv update plus a top-1 MoE of multi-scale 3x3 conv
  experts, inserted into the image-encoder self-attention;
- prompt encoder frozen; no prompts are used (the no-mask embedding is the
  constant prompt);
- mask decoder fully fine-tuned, reduced to one binary mask token;
- structure loss (weighted BCE + weighted IoU) and the official MoE
  load-balancing loss.

The official AutoGluon model uses HuggingFace SAM; this port uses the
Apache-2.0 Meta SAM implementation vendored under ``segment_anything`` and
only torch code so the offline server needs no ``transformers``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .layers import ConvLoRALinear
from .segment_anything import sam_model_registry


def _inject_convlora(encoder: nn.Module, rank: int, alpha: int,
                     experts: int) -> None:
    """Replace each ViT attention qkv projection with ConvLoRALinear."""
    for block in encoder.blocks:
        old = block.attn.qkv
        if not isinstance(old, nn.Linear):
            raise TypeError(f'unexpected qkv type: {type(old)}')
        layer = ConvLoRALinear(
            old.in_features,
            old.out_features,
            r=rank,
            lora_alpha=alpha,
            conv_lora_expert_num=experts,
        )
        layer.weight = old.weight
        layer.bias = old.bias
        block.attn.qkv = layer


def _to_binary_decoder(decoder: nn.Module) -> None:
    """Reduce SAM's four mask tokens to the paper's one binary token."""
    with torch.no_grad():
        token = decoder.mask_tokens.weight.data[0:1].clone()
    decoder.mask_tokens = nn.Embedding(1, decoder.transformer_dim)
    with torch.no_grad():
        decoder.mask_tokens.weight.copy_(token)
    decoder.output_hypernetworks_mlps = nn.ModuleList(
        [decoder.output_hypernetworks_mlps[0]])
    decoder.num_multimask_outputs = 0
    decoder.num_mask_tokens = 1


class ConvLoRASAM(nn.Module):
    # AutoGluon's Conv-LoRA semantic-segmentation config sets
    # ``image_norm: imagenet``; the released pipeline applies
    # Resize -> ToTensor (0-1) -> Normalize(ImageNet) before the HF SAM
    # encoder.  Meta SAM's own pixel mean/std are NOT used upstream.
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, sam: nn.Module, moe_weight: float = 1.0):
        super().__init__()
        self.sam = sam
        self.moe_weight = float(moe_weight)
        self.image_size = int(sam.image_encoder.img_size)
        self.register_buffer(
            'imagenet_mean',
            torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            'imagenet_std',
            torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False)

    def _moe_loss(self) -> torch.Tensor:
        losses = [
            module.last_moe_loss
            for module in self.sam.image_encoder.modules()
            if isinstance(module, ConvLoRALinear)
            and module.last_moe_loss is not None
        ]
        if not losses:
            return torch.zeros((), device=self.sam.pixel_mean.device)
        return torch.stack(losses).sum()

    def forward(self, images: torch.Tensor):
        # ``images`` are 0-255 RGB tensors, matching torchvision ToTensor
        # before AutoGluon's Normalize.  The official Conv-LoRA pipeline
        # uses ImageNet statistics rather than SAM's pixel mean/std.
        input_images = ((images / 255.0) - self.imagenet_mean) \
            / self.imagenet_std
        image_embeddings = self.sam.image_encoder(input_images)
        sparse, dense = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None)
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        # Standard SAM returns [B, 1, 256, 256] for the binary token.
        logits = self.sam.postprocess_masks(
            low_res_masks,
            input_size=(self.image_size, self.image_size),
            original_size=(self.image_size, self.image_size),
        )
        return logits, self._moe_loss()


def build_convlora(
    checkpoint: str | Path,
    model_type: str = 'vit_h',
    image_size: int = 1024,
    rank: int = 3,
    alpha: int = 8,
    experts: int = 8,
    moe_weight: float = 1.0,
) -> ConvLoRASAM:
    """Build Conv-LoRA around a local Meta SAM checkpoint.

    ``image_size`` must match the checkpoint architecture (1024 for the
    released ViT-B/L/H checkpoints).  Following the official AutoGluon
    recipe, ``ConvLoRASAM.forward`` normalizes 0-255 inputs with ImageNet
    statistics (not SAM's pixel mean/std).
    """
    if model_type not in sam_model_registry:
        raise ValueError(f'unknown SAM model type: {model_type}')
    if int(image_size) != 1024:
        raise ValueError(
            'Conv-LoRA uses the released SAM checkpoints, whose encoder '
            'image size is 1024')
    if rank < 1 or experts < 1:
        raise ValueError('rank and experts must be positive')
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    _to_binary_decoder(sam.mask_decoder)
    _inject_convlora(sam.image_encoder, rank, alpha, experts)

    # Official policy: image encoder mostly frozen (only Conv-LoRA), prompt
    # encoder frozen, mask decoder fully fine-tuned.
    for parameter in sam.parameters():
        parameter.requires_grad_(False)
    for parameter in sam.mask_decoder.parameters():
        parameter.requires_grad_(True)
    # The official AutoGluon config freezes the IoU prediction head (it is
    # not part of the binary mask objective) and the prompt encoder.
    sam.mask_decoder.iou_prediction_head.requires_grad_(False)
    for module in sam.image_encoder.modules():
        if not isinstance(module, ConvLoRALinear):
            continue
        for name, parameter in module.named_parameters():
            if name.startswith(('lora_A', 'lora_B', 'lora_moe')):
                parameter.requires_grad_(True)
    sam.prompt_encoder.requires_grad_(False)

    return ConvLoRASAM(sam, moe_weight=moe_weight)


__all__ = ['ConvLoRASAM', 'build_convlora']
