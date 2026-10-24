from .lora import LoRA_Sam
from .losses import DiceLoss, clDiceLoss
from .prompts import label_to_point_prompt_global
from .segment_anything import sam_model_registry


def build_sam_octa(checkpoint, model_type: str = 'vit_l', rank: int = 4):
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    return LoRA_Sam(sam, rank)


__all__ = ['DiceLoss', 'LoRA_Sam', 'build_sam_octa', 'clDiceLoss',
           'label_to_point_prompt_global', 'sam_model_registry']
