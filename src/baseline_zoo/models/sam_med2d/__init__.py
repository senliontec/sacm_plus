from types import SimpleNamespace

from .segment_anything import sam_model_registry


def build_sam_med2d(checkpoint, model_type: str = 'vit_l',
                     image_size: int = 256,
                     encoder_adapter: bool = True):
    args = SimpleNamespace(
        image_size=image_size,
        sam_checkpoint=str(checkpoint),
        encoder_adapter=encoder_adapter,
    )
    return sam_model_registry[model_type](args)


__all__ = ['build_sam_med2d', 'sam_model_registry']
