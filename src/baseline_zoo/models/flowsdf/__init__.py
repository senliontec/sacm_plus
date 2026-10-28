"""Self-contained FlowSDF port (IJCV 2025).

Image-guided conditional flow matching on truncated signed distance
functions.  The velocity network is the official UNet with an RRDB image
encoder (128 base channels, multipliers 1-1-2-2-4-4, three residual blocks
per level, attention at strides 8/16, 12 RRDB blocks).

Source: https://github.com/leabogensperger/FlowSDF
Paper: FlowSDF: Flow Matching for Medical Image Segmentation Using Distance
Transforms, International Journal of Computer Vision, 2025.
"""

from .sdf import (
    euler_sample,
    predict_tiled,
    predict_whole,
    signed_distance_map,
    tile_starts,
)
from .unet import UNetModel

__all__ = [
    'UNetModel',
    'euler_sample',
    'predict_tiled',
    'predict_whole',
    'signed_distance_map',
    'tile_starts',
]


def build_flowsdf(
    *,
    model_channels: int = 128,
    rrdb_blocks: int = 12,
    image_channels: int = 3,
) -> UNetModel:
    """Build the official FlowSDF velocity network (1-channel SDF)."""
    return UNetModel(
        in_channels=1,
        model_channels=model_channels,
        out_channels=1,
        num_res_blocks=3,
        attention_resolutions=(16, 8),
        dropout=0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        rrdb_blocks=rrdb_blocks,
        image_channels=image_channels,
    )
