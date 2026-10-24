from .sam import SAM

VIT_L_ENCODER = {
    'name': 'sam',
    'img_size': 1024,
    'mlp_ratio': 4,
    'patch_size': 16,
    'qkv_bias': True,
    'use_rel_pos': True,
    'window_size': 14,
    'out_chans': 256,
    'scale_factor': 32,
    'input_type': 'fft',
    'freq_nums': 0.25,
    'prompt_type': 'highpass',
    'prompt_embed_dim': 256,
    'tuning_stage': 1234,
    'handcrafted_tune': True,
    'embedding_tune': True,
    'adaptor': 'adaptor',
    'embed_dim': 1024,
    'depth': 24,
    'num_heads': 16,
    'global_attn_indexes': [5, 11, 17, 23],
}


def build_cwsam(num_classes: int = 2, inp_size: int = 1024) -> SAM:
    return SAM(inp_size=inp_size, encoder_mode=dict(VIT_L_ENCODER),
               loss='iou', num_classes=num_classes, loss_weight=None)


__all__ = ['SAM', 'VIT_L_ENCODER', 'build_cwsam']
