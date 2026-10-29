from .sam import SAM

VIT_B_ENCODER = {
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
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'global_attn_indexes': [2, 5, 8, 11],
}


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


ENCODERS = {'vit_b': VIT_B_ENCODER, 'vit_l': VIT_L_ENCODER}


def build_cwsam(num_classes: int = 2, inp_size: int = 1024,
                encoder: str = 'vit_b') -> SAM:
    if encoder not in ENCODERS:
        raise ValueError(f'unknown CWSAM encoder: {encoder!r}')
    return SAM(inp_size=inp_size, encoder_mode=dict(ENCODERS[encoder]),
               loss='iou', num_classes=num_classes, loss_weight=None)


__all__ = ['SAM', 'VIT_B_ENCODER', 'VIT_L_ENCODER', 'ENCODERS', 'build_cwsam']
