"""Tests for the Conv-LoRA + EVP encoder adaptation used by
``full_fixed2_lora_evp``."""

import torch

from models.sacm_x.configs import PRESETS
from models.sacm_x.specs import freeze_sam
from models.sam_network.build_sam import build_sam_vit_b
from models.sam_network.convlora import ConvLoRABranch
from models.sam_network.evp import EVPPromptGenerator
from models.sam_network.image_encoder import ImageEncoderViT


def test_convlora_branch_shape_and_gradients():
    branch = ConvLoRABranch(32, rank=4, num_experts=4)
    x = torch.randn(2, 8, 8, 32)
    out = branch(x)
    assert out.shape == (2, 8, 8, 96)
    # up is zero-initialised; a non-zero mean loss still produces a gradient.
    out.mean().backward()
    assert branch.up.weight.grad is not None
    assert branch.up.weight.grad.abs().sum() > 0
    assert branch.down.weight.grad is not None
    assert branch.last_moe_loss is not None


def test_convlora_keeps_low_rank_identity_path():
    branch = ConvLoRABranch(32, rank=4, num_experts=4)
    with torch.no_grad():
        for expert in branch.experts:
            expert[0].weight.zero_()
            expert[0].bias.zero_()
        branch.up.weight.fill_(1.0)
    out = branch(torch.randn(2, 8, 8, 32))
    # With every conv expert zeroed, the official A x identity path must
    # still produce a non-zero qkv residual.
    assert out.abs().sum() > 0


def test_evp_prompt_shapes_and_zero_init():
    generator = EVPPromptGenerator(
        embed_dim=64, depth=2, img_size=128, patch_size=16,
        scale_factor=32, freq_nums=0.25, zero_init=True)
    image = torch.randn(1, 3, 128, 128)
    tokens = torch.randn(1, 8, 8, 64)
    prompts = generator(image, tokens)
    assert len(prompts) == 2
    assert prompts[0].shape == (1, 64, 64)
    assert sum(float(p.abs().sum().detach()) for p in prompts) == 0.0
    sum(p.mean() for p in prompts).backward()
    assert generator.shared_mlp.weight.grad is not None
    assert generator.shared_mlp.weight.grad.abs().sum() > 0
    assert generator.patch_embed.weight.grad is not None


def test_encoder_convlora_evp_forward_and_gradients():
    encoder = ImageEncoderViT(
        img_size=128, patch_size=16, in_chans=3, embed_dim=64,
        depth=2, num_heads=2, out_chans=32, use_rel_pos=True,
        global_attn_indexes=(0, 1), window_size=14,
        use_adapter=False, use_internal_adapter=False,
        collect_adapter_features=False, use_convlora=True,
        convlora_rank=4, convlora_experts=4, use_evp=True,
        evp_scale_factor=32,
    )
    output, intermediate = encoder(
        torch.randn(1, 3, 128, 128), return_features=True)
    assert output.shape == (1, 32, 8, 8)
    assert len(intermediate) == 2
    assert encoder.last_moe_loss is not None
    output.mean().backward()
    params = dict(encoder.named_parameters())
    for name in ('blocks.0.attn.qkv_lora.up.weight',
                 'evp.shared_mlp.weight'):
        grad = params[name].grad
        assert grad is not None, name
        assert grad.abs().sum() > 0, name


def test_full_fixed2_lora_evp_preset_fields():
    preset = PRESETS['full_fixed2_lora_evp']
    # Hybrid: keep SACM external adapters + FusionV2; drop only the
    # duplicate block-internal adapter; add Conv-LoRA/EVP.
    assert preset['use_adapter'] is True
    assert preset['use_geo_i'] is False
    assert preset['use_geo_e'] is True
    assert preset['use_internal_adapter'] is False
    assert preset['collect_adapter_features'] is True
    assert preset['use_fusion_v2'] is True
    assert preset['use_convlora'] is True
    assert preset['use_evp'] is True
    assert preset['evp_zero_init'] is True
    assert preset['moe_loss_weight'] == 1.0
    assert preset['use_multi_depth'] is True
    assert preset['use_coarse_to_fine'] is True
    assert preset['use_detail_stem'] is True
    assert preset['high_res_output'] is True
    assert preset['high_res_size'] == 512
    assert preset['fixed_head_loss_weight'] == 0.5
    assert preset['fixed_supervision_head'] == 2
    assert preset['topology_loss'] == 'skeleton_recall'

    peft = PRESETS['full_fixed2_peft_lora_evp']
    assert peft['use_adapter'] is False
    assert peft['use_geo_e'] is False
    assert peft['collect_adapter_features'] is False
    assert peft['use_fusion_v2'] is False
    assert peft['use_convlora'] is True
    assert peft['use_evp'] is True


def test_hybrid_builder_keeps_external_adapters():
    model = build_sam_vit_b(
        checkpoint=None,
        use_adapter=True,
        use_geo_i=False,
        use_geo_e=True,
        use_internal_adapter=False,
        collect_adapter_features=True,
        use_coarse_to_fine=True,
        use_fusion_v2=True,
        use_multi_depth=True,
        use_convlora=True,
        convlora_rank=4,
        convlora_experts=8,
        use_evp=True,
        evp_zero_init=True,
        use_detail_stem=True,
        high_res_output=True,
        high_res_size=512,
        high_res_refine=False,
        use_structural_aux=True,
        init_stage2_from_stage1=False,
    )
    assert model.image_encoder.use_adapter is True
    assert len(model.image_encoder.external_adapters) > 0
    assert model.image_encoder.evp is not None
    freeze_sam(model)
    params = dict(model.named_parameters())
    assert params['image_encoder.blocks.0.attn.qkv_lora.up.weight'].requires_grad
    assert params['image_encoder.evp.shared_mlp.weight'].requires_grad
    assert any(
        p.requires_grad for n, p in params.items()
        if n.startswith('image_encoder.external_adapters'))


def test_peft_builder_and_freeze_protocol():
    model = build_sam_vit_b(
        checkpoint=None,
        use_adapter=False,
        use_geo_i=False,
        use_geo_e=False,
        use_internal_adapter=False,
        collect_adapter_features=False,
        use_coarse_to_fine=True,
        use_fusion_v2=False,
        use_multi_depth=True,
        use_convlora=True,
        convlora_rank=4,
        convlora_experts=8,
        use_evp=True,
        use_detail_stem=True,
        high_res_output=True,
        high_res_size=512,
        high_res_refine=False,
        use_structural_aux=True,
        init_stage2_from_stage1=False,
    )
    assert model.image_encoder.use_adapter is False
    assert len(model.image_encoder.external_adapters) == 0
    assert model.image_encoder.evp is not None
    freeze_sam(model)
    params = dict(model.named_parameters())
    assert not params['image_encoder.blocks.0.attn.qkv.weight'].requires_grad
    assert params['image_encoder.blocks.0.attn.qkv_lora.up.weight'].requires_grad
    assert params['image_encoder.evp.shared_mlp.weight'].requires_grad
    assert any(
        p.requires_grad for n, p in params.items()
        if n.startswith('mask_decoder'))
