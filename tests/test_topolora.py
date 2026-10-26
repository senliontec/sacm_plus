"""Fast TopoLoRA-SAM adapter tests (no ViT-B build)."""

import random
from functools import partial
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.adapters.topolora import AUG, TopoLoRAAdapter
from baseline_zoo.data import Sample
from baseline_zoo.models.topolora import build_transforms
from baseline_zoo.models.topolora.adapters import ConvAdapter
from baseline_zoo.models.topolora.lora import inject_lora
from baseline_zoo.models.topolora.model import TopoLoRASAM
from baseline_zoo.models.topolora.segment_anything.modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)


def _sample(tmp_path):
    image = tmp_path / 'TEST_0_image.png'
    mask = tmp_path / 'TEST_0_mask.png'
    Image.new('RGB', (32, 24), (120, 120, 120)).save(image)
    label = Image.new('L', (32, 24), 0)
    for x in range(16):
        for y in range(24):
            label.putpixel((x, y), 255)
    label.save(mask)
    return Sample(image_path=image, mask_path=mask,
                  dataset='TEST', stem='TEST_0')


def _tiny_model():
    encoder = ImageEncoderViT(
        depth=2, embed_dim=64, img_size=64, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_heads=2,
        patch_size=16, qkv_bias=True, use_rel_pos=True,
        global_attn_indexes=[1], window_size=2, out_chans=32)
    prompt_encoder = PromptEncoder(
        embed_dim=32, image_embedding_size=(4, 4),
        input_image_size=(64, 64), mask_in_chans=16)
    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2, embedding_dim=32, mlp_dim=64, num_heads=2),
        transformer_dim=32, iou_head_depth=2, iou_head_hidden_dim=32)
    sam = Sam(encoder, prompt_encoder, mask_decoder)
    for parameter in sam.parameters():
        parameter.requires_grad = False
    inject_lora(sam.image_encoder, ('mlp', 'fc1', 'fc2'),
                r=4, alpha=8, dropout=0.0)
    return TopoLoRASAM(
        sam, ConvAdapter(channels=32, hidden_dim=32, kernel_size=3))


def test_registry_and_metadata():
    assert 'topolora' in available_adapters()
    adapter = load_adapter('topolora')
    assert isinstance(adapter, TopoLoRAAdapter)
    assert adapter.id == 'topolora'
    assert adapter.input_size == 384
    assert adapter.train_batch_size == 1
    assert adapter.accumulation_steps == 4
    assert 'ViT-B' in adapter.deviations[0]


def test_preprocess_shapes(tmp_path):
    adapter = TopoLoRAAdapter()
    adapter.input_size = 384
    adapter._train_tf = build_transforms(
        384, train=True, sar=False, aug=AUG)
    adapter._eval_tf = build_transforms(384, train=False, sar=False)
    sample = _sample(tmp_path)

    image, mask = adapter._preprocess(
        sample, random.Random(0), train=True)
    assert image.shape == (3, 384, 384)
    assert mask.shape == (384, 384)
    assert image.dtype == torch.float32
    assert set(mask.unique().tolist()) <= {0, 1}
    assert torch.isfinite(image).all()

    eval_image, eval_mask = adapter._preprocess(
        sample, random.Random(0), train=False)
    assert eval_image.shape == (3, 384, 384)
    assert eval_mask.shape == (384, 384)


def test_loss_is_finite_and_differentiable():
    from baseline_zoo.models.topolora.topo_losses import cldice_loss, dice_loss

    logits = torch.randn(1, 1, 32, 32, requires_grad=True)
    target = torch.zeros(1, 1, 32, 32)
    target[..., :16] = 1
    prob = torch.sigmoid(logits)
    loss = (torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target)
        + dice_loss(prob, target)
        + 0.5 * cldice_loss(prob, target))
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_tiny_forward_and_trainable_policy():
    adapter = TopoLoRAAdapter()
    model = _tiny_model()
    adapter.configure_trainable(model)

    assert not model.sam.image_encoder.patch_embed.proj.weight.requires_grad
    assert model.sam.mask_decoder.iou_token.weight.requires_grad
    assert model.sam.image_encoder.blocks[0].mlp.lin1.lora_A.requires_grad
    assert model.adapter.dw.weight.requires_grad

    logits = model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 1, 64, 64)
    assert torch.isfinite(logits).all()


def test_config_roundtrip(tmp_path):
    adapter = TopoLoRAAdapter()
    adapter.input_size = 384
    adapter._sam_sha = 'abc'
    args = SimpleNamespace(
        sam_checkpoint=tmp_path / 'missing.pth', sam_type='vit_b',
        lora_r=8, lora_alpha=16.0, lora_dropout=0.0,
        adapter_hidden=256, adapter_kernel=3)
    config = adapter.save_config(args)
    assert config['input_size'] == 384
    assert config['sam_checkpoint_sha256'] == 'abc'

    restore = SimpleNamespace(sam_checkpoint=None, input_size=0)
    adapter.restore_config(restore, config)
    assert restore.input_size == 384
    assert restore.lora_r == 8
    assert restore.sam_type == 'vit_b'
    assert adapter.input_size == 384
