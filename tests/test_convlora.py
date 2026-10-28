"""Fast Conv-LoRA adapter tests (no full SAM run by default)."""

import random
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.convlora import ConvLoRAAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.models.convlora.layers import ConvLoRALinear
from baseline_zoo.models.convlora.losses import StructureLoss


def _sample(tmp_path):
    image = tmp_path / 'TEST_0_image.png'
    mask = tmp_path / 'TEST_0_mask.png'
    Image.new('RGB', (40, 32), (120, 120, 120)).save(image)
    label = Image.new('L', (40, 32), 0)
    for x in range(8, 28):
        label.putpixel((x, 16), 255)
    label.save(mask)
    return Sample(image_path=image, mask_path=mask,
                  dataset='TEST', stem='TEST_0')


def test_registry_and_metadata():
    assert 'convlora' in available_adapters()
    adapter = load_adapter('convlora')
    assert isinstance(adapter, ConvLoRAAdapter)
    assert adapter.id == 'convlora'
    assert adapter.input_size == 1024
    assert adapter.accumulation_steps == 4
    assert any('MoE' in item for item in adapter.deviations)


def test_conv_lora_layer_shapes_and_moe():
    layer = ConvLoRALinear(
        16, 48, r=3, lora_alpha=8, conv_lora_expert_num=4)
    x = torch.randn(2, 8, 8, 16, requires_grad=True)
    out, moe = layer(x)
    assert out.shape == (2, 8, 8, 48)
    assert moe is not None and torch.isfinite(moe)
    assert not layer.weight.requires_grad
    (out.mean() + moe).backward()
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None


def test_imagenet_normalization_matches_official_pipeline():
    from baseline_zoo.models.convlora.model import ConvLoRASAM

    captured = {}

    class _Encoder(torch.nn.Module):
        img_size = 1024

        def forward(self, x):
            captured['input'] = x
            return torch.zeros(1, 256, 64, 64)

    class _PromptEncoder(torch.nn.Module):
        def forward(self, points=None, boxes=None, masks=None):
            return (torch.zeros(1, 0, 256),
                    torch.zeros(1, 256, 64, 64))

        def get_dense_pe(self):
            return torch.zeros(1, 256, 64, 64)

    class _MaskDecoder(torch.nn.Module):
        def forward(self, **kwargs):
            return torch.zeros(1, 1, 256, 256), None

    sam = SimpleNamespace(
        image_encoder=_Encoder(),
        prompt_encoder=_PromptEncoder(),
        mask_decoder=_MaskDecoder(),
        pixel_mean=torch.zeros(3, 1, 1),
        postprocess_masks=lambda low, input_size, original_size: torch.zeros(
            1, 1, 1024, 1024),
    )
    model = ConvLoRASAM(sam)
    model(torch.full((1, 3, 8, 8), 255.0))

    expected = (
        1.0 - torch.tensor(ConvLoRASAM.IMAGENET_MEAN).view(1, 3, 1, 1)
    ) / torch.tensor(ConvLoRASAM.IMAGENET_STD).view(1, 3, 1, 1)
    assert torch.allclose(
        captured['input'][0, :, 0, 0], expected[0, :, 0, 0], atol=1e-6)


def test_structure_loss_is_finite_and_differentiable():
    logits = torch.zeros(2, 1, 64, 64, requires_grad=True)
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 20:30, 20:40] = 1
    loss = StructureLoss()(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_pair_shapes(tmp_path):
    adapter = ConvLoRAAdapter()
    image, mask = adapter._pair(_sample(tmp_path), random.Random(0))
    assert image.shape == (3, 1024, 1024)
    assert mask.shape == (1, 1024, 1024)
    assert image.min() >= 0 and image.max() <= 255
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_predict_resizes_logits_to_gt():
    adapter = ConvLoRAAdapter()
    adapter.input_size = 1024

    class _Fake(torch.nn.Module):
        def forward(self, image):
            logits = torch.full((1, 1, 1024, 1024), -10.0)
            logits[:, :, 500:524, :] = 10.0
            return logits, torch.zeros((), device=image.device)

    prepared = (torch.zeros(1, 3, 1024, 1024), (32, 48))
    pred = adapter.predict_sample(
        _Fake(), prepared,
        Sample(image_path='x', mask_path='y', dataset='T', stem='s'))
    assert pred.shape == (48, 32)
    assert pred.dtype == bool
    assert pred.any()


def test_config_roundtrip():
    adapter = ConvLoRAAdapter()
    args = SimpleNamespace(
        input_size=1024,
        convlora_model_type='vit_h',
        convlora_rank=3,
        convlora_alpha=8,
        convlora_experts=8,
        convlora_moe_weight=1.0,
        convlora_checkpoint='checkpoints/sam_vit_h_4b8939.pth',
    )
    config = adapter.save_config(args)
    restored = ConvLoRAAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert restored.input_size == 1024
    assert target.convlora_model_type == 'vit_h'
    assert target.convlora_rank == 3
    assert target.convlora_alpha == 8
    assert target.convlora_experts == 8
    assert target.convlora_moe_weight == 1.0


@pytest.mark.slow
def test_real_sam_build_and_forward():
    from pathlib import Path

    from baseline_zoo.models.convlora import build_convlora
    checkpoint = Path('checkpoints/sam_vit_b_01ec64.pth')
    if not checkpoint.is_file():
        pytest.skip('missing SAM ViT-B checkpoint')
    model = build_convlora(checkpoint, model_type='vit_b')
    model.eval()
    with torch.no_grad():
        logits, moe = model(torch.zeros(1, 3, 1024, 1024))
    assert logits.shape == (1, 1, 1024, 1024)
    assert torch.isfinite(moe)
