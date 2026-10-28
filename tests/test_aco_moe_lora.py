"""Fast ACO-MoE-LoRA tests (real Hiera-L forward is slow/optional)."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.aco_moe_lora import ACOMoeLoRAAdapter, structure_loss
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.models.aco_moe_lora import (
    ACOConvLoRA,
    DynamicLoRALayer,
    build_hiera_trunk,
)


def _sample(tmp_path):
    image = tmp_path / 'TEST_0_image.png'
    mask = tmp_path / 'TEST_0_mask.png'
    Image.new('RGB', (64, 48), (120, 120, 120)).save(image)
    label = Image.new('L', (64, 48), 0)
    for x in range(8, 56):
        label.putpixel((x, 24), 255)
    label.save(mask)
    return Sample(image_path=image, mask_path=mask,
                  dataset='TEST', stem='TEST_0')


def test_registry_and_metadata():
    assert 'aco_moe_lora' in available_adapters()
    adapter = load_adapter('aco_moe_lora')
    assert isinstance(adapter, ACOMoeLoRAAdapter)
    assert adapter.id == 'aco_moe_lora'
    assert adapter.input_size == 352
    assert adapter.official_lr == pytest.approx(1e-3)
    assert adapter.train_batch_size == 4
    assert adapter.accumulation_steps == 3
    assert any('validation Dice' in item for item in adapter.deviations)
    assert any('batch 12' in item for item in adapter.deviations)


def test_structure_loss_is_finite_and_differentiable():
    logits = torch.zeros(1, 1, 32, 32, requires_grad=True)
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 12:20, 10:24] = 1
    loss = structure_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_pair_shapes_and_normalization(tmp_path):
    adapter = ACOMoeLoRAAdapter()
    adapter.input_size = 352
    image, mask = adapter._resize_pair(_sample(tmp_path), False, None)
    assert image.shape == (3, 352, 352)
    assert mask.shape == (1, 352, 352)
    assert torch.all(mask >= 0) and torch.all(mask <= 1)
    # Grayscale 120 -> mean over the three ImageNet-normalized channels.
    assert image.mean() == pytest.approx(0.0964, abs=0.05)


def test_dynamic_lora_and_aco_conv_forward():
    lora = DynamicLoRALayer(8, 8, max_rank=4)
    lora.set_hyperparams(rank=2, alpha=4.0)
    out = lora(torch.randn(2, 4, 8))
    assert out.shape == (2, 4, 8)
    layer = ACOConvLoRA(4, 4, num_experts=2)
    layer.set_hyperparams(rank=2, alpha=4.0, top_k=1)
    out, aux = layer(torch.randn(2, 4, 16, 16))
    assert out.shape == (2, 4, 16, 16)
    assert aux.dim() == 0


def test_config_roundtrip():
    adapter = ACOMoeLoRAAdapter()
    args = SimpleNamespace(
        input_size=352,
        aco_sam2_checkpoint='checkpoints/sam2_hiera_large.pt',
        aco_num_experts=8,
        aco_use_multi_start=False,
        aco_num_ants=20,
        aco_diversification_interval=50,
        aco_loss_weight=1.0,
        aco_random_init=False,
        epochs=800,
        data_root='data/sacm_3shot_seed0',
    )
    config = adapter.save_config(args)
    restored = ACOMoeLoRAAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert target.input_size == 352
    assert target.aco_num_experts == 8
    assert target.aco_num_ants == 20
    assert str(target.aco_sam2_checkpoint).endswith(
        'sam2_hiera_large.pt')


@pytest.mark.slow
def test_real_hiera_and_aco_forward_if_present():
    weights = 'checkpoints/sam2_hiera_large.pt'
    trunk = build_hiera_trunk(weights)
    features = trunk(torch.zeros(1, 3, 64, 64))
    assert [tuple(f.shape) for f in features] == [
        (1, 144, 16, 16), (1, 288, 8, 8), (1, 576, 4, 4), (1, 1152, 2, 2)]
    from baseline_zoo.models.aco_moe_lora import ACOMoELoRA
    model = ACOMoELoRA(
        checkpoint_path=weights, num_experts=4, use_multi_start=False,
        aco_optimizer_config={
            'rank_candidates': [4, 8],
            'alpha_candidates': [4.0, 8.0],
            'top_k_candidates': [1, 2],
            'num_ants': 2,
            'evaporation_rate': 0.3,
            'diversification_interval': 50,
            'mutation_prob': 0.2,
            'total_epochs': 2,
        })
    model.eval()
    with torch.no_grad():
        out, _ = model(torch.zeros(1, 3, 64, 64),
                       update_hyperparams=False)
    assert out.shape == (1, 1, 64, 64)
