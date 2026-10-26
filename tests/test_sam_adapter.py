"""Fast SAM-Adapter adapter tests (no ViT-L build)."""

import random
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.adapters.sam_adapter import SAMAdapter
from baseline_zoo.data import Sample
from baseline_zoo.runtime import build_parser

REPO_ROOT = Path(__file__).resolve().parents[1]

TINY_ENCODER_MODE = {
    'name': 'sam',
    'img_size': 64,
    'mlp_ratio': 4,
    'patch_size': 16,
    'qkv_bias': True,
    'use_rel_pos': True,
    'window_size': 2,
    'out_chans': 32,
    'scale_factor': 32,
    'input_type': 'fft',
    'freq_nums': 0.25,
    'prompt_type': 'highpass',
    'prompt_embed_dim': 32,
    'tuning_stage': 1234,
    'handcrafted_tune': True,
    'embedding_tune': True,
    'adaptor': 'adaptor',
    'embed_dim': 64,
    'depth': 2,
    'num_heads': 2,
    'global_attn_indexes': [1],
}


class _FixedRandom:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


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


def test_registry_and_metadata():
    assert 'sam_adapter' in available_adapters()
    adapter = load_adapter('sam_adapter')
    assert isinstance(adapter, SAMAdapter)
    assert adapter.id == 'sam_adapter'
    assert adapter.input_size == 1024
    assert adapter.official_epochs == 20
    assert adapter.official_weight_decay == 0.01
    assert adapter.train_batch_size == 2
    assert 'Retrained' in adapter.deviations[0]


def test_optimizer_uses_executable_official_weight_decay():
    adapter = SAMAdapter()
    model = torch.nn.Linear(2, 2)
    args = SimpleNamespace(
        lr=adapter.official_lr,
        weight_decay=adapter.official_weight_decay)
    optimizer = adapter.make_optimizer(model, args)
    assert optimizer.param_groups[0]['weight_decay'] == 0.01


def test_preprocess_shapes_and_flip(tmp_path):
    adapter = SAMAdapter()
    adapter.device = torch.device('cpu')
    adapter.input_size = 1024
    sample = _sample(tmp_path)

    image, mask = adapter._preprocess(
        sample, _FixedRandom(1.0), train=True)
    assert image.shape == (3, 1024, 1024)
    assert mask.shape == (1, 1024, 1024)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}

    flipped, flipped_mask = adapter._preprocess(
        sample, _FixedRandom(0.0), train=True)
    assert torch.allclose(flipped, torch.flip(image, dims=[2]))
    assert torch.allclose(flipped_mask, torch.flip(mask, dims=[2]))

    eval_image, _ = adapter._preprocess(
        sample, random.Random(0), train=False)
    assert torch.allclose(eval_image, image)


def test_loss_is_finite_and_differentiable():
    adapter = SAMAdapter()
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = torch.zeros(2, 1, 32, 32)
    target[..., :16] = 1
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target)
    loss = bce + adapter._iou_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_tiny_prompt_free_forward_and_freeze_policy():
    from baseline_zoo.models.sam_adapter import build_sam_adapter

    model = build_sam_adapter(
        inp_size=64, encoder_mode=TINY_ENCODER_MODE)
    adapter = SAMAdapter()
    adapter.configure_trainable(model)

    logits = model.infer(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 1, 64, 64)
    assert not model.image_encoder.patch_embed.proj.weight.requires_grad
    assert model.image_encoder.prompt_generator.shared_mlp.weight.requires_grad
    assert model.mask_decoder.iou_token.weight.requires_grad
    assert model.no_mask_embed.weight.requires_grad


def test_config_roundtrip(tmp_path):
    adapter = SAMAdapter()
    adapter.input_size = 1024
    adapter._sam_sha = 'abc'
    args = SimpleNamespace(sam_checkpoint=tmp_path / 'missing.pth')
    config = adapter.save_config(args)
    assert config['input_size'] == 1024
    assert config['sam_checkpoint_sha256'] == 'abc'

    restore = SimpleNamespace(input_size=0, sam_checkpoint=None)
    adapter.restore_config(restore, config)
    assert restore.input_size == 1024
    assert adapter.input_size == 1024


def test_parser_default_matches_official_weight_decay():
    parser = build_parser(load_adapter('sam_adapter'))
    assert parser.parse_args([]).weight_decay == 0.01
