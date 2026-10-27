"""Fast MedCAGD adapter tests (no full-size training)."""

import random
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.medcagd import (
    MedCAGDAdapter,
    resolve_medcagd_encoder_weights,
)
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.models.medcagd.losses import (
    MedCAGDLoss,
    generate_edge_targets,
)
from baseline_zoo.models.medcagd.model import load_encoder_state


def _sample(tmp_path, dataset='TEST'):
    image = tmp_path / f'{dataset}_0_image.png'
    mask = tmp_path / f'{dataset}_0_mask.png'
    Image.new('RGB', (40, 32), (120, 120, 120)).save(image)
    label = Image.new('L', (40, 32), 0)
    for x in range(8, 28):
        label.putpixel((x, 16), 255)
    label.save(mask)
    return Sample(image_path=image, mask_path=mask,
                  dataset=dataset, stem=f'{dataset}_0')


def test_registry_and_metadata():
    assert 'medcagd' in available_adapters()
    adapter = load_adapter('medcagd')
    assert isinstance(adapter, MedCAGDAdapter)
    assert adapter.id == 'medcagd'
    assert adapter.input_size == 256
    assert adapter.train_batch_size == 8
    assert 'MedCAGD' in adapter.official_source
    assert any('256' in item for item in adapter.deviations)


def test_edge_targets_and_loss():
    masks = torch.zeros(1, 1, 16, 16)
    masks[:, :, 4:12, 4:12] = 1
    edges = generate_edge_targets(masks, radius=1)
    assert edges.shape == masks.shape
    assert edges[:, :, 4:12, 4:12].min() == 0
    assert edges.sum() > 0
    loss = MedCAGDLoss()(
        torch.zeros(1, 1, 16, 16, requires_grad=True),
        [torch.zeros(1, 1, 16, 16, requires_grad=True)] * 3,
        [torch.zeros(1, 1, 16, 16, requires_grad=True)] * 3,
        masks,
    )
    assert torch.isfinite(loss)
    loss.backward()


def test_train_batch_and_tiling(tmp_path):
    adapter = MedCAGDAdapter()
    adapter.input_size = 16
    for dataset in ('DRIVE', 'CHASEDB1'):
        image, mask = adapter._train_pair(
            _sample(tmp_path, dataset), random.Random(0))
        assert image.shape == (3, 16, 16)  # default: patches everywhere
        assert mask.shape == (1, 16, 16)
        assert image.min() >= 0 and image.max() <= 1
        assert set(mask.unique().tolist()) <= {0.0, 1.0}
    adapter._input_policy = 'paper'
    adapter._full_size = 24
    image, mask = adapter._train_pair(
        _sample(tmp_path, 'CHASEDB1'), random.Random(0))
    assert image.shape == (3, 24, 24)
    assert mask.shape == (1, 24, 24)
    image, mask = adapter._train_pair(
        _sample(tmp_path, 'DRIVE'), random.Random(0))
    assert image.shape == (3, 16, 16)
    assert adapter._tile_positions(16, 16, 8) == [0]
    assert adapter._tile_positions(40, 16, 8) == [0, 8, 16, 24]
    assert adapter._tile_positions(41, 16, 8) == [0, 8, 16, 24, 25]


def test_local_encoder_state_formats(tmp_path):
    import safetensors.torch as st
    state = {'patch_embed.proj.weight': torch.ones(2, 2),
             'stages.0.blocks.0.norm1.weight': torch.ones(2),
             'head.weight': torch.zeros(1)}
    safe_path = tmp_path / 'weights.safetensors'
    st.save_file(state, str(safe_path))
    expected = {'patch_embed.proj.weight',
                'stages_0.blocks.0.norm1.weight',
                'head.weight'}
    loaded = load_encoder_state(safe_path)
    assert set(loaded) == expected
    torch_path = tmp_path / 'weights.pth'
    torch.save({'state_dict': state}, torch_path)
    loaded = load_encoder_state(torch_path)
    assert set(loaded) == expected
    prefixed_path = tmp_path / 'prefixed.pth'
    torch.save({f'module.{key}': value for key, value in state.items()},
               prefixed_path)
    loaded = load_encoder_state(prefixed_path)
    assert set(loaded) == expected


def test_missing_encoder_weights_fails_before_network(tmp_path, monkeypatch):
    import baseline_zoo.adapters.medcagd as medcagd_module
    monkeypatch.setattr(medcagd_module, 'REPO_ROOT', tmp_path)
    adapter = MedCAGDAdapter()
    missing = tmp_path / 'missing.safetensors'
    args = SimpleNamespace(
        stage='train', input_size=256, medcagd_pretrained=1,
        medcagd_encoder_weights=missing,
        medcagd_input_policy='patches', medcagd_full_size=224,
        medcagd_eval_stride=128, medcagd_eval_batch=8,
        medcagd_threshold=0.5, medcagd_edge_weight=1.0,
        medcagd_encoder='pvt_v2_b2', medcagd_use_stage0=0,
    )
    try:
        adapter.build_model(args)
    except SystemExit as error:
        message = str(error)
        assert 'missing MedCAGD encoder weights' in message
        assert 'huggingface.co/timm/pvt_v2_b2.in1k' in message
        assert 'No network access' in message
    else:
        raise AssertionError('expected SystemExit for missing weights')


def test_encoder_weight_named_fallback(tmp_path, monkeypatch):
    import baseline_zoo.adapters.medcagd as medcagd_module
    checkpoint_dir = tmp_path / 'checkpoints'
    checkpoint_dir.mkdir()
    fallback = checkpoint_dir / 'pvt_v2_b2.in1k.pth'
    fallback.write_bytes(b'placeholder')
    monkeypatch.setattr(medcagd_module, 'REPO_ROOT', tmp_path)
    requested = tmp_path / 'missing.safetensors'
    assert resolve_medcagd_encoder_weights(requested) == fallback


def test_config_roundtrip():
    adapter = MedCAGDAdapter()
    args = SimpleNamespace(
        input_size=256,
        medcagd_full_size=224,
        medcagd_input_policy='patches',
        medcagd_encoder='pvt_v2_b2',
        medcagd_encoder_weights='checkpoints/pvt_v2_b2.in1k.safetensors',
        medcagd_eval_stride=128,
        medcagd_eval_batch=8,
        medcagd_threshold=0.5,
        medcagd_edge_weight=1.0,
        medcagd_use_stage0=0,
    )
    config = adapter.save_config(args)
    restored = MedCAGDAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert restored.input_size == 256
    assert restored._full_size == 224
    assert restored._input_policy == 'patches'
    assert target.medcagd_full_size == 224
    assert target.medcagd_input_policy == 'patches'
    assert target.medcagd_encoder == 'pvt_v2_b2'
    assert str(target.medcagd_encoder_weights) == (
        'checkpoints/pvt_v2_b2.in1k.safetensors')
    assert target.medcagd_eval_stride == 128
    assert target.medcagd_threshold == 0.5
    assert target.medcagd_use_stage0 == 0


class _TinyMedCAGD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1)

    def forward(self, x):
        main = self.conv(x)
        return main, [main] * 3, [main] * 3


def test_mixed_size_train_batch_groups(tmp_path):
    adapter = MedCAGDAdapter()
    adapter.device = torch.device('cpu')
    adapter.input_size = 16
    adapter._full_size = 8
    adapter._input_policy = 'paper'
    samples = [_sample(tmp_path, 'DRIVE'), _sample(tmp_path, 'CHASEDB1')]
    groups = adapter.prepare_train_batch(samples, random.Random(0))
    shapes = sorted(
        (tuple(images.shape), count) for images, _, count in groups)
    assert shapes == [((1, 3, 8, 8), 1), ((1, 3, 16, 16), 1)]
    loss = adapter.train_step(_TinyMedCAGD(), groups)
    assert torch.isfinite(loss)
    loss.backward()


class _PositiveModel(torch.nn.Module):
    def forward(self, x):
        main = torch.full((x.shape[0], 1, *x.shape[-2:]), 2.0,
                          device=x.device)
        return main, [main] * 3, [main] * 3


def test_predict_full_image_and_patch_branch(tmp_path):
    adapter = MedCAGDAdapter()
    adapter.input_size = 16
    adapter._full_size = 8
    adapter._eval_stride = 8
    for policy in ('patches', 'paper'):
        adapter._input_policy = policy
        for dataset in ('CHASEDB1', 'DRIVE'):
            prepared = adapter.prepare_eval_sample(
                _sample(tmp_path, dataset), 'cpu')
            pred = adapter.predict_sample(
                _PositiveModel(), prepared, _sample(tmp_path, dataset))
            assert pred.shape == (32, 40)  # original mask size
            assert pred.dtype == bool
            assert pred.all()


@pytest.mark.slow
def test_local_imagenet_encoder_load(tmp_path):
    pytest.importorskip('timm')
    import safetensors.torch as st
    from timm import create_model

    from baseline_zoo.models.medcagd import build_medcagd

    # Hugging Face publishes the classification state dict (stages.0.*
    # plus head.*), not the features-only wrapper state dict.
    source = create_model('pvt_v2_b2', pretrained=False)
    path = tmp_path / 'pvt_v2_b2.in1k.safetensors'
    st.save_file(source.state_dict(), str(path))
    model = build_medcagd(
        input_size=64, num_classes=1, pretrained=True,
        pretrained_weights=path)
    assert model.encoder is not None


@pytest.mark.slow
def test_model_forward():
    pytest.importorskip('timm')
    from baseline_zoo.models.medcagd import build_medcagd

    model = build_medcagd(input_size=64, num_classes=1, pretrained=False)
    output = model(torch.zeros(1, 3, 64, 64))
    main, auxiliaries, edges = output
    assert main.shape == (1, 1, 64, 64)
    assert len(auxiliaries) == 3 and len(edges) == 3
