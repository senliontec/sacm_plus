"""SAM2-UNet adapter tests.

Fast tests do not build the 216M-parameter Hiera-L model; the two slow tests
are skipped unless ``pytest --run-slow`` is passed.
"""

import importlib
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

from baseline_zoo.adapters.sam2_unet import SAM2UNetAdapter, structure_loss
from baseline_zoo.data import Sample

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / 'src'


def _write_pair(tmp_path, width=13, height=17):
    image_path = tmp_path / 'TEST_0_image.png'
    mask_path = tmp_path / 'TEST_0_mask.png'
    Image.new('RGB', (width, height), (120, 120, 120)).save(image_path)
    mask = Image.new('L', (width, height), 0)
    for x in range(width // 2):
        for y in range(height):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)
    return Sample(image_path=image_path, mask_path=mask_path,
                  dataset='TEST', stem='TEST_0')


def test_adapter_metadata():
    adapter = SAM2UNetAdapter()
    assert adapter.id == 'sam2_unet'
    assert adapter.display_name == 'SAM2-UNet'
    assert adapter.input_size == 352
    assert adapter.train_batch_size == 4
    assert adapter.selection == 'dice'
    assert 'Appendix' in adapter.deviations[0]


def test_structure_loss_is_finite_and_differentiable():
    pred = torch.full((2, 1, 16, 16), 20.0, requires_grad=True)
    mask = torch.ones(2, 1, 16, 16)
    loss = structure_loss(pred, mask)
    assert torch.isfinite(loss)
    assert loss.item() < 0.01
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_prepare_train_batch_shapes_and_soft_mask(tmp_path):
    adapter = SAM2UNetAdapter()
    sample = _write_pair(tmp_path)
    images, masks = adapter.prepare_train_batch(
        [sample], random.Random(0))
    assert images.shape == (1, 3, 352, 352)
    assert masks.shape == (1, 1, 352, 352)
    assert torch.isfinite(images).all()
    assert float(masks.min()) >= 0.0
    assert float(masks.max()) <= 1.0


def test_predict_sample_restores_gt_resolution(tmp_path):
    adapter = SAM2UNetAdapter()
    sample = _write_pair(tmp_path, width=13, height=17)
    logits = torch.full((1, 1, 32, 32), -10.0)
    logits[..., :16] = 10.0

    class Dummy(torch.nn.Module):
        def forward(self, x):
            return logits, logits, logits

    pred = adapter.predict_sample(
        Dummy(), torch.zeros(1, 3, 352, 352), sample)
    assert pred.dtype == bool
    assert pred.shape == (17, 13)
    assert pred[:, :6].all()
    assert not pred[:, 7:].any()


def test_training_requires_checkpoint_unless_random_init(
        tmp_path, monkeypatch):
    import baseline_zoo.models.sam2_unet as package

    calls = []

    class FakeModel:
        def __init__(self, checkpoint_path=None, device='cpu'):
            calls.append((checkpoint_path, device))

    monkeypatch.setattr(package, 'SAM2UNet', FakeModel)
    adapter = SAM2UNetAdapter()
    args = SimpleNamespace(
        input_size=352, stage='train',
        sam2_checkpoint=tmp_path / 'missing.pt',
        random_init=False, allow_random_init=False)
    with pytest.raises(SystemExit):
        adapter.build_model(args)

    args.random_init = True
    adapter.build_model(args)
    assert calls[-1] == (None, 'cpu')

    # --random-init must also win when the configured checkpoint exists.
    existing = tmp_path / 'existing.pt'
    existing.write_bytes(b'not-a-real-checkpoint')
    args.sam2_checkpoint = existing
    adapter.build_model(args)
    assert calls[-1] == (None, 'cpu')


def test_save_and_restore_config(tmp_path):
    adapter = SAM2UNetAdapter()
    adapter.input_size = 352
    adapter._sam2_sha = 'abc'
    args = SimpleNamespace(
        sam2_checkpoint=tmp_path / 'sam2.pt', random_init=False)
    config = adapter.save_config(args)
    assert config['input_size'] == 352
    assert config['sam2_checkpoint_sha256'] == 'abc'
    assert config['random_init'] is False

    restore = SimpleNamespace(
        input_size=0, sam2_checkpoint=tmp_path / 'missing.pt',
        allow_random_init=False)
    with pytest.raises(SystemExit):
        adapter.restore_config(
            restore, {'input_size': 352, 'random_init': True})
    restore.allow_random_init = True
    adapter.restore_config(
        restore, {'input_size': 352, 'random_init': True})
    assert restore.input_size == 352


def test_config_targets_are_package_internal():
    from baseline_zoo.models.sam2_unet.sam2.build_sam import CONFIG_DIR

    config = OmegaConf.load(CONFIG_DIR / 'sam2_hiera_l.yaml')
    targets = []

    def walk(node):
        if isinstance(node, (dict, DictConfig)):
            for key, value in node.items():
                if key == '_target_':
                    targets.append(str(value))
                walk(value)
        elif isinstance(node, (list, ListConfig)):
            for value in node:
                walk(value)

    walk(config)
    assert targets
    for target in targets:
        assert target.startswith('baseline_zoo.models.sam2_unet.')
        importlib.import_module(target.rsplit('.', 1)[0])


def test_load_checkpoint_payload_validation(tmp_path):
    from baseline_zoo.models.sam2_unet.sam2.build_sam import _load_checkpoint

    model = torch.nn.Linear(2, 2)
    good = tmp_path / 'good.pt'
    torch.save({'model': model.state_dict()}, good)
    _load_checkpoint(model, good)

    bad = tmp_path / 'bad.pt'
    torch.save({'not_model': {}}, bad)
    with pytest.raises(RuntimeError):
        _load_checkpoint(model, bad)

    mismatch = tmp_path / 'mismatch.pt'
    torch.save({'model': {'bad': torch.zeros(1)}}, mismatch)
    with pytest.raises(RuntimeError):
        _load_checkpoint(model, mismatch)


def test_adapter_import_is_lazy(tmp_path):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC)
    code = (
        "import sys; import baseline_zoo.adapters.sam2_unet; "
        "print('hydra' in sys.modules, "
        "'baseline_zoo.models.sam2_unet' in sys.modules)")
    output = subprocess.check_output(
        [sys.executable, '-c', code], cwd=tmp_path, env=env, text=True)
    assert output.strip() == 'False False'


@pytest.mark.slow
@pytest.mark.skipif("not config.getoption('--run-slow')")
def test_random_init_forward_shapes():
    from baseline_zoo.models.sam2_unet import SAM2UNet

    model = SAM2UNet(None, device='cpu').eval()
    with torch.no_grad():
        out, out1, out2 = model(torch.zeros(1, 3, 352, 352))
    assert out.shape == out1.shape == out2.shape == (1, 1, 352, 352)
    assert torch.isfinite(out).all()


@pytest.mark.slow
@pytest.mark.skipif("not config.getoption('--run-slow')")
def test_freeze_policy_keeps_prompt_learn_trainable():
    from baseline_zoo.models.sam2_unet import SAM2UNet

    adapter = SAM2UNetAdapter()
    model = SAM2UNet(None, device='cpu')
    adapter.configure_trainable(model)
    block_weight = model.encoder.blocks[0].block.attn.qkv.weight
    assert not block_weight.requires_grad
    prompts = [
        parameter for name, parameter in model.encoder.named_parameters()
        if 'prompt_learn' in name]
    assert prompts and all(p.requires_grad for p in prompts)
    assert any(p.requires_grad for p in model.head.parameters())
