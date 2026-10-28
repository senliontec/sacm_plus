"""A2SAM fidelity, component and adapter tests."""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.a2sam import A2SAMAdapter, _sample_seed
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.models.a2sam import (
    IOU,
    CoarseUNet,
    PersistencePrior,
    PromptFusion,
    ThicknessAwareLoss,
    build_a2sam,
    build_morse_complex,
    compute_persistence,
    contrastive_scores,
    dice_optimal_xi,
    f_ms,
    sam_cps_candidates,
    select_freeze_masks,
    uncertainty_from_samples,
    zero_cps_gradients,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_PROMPT = (REPO_ROOT.parent / '3rd/A2SAM/models/mmseg/models/a2sam/'
                   'prompt_discriminator.py')


def _sample(tmp_path, width=64, height=48):
    image = tmp_path / 'TEST_0_image.png'
    mask = tmp_path / 'TEST_0_mask.png'
    Image.new('RGB', (width, height), (120, 120, 120)).save(image)
    label = Image.new('L', (width, height), 0)
    for x in range(8, width - 8):
        label.putpixel((x, height // 2), 255)
    label.save(mask)
    return Sample(image_path=image, mask_path=mask,
                  dataset='TEST', stem='TEST_0')


def _args(**overrides):
    values = dict(
        input_size=1024,
        a2sam_sam_checkpoint=Path('checkpoints/sam_vit_b_01ec64.pth'),
        a2sam_unet_epochs=100,
        a2sam_prior_epochs=100,
        a2sam_unet_lr=1e-3,
        a2sam_unet_pos_weight=10.0,
        a2sam_prior_lr=1e-3,
        a2sam_prior_samples=5,
        a2sam_prior_min_sigma=0.05,
        a2sam_keep_ratio=0.035,
        a2sam_cps_alpha=0.4,
        a2sam_cps_mode='official',
        a2sam_ta_mode='paper',
        a2sam_fusion_inputs='release',
        a2sam_prompt_detach=False,
        a2sam_topology_backend='cripser',
        a2sam_coarse_threshold=0.5,
        a2sam_min_persistence=1e-3,
        a2sam_max_branches=256,
        a2sam_prior_grid_size=32,
        a2sam_optimizer='adamw',
        a2sam_lr_min=1e-7,
        a2sam_release_parity=False,
        a2sam_random_init=False,
        a2sam_no_augment=False,
        epochs=800, lr=2e-4, weight_decay=0.01, grad_clip=0.0,
        accum_steps=1, seed=42, data_root='data/sacm_3shot',
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_registry_and_metadata():
    assert 'a2sam' in available_adapters()
    adapter = load_adapter('a2sam')
    assert isinstance(adapter, A2SAMAdapter)
    assert adapter.id == 'a2sam'
    assert adapter.input_size == 1024
    assert adapter.official_lr == pytest.approx(2e-4)
    assert adapter.official_weight_decay == pytest.approx(0.01)
    assert any('paper-level' in item for item in adapter.deviations)


def test_prompt_fusion_matches_official():
    if not OFFICIAL_PROMPT.is_file():
        pytest.skip(f'missing official reference: {OFFICIAL_PROMPT}')
    spec = importlib.util.spec_from_file_location(
        'official_a2sam_prompt', OFFICIAL_PROMPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.PromptGenerator(
        in_channels=1, out_conv_channels=1).eval()
    ours = PromptFusion().eval()
    ours.load_state_dict(reference.state_dict())
    uncertainty = torch.rand(1, 32, 32)
    coarse = torch.randn(1, 32, 32)
    with torch.no_grad():
        expected = reference(uncertainty, coarse)
        actual = ours(uncertainty.unsqueeze(0), coarse.unsqueeze(0))
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-6)
    assert sum(p.numel() for p in ours.parameters()) == 22482


def test_prompt_fusion_paper_mode_and_detach():
    fusion = PromptFusion()
    uncertainty = torch.rand(2, 1, 16, 16, requires_grad=True)
    coarse = torch.randn(2, 1, 16, 16)
    output = fusion(uncertainty, coarse, mode='paper')
    assert output.shape == uncertainty.shape
    output.mean().backward()
    assert uncertainty.grad is not None
    detached = fusion(uncertainty.detach(), coarse, detach=True)
    assert not detached.requires_grad


def test_sam_checkpoint_allowlist(tmp_path):
    source = build_a2sam(image_size=256, prompt_size=64)
    target = build_a2sam(image_size=256, prompt_size=64)
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.add_(0.01)
    captured = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
        if any(excluded in name for excluded in (
            'mask_tokens', 'output_hypernetworks_mlps',
            'iou_prediction_head'))
    }
    checkpoint = tmp_path / 'sam_fixture.pth'
    torch.save(source.state_dict(), checkpoint)
    info = target.load_sam_checkpoint(checkpoint)
    assert info['loaded'] == {
        'image_encoder': 177, 'prompt_encoder': 17,
        'mask_decoder': 89,
    }
    random_params = sum(
        item['parameters'] for item in info['random_init'].values())
    random_keys = sum(
        len(item['keys']) for item in info['random_init'].values())
    assert random_keys == 31
    assert random_params == 692_868
    target_state = target.state_dict()
    source_state = source.state_dict()
    for name, before in captured.items():
        assert torch.equal(target_state[name], before)
    for name, value in source_state.items():
        if name.startswith(('image_encoder.', 'prompt_encoder.')) or (
                name.startswith('mask_decoder.')
                and not any(excluded in name for excluded in (
                    'mask_tokens', 'output_hypernetworks_mlps',
                    'iou_prediction_head'))):
            assert torch.equal(target_state[name], value)


def test_cps_official_freeze_semantics():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = torch.nn.Module()
            self.image_encoder.weight = torch.nn.Parameter(torch.randn(4, 5))
            self.image_encoder.norm = torch.nn.Parameter(torch.ones(5))
            self.prompt_encoder = torch.nn.Module()
            self.prompt_encoder.weight = torch.nn.Parameter(torch.randn(4, 5))
            self.mask_decoder = torch.nn.Module()
            self.mask_decoder.weight = torch.nn.Parameter(torch.randn(4, 5))
            self.mask_decoder.iou_prediction_head = torch.nn.Module()
            self.mask_decoder.iou_prediction_head.weight = (
                torch.nn.Parameter(torch.randn(4)))

    model = Toy()
    candidates = sam_cps_candidates(model)
    assert 'mask_decoder.iou_prediction_head.weight' not in candidates
    grads = {name: torch.randn_like(parameter)
             for name, parameter in candidates.items()}
    for name, parameter in candidates.items():
        parameter.grad = grads[name].clone()
    model.mask_decoder.iou_prediction_head.weight.grad = torch.ones(4)
    scores = contrastive_scores(grads, candidates, mode='official')
    masks = select_freeze_masks(
        scores, candidates, keep_ratio=0.5, mode='official')
    assert int(masks['image_encoder.norm'].sum()) == 5
    assert model.mask_decoder.iou_prediction_head.weight.requires_grad
    stats = zero_cps_gradients(model, masks)
    assert stats['trainable_elements'] + stats['frozen_elements'] == (
        stats['total_elements'])
    assert torch.count_nonzero(
        model.image_encoder.norm.grad).item() == 0
    assert torch.count_nonzero(
        model.mask_decoder.iou_prediction_head.weight.grad).item() == 4
    assert model.image_encoder.norm.requires_grad


def test_cps_paper_formula_and_count():
    parameter = torch.nn.Parameter(torch.randn(100))
    grads = torch.randn_like(parameter)
    params = {'mask_decoder.weight': parameter}
    scores = contrastive_scores(
        {'mask_decoder.weight': grads}, params, alpha=0.4, mode='paper')
    expected = (0.4 * (grads.pow(2) / (parameter.abs() + 1))
                + 0.6 * grads.pow(2))
    assert torch.allclose(scores['mask_decoder.weight'], expected)
    masks = select_freeze_masks(
        scores, params, keep_ratio=0.035, mode='paper')
    # paper Eq. (21): floor(100 * 0.035) = 3 trainable elements.
    assert int(masks['mask_decoder.weight'].sum()) == 97


def test_losses_modes_finite_and_empty():
    logits = torch.randn(1, 1, 64, 64, requires_grad=True)
    target = torch.zeros(1, 1, 64, 64)
    target[:, :, 30, 10:54] = 1
    for mode in ('paper', 'release'):
        loss = IOU()(logits, target) + 0.7 * ThicknessAwareLoss(mode=mode)(
            logits, target)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)
    empty = torch.zeros_like(target)
    assert ThicknessAwareLoss()(logits, empty).item() == 0.0
    assert torch.isfinite(IOU()(logits, empty))


def test_persistence_and_fms_synthetic():
    prob = np.zeros((33, 33), np.float32)
    for row in range(33):
        for col in range(33):
            prob[row, col] = max(
                0.9 - 0.02 * abs(row - 8) - 0.03 * abs(col - 16),
                0.7 - 0.02 * abs(row - 24) - 0.03 * abs(col - 16))
    prob[16, 16] = 0.1
    bars = compute_persistence(prob)
    assert len(bars) >= 2
    assert bars.essential.sum() == 1
    join_bars = compute_persistence(prob, backend='join_tree')
    auto_bars = compute_persistence(prob, backend='auto')
    np.testing.assert_allclose(
        np.sort(bars.persistence), np.sort(join_bars.persistence), atol=1e-6)
    np.testing.assert_allclose(
        np.sort(bars.persistence), np.sort(auto_bars.persistence), atol=1e-6)
    if importlib.util.find_spec('gudhi') is not None:
        # Gudhi builds a cubical complex, so diagonal cofaces are connected
        # through shared vertices; cripser/join-tree use 4-connected pixels.
        # The bar multiset can therefore differ on this asymmetric map.
        gudhi_bars = compute_persistence(prob, backend='gudhi')
        assert len(gudhi_bars) >= 2
        assert gudhi_bars.essential.sum() == 1
        assert np.isfinite(gudhi_bars.persistence).all()
    complex_ = build_morse_complex(prob, min_persistence=1e-4)
    assert complex_.n_branches >= 2
    low = f_ms(complex_, 0.0)
    high = f_ms(complex_, 0.5)
    assert low.sum() >= high.sum()
    assert low.any()
    assert np.isfinite(dice_optimal_xi(complex_, prob > 0.5))
    samples = np.stack([f_ms(complex_, 0.0), f_ms(complex_, 0.5)])
    uncertainty = uncertainty_from_samples(samples)
    assert uncertainty.shape == prob.shape
    assert uncertainty.min() >= 0.0
    assert uncertainty.max() <= 0.25
    empty = build_morse_complex(np.zeros_like(prob))
    assert empty.n_branches == 0
    assert not f_ms(empty, 0.0).any()


def test_unet_prior_model_parameter_budgets_and_forward():
    unet = CoarseUNet()
    prior = PersistencePrior()
    fusion = PromptFusion()
    assert 4.0e6 < sum(p.numel() for p in unet.parameters()) < 4.6e6
    assert 2.5e6 < sum(p.numel() for p in prior.parameters()) < 3.0e6
    assert 20_000 < sum(p.numel() for p in fusion.parameters()) < 25_000
    model = build_a2sam(image_size=256, prompt_size=64).eval()
    image = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        logits, probability, coarse = model.coarse(image)
        mu, log_var = model.prior_stats(image, probability)
        generator = torch.Generator().manual_seed(0)
        xi_a = model.prior.sample_xi(mu, log_var, k=5, generator=generator)
        generator = torch.Generator().manual_seed(0)
        xi_b = model.prior.sample_xi(mu, log_var, k=5, generator=generator)
        prompt = model.make_prompt(torch.zeros_like(probability), logits)
        masks, low_res, iou = model(image, prompt, (200, 180))
    assert logits.shape == probability.shape == coarse.shape
    assert mu.shape == log_var.shape == (1,)
    assert torch.all(mu >= 0.0) and torch.all(mu <= 1.0)
    assert torch.all(log_var >= model.prior.log_var_min - 1e-6)
    assert xi_a.shape == (5, 1)
    assert torch.equal(xi_a, xi_b)
    assert prompt.shape == (1, 1, 64, 64)
    assert masks.shape == (1, 1, 200, 180)
    assert low_res.shape[0] == 1
    assert iou.shape[0] == 1


def test_adapter_stage_config_and_data_pipeline(tmp_path):
    adapter = A2SAMAdapter()
    adapter.args = _args()
    adapter.device = torch.device('cpu')
    assert adapter._stage_for_epoch(1) == 'unet'
    assert adapter._stage_for_epoch(100) == 'unet'
    assert adapter._stage_for_epoch(101) == 'prior'
    assert adapter._stage_for_epoch(200) == 'prior'
    assert adapter._stage_for_epoch(201) == 'a2sam'
    adapter._stage = 'unet'
    assert not adapter.checkpoint_eligible(1)
    adapter._stage = 'a2sam'
    assert not adapter.checkpoint_eligible(201)
    assert adapter.checkpoint_eligible(202)
    adapter.args = _args(
        a2sam_fusion_inputs='paper', a2sam_ta_mode='release',
        a2sam_release_parity=True)
    assert adapter._fusion_mode() == 'release'
    assert adapter._ta_mode() == 'release'
    assert adapter._optimizer_name() == 'adam'
    assert adapter._prompt_detach()
    sample = _sample(tmp_path)
    batch = adapter._prepare_unet_batch([sample], random.Random(0))
    assert batch['image'].shape == (1, 3, 1024, 1024)
    assert batch['target'].shape == (1, 1, 1024, 1024)
    adapter._unet_cache[sample.stem] = {
        'probability': torch.zeros(1, 8, 8),
        'xi_star': 0.1,
    }
    prior_batch = adapter._prepare_prior_batch([sample], random.Random(0))
    assert prior_batch['probability'].shape == (1, 1, 8, 8)
    assert prior_batch['target'].shape == (1,)
    adapter._stage3_cache[sample.stem] = {
        'logits': torch.zeros(1, 8, 8),
        'y_c': torch.zeros(1, 8, 8),
        'uncertainty': torch.zeros(1, 8, 8),
        'gt': torch.zeros(1, 48, 64),
    }
    a2sam_batch = adapter._prepare_a2sam_batch([sample], random.Random(0))
    assert a2sam_batch['image'].shape == (1, 3, 1024, 1024)
    assert a2sam_batch['uncertainty'].shape == (1, 1, 8, 8)
    assert a2sam_batch['coarse'].shape == (1, 1, 8, 8)
    config = adapter.save_config(adapter.args)
    restored = A2SAMAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert target.a2sam_unet_epochs == 100
    assert target.a2sam_prior_epochs == 100
    assert target.a2sam_topology_backend == 'cripser'
    assert target.a2sam_cps_mode == 'official'
    assert restored.input_size == 1024


def test_seed_is_order_independent():
    assert _sample_seed(42, 'DRIVE_1', 'predict') == _sample_seed(
        42, 'DRIVE_1', 'predict')
    assert _sample_seed(42, 'DRIVE_1', 'predict') != _sample_seed(
        42, 'DRIVE_2', 'predict')


def test_adapter_stage_snapshot(tmp_path):
    adapter = A2SAMAdapter()
    adapter.args = _args(workdir=str(tmp_path))
    adapter.device = torch.device('cpu')
    adapter._epoch = 100
    adapter._save_stage_snapshot(
        build_a2sam(image_size=256, prompt_size=64), 'stage0_unet')
    path = tmp_path / 'stage0_unet.pt'
    assert path.is_file()
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['stage'] == 'stage0_unet'
    assert payload['epoch'] == 100
    assert payload['unet_state_dict']
    assert payload['prior_state_dict']
    assert payload['config']['unet_epochs'] == 100


@pytest.mark.slow
def test_real_sam_checkpoint_allowlist():
    path = REPO_ROOT / 'checkpoints/sam_vit_b_01ec64.pth'
    if not path.is_file():
        pytest.skip(f'missing local SAM ViT-B checkpoint: {path}')
    model = build_a2sam()
    info = model.load_sam_checkpoint(path)
    assert info['loaded'] == {
        'image_encoder': 177, 'prompt_encoder': 17,
        'mask_decoder': 89,
    }
    assert sum(
        item['parameters'] for item in info['random_init'].values()
    ) == 692_868
