"""Fast and asset-gated A2SAM tests."""

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.a2sam import A2SAMAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.models.a2sam import (
    ProbabilisticUNet,
    StructuralPromptGenerator,
    build_a2sam,
    contrastive_scores,
    extract_structural_points,
    select_contrastive_masks,
)
from baseline_zoo.models.a2sam.loss import IOU, ThicknessAwareLoss


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
        a2sam_prior_epochs=100,
        a2sam_prior_lr=1e-3,
        a2sam_prior_beta=1e-5,
        a2sam_prior_base=32,
        a2sam_latent_dim=16,
        a2sam_prior_samples=5,
        a2sam_keep_ratio=0.035,
        a2sam_cps_alpha=0.4,
        a2sam_cps_batches=0,
        a2sam_thickness_weight=0.7,
        a2sam_spg_points=512,
        a2sam_spg_stride=3,
        a2sam_spg_hidden=32,
        a2sam_persistence=5.0,
        a2sam_relative_persistence=0.1,
        a2sam_lr_min=1e-7,
        a2sam_random_init=False,
        a2sam_no_augment=False,
        a2sam_sam_checkpoint=Path('checkpoints/sam_vit_b_01ec64.pth'),
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
    assert any('does not contain' in item for item in adapter.deviations)


def test_prior_shapes_kl_and_samples():
    prior = ProbabilisticUNet(base=8, latent_dim=8)
    image = torch.randn(1, 3, 64, 64)
    gt = torch.zeros(1, 1, 64, 64)
    gt[:, :, 20:40, 20:40] = 1
    logits, mu, logvar = prior(image, gt)
    assert logits.shape == (1, 1, 64, 64)
    assert mu.shape == (1, 8) and logvar.shape == (1, 8)
    assert torch.isfinite(prior.kl_divergence(mu, logvar))
    samples = prior.sample_prior(image, k=5)
    assert samples.shape == (5, 1, 1, 64, 64)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, gt) + prior.kl_divergence(mu, logvar)
    loss.backward()
    assert any(p.grad is not None for p in prior.parameters())


def test_topology_extracts_y_branches_and_prunes_empty():
    prob = np.zeros((128, 128), np.float32)
    for y in range(80, 120):
        prob[y, 64] = 0.9
    for i in range(40):
        prob[80 - i, 64 - i] = 0.9
        prob[80 - i, 64 + i] = 0.9
    points = extract_structural_points(
        prob, np.zeros_like(prob), persistence=3, stride=4)
    assert points.n_critical >= 4
    assert points.n_branches >= 3
    assert points.points.shape[1] == 2
    assert points.features.shape == (len(points.points), 7)
    empty = extract_structural_points(
        np.zeros((32, 32), np.float32), np.zeros((32, 32), np.float32))
    assert empty.empty
    assert empty.features.shape == (0, 7)


def test_spg_prompt_shape_and_gradients():
    spg = StructuralPromptGenerator(
        prompt_size=32, max_points=100, stride=4)
    prob = torch.zeros(1, 1, 64, 64)
    prob[:, :, 32, 10:54] = 0.9
    uncertainty = torch.zeros_like(prob)
    prompt = spg(prob, uncertainty)
    assert prompt.shape == (1, 1, 32, 32)
    assert torch.isfinite(prompt).all()
    prompt.mean().backward()
    assert spg.branch_gate[0].weight.grad is not None
    assert spg.fuse[0].weight.grad is not None
    structural = spg.prepare(prob, uncertainty)
    with torch.no_grad():
        again = spg.render(structural, prob, uncertainty)
    assert torch.allclose(prompt.detach(), again)
    empty_prompt = spg(torch.zeros(1, 1, 64, 64),
                       torch.zeros(1, 1, 64, 64))
    assert empty_prompt.shape == (1, 1, 32, 32)
    assert torch.isfinite(empty_prompt).all()


def test_cps_selects_top_fraction_and_masks_gradients():
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(10, 10))
            self.norm = torch.nn.Parameter(torch.ones(10))

    model = Dummy()
    params = dict(model.named_parameters())
    grads = {name: torch.randn_like(param)
             for name, param in params.items()}
    scores = contrastive_scores(grads, params, alpha=0.4)
    masks = select_contrastive_masks(
        scores, params, keep_ratio=0.1, freeze_norm=True)
    assert masks['norm'].sum() == 0
    keep = int(masks['weight'].sum())
    assert 1 <= keep <= 100
    from baseline_zoo.models.a2sam.cps import apply_cps_masks
    stats = apply_cps_masks(model, masks)
    assert stats['kept_elements'] == keep
    model.zero_grad()
    model.weight.sum().backward()
    assert int((model.weight.grad != 0).sum()) == keep


def test_losses_are_finite_and_differentiable():
    logits = torch.randn(1, 1, 64, 64, requires_grad=True)
    gt = torch.zeros(1, 1, 64, 64)
    gt[:, :, 30, 10:54] = 1
    loss = IOU()(logits, gt) + 0.7 * ThicknessAwareLoss()(logits, gt)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_model_build_and_forward_small():
    model = build_a2sam(
        image_size=256, prompt_size=64, prior_base=8,
        latent_dim=8, spg_points=100).eval()
    image = torch.randn(1, 3, 256, 256)
    prob = torch.rand(1, 1, 128, 128)
    uncertainty = torch.rand(1, 1, 128, 128)
    with torch.no_grad():
        masks, low_res, iou = model(
            image, prob, uncertainty, (200, 180))
    assert masks.shape == (1, 1, 200, 180)
    assert low_res.shape == (1, 1, 64, 64)
    assert iou.shape[0] == 1


def test_adapter_data_pipeline(tmp_path):
    adapter = A2SAMAdapter()
    adapter.args = _args()
    adapter.device = torch.device('cpu')
    adapter._stage = 'prior'
    sample = _sample(tmp_path)
    batch = adapter._prepare_batch(
        [sample], random.Random(0), 'prior')
    assert batch['prior_image'].shape == (1, 3, 512, 512)
    assert batch['gt_prior'].shape == (1, 1, 512, 512)
    assert batch['image'].shape == (1, 3, 1024, 1024)
    assert batch['gt'].shape == (1, 1, 48, 64)
    adapter._stage = 'sam'
    with pytest.raises(RuntimeError):
        adapter._prepare_batch([sample], random.Random(0), 'sam')
    adapter._prior_cache[sample.stem] = (
        torch.zeros(1, 512, 512), torch.zeros(1, 512, 512))
    empty = extract_structural_points(
        np.zeros((512, 512), np.float32),
        np.zeros((512, 512), np.float32))
    adapter._structural_cache[sample.stem] = (empty, empty)
    batch = adapter._prepare_batch([sample], random.Random(0), 'sam')
    assert batch['prob'].shape == (1, 1, 512, 512)
    assert batch['uncertainty'].shape == (1, 1, 512, 512)
    assert len(batch['structural']) == 1


def test_adapter_prior_loss_small():
    adapter = A2SAMAdapter()
    adapter.args = _args(a2sam_prior_beta=1e-5)
    adapter.device = torch.device('cpu')
    model = build_a2sam(
        image_size=256, prompt_size=64, prior_base=8,
        latent_dim=8, spg_points=100)
    batch = {
        'prior_image': torch.randn(1, 3, 64, 64),
        'gt_prior': torch.zeros(1, 1, 64, 64),
    }
    batch['gt_prior'][:, :, 20:40, 20:40] = 1
    loss = adapter._prior_loss(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.prior.parameters())


def test_config_roundtrip():
    adapter = A2SAMAdapter()
    adapter.input_size = 1024
    adapter._sam_ckpt = 'checkpoints/sam_vit_b_01ec64.pth'
    adapter._sam_sha = 'abc'
    args = _args()
    config = adapter.save_config(args)
    restored = A2SAMAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert target.input_size == 1024
    assert target.a2sam_prior_epochs == 100
    assert target.a2sam_keep_ratio == pytest.approx(0.035)
    assert target.a2sam_no_augment is False


@pytest.mark.slow
def test_load_sam_checkpoint_if_present():
    path = Path('checkpoints/sam_vit_b_01ec64.pth')
    if not path.is_file():
        pytest.skip(f'missing local SAM ViT-B checkpoint: {path}')
    model = build_a2sam(
        image_size=1024, prior_base=8, latent_dim=8, spg_points=100)
    counts = model.load_sam_checkpoint(path)
    assert counts['image_encoder'] > 0
    assert counts['prompt_encoder'] > 0
    assert counts['mask_decoder'] > 0


@pytest.mark.slow
def test_adapter_sam_loss_backward_small():
    adapter = A2SAMAdapter()
    adapter.args = _args()
    adapter.device = torch.device('cpu')
    model = build_a2sam(
        image_size=256, prompt_size=64, prior_base=8,
        latent_dim=8, spg_points=100)
    batch = {
        'image': torch.randn(1, 3, 256, 256),
        'prob': torch.rand(1, 1, 128, 128),
        'uncertainty': torch.rand(1, 1, 128, 128),
        'gt': torch.zeros(1, 1, 120, 100),
    }
    batch['gt'][:, :, 50:70, 30:70] = 1
    loss = adapter._sam_loss(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.spg.parameters())
