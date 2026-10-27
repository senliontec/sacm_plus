"""Fast SAMed adapter tests (no ViT-B build)."""

import random
from functools import partial
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.adapters.samed import SAMedAdapter
from baseline_zoo.data import Sample
from baseline_zoo.models.samed import (
    DiceLoss,
    ForegroundDiceLoss,
    SAMedNet,
)
from baseline_zoo.models.samed.segment_anything.modeling import (
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
        num_multimask_outputs=1,
        transformer=TwoWayTransformer(
            depth=2, embedding_dim=32, mlp_dim=64, num_heads=2),
        transformer_dim=32, iou_head_depth=2, iou_head_hidden_dim=32)
    sam = Sam(encoder, prompt_encoder, mask_decoder,
              pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
    return SAMedNet(sam, rank=4)


def test_registry_and_metadata():
    assert 'samed' in available_adapters()
    adapter = load_adapter('samed')
    assert isinstance(adapter, SAMedAdapter)
    assert adapter.id == 'samed'
    assert adapter.input_size == 512
    assert adapter.train_batch_size == 4
    assert adapter.paper_lr == 5e-3
    assert adapter.official_lr == 1e-3
    assert 'fully supervised' in adapter.deviations[0]


def test_tiny_forward_loss_and_policy():
    adapter = SAMedAdapter()
    model = _tiny_model()
    adapter.configure_trainable(model)

    assert not model.lora_sam.sam.image_encoder.patch_embed.proj.weight.requires_grad
    assert model.lora_sam.sam.image_encoder.blocks[0].attn.qkv.linear_a_q.weight.requires_grad
    assert model.lora_sam.sam.image_encoder.blocks[0].attn.qkv.linear_b_q.weight.requires_grad
    assert model.lora_sam.sam.prompt_encoder.no_mask_embed.weight.requires_grad
    assert model.lora_sam.sam.mask_decoder.iou_token.weight.requires_grad

    low, masks = model(torch.randn(1, 3, 64, 64))
    assert low.shape == (1, 2, 16, 16)
    assert masks.shape == (1, 2, 64, 64)

    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, :8] = 1
    loss = (0.2 * torch.nn.functional.cross_entropy(low, target)
            + 0.8 * DiceLoss(2)(low, target, softmax=True))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.lora_sam.sam.image_encoder.blocks[0].attn.qkv.linear_a_q.weight.grad is not None


def test_foreground_dice_penalizes_all_background():
    loss_fn = ForegroundDiceLoss()
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, :8] = 1
    background_logits = torch.stack([
        torch.full_like(target, 10.0),
        torch.full_like(target, -10.0),
    ], dim=1).float()
    perfect_logits = torch.stack([
        torch.where(target == 1, -10.0, 10.0),
        torch.where(target == 1, 10.0, -10.0),
    ], dim=1).float()
    assert loss_fn(background_logits, target) > 0.9
    assert loss_fn(perfect_logits, target) < 0.01


def test_train_step_weighted_aux_loss_and_backward():
    adapter = SAMedAdapter()
    adapter.input_size = 64
    adapter.device = torch.device('cpu')
    adapter.args = SimpleNamespace(
        samed_bg_weight=0.1, samed_fg_weight=1.0,
        samed_aux_weight=0.5)
    model = _tiny_model()
    adapter.configure_trainable(model)
    images = torch.randn(1, 3, 64, 64)
    targets = torch.zeros(1, 64, 64, dtype=torch.long)
    targets[:, :32] = 1
    low_res = torch.zeros(1, 16, 16, dtype=torch.long)
    low_res[:, :8] = 1
    loss = adapter.train_step(model, (images, targets, low_res))
    assert torch.isfinite(loss)
    loss.backward()
    assert (model.lora_sam.sam.image_encoder.blocks[0].attn.qkv
            .linear_a_q.weight.grad is not None)
    assert (model.lora_sam.sam.mask_decoder.mask_tokens.weight.grad
            is not None)


def test_predict_sample_uses_argmax_and_nearest(tmp_path):
    adapter = SAMedAdapter()
    adapter.input_size = 64
    adapter.device = torch.device('cpu')
    sample = _sample(tmp_path)
    prepared = adapter.prepare_eval_sample(
        sample, torch.device('cpu'))
    model = _tiny_model()
    model.eval()
    pred = adapter.predict_sample(model, prepared, sample)
    assert pred.shape == (24, 32)
    assert pred.dtype == bool


def test_preprocess_shapes(tmp_path):
    adapter = SAMedAdapter()
    adapter.input_size = 512
    sample = _sample(tmp_path)
    image, mask, low = adapter._preprocess(
        sample, random.Random(0), train=True)
    assert image.shape == (3, 512, 512)
    assert mask.shape == (512, 512)
    assert low.shape == (128, 128)
    assert image.dtype == torch.float32
    assert set(mask.unique().tolist()) <= {0, 1}
    assert set(low.unique().tolist()) <= {0, 1}
    assert torch.isfinite(image).all()


def test_config_roundtrip(tmp_path):
    adapter = SAMedAdapter()
    adapter.input_size = 512
    adapter._sam_sha = 'abc'
    args = SimpleNamespace(
        sam_checkpoint=tmp_path / 'missing.pth', sam_type='vit_b',
        lora_rank=4, samed_bg_weight=0.1, samed_fg_weight=1.0,
        samed_aux_weight=0.5)
    config = adapter.save_config(args)
    assert config['input_size'] == 512
    assert config['sam_checkpoint_sha256'] == 'abc'
    assert config['num_classes'] == 1
    assert config['samed_bg_weight'] == 0.1
    assert config['samed_aux_weight'] == 0.5

    restore = SimpleNamespace(sam_checkpoint=None, input_size=0,
                              lora_rank=0)
    adapter.restore_config(restore, config)
    assert restore.input_size == 512
    assert restore.lora_rank == 4
    assert restore.sam_type == 'vit_b'
    assert restore.samed_bg_weight == 0.1
    assert restore.samed_aux_weight == 0.5


def test_low_res_tracks_input_size(tmp_path):
    adapter = SAMedAdapter()
    adapter.input_size = 256
    _, _, low = adapter._preprocess(
        _sample(tmp_path), random.Random(0), train=False)
    assert low.shape == (64, 64)


def test_poly_scheduler_does_not_zero_last_epoch():
    adapter = SAMedAdapter()
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    scheduler = adapter.make_scheduler(
        optimizer, SimpleNamespace(epochs=160))
    for _ in range(160):
        scheduler.step()
    assert optimizer.param_groups[0]['lr'] > 0


def test_warmup_is_scaled_by_iterations():
    adapter = SAMedAdapter()
    args = SimpleNamespace(epochs=200, batch_size=4)
    # 18 images / batch 4 -> 5 batches per epoch; 250 iterations -> 50 epochs.
    assert adapter._warmup_epochs(args) == 50

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    adapter.make_scheduler(optimizer, args)
    # The first LR must be about base_lr / 50, not the old base_lr / 5.
    assert 8e-5 < optimizer.param_groups[0]['lr'] < 1.2e-4
