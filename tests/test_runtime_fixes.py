"""Regression tests for issues found in the full porting audit."""

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.registry import load_adapter
from baseline_zoo.data import load_split_samples
from baseline_zoo.models.segdino import find_dinov3_checkpoint
from baseline_zoo.protocols import load_protocols
from baseline_zoo.registry import load_models
from baseline_zoo.runtime import binary_metrics, build_parser
from core.metrics import cldice_score, dice_score, hd95_score, iou_score


def _write_pair(root: Path, stem: str, size=(16, 16)):
    image_dir, mask_dir = root / 'images', root / 'masks'
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[4:12, 4:12] = 200
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    Image.fromarray(image).save(image_dir / f'{stem}.png')
    Image.fromarray(mask).save(mask_dir / f'{stem}.png')


def _manifest(root: Path, rows):
    with (root / 'split_manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['dataset', 'split', 'source_image',
                                'output', 'group'])
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_membership_rejects_stray_and_missing(tmp_path):
    train = tmp_path / 'train'
    for stem in ('DRIVE_a', 'DRIVE_b', 'DRIVE_c'):
        _write_pair(train, stem)
    _manifest(tmp_path, [
        {'dataset': 'DRIVE', 'split': 'train', 'source_image': 'a.tif',
         'output': 'DRIVE_a', 'group': 'DRIVE/a'},
    ])
    with pytest.raises(SystemExit):
        load_split_samples(tmp_path, 'train')

    _manifest(tmp_path, [
        {'dataset': 'DRIVE', 'split': 'train', 'source_image': 'a.tif',
         'output': 'DRIVE_a', 'group': 'DRIVE/a'},
        {'dataset': 'DRIVE', 'split': 'train', 'source_image': 'b.tif',
         'output': 'DRIVE_b', 'group': 'DRIVE/b'},
        {'dataset': 'DRIVE', 'split': 'train', 'source_image': 'c.tif',
         'output': 'DRIVE_c', 'group': 'DRIVE/c'},
    ])
    samples = load_split_samples(tmp_path, 'train')
    assert [sample.stem for sample in samples] == [
        'DRIVE_a', 'DRIVE_b', 'DRIVE_c']


def test_fixed_protocol_resolves_frozen_split():
    protocol = load_protocols()['fixed']
    split = protocol.split_for(42)
    assert split.split_root.name == 'sacm_3shot'
    assert split.test_manifest.name == 'protocol_manifest.csv'


def test_sacm_official_clip_default_is_effective():
    parser = build_parser(load_adapter('sacm_official'))
    assert parser.parse_args([]).grad_clip == 1.0
    assert parser.parse_args(['--grad-clip', '0.25']).grad_clip == 0.25


def test_pipeline_commands_use_current_interpreter():
    import shlex
    import sys

    from baseline_zoo.__main__ import _pythonize_command

    assert shlex.split(_pythonize_command(
        'python -m baseline_zoo.run_one --model unet'))[0] == sys.executable
    assert shlex.split(_pythonize_command(
        'python3 scripts/pipeline/x.py'))[0] == sys.executable
    unchanged = '/usr/bin/python -m example'
    assert _pythonize_command(unchanged) == unchanged


def test_begin_prediction_resets_seed_latch():
    adapter = load_adapter('matcher')
    adapter.prediction_mode = False
    adapter._prediction_seeded = True
    adapter.begin_prediction()
    assert adapter.prediction_mode is True
    assert adapter._prediction_seeded is False


def test_generic_scheduler_modes():
    from baseline_zoo.runtime import make_training_scheduler

    adapter = load_adapter('unet')
    for mode in ('constant', 'cosine', 'poly', 'onecycle'):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        args = argparse.Namespace(
            epochs=5, warmup_epochs=1, scheduler=mode, lr=1e-3)
        scheduler = make_training_scheduler(adapter, optimizer, args)
        lrs = []
        for _ in range(5):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])
        assert all(lr >= 0 and lr == lr for lr in lrs)
        if mode == 'constant':
            assert lrs[-1] == pytest.approx(1e-3)
        else:
            assert lrs[-1] < 1e-3


def test_official_scheduler_override_delegates():
    from baseline_zoo.runtime import make_training_scheduler

    class DummyAdapter:
        def make_scheduler(self, optimizer, args):
            return 'official-scheduler'

    assert make_training_scheduler(
        DummyAdapter(), object(),
        argparse.Namespace(scheduler='official')) == 'official-scheduler'


def test_scheduler_parser_defaults_preserve_official():
    parser = build_parser(load_adapter('unet'))
    args = parser.parse_args([])
    assert args.scheduler == 'official'
    assert args.warmup_epochs is None


def test_parser_epochs_default_from_model_config():
    models = load_models()
    for model_id in ('unet', 'dscnet', 'samed', 'fob_sam', 'cs2net'):
        parser = build_parser(load_adapter(model_id))
        assert parser.parse_args([]).epochs == models[model_id].defaults['epochs']


def test_cs2net_schedule_does_not_decay_before_first_epoch():
    adapter = load_adapter('cs2net')
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = adapter.make_scheduler(
        optimizer, argparse.Namespace(epochs=4))
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-4)
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-4)
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] < 1e-4


def test_sam_octa_does_not_prepad_non_square_image(tmp_path):
    _write_pair(tmp_path, 'LES-AV_1', size=(60, 80))
    from baseline_zoo.data import RawPairDataset
    sample = RawPairDataset(tmp_path).samples[0]
    adapter = load_adapter('sam_octa')
    adapter.args = argparse.Namespace()
    adapter.device = torch.device('cpu')
    prepared = adapter.prepare_eval_sample(sample, torch.device('cpu'))
    # Longest side is resized to 1024; the short side must stay unpadded.
    assert prepared['image'].shape == (1, 3, 1024, 768)
    assert prepared['original_size'] == (80, 60)


def test_sam_vit_l_unet_state_dict_is_decoder_only():
    adapter = load_adapter('sam_vit_l_unet')

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.decoder = torch.nn.Linear(2, 2)

    model = Dummy()
    state = adapter.state_dict(model)
    assert all(not key.startswith('encoder.') for key in state)
    assert all(key in model.decoder.state_dict() for key in state)


def test_knowsam_shared_wandb_interval_default():
    parser = build_parser(load_adapter('knowsam'))
    args = parser.parse_args([])
    assert args.log_interval == 50
    assert args.val_interval == 200


def test_knowsam_unlabeled_pool_excludes_protocol_images(tmp_path):
    adapter = load_adapter('knowsam')
    root = tmp_path / 'stage'
    adapter._stage(root, Path('data/sacm_3shot_seed0'), 8)
    staged = {path.name for path in (root / 'unlabeled' / 'image').glob('*')}
    manifest = Path('data/sacm_3shot_seed0/split_manifest.csv')
    with manifest.open(newline='') as handle:
        for row in csv.DictReader(handle):
            # The unlabeled pool must exclude train, val and test rows.
            name = f"{row['dataset']}_{Path(row['source_image']).stem}_image.png"
            assert name not in staged


def test_knowsam_largest_cc():
    from baseline_zoo.adapters.knowsam import _largest_cc

    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True       # 4 pixels
    mask[5:8, 5:8] = True       # 9 pixels
    result = _largest_cc(mask)
    assert int(result.sum()) == 9
    assert result[5:8, 5:8].all()
    assert not result[1:3, 1:3].any()

    empty = np.zeros((4, 4), dtype=bool)
    assert np.array_equal(_largest_cc(empty), empty)


def test_nnwnet_plan_command_skips_first_preprocess():
    adapter = load_adapter('nnwnet')
    command = adapter._plan_command()
    assert '--no_pp' in command
    assert 'plan_and_preprocess_entrypoints' in ' '.join(command)


def test_nnwnet_syncs_preprocessed_dataset_json(tmp_path):
    adapter = load_adapter('nnwnet')
    raw = tmp_path / 'raw' / 'Dataset501_Curvilinear3Shot'
    raw.mkdir(parents=True)
    (raw / 'dataset.json').write_text(json.dumps({'numTraining': 24}))
    preprocessed = tmp_path / 'preprocessed' / 'Dataset501_Curvilinear3Shot'
    adapter._sync_preprocessed_dataset_json(tmp_path, preprocessed)
    assert json.loads(
        (preprocessed / 'dataset.json').read_text())['numTraining'] == 24


def test_nnwnet_planning_stage_excludes_val(tmp_path):
    adapter = load_adapter('nnwnet')
    raw = tmp_path / 'raw'
    train, val = adapter._stage_raw(
        raw, Path('data/sacm_3shot_seed0'), include_val=False)
    assert len(train) == 18
    assert val == []
    dataset_json = json.loads(
        (raw / 'Dataset501_Curvilinear3Shot' / 'dataset.json').read_text())
    assert dataset_json['numTraining'] == 18
    labels = list(
        (raw / 'Dataset501_Curvilinear3Shot' / 'labelsTr').glob('*.png'))
    assert len(labels) == 18

    # The real two-phase flow stages val only after planning.  Re-staging
    # with val must keep the 18 train cases and add exactly 6 disjoint val
    # cases; the raw dataset.json must advertise 24 cases for the second
    # preprocessing pass.
    train_all, val_all = adapter._stage_raw(
        raw, Path('data/sacm_3shot_seed0'), include_val=True)
    assert len(train_all) == 18
    assert len(val_all) == 6
    assert set(train_all) == set(train)
    assert not set(train_all) & set(val_all)
    dataset_json = json.loads(
        (raw / 'Dataset501_Curvilinear3Shot' / 'dataset.json').read_text())
    assert dataset_json['numTraining'] == 24


def test_segdino_auto_detects_size_matched_checkpoint(tmp_path):
    dino = tmp_path / 'dinov3'
    dino.mkdir()
    generic = dino / 'pytorch_model.bin'
    generic.write_bytes(b'x')
    base = tmp_path / 'vit_base_patch16_dinov3.pth'
    base.write_bytes(b'x')
    assert find_dinov3_checkpoint('small', tmp_path) == generic
    assert find_dinov3_checkpoint('base', tmp_path) == base


def test_segdino_ignores_wrong_size_checkpoint(tmp_path):
    (tmp_path / 'vit_base_patch16_dinov3.pth').write_bytes(b'x')
    assert find_dinov3_checkpoint('small', tmp_path) is None


def test_segdino_ambiguous_checkpoints_fail(tmp_path):
    dino = tmp_path / 'dinov3'
    dino.mkdir()
    (dino / 'model.pth').write_bytes(b'x')
    (dino / 'dinov3.pth').write_bytes(b'x')
    with pytest.raises(ValueError, match='multiple DINOv3'):
        find_dinov3_checkpoint('small', tmp_path)


def test_segdino_restores_backbone_config():
    adapter = load_adapter('segdino')
    args = argparse.Namespace(
        dino_size='large', dino_ckpt=Path('weights.pth'), random_init=False)
    saved = adapter.save_config(args)
    target = argparse.Namespace(
        dino_size='small', dino_ckpt=None, random_init=False)
    adapter.restore_config(target, saved)
    assert target.dino_size == 'large'
    assert target.random_init is True
    assert target.dino_ckpt is None


def test_binary_metrics_match_canonical_paper_metrics():
    pred = np.zeros((32, 32), dtype=bool)
    gt = np.zeros((32, 32), dtype=bool)
    pred[4:20, 5:18] = True
    gt[6:22, 8:20] = True
    metrics = binary_metrics(pred, gt)
    assert metrics['dice'] == dice_score(pred, gt)
    assert metrics['iou'] == iou_score(pred, gt)
    assert metrics['cldice'] == cldice_score(pred, gt)
    assert metrics['hd95'] == hd95_score(pred, gt)


def test_nnwnet_discovers_checkpoint_extension(tmp_path):
    adapter = load_adapter('nnwnet')
    fold = tmp_path / 'fold_0'
    fold.mkdir()
    (fold / 'checkpoint_best.pth').write_bytes(b'pth')
    assert adapter._find_checkpoint(tmp_path, 'best').name == 'checkpoint_best.pth'
    (fold / 'checkpoint_best.pth').unlink()
    (fold / 'checkpoint_best.safetensors').write_bytes(b'safe')
    assert adapter._find_checkpoint(tmp_path, 'best').name == \
        'checkpoint_best.safetensors'


def test_nnwnet_uses_requested_cuda_device(monkeypatch):
    from baseline_zoo.adapters import nnwnet

    calls = []
    monkeypatch.setattr(
        nnwnet.torch.cuda, 'set_device',
        lambda device: calls.append(device))
    monkeypatch.setattr(nnwnet.torch.cuda, 'current_device', lambda: 2)

    assert nnwnet._resolve_device('cuda:4') == torch.device('cuda:4')
    assert calls == [torch.device('cuda:4')]

    calls.clear()
    assert nnwnet._resolve_device('cuda') == torch.device('cuda:2')
    assert calls == [torch.device('cuda:2')]

    calls.clear()
    assert nnwnet._resolve_device('cpu') == torch.device('cpu')
    assert calls == []


def test_nnwnet_restores_trainer_device():
    from baseline_zoo.adapters import nnwnet

    class DummyTrainer:
        def __init__(self):
            self.device = torch.device('cuda:0')
            self.my_init_kwargs = {'device': torch.device('cuda:0')}

    trainer = DummyTrainer()
    requested = torch.device('cuda:4')
    nnwnet._restore_trainer_device(trainer, requested)
    assert trainer.device == requested
    assert trainer.my_init_kwargs['device'] == requested


def test_sam_octa_eval_seed_and_batch_guard(tmp_path):
    _write_pair(tmp_path, 'TEST_0', size=(32, 24))
    from baseline_zoo.data import RawPairDataset
    sample = RawPairDataset(tmp_path).samples[0]

    def prepare(prior_seed):
        adapter = load_adapter('sam_octa')
        adapter.args = argparse.Namespace(seed=5, stage='predict')
        adapter.device = torch.device('cpu')
        np.random.seed(prior_seed)
        random.seed(prior_seed)
        return adapter.prepare_eval_sample(sample, torch.device('cpu'))

    first = prepare(999)
    second = prepare(111)
    assert torch.equal(first['point_coords'], second['point_coords'])
    assert torch.equal(first['point_labels'], second['point_labels'])

    adapter = load_adapter('sam_octa')
    adapter.args = argparse.Namespace(seed=5, stage='train')
    adapter.device = torch.device('cpu')
    with pytest.raises(RuntimeError, match='batch-size 1'):
        adapter.prepare_train_batch([sample, sample], random.Random(0))


def test_bcunet_prediction_skips_pretrained_backbone(monkeypatch):
    captured = {}
    loaded = []

    class DummyBCUNet:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr('baseline_zoo.adapters.bcunet.BCUNet', DummyBCUNet)
    adapter = load_adapter('bcunet')
    monkeypatch.setattr(
        adapter, '_load_local_convnext',
        lambda model, args: loaded.append(True))

    adapter.prediction_mode = True
    adapter.build_model(argparse.Namespace(
        pretrained=True, checkpoint=None, stage='predict'))
    assert captured['pretrained'] is False
    assert loaded == []

    captured.clear()
    adapter.prediction_mode = False
    adapter.build_model(argparse.Namespace(
        pretrained=True, checkpoint=None, stage='train',
        convnext_checkpoint=Path('unused.pth')))
    assert captured['pretrained'] is False
    assert loaded == [True]


def test_sacm_prediction_skips_missing_base_checkpoint(monkeypatch):
    captured = {}

    class Dummy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        'baseline_zoo.adapters.sacm_official.build_sam_vit_l', Dummy)
    adapter = load_adapter('sacm_official')
    adapter.prediction_mode = True
    adapter.build_model(argparse.Namespace(
        sam_checkpoint=Path('/missing/vit_l.pth'),
        adapter_dim_ratio=0.2))
    assert captured['checkpoint'] is None
    assert captured['adapter_dim_ratio'] == 0.2

    adapter.prediction_mode = False
    with pytest.raises(SystemExit):
        adapter.build_model(argparse.Namespace(
            sam_checkpoint=Path('/missing/vit_l.pth'),
            adapter_dim_ratio=0.2))


def test_sacm_adapter_dim_ratio_roundtrip():
    adapter = load_adapter('sacm_official')
    args = argparse.Namespace(
        adapter_dim_ratio=0.25, bce_weight=0.6, dice_weight=0.4)
    config = adapter.save_config(args)
    restored = argparse.Namespace()
    adapter.restore_config(restored, config)
    assert restored.adapter_dim_ratio == 0.25
    assert restored.bce_weight == 0.6
    assert restored.dice_weight == 0.4


def test_sam_vit_l_unet_records_and_checks_base_sha(tmp_path, capsys):
    from baseline_zoo.runtime import file_sha256
    adapter = load_adapter('sam_vit_l_unet')
    first = tmp_path / 'sam_a.pth'
    second = tmp_path / 'sam_b.pth'
    first.write_bytes(b'a')
    second.write_bytes(b'b')
    adapter._sam_sha = file_sha256(first)
    config = adapter.save_config(
        argparse.Namespace(sam_checkpoint=first))
    assert config['sam_checkpoint_sha256'] == file_sha256(first)
    restored = argparse.Namespace(
        sam_checkpoint=second, input_size=0)
    adapter.restore_config(restored, config)
    assert restored.input_size == 1024
    assert 'WARNING: SAM base checkpoint differs' in capsys.readouterr().out


def _load_tuning_module():
    path = (Path(__file__).resolve().parents[1] /
            'scripts' / 'tools' / 'tune_baselines.py')
    spec = importlib.util.spec_from_file_location(
        'tune_baselines_regression', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tuning_seed_and_gpu_parsing():
    tune = _load_tuning_module()
    protocol = load_protocols()['seeded5']
    assert tune._parse_seeds(['0,1', '2'], None, protocol) == [0, 1, 2]
    assert tune._parse_seeds(['all'], None, protocol) == [0, 1, 2, 3, 4]
    with pytest.raises(SystemExit):
        tune._parse_seeds(['9'], None, protocol)
    with pytest.raises(SystemExit):
        tune._parse_gpus('0,0')


def test_tuning_trial_command_passes_selection_and_seed():
    tune = _load_tuning_module()
    model = load_models()['unet']
    protocol = load_protocols()['seeded5']
    command = tune._trial_command(
        model=model,
        model_id='unet',
        protocol=protocol,
        seed=3,
        gpu='2',
        trial_dir=Path('/tmp/tune_trial'),
        params={
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'scheduler': 'cosine',
            'warmup_epochs': 5,
            '_trial_number': 7,
        },
        epochs=50,
        selection='iou',
        train_entrypoint=model.train_entrypoint,
    )
    assert '--device cuda:2' in command
    assert '--selection iou' in command
    assert '--scheduler cosine' in command
    assert '--warmup-epochs 5' in command
    assert '--wandb_name unet-seed3-trial7' in command
