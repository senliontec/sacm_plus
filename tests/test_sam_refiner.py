"""Fast SAMRefiner tests (no SAM checkpoint)."""

import importlib.util
from pathlib import Path

import torch

from core.postprocess.sam_refiner import SAMRefiner
from core.postprocess.sam_refiner.utils import _generalised_geodesic2d

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_wrapper_exports_class():
    assert SAMRefiner.__name__ == 'SAMRefiner'


def test_distance_fallback_is_positive_inside_mask():
    mask = torch.zeros(1, 1, 16, 16)
    mask[0, 0, 3:13, 3:13] = 1
    distance = _generalised_geodesic2d(
        torch.ones(1, 1, 16, 16), mask, 1e10, 0.0, 2)
    assert distance.shape == mask.shape
    assert float(distance.max()) > 0
    assert float(distance[0, 0, 0, 0]) == 0


def _load_cli():
    path = REPO_ROOT / 'scripts' / 'tools' / 'apply_sam_refiner.py'
    spec = importlib.util.spec_from_file_location('apply_sam_refiner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_resolves_image_from_manifest(tmp_path):
    cli = _load_cli()
    image = tmp_path / 'images' / 'DRIVE_1.png'
    image.parent.mkdir(parents=True)
    image.write_bytes(b'not an image; path resolution only')
    manifest = tmp_path / 'protocol_manifest.csv'
    manifest.write_text(
        'dataset,image_path,mask_path\n'
        f'DRIVE,{image},data/masks/DRIVE_1.png\n')
    mapping = cli.load_manifest(manifest)
    prediction = tmp_path / 'pred' / 'DRIVE' / 'DRIVE_1.png'
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b'mask')
    resolved = cli.resolve_image(
        prediction, tmp_path / 'pred', mapping, None)
    assert resolved == image


def test_cli_single_file_requires_image(tmp_path):
    cli = _load_cli()
    prediction = tmp_path / 'mask.png'
    prediction.write_bytes(b'mask')
    try:
        cli.resolve_image(prediction, prediction, {}, None)
    except SystemExit as error:
        assert '--image' in str(error)
    else:
        raise AssertionError('expected SystemExit')
