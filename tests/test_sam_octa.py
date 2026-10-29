"""SAM-OCTA backbone/config regression tests (no heavy forward)."""

from types import SimpleNamespace

from baseline_zoo.adapters.sam_octa import SAMOCTAAdapter


def test_checkpoint_matches_model_type():
    adapter = SAMOCTAAdapter()
    assert adapter._default_checkpoint('vit_b').name == 'sam_vit_b_01ec64.pth'
    assert adapter._default_checkpoint('vit_l').name == 'sam_vit_l_0b3195.pth'
    assert adapter._default_checkpoint('vit_h').name == 'sam_vit_h_4b8939.pth'


def test_config_roundtrip_keeps_backbone(tmp_path):
    adapter = SAMOCTAAdapter()
    args = SimpleNamespace(
        input_size=1024, sam_octa_model_type='vit_h',
        sam_checkpoint=tmp_path / 'missing.pth')
    config = adapter.save_config(args)
    assert config['model_type'] == 'vit_h'
    restored = SimpleNamespace()
    adapter.restore_config(restored, config)
    assert restored.sam_octa_model_type == 'vit_h'
    assert restored.input_size == 1024
