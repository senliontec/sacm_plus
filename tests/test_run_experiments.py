"""Launcher layout/control-flow tests."""

import importlib.util
import os
from argparse import Namespace

_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'scripts',
                       'run_experiments.py')
_spec = importlib.util.spec_from_file_location('run_experiments', _SCRIPT)
rex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rex)


def _args():
    return Namespace(
        train_root='data/sacm_3shot',
        checkpoint='checkpoints/sam_vit_l.pth',
        epochs=50,
        val_interval=10,
        seed=42,
        pred_threshold=0.0,
        extra_train_args='',
        output_root='results',
        tta=False,
    )


def test_train_once_checkpoint_and_test_output_layout():
    args = _args()
    ckpt = os.path.join('results', 'full', 'best_model.pth')
    train = rex.train_command(args, 'full', ckpt, 'iou')
    assert train[train.index('--save_path') + 1] == ckpt

    test = rex.test_command(
        args, 'full', 'DRIVE', '/data/DRIVE', ckpt, 'iou')
    assert test[test.index('--trained_weights') + 1] == ckpt
    assert test[test.index('--output_dir') + 1] == os.path.join(
        'results', 'full', 'DRIVE')


def test_resolve_selection_uses_preset_iou_head():
    args = _args()
    args.selection = None
    assert rex.resolve_selection(args, 'sacm') == 'gating'
    assert rex.resolve_selection(args, 'full') == 'iou'


def test_train_failure_is_terminal_and_marks_all_datasets():
    class _Proc:
        returncode = 1

    class _Log:
        def close(self):
            pass

    slot = {
        'preset': 'full', 'gpu': 0, 'phase': 'train',
        'proc': _Proc(), 'datasets': [('DRIVE', '/d'), ('CORN', '/c')],
        'test_queue': [], 'rows': [], 'status': 'running',
        'log': _Log(), 'ckpt': 'x', 'selection': 'iou', 'env': {},
        'current_dataset': None,
    }
    done = rex.advance(slot, Namespace(skip_test=False))
    assert done is True
    assert slot['status'] == 'train_failed'
    assert [r['dataset'] for r in slot['rows']] == ['DRIVE', 'CORN']
