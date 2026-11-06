"""Guard the shared 800-epoch baseline budget."""

from baseline_zoo.registry import load_models

TRAINING_FREE = {'c2p', 'matcher', 'persam', 'rsrm'}
# KnowSAM's --epochs flag is an iteration budget in the official
# semi-supervised loop, so its 50,000 iterations are not an epoch count.
KNOWSAM_ITERATIONS = 50000


def test_trainable_baselines_use_shared_800_epoch_budget():
    models = load_models()
    assert TRAINING_FREE <= set(models), 'training-free model missing'
    assert 'knowsam' in models

    for model_id, model in models.items():
        epochs = int(model.defaults['epochs'])
        if model_id in TRAINING_FREE:
            assert epochs == 0, f'{model_id} must stay training-free'
        elif model_id == 'knowsam':
            assert epochs == KNOWSAM_ITERATIONS, (
                'knowsam defaults.epochs is its iteration budget')
        else:
            assert epochs == 800, (
                f'{model_id} must use the shared 800-epoch budget, '
                f'got {epochs}')
