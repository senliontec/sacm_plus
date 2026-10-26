"""Official inference threshold (synthetic calibration case)."""


def predict(logits):
    """Official test path: threshold raw logits at 0.5."""
    return (logits > 0.5).float()
