"""Official training-free postprocess (synthetic calibration case)."""


def predict(probabilities, min_votes=3):
    binary = probabilities > 0.5
    return binary, min_votes
