import numpy as np

from utils.qc import registration_quality


def test_registration_quality_is_bounded_and_identity_is_high():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    identity = registration_quality(image, image)
    shifted = registration_quality(image, np.roll(image, 12, axis=0))
    assert 0 <= identity["q_reg"] <= 1
    assert identity["q_reg"] > 0.9
    assert shifted["residual_shift"] > identity["residual_shift"]
