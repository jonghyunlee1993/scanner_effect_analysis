import numpy as np

from analyze_e4_control_frontier import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COLLAPSE_LOWER_THRESHOLD,
    COLLAPSE_POINT_THRESHOLD,
    CONTENT_NONINFERIORITY_MARGIN,
    RAW_RADIUS_EPSILON,
    bootstrap_indices,
)


def test_frozen_e4_thresholds_and_bootstrap():
    assert RAW_RADIUS_EPSILON == 0.01
    assert CONTENT_NONINFERIORITY_MARGIN == -0.02
    assert COLLAPSE_POINT_THRESHOLD == 0.90
    assert COLLAPSE_LOWER_THRESHOLD == 0.85
    assert BOOTSTRAP_REPLICATES == 5000
    assert BOOTSTRAP_SEED == 20260803
    first = bootstrap_indices(109)
    second = bootstrap_indices(109)
    assert first.shape == (5000, 109)
    assert np.array_equal(first, second)

