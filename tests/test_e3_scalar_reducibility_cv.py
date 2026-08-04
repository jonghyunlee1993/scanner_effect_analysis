import numpy as np

from analyze_e3_scalar_reducibility_cv import (
    DENSE_GAINS,
    dense_manifold,
    interpolate_curve,
    project_curve,
)


def test_dense_manifold_and_projection_recover_scalar_gain():
    gains = np.asarray([0.0, 1.0, 2.0])
    frequency = np.linspace(0.0, 1.0, 7)
    curves = np.stack([gain * frequency for gain in gains])
    manifold = dense_manifold(gains, curves)
    target = interpolate_curve(1.25, gains, curves)
    best, error = project_curve(target, manifold, np.ones(len(frequency), dtype=bool))
    assert np.isclose(DENSE_GAINS[best], 1.25)
    assert error < 1e-12


def test_off_axis_curve_has_positive_residual():
    gains = np.asarray([0.0, 1.0, 2.0])
    frequency = np.linspace(0.0, 1.0, 7)
    curves = np.stack([gain * frequency for gain in gains])
    manifold = dense_manifold(gains, curves)
    target = 1.25 * frequency + 0.2 * frequency**2
    _, error = project_curve(target, manifold, np.ones(len(frequency), dtype=bool))
    assert error > 0.01
