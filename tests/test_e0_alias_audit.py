import numpy as np
import pandas as pd

from run_e0_alias_audit import (
    alias_ratio_from_matrix,
    radial_periodogram,
    select_transform_profiles,
)


def test_radial_periodogram_localizes_a_sinusoid():
    size = 256
    mpp = 0.5
    frequency = 0.4
    axis = (np.arange(size) - size / 2) * mpp
    image = np.cos(2 * np.pi * frequency * axis)[None, :].repeat(size, axis=0)
    edges = np.arange(0, 1.025, 0.025)
    centers = (edges[:-1] + edges[1:]) / 2
    power = radial_periodogram(image, mpp, edges, input_variance=np.var(image))
    assert abs(centers[np.argmax(power)] - frequency) <= 0.025
    assert 0.9 <= power.sum() <= 1.1


def test_alias_ratio_uses_2d_radial_frequency_weighting():
    input_frequency = np.array([0.7, 1.1])
    output_centers = np.array([0.7])
    mixing = np.array([[2.0], [1.0]])
    result = alias_ratio_from_matrix(
        mixing, input_frequency, output_centers, target_nyquist=1.0
    )
    assert np.isclose(result["sinusoid_alias_to_inband_ratio"], 1.1 / 1.4)


def test_transform_profile_selection_returns_observed_quantiles():
    frame = pd.DataFrame(
        {
            "scanner": ["gt450"] * 5,
            "slide_id": ["a", "b", "c", "d", "e"],
            "output_px_per_native_px_geom": [0.40, 0.45, 0.50, 0.55, 0.60],
        }
    )
    selected = select_transform_profiles(frame)
    assert selected["transform_profile"].tolist() == ["q05", "q50", "q95"]
    assert selected["slide_id"].tolist() == ["a", "c", "e"]
