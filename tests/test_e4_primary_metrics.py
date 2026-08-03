import numpy as np
import pytest

from e4_primary_metrics import (
    bootstrap_mean_ci,
    collapse_metric_values,
    content_margin_by_scanner,
    l2_normalize,
    scanner_centroid_rms,
    unmatched_q95,
)


def test_scanner_radius_zero_for_identical_scanners():
    rng = np.random.default_rng(1)
    locations = rng.normal(size=(100, 8))
    features = np.repeat(locations[None], 6, axis=0)
    assert scanner_centroid_rms(features) < 1e-12


def test_scanner_radius_detects_scanner_offsets():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(6, 100, 8))
    assert scanner_centroid_rms(features) > 0.1


def test_content_margin_rewards_exact_match():
    reference = np.eye(100)
    features = np.repeat(reference[None], 6, axis=0)
    margin = content_margin_by_scanner(features, reference)
    assert margin.shape == (5, 100)
    assert np.allclose(unmatched_q95(reference), 0.0)
    assert np.allclose(margin, 1.0)


def test_collapse_metrics_change_under_rank_collapse():
    rng = np.random.default_rng(3)
    rich = rng.normal(size=(100, 20))
    scalar = rng.normal(size=(100, 1))
    collapsed = np.repeat(scalar, 20, axis=1)
    rich_metrics = collapse_metric_values(rich)
    collapsed_metrics = collapse_metric_values(collapsed)
    assert rich_metrics["entropy_effective_rank"] > collapsed_metrics["entropy_effective_rank"]
    assert rich_metrics["variance_trace"] > 0
    assert collapsed_metrics["median_pairwise_distance"] >= 0


def test_bootstrap_contract():
    values = np.arange(1, 6, dtype=float)
    indices = np.tile(np.arange(5), (10, 1))
    point, lower, upper = bootstrap_mean_ci(values, indices)
    assert point == lower == upper == 3.0
    with pytest.raises(ValueError):
        bootstrap_mean_ci(values, indices[:, :4])


def test_l2_normalize_rejects_zero():
    with pytest.raises(ValueError):
        l2_normalize(np.zeros((2, 3)))
