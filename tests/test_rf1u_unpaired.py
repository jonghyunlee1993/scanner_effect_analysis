"""Contract tests for the RF1U unpaired multi-target estimator."""

from __future__ import annotations

import numpy as np
import pytest

from e5_comparator_population import SCANNERS
from rf1m_combined import RF1M_SIGMAS
from rf1u_unpaired import (
    RF1U_HARD_CAP,
    RF1U_SHRINKAGE_SE,
    RF1U_TARGETS,
    bootstrap_log_gain_se,
    fitted_scanner_gains,
    log_population_gain,
    reliability_alpha,
    shrunk_band_gains,
    source_indices,
    target_index,
)


N_BANDS = len(RF1M_SIGMAS)


def test_targets_are_known_scanners_and_sources_exclude_them():
    for target in RF1U_TARGETS:
        index = target_index(target)
        assert SCANNERS[index] == target
        sources = source_indices(target)
        assert len(sources) == len(SCANNERS) - 1
        assert index not in sources
        assert list(sources) == sorted(sources)


def test_unknown_target_is_rejected():
    with pytest.raises(ValueError):
        target_index("versa")


def test_population_gain_pools_before_dividing():
    """The estimator must not depend on how slides are paired across the sets."""
    generator = np.random.default_rng(20260804)
    target = generator.uniform(1.0, 4.0, size=(40, N_BANDS))
    source = generator.uniform(1.0, 4.0, size=(40, N_BANDS))
    expected = 0.5 * (np.log(target.mean(axis=0)) - np.log(source.mean(axis=0)))
    assert np.allclose(log_population_gain(target, source), expected)
    order = generator.permutation(40)
    assert np.allclose(
        log_population_gain(target, source), log_population_gain(target, source[order])
    )


def test_population_gain_is_zero_for_identical_populations():
    energy = np.full((10, N_BANDS), 2.5)
    assert np.allclose(log_population_gain(energy, energy), 0.0)


def test_population_gain_rejects_malformed_input():
    energy = np.full((10, N_BANDS), 2.0)
    with pytest.raises(ValueError):
        log_population_gain(energy, energy[:5])
    with pytest.raises(ValueError):
        log_population_gain(energy, -energy)
    with pytest.raises(ValueError):
        log_population_gain(energy[0], energy[0])


def test_bootstrap_se_is_deterministic_and_shrinks_with_agreement():
    generator = np.random.default_rng(20260804)
    target = generator.uniform(2.0, 3.0, size=(60, N_BANDS))
    noisy = generator.uniform(0.5, 6.0, size=(60, N_BANDS))
    tight = np.full((60, N_BANDS), 2.5)
    first = bootstrap_log_gain_se(target, noisy)
    assert np.allclose(first, bootstrap_log_gain_se(target, noisy))
    assert np.all(bootstrap_log_gain_se(target, tight) < first)


def test_alpha_is_one_when_certain_and_zero_when_noise_dominates():
    gain = np.asarray([0.5, 0.5, 0.5])
    assert np.allclose(reliability_alpha(gain, np.zeros(3)), 1.0)
    assert np.allclose(reliability_alpha(gain, np.full(3, 0.25), 2.0), 0.0)
    middle = reliability_alpha(gain, np.full(3, 0.05), 2.0)
    assert np.allclose(middle, 1.0 - 2.0 * 0.1)


def test_alpha_is_zero_for_a_vanishing_gain():
    """A band with nothing to correct must not be amplified by a tiny divisor."""
    assert np.allclose(reliability_alpha(np.zeros(3), np.full(3, 1e-6)), 0.0)


def test_alpha_rejects_a_negative_multiplier_or_error():
    with pytest.raises(ValueError):
        reliability_alpha(np.ones(3), np.ones(3), -1.0)
    with pytest.raises(ValueError):
        reliability_alpha(np.ones(3), -np.ones(3))


def test_shrunk_gain_lies_between_identity_and_the_raw_gain():
    generator = np.random.default_rng(20260804)
    target = generator.uniform(2.0, 3.0, size=(50, N_BANDS))
    source = generator.uniform(0.8, 1.4, size=(50, N_BANDS))
    fitted = shrunk_band_gains(target, source)
    assert np.all(np.abs(fitted["log_gain"]) <= np.abs(fitted["raw_log_gain"]) + 1e-12)
    assert np.all((fitted["alpha"] >= 0.0) & (fitted["alpha"] <= 1.0))
    assert np.allclose(fitted["gain"], np.exp(fitted["log_gain"]))
    assert np.all(np.abs(fitted["log_gain"]) <= np.log(RF1U_HARD_CAP) + 1e-12)


def test_identical_populations_give_exact_identity_gains():
    energy = np.tile(np.linspace(1.0, 2.0, 30)[:, None], (1, N_BANDS))
    fitted = shrunk_band_gains(energy, energy)
    assert np.allclose(fitted["gain"], 1.0)


def test_zero_shrinkage_multiplier_returns_the_raw_gain():
    generator = np.random.default_rng(20260804)
    target = generator.uniform(2.0, 3.0, size=(50, N_BANDS))
    source = generator.uniform(0.8, 1.4, size=(50, N_BANDS))
    fitted = shrunk_band_gains(target, source, shrinkage_se=0.0)
    assert np.allclose(fitted["log_gain"], fitted["raw_log_gain"])


def test_fitted_scanner_gains_match_the_per_scanner_fit():
    generator = np.random.default_rng(20260804)
    target = generator.uniform(2.0, 3.0, size=(40, N_BANDS))
    source = generator.uniform(0.8, 1.4, size=(40, 5, N_BANDS))
    stacked = fitted_scanner_gains(target, source)
    assert stacked["gain"].shape == (5, N_BANDS)
    for index in range(5):
        single = shrunk_band_gains(target, source[:, index])
        assert np.allclose(stacked["gain"][index], single["gain"])
        assert np.allclose(stacked["alpha"][index], single["alpha"])


def test_default_shrinkage_is_the_frozen_two_standard_errors():
    assert RF1U_SHRINKAGE_SE == 2.0
