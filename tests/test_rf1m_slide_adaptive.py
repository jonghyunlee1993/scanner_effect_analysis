"""Tests for the RF1M slide-adaptive gain feasibility test."""

from __future__ import annotations

import numpy as np
import pytest

from analyze_rf1m_slide_adaptive import ESTIMATORS, fit_log_energy_line, predicted_target_log


def test_fit_recovers_a_known_line():
    x = np.linspace(-2.0, 2.0, 40)
    intercept, slope = fit_log_energy_line(x, 0.7 + 0.4 * x)
    assert np.isclose(intercept, 0.7)
    assert np.isclose(slope, 0.4)


def test_fit_returns_zero_slope_for_constant_source():
    x = np.full(10, 1.5)
    intercept, slope = fit_log_energy_line(x, np.linspace(0.0, 1.0, 10))
    assert slope == 0.0
    assert np.isclose(intercept, 0.5)


def test_fit_rejects_malformed_input():
    with pytest.raises(ValueError):
        fit_log_energy_line(np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError):
        fit_log_energy_line(np.zeros(5), np.zeros(4))
    with pytest.raises(ValueError):
        fit_log_energy_line(np.asarray([0.0, 1.0, np.nan]), np.zeros(3))


def test_pooled_estimator_shifts_the_slide_by_a_constant():
    train_source = np.asarray([0.0, 1.0, 2.0])
    train_target = np.asarray([1.0, 1.5, 3.0])
    source = np.asarray([0.5, 1.5])
    predicted = predicted_target_log("pooled", source, train_source, train_target)
    offset = predicted - source
    assert np.allclose(offset, offset[0])


def test_per_slide_estimator_ignores_the_slide():
    train_source = np.asarray([0.0, 1.0, 2.0])
    train_target = np.asarray([1.0, 1.5, 3.0])
    predicted = predicted_target_log(
        "per_slide", np.asarray([0.5, 5.0]), train_source, train_target
    )
    assert np.allclose(predicted, predicted[0])


def test_regression_reduces_to_pooled_when_the_slope_is_one():
    """b=1 makes the regression an additive offset, which is the pooled form."""
    train_source = np.asarray([0.0, 1.0, 2.0, 3.0])
    train_target = train_source + 0.8
    source = np.asarray([0.5, 2.5])
    regression = predicted_target_log("regression", source, train_source, train_target)
    assert np.allclose(regression, source + 0.8)


def test_regression_reduces_to_per_slide_when_the_slope_is_zero():
    train_source = np.asarray([0.0, 1.0, 2.0, 3.0])
    train_target = np.full(4, 1.25)
    source = np.asarray([0.5, 2.5])
    regression = predicted_target_log("regression", source, train_source, train_target)
    per_slide = predicted_target_log("per_slide", source, train_source, train_target)
    assert np.allclose(regression, per_slide)


def test_regression_beats_both_endpoints_on_an_intermediate_slope():
    generator = np.random.default_rng(20260803)
    source = generator.normal(0.0, 1.0, size=200)
    target = 0.5 + 0.5 * source + generator.normal(0.0, 0.05, size=200)
    errors = {
        name: predicted_target_log(name, source, source, target) - target
        for name in ESTIMATORS
    }
    rmse = {name: float(np.sqrt((value**2).mean())) for name, value in errors.items()}
    assert rmse["regression"] < rmse["pooled"]
    assert rmse["regression"] < rmse["per_slide"]


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError):
        predicted_target_log("magic", np.zeros(3), np.zeros(3), np.zeros(3))
