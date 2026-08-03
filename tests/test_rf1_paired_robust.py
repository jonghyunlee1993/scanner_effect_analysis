import numpy as np
import torch

from analyze_rf1_paired_robust import (
    GAIN_CAP,
    batch_radial_power_per_image,
    fit_paired_robust_gains,
    huber_location,
    paired_anchor_log_ratio,
)
from e5_comparator_population import batch_radial_power, radial_geometry


def test_per_image_radial_power_sums_to_locked_batch_operation():
    generator = torch.Generator().manual_seed(7)
    rgb = torch.rand((3, 224, 224, 3), generator=generator)
    geometry = radial_geometry(224)
    per_image = batch_radial_power_per_image(rgb, geometry)
    pooled = batch_radial_power(rgb, geometry)
    assert per_image.shape == (3, 72)
    # The two implementations differ only in GPU/CPU reduction order.
    assert torch.allclose(per_image.sum(dim=0), pooled, rtol=1e-7, atol=1e-5)


def test_paired_log_ratio_is_anchor_centered_per_patch():
    frequency = np.linspace(0.01, 0.99, 72)
    source = np.ones((4, 72))
    target = np.exp(np.linspace(-0.4, 0.8, 72))[None] * np.ones((4, 1))
    ratio = paired_anchor_log_ratio(source, target, frequency)
    anchor = (frequency >= 0.03) & (frequency <= 0.10)
    assert np.allclose(ratio[:, anchor].mean(axis=1), 0.0, atol=1e-12)


def test_huber_location_rejects_single_extreme_slide():
    values = np.zeros((11, 3))
    values[-1] = 1000.0
    location = huber_location(values)
    assert np.max(np.abs(location)) < 1e-6


def test_reliability_shrinks_uncertain_effect_and_preserves_constraints():
    rng = np.random.default_rng(11)
    frequency = np.linspace(0.01, 0.99, 72)
    shape = (20, 12, 5, 72)
    stable = np.zeros(shape)
    stable[..., frequency >= 0.2] = 0.12
    stable += rng.normal(0.0, 0.004, shape)
    unstable = stable.copy()
    unstable[:, :, 1, frequency >= 0.2] += rng.normal(
        0.0, 0.8, (20, 12, int((frequency >= 0.2).sum()))
    )
    fit = fit_paired_robust_gains(unstable, frequency)
    selected = frequency >= 0.2
    assert fit["reliability"][0, selected].mean() > fit["reliability"][1, selected].mean()
    assert np.all(fit["paired_robust_gain"][:, frequency < 0.10] == 1.0)
    assert np.all(fit["paired_reliable_gain"][:, frequency < 0.10] == 1.0)
    assert fit["paired_robust_gain"].min() >= 1.0 / GAIN_CAP - 1e-12
    assert fit["paired_robust_gain"].max() <= GAIN_CAP + 1e-12
    assert np.abs(np.log(fit["paired_reliable_gain"][1, selected])).mean() < np.abs(
        np.log(fit["paired_robust_gain"][1, selected])
    ).mean()
