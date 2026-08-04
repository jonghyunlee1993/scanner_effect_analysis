import numpy as np
import torch

from e5_comparator_population import radial_geometry, rgb01_to_od
from e5_reinhard_residual_frequency import (
    OD_RGB8_MAX,
    RF1_GAIN_CAP_CANDIDATES,
    RF1_VERSION,
    fold_assignments,
    fold_counts,
    log_spectrum_rmse,
    shared_od_residual_frequency,
    shared_od_residual_frequency_many,
)


def test_rf1_frozen_identifiers_and_balanced_folds():
    slide_ids = [f"slide_{index:03d}" for index in range(109)]
    first = fold_assignments(slide_ids)
    second = fold_assignments(list(reversed(slide_ids)))
    assert first == second
    assert sorted(fold_counts(first)) == [21, 22, 22, 22, 22]
    assert RF1_VERSION == "e5_rf1_reinhard_residual_frequency_v1"
    assert RF1_GAIN_CAP_CANDIDATES[0] == 1.01
    assert RF1_GAIN_CAP_CANDIDATES[-1] == 4.0


def test_identity_gain_is_identity_without_projection():
    rng = np.random.default_rng(7)
    rgb = torch.tensor(rng.uniform(0.05, 0.95, size=(3, 32, 32, 3)), dtype=torch.float32)
    geometry = radial_geometry(32)
    report = shared_od_residual_frequency(
        rgb, np.ones_like(geometry.frequency), geometry.frequency
    )
    assert torch.allclose(report["output"], rgb, atol=2e-6, rtol=0)
    assert torch.max(report["projection_fraction"]) == 0
    assert torch.max(report["final_clamp_mae"]) < 1e-8


def test_gamut_projection_preserves_od_chromatic_differences():
    y, x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    pattern = ((x + y) % 2).float()
    rgb = torch.stack(
        [0.92 - 0.70 * pattern, 0.80 - 0.55 * pattern, 0.68 - 0.40 * pattern], dim=-1
    )[None]
    geometry = radial_geometry(32)
    gain = np.full_like(geometry.frequency, 4.0)
    gain[geometry.frequency < 0.10] = 1.0
    report = shared_od_residual_frequency(rgb, gain, geometry.frequency)
    output = report["output"]
    assert torch.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
    input_od = rgb01_to_od(rgb)
    output_od = rgb01_to_od(output)
    assert torch.allclose(
        input_od[..., 0] - input_od[..., 1],
        output_od[..., 0] - output_od[..., 1],
        atol=2e-5,
        rtol=0,
    )
    assert float(output_od.min()) >= -1e-6
    assert float(output_od.max()) <= OD_RGB8_MAX + 1e-5


def test_vectorized_gain_grid_matches_single_candidate_rendering():
    rng = np.random.default_rng(21)
    rgb = torch.tensor(rng.uniform(0.01, 0.99, size=(2, 32, 32, 3)), dtype=torch.float32)
    geometry = radial_geometry(32)
    gains = np.stack(
        [
            np.ones_like(geometry.frequency),
            np.linspace(0.91, 1.09, len(geometry.frequency)),
        ]
    )
    many = shared_od_residual_frequency_many(rgb, gains, geometry.frequency)
    for index, gain in enumerate(gains):
        single = shared_od_residual_frequency(rgb, gain, geometry.frequency)
        for name in (
            "output",
            "proposed_delta",
            "projected_delta",
            "material_range_fraction",
            "projection_rgb_mae",
            "final_clamp_mae",
        ):
            assert torch.allclose(many[name][index], single[name], atol=1e-7, rtol=0)


def test_log_spectrum_rmse_is_zero_for_identical_curves():
    frequency = np.linspace(0.01, 1.0, 72)
    power = np.linspace(1.0, 2.0, 72)
    assert log_spectrum_rmse(power, power, frequency) == 0.0
