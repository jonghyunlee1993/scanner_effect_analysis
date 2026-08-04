"""Contract tests for the E5-RF1M combined multiscale/no-harm candidate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from analyze_rf1_multiscale import laplacian_pyramid, shared_od_multiscale
from build_rf1m_cell import cell_name, directional_pairs, selected_split
from e5_comparator_population import FOVS, SCANNERS, rgb01_to_od
from e5_reinhard_residual_frequency import RF1_FOLDS, RF1_GAIN_CAP_CANDIDATES
from rf1m_combined import (
    RF1M_CAP_CANDIDATES,
    RF1M_IDENTITY_CAP,
    RF1M_SIGMAS,
    accumulate_band_energy,
    band_energy,
    candidate_band_gains,
    fitted_band_gains,
    shared_od_multiscale_many,
)


N_SOURCES = len(SCANNERS) - 1
N_BANDS = len(RF1M_SIGMAS)


def random_rgb(count: int = 3, size: int = 32, seed: int = 20260803) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return 0.15 + 0.7 * torch.rand(count, size, size, 3, generator=generator, dtype=torch.float64)


def test_candidate_caps_start_at_identity_and_extend_the_frozen_rf1_grid():
    assert RF1M_CAP_CANDIDATES[0] == RF1M_IDENTITY_CAP
    assert RF1M_CAP_CANDIDATES[1:] == tuple(RF1_GAIN_CAP_CANDIDATES)
    assert list(RF1M_CAP_CANDIDATES) == sorted(RF1M_CAP_CANDIDATES)


def test_laplacian_bands_and_base_reconstruct_the_mean_od_exactly():
    mean_od = rgb01_to_od(random_rgb()).mean(dim=-1)
    bands, base = laplacian_pyramid(mean_od, RF1M_SIGMAS)
    reconstructed = base + sum(bands)
    assert torch.allclose(reconstructed, mean_od, atol=1e-12)


def test_band_energy_matches_the_per_band_mean_square():
    mean_od = rgb01_to_od(random_rgb()).mean(dim=-1)
    bands, _ = laplacian_pyramid(mean_od, RF1M_SIGMAS)
    energy = band_energy(mean_od, RF1M_SIGMAS)
    assert energy.shape == (mean_od.shape[0], N_BANDS)
    for index, band in enumerate(bands):
        assert torch.allclose(energy[:, index], band.square().mean(dim=(1, 2)), atol=1e-12)


def test_accumulate_band_energy_sums_over_the_batch():
    rgb = random_rgb(count=4)
    total, count = accumulate_band_energy(rgb)
    assert count == 4
    per_patch = band_energy(rgb01_to_od(rgb).mean(dim=-1), RF1M_SIGMAS).double().numpy()
    assert np.allclose(total, per_patch.sum(axis=0), atol=1e-12)


def test_fitted_band_gains_are_the_capped_energy_ratio():
    source = np.full((N_SOURCES, N_BANDS), 4.0)
    target = np.full(N_BANDS, 9.0)
    uncapped = fitted_band_gains(source, target, 4.0)
    assert np.allclose(uncapped, 1.5)
    capped = fitted_band_gains(source, target, 1.25)
    assert np.allclose(capped, 1.25)
    attenuating = fitted_band_gains(np.full((N_SOURCES, N_BANDS), 9.0), np.full(N_BANDS, 4.0), 1.25)
    assert np.allclose(attenuating, 1.0 / 1.25)


def test_identity_cap_gives_unit_gains_regardless_of_energy():
    generator = np.random.default_rng(20260803)
    source = generator.uniform(0.1, 5.0, size=(N_SOURCES, N_BANDS))
    target = generator.uniform(0.1, 5.0, size=N_BANDS)
    assert np.allclose(fitted_band_gains(source, target, RF1M_IDENTITY_CAP), 1.0)
    assert np.allclose(candidate_band_gains(source, target)[0], 1.0)


def test_candidate_band_gains_requires_ascending_caps_starting_at_identity():
    source = np.full((N_SOURCES, N_BANDS), 4.0)
    target = np.full(N_BANDS, 9.0)
    with pytest.raises(ValueError):
        candidate_band_gains(source, target, (1.25, 1.5))
    with pytest.raises(ValueError):
        candidate_band_gains(source, target, (1.0, 1.5, 1.25))


def test_fitted_band_gains_reject_malformed_inputs():
    source = np.full((N_SOURCES, N_BANDS), 4.0)
    target = np.full(N_BANDS, 9.0)
    with pytest.raises(ValueError):
        fitted_band_gains(source[:, :-1], target, 1.25)
    with pytest.raises(ValueError):
        fitted_band_gains(source, target[:-1], 1.25)
    with pytest.raises(ValueError):
        fitted_band_gains(source, target, 0.9)
    with pytest.raises(ValueError):
        fitted_band_gains(np.full((N_SOURCES, N_BANDS), -1.0), target, 1.25)


def test_identity_candidate_reproduces_the_input_and_never_projects():
    rgb = random_rgb()
    gains = np.ones((1, N_BANDS))
    report = shared_od_multiscale_many(rgb, gains)
    assert torch.allclose(report["output"][0], rgb, atol=1e-9)
    assert float(report["projected_delta"].abs().max()) < 1e-9
    assert float(report["material_range_fraction"].max()) == 0.0
    assert float(report["final_clamp_mae"].max()) == 0.0


def test_many_matches_the_single_candidate_renderer():
    rgb = random_rgb()
    gains = np.asarray([[1.0, 1.0, 1.0], [1.2, 1.1, 1.05], [0.9, 0.8, 0.75]])
    report = shared_od_multiscale_many(rgb, gains)
    for index, gain in enumerate(gains):
        single = shared_od_multiscale(rgb, gain)
        assert torch.allclose(report["output"][index], single["output"], atol=1e-12)
        assert torch.allclose(
            report["projection_rgb_mae"][index], single["projection_rgb_mae"], atol=1e-12
        )


def test_output_stays_inside_the_rgb_cube_for_extreme_gains():
    rgb = random_rgb()
    gains = np.asarray([[4.0, 4.0, 4.0], [0.25, 0.25, 0.25]])
    report = shared_od_multiscale_many(rgb, gains)
    assert float(report["output"].min()) >= 0.0
    assert float(report["output"].max()) <= 1.0
    assert float(report["final_clamp_mae"].max()) <= 1e-6


def test_shared_residual_preserves_chromatic_od_differences():
    rgb = random_rgb()
    gains = np.asarray([[1.3, 1.2, 1.1]])
    report = shared_od_multiscale_many(rgb, gains)
    before = rgb01_to_od(rgb)
    after = rgb01_to_od(report["output"][0].clamp(1e-8, 1.0))
    difference = (after - after.mean(dim=-1, keepdim=True)) - (
        before - before.mean(dim=-1, keepdim=True)
    )
    assert float(difference.abs().max()) < 1e-6


def test_shared_od_multiscale_many_rejects_malformed_inputs():
    rgb = random_rgb()
    with pytest.raises(ValueError):
        shared_od_multiscale_many(rgb, np.ones((2, N_BANDS + 1)))
    with pytest.raises(ValueError):
        shared_od_multiscale_many(rgb, np.zeros((1, N_BANDS)))
    with pytest.raises(ValueError):
        shared_od_multiscale_many(rgb[0], np.ones((1, N_BANDS)))


def test_outer_task_indices_cover_every_fov_and_fold_once():
    splits = [selected_split("outer", index) for index in range(len(FOVS) * RF1_FOLDS)]
    assert len(set(splits)) == len(splits)
    assert {fov for fov, _ in splits} == set(FOVS)
    for fov in FOVS:
        folds = sorted(excluded[0] for value, excluded in splits if value == fov)
        assert folds == list(range(RF1_FOLDS))
    with pytest.raises(ValueError):
        selected_split("outer", len(FOVS) * RF1_FOLDS)


def test_inner_task_indices_cover_every_unordered_pair_once():
    tasks = len(FOVS) * 10
    splits = [selected_split("inner", index) for index in range(tasks)]
    assert len(set(splits)) == tasks
    for fov in FOVS:
        pairs = sorted(excluded for value, excluded in splits if value == fov)
        assert len(pairs) == 10
        assert all(first < second for first, second in pairs)
    with pytest.raises(ValueError):
        selected_split("inner", tasks)


def test_directional_pairs_exclude_the_validation_fold_from_training():
    assert directional_pairs("outer", (2,)) == ((2, 2),)
    assert directional_pairs("inner", (1, 3)) == ((1, 3), (3, 1))


def test_cell_names_separate_the_two_stages():
    assert cell_name(256, "outer", 2, 2) == "fov_256_outer_2"
    assert cell_name(256, "inner", 2, 3) == "fov_256_outer_2_inner_3"


def test_k_se_rule_shrinks_further_toward_identity_than_one_se():
    """Amendment 1: two standard errors must pick a weaker cap than one.

    Reproduces the S360 outer-fold-4 shape that failed the frozen gate: every
    candidate up to cap_1.1 is eligible, the mean-delta curve is steep, and the
    one-SE threshold lands between cap_1.05 and cap_1.1.
    """
    from analyze_rf1_noharm_gain import choose_candidate_one_se
    from rf1m_combined import choose_candidate_k_se

    caps = np.asarray([1.0, 1.05, 1.1])
    base = np.zeros(4)
    # Observed shape: cap_1.1 mean delta -0.0818 with SE 0.0219, cap_1.05 -0.0574.
    strong = np.asarray([-0.03090, -0.06483, -0.09877, -0.13270])
    candidate = np.stack([np.zeros(4), np.full(4, -0.0574), strong], axis=1)
    gamut = np.ones_like(candidate, dtype=bool)
    assert np.isclose(strong.mean(), -0.0818, atol=1e-4)
    assert np.isclose(strong.std(ddof=1) / 2.0, 0.0219, atol=1e-4)

    one_se = choose_candidate_one_se(caps, base, candidate, gamut)[0]
    two_se = choose_candidate_k_se(caps, base, candidate, gamut, 2.0)[0]
    assert one_se == 2
    assert two_se == 1
    assert caps[two_se] < caps[one_se]


def test_k_se_rule_keeps_the_one_se_result_when_the_multiplier_is_one():
    from analyze_rf1_noharm_gain import choose_candidate_one_se
    from rf1m_combined import choose_candidate_k_se

    caps = np.asarray([1.0, 1.05, 1.1, 1.25])
    base = np.zeros(4)
    generator = np.random.default_rng(20260803)
    candidate = np.sort(
        -np.abs(generator.normal(0.05, 0.01, size=(4, 4))).cumsum(axis=1), axis=1
    )
    candidate[:, 0] = 0.0
    gamut = np.ones_like(candidate, dtype=bool)
    assert (
        choose_candidate_k_se(caps, base, candidate, gamut, 1.0)[0]
        == choose_candidate_one_se(caps, base, candidate, gamut)[0]
    )


def test_k_se_rule_rejects_a_negative_multiplier():
    from rf1m_combined import choose_candidate_k_se

    caps = np.asarray([1.0, 1.05])
    base = np.zeros(4)
    candidate = np.stack([np.zeros(4), np.full(4, -0.01)], axis=1)
    gamut = np.ones_like(candidate, dtype=bool)
    with pytest.raises(ValueError):
        choose_candidate_k_se(caps, base, candidate, gamut, -1.0)


def test_identity_candidate_power_is_the_uncorrected_reinhard_base():
    """Identity must compare exactly against the base, not its float32 re-render.

    The cell stores an explicitly rendered identity candidate whose radial power
    agrees with the base only to float32 rounding.  Selecting on that render
    would make identity fail the 1e-12 non-worse tolerance against itself.
    """
    from analyze_rf1m_selection import candidate_power

    base = np.arange(3 * 8, dtype=np.float64).reshape(3, 8) + 1.0
    rendered = np.stack([base * (1.0 + 3e-8), base * 1.5, base * 2.0], axis=0)
    cell = {"validation_base_power": base, "validation_output_power": rendered}
    for scanner_index in range(3):
        assert np.array_equal(
            candidate_power(cell, 0, scanner_index), base[scanner_index]
        )
        assert not np.array_equal(
            candidate_power(cell, 0, scanner_index), rendered[0, scanner_index]
        )
        assert np.array_equal(
            candidate_power(cell, 2, scanner_index), rendered[2, scanner_index]
        )
