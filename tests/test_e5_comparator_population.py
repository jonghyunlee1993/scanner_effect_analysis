import numpy as np
import torch

from e5_comparator_population import (
    FEATURE_CONDITIONS,
    IMAGE_CONDITIONS,
    batch_radial_power,
    centered_covariance,
    coral_transform,
    feature_sufficient_statistics,
    fitted_frequency_gain,
    frequency_calibration,
    lab_to_rgb01,
    mean_std,
    od_affine_statistics,
    paired_od_affine,
    procrustes_transform,
    radial_geometry,
    reinhard_lab,
    rgb01_to_lab,
    solve_od_affine,
)
from extract_e5_image_features import heldout_parameters


def test_frozen_method_sets_are_complete():
    assert IMAGE_CONDITIONS == (
        "reinhard_lab",
        "paired_od_affine",
        "frequency_calibration",
    )
    assert FEATURE_CONDITIONS == ("coral", "orthogonal_procrustes")


def test_lab_roundtrip_and_identity_reinhard():
    generator = torch.Generator().manual_seed(4)
    rgb = torch.rand((3, 17, 19, 3), generator=generator) * 0.9 + 0.05
    recovered = lab_to_rgb01(rgb01_to_lab(rgb))
    assert torch.max(torch.abs(recovered - rgb)) < 3e-5
    lab = rgb01_to_lab(rgb)
    source_mean = lab.reshape(-1, 3).mean(0)
    source_std = lab.reshape(-1, 3).std(0, unbiased=False)
    report = reinhard_lab(rgb, source_mean, source_std, source_mean, source_std)
    assert torch.max(torch.abs(report["output"] - rgb)) < 3e-5
    assert torch.all(report["preclip_range_fraction"] == 0)


def test_moments_and_od_affine_recover_known_mapping():
    generator = torch.Generator().manual_seed(12)
    source = torch.rand((2, 16, 16, 3), generator=generator) * 0.7 + 0.15
    known = torch.tensor(
        [[1.05, 0.02, 0.01], [0.01, 0.92, 0.03], [0.02, 0.01, 1.08], [0.01, 0.02, 0.03]],
        dtype=torch.float64,
    )
    from e5_comparator_population import od_to_rgb01, rgb01_to_od

    od = rgb01_to_od(source).double()
    design = torch.cat([od, torch.ones_like(od[..., :1])], -1)
    target = od_to_rgb01(design @ known).float()
    xtx, xty, count = od_affine_statistics(source, target)
    fitted = solve_od_affine(xtx.numpy(), xty.numpy())
    assert count == 512
    assert np.allclose(fitted, known.numpy(), atol=3e-4)
    report = paired_od_affine(source, torch.from_numpy(fitted).float())
    assert torch.mean(torch.abs(report["output"] - target)) < 2e-5


def test_radial_gain_is_identity_for_equal_power_and_transform():
    generator = torch.Generator().manual_seed(7)
    rgb = torch.rand((2, 32, 32, 3), generator=generator) * 0.8 + 0.1
    geometry = radial_geometry(32)
    power = batch_radial_power(rgb, geometry).numpy()
    gain = fitted_frequency_gain(power, power, geometry.frequency, 2.0)
    assert np.allclose(gain, 1.0)
    report = frequency_calibration(rgb, gain, geometry.frequency)
    assert torch.max(torch.abs(report["output"] - rgb)) < 2e-5


def test_feature_statistics_covariance_and_identity_coral():
    generator = torch.Generator().manual_seed(18)
    base = torch.randn((20, 7), generator=generator)
    features = torch.stack([base + 0.1 * index for index in range(6)])
    sums, grams, cross, count = feature_sufficient_statistics(features)
    assert sums.shape == (6, 7)
    assert grams.shape == (6, 7, 7)
    assert cross.shape == (5, 7, 7)
    expected = torch.from_numpy(np.cov(features[0].numpy(), rowvar=False)).double()
    observed = centered_covariance(sums[0], grams[0], count)
    assert torch.allclose(observed, expected, atol=1e-10)
    transformed = coral_transform(
        features[0], sums[0], grams[0], sums[0], grams[0], count, 0.1
    )
    assert torch.allclose(transformed, features[0].double(), atol=2e-7)


def test_procrustes_recovers_orthogonal_map_without_scale():
    generator = torch.Generator().manual_seed(33)
    source = torch.randn((60, 6), generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn((6, 6), generator=generator, dtype=torch.float64))
    target = source @ q + torch.arange(6, dtype=torch.float64)
    features = torch.stack(
        [target, source, source, source, source, source], dim=0
    )
    sums, _, cross, count = feature_sufficient_statistics(features)
    heldout = torch.randn((9, 6), generator=generator, dtype=torch.float64)
    transformed = procrustes_transform(
        heldout, sums[1], sums[0], cross[0], count
    )
    assert torch.allclose(transformed, heldout @ q + torch.arange(6), atol=1e-9)


def test_mean_std_uses_population_moments_and_floor():
    mean, std = mean_std(np.array([3.0, 6.0]), np.array([5.0, 18.0]), 2)
    assert np.allclose(mean, [1.5, 3.0])
    assert np.allclose(std, [0.5, 0.001])


def test_image_parameters_subtract_the_heldout_slide():
    slides = 109
    lab_count = np.full((slides, 6), 10, dtype=np.int64)
    lab_sum = np.zeros((slides, 6, 3), dtype=np.float64)
    lab_square = np.full((slides, 6, 3), 10.0, dtype=np.float64)
    lab_sum[0] = 10.0
    lab_square[0] = 100.0
    od_xtx = np.broadcast_to(np.eye(4), (slides, 5, 4, 4)).copy()
    coefficient = np.zeros((4, 3), dtype=np.float64)
    coefficient[:3] = np.eye(3)
    od_xty = np.broadcast_to(coefficient, (slides, 5, 4, 3)).copy()
    radial = np.ones((slides, 6, 72), dtype=np.float64)
    statistics = {
        "slide_ids": np.asarray([f"slide-{index:03d}" for index in range(slides)]),
        "lab_sum": lab_sum,
        "lab_square_sum": lab_square,
        "lab_count": lab_count,
        "od_xtx": od_xtx,
        "od_xty": od_xty,
        "radial_power": radial,
        "radial_frequency": radial_geometry(32).frequency,
    }
    parameters = heldout_parameters(statistics, "slide-000", 2.0)
    assert parameters["train_slides"] == 108
    # The outlying held-out slide does not affect the train-fold mean.
    assert np.allclose(parameters["lab"][0][0], 0.0)
    assert np.allclose(parameters["frequency_gain"], 1.0)
    assert np.allclose(parameters["od"][0][:3], np.eye(3), atol=2e-6)
