import numpy as np
import pytest
import torch

from e4_control_population import (
    CONTROL_SPECS,
    PYRAMID_LEVELS,
    analysis_tensor_to_uint8,
    centered_crop,
    decompose_scanner_tensor,
    detail_rms,
    heldout_global_mean_bands,
    registered_loo_bands,
    render_control,
    requested_bands,
    slide_band_sums,
    operational_detail_rms,
    uint8_to_analysis_tensor,
)
from prenorm.exp01.frequency import FixedLaplacianPyramid


def test_frozen_condition_grid():
    assert len(CONTROL_SPECS) == 9
    assert [spec.value for spec in CONTROL_SPECS if spec.family == "gain"] == [
        0.75,
        0.5,
        0.25,
        0.0,
        1.25,
        1.5,
        2.0,
    ]
    assert CONTROL_SPECS[-2].family == "registered_loo_mix"
    assert CONTROL_SPECS[-2].value == 0.25
    assert CONTROL_SPECS[-1].family == "global_train_mix"
    assert CONTROL_SPECS[-1].value == 1.0


def test_center_crop_and_roundtrip():
    grid = np.zeros((6, 2, 512, 512, 3), dtype=np.uint8)
    grid[:, :, 128:384, 128:384] = 117
    cropped = centered_crop(grid, 256)
    assert cropped.shape == (6, 2, 256, 256, 3)
    tensor = uint8_to_analysis_tensor(cropped)
    assert np.array_equal(analysis_tensor_to_uint8(tensor), cropped)
    with pytest.raises(ValueError):
        centered_crop(grid, 128)


def test_registered_leave_one_scanner_mean():
    bands = [
        torch.arange(6, dtype=torch.float32)[:, None, None, None, None]
        .expand(6, 2, 3, 4, 4)
        for _ in range(PYRAMID_LEVELS)
    ]
    loo = registered_loo_bands(bands)
    assert torch.allclose(loo[0][0], torch.full_like(loo[0][0], 3.0))
    assert torch.allclose(loo[0][5], torch.full_like(loo[0][5], 2.0))


def test_heldout_global_mean_is_exact():
    total = [np.full((3, 4, 4), 70.0 + index) for index in range(4)]
    heldout = [np.full((3, 4, 4), 10.0 + index) for index in range(4)]
    means = heldout_global_mean_bands(total, heldout, 7, 1, "cpu")
    for index, mean in enumerate(means):
        assert torch.allclose(
            mean,
            torch.full_like(mean, (60.0) / 6.0),
        )


def test_gain_one_reconstructs_uint8_input():
    rng = np.random.default_rng(17)
    value = rng.integers(0, 256, size=(6, 2, 224, 224, 3), dtype=np.uint8)
    images = uint8_to_analysis_tensor(value)
    pyramid = FixedLaplacianPyramid(PYRAMID_LEVELS)
    coarse, bands = decompose_scanner_tensor(pyramid, images)
    reconstructed = render_control(pyramid, coarse, bands)["image"]
    error = (reconstructed - images).abs().max().item()
    assert error < 2e-6
    assert np.array_equal(analysis_tensor_to_uint8(reconstructed), value)
    requested = detail_rms(pyramid, coarse, bands)
    operational = operational_detail_rms(pyramid, reconstructed)
    assert torch.allclose(requested, operational, atol=2e-6, rtol=1e-5)


def test_control_requests_have_expected_limits():
    source = [torch.ones(6, 2, 3, 4, 4) for _ in range(4)]
    registered = [2.0 * value for value in source]
    global_mean = [torch.full((3, 4, 4), 3.0) for _ in range(4)]
    complete_blur = next(spec for spec in CONTROL_SPECS if spec.name.endswith("0p00"))
    oracle = next(spec for spec in CONTROL_SPECS if spec.family == "registered_loo_mix")
    negative = next(spec for spec in CONTROL_SPECS if spec.family == "global_train_mix")
    assert all(torch.count_nonzero(value) == 0 for value in requested_bands(complete_blur, source, registered, global_mean))
    assert torch.allclose(requested_bands(oracle, source, registered, global_mean)[0], torch.full_like(source[0], 1.25))
    assert torch.allclose(requested_bands(negative, source, registered, global_mean)[0], torch.full_like(source[0], 3.0))


def test_slide_band_sums_aggregate_scanners_and_locations():
    bands = [torch.ones(6, 2, 3, 4, 4) * (index + 1) for index in range(4)]
    sums = slide_band_sums(bands)
    assert np.all(sums[0] == 12)
    assert np.all(sums[3] == 48)
