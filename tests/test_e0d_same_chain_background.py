import numpy as np

from e0d_same_chain_helpers import (
    batch_total_od_fft as e0d_batch_total_od_fft,
    glass_rejection_reasons as e0d_glass_rejection_reasons,
    patch_statistics as e0d_patch_statistics,
)
from render_e0d_same_chain_background_slide import (
    mapped_target_top_left,
    spectrum_rows,
)
from analyze_e0d_same_chain_noise_floor import (
    merge_tissue_background,
    noise_floor_band_sensitivity,
)
from run_e1_native_aa_spectral_slide import batch_total_od_fft as e1_batch_total_od_fft
from run_exp06_raw_nps_pilot import (
    glass_rejection_reasons as exp07_glass_rejection_reasons,
    patch_statistics as exp07_patch_statistics,
)


def test_mapped_target_top_left_uses_native_patch_centre():
    matrix = np.array(
        [[0.5, 0.0, 10.0], [0.0, 0.5, -4.0], [0.0, 0.0, 1.0]]
    )
    x, y, center_x, center_y = mapped_target_top_left(matrix, 100, 200, 512)
    assert (center_x, center_y) == (188.0, 224.0)
    assert (x, y) == (60, 96)


def test_spectrum_rows_match_frozen_shape_and_replicates():
    rng = np.random.default_rng(7)
    images = rng.integers(180, 256, size=(100, 256, 256, 3), dtype=np.uint8)
    spectra, replicates = spectrum_rows(images, "at2", "slide", 72)
    assert len(spectra) == 72
    assert len(replicates) == 5 * 72
    assert {row["patches_in_replicate"] for row in replicates} == {20}
    assert all(row["background_radial_power"] >= 0 for row in spectra)


def test_dependency_light_helpers_equal_frozen_e1_and_exp07_definitions():
    rng = np.random.default_rng(12)
    images = rng.integers(0, 256, size=(3, 32, 32, 3), dtype=np.uint8)
    window = np.hanning(32).astype(np.float32)
    window = window[:, None] * window[None, :]
    assert np.allclose(
        e0d_batch_total_od_fft(images, window),
        e1_batch_total_od_fft(images, window),
    )
    image = images[0]
    observed = e0d_patch_statistics(image)
    expected = exp07_patch_statistics(image)
    assert observed.keys() == expected.keys()
    assert all(np.isclose(observed[key], expected[key]) for key in observed)
    assert e0d_glass_rejection_reasons(observed) == exp07_glass_rejection_reasons(
        expected
    )


def synthetic_spectra(background_fraction=0.1):
    frequency = np.linspace(0.01, 0.98, 72)
    tissue_rows = []
    background_rows = []
    for slide_id in [str(index) for index in range(109)]:
        for scanner in ["at2", "gt450", "versa", "akoya", "s60", "s360"]:
            scanner_scale = 2.0 if scanner == "gt450" else 1.0
            for freq in frequency:
                tissue_power = scanner_scale * (1.0 + freq)
                tissue_rows.append(
                    {
                        "slide_id": slide_id,
                        "scanner": scanner,
                        "frequency_cyc_per_um": freq,
                        # A frequency-constant scanner scale is removed by the
                        # frozen anchor normalization.
                        "relative_transfer": 1.0,
                        "log2_relative_transfer": 0.0,
                        "coherence_to_at2": 1.0,
                        "radial_power": tissue_power,
                    }
                )
                background_rows.append(
                    {
                        "slide_id": slide_id,
                        "scanner": scanner,
                        "frequency_cyc_per_um": freq,
                        "tissue_type": "normal",
                        "background_radial_power": background_fraction * tissue_power,
                    }
                )
    return np.array(tissue_rows, dtype=object), np.array(background_rows, dtype=object)


def test_noise_floor_sensitivity_preserves_ratio_for_proportional_background():
    import pandas as pd

    tissue, background = synthetic_spectra(background_fraction=0.05)
    merged = merge_tissue_background(
        pd.DataFrame(tissue.tolist()), pd.DataFrame(background.tolist())
    )
    rows = noise_floor_band_sensitivity(merged)
    gt450 = rows[
        (rows["scanner"] == "gt450")
        & (rows["band"] == "high")
        & (rows["minimum_snr_threshold"] == 10.0)
    ]
    assert gt450["eligible"].all()
    assert np.allclose(gt450["delta_corrected_minus_raw_log2"], 0.0, atol=1e-12)


def test_noise_floor_sensitivity_reports_nonpositive_subtraction():
    import pandas as pd

    tissue, background = synthetic_spectra(background_fraction=1.1)
    merged = merge_tissue_background(
        pd.DataFrame(tissue.tolist()), pd.DataFrame(background.tolist())
    )
    assert merged["negative_after_subtraction"].all()
    rows = noise_floor_band_sensitivity(merged)
    assert not rows["eligible"].any()
    assert rows["noise_corrected_log2_relative_transfer"].isna().all()
