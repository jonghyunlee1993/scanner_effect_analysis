import numpy as np

from run_e1_native_aa_spectral_slide import batch_total_od_fft, central_256


def test_central_256_preserves_the_frozen_center():
    grid = np.arange(2 * 512 * 512 * 3, dtype=np.int64).reshape(2, 512, 512, 3)
    crop = central_256(grid)
    assert crop.shape == (2, 256, 256, 3)
    assert np.array_equal(crop[:, 0, 0], grid[:, 128, 128])
    assert np.array_equal(crop[:, -1, -1], grid[:, 383, 383])


def test_od_fft_removes_per_patch_dc_component():
    rgb = np.full((3, 8, 8, 3), 128, dtype=np.uint8)
    window = np.ones((8, 8), dtype=np.float32)
    fourier = batch_total_od_fft(rgb, window)
    assert fourier.shape == (3, 8, 8)
    assert np.allclose(fourier, 0.0, atol=5e-6)
