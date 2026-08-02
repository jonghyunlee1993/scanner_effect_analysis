import numpy as np

from utils.align import pad_frac


def test_pad_frac_detects_constant_missing_region():
    reference = np.full((64, 64, 3), (130, 60, 100), dtype=np.uint8)
    crop = reference.copy()
    crop[:, :32] = (220, 228, 226)

    assert pad_frac(crop, reference) > 0.45


def test_pad_frac_ignores_low_amplitude_textured_tissue():
    reference = np.full((64, 64, 3), (130, 60, 100), dtype=np.uint8)
    yy, xx = np.indices((64, 64))
    texture = ((xx + 2 * yy) % 3).astype(np.uint8)
    crop = np.stack((215 + texture, 218 + texture, 220 + texture), axis=-1)

    assert pad_frac(crop, reference) < 0.01
