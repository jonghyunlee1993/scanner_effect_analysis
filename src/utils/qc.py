"""Per-tile tissue, focus, and appearance-robust registration metrics.

All image inputs are RGB uint8 arrays. Registration confidence deliberately uses
low-pass luminance so scanner-specific colour and fine texture do not become a
selection criterion.
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity


def _gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def tissue_fraction(rgb):
    """Fraction of tissue pixels via Otsu on HSV saturation."""
    s = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    t, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((s > t).mean())


def focus_score(rgb):
    """Sharpness = variance of the Laplacian of gray."""
    return float(cv2.Laplacian(_gray(rgb), cv2.CV_64F).var())


def local_ncc(a_rgb, b_rgb, sigma=3.0):
    """Return low-pass luminance NCC in ``[-1, 1]``."""
    a = cv2.GaussianBlur(_gray(a_rgb).astype(np.float32), (0, 0), sigma)
    b = cv2.GaussianBlur(_gray(b_rgb).astype(np.float32), (0, 0), sigma)
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def local_ssim(a_rgb, b_rgb):
    """Structural similarity on grayscale."""
    return float(structural_similarity(_gray(a_rgb), _gray(b_rgb)))


def phase_shift(a_rgb, b_rgb):
    """Return residual registration shift magnitude in pixels."""
    return phase_correlation(a_rgb, b_rgb)[0]


def phase_correlation(a_rgb, b_rgb, sigma=3.0):
    """Return ``(shift_magnitude, response)`` from low-pass phase correlation."""
    a = cv2.GaussianBlur(_gray(a_rgb).astype(np.float32), (0, 0), sigma)
    b = cv2.GaussianBlur(_gray(b_rgb).astype(np.float32), (0, 0), sigma)
    (dx, dy), response = cv2.phaseCorrelate(a, b)
    return float(np.hypot(dx, dy)), float(response)


def registration_quality(a_rgb, b_rgb, sigma=3.0):
    """Return the v3 registration confidence and its diagnostic components.

    The geometric mean makes a high score require agreement between low-pass NCC
    and phase correlation while retaining a continuous supervision weight.
    """
    ncc_lp = local_ncc(a_rgb, b_rgb, sigma=sigma)
    residual_shift, phase_response = phase_correlation(a_rgb, b_rgb, sigma=sigma)
    q_ncc = np.clip((ncc_lp - 0.2) / 0.6, 0.0, 1.0)
    q_phase = np.clip((phase_response - 0.1) / 0.7, 0.0, 1.0)
    q_reg = np.sqrt(q_ncc * q_phase)
    return {
        "q_reg": float(q_reg),
        "ncc_lp": float(ncc_lp),
        "phase_response": float(phase_response),
        "residual_shift": float(residual_shift),
    }
