"""Dependency-light frozen spectral and glass-QC helpers for E0d rendering."""

from __future__ import annotations

import cv2
import numpy as np


QC_THRESHOLDS = {
    "gray_mean_min": 205.0,
    "gray_p05_min": 175.0,
    "gray_std_min": 0.35,
    "gray_std_max": 20.0,
    "saturation_p95_max": 50.0,
    "stained_fraction_max": 0.0002,
    "dark_fraction_max": 0.002,
    "dark_outlier_fraction_10_max": 0.003,
    "dark_outlier_fraction_20_max": 0.0005,
    "dark_component_fraction_15_max": 0.0003,
    "chromatic_outlier_fraction_15_max": 0.0002,
    "chromatic_component_fraction_15_max": 0.0001,
    "exact_white_fraction_max": 0.980,
    "flat_fraction_max": 0.995,
    "gradient_mean_max": 8.0,
}


def frequency_geometry(size: int, mpp: float, bins: int):
    one_d = np.fft.fftfreq(size) / mpp
    radius = np.sqrt(one_d[:, None] ** 2 + one_d[None, :] ** 2)
    nyquist = 1.0 / (2.0 * mpp)
    edges = np.linspace(0.0, nyquist, bins + 1)
    index = np.digitize(radius.ravel(), edges) - 1
    valid = (index >= 0) & (index < bins)
    counts = np.bincount(index[valid], minlength=bins).astype(np.float64)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return index, valid, counts, centres, nyquist


def radial_mean(value, index, valid, counts):
    return np.bincount(
        index[valid],
        weights=np.asarray(value).ravel()[valid],
        minlength=len(counts),
    ) / np.maximum(counts, 1.0)


def batch_total_od_fft(rgb: np.ndarray, window: np.ndarray):
    value = -np.log((np.asarray(rgb, dtype=np.float32) + 1.0) / 256.0).mean(
        axis=3
    )
    value -= value.mean(axis=(1, 2), keepdims=True)
    return np.fft.fft2(value * window[None, :, :], axes=(-2, -1))


def flat_fraction(rgb: np.ndarray, win: int = 5, thresh: float = 1.0):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (win, win))
    variance = cv2.blur(gray * gray, (win, win)) - mean * mean
    return float((np.sqrt(np.maximum(variance, 0.0)) < thresh).mean())


def patch_statistics(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    stained = (saturation >= 31) & (gray <= 220)
    median_gray = float(np.median(gray))
    median_rgb = np.median(rgb.reshape(-1, 3), axis=0)
    rgb_delta = rgb.astype(np.float32) - median_rgb[None, None, :]
    luminance_delta = rgb_delta.mean(axis=2, keepdims=True)
    chromatic_residual = np.sqrt(
        np.square(rgb_delta - luminance_delta).sum(axis=2)
    )
    dark_10 = gray < median_gray - 10.0
    dark_15 = gray < median_gray - 15.0
    dark_20 = gray < median_gray - 20.0
    chromatic_15 = chromatic_residual > 15.0

    def largest_component_fraction(mask):
        count, _, component_stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if count <= 1:
            return 0.0
        return float(component_stats[1:, cv2.CC_STAT_AREA].max() / mask.size)

    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "gray_p05": float(np.quantile(gray, 0.05)),
        "gray_p95": float(np.quantile(gray, 0.95)),
        "saturation_mean": float(saturation.mean()),
        "saturation_p95": float(np.quantile(saturation, 0.95)),
        "stained_fraction": float(stained.mean()),
        "dark_fraction": float((gray < 160).mean()),
        "dark_outlier_fraction_10": float(dark_10.mean()),
        "dark_outlier_fraction_20": float(dark_20.mean()),
        "dark_component_fraction_15": largest_component_fraction(dark_15),
        "chromatic_outlier_fraction_15": float(chromatic_15.mean()),
        "chromatic_component_fraction_15": largest_component_fraction(
            chromatic_15
        ),
        "exact_white_fraction": float((rgb >= 254).all(axis=2).mean()),
        "flat_fraction": flat_fraction(rgb, win=5, thresh=0.5),
        "gradient_mean": float(gradient.mean()),
        "gradient_p95": float(np.quantile(gradient, 0.95)),
    }


def glass_rejection_reasons(stats):
    comparisons = (
        (stats["gray_mean"] < QC_THRESHOLDS["gray_mean_min"], "dim_mean"),
        (stats["gray_p05"] < QC_THRESHOLDS["gray_p05_min"], "dim_tail"),
        (stats["gray_std"] < QC_THRESHOLDS["gray_std_min"], "too_flat_std"),
        (stats["gray_std"] > QC_THRESHOLDS["gray_std_max"], "high_std"),
        (
            stats["saturation_p95"] > QC_THRESHOLDS["saturation_p95_max"],
            "saturated_colour",
        ),
        (
            stats["stained_fraction"] > QC_THRESHOLDS["stained_fraction_max"],
            "stain",
        ),
        (
            stats["dark_fraction"] > QC_THRESHOLDS["dark_fraction_max"],
            "dark_pixels",
        ),
        (
            stats["dark_outlier_fraction_10"]
            > QC_THRESHOLDS["dark_outlier_fraction_10_max"],
            "local_dark_outliers",
        ),
        (
            stats["dark_outlier_fraction_20"]
            > QC_THRESHOLDS["dark_outlier_fraction_20_max"],
            "deep_dark_outliers",
        ),
        (
            stats["dark_component_fraction_15"]
            > QC_THRESHOLDS["dark_component_fraction_15_max"],
            "connected_dark_debris",
        ),
        (
            stats["chromatic_outlier_fraction_15"]
            > QC_THRESHOLDS["chromatic_outlier_fraction_15_max"],
            "chromatic_outliers",
        ),
        (
            stats["chromatic_component_fraction_15"]
            > QC_THRESHOLDS["chromatic_component_fraction_15_max"],
            "connected_chromatic_debris",
        ),
        (
            stats["exact_white_fraction"]
            > QC_THRESHOLDS["exact_white_fraction_max"],
            "clipped_white",
        ),
        (
            stats["flat_fraction"] > QC_THRESHOLDS["flat_fraction_max"],
            "synthetic_flat",
        ),
        (
            stats["gradient_mean"] > QC_THRESHOLDS["gradient_mean_max"],
            "edges_or_debris",
        ),
    )
    return [name for failed, name in comparisons if failed]
