"""Fixed single-factor image perturbations shared by the paper analyses."""

from __future__ import annotations

import cv2
import numpy as np


SWEEPS = (
    ("Gaussian blur", "blur_sigma", (0.0, 0.5, 1.0, 2.0, 3.0), "High-band log2 vs AT2"),
    ("Unsharp mask", "sharpen_amount", (0.0, 0.25, 0.5, 1.0, 2.0), "High-band log2 vs AT2"),
    ("Gamma", "gamma", (1.0, 0.9, 0.75, 0.6, 0.5), "Mean-OD log2 vs AT2"),
)


def transform(source: np.ndarray, parameter: str, value: float) -> np.ndarray:
    """Apply one frozen sweep setting to a uint8 RGB image."""
    if parameter == "blur_sigma":
        return source.copy() if value == 0 else cv2.GaussianBlur(source, (0, 0), value)
    rgb = source.astype(np.float32) / 255.0
    if parameter == "sharpen_amount":
        lowpass = cv2.GaussianBlur(rgb, (0, 0), 1.0)
        rgb = rgb + value * (rgb - lowpass)
    elif parameter == "gamma":
        rgb = np.power(rgb, value)
    else:
        raise ValueError(parameter)
    return np.clip(np.rint(np.clip(rgb, 0, 1) * 255), 0, 255).astype(np.uint8)
