"""Gross scanner-to-reference alignment by tissue-mask overlap.

Every thumbnail is resampled to exactly 16 um/px from that scanner's own MPP, so
**the scale between two thumbnails is 1.0 by construction**.  All that is unknown
is a small rotation -- near 0, or near 180 for the Philips unit, which scans the
slide upside down -- and a translation.  The keypoint fit of Amendment 1
estimates four to eight degrees of freedom instead, which on a forty-six core
lattice leaves room for a wrong-but-consistent solution one 3.6 mm core pitch
off.  Matching whole tissue masks cannot make that mistake: a one-core shift
misaligns all forty-six cores at once, so the wrong solution is not a local
optimum.

This was written expecting it to rescue the three blocks that E9 loses entirely
-- SQ on GIVH and HRH, S60 on HRH.  **It does not, and the reason is worth
recording.**  Measured against the mask fit, the existing keypoint fit is right:
those blocks return mask Dice 0.93 and the two transforms agree to 40-52 um,
against 6-13 um on healthy blocks and a 3600 um core pitch.  The keypoint fit
never landed on the wrong core.  What fails is downstream of registration --
S60/HRH is an out-of-focus scan (high-band power down five to tenfold, position
correct to 4-11 um), and SQ/GIVH and SQ/HRH do not match the reference at patch
scale under any offset searched.  See Amendment 2 of the contract.

So what this module actually earns is a *number*, not a repair: the Dice between
two section outlines, which is what separates "the transform is in the wrong
place" from "the transform is in the right place and the pixels still disagree".
That distinction is the whole diagnosis, and nothing else in the pipeline
measures it.  It is also cheap insurance -- matches that disagree with the mask
fit are dropped before RANSAC, which on healthy blocks moves the answer by
0.0-2.4 um, so the guard costs nothing where it is not needed.

Contract: docs/e9_plism_native_ert_contract.md, Amendment 2.
"""

from __future__ import annotations

import numpy as np

from extract_plism_native_psd import TISSUE_OD_FLOOR

# Rotation is either near 0 or near 180; nothing else is physically possible for
# a slide on a stage.  Coarse pass locates which, fine pass pins it down.
ROTATION_COARSE = np.concatenate([np.arange(-5.0, 5.01, 1.0), np.arange(175.0, 185.01, 1.0)])
ROTATION_FINE = 0.05
ROTATION_SPAN = 0.6

# A match may disagree with the mask fit by this much and still be believed.
# The core pitch is 3.6 mm = 225 thumbnail pixels, so this cannot admit a
# one-core-off cluster while still keeping honest keypoints.
GUIDE_TOLERANCE_PX = 12.0
MIN_GUIDED_MATCHES = 8

# Below this the two sections do not overlap at all and no transform is trusted.
MIN_DICE = 0.55


def white_level(thumb: np.ndarray) -> float:
    """Brightness of the glass, taken as the mode of the bright half.

    A high percentile is the usual choice and works for six of the seven
    scanners, but the Philips unit carries saturated pixels that drag its 99th
    percentile to 254 while the glass itself sits near 220.  Optical density is
    then inflated everywhere and 86% of the slide is called tissue, against 28-31%
    for the others.  Glass covers most of a TMA slide, so the histogram peak of
    the bright half *is* the glass, and it is the same rule for every scanner.
    """
    bright = thumb[thumb >= np.median(thumb)]
    if bright.size == 0:
        return max(float(np.percentile(thumb, 99.0)), 1.0)
    counts, edges = np.histogram(bright, bins=64)
    peak = int(np.argmax(counts))
    return max(float(0.5 * (edges[peak] + edges[peak + 1])), 1.0)


def tissue_mask(thumb: np.ndarray) -> np.ndarray:
    """The contract's rule: mean optical density above a fixed floor."""
    optical_density = -np.log10(np.clip(thumb.astype(np.float32), 1.0, None)
                                / white_level(thumb))
    return (optical_density > TISSUE_OD_FLOOR).astype(np.float32)


def rotate_into_canvas(mask: np.ndarray, rotation: float) -> tuple[np.ndarray, np.ndarray]:
    """Mask rotated about its own centre, in a canvas wide enough to hold it."""
    import cv2

    height, width = mask.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -rotation, 1.0)
    diagonal = int(np.ceil(np.hypot(width, height)))
    matrix[0, 2] += (diagonal - width) / 2
    matrix[1, 2] += (diagonal - height) / 2
    rotated = cv2.warpAffine(mask, matrix, (diagonal, diagonal), flags=cv2.INTER_NEAREST)
    return rotated, matrix


def _spectrum(fixed: np.ndarray, shape: tuple[int, int], cache: dict) -> dict:
    height = int(2 ** np.ceil(np.log2(max(shape[0], fixed.shape[0]) * 2)))
    width = int(2 ** np.ceil(np.log2(max(shape[1], fixed.shape[1]) * 2)))
    if (height, width) not in cache:
        padded = np.zeros((height, width), dtype=np.float32)
        padded[: fixed.shape[0], : fixed.shape[1]] = fixed
        cache[(height, width)] = {"fft": np.fft.rfft2(padded), "shape": (height, width),
                                  "sum": float(fixed.sum())}
    return cache[(height, width)]


def best_shift(moving: np.ndarray, spectrum: dict) -> tuple[np.ndarray, float]:
    """Translation maximising binary overlap, by FFT cross-correlation."""
    height, width = spectrum["shape"]
    padded = np.zeros((height, width), dtype=np.float32)
    padded[: moving.shape[0], : moving.shape[1]] = moving
    correlation = np.fft.irfft2(spectrum["fft"] * np.conj(np.fft.rfft2(padded)), s=(height, width))
    peak_y, peak_x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    overlap = float(correlation[peak_y, peak_x])
    dy = peak_y - height if peak_y > height // 2 else peak_y
    dx = peak_x - width if peak_x > width // 2 else peak_x
    total = spectrum["sum"] + float(padded.sum())
    return np.array([dx, dy], dtype=np.float64), (2.0 * overlap / total if total > 0 else 0.0)


def gross_transform(moving_thumb: np.ndarray, fixed_thumb: np.ndarray) -> dict:
    """Rigid transform taking `moving` thumbnail coordinates into `fixed`.

    Scale is fixed at 1.0 -- both thumbnails are already at 16 um/px -- so only
    rotation and translation are searched.
    """
    fixed = tissue_mask(fixed_thumb)
    moving = tissue_mask(moving_thumb)
    cache: dict = {}

    def evaluate(rotation: float) -> dict:
        rotated, matrix = rotate_into_canvas(moving, rotation)
        shift, dice = best_shift(rotated, _spectrum(fixed, rotated.shape, cache))
        full = np.array(matrix, dtype=np.float64)
        full[0, 2] += shift[0]
        full[1, 2] += shift[1]
        return {"rotation": float(rotation), "shift": shift, "dice": dice, "matrix": full}

    best = max((evaluate(r) for r in ROTATION_COARSE), key=lambda e: e["dice"])
    fine = np.arange(best["rotation"] - ROTATION_SPAN, best["rotation"] + ROTATION_SPAN + 1e-9,
                     ROTATION_FINE)
    best = max((evaluate(r) for r in fine), key=lambda e: e["dice"])
    best["ok"] = best["dice"] >= MIN_DICE
    return best


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ np.asarray(matrix)[:, :2].T + np.asarray(matrix)[:, 2]


def guided_matches(source: np.ndarray, target: np.ndarray, guide: np.ndarray,
                   tolerance: float = GUIDE_TOLERANCE_PX):
    """Keypoint matches that agree with the mask fit, and how many were dropped."""
    flat_source = source.reshape(-1, 2).astype(np.float64)
    flat_target = target.reshape(-1, 2).astype(np.float64)
    predicted = apply_affine(guide, flat_source)
    keep = np.hypot(*(predicted - flat_target).T) <= tolerance
    return source[keep], target[keep], int(keep.sum()), int((~keep).sum())
