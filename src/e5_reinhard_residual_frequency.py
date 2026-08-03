"""Pure operations for the post-core E5-RF1 image harmonization extension."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from e5_comparator_population import (
    GAIN_CAP_CANDIDATES,
    RADIAL_BINS,
    SCANNERS,
    TARGET_MPP,
    fitted_frequency_gain,
    od_to_rgb01,
    radial_gain_map,
    rgb01_to_od,
)


RF1_VERSION = "e5_rf1_reinhard_residual_frequency_v1"
RF1_CONDITION = "reinhard_residual_frequency"
RF1_FOLDS = 5
RF1_FOLD_SALT = "e5_rf1_fold_v1:"
RF1_GAIN_CAP_CANDIDATES = GAIN_CAP_CANDIDATES
MATERIAL_EXCURSION = 1.0 / 255.0
OD_RGB8_MAX = float(np.log(256.0))


def fold_assignments(slide_ids: Sequence[str]) -> dict[str, int]:
    """Return balanced outcome-blind folds from salted slide-id hashes."""
    values = [str(value) for value in slide_ids]
    if len(values) != len(set(values)):
        raise ValueError("slide ids must be unique")
    ordered = sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{RF1_FOLD_SALT}{value}".encode()).hexdigest(),
            value,
        ),
    )
    return {slide_id: index % RF1_FOLDS for index, slide_id in enumerate(ordered)}


def fold_counts(assignments: dict[str, int]) -> tuple[int, ...]:
    return tuple(sum(fold == value for fold in assignments.values()) for value in range(RF1_FOLDS))


def fitted_residual_gains(
    source_power: np.ndarray,
    target_power: np.ndarray,
    frequency: np.ndarray,
    cap: float,
) -> np.ndarray:
    """Fit one post-Reinhard residual gain for each non-AT2 source scanner."""
    source = np.asarray(source_power, dtype=np.float64)
    target = np.asarray(target_power, dtype=np.float64)
    if source.shape != (len(SCANNERS) - 1, RADIAL_BINS):
        raise ValueError(f"invalid post-Reinhard source power shape: {source.shape}")
    if target.shape != (RADIAL_BINS,):
        raise ValueError(f"invalid AT2 target power shape: {target.shape}")
    return np.stack(
        [fitted_frequency_gain(row, target, frequency, cap) for row in source], axis=0
    )


def _batch_metrics(value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2:
        raise ValueError("patch metric input must include batch and spatial dimensions")
    return value.float().mean(dim=tuple(range(1, value.ndim)))


def shared_od_residual_frequency(
    rgb01: torch.Tensor,
    gain: np.ndarray,
    frequency: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Apply a radial mean-OD residual and project it to exact RGB8 OD support.

    The same scalar residual is applied to all three OD channels. Projecting that
    scalar, instead of clipping RGB channels independently, preserves the two
    within-pixel chromatic OD differences.
    """
    report = shared_od_residual_frequency_many(
        rgb01, np.asarray(gain, dtype=np.float64)[None], frequency
    )
    return {name: value[0] for name, value in report.items()}


def shared_od_residual_frequency_many(
    rgb01: torch.Tensor,
    gains: np.ndarray,
    frequency: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Vectorized RF1 rendering for a candidate-cap gain grid.

    Returned tensors have candidate as the first dimension and batch as the
    second dimension. This avoids recomputing the input FFT for every cap in the
    input-only audit.
    """
    if rgb01.ndim != 4 or rgb01.shape[-1] != 3 or rgb01.shape[1] != rgb01.shape[2]:
        raise ValueError("RF1 correction requires NxHxWx3 square RGB")
    if not torch.isfinite(rgb01).all():
        raise ValueError("RF1 input contains non-finite values")
    gain_values = np.asarray(gains, dtype=np.float64)
    if gain_values.ndim != 2 or gain_values.shape[1] != len(frequency):
        raise ValueError("candidate gains must have shape CxF")

    od = rgb01_to_od(rgb01)
    mean_od = od.mean(dim=-1)
    size = int(mean_od.shape[-1])
    padding = size // 4
    padded = F.pad(mean_od[:, None], (padding, padding, padding, padding), mode="reflect")[:, 0]
    spatial_mean = padded.mean(dim=(-2, -1), keepdim=True)
    centered = padded - spatial_mean
    spectrum = torch.fft.rfft2(centered)
    gain_maps = torch.stack(
        [
            radial_gain_map(
                padded.shape[-1], gain, frequency, rgb01.device, rgb01.dtype
            )
            for gain in gain_values
        ],
        dim=0,
    )
    corrected = torch.fft.irfft2(
        spectrum[:, None] * gain_maps[None], s=padded.shape[-2:]
    ).real
    corrected = corrected + spatial_mean[:, None]
    corrected = corrected[
        :, :, padding : padding + size, padding : padding + size
    ]
    proposed_delta = corrected - mean_od[:, None]

    lower = -od.amin(dim=-1)
    upper = OD_RGB8_MAX - od.amax(dim=-1)
    projected_delta = torch.minimum(
        torch.maximum(proposed_delta, lower[:, None]), upper[:, None]
    )

    proposed_od = od[:, None] + proposed_delta[..., None]
    projected_od = od[:, None] + projected_delta[..., None]
    proposed_rgb = od_to_rgb01(proposed_od)
    projected_rgb = od_to_rgb01(projected_od)
    output = projected_rgb.clamp(0.0, 1.0)

    out_of_range = (proposed_rgb < 0.0) | (proposed_rgb > 1.0)
    material = (proposed_rgb < -MATERIAL_EXCURSION) | (
        proposed_rgb > 1.0 + MATERIAL_EXCURSION
    )
    projection = (projected_delta - proposed_delta).abs()

    # Internally tensors are BxC; expose CxB consistently.
    metric_dims = (2, 3, 4)
    pixel_dims = (2, 3)
    return {
        "output": output.permute(1, 0, 2, 3, 4),
        "proposed_rgb": proposed_rgb.permute(1, 0, 2, 3, 4),
        "projected_rgb": projected_rgb.permute(1, 0, 2, 3, 4),
        "proposed_delta": proposed_delta.permute(1, 0, 2, 3),
        "projected_delta": projected_delta.permute(1, 0, 2, 3),
        "preproject_range_fraction": out_of_range.float().mean(dim=metric_dims).T,
        "material_range_fraction": material.float().mean(dim=metric_dims).T,
        "projection_fraction": (projection > 1e-7).float().mean(dim=pixel_dims).T,
        "projection_rgb_mae": (projected_rgb - proposed_rgb)
        .abs()
        .mean(dim=metric_dims)
        .T,
        "final_clamp_mae": (output - projected_rgb).abs().mean(dim=metric_dims).T,
    }


def log_spectrum_rmse(
    observed_power: np.ndarray,
    target_power: np.ndarray,
    frequency: np.ndarray,
    lower: float = 0.10,
) -> float:
    observed = np.asarray(observed_power, dtype=np.float64)
    target = np.asarray(target_power, dtype=np.float64)
    freq = np.asarray(frequency, dtype=np.float64)
    if observed.shape != target.shape or observed.shape != freq.shape:
        raise ValueError("spectrum arrays differ in shape")
    selected = freq >= float(lower)
    if not np.any(selected) or np.any(observed < 0) or np.any(target < 0):
        raise ValueError("invalid spectrum values")
    difference = np.log(np.maximum(observed[selected], 1e-20)) - np.log(
        np.maximum(target[selected], 1e-20)
    )
    return float(np.sqrt(np.mean(difference**2)))
