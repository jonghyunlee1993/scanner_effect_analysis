"""High-frequency perturbations and embedding-trajectory measurements."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def reconstruct_with_bands(
    pyramid,
    low: torch.Tensor,
    bands: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Reconstruct requested coefficients and explicitly clip to valid RGB."""
    preclip = pyramid.reconstruct(low, list(bands))
    return {
        "preclip": preclip,
        "image": preclip.clamp(-1.0, 1.0),
    }


def gain_high_frequency(
    pyramid,
    low: torch.Tensor,
    bands: Sequence[torch.Tensor],
    gain: float,
) -> dict[str, torch.Tensor]:
    """Scale every Laplacian band by one common high-frequency gain."""
    return reconstruct_with_bands(
        pyramid,
        low,
        [float(gain) * band for band in bands],
    )


def mix_high_frequency(
    pyramid,
    low: torch.Tensor,
    source_bands: Sequence[torch.Tensor],
    reference_bands: Sequence[torch.Tensor],
    amount: float,
) -> dict[str, torch.Tensor]:
    """Interpolate source detail toward a virtual high-frequency reference."""
    value = float(amount)
    if not 0.0 <= value <= 1.0:
        raise ValueError("high-frequency mix amount must be in [0, 1]")
    if len(source_bands) != len(reference_bands):
        raise ValueError("source and reference must have the same band count")
    return reconstruct_with_bands(
        pyramid,
        low,
        [
            (1.0 - value) * source + value * reference.to(source)
            for source, reference in zip(source_bands, reference_bands)
        ],
    )


def spherical_mean(embedding: np.ndarray, axis: int = 0) -> np.ndarray:
    """Normalized Euclidean mean for unit-length embedding vectors."""
    value = np.asarray(embedding, dtype=np.float64).mean(axis=axis)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norm, 1e-12)


def chord_direction(
    endpoint: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    """Unit embedding-space chord from a baseline to an endpoint."""
    displacement = (
        np.asarray(endpoint, dtype=np.float64)
        - np.asarray(baseline, dtype=np.float64)
    )
    norm = np.linalg.norm(displacement, axis=-1, keepdims=True)
    return displacement / np.maximum(norm, 1e-12)


def pairwise_direction_cosine(directions: np.ndarray) -> np.ndarray:
    """Cosine matrix between unit-normalized embedding displacements."""
    values = np.asarray(directions, dtype=np.float64)
    values = values / np.maximum(
        np.linalg.norm(values, axis=-1, keepdims=True), 1e-12
    )
    return values @ values.T


def embedding_trajectory_metrics(
    embedding: np.ndarray,
    raw: np.ndarray,
    consensus: np.ndarray,
) -> dict[str, np.ndarray]:
    """Measure displacement and progress toward a fixed embedding consensus.

    Direction is measured in the tangent plane of the unit embedding sphere at
    the raw point. This removes the radial component introduced by a finite
    chord and makes the angle describe genuine angular motion.
    """
    embedding = np.asarray(embedding, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    consensus = np.asarray(consensus, dtype=np.float64)
    displacement = embedding - raw
    displacement_norm = np.linalg.norm(displacement, axis=-1)
    displacement_tangent = displacement - (
        np.sum(displacement * raw, axis=-1, keepdims=True) * raw
    )
    desired_tangent = consensus - (
        np.sum(consensus * raw, axis=-1, keepdims=True) * raw
    )
    displacement_tangent_norm = np.linalg.norm(
        displacement_tangent, axis=-1
    )
    desired_tangent_norm = np.linalg.norm(desired_tangent, axis=-1)
    denominator = displacement_tangent_norm * desired_tangent_norm
    direction_cosine = np.divide(
        np.sum(displacement_tangent * desired_tangent, axis=-1),
        denominator,
        out=np.full_like(displacement_norm, np.nan),
        where=denominator > 1e-12,
    )
    self_cosine = np.sum(embedding * raw, axis=-1)
    consensus_cosine = np.sum(embedding * consensus, axis=-1)
    raw_consensus_cosine = np.sum(raw * consensus, axis=-1)
    desired_unit = desired_tangent / np.maximum(
        desired_tangent_norm[..., None], 1e-12
    )
    signed_progress = np.sum(displacement_tangent * desired_unit, axis=-1)
    orthogonal = (
        displacement_tangent - signed_progress[..., None] * desired_unit
    )
    return {
        "embedding_displacement": displacement_norm,
        "direction_cosine_to_consensus": direction_cosine,
        "self_cosine": self_cosine,
        "consensus_cosine": consensus_cosine,
        "consensus_cosine_gain": consensus_cosine - raw_consensus_cosine,
        "signed_consensus_progress": signed_progress,
        "orthogonal_drift": np.linalg.norm(orthogonal, axis=-1),
    }
