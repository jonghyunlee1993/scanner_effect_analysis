"""Pure implementations of the frozen E4 primary and collapse endpoints."""

from __future__ import annotations

import numpy as np


COLLAPSE_METRICS = ("variance_trace", "entropy_effective_rank", "median_pairwise_distance")


def l2_normalize(features: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norms <= 0):
        raise ValueError("features must be finite and have positive L2 norm")
    return value / norms


def scanner_centroid_rms(features: np.ndarray) -> float:
    value = l2_normalize(features)
    if value.ndim != 3 or value.shape[:2] != (6, 100):
        raise ValueError(f"expected 6×100×D features, got {value.shape}")
    centroid = value.mean(axis=0)
    per_location = np.sqrt(np.mean(np.sum((value - centroid[None]) ** 2, axis=-1), axis=0))
    return float(per_location.mean())


def unmatched_q95(raw_at2: np.ndarray) -> np.ndarray:
    reference = l2_normalize(raw_at2)
    if reference.ndim != 2 or reference.shape[0] != 100:
        raise ValueError(f"expected 100×D raw AT2 features, got {reference.shape}")
    similarities = reference @ reference.T
    return np.asarray(
        [np.quantile(np.delete(similarities[index], index), 0.95) for index in range(100)],
        dtype=np.float64,
    )


def content_margin_by_scanner(features: np.ndarray, raw_at2: np.ndarray) -> np.ndarray:
    value = l2_normalize(features)
    reference = l2_normalize(raw_at2)
    if value.ndim != 3 or value.shape[:2] != (6, 100):
        raise ValueError(f"expected 6×100×D features, got {value.shape}")
    if reference.shape != value.shape[1:]:
        raise ValueError("raw AT2 and condition feature shapes differ")
    baseline = unmatched_q95(reference)
    matched = np.sum(value[1:] * reference[None], axis=-1)
    return matched - baseline[None]


def collapse_metric_values(features: np.ndarray) -> dict[str, float]:
    value = l2_normalize(features)
    if value.ndim != 2 or value.shape[0] != 100:
        raise ValueError(f"expected 100×D features, got {value.shape}")
    centered = value - value.mean(axis=0, keepdims=True)
    covariance_gram = centered @ centered.T / (len(value) - 1)
    eigenvalues = np.linalg.eigvalsh(covariance_gram)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    variance_trace = float(eigenvalues.sum())
    if variance_trace <= 0:
        effective_rank = 0.0
    else:
        probabilities = eigenvalues[eigenvalues > 0] / variance_trace
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    similarity = np.clip(value @ value.T, -1.0, 1.0)
    upper = np.triu_indices(len(value), k=1)
    pairwise = np.sqrt(np.maximum(2.0 - 2.0 * similarity[upper], 0.0))
    return {
        "variance_trace": variance_trace,
        "entropy_effective_rank": effective_rank,
        "median_pairwise_distance": float(np.median(pairwise)),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    indices: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    value = np.asarray(values, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("bootstrap values must be one-dimensional and finite")
    if indices.ndim != 2 or indices.shape[1] != len(value):
        raise ValueError("bootstrap index matrix has the wrong shape")
    replicates = value[indices].mean(axis=1)
    return (
        float(value.mean()),
        float(np.quantile(replicates, alpha / 2)),
        float(np.quantile(replicates, 1 - alpha / 2)),
    )

