"""Frozen-UNI evaluation helpers for scanner-to-AT2 translation.

This module keeps the scientific endpoint separate from feature extraction.
The primary functions, :func:`paired_uni_metrics` and
:func:`embedding_collapse_diagnostics`, operate only on NumPy feature arrays,
so their arithmetic can be tested without loading UNI or using a GPU.

For a source-scanner patch, its translated patch, and the real same-location
AT2 patch, the paired endpoint is::

    d_raw    = 1 - cosine(z_raw, z_AT2)
    d_method = 1 - cosine(z_method, z_AT2)
    gain     = d_raw - d_method
    closure  = gain / d_raw

Positive gain means movement toward the paired AT2 observation.  Fractional
closure is left unbounded: negative values and values above one are meaningful
and are never clipped.  It is undefined (NaN) when ``d_raw`` is at or below
``DEFAULT_NEAR_ZERO_DISTANCE``; the returned ``closure_defined`` flag records
that predeclared guard explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_NEAR_ZERO_DISTANCE = 1e-6
DEFAULT_NN_CHUNK_SIZE = 2048

REQUIRED_METADATA_ALIASES = {
    "source_scanner": ("source_scanner",),
    "slide": ("slide_id",),
    "tissue": ("tissue_type", "tissue"),
    "fold": ("fold",),
    "location": ("location_index", "location"),
}

UNI_METRIC_COLUMNS = (
    "raw_to_at2_distance",
    "method_to_at2_distance",
    "gain_to_at2",
    "fractional_closure",
    "method_to_raw_distance",
)

PLISM_EXTERNAL_SHIFT_NOTE = (
    "Every PLISM image is externally shifted relative to PanNormal by site, "
    "physical device, section, stain, and acquisition context. 'ID-like' "
    "describes only scanner-model and tissue-label membership; it never means "
    "that a PLISM image is fully in-distribution."
)


def _as_finite_feature_matrix(value: Any, name: str) -> np.ndarray:
    """Return a finite floating-point ``(sample, feature)`` array."""

    result = np.asarray(value)
    if result.ndim != 2:
        raise ValueError(f"{name} must have shape (sample, feature), got {result.shape}")
    if result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got {result.dtype}")
    # float64 avoids avoidable cancellation when two normalized embeddings are
    # extremely close, which matters for the fractional-closure guard.
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return result


def _validate_feature_triplet(
    raw_features: Any,
    method_features: Any,
    at2_features: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = _as_finite_feature_matrix(raw_features, "raw_features")
    method = _as_finite_feature_matrix(method_features, "method_features")
    at2 = _as_finite_feature_matrix(at2_features, "at2_features")
    if raw.shape != method.shape or raw.shape != at2.shape:
        raise ValueError(
            "raw_features, method_features, and at2_features must have identical "
            f"shape; got {raw.shape}, {method.shape}, and {at2.shape}"
        )
    return raw, method, at2


def _unit_normalize(features: Any, name: str, eps: float = 1e-12) -> np.ndarray:
    values = _as_finite_feature_matrix(features, name)
    if eps <= 0:
        raise ValueError("eps must be positive")
    norms = np.linalg.norm(values, axis=1)
    invalid = norms <= eps
    if invalid.any():
        indices = np.flatnonzero(invalid)[:5].tolist()
        raise ValueError(f"{name} contains zero/near-zero norm rows at {indices}")
    return values / norms[:, None]


def cosine_distance_rows(left: Any, right: Any, *, eps: float = 1e-12) -> np.ndarray:
    """Compute row-aligned cosine distance without assuming normalized inputs."""

    left_values = _as_finite_feature_matrix(left, "left")
    right_values = _as_finite_feature_matrix(right, "right")
    if left_values.shape != right_values.shape:
        raise ValueError(
            f"left and right must have identical shape, got {left_values.shape} "
            f"and {right_values.shape}"
        )
    left_unit = _unit_normalize(left_values, "left", eps=eps)
    right_unit = _unit_normalize(right_values, "right", eps=eps)
    # Clipping only corrects floating-point excursions outside the mathematical
    # cosine range.  It does not clip gain or fractional closure.
    similarity = np.clip(np.einsum("ij,ij->i", left_unit, right_unit), -1.0, 1.0)
    return 1.0 - similarity


def _metadata_frame(metadata: Any, sample_count: int) -> pd.DataFrame:
    if isinstance(metadata, pd.DataFrame):
        frame = metadata.copy().reset_index(drop=True)
    elif isinstance(metadata, Mapping):
        frame = pd.DataFrame(metadata).reset_index(drop=True)
    else:
        frame = pd.DataFrame(metadata).reset_index(drop=True)
    if len(frame) != sample_count:
        raise ValueError(
            f"metadata has {len(frame)} rows but embeddings have {sample_count} rows"
        )
    missing = []
    for semantic_name, aliases in REQUIRED_METADATA_ALIASES.items():
        if not any(column in frame.columns for column in aliases):
            missing.append(f"{semantic_name} ({' or '.join(aliases)})")
    if missing:
        raise ValueError(f"metadata is missing required fields: {', '.join(missing)}")
    return frame


def paired_uni_metrics(
    raw_features: Any,
    method_features: Any,
    at2_features: Any,
    metadata: Any,
    *,
    method: str = "pix2pix",
    target_scanner: str = "AT2",
    near_zero_distance: float = DEFAULT_NEAR_ZERO_DISTANCE,
) -> pd.DataFrame:
    """Return location-level paired frozen-UNI metrics with metadata intact.

    Parameters
    ----------
    raw_features, method_features, at2_features:
        Row-aligned feature matrices.  Each row must describe the same physical
        location in all three arrays.  Inputs need not already have unit norm.
    metadata:
        A data frame or mapping with ``source_scanner``, ``slide_id``, ``fold``,
        one of ``tissue_type``/``tissue``, and one of
        ``location_index``/``location``.  Every supplied column is copied to the
        result, rather than reducing metadata to the minimum contract.
    near_zero_distance:
        ``fractional_closure`` is NaN when ``d_raw <= near_zero_distance``.
        Gain and both distances remain reported.  Closure is not clipped.
    """

    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a non-empty string")
    if not isinstance(target_scanner, str) or not target_scanner.strip():
        raise ValueError("target_scanner must be a non-empty string")
    if not np.isfinite(near_zero_distance) or near_zero_distance < 0:
        raise ValueError("near_zero_distance must be finite and non-negative")

    raw, translated, at2 = _validate_feature_triplet(
        raw_features, method_features, at2_features
    )
    result = _metadata_frame(metadata, len(raw))
    reserved = {
        "method",
        "target_scanner",
        "near_zero_distance",
        "closure_defined",
        *UNI_METRIC_COLUMNS,
    }
    overlap = reserved.intersection(result.columns)
    if overlap:
        raise ValueError(f"metadata uses reserved result columns: {sorted(overlap)}")

    d_raw = cosine_distance_rows(raw, at2)
    d_method = cosine_distance_rows(translated, at2)
    gain = d_raw - d_method
    closure_defined = d_raw > near_zero_distance
    closure = np.full(len(raw), np.nan, dtype=np.float64)
    np.divide(gain, d_raw, out=closure, where=closure_defined)

    result["method"] = method.strip()
    result["target_scanner"] = target_scanner.strip()
    result["raw_to_at2_distance"] = d_raw
    result["method_to_at2_distance"] = d_method
    result["gain_to_at2"] = gain
    result["fractional_closure"] = closure
    result["closure_defined"] = closure_defined
    result["near_zero_distance"] = float(near_zero_distance)
    result["method_to_raw_distance"] = cosine_distance_rows(translated, raw)
    return result


def nearest_neighbor_assignments(
    query_features: Any,
    reference_features: Any,
    *,
    exclude_same_index: bool = False,
    chunk_size: int = DEFAULT_NN_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cosine-nearest reference indices and similarities in chunks.

    ``exclude_same_index`` is intended for self-neighbour diagnostics and
    requires the query and reference matrices to have the same row count.
    """

    query = _unit_normalize(query_features, "query_features")
    reference = _unit_normalize(reference_features, "reference_features")
    if query.shape[1] != reference.shape[1]:
        raise ValueError(
            f"feature dimensions differ: {query.shape[1]} and {reference.shape[1]}"
        )
    if not isinstance(chunk_size, (int, np.integer)) or int(chunk_size) <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if exclude_same_index:
        if len(query) != len(reference):
            raise ValueError("exclude_same_index requires equal row counts")
        if len(reference) < 2:
            raise ValueError("at least two rows are required after self-exclusion")

    indices = np.empty(len(query), dtype=np.int64)
    similarities = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), int(chunk_size)):
        stop = min(start + int(chunk_size), len(query))
        block = query[start:stop] @ reference.T
        if exclude_same_index:
            rows = np.arange(stop - start)
            block[rows, np.arange(start, stop)] = -np.inf
        block_indices = np.argmax(block, axis=1)
        indices[start:stop] = block_indices
        similarities[start:stop] = block[np.arange(stop - start), block_indices]
    return indices, similarities


def embedding_collapse_diagnostics(
    method_features: Any,
    reference_features: Any | None = None,
    *,
    chunk_size: int = DEFAULT_NN_CHUNK_SIZE,
) -> dict[str, float | int | bool]:
    """Summarize feature collapse and nearest-neighbour concentration.

    Norm and variance summaries describe the translated embeddings themselves.
    Nearest-neighbour summaries use ``reference_features`` when supplied (the
    paired AT2 pool is the usual choice), otherwise they use the translated
    pool with each sample's self-match excluded.  Concentration is exposed as
    the maximum target share, Herfindahl index, normalized entropy, and the
    fraction of reference rows selected at least once.  Lower entropy or a high
    maximum share/HHI is compatible with many outputs collapsing onto a small
    part of the reference space and should be compared with raw and real-AT2
    baselines, not interpreted from a universal threshold.
    """

    method = _as_finite_feature_matrix(method_features, "method_features")
    norms = np.linalg.norm(method, axis=1)
    unit = _unit_normalize(method, "method_features")
    centered = unit - unit.mean(axis=0, keepdims=True)
    dimension_variance = np.var(unit, axis=0, ddof=0)

    if reference_features is None:
        reference = method
        exclude_same_index = True
        reference_is_method = True
    else:
        reference = _as_finite_feature_matrix(reference_features, "reference_features")
        exclude_same_index = False
        reference_is_method = False
    indices, similarities = nearest_neighbor_assignments(
        method,
        reference,
        exclude_same_index=exclude_same_index,
        chunk_size=chunk_size,
    )
    counts = np.bincount(indices, minlength=len(reference)).astype(np.float64)
    probabilities = counts[counts > 0] / len(method)
    hhi = float(np.square(probabilities).sum())
    if len(reference) > 1:
        normalized_entropy = float(
            -(probabilities * np.log(probabilities)).sum() / np.log(len(reference))
        )
    else:
        normalized_entropy = 0.0

    result: dict[str, float | int | bool] = {
        "n_embeddings": int(len(method)),
        "feature_dim": int(method.shape[1]),
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_std": float(norms.std(ddof=0)),
        "embedding_norm_min": float(norms.min()),
        "embedding_norm_max": float(norms.max()),
        "embedding_total_variance": float(np.square(centered).sum(axis=1).mean()),
        "embedding_mean_dimension_variance": float(dimension_variance.mean()),
        "nearest_reference_cosine_mean": float(similarities.mean()),
        "nearest_reference_cosine_p95": float(np.quantile(similarities, 0.95)),
        "nearest_reference_unique_fraction": float(np.count_nonzero(counts) / len(reference)),
        "nearest_reference_max_share": float(counts.max() / len(method)),
        "nearest_reference_hhi": hhi,
        "nearest_reference_normalized_entropy": normalized_entropy,
        "reference_is_method": reference_is_method,
    }
    if reference_features is not None and len(method) == len(reference):
        result["paired_reference_top1_rate"] = float(
            np.mean(indices == np.arange(len(method)))
        )
    return result


def collapse_diagnostics_by_group(
    method_features: Any,
    metadata: pd.DataFrame,
    *,
    reference_features: Any | None = None,
    group_columns: Sequence[str] = ("source_scanner", "method"),
    chunk_size: int = DEFAULT_NN_CHUNK_SIZE,
) -> pd.DataFrame:
    """Compute collapse diagnostics for metadata-defined, row-aligned groups."""

    method = _as_finite_feature_matrix(method_features, "method_features")
    frame = metadata.copy().reset_index(drop=True)
    if len(frame) != len(method):
        raise ValueError("metadata and method_features row counts differ")
    reference = None
    if reference_features is not None:
        reference = _as_finite_feature_matrix(reference_features, "reference_features")
        if reference.shape != method.shape:
            raise ValueError("reference_features and method_features shapes differ")
    groups = list(group_columns)
    missing = set(groups) - set(frame.columns)
    if missing:
        raise ValueError(f"metadata is missing group columns: {sorted(missing)}")

    rows = []
    grouped = frame.groupby(groups, sort=True, dropna=False).indices
    for key, positions in grouped.items():
        key_values = key if isinstance(key, tuple) else (key,)
        positions = np.asarray(positions, dtype=np.int64)
        values = embedding_collapse_diagnostics(
            method[positions],
            None if reference is None else reference[positions],
            chunk_size=chunk_size,
        )
        rows.append({**dict(zip(groups, key_values)), **values})
    return pd.DataFrame(rows)


def aggregate_physical_slides(
    location_metrics: pd.DataFrame,
    *,
    physical_slide_column: str = "slide_id",
    strata_columns: Sequence[str] = ("source_scanner", "method"),
    metric_columns: Sequence[str] = UNI_METRIC_COLUMNS,
) -> pd.DataFrame:
    """Average locations so each physical slide contributes one observation.

    For PanNormal, ``physical_slide_column='slide_id'`` is appropriate.  For
    PLISM, pass the serial-section identifier (for example ``'stain'``), not a
    TMA core or location.  Include an OOD quadrant in ``strata_columns`` when
    producing the PLISM 2x2 summary.
    """

    frame = location_metrics.copy()
    group_columns = [physical_slide_column, *strata_columns]
    missing = set(group_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"location_metrics is missing grouping columns: {sorted(missing)}")
    metrics = list(metric_columns)
    missing_metrics = set(metrics) - set(frame.columns)
    if missing_metrics:
        raise ValueError(f"location_metrics is missing metrics: {sorted(missing_metrics)}")
    if frame.empty:
        raise ValueError("location_metrics must not be empty")

    numeric = frame[metrics].apply(pd.to_numeric, errors="raise")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("metric columns contain infinite values")
    working = frame[group_columns].copy()
    working[metrics] = numeric
    if "closure_defined" in frame:
        working["closure_defined"] = frame["closure_defined"].astype(bool)

    grouped = working.groupby(group_columns, sort=True, dropna=False)
    result = grouped[metrics].mean().reset_index()
    counts = grouped.size().rename("n_locations").reset_index()
    result = result.merge(counts, on=group_columns, validate="one_to_one")
    if "closure_defined" in working:
        closure = (
            grouped["closure_defined"]
            .agg(closure_defined_count="sum", closure_defined_fraction="mean")
            .reset_index()
        )
        result = result.merge(closure, on=group_columns, validate="one_to_one")
    return result


def summarize_physical_slides(
    slide_metrics: pd.DataFrame,
    *,
    physical_slide_column: str = "slide_id",
    group_columns: Sequence[str] = ("source_scanner", "method"),
    metric_columns: Sequence[str] = UNI_METRIC_COLUMNS,
    bootstrap_replicates: int = 20_000,
    confidence_level: float = 0.95,
    seed: int = 20260917,
) -> pd.DataFrame:
    """Return equal-physical-slide means and percentile bootstrap intervals."""

    groups = list(group_columns)
    required = {physical_slide_column, *groups, *metric_columns}
    missing = required - set(slide_metrics.columns)
    if missing:
        raise ValueError(f"slide_metrics is missing columns: {sorted(missing)}")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    duplicate = slide_metrics.duplicated([physical_slide_column, *groups])
    if duplicate.any():
        raise ValueError(
            "slide_metrics must have one row per physical slide and analysis stratum; "
            "run aggregate_physical_slides first"
        )

    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence_level) / 2.0
    rows: list[dict[str, Any]] = []
    grouped = slide_metrics.groupby(groups, sort=True, dropna=False) if groups else [((), slide_metrics)]
    for key, block in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(groups, key_values))
        n_slides = int(block[physical_slide_column].nunique())
        for metric in metric_columns:
            values = pd.to_numeric(block[metric], errors="raise").to_numpy(float)
            values = values[np.isfinite(values)]
            if not len(values):
                estimate = low = high = float("nan")
            else:
                estimate = float(values.mean())
                draws = rng.choice(values, size=(int(bootstrap_replicates), len(values)), replace=True)
                means = draws.mean(axis=1)
                low, high = np.quantile(means, [alpha, 1.0 - alpha]).astype(float)
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_physical_slides": n_slides,
                    "n_finite_slides": int(len(values)),
                    "bootstrap_replicates": int(bootstrap_replicates),
                    "confidence_level": float(confidence_level),
                }
            )
    return pd.DataFrame(rows)


def plism_model_status(
    source_scanner: str,
    shared_tissue: bool,
    *,
    model_source_scanner: str = "GT450",
) -> dict[str, str | bool]:
    """Describe one PLISM observation in a generator-specific 2x2 design.

    For the primary GT450-specific generator, only GT450 is scanner-ID-like;
    every other input scanner is scanner-OOD.  ``shared_tissue`` must be passed
    explicitly from the frozen tissue crosswalk.  Even the shared/shared cell
    remains an external replication because all PLISM observations are shifted
    in site/device/section/stain/acquisition context.
    """

    scanner = str(source_scanner).strip()
    model_scanner = str(model_source_scanner).strip()
    if not scanner or not model_scanner:
        raise ValueError("source_scanner and model_source_scanner must be non-empty")
    if not isinstance(shared_tissue, (bool, np.bool_)):
        raise TypeError("shared_tissue must be an explicit boolean")

    scanner_id_like = scanner.casefold() == model_scanner.casefold()
    tissue_id_like = bool(shared_tissue)
    if scanner_id_like and tissue_id_like:
        quadrant = "shared/shared"
    elif not scanner_id_like and tissue_id_like:
        quadrant = "scanner-OOD"
    elif scanner_id_like and not tissue_id_like:
        quadrant = "tissue-OOD"
    else:
        quadrant = "double-OOD"
    return {
        "source_scanner": scanner,
        "model_source_scanner": model_scanner,
        "scanner_status": "scanner-ID-like" if scanner_id_like else "scanner-OOD",
        "tissue_status": "tissue-ID-like" if tissue_id_like else "tissue-OOD",
        "quadrant": quadrant,
        "plism_external_shifted": True,
        "fully_in_distribution": False,
        "id_like_scope": "scanner-model and tissue-label only",
        "external_shift_note": PLISM_EXTERNAL_SHIFT_NOTE,
    }


def add_plism_model_status(
    metadata: pd.DataFrame,
    *,
    scanner_column: str = "source_scanner",
    shared_tissue_column: str = "shared_tissue",
    model_source_scanner: str = "GT450",
) -> pd.DataFrame:
    """Attach generator-specific PLISM status columns to a metadata frame."""

    missing = {scanner_column, shared_tissue_column} - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing PLISM status columns: {sorted(missing)}")
    result = metadata.copy().reset_index(drop=True)
    status_rows = [
        plism_model_status(
            row[scanner_column],
            row[shared_tissue_column],
            model_source_scanner=model_source_scanner,
        )
        for _, row in result.iterrows()
    ]
    status = pd.DataFrame(status_rows)
    # The original scanner column is retained; avoid duplicating it when it has
    # the canonical name used by plism_model_status.
    for column in status.columns:
        if column == "source_scanner" and column in result.columns:
            continue
        if column in result.columns:
            raise ValueError(f"metadata already contains generated status column {column!r}")
        result[column] = status[column].to_numpy()
    return result


def load_frozen_uni(device: Any):
    """Load the repository's sealed UNI model and disable all gradients."""

    from prenorm.embedding import load_uni

    model, size, mean, std = load_uni(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, size, mean, std


def embed_frozen_uni(
    model: Any,
    images: Any,
    size: int,
    mean: Any,
    std: Any,
    device: Any,
    *,
    batch_size: int = 32,
    value_range: str = "minus_one_one",
) -> np.ndarray:
    """Embed RGB images with frozen UNI and return normalized NumPy features.

    ``images`` may be NHWC or NCHW.  ``value_range`` is explicit to prevent a
    generated ``[-1, 1]`` tensor from being mistaken for uint8 pixels; accepted
    values are ``'minus_one_one'``, ``'zero_one'``, and ``'uint8'``.
    """

    import torch

    from prenorm.embedding import embed_uni

    if not isinstance(batch_size, (int, np.integer)) or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(images, torch.Tensor):
        tensor = images.detach()
    else:
        tensor = torch.as_tensor(np.asarray(images))
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"images must have rank 4 after batching, got {tuple(tensor.shape)}")
    if tensor.shape[1] == 3:
        pass
    elif tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"images must have three RGB channels, got {tuple(tensor.shape)}")

    tensor = tensor.to(dtype=torch.float32)
    if value_range == "uint8":
        if tensor.min().item() < 0 or tensor.max().item() > 255:
            raise ValueError("uint8-range images must lie in [0, 255]")
        tensor = tensor.div(127.5).sub(1.0)
    elif value_range == "zero_one":
        if tensor.min().item() < 0 or tensor.max().item() > 1:
            raise ValueError("zero_one images must lie in [0, 1]")
        tensor = tensor.mul(2.0).sub(1.0)
    elif value_range == "minus_one_one":
        if tensor.min().item() < -1 or tensor.max().item() > 1:
            raise ValueError("minus_one_one images must lie in [-1, 1]")
    else:
        raise ValueError(
            "value_range must be 'minus_one_one', 'zero_one', or 'uint8'"
        )
    features = embed_uni(
        model,
        tensor,
        int(size),
        mean,
        std,
        device,
        batch_size=int(batch_size),
    )
    result = features.detach().cpu().numpy().astype(np.float32, copy=False)
    if result.ndim != 2 or len(result) != len(tensor) or not np.isfinite(result).all():
        raise ValueError(f"invalid UNI embedding output shape {result.shape}")
    return result
