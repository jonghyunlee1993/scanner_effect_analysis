"""Leakage-safe analytic controls for scanner-to-AT2 translation.

The learned baselines translate GT450, VERSA, or AKOYA images to the AT2
appearance.  The existing scanner-batch analysis fitted corrections in the
opposite direction, so its saved parameters must not be reused here.  This
module refits three image-only controls on the *training slides* of each outer
fold:

``reinhard``
    Match the source-scanner LAB mean and standard deviation to AT2.
``frequency``
    Preserve phase while multiplying the source OD amplitude spectrum by the
    paired AT2/source radial gain.
``combined``
    Apply Reinhard first and the frequency correction second.

The fit API deliberately requires a ``split_role`` column.  It rejects split
tables that do not have the formal three-train/one-validation/one-test fold
layout and records the exact training slides in every parameter block.  The
frequency sign is estimated by joining source and AT2 rows from the same
training slide and frequency bin; it is never obtained from a separately
fitted or oppositely split curve.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from skimage.color import lab2rgb, rgb2lab


ANALYTIC_VERSION = "scanner_to_at2_analytic_controls_v1"
FORMAL_SOURCE_SCANNERS = ("gt450", "versa", "akoya")
FORMAL_TARGET_SCANNER = "at2"
ANALYTIC_ARMS = ("raw", "reinhard", "frequency", "combined")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_frame(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.read_csv(Path(value), dtype={"slide_id": str})


def _frame_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    """Hash a canonical view of the rows actually used for a fit."""

    canonical = frame.loc[:, columns].copy()
    canonical["slide_id"] = canonical["slide_id"].astype(str)
    canonical = canonical.sort_values(columns, kind="mergesort").reset_index(drop=True)
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scanner_name(value: str) -> str:
    return str(value).strip().lower()


def _formal_split(sample_index: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"slide_id", "fold", "split_role"}
    missing = required - set(sample_index.columns)
    if missing:
        raise ValueError(f"sample index is missing columns: {sorted(missing)}")

    split = sample_index.copy()
    split["slide_id"] = split["slide_id"].astype(str)
    split["fold"] = pd.to_numeric(split["fold"], errors="raise").astype(int)
    split["split_role"] = split["split_role"].astype(str).str.strip().str.lower()
    allowed_roles = {"train", "validation", "test"}
    unexpected = set(split["split_role"].unique()) - allowed_roles
    if unexpected:
        raise ValueError(f"unexpected split_role values: {sorted(unexpected)}")

    slide_layout = split[["slide_id", "fold", "split_role"]].drop_duplicates()
    multiplicity = slide_layout.groupby("slide_id", sort=False).size()
    if not multiplicity.eq(1).all():
        offenders = multiplicity[multiplicity.ne(1)].index.tolist()
        raise ValueError(f"slides assigned to multiple folds or roles: {offenders[:5]}")

    role_folds = {
        role: sorted(
            slide_layout.loc[slide_layout["split_role"] == role, "fold"].unique().tolist()
        )
        for role in sorted(allowed_roles)
    }
    if len(role_folds["train"]) != 3:
        raise ValueError(
            "formal analytic controls require exactly three training folds; "
            f"got {role_folds['train']}"
        )
    if len(role_folds["validation"]) != 1 or len(role_folds["test"]) != 1:
        raise ValueError(
            "formal analytic controls require one validation and one test fold; "
            f"got validation={role_folds['validation']}, test={role_folds['test']}"
        )
    if set(role_folds["train"]) & set(role_folds["validation"] + role_folds["test"]):
        raise ValueError("train, validation, and test folds must be disjoint")

    train_slides = sorted(
        slide_layout.loc[slide_layout["split_role"] == "train", "slide_id"].tolist()
    )
    if not train_slides:
        raise ValueError("split contains no training slides")
    provenance = {
        "outer_test_fold": int(role_folds["test"][0]),
        "validation_fold": int(role_folds["validation"][0]),
        "training_folds": [int(value) for value in role_folds["train"]],
        "training_slides": train_slides,
        "training_slide_count": len(train_slides),
        "fit_split_role": "train",
    }
    return slide_layout, provenance


def _validate_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "slide_id",
        "scanner",
        "pixels",
        "sum_l",
        "sumsq_l",
        "sum_a",
        "sumsq_a",
        "sum_b",
        "sumsq_b",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"LAB statistics are missing columns: {sorted(missing)}")
    result = frame.copy()
    result["slide_id"] = result["slide_id"].astype(str)
    result["scanner"] = result["scanner"].astype(str).str.strip().str.lower()
    numeric = sorted(required - {"slide_id", "scanner"})
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(result[numeric].to_numpy(float)).all():
        raise ValueError("LAB statistics contain non-finite values")
    if (result["pixels"] <= 0).any():
        raise ValueError("LAB statistics contain non-positive pixel counts")
    if result.duplicated(["slide_id", "scanner"]).any():
        raise ValueError("LAB statistics contain duplicate slide/scanner rows")
    return result


def _lab_moments(
    statistics: pd.DataFrame,
    training_slides: set[str],
    scanner: str,
) -> tuple[list[float], list[float], pd.DataFrame]:
    selected = statistics[
        statistics["slide_id"].isin(training_slides)
        & statistics["scanner"].eq(scanner)
    ].copy()
    observed = set(selected["slide_id"])
    if observed != training_slides:
        missing = sorted(training_slides - observed)
        extra = sorted(observed - training_slides)
        raise ValueError(
            f"LAB statistics for {scanner} do not match training slides; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    count = float(selected["pixels"].sum())
    means: list[float] = []
    stds: list[float] = []
    for label in ("l", "a", "b"):
        mean = float(selected[f"sum_{label}"].sum() / count)
        second = float(selected[f"sumsq_{label}"].sum() / count)
        means.append(mean)
        stds.append(max(math.sqrt(max(second - mean * mean, 0.0)), 1e-3))
    return means, stds, selected


def _paired_frequency_gain(
    spectra: pd.DataFrame,
    training_slides: set[str],
    source_scanner: str,
    target_scanner: str,
    *,
    smoothing_sigma_bins: float,
    log2_gain_clip: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    required = {
        "slide_id",
        "scanner",
        "frequency_cyc_per_pixel",
        "log2_relative_transfer",
    }
    missing = required - set(spectra.columns)
    if missing:
        raise ValueError(f"spectra are missing columns: {sorted(missing)}")
    frame = spectra.copy()
    frame["slide_id"] = frame["slide_id"].astype(str)
    frame["scanner"] = frame["scanner"].astype(str).str.strip().str.lower()
    for column in ("frequency_cyc_per_pixel", "log2_relative_transfer"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    selected = frame[
        frame["slide_id"].isin(training_slides)
        & frame["scanner"].isin((source_scanner, target_scanner))
    ].copy()
    if selected.empty:
        raise ValueError("no training spectra were selected")
    if not np.isfinite(
        selected[["frequency_cyc_per_pixel", "log2_relative_transfer"]].to_numpy(float)
    ).all():
        raise ValueError("spectra contain non-finite values")
    if selected.duplicated(["slide_id", "scanner", "frequency_cyc_per_pixel"]).any():
        raise ValueError("spectra contain duplicate slide/scanner/frequency rows")

    key = ["slide_id", "frequency_cyc_per_pixel"]
    source = selected[selected["scanner"].eq(source_scanner)][
        key + ["log2_relative_transfer"]
    ].rename(columns={"log2_relative_transfer": "source_log2_transfer"})
    target = selected[selected["scanner"].eq(target_scanner)][
        key + ["log2_relative_transfer"]
    ].rename(columns={"log2_relative_transfer": "target_log2_transfer"})
    paired = source.merge(target, on=key, how="outer", validate="one_to_one", indicator=True)
    if not paired["_merge"].eq("both").all():
        bad = paired.loc[paired["_merge"].ne("both"), key + ["_merge"]]
        raise ValueError(
            "source and target spectra must be paired on the same training rows; "
            f"unpaired examples={bad.head().to_dict('records')}"
        )
    paired = paired.drop(columns="_merge")
    observed_slides = set(paired["slide_id"])
    if observed_slides != training_slides:
        raise ValueError(
            "frequency rows do not cover all training slides; "
            f"missing={sorted(training_slides - observed_slides)[:5]}"
        )

    counts = paired.groupby("frequency_cyc_per_pixel")["slide_id"].nunique()
    if not counts.eq(len(training_slides)).all():
        bad_frequency = counts[counts.ne(len(training_slides))].index.tolist()
        raise ValueError(
            "frequency grid is incomplete across training slides; "
            f"examples={bad_frequency[:5]}"
        )
    # Existing spectra express each scanner relative to AT2.  Subtracting the
    # same-slide source curve from the same-slide AT2 curve produces the exact
    # reverse AT2/source log2 amplitude gain without cross-split inversion.
    paired["log2_at2_over_source"] = (
        paired["target_log2_transfer"] - paired["source_log2_transfer"]
    )
    curve = (
        paired.groupby("frequency_cyc_per_pixel", sort=True)["log2_at2_over_source"]
        .mean()
        .sort_index()
    )
    frequency = curve.index.to_numpy(float)
    if len(frequency) < 2 or np.any(np.diff(frequency) <= 0):
        raise ValueError("frequency grid must contain at least two increasing values")
    log_gain = curve.to_numpy(float)
    if smoothing_sigma_bins > 0:
        log_gain = gaussian_filter1d(
            log_gain, sigma=float(smoothing_sigma_bins), mode="nearest"
        )
    lower, upper = map(float, log2_gain_clip)
    if not lower < upper:
        raise ValueError("log2_gain_clip must be increasing")
    log_gain = np.clip(log_gain, lower, upper)
    return frequency, log_gain, paired


def fit_reverse_analytic_controls(
    sample_index: str | Path | pd.DataFrame,
    lab_statistics: str | Path | pd.DataFrame,
    spectra: str | Path | pd.DataFrame,
    source_scanner: str,
    *,
    target_scanner: str = FORMAL_TARGET_SCANNER,
    smoothing_sigma_bins: float = 1.5,
    log2_gain_clip: tuple[float, float] = (-1.0, 1.0),
) -> dict[str, Any]:
    """Fit source→AT2 Reinhard/frequency parameters using train slides only.

    ``sample_index`` must already have slide-level ``split_role`` assignments.
    Location rows may be repeated; the fit uses the unique training-slide set.
    LAB sufficient statistics and radial spectra may contain all slides, but
    validation/test rows are filtered before any aggregate is calculated.
    """

    source_scanner = _scanner_name(source_scanner)
    target_scanner = _scanner_name(target_scanner)
    if source_scanner not in FORMAL_SOURCE_SCANNERS:
        raise ValueError(f"source_scanner must be one of {FORMAL_SOURCE_SCANNERS}")
    if target_scanner != FORMAL_TARGET_SCANNER:
        raise ValueError(f"formal target scanner is fixed to {FORMAL_TARGET_SCANNER}")
    if source_scanner == target_scanner:
        raise ValueError("source and target scanners must differ")
    if smoothing_sigma_bins < 0:
        raise ValueError("smoothing_sigma_bins must be non-negative")

    split_input = _read_frame(sample_index)
    slide_layout, split_provenance = _formal_split(split_input)
    training_slides = set(split_provenance["training_slides"])
    statistics = _validate_statistics(_read_frame(lab_statistics))
    spectra_frame = _read_frame(spectra)

    source_mean, source_std, source_stats = _lab_moments(
        statistics, training_slides, source_scanner
    )
    target_mean, target_std, target_stats = _lab_moments(
        statistics, training_slides, target_scanner
    )
    frequency, log_gain, paired_spectra = _paired_frequency_gain(
        spectra_frame,
        training_slides,
        source_scanner,
        target_scanner,
        smoothing_sigma_bins=smoothing_sigma_bins,
        log2_gain_clip=log2_gain_clip,
    )

    lab_hash_columns = [
        "slide_id",
        "scanner",
        "pixels",
        "sum_l",
        "sumsq_l",
        "sum_a",
        "sumsq_a",
        "sum_b",
        "sumsq_b",
    ]
    frequency_hash_columns = [
        "slide_id",
        "frequency_cyc_per_pixel",
        "source_log2_transfer",
        "target_log2_transfer",
        "log2_at2_over_source",
    ]
    used_lab = pd.concat([source_stats, target_stats], ignore_index=True)
    split_hash = _frame_sha256(
        slide_layout[slide_layout["slide_id"].isin(training_slides)],
        ["slide_id", "fold", "split_role"],
    )
    payload: dict[str, Any] = {
        "analytic_version": ANALYTIC_VERSION,
        "created_utc": _utc_now(),
        "direction": f"{source_scanner}->{target_scanner}",
        "source_scanner": source_scanner,
        "target_scanner": target_scanner,
        **split_provenance,
        "reinhard": {
            "source_mean": source_mean,
            "source_std": source_std,
            "target_mean": target_mean,
            "target_std": target_std,
        },
        "frequency": {
            "frequency_cyc_per_pixel": frequency.tolist(),
            "log2_amplitude_gain": log_gain.tolist(),
            "definition": "paired target_log2_transfer - source_log2_transfer",
            "smoothing_sigma_bins": float(smoothing_sigma_bins),
            "log2_gain_clip": [float(log2_gain_clip[0]), float(log2_gain_clip[1])],
        },
        "fit_provenance": {
            "split_rule": "split_role == train only",
            "primary_fairness_rule": "three training folds; validation and test excluded",
            "sample_index_training_sha256": split_hash,
            "lab_training_rows_sha256": _frame_sha256(used_lab, lab_hash_columns),
            "paired_frequency_training_rows_sha256": _frame_sha256(
                paired_spectra, frequency_hash_columns
            ),
            "lab_rows_used": int(len(used_lab)),
            "paired_frequency_rows_used": int(len(paired_spectra)),
        },
    }
    return payload


def _validate_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 image, got {array.shape}")
    if array.dtype != np.uint8:
        raise TypeError(f"expected uint8 image, got {array.dtype}")
    return array


def _validate_parameter_direction(parameter: Mapping[str, Any]) -> None:
    source = _scanner_name(str(parameter.get("source_scanner", "")))
    target = _scanner_name(str(parameter.get("target_scanner", "")))
    if source not in FORMAL_SOURCE_SCANNERS or target != FORMAL_TARGET_SCANNER:
        raise ValueError(
            "parameter direction must be one of GT450/VERSA/AKOYA->AT2; "
            f"got {source!r}->{target!r}"
        )
    if parameter.get("fit_split_role") != "train":
        raise ValueError("analytic parameter was not marked as train-only")


def apply_reinhard(image: np.ndarray, parameter: Mapping[str, Any]) -> np.ndarray:
    """Apply the fitted source→AT2 LAB moment transform to one RGB image."""

    array = _validate_image(image)
    block = parameter["reinhard"] if "reinhard" in parameter else parameter
    lab = rgb2lab(array.astype(np.float32) / 255.0).astype(np.float32)
    source_mean = np.asarray(block["source_mean"], dtype=np.float32)
    source_std = np.asarray(block["source_std"], dtype=np.float32)
    target_mean = np.asarray(block["target_mean"], dtype=np.float32)
    target_std = np.asarray(block["target_std"], dtype=np.float32)
    if any(value.shape != (3,) for value in (source_mean, source_std, target_mean, target_std)):
        raise ValueError("Reinhard means and standard deviations must have length three")
    if np.any(source_std <= 0) or np.any(target_std <= 0):
        raise ValueError("Reinhard standard deviations must be positive")
    corrected = (lab - source_mean) / source_std * target_std + target_mean
    corrected[..., 0] = np.clip(corrected[..., 0], 0.0, 100.0)
    corrected[..., 1:] = np.clip(corrected[..., 1:], -127.0, 127.0)
    rgb = lab2rgb(corrected)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _replace_od_luminance(image: np.ndarray, corrected_mean_od: np.ndarray) -> np.ndarray:
    channel_od = -np.log((image.astype(np.float32) + 1.0) / 256.0)
    original_mean = channel_od.mean(axis=-1)
    shifted = channel_od + (corrected_mean_od - original_mean)[..., None]
    rgb = np.exp(-np.clip(shifted, 0.0, 8.0)) * 256.0 - 1.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def apply_frequency(image: np.ndarray, parameter: Mapping[str, Any]) -> np.ndarray:
    """Apply the fitted phase-preserving AT2/source radial amplitude gain."""

    array = _validate_image(image)
    block = parameter["frequency"] if "frequency" in parameter else parameter
    frequency = np.asarray(block["frequency_cyc_per_pixel"], dtype=float)
    log_gain = np.asarray(block["log2_amplitude_gain"], dtype=float)
    if frequency.ndim != 1 or log_gain.shape != frequency.shape or len(frequency) < 2:
        raise ValueError("frequency and log2 gain must be equal-length 1D arrays")
    if not np.isfinite(frequency).all() or not np.isfinite(log_gain).all():
        raise ValueError("frequency parameters contain non-finite values")
    if np.any(np.diff(frequency) <= 0):
        raise ValueError("frequency grid must be strictly increasing")

    channel_od = -np.log((array.astype(np.float32) + 1.0) / 256.0)
    mean_od = channel_od.mean(axis=-1)
    pad = 32
    padded = np.pad(mean_od, pad, mode="reflect")
    centered = padded - padded.mean()
    spectrum = np.fft.fft2(centered)
    fy = np.fft.fftfreq(padded.shape[0])[:, None]
    fx = np.fft.fftfreq(padded.shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    gain = np.exp2(np.interp(np.minimum(radius, frequency[-1]), frequency, log_gain))
    # The fitted curve is already clipped, but this guard also protects a
    # hand-edited parameter file at application time.
    gain = np.clip(gain, 0.5, 2.0)
    corrected = np.fft.ifft2(spectrum * gain).real + padded.mean()
    corrected = corrected[pad:-pad, pad:-pad].astype(np.float32)
    return _replace_od_luminance(array, corrected)


def apply_analytic_control(
    image: np.ndarray,
    parameter: Mapping[str, Any],
    arm: str,
) -> np.ndarray:
    """Apply one of raw, Reinhard, frequency-only, or combined."""

    _validate_parameter_direction(parameter)
    normalized = str(arm).strip().lower()
    if normalized not in ANALYTIC_ARMS:
        raise ValueError(f"arm must be one of {ANALYTIC_ARMS}")
    array = _validate_image(image)
    if normalized == "raw":
        return array.copy()
    if normalized == "reinhard":
        return apply_reinhard(array, parameter)
    if normalized == "frequency":
        return apply_frequency(array, parameter)
    return apply_frequency(apply_reinhard(array, parameter), parameter)


def apply_analytic_controls(
    image: np.ndarray,
    parameter: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Render all formal analytic arms for one source-scanner image."""

    return {
        arm: apply_analytic_control(image, parameter, arm) for arm in ANALYTIC_ARMS
    }


def save_parameters(path: str | Path, parameter: Mapping[str, Any]) -> None:
    """Atomically save a fitted parameter block as JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(parameter), indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def load_parameters(path: str | Path) -> dict[str, Any]:
    """Load a parameter JSON and reject wrong-direction or non-training fits."""

    payload = json.loads(Path(path).read_text())
    if payload.get("analytic_version") != ANALYTIC_VERSION:
        raise ValueError(f"unsupported analytic version: {payload.get('analytic_version')!r}")
    _validate_parameter_direction(payload)
    return payload

