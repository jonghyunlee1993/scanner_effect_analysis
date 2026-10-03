"""Frequency perturbations for matched information-removal and UNI sensitivity.

All filters operate on patch-mean-removed mean optical density with reflected
padding.  Colour differences are retained by adding the corrected luminance
back to all three OD channels.  The primary axis is cycles/pixel.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd

from manuscript_completion.contract import (
    ANALYSIS_ROOT,
    BANDS_CYC_PER_PIXEL,
    OUTPUT_ROOT,
    SCANNERS,
    UNI_V1_CHECKPOINT_SHA256,
    sha256,
    utc_now,
)


PAD = 32
ANALYSIS_VERSION = "manuscript_completion_frequency_v2"
TAPER_CYC_PER_PIXEL = 0.015
LOWPASS_CUTOFFS = (0.50, 0.35, 0.25, 0.18, 0.125)
DOSE_FRACTIONS = (0.25, 0.50)
MATCHED_BANDSTOP_FRACTION = 0.50
CALIBRATION_QUANTILE = 0.10
RMS_MEDIAN_RELATIVE_ERROR_MAX = 0.10
RMS_P90_RELATIVE_ERROR_MAX = 0.25
RMS_BAND_MEDIAN_RATIO_MAX = 1.10
IMPLEMENTATION_MANIFEST_ENV = "FREQUENCY_IMPLEMENTATION_MANIFEST"
IMPLEMENTATION_MANIFEST_SHA256_ENV = "FREQUENCY_IMPLEMENTATION_MANIFEST_SHA256"
METRIC_NAMES = (
    "perturbation_rms_od",
    "gradient_ncc_to_raw",
    "gradient_energy",
    "saturation_fraction",
    "band_power_low_mid",
    "band_power_mid",
    "band_power_high",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_implementation_manifest(
    manifest_path: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
    *,
    repository: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless code and immutable inputs match the submitted manifest."""

    selected = manifest_path or os.environ.get(IMPLEMENTATION_MANIFEST_ENV)
    expected = expected_manifest_sha256 or os.environ.get(
        IMPLEMENTATION_MANIFEST_SHA256_ENV
    )
    if not selected or not expected:
        raise RuntimeError(
            f"{IMPLEMENTATION_MANIFEST_ENV} and "
            f"{IMPLEMENTATION_MANIFEST_SHA256_ENV} are required"
        )
    path = Path(selected).resolve()
    observed_manifest_sha256 = sha256(path) if path.is_file() else "missing"
    if observed_manifest_sha256 != str(expected).lower():
        raise RuntimeError(
            "frequency implementation manifest SHA256 mismatch: "
            f"expected {expected}, observed {observed_manifest_sha256}"
        )
    payload = json.loads(path.read_text())
    if (
        payload.get("status")
        not in {"frozen_before_uni_outcomes", "frozen_after_computational_bugfix"}
        or payload.get("analysis_version") != ANALYSIS_VERSION
        or not isinstance(payload.get("files"), dict)
        or not isinstance(payload.get("inputs"), dict)
    ):
        raise RuntimeError(f"invalid frozen frequency implementation manifest: {path}")
    root = (
        Path(repository).resolve()
        if repository is not None
        else Path(__file__).resolve().parents[2]
    )
    mismatches: list[str] = []
    for relative, file_sha256 in payload["files"].items():
        candidate = (root / str(relative)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            mismatches.append(f"file escapes repository: {relative}")
            continue
        observed = sha256(candidate) if candidate.is_file() else "missing"
        if observed != str(file_sha256).lower():
            mismatches.append(
                f"{relative}: expected {file_sha256}, observed {observed}"
            )
    for name, item in payload["inputs"].items():
        if not isinstance(item, dict) or set(item) < {"path", "sha256"}:
            mismatches.append(f"input {name}: malformed manifest entry")
            continue
        candidate = Path(str(item["path"])).resolve()
        observed = sha256(candidate) if candidate.is_file() else "missing"
        if observed != str(item["sha256"]).lower():
            mismatches.append(
                f"input {name}: expected {item['sha256']}, observed {observed}"
            )
    if mismatches:
        raise RuntimeError(
            "frequency implementation contract mismatch:\n" + "\n".join(mismatches)
        )
    return {
        "path": str(path),
        "sha256": observed_manifest_sha256,
        "payload": payload,
    }


def expected_condition_names(mode: str) -> list[str]:
    if mode == "matched":
        return [
            "raw",
            *(f"lowpass_{str(value).replace('.', 'p')}" for value in LOWPASS_CUTOFFS),
            *(f"bandstop_{name}_rms" for name in BANDS_CYC_PER_PIXEL),
        ]
    if mode != "sensitivity":
        raise ValueError(f"unknown mode: {mode}")
    names = ["raw"]
    for dose_fraction in DOSE_FRACTIONS:
        dose_label = f"d{int(round(dose_fraction * 100)):02d}"
        for name in BANDS_CYC_PER_PIXEL:
            names.extend(
                [
                    f"amplitude_attenuation_{name}_{dose_label}",
                    f"amplitude_gain_{name}_{dose_label}",
                    f"phase_{name}_{dose_label}",
                ]
            )
    return names


def _contract_path(output_root: Path, mode: str) -> Path:
    relative = {
        "matched": "04_frequency_mechanism/02_matched_information_removal/matched_filter_contract_v2.json",
        "sensitivity": "04_frequency_mechanism/03_uni_band_sensitivity/band_perturbation_contract_v2.json",
    }[mode]
    return output_root / relative


def _legacy_contract_path(output_root: Path, mode: str) -> Path:
    relative = {
        "matched": "04_frequency_mechanism/02_matched_information_removal/matched_filter_contract.json",
        "sensitivity": "04_frequency_mechanism/03_uni_band_sensitivity/band_perturbation_contract.json",
    }[mode]
    return output_root / relative


def _calibration_path(output_root: Path) -> Path:
    return (
        output_root
        / "04_frequency_mechanism/03_uni_band_sensitivity/dose_calibration_v2.json"
    )


def prepare_contract(output_root: Path) -> dict[str, Any]:
    common = {
        "analysis_version": ANALYSIS_VERSION,
        "created_utc": utc_now(),
        "status": "frozen_before_uni_outcomes",
        "input_space": "patch-mean-removed mean optical density",
        "padding": {"pixels": PAD, "mode": "reflect"},
        "frequency_axis": "cycles_per_pixel",
        "bands_cyc_per_pixel": {
            name: list(bounds) for name, bounds in BANDS_CYC_PER_PIXEL.items()
        },
        "smooth_taper_cyc_per_pixel": TAPER_CYC_PER_PIXEL,
        "paired_rule": "the same filter and coefficient are applied to all six scanner images at a location",
        "uni_role": "frozen evaluator; no dose or condition is selected using embeddings",
        "uni_v1_checkpoint_sha256": UNI_V1_CHECKPOINT_SHA256,
        "inference_unit": "physical_slide",
        "revision_reason": (
            "Pre-outcome audit replaced held-out-image adaptive UNI-sensitivity doses with "
            "outer-fold training-slide-only calibration, corrected the tissue-probe estimand, "
            "added prespecified gradient-stratified nested models, and strengthened schema/RMS gates."
        ),
    }
    matched = {
        **common,
        "experiment": "matched_information_removal",
        "conditions": {
            "raw": True,
            "lowpass_cutoffs_cyc_per_pixel": list(LOWPASS_CUTOFFS),
            "bandstop": list(BANDS_CYC_PER_PIXEL),
        },
        "bandstop_calibration": (
            "per physical slide, target OD RMS is 0.5 times the minimum median band-component RMS; "
            "one band-specific coefficient is shared by all locations and scanners"
        ),
        "expected_conditions": expected_condition_names("matched"),
    }
    sensitivity = {
        **common,
        "experiment": "uni_band_sensitivity",
        "dose_definition": (
            "fraction of the outer-fold training-panel 10th percentile of per-image minimum "
            "low/mid/high band-component OD RMS; one frozen scalar is applied to every held-out "
            "scanner and location in that fold"
        ),
        "calibration_quantile": CALIBRATION_QUANTILE,
        "calibration_population": "all six scanner inputs from training physical slides only",
        "dose_fractions": list(DOSE_FRACTIONS),
        "amplitude_conditions": ["attenuation", "gain"],
        "phase_condition": (
            "deterministic partial phase replacement on a 17-point blend grid; "
            "original Fourier amplitude retained and achieved OD RMS matched input-only"
        ),
        "phase_seed": "sha256(slide_id, flattened image index, band, dose)",
        "gradient_quartiles": "within-scanner quartiles of raw-input gradient energy",
        "mixed_model": (
            "high-minus-comparator nested tissue/slide REML, reported overall and by scanner "
            "and raw-gradient quartile"
        ),
        "gradient_stratum_minimum_coverage": {
            "physical_slides": 80,
            "tissues": 30,
        },
        "achieved_rms_gate": {
            "amplitude_group_median_relative_error_max": RMS_MEDIAN_RELATIVE_ERROR_MAX,
            "amplitude_group_p90_relative_error_max": RMS_P90_RELATIVE_ERROR_MAX,
            "amplitude_band_median_max_to_min_ratio_max": RMS_BAND_MEDIAN_RATIO_MAX,
            "claim_rule": "primary frequency-sensitivity claim requires all amplitude RMS gates",
        },
        "expected_conditions": expected_condition_names("sensitivity"),
    }

    matched_path = _contract_path(output_root, "matched")
    sensitivity_path = _contract_path(output_root, "sensitivity")
    for mode, payload, destination in (
        ("matched", matched, matched_path),
        ("sensitivity", sensitivity, sensitivity_path),
    ):
        legacy = _legacy_contract_path(output_root, mode)
        payload["supersedes"] = (
            {"path": str(legacy.resolve()), "sha256": sha256(legacy)}
            if legacy.is_file()
            else None
        )
        if destination.exists():
            existing = json.loads(destination.read_text())
            if existing.get("status") != "frozen_before_uni_outcomes":
                raise ValueError(f"existing v2 contract is not frozen: {destination}")
        else:
            _write_json(destination, payload)
    result = {
        "status": "pass",
        "matched_contract": str(matched_path),
        "matched_contract_sha256": sha256(matched_path),
        "sensitivity_contract": str(sensitivity_path),
        "sensitivity_contract_sha256": sha256(sensitivity_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _calibration_slide(payload: dict[str, Any]) -> dict[str, Any]:
    slide_id = str(payload["slide_id"])
    locations = _location_indices(slide_id)
    with h5py.File(Path(payload["cache_path"]), "r") as store:
        images = np.asarray(store["images"][locations], dtype=np.uint8)
        scanner_names = tuple(value.decode().lower() for value in store["scanner_names"][:])
    if scanner_names != SCANNERS or images.shape[:2] != (20, len(SCANNERS)):
        raise ValueError(f"{slide_id}: calibration input schema mismatch")
    flat = images.reshape(-1, *images.shape[2:])
    _, _, spectrum, radius = od_spectrum(flat)
    components = band_components(spectrum, radius)
    rms = np.stack([_rms(components[name]) for name in BANDS_CYC_PER_PIXEL], axis=1)
    return {
        "slide_id": slide_id,
        "fold": int(payload["fold"]),
        "minimum_rms": rms.min(axis=1).reshape(20, len(SCANNERS)),
    }


def prepare_dose_calibration(output_root: Path, workers: int) -> dict[str, Any]:
    """Freeze outer-fold input-only dose scalars before any UNI outcomes are read."""
    contract_path = _contract_path(output_root, "sensitivity")
    if not contract_path.is_file():
        raise FileNotFoundError(f"prepare the v2 contract first: {contract_path}")
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != "frozen_before_uni_outcomes":
        raise ValueError("sensitivity contract is not frozen")
    destination = _calibration_path(output_root)
    if destination.exists():
        existing = json.loads(destination.read_text())
        if (
            existing.get("status") == "frozen_before_uni_outcomes"
            and existing.get("contract_sha256") == sha256(contract_path)
        ):
            print(json.dumps({"status": "already_complete", "path": str(destination)}))
            return existing
        raise ValueError(f"stale or incompatible calibration exists: {destination}")

    cohort_path = ANALYSIS_ROOT / "00_contract/cohort.csv"
    cohort = pd.read_csv(cohort_path, dtype={"slide_id": str})
    if len(cohort) != 103 or set(cohort["fold"]) != set(range(5)):
        raise ValueError("dose calibration requires the locked 103-slide five-fold cohort")
    records = cohort[["slide_id", "fold", "cache_path"]].to_dict("records")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        slide_results = list(executor.map(_calibration_slide, records))
    values = np.stack([item["minimum_rms"] for item in slide_results])
    folds = np.asarray([item["fold"] for item in slide_results], dtype=int)
    fold_payload: dict[str, Any] = {}
    for outer_fold in range(5):
        training = values[folds != outer_fold]
        base = float(np.quantile(training, CALIBRATION_QUANTILE))
        if not np.isfinite(base) or base <= 0:
            raise ValueError(f"fold {outer_fold}: invalid calibrated OD-RMS {base}")
        fold_payload[str(outer_fold)] = {
            "training_slides": int(np.sum(folds != outer_fold)),
            "held_out_slides": int(np.sum(folds == outer_fold)),
            "training_images": int(training.size),
            "base_rms_od": base,
            "training_minimum_rms_median": float(np.median(training)),
            "training_minimum_rms_q10": float(np.quantile(training, 0.10)),
            "training_minimum_rms_q90": float(np.quantile(training, 0.90)),
            "dose_target_rms_od": {
                f"d{int(round(dose * 100)):02d}": base * dose for dose in DOSE_FRACTIONS
            },
            "training_slide_ids": sorted(cohort.loc[cohort.fold != outer_fold, "slide_id"]),
        }
    result = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "frozen_before_uni_outcomes",
        "created_utc": utc_now(),
        "contract": str(contract_path.resolve()),
        "contract_sha256": sha256(contract_path),
        "cohort": str(cohort_path.resolve()),
        "cohort_sha256": sha256(cohort_path),
        "calibration_quantile": CALIBRATION_QUANTILE,
        "calibration_rule": (
            "For each outer fold, pool the per-image minimum low/mid/high component RMS over "
            "all six scanner inputs and 20 locked locations from training slides only; freeze q10."
        ),
        "folds": fold_payload,
    }
    _write_json(destination, result)
    print(json.dumps({"status": "pass", "path": str(destination), "sha256": sha256(destination)}))
    return result


def _smooth_step(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def frequency_radius(height: int, width: int) -> np.ndarray:
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    return np.sqrt(fx * fx + fy * fy)


def smooth_band_mask(
    radius: np.ndarray,
    lower: float,
    upper: float,
    taper: float = TAPER_CYC_PER_PIXEL,
) -> np.ndarray:
    if not 0.0 <= lower < upper <= 0.5:
        raise ValueError("band must lie within [0, 0.5] cycles/pixel")
    if taper <= 0:
        return ((radius >= lower) & (radius <= upper)).astype(np.float32)
    rising = _smooth_step((radius - (lower - taper)) / (2.0 * taper))
    falling = 1.0 - _smooth_step((radius - (upper - taper)) / (2.0 * taper))
    return np.minimum(rising, falling).astype(np.float32)


def smooth_lowpass_mask(
    radius: np.ndarray,
    cutoff: float,
    taper: float = TAPER_CYC_PER_PIXEL,
) -> np.ndarray:
    if not 0.0 < cutoff <= 0.5:
        raise ValueError("cutoff must lie within (0, 0.5]")
    start = max(cutoff - taper, 0.0)
    stop = min(cutoff + taper, math.sqrt(0.5), 0.5)
    if stop <= start:
        return (radius <= cutoff).astype(np.float32)
    return (1.0 - _smooth_step((radius - start) / (stop - start))).astype(np.float32)


def rgb_to_mean_od(images: np.ndarray) -> np.ndarray:
    array = np.asarray(images)
    if array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError("images must be uint8 [..., H, W, 3]")
    od = -np.log((array.astype(np.float32) + 1.0) / 256.0)
    return od.mean(axis=-1)


def replace_mean_od(images: np.ndarray, corrected_mean_od: np.ndarray) -> np.ndarray:
    array = np.asarray(images)
    original = rgb_to_mean_od(array)
    if corrected_mean_od.shape != original.shape:
        raise ValueError("corrected mean OD shape mismatch")
    channel_od = -np.log((array.astype(np.float32) + 1.0) / 256.0)
    shifted = channel_od + (corrected_mean_od - original)[..., None]
    rgb = np.exp(-np.clip(shifted, 0.0, 8.0)) * 256.0 - 1.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def od_spectrum(images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean_od = rgb_to_mean_od(images)
    padded = np.pad(mean_od, ((0, 0), (PAD, PAD), (PAD, PAD)), mode="reflect")
    mean = padded.mean(axis=(-2, -1), keepdims=True)
    spectrum = np.fft.fft2(padded - mean, axes=(-2, -1))
    radius = frequency_radius(padded.shape[-2], padded.shape[-1])
    return mean_od, mean, spectrum, radius


def reconstruct_mean_od(
    spectrum: np.ndarray,
    mean: np.ndarray,
    multiplier: np.ndarray,
) -> np.ndarray:
    restored = np.fft.ifft2(spectrum * multiplier, axes=(-2, -1)).real + mean
    return restored[..., PAD:-PAD, PAD:-PAD].astype(np.float32)


def band_components(
    spectrum: np.ndarray,
    radius: np.ndarray,
) -> dict[str, np.ndarray]:
    zero = np.zeros((*spectrum.shape[:-2], 1, 1), dtype=np.float32)
    return {
        name: reconstruct_mean_od(
            spectrum,
            zero,
            smooth_band_mask(radius, bounds[0], bounds[1]),
        )
        for name, bounds in BANDS_CYC_PER_PIXEL.items()
    }


def _rms(value: np.ndarray, axis: tuple[int, ...] = (-2, -1)) -> np.ndarray:
    return np.sqrt(np.mean(np.square(value, dtype=np.float64), axis=axis))


def _gradient(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gy = np.gradient(gray, axis=-2)
    gx = np.gradient(gray, axis=-1)
    return gx, gy


def _row_ncc(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = left.reshape(len(left), -1).astype(np.float64)
    b = right.reshape(len(right), -1).astype(np.float64)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(denominator, 1e-12)


def image_metrics(raw: np.ndarray, changed: np.ndarray) -> np.ndarray:
    raw_od = rgb_to_mean_od(raw)
    changed_od = rgb_to_mean_od(changed)
    gx_raw, gy_raw = _gradient(raw_od)
    gx, gy = _gradient(changed_od)
    raw_vector = np.concatenate([gx_raw.reshape(len(raw), -1), gy_raw.reshape(len(raw), -1)], axis=1)
    changed_vector = np.concatenate([gx.reshape(len(raw), -1), gy.reshape(len(raw), -1)], axis=1)
    gradient_energy = np.sqrt(np.mean(gx * gx + gy * gy, axis=(-2, -1)))
    saturation = ((changed <= 1) | (changed >= 254)).any(axis=-1).mean(axis=(-2, -1))
    _, _, spectrum, radius = od_spectrum(changed)
    power = np.abs(spectrum) ** 2
    band_values = []
    for bounds in BANDS_CYC_PER_PIXEL.values():
        mask = smooth_band_mask(radius, *bounds) > 0.5
        band_values.append(np.log10(np.maximum(power[..., mask].mean(axis=-1), 1e-12)))
    return np.column_stack(
        [
            _rms(changed_od - raw_od),
            _row_ncc(raw_vector, changed_vector),
            gradient_energy,
            saturation,
            *band_values,
        ]
    ).astype(np.float32)


def matched_removal_conditions(images: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply identical filters to every scanner at a location panel.

    ``images`` has shape ``[locations, scanners, H, W, 3]``.  Band-stop
    coefficients are calibrated once per slide across all locations/scanners,
    then shared by both sides of every paired comparison.
    """

    if images.ndim != 5 or images.shape[1] != len(SCANNERS):
        raise ValueError("expected [locations, 6, H, W, 3]")
    flat = images.reshape(-1, *images.shape[2:])
    raw_mean, mean, spectrum, radius = od_spectrum(flat)
    components = band_components(spectrum, radius)
    component_rms = {name: _rms(value) for name, value in components.items()}
    median_rms = {name: float(np.median(value)) for name, value in component_rms.items()}
    target_rms = MATCHED_BANDSTOP_FRACTION * min(median_rms.values())
    alpha = {
        name: float(np.clip(target_rms / max(value, 1e-8), 0.0, 1.0))
        for name, value in median_rms.items()
    }

    names = ["raw"]
    rendered = [flat]
    for cutoff in LOWPASS_CUTOFFS:
        label = f"lowpass_{str(cutoff).replace('.', 'p')}"
        names.append(label)
        corrected = reconstruct_mean_od(spectrum, mean, smooth_lowpass_mask(radius, cutoff))
        rendered.append(replace_mean_od(flat, corrected))
    for name, bounds in BANDS_CYC_PER_PIXEL.items():
        names.append(f"bandstop_{name}_rms")
        multiplier = 1.0 - alpha[name] * smooth_band_mask(radius, *bounds)
        corrected = reconstruct_mean_od(spectrum, mean, multiplier)
        rendered.append(replace_mean_od(flat, corrected))

    output = np.stack(rendered, axis=1)
    metric = np.stack([image_metrics(flat, output[:, index]) for index in range(len(names))], axis=1)
    shape = (images.shape[0], images.shape[1], len(names))
    return (
        names,
        output.reshape(*shape, *images.shape[2:]),
        metric.reshape(*shape, len(METRIC_NAMES)),
        {
            "band_component_median_rms": median_rms,
            "matched_target_rms": target_rms,
            "bandstop_alpha": alpha,
        },
    )


def _phase_scramble(
    spectrum: np.ndarray,
    mask: np.ndarray,
    blend: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(spectrum.shape[-2:])
    random_phase = np.angle(np.fft.fft2(noise))
    original_phase = np.angle(spectrum)
    delta = np.angle(np.exp(1j * (random_phase - original_phase)))
    phase = original_phase + blend * mask * delta
    changed = np.abs(spectrum) * np.exp(1j * phase)
    # Numerical symmetrization is supplied by taking the real inverse transform.
    return changed


def band_sensitivity_conditions(
    images: np.ndarray,
    *,
    slide_id: str,
    dose_base_rms_od: float,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    """Create equal-RMS perturbations using a frozen training-slide dose scalar."""

    if images.ndim != 5 or images.shape[1] != len(SCANNERS):
        raise ValueError("expected [locations, 6, H, W, 3]")
    if not np.isfinite(dose_base_rms_od) or dose_base_rms_od <= 0:
        raise ValueError("dose_base_rms_od must be finite and positive")
    flat = images.reshape(-1, *images.shape[2:])
    raw_mean, mean, spectrum, radius = od_spectrum(flat)
    components = band_components(spectrum, radius)
    component_rms = np.stack([_rms(components[name]) for name in BANDS_CYC_PER_PIXEL], axis=1)

    names = ["raw"]
    rendered = [flat]
    clipped_coefficients: dict[str, float] = {}
    target_by_condition: dict[str, float] = {"raw": 0.0}
    for dose_fraction in DOSE_FRACTIONS:
        dose_label = f"d{int(round(dose_fraction * 100)):02d}"
        target = float(dose_base_rms_od * dose_fraction)
        for band_index, (name, bounds) in enumerate(BANDS_CYC_PER_PIXEL.items()):
            alpha = np.clip(target / np.maximum(component_rms[:, band_index], 1e-8), 0.0, 1.0)
            mask = smooth_band_mask(radius, *bounds)
            for direction, sign in (("attenuation", -1.0), ("gain", 1.0)):
                condition = f"amplitude_{direction}_{name}_{dose_label}"
                names.append(condition)
                target_by_condition[condition] = target
                clipped_coefficients[condition] = float(np.mean(alpha >= 1.0 - 1e-8))
                multiplier = 1.0 + sign * alpha[:, None, None] * mask
                corrected = reconstruct_mean_od(spectrum, mean, multiplier)
                rendered.append(replace_mean_od(flat, corrected))

            phase_condition = f"phase_{name}_{dose_label}"
            names.append(phase_condition)
            target_by_condition[phase_condition] = target
            phase_images = []
            for image_index in range(len(flat)):
                seed_payload = f"{slide_id}|{image_index}|{name}|{dose_label}".encode()
                seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "little")
                best = None
                for blend in np.linspace(0.0, 1.0, 17):
                    changed_spectrum = _phase_scramble(
                        spectrum[image_index], mask, float(blend), seed
                    )
                    candidate = reconstruct_mean_od(
                        changed_spectrum[None], mean[image_index : image_index + 1], 1.0
                    )[0]
                    error = abs(float(_rms(candidate - raw_mean[image_index])) - target)
                    if best is None or error < best[0]:
                        best = (error, candidate)
                assert best is not None
                phase_images.append(best[1])
            rendered.append(replace_mean_od(flat, np.stack(phase_images)))

    output = np.stack(rendered, axis=1)
    metric = np.stack([image_metrics(flat, output[:, index]) for index in range(len(names))], axis=1)
    shape = (images.shape[0], images.shape[1], len(names))
    return (
        names,
        output.reshape(*shape, *images.shape[2:]),
        metric.reshape(*shape, len(METRIC_NAMES)),
        {
            "dose_definition": "fraction of a frozen outer-fold training-slide OD-RMS scalar",
            "dose_base_rms_od": float(dose_base_rms_od),
            "dose_fractions": list(DOSE_FRACTIONS),
            "phase_blend_grid": 17,
            "target_rms_od_by_condition": target_by_condition,
            "coefficient_clipped_fraction_by_condition": clipped_coefficients,
        },
    )


def _embed_images(images: np.ndarray, batch_size: int) -> np.ndarray:
    import torch
    import torch.nn.functional as torch_functional

    from prenorm.embedding import load_uni

    if not torch.cuda.is_available():
        raise RuntimeError("frequency perturbation UNI extraction requires CUDA")
    device = torch.device("cuda")
    model, size, mean, std = load_uni(device)
    flat = images.reshape(-1, *images.shape[-3:])
    output = []
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            block = flat[start : start + batch_size]
            tensor = torch.from_numpy(block.astype(np.float32)).permute(0, 3, 1, 2).div_(255.0)
            tensor = tensor.to(device, non_blocking=True)
            tensor = torch_functional.interpolate(
                tensor, size=(size, size), mode="bicubic", align_corners=False
            )
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with context:
                feature = model((tensor - mean) / std)
            feature = torch_functional.normalize(feature.float(), dim=1)
            output.append(feature.cpu().numpy())
    return np.concatenate(output).astype(np.float32).reshape(*images.shape[:-3], -1)


def _location_indices(slide_id: str) -> np.ndarray:
    path = ANALYSIS_ROOT / "03_uni/shards" / f"{slide_id}.h5"
    with h5py.File(path, "r") as store:
        return np.asarray(store["location_index"], dtype=np.int64)


def extract_task(
    mode: str,
    task_index: int,
    task_count: int,
    output_root: Path,
    batch_size: int,
) -> None:
    implementation = verify_implementation_manifest()
    implementation_sha256 = str(implementation["sha256"])
    cohort = pd.read_csv(ANALYSIS_ROOT / "00_contract/cohort.csv", dtype={"slide_id": str})
    if not 0 <= task_index < task_count:
        raise ValueError("task_index must be in [0, task_count)")
    relative = {
        "matched": "04_frequency_mechanism/02_matched_information_removal/shards",
        "sensitivity": "04_frequency_mechanism/03_uni_band_sensitivity/shards",
    }[mode]
    destination_root = output_root / relative
    destination_root.mkdir(parents=True, exist_ok=True)
    contract_path = _contract_path(output_root, mode)
    if not contract_path.is_file():
        raise FileNotFoundError(f"missing frozen v2 contract: {contract_path}")
    contract = json.loads(contract_path.read_text())
    if (
        contract.get("analysis_version") != ANALYSIS_VERSION
        or contract.get("status") != "frozen_before_uni_outcomes"
    ):
        raise ValueError(f"invalid v2 contract: {contract_path}")
    contract_hash = sha256(contract_path)
    dose_calibration = None
    dose_calibration_hash = None
    if mode == "sensitivity":
        calibration_path = _calibration_path(output_root)
        if not calibration_path.is_file():
            raise FileNotFoundError(f"missing frozen dose calibration: {calibration_path}")
        dose_calibration = json.loads(calibration_path.read_text())
        dose_calibration_hash = sha256(calibration_path)
        if (
            dose_calibration.get("status") != "frozen_before_uni_outcomes"
            or dose_calibration.get("contract_sha256") != contract_hash
        ):
            raise ValueError("dose calibration does not match the sensitivity contract")
    for row in cohort.iloc[task_index::task_count].itertuples(index=False):
        slide_id = str(row.slide_id)
        destination = destination_root / f"{slide_id}.h5"
        summary_path = destination.with_suffix(".summary.json")
        if destination.exists() and summary_path.exists():
            old = json.loads(summary_path.read_text())
            if (
                old.get("status") == "pass"
                and old.get("analysis_version") == ANALYSIS_VERSION
                and old.get("contract_sha256") == contract_hash
                and old.get("dose_calibration_sha256") == dose_calibration_hash
                and old.get("implementation_manifest_sha256") == implementation_sha256
                and old.get("output_sha256") == sha256(destination)
            ):
                print(json.dumps({"status": "already_complete", "slide_id": slide_id}), flush=True)
                continue
            raise ValueError(f"{slide_id}: incompatible pre-existing frequency shard")
        locations = _location_indices(slide_id)
        phenotype_path = (
            Path(__file__).resolve().parents[2]
            / "outputs/final_image_study_v1/02_image_phenotypes/shards"
            / slide_id
            / "location_metrics.h5"
        )
        with h5py.File(phenotype_path, "r") as phenotype_store:
            replicate_id = np.asarray(
                phenotype_store["replicate_id"][locations], dtype=np.int16
            )
        with h5py.File(Path(row.cache_path), "r") as store:
            images = np.asarray(store["images"][locations], dtype=np.uint8)
            names = tuple(value.decode().lower() for value in store["scanner_names"][:])
            source_index = np.asarray(store["source_index"][locations], dtype=np.int64)
        if names != SCANNERS:
            raise ValueError(f"{slide_id}: scanner order mismatch: {names}")
        if mode == "matched":
            condition_names, rendered, metrics, calibration = matched_removal_conditions(images)
        else:
            assert dose_calibration is not None
            dose_base = float(dose_calibration["folds"][str(int(row.fold))]["base_rms_od"])
            condition_names, rendered, metrics, calibration = band_sensitivity_conditions(
                images, slide_id=slide_id, dose_base_rms_od=dose_base
            )
        if condition_names != expected_condition_names(mode):
            raise AssertionError(f"{slide_id}: generated condition schema differs from contract")
        features = _embed_images(rendered, batch_size)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        with h5py.File(temporary, "w") as store:
            store.attrs["analysis_version"] = ANALYSIS_VERSION
            store.attrs["mode"] = mode
            store.attrs["slide_id"] = slide_id
            store.attrs["tissue_type"] = str(row.tissue_type)
            store.attrs["fold"] = int(row.fold)
            store.attrs["calibration_json"] = json.dumps(calibration, sort_keys=True)
            store.attrs["contract_sha256"] = contract_hash
            store.attrs["dose_calibration_sha256"] = dose_calibration_hash or ""
            store.attrs["implementation_manifest_sha256"] = implementation_sha256
            store.attrs["uni_v1_checkpoint_sha256"] = UNI_V1_CHECKPOINT_SHA256
            store.create_dataset("features", data=features, compression="lzf")
            store.create_dataset("image_metrics", data=metrics, compression="lzf")
            store.create_dataset("condition_names", data=np.asarray(condition_names, dtype="S64"))
            store.create_dataset("metric_names", data=np.asarray(METRIC_NAMES, dtype="S48"))
            store.create_dataset("scanner_names", data=np.asarray(SCANNERS, dtype="S8"))
            store.create_dataset("location_index", data=locations)
            store.create_dataset("source_index", data=source_index)
            store.create_dataset("replicate_id", data=replicate_id)
        temporary.replace(destination)
        summary = {
            "status": "pass",
            "analysis_version": ANALYSIS_VERSION,
            "mode": mode,
            "slide_id": slide_id,
            "locations": len(locations),
            "scanners": len(SCANNERS),
            "conditions": len(condition_names),
            "feature_dim": int(features.shape[-1]),
            "contract_sha256": contract_hash,
            "dose_calibration_sha256": dose_calibration_hash,
            "implementation_manifest": str(implementation["path"]),
            "implementation_manifest_sha256": implementation_sha256,
            "uni_v1_checkpoint_sha256": UNI_V1_CHECKPOINT_SHA256,
            "output_sha256": sha256(destination),
            "completed_utc": utc_now(),
        }
        temporary_summary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
        temporary_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        temporary_summary.replace(summary_path)
        print(json.dumps(summary), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-contract")
    calibration = subparsers.add_parser("prepare-calibration")
    calibration.add_argument("--workers", type=int, default=1)
    extract = subparsers.add_parser("extract-task")
    extract.add_argument("--mode", choices=("matched", "sensitivity"), required=True)
    extract.add_argument("--task-index", type=int, required=True)
    extract.add_argument("--task-count", type=int, required=True)
    extract.add_argument("--batch-size", type=int, default=48)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare-contract":
        prepare_contract(args.output_root)
        return 0
    if args.command == "prepare-calibration":
        prepare_dose_calibration(args.output_root, args.workers)
        return 0
    extract_task(args.mode, args.task_index, args.task_count, args.output_root, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
