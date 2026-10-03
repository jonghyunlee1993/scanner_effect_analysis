#!/usr/bin/env python3
"""Test whether common augmentations can reach paired scanner phenotypes.

This is an intentionally generous test of the augmentation-support hypothesis.
For every selected AT2 patch, a fixed Sobol library of conventional pathology
augmentations is rendered.  Both a scanner-global choice and a per-image
finite-search oracle are evaluated on held-out slides.  The oracle is allowed
to inspect the paired target phenotype, so failure is evidence against the
augmentation family rather than a parameter-selection algorithm.

The experiment is image-first.  A representative frozen-PFM bridge is handled
by ``augmentation_ood_uni.py`` after the image-domain choices are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fft
from scipy.stats import qmc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.color import hed2rgb, rgb2hed
import cv2

from final_image_study import (
    BANDS_CYC_PER_UM,
    PROVISIONAL_MPP,
    REPLICATE_GROUPS,
    SCANNERS,
    SPECTRAL_BINS,
    frequency_geometry,
    geometric_band,
    image_metrics,
    normalized_transfer,
    radial_mean,
    rowwise_correlation,
)


ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = ROOT / "outputs/final_image_study_v1"
OUTPUT_ROOT = ROOT / "outputs/augmentation_ood_v1"
ANALYSIS_VERSION = "augmentation_ood_v1.1"
REFERENCE = "at2"
TARGET_SCANNERS = SCANNERS[1:]
PATCHES_PER_SLIDE = 40
FIDELITY_NCC_MIN = 0.90
SATURATION_INCREASE_MAX = 0.05
SATURATION_ABSOLUTE_MAX = 0.10
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260916

PARAMETERS = (
    "brightness",
    "contrast",
    "saturation",
    "gamma",
    "hed_h_scale",
    "hed_e_scale",
    "blur_sigma",
    "sharpen_amount",
    "noise_sd",
)

RANGES = {
    "default": {
        "brightness": (0.80, 1.20),
        "contrast": (0.80, 1.20),
        "saturation": (0.80, 1.20),
        "gamma": (0.80, 1.20),
        "hed_h_scale": (0.85, 1.15),
        "hed_e_scale": (0.85, 1.15),
        "blur_sigma": (0.0, 1.0),
        "sharpen_amount": (0.0, 0.5),
        "noise_sd": (0.0, 0.025),
    },
    "strong": {
        "brightness": (0.40, 1.60),
        "contrast": (0.40, 1.80),
        "saturation": (0.40, 1.80),
        "gamma": (0.50, 1.80),
        "hed_h_scale": (0.50, 1.60),
        "hed_e_scale": (0.50, 1.60),
        "blur_sigma": (0.0, 3.0),
        "sharpen_amount": (0.0, 2.0),
        "noise_sd": (0.0, 0.060),
    },
}

LIBRARY_SIZES = {"default": 64, "strong": 256}

ENDPOINTS = (
    "delta_lab_l",
    "delta_lab_a",
    "delta_lab_b",
    "log2_mean_od_ratio",
    "log2_lab_l_sd_ratio",
    "log2_od_sd_ratio",
    "log2_gradient_rms_ratio",
    "frequency_low_mid",
    "frequency_mid",
    "frequency_high",
)

ARMS = (
    "identity",
    "default_global",
    "default_oracle",
    "strong_global",
    "strong_unconstrained_oracle",
    "strong_oracle",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*values: object) -> int:
    token = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)


def augmentation_seed(
    slide_id: str,
    location_index: int,
    parameters: np.ndarray,
) -> int:
    """Bind stochastic rendering to the transform, not its library label."""
    parameter_token = np.asarray(parameters, dtype=np.float32).tobytes().hex()
    return stable_seed(slide_id, location_index, parameter_token)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def make_library(name: str) -> np.ndarray:
    size = LIBRARY_SIZES[name]
    power = int(round(math.log2(size)))
    if 2**power != size:
        raise ValueError("Sobol library size must be a power of two")
    # Five extra Sobol dimensions gate operation groups.  Conventional
    # pipelines do not apply blur, sharpening, noise, stain, and colour at
    # maximum strength simultaneously on every view.
    unit = qmc.Sobol(
        d=len(PARAMETERS) + 5,
        scramble=True,
        seed=stable_seed(ANALYSIS_VERSION, name),
    ).random_base2(power)
    ranges = RANGES[name]
    values = np.empty((size, len(PARAMETERS)), dtype=np.float32)
    for index, parameter in enumerate(PARAMETERS):
        low, high = ranges[parameter]
        values[:, index] = low + unit[:, index] * (high - low)
    color_gate, stain_gate, blur_gate, sharpen_gate, noise_gate = (
        unit[:, len(PARAMETERS) + offset] for offset in range(5)
    )
    values[color_gate >= 0.85, 0:4] = 1.0
    values[stain_gate >= 0.65, 4:6] = 1.0
    values[blur_gate >= 0.35, 6] = 0.0
    values[sharpen_gate >= 0.35, 7] = 0.0
    both = (values[:, 6] > 0) & (values[:, 7] > 0)
    keep_blur = blur_gate < sharpen_gate
    values[both & keep_blur, 7] = 0.0
    values[both & ~keep_blur, 6] = 0.0
    values[noise_gate >= 0.35, 8] = 0.0
    identity = np.asarray(
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    if name == "strong":
        # A wider search must be a strict superset of the default search for
        # the oracle comparison to be monotone.  Keep the declared total at
        # 257 candidates: identity + 64 default + 192 wide-range draws.
        default_nonidentity = make_library("default")[1:]
        wide_draws = values[: LIBRARY_SIZES["strong"] - len(default_nonidentity)]
        return np.concatenate([identity, default_nonidentity, wide_draws], axis=0)
    return np.concatenate([identity, values], axis=0)


def prepare(source_root: Path, output_root: Path) -> None:
    source_cohort = source_root / "00_contract/cohort.csv"
    source_summary = source_root / "summary.json"
    cohort = pd.read_csv(source_cohort, dtype={"slide_id": str})
    if len(cohort) != 103:
        raise ValueError(f"expected 103 slides, got {len(cohort)}")
    for relative in (
        "00_contract",
        "01_reachability/shards",
        "02_aggregate",
        "03_pfm_uni/shards",
        "logs",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    cohort_path = output_root / "00_contract/cohort.csv"
    cohort.to_csv(cohort_path, index=False)

    libraries = {name: make_library(name).tolist() for name in RANGES}
    library_path = output_root / "00_contract/augmentation_libraries.json"
    write_json(
        library_path,
        {
            "analysis_version": ANALYSIS_VERSION,
            "parameters": list(PARAMETERS),
            "ranges": RANGES,
            "libraries_include_identity_at_index_zero": True,
            "libraries": libraries,
        },
    )
    contract = {
        "analysis_version": ANALYSIS_VERSION,
        "created_utc": utc_now(),
        "hypothesis": (
            "Common content-preserving augmentation families do not cover the paired "
            "scanner/acquisition-pipeline phenotype, even under strong per-image finite search."
        ),
        "primary_unit": "paired image location",
        "inference_unit": "physical slide",
        "reference_scanner": REFERENCE,
        "target_scanners": list(TARGET_SCANNERS),
        "slides": len(cohort),
        "locations_per_slide": PATCHES_PER_SLIDE,
        "selection": "four deterministic source-index locations from each of ten spatial replicates",
        "endpoints": list(ENDPOINTS),
        "arms": list(ARMS),
        "oracle_definition": (
            "minimum standardized endpoint RMSE over a fixed finite Sobol library; "
            "the paired target is visible only for candidate selection"
        ),
        "operation_application": (
            "Sobol draws with probabilistic colour, HED stain, mutually exclusive blur/sharpen, "
            "and noise groups; the strong library is a strict superset of the default library; "
            "both unconstrained and fidelity-constrained strong oracles"
        ),
        "fidelity_gate": {
            "gradient_ncc_to_source_minimum": FIDELITY_NCC_MIN,
            "saturation_increase_maximum": SATURATION_INCREASE_MAX,
            "saturation_absolute_minimum_ceiling": SATURATION_ABSOLUTE_MAX,
        },
        "physical_frequency_status": "provisional_at_0.5052_um_per_pixel",
        "claim_boundary": (
            "Tests a declared proxy for commonly used pathology augmentation, not the "
            "undisclosed exact pretraining distribution of every PFM.  The observed effect "
            "is acquisition-pipeline, not isolated optics."
        ),
        "decision_interpretation": {
            "default_fails_strong_succeeds": "range mismatch",
            "global_fails_oracle_succeeds": "image/tissue-conditional parameter mismatch",
            "unconstrained_succeeds_fidelity_oracle_fails": "destructive mimicry only",
            "strong_oracle_fails_scanner_informed_succeeds": "augmentation-family support mismatch",
            "image_mimic_succeeds_pfm_separates": "representation/objective mismatch",
        },
        "source": {
            "cohort": str(source_cohort.resolve()),
            "cohort_sha256": sha256(source_cohort),
            "summary": str(source_summary.resolve()),
            "summary_sha256": sha256(source_summary),
        },
        "augmentation_library_sha256": sha256(library_path),
    }
    write_json(output_root / "00_contract/analysis_contract.json", contract)
    print(json.dumps(contract, indent=2))


def load_libraries(output_root: Path) -> dict[str, np.ndarray]:
    payload = json.loads(
        (output_root / "00_contract/augmentation_libraries.json").read_text()
    )
    if payload["parameters"] != list(PARAMETERS):
        raise ValueError("augmentation parameter order changed")
    return {
        name: np.asarray(values, dtype=np.float32)
        for name, values in payload["libraries"].items()
    }


def choose_locations(metrics_path: Path) -> np.ndarray:
    with h5py.File(metrics_path, "r") as store:
        replicate = np.asarray(store["replicate_id"], dtype=int)
        source_index = np.asarray(store["source_index"], dtype=np.int64)
    selected = []
    for group in range(REPLICATE_GROUPS):
        indices = np.flatnonzero(replicate == group)
        order = indices[np.argsort(source_index[indices], kind="stable")]
        if len(order) < PATCHES_PER_SLIDE // REPLICATE_GROUPS:
            raise ValueError(f"replicate {group} has too few locations")
        selected.extend(order[: PATCHES_PER_SLIDE // REPLICATE_GROUPS].tolist())
    return np.sort(np.asarray(selected, dtype=np.int64))


def render_augmentation(
    source_rgb: np.ndarray,
    source_hed: np.ndarray,
    parameters: np.ndarray,
    noise_seed: int,
) -> np.ndarray:
    values = dict(zip(PARAMETERS, np.asarray(parameters, dtype=float)))
    if (
        np.allclose(parameters[:6], 1.0)
        and np.allclose(parameters[6:], 0.0)
    ):
        return source_rgb.copy()
    hed = source_hed.copy()
    hed[..., 0] *= values["hed_h_scale"]
    hed[..., 1] *= values["hed_e_scale"]
    rgb = np.clip(hed2rgb(hed), 0.0, 1.0).astype(np.float32)
    gray = np.sum(rgb * np.asarray([0.299, 0.587, 0.114], dtype=np.float32), axis=-1)
    rgb = gray[..., None] + values["saturation"] * (rgb - gray[..., None])
    mean_gray = float(gray.mean())
    rgb = mean_gray + values["contrast"] * (rgb - mean_gray)
    rgb *= values["brightness"]
    rgb = np.power(np.clip(rgb, 0.0, 1.0), values["gamma"])
    blur_sigma = values["blur_sigma"]
    if blur_sigma > 0.02:
        rgb = cv2.GaussianBlur(rgb, (0, 0), blur_sigma)
    sharpen = values["sharpen_amount"]
    if sharpen > 0.01:
        lowpass = cv2.GaussianBlur(rgb, (0, 0), 1.0)
        rgb = rgb + sharpen * (rgb - lowpass)
    noise_sd = values["noise_sd"]
    if noise_sd > 1e-6:
        rng = np.random.default_rng(noise_seed)
        rgb += rng.normal(0.0, noise_sd, rgb.shape).astype(np.float32)
    return np.clip(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)


def radial_power_batch(od: np.ndarray, workers: int) -> np.ndarray:
    size = od.shape[-1]
    window_1d = np.hanning(size).astype(np.float32)
    window = window_1d[:, None] * window_1d[None, :]
    centered = od - od.mean(axis=(1, 2), keepdims=True)
    fourier = fft.fft2(centered * window[None, :, :], axes=(-2, -1), workers=workers)
    power = np.abs(fourier) ** 2
    index, valid, counts, _ = frequency_geometry(size, SPECTRAL_BINS)
    result = np.empty((len(od), SPECTRAL_BINS), dtype=np.float64)
    for row in range(len(od)):
        result[row] = radial_mean(power[row], index, valid, counts)
    return result


def absolute_measurements(
    images: np.ndarray,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scalar, od, gradient = image_metrics(images)
    radial = radial_power_batch(od, workers)
    return scalar, radial, gradient, od


def contrasts_from_absolute(
    scalar: np.ndarray,
    radial: np.ndarray,
    source_scalar: np.ndarray,
    source_radial: np.ndarray,
) -> np.ndarray:
    if len(source_scalar) == 1 and len(scalar) != 1:
        source_scalar = np.repeat(source_scalar, len(scalar), axis=0)
        source_radial = np.repeat(source_radial, len(radial), axis=0)
    _, _, _, frequency_px = frequency_geometry(256, SPECTRAL_BINS)
    frequency_um = frequency_px / PROVISIONAL_MPP
    rows = []
    for index in range(len(scalar)):
        transfer = normalized_transfer(radial[index], source_radial[index], frequency_um)
        bands = [
            math.log2(geometric_band(transfer, frequency_um, bounds))
            for bounds in BANDS_CYC_PER_UM.values()
        ]
        rows.append(
            [
                scalar[index, 0] - source_scalar[index, 0],
                scalar[index, 1] - source_scalar[index, 1],
                scalar[index, 2] - source_scalar[index, 2],
                math.log2(max(float(scalar[index, 4]), 1e-12) / max(float(source_scalar[index, 4]), 1e-12)),
                math.log2(max(float(scalar[index, 3]), 1e-12) / max(float(source_scalar[index, 3]), 1e-12)),
                math.log2(max(float(scalar[index, 5]), 1e-12) / max(float(source_scalar[index, 5]), 1e-12)),
                math.log2(max(float(scalar[index, 6]), 1e-12) / max(float(source_scalar[index, 6]), 1e-12)),
                *bands,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def candidate_measurements(
    source_rgb: np.ndarray,
    parameters: np.ndarray,
    slide_id: str,
    location_index: int,
    arm: str,
    workers: int,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_scalar, source_radial, source_gradient, _ = absolute_measurements(
        source_rgb[None], workers
    )
    source_hed = rgb2hed(source_rgb.astype(np.float32) / 255.0).astype(np.float32)
    vectors = np.empty((len(parameters), len(ENDPOINTS)), dtype=np.float32)
    fidelity = np.empty(len(parameters), dtype=np.float32)
    saturation = np.empty(len(parameters), dtype=np.float32)
    for start in range(0, len(parameters), batch_size):
        stop = min(start + batch_size, len(parameters))
        images = np.stack(
            [
                render_augmentation(
                    source_rgb,
                    source_hed,
                    parameters[candidate],
                    augmentation_seed(
                        slide_id,
                        location_index,
                        parameters[candidate],
                    ),
                )
                for candidate in range(start, stop)
            ],
            axis=0,
        )
        scalar, radial, gradient, _ = absolute_measurements(images, workers)
        vectors[start:stop] = contrasts_from_absolute(
            scalar, radial, source_scalar, source_radial
        )
        fidelity[start:stop] = rowwise_correlation(
            gradient, np.repeat(source_gradient, len(images), axis=0)
        )
        saturation[start:stop] = scalar[:, 8]
    return vectors, fidelity, saturation


def extract_slide(
    slide_index: int,
    source_root: Path,
    output_root: Path,
    workers: int,
) -> None:
    cohort_path = output_root / "00_contract/cohort.csv"
    cohort = pd.read_csv(cohort_path, dtype={"slide_id": str})
    if slide_index < 0 or slide_index >= len(cohort):
        raise IndexError(slide_index)
    row = cohort.iloc[slide_index]
    slide_id = str(row["slide_id"])
    output = output_root / "01_reachability/shards" / f"{slide_id}.npz"
    summary_path = output.with_suffix(".summary.json")
    library_path = output_root / "00_contract/augmentation_libraries.json"
    if output.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("analysis_version") == ANALYSIS_VERSION
            and summary.get("status") == "pass"
            and summary.get("augmentation_library_sha256") == sha256(library_path)
            and summary.get("output_sha256") == sha256(output)
        ):
            print(json.dumps({"status": "already_complete", "slide_id": slide_id}))
            return

    libraries = load_libraries(output_root)
    metrics_path = source_root / "02_image_phenotypes/shards" / slide_id / "location_metrics.h5"
    selected = choose_locations(metrics_path)
    cache_path = Path(row["cache_path"])
    with h5py.File(cache_path, "r") as store:
        images = np.asarray(store["images"][selected], dtype=np.uint8)
        source_index = np.asarray(store["source_index"][selected], dtype=np.int64)
        coords = np.asarray(store["phase_coords_xy"][selected, 0], dtype=np.int64)

    source_scalar, source_radial, source_gradient, _ = absolute_measurements(
        images[:, 0], workers
    )
    target_vectors = np.empty(
        (len(selected), len(TARGET_SCANNERS), len(ENDPOINTS)), dtype=np.float32
    )
    target_fidelity = np.empty((len(selected), len(TARGET_SCANNERS)), dtype=np.float32)
    for scanner_index in range(1, len(SCANNERS)):
        scalar, radial, gradient, _ = absolute_measurements(
            images[:, scanner_index], workers
        )
        target_vectors[:, scanner_index - 1] = contrasts_from_absolute(
            scalar, radial, source_scalar, source_radial
        )
        target_fidelity[:, scanner_index - 1] = rowwise_correlation(
            gradient, source_gradient
        )

    payload: dict[str, np.ndarray] = {
        "location_index": selected,
        "source_index": source_index,
        "phase_coords_xy": coords,
        "target_vectors": target_vectors,
        "target_fidelity": target_fidelity,
        "source_saturation": source_scalar[:, 8].astype(np.float32),
        "endpoint_names": np.asarray(ENDPOINTS, dtype="S40"),
        "target_scanners": np.asarray(TARGET_SCANNERS, dtype="S8"),
    }
    for arm, parameters in libraries.items():
        vectors = np.empty(
            (len(selected), len(parameters), len(ENDPOINTS)), dtype=np.float32
        )
        fidelity = np.empty((len(selected), len(parameters)), dtype=np.float32)
        saturation = np.empty((len(selected), len(parameters)), dtype=np.float32)
        for image_index, location_index in enumerate(selected):
            result = candidate_measurements(
                images[image_index, 0],
                parameters,
                slide_id,
                int(location_index),
                arm,
                workers,
            )
            vectors[image_index], fidelity[image_index], saturation[image_index] = result
            print(
                f"[{slide_id}] {arm} {image_index + 1}/{len(selected)}",
                flush=True,
            )
        payload[f"{arm}_vectors"] = vectors
        payload[f"{arm}_fidelity"] = fidelity
        payload[f"{arm}_saturation"] = saturation

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(output)
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "pass",
        "completed_utc": utc_now(),
        "slide_id": slide_id,
        "tissue_type": str(row["tissue_type"]),
        "locations": len(selected),
        "target_scanners": list(TARGET_SCANNERS),
        "endpoints": list(ENDPOINTS),
        "library_sizes_including_identity": {
            name: len(values) for name, values in libraries.items()
        },
        "augmentation_library_sha256": sha256(library_path),
        "source_cache": str(cache_path),
        "source_cache_bytes": cache_path.stat().st_size,
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


def robust_scale(values: np.ndarray) -> np.ndarray:
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.std(values, axis=0, ddof=1)
    scale = np.where(scale > 1e-6, scale, fallback)
    return np.maximum(scale, 1e-4)


def candidate_valid(
    fidelity: np.ndarray,
    saturation: np.ndarray,
    source_saturation: np.ndarray,
) -> np.ndarray:
    ceiling = np.maximum(
        source_saturation[:, None] + SATURATION_INCREASE_MAX,
        SATURATION_ABSOLUTE_MAX,
    )
    return (fidelity >= FIDELITY_NCC_MIN) & (saturation <= ceiling)


def distances(candidates: np.ndarray, target: np.ndarray, scale: np.ndarray) -> np.ndarray:
    residual = (candidates - target[:, None, :]) / scale[None, None, :]
    return np.sqrt(np.mean(np.square(residual), axis=2))


def bootstrap_mean_ci(frame: pd.DataFrame, column: str, seed: int) -> tuple[float, float, float]:
    slide_values = frame.groupby("slide_id")[column].mean().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(slide_values, size=(BOOTSTRAP_REPLICATES, len(slide_values)), replace=True)
    distribution = draws.mean(axis=1)
    return (
        float(slide_values.mean()),
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    )


def classifier_predictions(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    endpoint_columns = [f"generated_{name}" for name in ENDPOINTS]
    target_columns = [f"target_{name}" for name in ENDPOINTS]
    for arm in ARMS:
        for scanner in TARGET_SCANNERS:
            subset = selected[
                (selected["arm"] == arm) & (selected["scanner"] == scanner)
            ]
            for fold in sorted(subset["fold"].unique()):
                train = subset[subset["fold"] != fold]
                test = subset[subset["fold"] == fold]
                x_train = np.concatenate(
                    [
                        train[endpoint_columns].to_numpy(float),
                        train[target_columns].to_numpy(float),
                    ],
                    axis=0,
                )
                y_train = np.concatenate(
                    [np.zeros(len(train), dtype=int), np.ones(len(train), dtype=int)]
                )
                x_test = np.concatenate(
                    [
                        test[endpoint_columns].to_numpy(float),
                        test[target_columns].to_numpy(float),
                    ],
                    axis=0,
                )
                y_test = np.concatenate(
                    [np.zeros(len(test), dtype=int), np.ones(len(test), dtype=int)]
                )
                classifier = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=0.1,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=BOOTSTRAP_SEED,
                    ),
                )
                classifier.fit(x_train, y_train)
                prediction = classifier.predict(x_test)
                metadata = pd.concat(
                    [
                        test[["slide_id", "location_index"]],
                        test[["slide_id", "location_index"]],
                    ],
                    ignore_index=True,
                )
                for index, item in metadata.reset_index(drop=True).iterrows():
                    rows.append(
                        {
                            "arm": arm,
                            "scanner": scanner,
                            "fold": int(fold),
                            "slide_id": str(item.slide_id),
                            "location_index": int(item.location_index),
                            "truth": int(y_test[index]),
                            "prediction": int(prediction[index]),
                            "correct": int(prediction[index] == y_test[index]),
                        }
                    )
    return pd.DataFrame(rows)


def make_figures(
    selected: pd.DataFrame,
    arm_summary: pd.DataFrame,
    output_root: Path,
) -> None:
    arm_order = list(ARMS)
    colors = {
        "identity": "#8c9698",
        "default_global": "#7aa4af",
        "default_oracle": "#175d72",
        "strong_global": "#d19a52",
        "strong_unconstrained_oracle": "#7c3aed",
        "strong_oracle": "#a63d40",
    }
    figure, axis = plt.subplots(figsize=(12.5, 6.2), constrained_layout=True)
    width = 0.13
    x = np.arange(len(TARGET_SCANNERS))
    for offset, arm in enumerate(arm_order):
        part = arm_summary[
            (arm_summary["arm"] == arm) & (arm_summary["scanner"] != "all")
        ].set_index("scanner").reindex(TARGET_SCANNERS)
        position = x + (offset - 2.5) * width
        mean = part["distance_mean"].to_numpy()
        axis.errorbar(
            position,
            mean,
            yerr=np.vstack(
                [
                    mean - part["distance_ci_low"].to_numpy(),
                    part["distance_ci_high"].to_numpy() - mean,
                ]
            ),
            fmt="o",
            capsize=3,
            color=colors[arm],
            label=arm.replace("_", " "),
        )
    axis.axhline(1.0, color="#5a6b70", ls="--", lw=1, label="1 robust-SD RMSE")
    axis.set_xticks(x, [scanner.upper() for scanner in TARGET_SCANNERS])
    axis.set_ylabel("Target-mimic error (standardized RMSE)")
    axis.set_title(
        "Can conventional augmentation reach the paired scanner phenotype?",
        loc="left",
        fontweight="bold",
    )
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(output_root / "02_aggregate/figure01_reachability.png", dpi=180)
    plt.close(figure)

    classifier = arm_summary[arm_summary["scanner"] != "all"]
    figure, axis = plt.subplots(figsize=(12.5, 6.2), constrained_layout=True)
    for offset, arm in enumerate(arm_order):
        part = classifier[classifier["arm"] == arm].set_index("scanner").reindex(
            TARGET_SCANNERS
        )
        position = x + (offset - 2.5) * width
        mean = part["classifier_balanced_accuracy"].to_numpy()
        axis.errorbar(
            position,
            mean,
            yerr=np.vstack(
                [
                    mean - part["classifier_ci_low"].to_numpy(),
                    part["classifier_ci_high"].to_numpy() - mean,
                ]
            ),
            fmt="o",
            capsize=3,
            color=colors[arm],
            label=arm.replace("_", " "),
        )
    axis.axhline(0.5, color="#5a6b70", ls="--", lw=1, label="chance")
    axis.set_xticks(x, [scanner.upper() for scanner in TARGET_SCANNERS])
    axis.set_ylim(0.45, 1.02)
    axis.set_ylabel("Held-out slide balanced accuracy")
    axis.set_title(
        "Real scanner vs best-matching augmented AT2 remains distinguishable",
        loc="left",
        fontweight="bold",
    )
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(output_root / "02_aggregate/figure02_real_vs_augmented.png", dpi=180)
    plt.close(figure)

    oracle = selected[selected["arm"] == "strong_oracle"].copy()
    residual = np.column_stack(
        [
            oracle[f"generated_{endpoint}"] - oracle[f"target_{endpoint}"]
            for endpoint in ENDPOINTS
        ]
    )
    scale_matrix = np.column_stack(
        [oracle[f"scale_{endpoint}"].to_numpy(float) for endpoint in ENDPOINTS]
    )
    standardized = residual / scale_matrix
    matrix = []
    for scanner in TARGET_SCANNERS:
        selected_scanner = oracle["scanner"].to_numpy() == scanner
        matrix.append(np.mean(np.abs(standardized[selected_scanner]), axis=0))
    matrix = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    image = axis.imshow(matrix, cmap="magma", vmin=0, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.1f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > np.nanmedian(matrix) else "#16252b",
                fontsize=8,
            )
    axis.set_yticks(range(len(TARGET_SCANNERS)), [item.upper() for item in TARGET_SCANNERS])
    axis.set_xticks(range(len(ENDPOINTS)), [item.replace("_", " ") for item in ENDPOINTS], rotation=55, ha="right")
    axis.set_title(
        "Residual phenotype after strong per-image finite-search oracle",
        loc="left",
        fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035)
    colorbar.set_label("Mean absolute residual (robust SD units)")
    figure.savefig(output_root / "02_aggregate/figure03_oracle_residual.png", dpi=180)
    plt.close(figure)

    tissue = (
        oracle.groupby(["scanner", "tissue_type"])["distance"]
        .mean()
        .unstack("tissue_type")
        .reindex(index=TARGET_SCANNERS)
    )
    order = tissue.mean(axis=0).sort_values().index
    tissue = tissue[order]
    figure, axis = plt.subplots(figsize=(16, 5.4), constrained_layout=True)
    image = axis.imshow(tissue, cmap="viridis", vmin=0, aspect="auto")
    axis.set_yticks(range(len(TARGET_SCANNERS)), [item.upper() for item in TARGET_SCANNERS])
    axis.set_xticks(range(tissue.shape[1]), tissue.columns, rotation=65, ha="right", fontsize=7)
    axis.set_title(
        "Strong-oracle mimic error remains tissue dependent",
        loc="left",
        fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025)
    colorbar.set_label("Standardized RMSE")
    figure.savefig(output_root / "02_aggregate/figure04_tissue_residual.png", dpi=180)
    plt.close(figure)


def aggregate(output_root: Path) -> None:
    cohort = pd.read_csv(output_root / "00_contract/cohort.csv", dtype={"slide_id": str})
    libraries = load_libraries(output_root)
    target_parts = []
    candidate_parts: dict[str, list[np.ndarray]] = {name: [] for name in libraries}
    fidelity_parts: dict[str, list[np.ndarray]] = {name: [] for name in libraries}
    saturation_parts: dict[str, list[np.ndarray]] = {name: [] for name in libraries}
    source_saturation_parts = []
    metadata = []
    for row in cohort.itertuples(index=False):
        path = output_root / "01_reachability/shards" / f"{row.slide_id}.npz"
        summary_path = path.with_suffix(".summary.json")
        if not path.exists() or not summary_path.exists():
            raise FileNotFoundError(path)
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "pass" or summary.get("output_sha256") != sha256(path):
            raise ValueError(f"invalid shard {path}")
        with np.load(path) as data:
            target_parts.append(np.asarray(data["target_vectors"], dtype=np.float32))
            for arm in libraries:
                candidate_parts[arm].append(np.asarray(data[f"{arm}_vectors"], dtype=np.float32))
                fidelity_parts[arm].append(np.asarray(data[f"{arm}_fidelity"], dtype=np.float32))
                saturation_parts[arm].append(np.asarray(data[f"{arm}_saturation"], dtype=np.float32))
            source_saturation_parts.append(np.asarray(data["source_saturation"], dtype=np.float32))
            for location_index, source_index in zip(data["location_index"], data["source_index"]):
                metadata.append(
                    {
                        "slide_id": str(row.slide_id),
                        "tissue_type": str(row.tissue_type),
                        "location_index": int(location_index),
                        "source_index": int(source_index),
                    }
                )
    target = np.concatenate(target_parts, axis=0)
    candidates = {name: np.concatenate(parts, axis=0) for name, parts in candidate_parts.items()}
    fidelities = {name: np.concatenate(parts, axis=0) for name, parts in fidelity_parts.items()}
    saturations = {name: np.concatenate(parts, axis=0) for name, parts in saturation_parts.items()}
    source_saturation = np.concatenate(source_saturation_parts)
    metadata_frame = pd.DataFrame(metadata)
    if len(metadata_frame) != len(cohort) * PATCHES_PER_SLIDE:
        raise ValueError("incomplete location population")
    if target.shape != (len(metadata_frame), len(TARGET_SCANNERS), len(ENDPOINTS)):
        raise ValueError(f"unexpected target shape {target.shape}")

    slides = metadata_frame[["slide_id"]].drop_duplicates().reset_index(drop=True)
    splitter = GroupKFold(n_splits=5)
    slide_fold: dict[str, int] = {}
    for fold, (_, test) in enumerate(splitter.split(slides, groups=slides["slide_id"])):
        for slide_id in slides.iloc[test]["slide_id"]:
            slide_fold[str(slide_id)] = fold
    metadata_frame["fold"] = metadata_frame["slide_id"].map(slide_fold).astype(int)
    write_frame(
        output_root / "00_contract/slide_folds.csv",
        pd.DataFrame(sorted(slide_fold.items()), columns=["slide_id", "fold"]),
    )

    selected_rows = []
    global_rows = []
    for fold in range(5):
        train = metadata_frame["fold"].to_numpy() != fold
        test = ~train
        scale = robust_scale(target[train].reshape(-1, len(ENDPOINTS)))
        valid = {
            name: candidate_valid(
                fidelities[name], saturations[name], source_saturation
            )
            for name in libraries
        }
        for scanner_index, scanner in enumerate(TARGET_SCANNERS):
            global_candidates = {}
            for library_name in libraries:
                train_distance = distances(
                    candidates[library_name][train],
                    target[train, scanner_index],
                    scale,
                )
                train_distance[~valid[library_name][train]] = np.inf
                mean_distance = np.mean(train_distance, axis=0)
                if not np.isfinite(mean_distance).any():
                    raise ValueError(f"no valid global candidate: {fold}/{scanner}/{library_name}")
                global_candidates[library_name] = int(np.nanargmin(mean_distance))
                global_rows.append(
                    {
                        "fold": fold,
                        "scanner": scanner,
                        "library": library_name,
                        "candidate_index": global_candidates[library_name],
                        "training_mean_distance": float(mean_distance[global_candidates[library_name]]),
                    }
                )

            test_indices = np.flatnonzero(test)
            for observation in test_indices:
                arm_choices = {"identity": ("strong", 0)}
                for library_name in libraries:
                    arm_choices[f"{library_name}_global"] = (
                        library_name,
                        global_candidates[library_name],
                    )
                    one_distance = distances(
                        candidates[library_name][observation : observation + 1],
                        target[observation : observation + 1, scanner_index],
                        scale,
                    )[0]
                    one_distance[~valid[library_name][observation]] = np.inf
                    if not np.isfinite(one_distance).any():
                        raise ValueError(f"no valid oracle candidate: {observation}/{library_name}")
                    arm_choices[f"{library_name}_oracle"] = (
                        library_name,
                        int(np.argmin(one_distance)),
                    )
                    if library_name == "strong":
                        unconstrained_distance = distances(
                            candidates[library_name][observation : observation + 1],
                            target[observation : observation + 1, scanner_index],
                            scale,
                        )[0]
                        arm_choices["strong_unconstrained_oracle"] = (
                            library_name,
                            int(np.argmin(unconstrained_distance)),
                        )

                for arm, (library_name, candidate_index) in arm_choices.items():
                    generated = candidates[library_name][observation, candidate_index]
                    target_vector = target[observation, scanner_index]
                    standardized = (generated - target_vector) / scale
                    endpoint_within_one = np.abs(standardized) <= 1.0
                    payload = {
                        **metadata_frame.iloc[observation].to_dict(),
                        "scanner": scanner,
                        "arm": arm,
                        "library": library_name,
                        "candidate_index": candidate_index,
                        "distance": float(np.sqrt(np.mean(np.square(standardized)))),
                        "endpoints_within_one_sd_fraction": float(endpoint_within_one.mean()),
                        "all_endpoints_within_one_sd": bool(endpoint_within_one.all()),
                        "fidelity_ncc": float(fidelities[library_name][observation, candidate_index]),
                        "saturation_fraction": float(saturations[library_name][observation, candidate_index]),
                    }
                    for endpoint_index, endpoint in enumerate(ENDPOINTS):
                        payload[f"generated_{endpoint}"] = float(generated[endpoint_index])
                        payload[f"target_{endpoint}"] = float(target_vector[endpoint_index])
                        payload[f"scale_{endpoint}"] = float(scale[endpoint_index])
                    selected_rows.append(payload)
    selected = pd.DataFrame(selected_rows)
    if len(selected) != len(metadata_frame) * len(TARGET_SCANNERS) * len(ARMS):
        raise ValueError("selected-candidate table is incomplete")
    write_frame(output_root / "02_aggregate/selected_candidates.csv", selected)
    write_frame(output_root / "02_aggregate/global_candidates.csv", pd.DataFrame(global_rows))

    classifier = classifier_predictions(selected)
    write_frame(output_root / "02_aggregate/classifier_predictions.csv", classifier)
    summaries = []
    for arm in ARMS:
        for scanner in (*TARGET_SCANNERS, "all"):
            subset = selected[selected["arm"] == arm]
            prediction = classifier[classifier["arm"] == arm]
            if scanner != "all":
                subset = subset[subset["scanner"] == scanner]
                prediction = prediction[prediction["scanner"] == scanner]
            distance_mean, distance_low, distance_high = bootstrap_mean_ci(
                subset, "distance", stable_seed(arm, scanner, "distance")
            )
            coverage_mean, coverage_low, coverage_high = bootstrap_mean_ci(
                subset,
                "endpoints_within_one_sd_fraction",
                stable_seed(arm, scanner, "coverage"),
            )
            per_slide_accuracy = (
                prediction.groupby("slide_id")["correct"].mean().rename("accuracy").reset_index()
            )
            accuracy_mean, accuracy_low, accuracy_high = bootstrap_mean_ci(
                per_slide_accuracy,
                "accuracy",
                stable_seed(arm, scanner, "classifier"),
            )
            summaries.append(
                {
                    "arm": arm,
                    "scanner": scanner,
                    "locations": subset[["slide_id", "location_index"]].drop_duplicates().shape[0],
                    "distance_mean": distance_mean,
                    "distance_ci_low": distance_low,
                    "distance_ci_high": distance_high,
                    "endpoint_coverage_mean": coverage_mean,
                    "endpoint_coverage_ci_low": coverage_low,
                    "endpoint_coverage_ci_high": coverage_high,
                    "all_endpoint_coverage_rate": float(subset["all_endpoints_within_one_sd"].mean()),
                    "mean_fidelity_ncc": float(subset["fidelity_ncc"].mean()),
                    "classifier_balanced_accuracy": accuracy_mean,
                    "classifier_ci_low": accuracy_low,
                    "classifier_ci_high": accuracy_high,
                }
            )
    arm_summary = pd.DataFrame(summaries)
    write_frame(output_root / "02_aggregate/arm_summary.csv", arm_summary)
    make_figures(selected, arm_summary, output_root)

    pooled = arm_summary[arm_summary["scanner"] == "all"].set_index("arm")
    strong = pooled.loc["strong_oracle"]
    strong_unconstrained = pooled.loc["strong_unconstrained_oracle"]
    default = pooled.loc["default_oracle"]
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "pass",
        "completed_utc": utc_now(),
        "slides": len(cohort),
        "locations": len(metadata_frame),
        "paired_target_comparisons": len(metadata_frame) * len(TARGET_SCANNERS),
        "augmentation_candidates_evaluated_per_source": {
            name: len(values) for name, values in libraries.items()
        },
        "primary": {
            "default_oracle_distance": float(default["distance_mean"]),
            "default_oracle_distance_ci": [
                float(default["distance_ci_low"]),
                float(default["distance_ci_high"]),
            ],
            "strong_oracle_distance": float(strong["distance_mean"]),
            "strong_oracle_distance_ci": [
                float(strong["distance_ci_low"]),
                float(strong["distance_ci_high"]),
            ],
            "strong_oracle_endpoint_coverage": float(strong["endpoint_coverage_mean"]),
            "strong_oracle_all_endpoint_coverage": float(strong["all_endpoint_coverage_rate"]),
            "strong_oracle_real_vs_augmented_balanced_accuracy": float(
                strong["classifier_balanced_accuracy"]
            ),
            "strong_oracle_classifier_ci": [
                float(strong["classifier_ci_low"]),
                float(strong["classifier_ci_high"]),
            ],
            "strong_unconstrained_oracle_distance": float(
                strong_unconstrained["distance_mean"]
            ),
            "strong_unconstrained_oracle_mean_fidelity_ncc": float(
                strong_unconstrained["mean_fidelity_ncc"]
            ),
        },
        "interpretation_status": "requires_result_reading_before_manuscript_update",
        "claim_boundary": (
            "The oracle searches a large but finite declared proxy augmentation library. "
            "Failure supports mismatch to this family; it does not prove exclusion from every "
            "possible augmentation or identify pure optics."
        ),
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    extract = commands.add_parser("extract-slide")
    extract.add_argument("--slide-index", type=int, required=True)
    extract.add_argument("--workers", type=int, default=4)
    commands.add_parser("aggregate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if args.command == "prepare":
        prepare(source_root, output_root)
    elif args.command == "extract-slide":
        extract_slide(args.slide_index, source_root, output_root, args.workers)
    elif args.command == "aggregate":
        aggregate(output_root)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
