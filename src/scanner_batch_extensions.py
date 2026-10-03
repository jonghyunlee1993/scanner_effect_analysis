#!/usr/bin/env python3
"""Current-data extensions for the scanner batch-effect image study.

The module implements three connected analyses on the locked 103-slide cohort:

* a directed six-scanner image-relation atlas;
* slide-held-out Reinhard/frequency correction with explicit fidelity controls;
* a frozen-UNI consequence bridge after every image-space choice is locked.

The main correction is fitted only from image statistics.  Target-informed
amplitude/phase arms are labelled mechanistic oracles and never presented as
deployable correction methods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.color import lab2rgb, rgb2lab

from augmentation_ood_study import (
    ENDPOINTS,
    TARGET_SCANNERS,
    absolute_measurements,
    choose_locations,
    contrasts_from_absolute,
    stable_seed,
)
from final_image_study import PROVISIONAL_MPP, SCANNERS, rowwise_correlation


ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = ROOT / "outputs/final_image_study_v1"
AUGMENTATION_ROOT = ROOT / "outputs/augmentation_ood_v1"
OUTPUT_ROOT = ROOT / "outputs/scanner_batch_effect_analysis_2026-09-17"
ANALYSIS_ID = "scanner_batch_effect_analysis_2026-09-17"
DATE = "2026-09-17"
REFERENCE = "at2"
BOOTSTRAPS = 2000
PRIMARY_ARMS = ("raw", "reinhard", "frequency", "combined")
SUPPLEMENT_ARMS = (
    "destructive_lowpass",
    "amplitude_oracle",
    "phase_oracle",
    "combined_loto",
)
ALL_ARMS = PRIMARY_ARMS + SUPPLEMENT_ARMS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def prepare(source_root: Path, augmentation_root: Path, output_root: Path) -> None:
    cohort = pd.read_csv(source_root / "00_contract/cohort.csv", dtype={"slide_id": str})
    folds = pd.read_csv(
        augmentation_root / "00_contract/slide_folds.csv", dtype={"slide_id": str}
    )
    if set(cohort["slide_id"]) != set(folds["slide_id"]):
        raise ValueError("cohort and slide-fold populations differ")
    cohort = cohort.merge(folds, on="slide_id", validate="one_to_one")
    directories = (
        "00_contract/target_references",
        "01_scanner_atlas",
        "02_correction/fit_shards",
        "02_correction/shards",
        "03_uni/shards",
        "04_mechanism",
        "05_supplement",
        "90_figures/main",
        "90_figures/supplement",
        "logs",
    )
    for relative in directories:
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    write_frame(output_root / "00_contract/cohort.csv", cohort)

    selected_path = augmentation_root / "02_aggregate/selected_candidates.csv"
    selected = pd.read_csv(selected_path, dtype={"slide_id": str})
    identity = selected[selected["arm"] == "identity"].copy()
    keep = ["slide_id", "tissue_type", "location_index", "source_index", "fold", "scanner"]
    for endpoint in ENDPOINTS:
        keep.extend([f"target_{endpoint}", f"scale_{endpoint}"])
    identity = identity[keep]
    expected = len(cohort) * 40 * len(TARGET_SCANNERS)
    if len(identity) != expected:
        raise ValueError(f"expected {expected} target-reference rows, got {len(identity)}")
    for slide_id, frame in identity.groupby("slide_id", sort=False):
        write_frame(output_root / "00_contract/target_references" / f"{slide_id}.csv", frame)

    contract = {
        "analysis_id": ANALYSIS_ID,
        "created_utc": utc_now(),
        "status": "prepared",
        "slides": len(cohort),
        "tissues": int(cohort["tissue_type"].nunique()),
        "scanners": list(SCANNERS),
        "reference": REFERENCE,
        "locations_per_slide": 40,
        "primary_arms": list(PRIMARY_ARMS),
        "supplement_arms": list(SUPPLEMENT_ARMS),
        "split_rule": "five slide-held-out folds frozen by augmentation_ood_v1",
        "loto_rule": "each slide also receives parameters fitted after excluding its entire tissue type",
        "frequency_rule": (
            "phase-preserving smooth radial amplitude gain estimated from training-slide "
            "paired spectra and clipped to 0.5x--2x"
        ),
        "carried_forward_fidelity_gates": {
            "gradient_ncc_to_source_minimum": 0.90,
            "saturation_absolute_ceiling": 0.10,
        },
        "physical_frequency_status": "provisional_at_0.5052_um_per_pixel",
        "uni_role": "locked secondary bridge; never used for image correction selection",
        "oracle_boundary": (
            "amplitude_oracle and phase_oracle inspect the paired target and are mechanism ceilings only"
        ),
        "unidentified": (
            "device unit, date, causal focus, optical MTF, vendor processing, and compression "
            "cannot be separated with the current observational metadata"
        ),
        "sources": {
            "cohort": str((source_root / "00_contract/cohort.csv").resolve()),
            "cohort_sha256": sha256(source_root / "00_contract/cohort.csv"),
            "selected_candidates": str(selected_path.resolve()),
            "selected_candidates_sha256": sha256(selected_path),
        },
    }
    write_json(output_root / "00_contract/analysis_contract.json", contract)
    print(json.dumps(contract, ensure_ascii=False, indent=2))


def fit_slide(slide_index: int, output_root: Path) -> None:
    cohort = pd.read_csv(output_root / "00_contract/cohort.csv", dtype={"slide_id": str})
    row = cohort.iloc[slide_index]
    slide_id = str(row.slide_id)
    output = output_root / "02_correction/fit_shards" / f"{slide_id}.csv"
    if output.exists():
        existing = pd.read_csv(output)
        if len(existing) == len(SCANNERS) and set(existing["scanner"]) == set(SCANNERS):
            print(json.dumps({"status": "already_complete", "slide_id": slide_id}))
            return
    metrics_path = SOURCE_ROOT / "02_image_phenotypes/shards" / slide_id / "location_metrics.h5"
    locations = choose_locations(metrics_path)
    with h5py.File(Path(row.cache_path), "r") as store:
        images = np.asarray(store["images"][locations], dtype=np.uint8)
    rows = []
    for scanner_index, scanner in enumerate(SCANNERS):
        lab = rgb2lab(images[:, scanner_index].astype(np.float32) / 255.0).astype(np.float64)
        flat = lab.reshape(-1, 3)
        payload = {
            "slide_id": slide_id,
            "tissue_type": str(row.tissue_type),
            "scanner": scanner,
            "pixels": len(flat),
        }
        for channel, label in enumerate(("l", "a", "b")):
            payload[f"sum_{label}"] = float(flat[:, channel].sum())
            payload[f"sumsq_{label}"] = float(np.square(flat[:, channel]).sum())
        rows.append(payload)
    write_frame(output, pd.DataFrame(rows))
    print(json.dumps({"status": "pass", "slide_id": slide_id, "locations": len(locations)}))


def aggregate_lab_stats(stats: pd.DataFrame, slides: set[str], scanner: str) -> tuple[list[float], list[float]]:
    selected = stats[(stats["slide_id"].isin(slides)) & (stats["scanner"] == scanner)]
    count = float(selected["pixels"].sum())
    if count <= 0:
        raise ValueError(f"no LAB statistics for {scanner}")
    means, stds = [], []
    for label in ("l", "a", "b"):
        mean = float(selected[f"sum_{label}"].sum() / count)
        second = float(selected[f"sumsq_{label}"].sum() / count)
        means.append(mean)
        stds.append(max(math.sqrt(max(second - mean * mean, 0.0)), 1e-3))
    return means, stds


def parameter_block(
    stats: pd.DataFrame,
    spectra: pd.DataFrame,
    train_slides: set[str],
) -> dict:
    source_mean, source_std = aggregate_lab_stats(stats, train_slides, REFERENCE)
    frequency = (
        spectra["frequency_cyc_per_pixel"].drop_duplicates().sort_values().to_numpy(float)
    )
    block = {}
    for scanner in TARGET_SCANNERS:
        target_mean, target_std = aggregate_lab_stats(stats, train_slides, scanner)
        subset = spectra[
            (spectra["slide_id"].isin(train_slides)) & (spectra["scanner"] == scanner)
        ]
        curve = (
            subset.groupby("frequency_cyc_per_pixel")["log2_relative_transfer"]
            .mean()
            .reindex(frequency)
            .to_numpy(float)
        )
        curve = np.clip(gaussian_filter1d(curve, sigma=1.5, mode="nearest"), -1.0, 1.0)
        block[scanner] = {
            "source_mean": source_mean,
            "source_std": source_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "frequency_cyc_per_pixel": frequency.tolist(),
            "log2_amplitude_gain": curve.tolist(),
            "training_slides": len(train_slides),
        }
    return block


def fit_aggregate(source_root: Path, output_root: Path) -> None:
    cohort = pd.read_csv(output_root / "00_contract/cohort.csv", dtype={"slide_id": str})
    frames = []
    for slide_id in cohort["slide_id"]:
        path = output_root / "02_correction/fit_shards" / f"{slide_id}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path, dtype={"slide_id": str}))
    stats = pd.concat(frames, ignore_index=True)
    spectra = pd.read_csv(
        source_root / "03_frequency/spectra.csv", dtype={"slide_id": str}
    )
    all_slides = set(cohort["slide_id"])
    parameters = {"fold": {}, "loto": {}}
    for fold in sorted(cohort["fold"].unique()):
        train = set(cohort.loc[cohort["fold"] != fold, "slide_id"])
        parameters["fold"][str(int(fold))] = parameter_block(stats, spectra, train)
    for tissue in sorted(cohort["tissue_type"].unique()):
        train = set(cohort.loc[cohort["tissue_type"] != tissue, "slide_id"])
        parameters["loto"][str(tissue)] = parameter_block(stats, spectra, train)
    parameters["full"] = parameter_block(stats, spectra, all_slides)
    parameters["analysis_id"] = ANALYSIS_ID
    parameters["created_utc"] = utc_now()
    write_json(output_root / "02_correction/parameters.json", parameters)
    write_frame(output_root / "02_correction/lab_sufficient_statistics.csv", stats)
    print(json.dumps({"status": "pass", "folds": 5, "tissues": len(parameters["loto"])}))


def reinhard_transform(image: np.ndarray, parameter: dict) -> np.ndarray:
    lab = rgb2lab(image.astype(np.float32) / 255.0).astype(np.float32)
    source_mean = np.asarray(parameter["source_mean"], dtype=np.float32)
    source_std = np.asarray(parameter["source_std"], dtype=np.float32)
    target_mean = np.asarray(parameter["target_mean"], dtype=np.float32)
    target_std = np.asarray(parameter["target_std"], dtype=np.float32)
    corrected = (lab - source_mean) / source_std * target_std + target_mean
    corrected[..., 0] = np.clip(corrected[..., 0], 0.0, 100.0)
    corrected[..., 1:] = np.clip(corrected[..., 1:], -127.0, 127.0)
    rgb = lab2rgb(corrected)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def replace_od_luminance(image: np.ndarray, corrected_mean_od: np.ndarray) -> np.ndarray:
    channel_od = -np.log((image.astype(np.float32) + 1.0) / 256.0)
    original_mean = channel_od.mean(axis=-1)
    channel_od = channel_od + (corrected_mean_od - original_mean)[..., None]
    rgb = np.exp(-np.clip(channel_od, 0.0, 8.0)) * 256.0 - 1.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def frequency_transform(image: np.ndarray, parameter: dict, destructive: bool = False) -> np.ndarray:
    channel_od = -np.log((image.astype(np.float32) + 1.0) / 256.0)
    mean_od = channel_od.mean(axis=-1)
    pad = 32
    padded = np.pad(mean_od, pad, mode="reflect")
    centered = padded - padded.mean()
    spectrum = np.fft.fft2(centered)
    fy = np.fft.fftfreq(padded.shape[0])[:, None]
    fx = np.fft.fftfreq(padded.shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    if destructive:
        gain = np.exp(-2.0 * (np.pi**2) * (3.0**2) * radius**2)
    else:
        frequency = np.asarray(parameter["frequency_cyc_per_pixel"], dtype=float)
        log_gain = np.asarray(parameter["log2_amplitude_gain"], dtype=float)
        gain = np.exp2(np.interp(np.minimum(radius, frequency[-1]), frequency, log_gain))
        gain = np.clip(gain, 0.5, 2.0)
    corrected = np.fft.ifft2(spectrum * gain).real + padded.mean()
    corrected = corrected[pad:-pad, pad:-pad].astype(np.float32)
    return replace_od_luminance(image, corrected)


def amplitude_phase_oracles(base: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def mean_od(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        channels = -np.log((image.astype(np.float32) + 1.0) / 256.0)
        value = channels.mean(axis=-1)
        return value, np.fft.fft2(value - value.mean())

    base_od, base_fft = mean_od(base)
    target_od, target_fft = mean_od(target)
    amplitude = np.abs(target_fft) * np.exp(1j * np.angle(base_fft))
    phase = np.abs(base_fft) * np.exp(1j * np.angle(target_fft))
    amplitude_od = np.fft.ifft2(amplitude).real + base_od.mean()
    phase_od = np.fft.ifft2(phase).real + base_od.mean()
    return (
        replace_od_luminance(base, amplitude_od.astype(np.float32)),
        replace_od_luminance(base, phase_od.astype(np.float32)),
    )


def render_conditions(source: np.ndarray, target: np.ndarray, fold_parameter: dict, loto_parameter: dict) -> dict:
    reinhard = reinhard_transform(source, fold_parameter)
    frequency = frequency_transform(source, fold_parameter)
    combined = frequency_transform(reinhard, fold_parameter)
    destructive = frequency_transform(reinhard, fold_parameter, destructive=True)
    amplitude, phase = amplitude_phase_oracles(reinhard, target)
    loto = frequency_transform(reinhard_transform(source, loto_parameter), loto_parameter)
    return {
        "raw": source,
        "reinhard": reinhard,
        "frequency": frequency,
        "combined": combined,
        "destructive_lowpass": destructive,
        "amplitude_oracle": amplitude,
        "phase_oracle": phase,
        "combined_loto": loto,
    }


def extract_slide(slide_index: int, output_root: Path, workers: int) -> None:
    cohort = pd.read_csv(output_root / "00_contract/cohort.csv", dtype={"slide_id": str})
    row = cohort.iloc[slide_index]
    slide_id = str(row.slide_id)
    output = output_root / "02_correction/shards" / f"{slide_id}.csv"
    summary_path = output.with_suffix(".summary.json")
    if output.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("status") == "pass" and summary.get("output_sha256") == sha256(output):
            print(json.dumps({"status": "already_complete", "slide_id": slide_id}))
            return
    parameters = json.loads((output_root / "02_correction/parameters.json").read_text())
    fold_block = parameters["fold"][str(int(row.fold))]
    loto_block = parameters["loto"][str(row.tissue_type)]
    metrics_path = SOURCE_ROOT / "02_image_phenotypes/shards" / slide_id / "location_metrics.h5"
    locations = choose_locations(metrics_path)
    with h5py.File(Path(row.cache_path), "r") as store:
        images = np.asarray(store["images"][locations], dtype=np.uint8)
        source_indices = np.asarray(store["source_index"][locations], dtype=np.int64)
    references = pd.read_csv(
        output_root / "00_contract/target_references" / f"{slide_id}.csv"
    ).set_index(["location_index", "scanner"])
    rows = []
    for offset, location_index in enumerate(locations):
        source = images[offset, 0]
        source_scalar, source_radial, source_gradient, _ = absolute_measurements(
            source[None], workers
        )
        for scanner_index, scanner in enumerate(TARGET_SCANNERS, start=1):
            target = images[offset, scanner_index]
            conditions = render_conditions(
                source, target, fold_block[scanner], loto_block[scanner]
            )
            condition_images = np.stack([conditions[arm] for arm in ALL_ARMS], axis=0)
            scalar, radial, gradient, _ = absolute_measurements(condition_images, workers)
            generated = contrasts_from_absolute(scalar, radial, source_scalar, source_radial)
            target_scalar, target_radial, target_gradient, _ = absolute_measurements(
                target[None], workers
            )
            target_vector = contrasts_from_absolute(
                target_scalar, target_radial, source_scalar, source_radial
            )[0]
            reference = references.loc[(int(location_index), scanner)]
            scale = np.asarray([float(reference[f"scale_{name}"]) for name in ENDPOINTS])
            scale = np.maximum(scale, 1e-6)
            reference_target = np.asarray(
                [float(reference[f"target_{name}"]) for name in ENDPOINTS]
            )
            if not np.allclose(target_vector, reference_target, atol=3e-3, rtol=3e-3):
                raise ValueError(f"target endpoint mismatch {slide_id}/{location_index}/{scanner}")
            fidelity = rowwise_correlation(
                gradient, np.repeat(source_gradient, len(ALL_ARMS), axis=0)
            )
            target_similarity = rowwise_correlation(
                gradient, np.repeat(target_gradient, len(ALL_ARMS), axis=0)
            )
            for arm_index, arm in enumerate(ALL_ARMS):
                residual = (generated[arm_index] - target_vector) / scale
                payload = {
                    "slide_id": slide_id,
                    "tissue_type": str(row.tissue_type),
                    "fold": int(row.fold),
                    "location_index": int(location_index),
                    "source_index": int(source_indices[offset]),
                    "scanner": scanner,
                    "arm": arm,
                    "distance": float(np.sqrt(np.mean(residual**2))),
                    "endpoint_coverage": float(np.mean(np.abs(residual) <= 1.0)),
                    "all_endpoints_covered": bool(np.all(np.abs(residual) <= 1.0)),
                    "fidelity_ncc": float(fidelity[arm_index]),
                    "target_gradient_ncc": float(target_similarity[arm_index]),
                    "saturation_fraction": float(scalar[arm_index, 8]),
                    "source_gradient_rms": float(source_scalar[0, 6]),
                    "target_fidelity_ncc": float(
                        rowwise_correlation(target_gradient, source_gradient)[0]
                    ),
                }
                for endpoint_index, endpoint in enumerate(ENDPOINTS):
                    payload[f"generated_{endpoint}"] = float(generated[arm_index, endpoint_index])
                    payload[f"target_{endpoint}"] = float(target_vector[endpoint_index])
                    payload[f"scaled_residual_{endpoint}"] = float(residual[endpoint_index])
                rows.append(payload)
        print(f"[{slide_id}] {offset + 1}/{len(locations)}", flush=True)
    frame = pd.DataFrame(rows)
    expected = len(locations) * len(TARGET_SCANNERS) * len(ALL_ARMS)
    if len(frame) != expected or not np.isfinite(frame.select_dtypes("number")).all().all():
        raise ValueError(f"invalid correction shard {slide_id}: {len(frame)} rows")
    write_frame(output, frame)
    summary = {
        "analysis_id": ANALYSIS_ID,
        "status": "pass",
        "completed_utc": utc_now(),
        "slide_id": slide_id,
        "locations": len(locations),
        "rows": len(frame),
        "output_sha256": sha256(output),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


def bootstrap_slide_mean(frame: pd.DataFrame, column: str, seed: int) -> tuple[float, float, float]:
    values = frame.groupby("slide_id")[column].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired_slide_contrast(
    frame: pd.DataFrame, arm_a: str, arm_b: str, column: str, seed: int
) -> tuple[float, float, float]:
    table = frame.pivot_table(index="slide_id", columns="arm", values=column, aggfunc="mean")
    values = (table[arm_a] - table[arm_b]).dropna().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def fit_equivalent_blur(frequency: np.ndarray, log2_curve: np.ndarray) -> tuple[float, float, float]:
    selected = (frequency >= 0.05) & (frequency <= 0.45)
    x = np.square(frequency[selected])
    y = np.log(2.0) * log2_curve[selected]
    design = np.column_stack([np.ones(len(x)), x])
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ coefficient
    ss_total = float(np.square(y - y.mean()).sum())
    ss_residual = float(np.square(y - fitted).sum())
    r2 = 1.0 - ss_residual / max(ss_total, 1e-12)
    slope = float(coefficient[1])
    sigma = math.copysign(math.sqrt(abs(slope) / (2.0 * math.pi**2)), -slope)
    rmse = math.sqrt(ss_residual / len(y))
    return sigma, r2, rmse


def scanner_atlas(source_root: Path, output_root: Path) -> dict:
    scalar = pd.read_csv(
        source_root / "02_image_phenotypes/replicate_scalar_summary.csv",
        dtype={"slide_id": str},
    )
    spectra = pd.read_csv(source_root / "03_frequency/spectra.csv", dtype={"slide_id": str})
    bands = pd.read_csv(
        source_root / "03_frequency/replicate_band_summary.csv", dtype={"slide_id": str}
    )
    scalar_slide = scalar.groupby(["slide_id", "tissue_type", "scanner"], as_index=False).mean(numeric_only=True)
    band_slide = bands.groupby(["slide_id", "tissue_type", "scanner", "band"], as_index=False)["log2_relative_transfer"].mean()
    frequency = spectra["frequency_cyc_per_pixel"].drop_duplicates().sort_values().to_numpy(float)
    curve_lookup = {
        (str(slide), str(scanner)): group.set_index("frequency_cyc_per_pixel")["log2_relative_transfer"].reindex(frequency).to_numpy(float)
        for (slide, scanner), group in spectra.groupby(["slide_id", "scanner"])
    }
    vector_rows, curve_rows, blur_rows = [], [], []
    metric_names = (
        "delta_lab_l", "delta_lab_a", "delta_lab_b", "log2_mean_od_ratio",
        "log2_lab_l_sd_ratio", "log2_od_sd_ratio", "log2_gradient_rms_ratio",
        "frequency_low_mid", "frequency_mid", "frequency_high",
    )
    raw_vectors = []
    for source_scanner in SCANNERS:
        for target_scanner in SCANNERS:
            if source_scanner == target_scanner:
                continue
            merged = scalar_slide[scalar_slide["scanner"] == source_scanner].merge(
                scalar_slide[scalar_slide["scanner"] == target_scanner],
                on=["slide_id", "tissue_type"], suffixes=("_a", "_b"), validate="one_to_one"
            )
            band_a = band_slide[band_slide["scanner"] == source_scanner].pivot(
                index="slide_id", columns="band", values="log2_relative_transfer"
            )
            band_b = band_slide[band_slide["scanner"] == target_scanner].pivot(
                index="slide_id", columns="band", values="log2_relative_transfer"
            )
            for item in merged.itertuples(index=False):
                slide_id = str(item.slide_id)
                vector = np.asarray([
                    item.lab_l_mean_b - item.lab_l_mean_a,
                    item.lab_a_mean_b - item.lab_a_mean_a,
                    item.lab_b_mean_b - item.lab_b_mean_a,
                    math.log2(max(item.mean_od_b, 1e-8) / max(item.mean_od_a, 1e-8)),
                    math.log2(max(item.lab_l_sd_b, 1e-8) / max(item.lab_l_sd_a, 1e-8)),
                    math.log2(max(item.od_sd_b, 1e-8) / max(item.od_sd_a, 1e-8)),
                    math.log2(max(item.gradient_rms_b, 1e-8) / max(item.gradient_rms_a, 1e-8)),
                    band_b.loc[slide_id, "low_mid"] - band_a.loc[slide_id, "low_mid"],
                    band_b.loc[slide_id, "mid"] - band_a.loc[slide_id, "mid"],
                    band_b.loc[slide_id, "high"] - band_a.loc[slide_id, "high"],
                ], dtype=float)
                raw_vectors.append(vector)
                payload = {
                    "slide_id": slide_id, "tissue_type": item.tissue_type,
                    "source_scanner": source_scanner, "target_scanner": target_scanner,
                }
                payload.update(dict(zip(metric_names, vector)))
                vector_rows.append(payload)
                curve = curve_lookup[(slide_id, target_scanner)] - curve_lookup[(slide_id, source_scanner)]
                sigma, r2, rmse = fit_equivalent_blur(frequency, curve)
                blur_rows.append({
                    "slide_id": slide_id, "tissue_type": item.tissue_type,
                    "source_scanner": source_scanner, "target_scanner": target_scanner,
                    "signed_equivalent_sigma_px": sigma,
                    "signed_equivalent_sigma_um_provisional": sigma * PROVISIONAL_MPP,
                    "gaussian_r2": r2, "gaussian_log_amplitude_rmse": rmse,
                })
                for f, value in zip(frequency, curve):
                    curve_rows.append({
                        "slide_id": slide_id, "source_scanner": source_scanner,
                        "target_scanner": target_scanner, "frequency_cyc_per_pixel": f,
                        "log2_amplitude_ratio": value,
                    })
    vector_frame = pd.DataFrame(vector_rows)
    scales = np.median(np.abs(np.asarray(raw_vectors) - np.median(raw_vectors, axis=0)), axis=0) * 1.4826
    scales = np.maximum(scales, 1e-3)
    vector_frame["composite_distance"] = np.sqrt(
        np.mean(np.square(vector_frame[list(metric_names)].to_numpy(float) / scales), axis=1)
    )
    curve_frame = pd.DataFrame(curve_rows)
    blur_frame = pd.DataFrame(blur_rows)
    summary_rows = []
    for (source_scanner, target_scanner), group in vector_frame.groupby(["source_scanner", "target_scanner"]):
        payload = {"source_scanner": source_scanner, "target_scanner": target_scanner}
        for metric in list(metric_names) + ["composite_distance"]:
            estimate, low, high = bootstrap_slide_mean(group, metric, stable_seed("atlas", source_scanner, target_scanner, metric))
            payload[metric] = estimate
            payload[f"{metric}_ci_low"] = low
            payload[f"{metric}_ci_high"] = high
        blur = blur_frame[(blur_frame["source_scanner"] == source_scanner) & (blur_frame["target_scanner"] == target_scanner)]
        for metric in ("signed_equivalent_sigma_px", "signed_equivalent_sigma_um_provisional", "gaussian_r2", "gaussian_log_amplitude_rmse"):
            estimate, low, high = bootstrap_slide_mean(blur, metric, stable_seed("blur", source_scanner, target_scanner, metric))
            payload[metric] = estimate
            payload[f"{metric}_ci_low"] = low
            payload[f"{metric}_ci_high"] = high
        mean_curve = curve_frame[(curve_frame["source_scanner"] == source_scanner) & (curve_frame["target_scanner"] == target_scanner)].groupby("frequency_cyc_per_pixel")["log2_amplitude_ratio"].mean().reindex(frequency).to_numpy(float)
        sigma, r2, rmse = fit_equivalent_blur(frequency, mean_curve)
        payload["mean_curve_signed_sigma_px"] = sigma
        payload["mean_curve_gaussian_r2"] = r2
        payload["mean_curve_gaussian_rmse"] = rmse
        summary_rows.append(payload)
    summary = pd.DataFrame(summary_rows)
    write_frame(output_root / "01_scanner_atlas/slide_vectors.csv", vector_frame)
    write_frame(output_root / "01_scanner_atlas/pairwise_curves.csv", curve_frame)
    write_frame(output_root / "01_scanner_atlas/slide_blur_fits.csv", blur_frame)
    write_frame(output_root / "01_scanner_atlas/pairwise_summary.csv", summary)
    make_atlas_figures(summary, curve_frame, frequency, output_root)
    return {
        "pairs": len(summary),
        "median_gaussian_r2": float(summary["mean_curve_gaussian_r2"].median()),
        "poor_scalar_blur_pairs_r2_below_0_8": int((summary["mean_curve_gaussian_r2"] < 0.8).sum()),
    }


def make_atlas_figures(summary: pd.DataFrame, curves: pd.DataFrame, frequency: np.ndarray, output_root: Path) -> None:
    labels = [scanner.upper() for scanner in SCANNERS]
    distance = np.zeros((len(SCANNERS), len(SCANNERS)))
    sigma = np.zeros_like(distance)
    adequacy = np.ones_like(distance)
    for row in summary.itertuples(index=False):
        i = SCANNERS.index(row.source_scanner)
        j = SCANNERS.index(row.target_scanner)
        distance[i, j] = row.composite_distance
        sigma[i, j] = row.signed_equivalent_sigma_px
        adequacy[i, j] = row.mean_curve_gaussian_r2
    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    panels = [
        (axes[0, 0], distance, "A  Multicomponent scanner difference", "robust-SD units", "viridis", None),
        (axes[0, 1], sigma, "B  Directed equivalent blur", "signed sigma (px)", "coolwarm", max(abs(sigma.min()), abs(sigma.max()))),
        (axes[1, 0], adequacy, "C  Can one Gaussian blur explain the curve?", "Gaussian fit R-squared", "magma", None),
    ]
    for axis, matrix, title, color_label, cmap, limit in panels:
        kwargs = {"cmap": cmap, "aspect": "auto"}
        if limit is not None:
            kwargs.update(vmin=-limit, vmax=limit)
        image = axis.imshow(matrix, **kwargs)
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("Target scanner B")
        axis.set_ylabel("Source scanner A")
        axis.set_title(title, loc="left", fontweight="bold")
        figure.colorbar(image, ax=axis, shrink=0.8, label=color_label)
    axis = axes[1, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(TARGET_SCANNERS)))
    for color, scanner in zip(colors, TARGET_SCANNERS):
        group = curves[(curves["source_scanner"] == REFERENCE) & (curves["target_scanner"] == scanner)]
        table = group.pivot(index="slide_id", columns="frequency_cyc_per_pixel", values="log2_amplitude_ratio").reindex(columns=frequency)
        mean = table.mean(axis=0).to_numpy(float)
        sem = table.std(axis=0, ddof=1).to_numpy(float) / math.sqrt(len(table))
        axis.plot(frequency, np.exp2(mean), color=color, label=scanner.upper(), linewidth=2)
        axis.fill_between(frequency, np.exp2(mean - 1.96 * sem), np.exp2(mean + 1.96 * sem), color=color, alpha=0.15)
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xlim(0.03, 0.5)
    axis.set_yscale("log", base=2)
    axis.set_xlabel("Spatial frequency (cycles/pixel)")
    axis.set_ylabel("Amplitude ratio to AT2")
    axis.set_title("D  Full curve before any scalar summary", loc="left", fontweight="bold")
    axis.legend(ncol=2, fontsize=8)
    figure.suptitle("Directed scanner-relation atlas", fontsize=17, fontweight="bold")
    figure.savefig(output_root / "90_figures/main/figure01_scanner_relation_atlas.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(len(SCANNERS), len(SCANNERS), figsize=(17, 17), sharex=True, sharey=True)
    for i, source_scanner in enumerate(SCANNERS):
        for j, target_scanner in enumerate(SCANNERS):
            axis = axes[i, j]
            if i == j:
                axis.axhline(1.0, color="black", linewidth=1)
                axis.text(0.5, 0.5, "identity", transform=axis.transAxes, ha="center", va="center")
            else:
                group = curves[(curves["source_scanner"] == source_scanner) & (curves["target_scanner"] == target_scanner)]
                table = group.pivot(index="slide_id", columns="frequency_cyc_per_pixel", values="log2_amplitude_ratio").reindex(columns=frequency)
                mean = table.mean(axis=0).to_numpy(float)
                q10 = table.quantile(0.10).to_numpy(float)
                q90 = table.quantile(0.90).to_numpy(float)
                axis.plot(frequency, np.exp2(mean), color="#1c5b6e", linewidth=1.2)
                axis.fill_between(frequency, np.exp2(q10), np.exp2(q90), color="#1c5b6e", alpha=0.18)
                axis.axhline(1.0, color="black", linewidth=0.5)
            if i == 0:
                axis.set_title(target_scanner.upper(), fontsize=9)
            if j == 0:
                axis.set_ylabel(source_scanner.upper(), fontsize=9)
            axis.set_xlim(0.03, 0.5)
            axis.set_ylim(0.25, 4.0)
            axis.set_yscale("log", base=2)
            axis.tick_params(labelsize=6)
    figure.supxlabel("Spatial frequency (cycles/pixel); columns are targets")
    figure.supylabel("Amplitude ratio; rows are sources")
    figure.suptitle("All directed scanner transfer curves (median-like mean and slide 10–90% band)", fontweight="bold")
    figure.savefig(output_root / "90_figures/supplement/supp_figure01_all_transfer_curves.png", dpi=180)
    plt.close(figure)


def correction_summaries(frame: pd.DataFrame, output_root: Path) -> dict:
    rows = []
    for arm in ALL_ARMS:
        group = frame[frame["arm"] == arm]
        row = {"arm": arm, "locations": group[["slide_id", "location_index"]].drop_duplicates().shape[0]}
        for column in ("distance", "endpoint_coverage", "all_endpoints_covered", "fidelity_ncc", "target_gradient_ncc", "saturation_fraction"):
            estimate, low, high = bootstrap_slide_mean(group, column, stable_seed("correction", arm, column))
            row[column] = estimate
            row[f"{column}_ci_low"] = low
            row[f"{column}_ci_high"] = high
        rows.append(row)
    arm_summary = pd.DataFrame(rows)
    scanner_rows = []
    for (arm, scanner), group in frame.groupby(["arm", "scanner"]):
        payload = {"arm": arm, "scanner": scanner}
        for column in ("distance", "endpoint_coverage", "all_endpoints_covered", "fidelity_ncc"):
            estimate, low, high = bootstrap_slide_mean(group, column, stable_seed("scanner", arm, scanner, column))
            payload[column] = estimate
            payload[f"{column}_ci_low"] = low
            payload[f"{column}_ci_high"] = high
        scanner_rows.append(payload)
    scanner_summary = pd.DataFrame(scanner_rows)
    endpoint_rows = []
    for (arm, scanner), group in frame.groupby(["arm", "scanner"]):
        for endpoint in ENDPOINTS:
            values = group.groupby("slide_id")[f"scaled_residual_{endpoint}"].apply(lambda x: np.mean(np.abs(x)))
            endpoint_rows.append({
                "arm": arm, "scanner": scanner, "endpoint": endpoint,
                "mean_absolute_scaled_residual": float(values.mean()),
            })
    endpoint_summary = pd.DataFrame(endpoint_rows)
    write_frame(output_root / "02_correction/arm_summary.csv", arm_summary)
    write_frame(output_root / "02_correction/scanner_summary.csv", scanner_summary)
    write_frame(output_root / "02_correction/endpoint_summary.csv", endpoint_summary)

    distance_contrast = paired_slide_contrast(frame, "combined", "reinhard", "distance", stable_seed("combined", "reinhard", "distance"))
    fidelity_contrast = paired_slide_contrast(frame, "combined", "reinhard", "fidelity_ncc", stable_seed("combined", "reinhard", "fidelity"))
    write_json(output_root / "02_correction/primary_contrast.json", {
        "combined_minus_reinhard_distance": {"estimate": distance_contrast[0], "ci_low": distance_contrast[1], "ci_high": distance_contrast[2]},
        "combined_minus_reinhard_fidelity_ncc": {"estimate": fidelity_contrast[0], "ci_low": fidelity_contrast[1], "ci_high": fidelity_contrast[2]},
    })
    make_correction_figures(frame, arm_summary, scanner_summary, endpoint_summary, output_root)
    return {
        "combined_minus_reinhard_distance": distance_contrast,
        "combined_minus_reinhard_fidelity_ncc": fidelity_contrast,
        "arm_summary": arm_summary.set_index("arm").to_dict(orient="index"),
    }


def make_correction_figures(frame: pd.DataFrame, arm_summary: pd.DataFrame, scanner_summary: pd.DataFrame, endpoint_summary: pd.DataFrame, output_root: Path) -> None:
    display = {"raw": "Raw", "reinhard": "Reinhard", "frequency": "Frequency", "combined": "Combined"}
    colors = ["#9aa6ac", "#3b78a8", "#bf8b2e", "#1f7a65"]
    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    primary = arm_summary.set_index("arm").loc[list(PRIMARY_ARMS)]
    x = np.arange(len(PRIMARY_ARMS))
    axes[0, 0].bar(x, primary["distance"], color=colors)
    axes[0, 0].errorbar(x, primary["distance"], yerr=[primary["distance"] - primary["distance_ci_low"], primary["distance_ci_high"] - primary["distance"]], fmt="none", color="black", capsize=3)
    axes[0, 0].set_xticks(x, [display[a] for a in PRIMARY_ARMS])
    axes[0, 0].set_ylabel("Target residual (robust-SD units)")
    axes[0, 0].set_title("A  Does frequency add beyond Reinhard?", loc="left", fontweight="bold")

    matrix = scanner_summary[scanner_summary["arm"].isin(PRIMARY_ARMS)].pivot(index="scanner", columns="arm", values="distance").reindex(index=TARGET_SCANNERS, columns=PRIMARY_ARMS)
    image = axes[0, 1].imshow(matrix, cmap="viridis", aspect="auto")
    axes[0, 1].set_xticks(range(len(PRIMARY_ARMS)), [display[a] for a in PRIMARY_ARMS], rotation=35, ha="right")
    axes[0, 1].set_yticks(range(len(TARGET_SCANNERS)), [s.upper() for s in TARGET_SCANNERS])
    axes[0, 1].set_title("B  Scanner-specific residual", loc="left", fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes[0, 1].text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", color="white" if matrix.iloc[i, j] > np.nanmedian(matrix.to_numpy()) else "black", fontsize=8)
    figure.colorbar(image, ax=axes[0, 1], shrink=0.8)

    slide = frame[frame["arm"].isin(["reinhard", "combined"])].pivot_table(index="slide_id", columns="arm", values="distance", aggfunc="mean")
    change = slide["combined"] - slide["reinhard"]
    axes[1, 0].axvline(0, color="black", linewidth=1)
    axes[1, 0].hist(change, bins=24, color="#1f7a65", alpha=0.85)
    axes[1, 0].set_xlabel("Combined minus Reinhard distance")
    axes[1, 0].set_ylabel("Slides")
    axes[1, 0].set_title("C  Slide-level paired change", loc="left", fontweight="bold")

    for arm, color in zip(ALL_ARMS, plt.cm.tab10(np.linspace(0, 1, len(ALL_ARMS)))):
        row = arm_summary.set_index("arm").loc[arm]
        axes[1, 1].scatter(row["fidelity_ncc"], row["distance"], color=color, s=70, label=arm)
    axes[1, 1].axvline(0.90, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_xlabel("Gradient fidelity to source")
    axes[1, 1].set_ylabel("Target residual")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("D  Correction–fidelity frontier", loc="left", fontweight="bold")
    axes[1, 1].legend(fontsize=7, ncol=2)
    figure.suptitle("Image-level correction scoreboard (held-out slides)", fontsize=17, fontweight="bold")
    figure.savefig(output_root / "90_figures/main/figure02_correction_scoreboard.png", dpi=180)
    plt.close(figure)

    oracle = arm_summary.set_index("arm").loc[["combined", "destructive_lowpass", "amplitude_oracle", "phase_oracle", "combined_loto"]]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].bar(np.arange(len(oracle)), oracle["distance"], color=plt.cm.Set2(np.linspace(0, 1, len(oracle))))
    axes[0].set_xticks(np.arange(len(oracle)), oracle.index, rotation=35, ha="right")
    axes[0].set_ylabel("Target residual")
    axes[0].set_title("A  Mechanism ceilings and destructive control", loc="left", fontweight="bold")
    axes[1].scatter(oracle["fidelity_ncc"], oracle["distance"], c=np.arange(len(oracle)), cmap="Set2", s=90)
    for arm, row in oracle.iterrows():
        axes[1].annotate(arm, (row["fidelity_ncc"], row["distance"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("Gradient fidelity to source")
    axes[1].set_ylabel("Target residual")
    axes[1].invert_yaxis()
    axes[1].set_title("B  Lower residual can still destroy content", loc="left", fontweight="bold")
    figure.savefig(output_root / "90_figures/supplement/supp_figure02_mechanism_frontier.png", dpi=180)
    plt.close(figure)

    combined = endpoint_summary[endpoint_summary["arm"].isin(["reinhard", "frequency", "combined"])].groupby(["arm", "endpoint"])["mean_absolute_scaled_residual"].mean().unstack(0).reindex(ENDPOINTS)
    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    image = axis.imshow(combined, cmap="magma", aspect="auto")
    axis.set_xticks(range(len(combined.columns)), [str(x).title() for x in combined.columns])
    axis.set_yticks(range(len(combined.index)), combined.index)
    axis.set_title("Endpoint residuals identify what each component fixes", loc="left", fontweight="bold")
    figure.colorbar(image, ax=axis, label="Mean absolute residual (robust SD)")
    figure.savefig(output_root / "90_figures/supplement/supp_figure03_endpoint_ablation.png", dpi=180)
    plt.close(figure)


def conditional_analyses(frame: pd.DataFrame, source_root: Path, augmentation_root: Path, output_root: Path) -> dict:
    unique_texture = frame[["slide_id", "location_index", "source_gradient_rms"]].drop_duplicates()
    unique_texture["texture_quartile"] = pd.qcut(unique_texture["source_gradient_rms"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    frame = frame.merge(unique_texture[["slide_id", "location_index", "texture_quartile"]], on=["slide_id", "location_index"], validate="many_to_one")
    texture = frame.groupby(["arm", "texture_quartile"], observed=True, as_index=False).agg(distance=("distance", "mean"), fidelity_ncc=("fidelity_ncc", "mean"), failure_rate=("all_endpoints_covered", lambda x: 1.0 - float(np.mean(x))))
    tissue = frame.groupby(["tissue_type", "scanner", "arm"], as_index=False).agg(distance=("distance", "mean"), fidelity_ncc=("fidelity_ncc", "mean"), locations=("location_index", "size"))
    reliability_rows = []
    for arm in ALL_ARMS:
        group = frame[frame["arm"] == arm]
        for threshold in np.linspace(0.80, 0.99, 20):
            accepted = group[group["fidelity_ncc"] >= threshold]
            reliability_rows.append({"arm": arm, "fidelity_threshold": threshold, "retained_fraction": len(accepted) / len(group), "mean_distance": float(accepted["distance"].mean()) if len(accepted) else np.nan})
    reliability = pd.DataFrame(reliability_rows)
    write_frame(output_root / "05_supplement/texture_strata.csv", texture)
    write_frame(output_root / "05_supplement/tissue_scanner_summary.csv", tissue)
    write_frame(output_root / "05_supplement/reliability_curves.csv", reliability)

    loto = frame[frame["arm"].isin(["combined", "combined_loto"])].pivot_table(index=["slide_id", "tissue_type", "scanner", "location_index"], columns="arm", values="distance").reset_index()
    loto["loto_minus_fold"] = loto["combined_loto"] - loto["combined"]
    loto_summary = loto.groupby(["tissue_type", "scanner"], as_index=False)["loto_minus_fold"].mean()
    write_frame(output_root / "05_supplement/loto_sensitivity.csv", loto_summary)

    augmentation = pd.read_csv(augmentation_root / "02_aggregate/selected_candidates.csv", dtype={"slide_id": str})
    augmentation = augmentation[augmentation["arm"] == "strong_oracle"]
    generated_cols = [f"generated_{endpoint}" for endpoint in ENDPOINTS]
    target_cols = [f"target_{endpoint}" for endpoint in ENDPOINTS]
    scales = np.asarray([augmentation[f"scale_{endpoint}"].median() for endpoint in ENDPOINTS])
    aug_matrix = augmentation[generated_cols].to_numpy(float) / scales
    target_matrix = augmentation[target_cols].to_numpy(float) / scales
    corr_aug = np.corrcoef(aug_matrix, rowvar=False)
    corr_target = np.corrcoef(target_matrix, rowvar=False)
    covariance_mismatch = float(np.linalg.norm(corr_aug - corr_target, ord="fro") / np.linalg.norm(corr_target, ord="fro"))
    np.savetxt(output_root / "04_mechanism/augmentation_correlation.csv", corr_aug, delimiter=",")
    np.savetxt(output_root / "04_mechanism/target_correlation.csv", corr_target, delimiter=",")
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for axis, matrix, title in zip(axes, [corr_target, corr_aug, corr_aug - corr_target], ["Real scanner", "Strong augmentation oracle", "Augmentation minus real"]):
        limit = 1 if "minus" not in title.lower() else max(abs(matrix.min()), abs(matrix.max()))
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_xticks(range(len(ENDPOINTS)), ENDPOINTS, rotation=90, fontsize=7)
        axis.set_yticks(range(len(ENDPOINTS)), ENDPOINTS, fontsize=7)
        axis.set_title(title, loc="left", fontweight="bold")
        figure.colorbar(image, ax=axis, shrink=0.75)
    figure.suptitle("Marginal reach does not guarantee the real joint endpoint geometry", fontweight="bold")
    figure.savefig(output_root / "90_figures/supplement/supp_figure04_augmentation_joint_manifold.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    combined_texture = texture[texture["arm"].isin(["reinhard", "combined"])]
    for arm, group in combined_texture.groupby("arm"):
        axes[0].plot(group["texture_quartile"].astype(str), group["distance"], marker="o", label=arm)
    axes[0].set_title("A  Residual by source texture", loc="left", fontweight="bold")
    axes[0].set_ylabel("Target residual")
    axes[0].legend()
    for arm in ["combined", "destructive_lowpass"]:
        group = reliability[reliability["arm"] == arm]
        axes[1].plot(group["fidelity_threshold"], group["retained_fraction"], label=arm)
    axes[1].set_title("B  Reliability as fidelity tightens", loc="left", fontweight="bold")
    axes[1].set_xlabel("Minimum gradient fidelity")
    axes[1].set_ylabel("Retained fraction")
    axes[1].legend()
    values = loto.groupby("tissue_type")["loto_minus_fold"].mean().sort_values()
    axes[2].barh(np.arange(len(values)), values, color=np.where(values > 0, "#b85c5c", "#4c8c75"))
    axes[2].set_yticks(np.arange(len(values)), values.index, fontsize=6)
    axes[2].axvline(0, color="black", linewidth=1)
    axes[2].set_title("C  Leave-one-tissue-out sensitivity", loc="left", fontweight="bold")
    axes[2].set_xlabel("LOTO minus fold-held-out residual")
    figure.savefig(output_root / "90_figures/supplement/supp_figure05_conditional_reliability.png", dpi=180)
    plt.close(figure)

    predictor = baseline_predictor(frame, source_root, output_root)
    return {
        "augmentation_joint_correlation_mismatch": covariance_mismatch,
        "maximum_absolute_loto_change": float(loto_summary["loto_minus_fold"].abs().max()),
        **predictor,
    }


def baseline_predictor(frame: pd.DataFrame, source_root: Path, output_root: Path) -> dict:
    spectra = pd.read_csv(source_root / "03_frequency/spectra.csv", dtype={"slide_id": str})
    at2 = spectra[spectra["scanner"] == REFERENCE].copy()
    features = []
    for slide_id, group in at2.groupby("slide_id"):
        f = group["frequency_cyc_per_pixel"].to_numpy(float)
        power = np.log10(np.maximum(group["radial_power"].to_numpy(float), 1e-12))
        payload = {"slide_id": slide_id}
        for name, bounds in {"low": (0.03, 0.12), "mid": (0.12, 0.28), "high": (0.28, 0.48)}.items():
            selected = (f >= bounds[0]) & (f < bounds[1])
            payload[f"source_log_power_{name}"] = float(power[selected].mean())
        features.append(payload)
    features = pd.DataFrame(features)
    outcomes = frame[frame["arm"].isin(["reinhard", "combined"])].pivot_table(index=["slide_id", "tissue_type", "scanner"], columns="arm", values="distance").reset_index()
    outcomes["benefit"] = outcomes["reinhard"] - outcomes["combined"]
    data = outcomes.merge(features, on="slide_id", validate="many_to_one")
    feature_cols = [column for column in features.columns if column != "slide_id"]
    predictions = []
    for scanner, scanner_data in data.groupby("scanner"):
        for tissue in scanner_data["tissue_type"].unique():
            train = scanner_data[scanner_data["tissue_type"] != tissue]
            test = scanner_data[scanner_data["tissue_type"] == tissue]
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            model.fit(train[feature_cols], train["benefit"])
            predicted = model.predict(test[feature_cols])
            for row, value in zip(test.itertuples(index=False), predicted):
                predictions.append({"slide_id": row.slide_id, "tissue_type": row.tissue_type, "scanner": scanner, "observed_benefit": row.benefit, "predicted_benefit": float(value)})
    predictions = pd.DataFrame(predictions)
    null = predictions.groupby("scanner")["observed_benefit"].transform("mean")
    ss_residual = float(np.square(predictions["observed_benefit"] - predictions["predicted_benefit"]).sum())
    ss_null = float(np.square(predictions["observed_benefit"] - null).sum())
    r2 = 1.0 - ss_residual / max(ss_null, 1e-12)
    mae = float(np.abs(predictions["observed_benefit"] - predictions["predicted_benefit"]).mean())
    null_mae = float(np.abs(predictions["observed_benefit"] - null).mean())
    write_frame(output_root / "05_supplement/baseline_spectrum_predictions.csv", predictions)
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for scanner, group in predictions.groupby("scanner"):
        axis.scatter(group["observed_benefit"], group["predicted_benefit"], s=22, alpha=0.75, label=scanner.upper())
    low = min(predictions["observed_benefit"].min(), predictions["predicted_benefit"].min())
    high = max(predictions["observed_benefit"].max(), predictions["predicted_benefit"].max())
    axis.plot([low, high], [low, high], color="black", linestyle="--")
    axis.set_xlabel("Observed combined-over-Reinhard benefit")
    axis.set_ylabel("Leave-one-tissue-out prediction")
    axis.set_title(f"Baseline spectrum prediction: R²={r2:.2f}, MAE={mae:.3f}", loc="left", fontweight="bold")
    axis.legend(fontsize=7, ncol=2)
    figure.savefig(output_root / "90_figures/supplement/supp_figure06_baseline_predictor.png", dpi=180)
    plt.close(figure)
    return {"baseline_predictor_loto_r2": r2, "baseline_predictor_mae": mae, "baseline_predictor_null_mae": null_mae}


def make_failure_gallery(frame: pd.DataFrame, output_root: Path) -> None:
    table = frame[frame["arm"].isin(["reinhard", "combined"])].pivot_table(index=["slide_id", "tissue_type", "scanner", "location_index"], columns="arm", values="distance").reset_index()
    table["benefit"] = table["reinhard"] - table["combined"]
    best = (
        table.sort_values("benefit", ascending=False)
        .groupby("scanner", as_index=False)
        .head(1)
        .assign(category="largest improvement")
    )
    worst = (
        table.sort_values("benefit", ascending=True)
        .groupby("scanner", as_index=False)
        .head(1)
        .assign(category="largest failure")
    )
    examples = pd.concat([best, worst], ignore_index=True).sort_values(["scanner", "category"])
    cohort = pd.read_csv(output_root / "00_contract/cohort.csv", dtype={"slide_id": str}).set_index("slide_id")
    parameters = json.loads((output_root / "02_correction/parameters.json").read_text())
    figure, axes = plt.subplots(len(examples), 4, figsize=(13, 3.1 * len(examples)), constrained_layout=True)
    for row_index, item in enumerate(examples.itertuples(index=False)):
        cohort_row = cohort.loc[str(item.slide_id)]
        with h5py.File(Path(cohort_row.cache_path), "r") as store:
            scanner_index = SCANNERS.index(item.scanner)
            source = np.asarray(store["images"][int(item.location_index), 0], dtype=np.uint8)
            target = np.asarray(store["images"][int(item.location_index), scanner_index], dtype=np.uint8)
        parameter = parameters["fold"][str(int(cohort_row["fold"]))][item.scanner]
        reinhard = reinhard_transform(source, parameter)
        combined = frequency_transform(reinhard, parameter)
        for axis, image, label in zip(axes[row_index], [source, target, reinhard, combined], ["AT2 source", f"{item.scanner.upper()} target", "Reinhard", "Combined"]):
            axis.imshow(image)
            axis.set_title(label, fontsize=9)
            axis.axis("off")
        axes[row_index, 0].set_ylabel(
            f"{item.scanner.upper()} · {item.category}\n{item.slide_id} · benefit {item.benefit:+.2f}",
            fontsize=8,
        )
    figure.suptitle("Scanner-balanced largest improvements and failures", fontsize=15, fontweight="bold")
    figure.savefig(output_root / "90_figures/supplement/supp_figure07_failure_gallery.png", dpi=160)
    plt.close(figure)
    write_frame(output_root / "05_supplement/failure_gallery_examples.csv", examples)


def aggregate(source_root: Path, augmentation_root: Path, output_root: Path) -> None:
    cohort = pd.read_csv(output_root / "00_contract/cohort.csv", dtype={"slide_id": str})
    frames = []
    for slide_id in cohort["slide_id"]:
        path = output_root / "02_correction/shards" / f"{slide_id}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path, dtype={"slide_id": str}))
    frame = pd.concat(frames, ignore_index=True)
    expected = len(cohort) * 40 * len(TARGET_SCANNERS) * len(ALL_ARMS)
    if len(frame) != expected:
        raise ValueError(f"expected {expected} correction rows, got {len(frame)}")
    write_frame(output_root / "02_correction/location_metrics.csv", frame)
    atlas_result = scanner_atlas(source_root, output_root)
    correction_result = correction_summaries(frame, output_root)
    conditional_result = conditional_analyses(frame, source_root, augmentation_root, output_root)
    make_failure_gallery(frame, output_root)
    summary = {
        "analysis_id": ANALYSIS_ID,
        "status": "image_analysis_pass",
        "completed_utc": utc_now(),
        "slides": len(cohort),
        "tissues": int(cohort["tissue_type"].nunique()),
        "locations": int(frame[["slide_id", "location_index"]].drop_duplicates().shape[0]),
        "target_comparisons": int(frame[["slide_id", "location_index", "scanner"]].drop_duplicates().shape[0]),
        "correction_rows": len(frame),
        "scanner_atlas": atlas_result,
        "correction": correction_result,
        "conditional": conditional_result,
        "claim_boundary": (
            "Frequency correction is an image-space scanner-global transform. Equivalent blur is a descriptive proxy; "
            "causal focus and optical MTF remain unidentified without calibration acquisitions."
        ),
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def uni_location_indices(slide_id: str, output_root: Path) -> list[int]:
    reference = pd.read_csv(output_root / "00_contract/target_references" / f"{slide_id}.csv")
    available = sorted(reference["location_index"].unique())
    metrics_path = SOURCE_ROOT / "02_image_phenotypes/shards" / slide_id / "location_metrics.h5"
    with h5py.File(metrics_path, "r") as store:
        replicate = np.asarray(store["replicate_id"], dtype=int)
    chosen = []
    for group in range(10):
        values = [int(index) for index in available if replicate[int(index)] == group]
        chosen.extend(values[:2])
    if len(chosen) != 20:
        raise ValueError(f"{slide_id}: expected 20 UNI locations, got {len(chosen)}")
    return sorted(chosen)


def uni_extract_task(task_index: int, task_count: int, output_root: Path, batch_size: int) -> None:
    import torch
    import torch.nn.functional as torch_functional
    from prenorm.embedding import load_uni

    if not torch.cuda.is_available():
        raise RuntimeError("UNI extraction requires CUDA")
    cohort = pd.read_csv(output_root / "00_contract/cohort.csv", dtype={"slide_id": str})
    assigned = cohort.iloc[task_index::task_count]
    parameters = json.loads((output_root / "02_correction/parameters.json").read_text())
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    model, size, mean, std = load_uni(device)
    for row in assigned.itertuples(index=False):
        slide_id = str(row.slide_id)
        output = output_root / "03_uni/shards" / f"{slide_id}.h5"
        summary_path = output.with_suffix(".summary.json")
        if output.exists() and summary_path.exists():
            old = json.loads(summary_path.read_text())
            if old.get("status") == "pass" and old.get("output_sha256") == sha256(output):
                print(json.dumps({"status": "already_complete", "slide_id": slide_id}))
                continue
        locations = uni_location_indices(slide_id, output_root)
        with h5py.File(Path(row.cache_path), "r") as store:
            rgb = np.asarray(store["images"][locations], dtype=np.uint8)
            source_indices = np.asarray(store["source_index"][locations], dtype=np.int64)
        names = ["source"] + [f"target:{scanner}" for scanner in TARGET_SCANNERS]
        for arm in ALL_ARMS[1:]:
            names.extend(f"{arm}:{scanner}" for scanner in TARGET_SCANNERS)
        name_to_index = {name: index for index, name in enumerate(names)}
        images = np.empty((len(locations), len(names), 256, 256, 3), dtype=np.uint8)
        images[:, 0] = rgb[:, 0]
        for scanner_index, scanner in enumerate(TARGET_SCANNERS, start=1):
            images[:, scanner_index] = rgb[:, scanner_index]
        for location_offset in range(len(locations)):
            source = rgb[location_offset, 0]
            for scanner_index, scanner in enumerate(TARGET_SCANNERS, start=1):
                target = rgb[location_offset, scanner_index]
                conditions = render_conditions(
                    source, target,
                    parameters["fold"][str(int(row.fold))][scanner],
                    parameters["loto"][str(row.tissue_type)][scanner],
                )
                for arm in ALL_ARMS[1:]:
                    images[location_offset, name_to_index[f"{arm}:{scanner}"]] = conditions[arm]
        flat = images.reshape(-1, 256, 256, 3)
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(flat), batch_size):
                stop = min(start + batch_size, len(flat))
                tensor = torch.from_numpy(flat[start:stop].astype(np.float32)).permute(0, 3, 1, 2).div_(255.0).to(device, non_blocking=True)
                tensor = torch_functional.interpolate(tensor, size=(size, size), mode="bicubic", align_corners=False)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feature = model((tensor - mean) / std)
                outputs.append(torch_functional.normalize(feature.float(), dim=1).cpu().numpy())
        features = np.concatenate(outputs).astype(np.float32).reshape(len(locations), len(names), -1)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        with h5py.File(temporary, "w") as store:
            store.attrs["analysis_id"] = ANALYSIS_ID
            store.attrs["slide_id"] = slide_id
            store.attrs["tissue_type"] = str(row.tissue_type)
            store.create_dataset("features", data=features, compression="lzf")
            store.create_dataset("condition_names", data=np.asarray(names, dtype="S48"))
            store.create_dataset("location_index", data=np.asarray(locations, dtype=np.int64))
            store.create_dataset("source_index", data=source_indices)
        temporary.replace(output)
        summary = {"analysis_id": ANALYSIS_ID, "status": "pass", "slide_id": slide_id, "locations": len(locations), "conditions": len(names), "feature_dim": features.shape[-1], "output_sha256": sha256(output)}
        write_json(summary_path, summary)
        print(json.dumps(summary), flush=True)


def uni_aggregate(output_root: Path) -> None:
    cohort = pd.read_csv(output_root / "00_contract/cohort.csv", dtype={"slide_id": str})
    rows = []
    feature_generated, feature_target = [], []
    for row in cohort.itertuples(index=False):
        path = output_root / "03_uni/shards" / f"{row.slide_id}.h5"
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as store:
            features = np.asarray(store["features"], dtype=np.float32)
            names = [value.decode() for value in store["condition_names"][:]]
            locations = np.asarray(store["location_index"], dtype=int)
        index = {name: offset for offset, name in enumerate(names)}
        source = features[:, index["source"]]
        for scanner in TARGET_SCANNERS:
            target = features[:, index[f"target:{scanner}"]]
            source_target = np.sum(source * target, axis=1)
            for arm in ALL_ARMS:
                generated = source if arm == "raw" else features[:, index[f"{arm}:{scanner}"]]
                generated_target = np.sum(generated * target, axis=1)
                generated_source = np.sum(generated * source, axis=1)
                for location_offset, location_index in enumerate(locations):
                    rows.append({
                        "slide_id": str(row.slide_id), "tissue_type": str(row.tissue_type), "fold": int(row.fold),
                        "location_index": int(location_index), "scanner": scanner, "arm": arm,
                        "source_target_cosine": float(source_target[location_offset]),
                        "generated_target_cosine": float(generated_target[location_offset]),
                        "generated_source_cosine": float(generated_source[location_offset]),
                        "target_gain": float(generated_target[location_offset] - source_target[location_offset]),
                    })
                    feature_generated.append(generated[location_offset])
                    feature_target.append(target[location_offset])
    frame = pd.DataFrame(rows)
    generated_matrix = np.asarray(feature_generated, dtype=np.float32)
    target_matrix = np.asarray(feature_target, dtype=np.float32)
    if len(frame) != len(generated_matrix):
        raise ValueError("UNI feature/metadata mismatch")
    prediction_rows = []
    for arm in ALL_ARMS:
        arm_mask = frame["arm"].to_numpy() == arm
        for fold in sorted(frame["fold"].unique()):
            train = arm_mask & (frame["fold"].to_numpy() != fold)
            test = arm_mask & (frame["fold"].to_numpy() == fold)
            x_train = np.concatenate([generated_matrix[train], target_matrix[train]])
            y_train = np.concatenate([np.zeros(train.sum()), np.ones(train.sum())])
            x_test = np.concatenate([generated_matrix[test], target_matrix[test]])
            y_test = np.concatenate([np.zeros(test.sum()), np.ones(test.sum())])
            classifier = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, solver="liblinear"))
            classifier.fit(x_train, y_train)
            predicted = classifier.predict(x_test)
            metadata = pd.concat([frame.loc[test, ["slide_id", "location_index"]], frame.loc[test, ["slide_id", "location_index"]]], ignore_index=True)
            for item, truth, guess in zip(metadata.itertuples(index=False), y_test, predicted):
                prediction_rows.append({"arm": arm, "fold": int(fold), "slide_id": item.slide_id, "location_index": int(item.location_index), "truth": int(truth), "predicted": int(guess), "correct": int(truth == guess)})
    predictions = pd.DataFrame(prediction_rows)
    summary_rows = []
    for arm in ALL_ARMS:
        group = frame[frame["arm"] == arm]
        accuracy = predictions[predictions["arm"] == arm].groupby("slide_id")["correct"].mean().reset_index()
        payload = {"arm": arm}
        for column in ("generated_target_cosine", "generated_source_cosine", "target_gain"):
            estimate, low, high = bootstrap_slide_mean(group, column, stable_seed("uni", arm, column))
            payload[column] = estimate
            payload[f"{column}_ci_low"] = low
            payload[f"{column}_ci_high"] = high
        estimate, low, high = bootstrap_slide_mean(accuracy, "correct", stable_seed("uni", arm, "accuracy"))
        payload["real_vs_corrected_balanced_accuracy"] = estimate
        payload["real_vs_corrected_balanced_accuracy_ci_low"] = low
        payload["real_vs_corrected_balanced_accuracy_ci_high"] = high
        summary_rows.append(payload)
    summary = pd.DataFrame(summary_rows)
    write_frame(output_root / "03_uni/location_metrics.csv", frame)
    write_frame(output_root / "03_uni/classifier_predictions.csv", predictions)
    write_frame(output_root / "03_uni/summary.csv", summary)
    contrast = paired_slide_contrast(frame, "combined", "reinhard", "target_gain", stable_seed("uni", "combined-reinhard"))
    result = {"analysis_id": ANALYSIS_ID, "status": "pass", "slides": len(cohort), "locations": int(frame[["slide_id", "location_index"]].drop_duplicates().shape[0]), "combined_minus_reinhard_target_gain": {"estimate": contrast[0], "ci_low": contrast[1], "ci_high": contrast[2]}}
    write_json(output_root / "03_uni/summary.json", result)

    display = {arm: arm.replace("_", " ").title() for arm in ALL_ARMS}
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    primary = summary.set_index("arm").loc[list(PRIMARY_ARMS)]
    x = np.arange(len(primary))
    palette = ["#9aa6ac", "#3b78a8", "#bf8b2e", "#1f7a65"]
    target_error = np.vstack([
        primary["target_gain"] - primary["target_gain_ci_low"],
        primary["target_gain_ci_high"] - primary["target_gain"],
    ])
    axes[0].bar(x, primary["target_gain"], color=palette, yerr=target_error, capsize=3)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xticks(x, [display[a] for a in PRIMARY_ARMS], rotation=30, ha="right")
    axes[0].set_ylabel("Cosine gain toward real target")
    axes[0].set_title("A  Frozen UNI target gain", loc="left", fontweight="bold")
    axes[0].text(
        0.02,
        0.96,
        f"Combined − Reinhard\n{contrast[0]:+.4f} [{contrast[1]:+.4f}, {contrast[2]:+.4f}]",
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#bcc8ce", "alpha": 0.92},
    )
    source_error = np.vstack([
        primary["generated_source_cosine"] - primary["generated_source_cosine_ci_low"],
        primary["generated_source_cosine_ci_high"] - primary["generated_source_cosine"],
    ])
    axes[1].bar(x, primary["generated_source_cosine"], color=palette, yerr=source_error, capsize=3)
    axes[1].set_xticks(x, [display[a] for a in PRIMARY_ARMS], rotation=30, ha="right")
    axes[1].set_ylabel("Cosine to source")
    axes[1].set_title("B  Representation fidelity", loc="left", fontweight="bold")
    classifier_error = np.vstack([
        primary["real_vs_corrected_balanced_accuracy"] - primary["real_vs_corrected_balanced_accuracy_ci_low"],
        primary["real_vs_corrected_balanced_accuracy_ci_high"] - primary["real_vs_corrected_balanced_accuracy"],
    ])
    axes[2].bar(
        x,
        primary["real_vs_corrected_balanced_accuracy"],
        color=palette,
        yerr=classifier_error,
        capsize=3,
    )
    axes[2].axhline(0.5, color="black", linestyle="--")
    axes[2].set_xticks(x, [display[a] for a in PRIMARY_ARMS], rotation=30, ha="right")
    axes[2].set_ylabel("Balanced accuracy")
    axes[2].set_title("C  Real target remains distinguishable", loc="left", fontweight="bold")
    figure.suptitle("Locked frozen-UNI consequence test", fontsize=16, fontweight="bold")
    figure.savefig(output_root / "90_figures/main/figure03_uni_correction_bridge.png", dpi=180)
    plt.close(figure)
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--augmentation-root", type=Path, default=AUGMENTATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    fit = subparsers.add_parser("fit-slide")
    fit.add_argument("--slide-index", type=int, required=True)
    subparsers.add_parser("fit-aggregate")
    extract = subparsers.add_parser("extract-slide")
    extract.add_argument("--slide-index", type=int, required=True)
    extract.add_argument("--workers", type=int, default=2)
    subparsers.add_parser("aggregate")
    uni = subparsers.add_parser("uni-extract-task")
    uni.add_argument("--task-index", type=int, required=True)
    uni.add_argument("--task-count", type=int, required=True)
    uni.add_argument("--batch-size", type=int, default=32)
    subparsers.add_parser("uni-aggregate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    augmentation_root = args.augmentation_root.resolve()
    output_root = args.output_root.resolve()
    if args.command == "prepare":
        prepare(source_root, augmentation_root, output_root)
    elif args.command == "fit-slide":
        fit_slide(args.slide_index, output_root)
    elif args.command == "fit-aggregate":
        fit_aggregate(source_root, output_root)
    elif args.command == "extract-slide":
        extract_slide(args.slide_index, output_root, args.workers)
    elif args.command == "aggregate":
        aggregate(source_root, augmentation_root, output_root)
    elif args.command == "uni-extract-task":
        uni_extract_task(args.task_index, args.task_count, output_root, args.batch_size)
    elif args.command == "uni-aggregate":
        uni_aggregate(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
