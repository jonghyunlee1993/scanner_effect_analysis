#!/usr/bin/env python3
"""Image-level paired scanner study for the final PanNormal manuscript.

The program has three stages:

``prepare``
    Freeze the 103-slide cohort, tissue annotations, estimands, and provenance.
``extract-slide``
    Measure one HDF5 cache shard.  This is the SLURM-array unit.
``aggregate``
    Combine completed shards, derive paired contrasts, fit nested REML models,
    and render manuscript-facing figures and evidence tables.

The primary unit is a paired image location.  Ten spatially distributed groups
per slide are used only to estimate within-slide sampling error in the LMM.
No learned features are used by this analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
from datetime import datetime, timezone

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fft, linalg, optimize, stats
from scipy.cluster.hierarchy import leaves_list, linkage


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data/preprocessing_patch_extraction_matching_and_QC"
OUTPUT_ROOT = ROOT / "outputs/final_image_study_v1"
TISSUE_SOURCE = (
    ROOT
    / "data/provenance/pannormal_native_geometry_manifest.csv"
)
ANALYSIS_VERSION = "final_image_study_v1"
SCANNERS = ("at2", "versa", "akoya", "gt450", "s360", "s60")
REFERENCE = "at2"
PATCH_SIZE = 256
REPLICATE_GROUPS = 10
SPECTRAL_BINS = 72

# The registered image plane is believed to be the historical AT2 20x grid.
# The cache does not encode MPP, so physical units remain a provisional bridge
# until the preprocessing provenance is amended.  Pixel-frequency values are
# always emitted as the provenance-independent source measurement.
PROVISIONAL_MPP = 0.5052
BANDS_CYC_PER_UM = {
    "low_mid": (0.10, 0.30),
    "mid": (0.30, 0.60),
    "high": (0.60, 0.90),
}

SCALAR_METRICS = (
    "lab_l_mean",
    "lab_a_mean",
    "lab_b_mean",
    "lab_l_sd",
    "mean_od",
    "od_sd",
    "gradient_rms",
    "gradient_ncc_at2",
    "saturation_fraction",
    "tissue_fraction",
    "laplacian_variance",
)

MODEL_ENDPOINTS = {
    "delta_lab_l": {
        "source": "lab_l_mean",
        "transform": "difference",
        "label": "Δ CIELAB L*",
        "family": "color",
    },
    "delta_lab_a": {
        "source": "lab_a_mean",
        "transform": "difference",
        "label": "Δ CIELAB a*",
        "family": "color",
    },
    "delta_lab_b": {
        "source": "lab_b_mean",
        "transform": "difference",
        "label": "Δ CIELAB b*",
        "family": "color",
    },
    "log2_mean_od_ratio": {
        "source": "mean_od",
        "transform": "log2_ratio",
        "label": "log₂ mean-OD ratio",
        "family": "color",
    },
    "log2_lab_l_sd_ratio": {
        "source": "lab_l_sd",
        "transform": "log2_ratio",
        "label": "log₂ L* contrast ratio",
        "family": "contrast",
    },
    "log2_od_sd_ratio": {
        "source": "od_sd",
        "transform": "log2_ratio",
        "label": "log₂ OD contrast ratio",
        "family": "contrast",
    },
    "log2_gradient_rms_ratio": {
        "source": "gradient_rms",
        "transform": "log2_ratio",
        "label": "log₂ gradient-energy ratio",
        "family": "structure",
    },
    "gradient_dissimilarity": {
        "source": "gradient_ncc_at2",
        "transform": "one_minus",
        "label": "1 − gradient NCC to AT2",
        "family": "structure",
    },
    "delta_tissue_fraction": {
        "source": "tissue_fraction",
        "transform": "difference",
        "label": "Δ tissue fraction",
        "family": "selection diagnostic",
    },
    "log2_laplacian_ratio": {
        "source": "laplacian_variance",
        "transform": "log2_ratio",
        "label": "log₂ Laplacian-variance ratio",
        "family": "QC diagnostic",
    },
}


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


def read_tissue_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            slide_id = str(row["slide_id"])
            tissue = str(row["tissue_type"]).strip()
            previous = mapping.setdefault(slide_id, tissue)
            if previous != tissue:
                raise ValueError(f"multiple tissue labels for {slide_id}: {previous}, {tissue}")
    return mapping


def prepare(data_root: Path, output_root: Path, tissue_source: Path) -> None:
    caches = sorted((data_root / "cache").glob("*.h5"), key=lambda p: p.stem)
    if not caches:
        raise FileNotFoundError(f"no HDF5 cache files under {data_root / 'cache'}")
    tissue_map = read_tissue_map(tissue_source)
    qc = pd.read_csv(data_root / "qc/slide_summary.csv", dtype={"slide_id": str})
    qc = qc.set_index("slide_id", verify_integrity=True)
    rows = []
    for array_index, path in enumerate(caches):
        slide_id = path.stem
        if slide_id not in tissue_map:
            raise KeyError(f"missing tissue annotation for {slide_id}")
        if slide_id not in qc.index:
            raise KeyError(f"missing preprocessing QC row for {slide_id}")
        with h5py.File(path, "r") as store:
            scanners = [value.decode() for value in store["scanner_names"][:]]
            pair_count = int(store["images"].shape[0])
            created_utc = str(store.attrs["created_utc"])
        if scanners != list(SCANNERS):
            raise ValueError(f"{slide_id}: unexpected scanner order {scanners}")
        stat = path.stat()
        row = qc.loc[slide_id]
        rows.append(
            {
                "array_index": array_index,
                "slide_id": slide_id,
                "tissue_type": tissue_map[slide_id],
                "pair_count": pair_count,
                "source_candidate_count": int(row["source_candidate_count"]),
                "processed_candidate_count": int(row["processed_candidate_count"]),
                "cap_reached": bool(row["cap_reached"]),
                "cache_path": str(path.resolve()),
                "cache_bytes": stat.st_size,
                "cache_mtime_ns": stat.st_mtime_ns,
                "cache_created_utc": created_utc,
            }
        )
    cohort = pd.DataFrame(rows)
    if len(cohort) != 103:
        raise ValueError(f"frozen final cohort requires 103 slides, got {len(cohort)}")

    for relative in (
        "00_contract",
        "01_pairing_audit",
        "02_image_phenotypes/shards",
        "03_frequency/shards",
        "04_lmm",
        "90_report_assets",
        "logs",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    cohort_path = output_root / "00_contract/cohort.csv"
    cohort.to_csv(cohort_path, index=False)

    run_summary_path = data_root / "qc/run_summary.json"
    preprocessing_config = data_root / "config/config.json"
    contract = {
        "analysis_version": ANALYSIS_VERSION,
        "created_utc": utc_now(),
        "primary_question": (
            "Do paired scanner/acquisition effects comprise separable color, contrast, "
            "and spatial-frequency components with tissue- and slide-dependent heterogeneity?"
        ),
        "primary_unit": "paired image location",
        "inference_unit": "slide nested in tissue type",
        "reference_scanner": REFERENCE,
        "scanners": list(SCANNERS),
        "slides": len(cohort),
        "tissues": int(cohort["tissue_type"].nunique()),
        "paired_locations": int(cohort["pair_count"].sum()),
        "spatial_replicates_per_slide": REPLICATE_GROUPS,
        "replicate_purpose": "estimate spatial sampling error; not independent biological n",
        "scalar_metrics": list(SCALAR_METRICS),
        "model_endpoints": MODEL_ENDPOINTS,
        "spectrum": {
            "estimator": (
                "mean total OD; patch-mean removal; 2D Hann; radial mean power; "
                "AT2-normalized amplitude ratio; 0.03-0.10 cyc/um scale normalization"
            ),
            "radial_bins": SPECTRAL_BINS,
            "bands_cyc_per_um": BANDS_CYC_PER_UM,
            "frequency_source_measurement": "cycles_per_pixel",
            "physical_scale_mpp": PROVISIONAL_MPP,
            "physical_scale_status": "provisional_until_cache_or_registration_provenance_is_amended",
        },
        "lmm": (
            "endpoint = scanner fixed mean + scanner-by-tissue random intercept + "
            "scanner-by-slide-within-tissue random intercept + replicate sampling error"
        ),
        "claim_boundary": (
            "The estimand is the registered acquisition pipeline represented in the cache; "
            "a hardware-only scanner effect is not identified."
        ),
        "input_provenance": {
            "preprocessing_run_summary": str(run_summary_path.resolve()),
            "preprocessing_run_summary_sha256": sha256(run_summary_path),
            "preprocessing_config": str(preprocessing_config.resolve()),
            "preprocessing_config_sha256": sha256(preprocessing_config),
            "tissue_annotation_source": str(tissue_source.resolve()),
            "tissue_annotation_source_sha256": sha256(tissue_source),
            "cohort_sha256": sha256(cohort_path),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "h5py": h5py.__version__,
        },
    }
    write_json(output_root / "00_contract/analysis_contract.json", contract)

    audit = {
        "analysis_version": ANALYSIS_VERSION,
        "slides": len(cohort),
        "tissues": int(cohort["tissue_type"].nunique()),
        "paired_locations": int(cohort["pair_count"].sum()),
        "pair_count": {
            "minimum": int(cohort["pair_count"].min()),
            "median": float(cohort["pair_count"].median()),
            "maximum": int(cohort["pair_count"].max()),
        },
        "cap_reached_slides": int(cohort["cap_reached"].sum()),
        "source_candidates": int(cohort["source_candidate_count"].sum()),
        "processed_candidates": int(cohort["processed_candidate_count"].sum()),
        "accepted_pairs": int(cohort["pair_count"].sum()),
        "accepted_fraction_among_processed": float(
            cohort["pair_count"].sum() / cohort["processed_candidate_count"].sum()
        ),
        "tissue_slide_counts": {
            str(k): int(v)
            for k, v in cohort.groupby("tissue_type")["slide_id"].nunique().sort_index().items()
        },
    }
    write_json(output_root / "01_pairing_audit/summary.json", audit)
    print(json.dumps({"status": "prepared", **audit}, indent=2))


def audit_plism(plism_root: Path, output_root: Path) -> None:
    """Record whether the current PLISM tree can support the external arm."""

    resolved = plism_root.resolve()
    manifest_path = plism_root / "manifest_original.csv"
    provenance_path = plism_root / "PROVENANCE.md"
    alignment_root = plism_root / "alignment"
    manifest = pd.read_csv(manifest_path, dtype={"stain": str, "scanner": str})
    file_rows = []
    for row in manifest.itertuples(index=False):
        path = plism_root / "original_wsi" / str(row.name)
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        file_rows.append(
            {
                "name": str(row.name),
                "exists": exists,
                "expected_bytes": int(row.size_bytes),
                "actual_bytes": actual_bytes,
                "size_match": bool(exists and actual_bytes == int(row.size_bytes)),
            }
        )
    common = sorted(
        set(manifest["scanner"].astype(str).str.upper())
        & {scanner.upper() for scanner in SCANNERS}
    )
    required_alignment = (
        "per_section.csv",
        "per_location.csv.gz",
        "section_transform.csv",
        "file_corrections.csv",
        "core_map.csv",
    )
    payload = {
        "audit_version": "plism_readiness_v1",
        "created_utc": utc_now(),
        "status": "usable_as_external_robustness_not_poolable_as_confirmatory_replication",
        "root_symlink": str(plism_root),
        "resolved_root": str(resolved),
        "root_exists": resolved.is_dir(),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "provenance": str(provenance_path.resolve()),
        "provenance_sha256": sha256(provenance_path),
        "expected_wsi": int(len(manifest)),
        "present_wsi": int(sum(item["exists"] for item in file_rows)),
        "size_matched_wsi": int(sum(item["size_match"] for item in file_rows)),
        "md5_status": "verified_at_download_per_provenance_not_recomputed_in_this_audit",
        "sections": int(manifest["stain"].nunique()),
        "scanners": int(manifest["scanner"].nunique()),
        "scanner_names": sorted(manifest["scanner"].unique().tolist()),
        "shared_scanners": common,
        "native_mpp_range": [
            float(manifest["native_mpp"].min()),
            float(manifest["native_mpp"].max()),
        ],
        "alignment_artifacts": {
            name: (alignment_root / name).is_file() for name in required_alignment
        },
        "required_corrections": {
            "sq_givh_hrh_file_swap": True,
            "joint_location_gate": "residual_um <= 1.0 and response >= 0.3",
            "s60_hrh_out_of_focus": (
                "retain_and_flag_in_primary_then_report_sensitivity_exclusion"
            ),
            "resampling_unit": "section",
        },
        "endpoint_compatibility": {
            "mean_lab_od_contrast": "usable_with_identical_definition",
            "native_physical_spectrum": (
                "usable_after_internal_effective_mpp_gate"
            ),
            "pixelwise_gradient_ncc": "not_equivalent_without_common_grid",
            "laplacian_variance": "requires_physical_scale_normalization",
            "scanner_by_tissue_lmm": (
                "not_identically_replicable_due_to_stain_section_confounding"
            ),
        },
        "file_audit_failures": [
            item
            for item in file_rows
            if not item["exists"] or not item["size_match"]
        ],
    }
    if payload["present_wsi"] != 91 or payload["size_matched_wsi"] != 91:
        payload["status"] = "blocked_by_missing_or_size_mismatched_wsi"
    write_json(output_root / "00_contract/plism_readiness.json", payload)
    print(json.dumps(payload, indent=2))


def spatial_replicates(coords: np.ndarray, source_index: np.ndarray) -> np.ndarray:
    """Distribute every spatial cell across all replicate groups deterministically."""

    coords = np.asarray(coords, dtype=np.float64)
    bins = []
    for axis in range(2):
        values = coords[:, axis]
        low, high = float(values.min()), float(values.max())
        if high <= low:
            bins.append(np.zeros(len(values), dtype=np.int64))
        else:
            scaled = np.floor(10.0 * (values - low) / (high - low + 1e-9)).astype(int)
            bins.append(np.clip(scaled, 0, 9))
    cells = bins[1] * 10 + bins[0]
    result = np.empty(len(coords), dtype=np.int16)
    for cell in np.unique(cells):
        selected = np.flatnonzero(cells == cell)
        order = selected[np.argsort(source_index[selected], kind="stable")]
        result[order] = (np.arange(len(order)) + int(cell)) % REPLICATE_GROUPS
    if set(result.tolist()) != set(range(REPLICATE_GROUPS)):
        raise ValueError("spatial replicate allocation produced an empty group")
    return result


def srgb_to_lab(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(rgb, dtype=np.float32) / 255.0
    linear = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
    x = 0.4124564 * linear[..., 0] + 0.3575761 * linear[..., 1] + 0.1804375 * linear[..., 2]
    y = 0.2126729 * linear[..., 0] + 0.7151522 * linear[..., 1] + 0.0721750 * linear[..., 2]
    z = 0.0193339 * linear[..., 0] + 0.1191920 * linear[..., 1] + 0.9503041 * linear[..., 2]
    xyz = (x / 0.95047, y, z / 1.08883)

    def f(component: np.ndarray) -> np.ndarray:
        delta = 6.0 / 29.0
        return np.where(
            component > delta**3,
            np.cbrt(component),
            component / (3.0 * delta**2) + 4.0 / 29.0,
        )

    fx, fy, fz = (f(component) for component in xyz)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def image_metrics(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return scalar metrics, mean OD image, and OD-gradient magnitude."""

    l_star, a_star, b_star = srgb_to_lab(rgb)
    od = -np.log((np.asarray(rgb, dtype=np.float32) + 1.0) / 256.0).mean(axis=-1)
    gx = od[:, 1:-1, 2:] - od[:, 1:-1, :-2]
    gy = od[:, 2:, 1:-1] - od[:, :-2, 1:-1]
    gradient = np.sqrt(np.maximum(gx * gx + gy * gy, 0.0))
    saturation = ((rgb <= 1) | (rgb >= 254)).any(axis=-1)
    metrics = np.column_stack(
        [
            l_star.mean(axis=(1, 2)),
            a_star.mean(axis=(1, 2)),
            b_star.mean(axis=(1, 2)),
            l_star.std(axis=(1, 2)),
            od.mean(axis=(1, 2)),
            od.std(axis=(1, 2)),
            np.sqrt(np.mean(gx * gx + gy * gy, axis=(1, 2))),
            np.full(len(rgb), np.nan),
            saturation.mean(axis=(1, 2)),
        ]
    ).astype(np.float32)
    return metrics, od, gradient


def rowwise_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = left.reshape(len(left), -1).astype(np.float64)
    y = right.reshape(len(right), -1).astype(np.float64)
    x -= x.mean(axis=1, keepdims=True)
    y -= y.mean(axis=1, keepdims=True)
    numerator = np.sum(x * y, axis=1)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def frequency_geometry(size: int, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    one_d = np.fft.fftfreq(size)
    radius = np.sqrt(one_d[:, None] ** 2 + one_d[None, :] ** 2)
    edges = np.linspace(0.0, 0.5, bins + 1)
    index = np.digitize(radius.ravel(), edges) - 1
    valid = (index >= 0) & (index < bins)
    counts = np.bincount(index[valid], minlength=bins).astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return index, valid, counts, centers


def radial_mean(value: np.ndarray, index: np.ndarray, valid: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.bincount(
        index[valid], weights=np.asarray(value).ravel()[valid], minlength=len(counts)
    ) / np.maximum(counts, 1.0)


def normalized_transfer(power: np.ndarray, reference: np.ndarray, frequency_um: np.ndarray) -> np.ndarray:
    ratio = np.sqrt(np.maximum(power, 1e-20) / np.maximum(reference, 1e-20))
    normalization = (frequency_um >= 0.03) & (frequency_um <= 0.10)
    scale = np.exp(np.mean(np.log(np.maximum(ratio[normalization], 1e-20))))
    return ratio / scale


def geometric_band(curve: np.ndarray, frequency_um: np.ndarray, bounds: tuple[float, float]) -> float:
    selected = (frequency_um >= bounds[0]) & (frequency_um < bounds[1])
    if not selected.any():
        raise ValueError(f"empty spectral band {bounds}")
    return float(np.exp(np.mean(np.log(np.maximum(curve[selected], 1e-20)))))


def write_frame_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def extract_slide(
    slide_index: int,
    output_root: Path,
    chunk_size: int,
    workers: int,
) -> None:
    cohort_path = output_root / "00_contract/cohort.csv"
    cohort_hash = sha256(cohort_path)
    cohort = pd.read_csv(cohort_path, dtype={"slide_id": str})
    if slide_index < 0 or slide_index >= len(cohort):
        raise IndexError(f"slide index {slide_index} outside 0..{len(cohort) - 1}")
    row = cohort.iloc[slide_index]
    slide_id = str(row["slide_id"])
    source = Path(row["cache_path"])
    slide_output = output_root / "02_image_phenotypes/shards" / slide_id
    spectrum_output = output_root / "03_frequency/shards" / slide_id
    summary_path = slide_output / "summary.json"
    required = (
        slide_output / "location_metrics.h5",
        slide_output / "replicate_scalar_summary.csv",
        spectrum_output / "spectra.csv",
        spectrum_output / "band_summary.csv",
        spectrum_output / "replicate_band_summary.csv",
    )
    if summary_path.exists() and all(path.exists() for path in required):
        existing = json.loads(summary_path.read_text())
        if (
            existing.get("analysis_version") == ANALYSIS_VERSION
            and existing.get("cohort_sha256") == cohort_hash
            and existing.get("source_cache_bytes") == source.stat().st_size
            and existing.get("status") == "pass"
        ):
            print(json.dumps({"status": "already_complete", "slide_id": slide_id}))
            return

    slide_output.mkdir(parents=True, exist_ok=True)
    spectrum_output.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as store:
        scanner_names = [value.decode() for value in store["scanner_names"][:]]
        if scanner_names != list(SCANNERS):
            raise ValueError(f"{slide_id}: scanner order mismatch")
        n_locations = int(store["images"].shape[0])
        if n_locations != int(row["pair_count"]):
            raise ValueError(f"{slide_id}: cohort/cache pair count mismatch")
        source_index = np.asarray(store["source_index"], dtype=np.int64)
        coords = np.asarray(store["phase_coords_xy"][:, 0, :], dtype=np.int64)
        replicate_id = spatial_replicates(coords, source_index)
        metrics = np.empty((n_locations, len(SCANNERS), len(SCALAR_METRICS)), dtype=np.float32)
        metrics[:, :, -2] = np.asarray(store["quality/tissue_fraction"], dtype=np.float32)
        metrics[:, :, -1] = np.asarray(store["quality/laplacian_variance"], dtype=np.float32)

        power_sum = np.zeros(
            (REPLICATE_GROUPS, len(SCANNERS), PATCH_SIZE, PATCH_SIZE), dtype=np.float64
        )
        cross_sum = np.zeros(
            (REPLICATE_GROUPS, len(SCANNERS), PATCH_SIZE, PATCH_SIZE), dtype=np.complex128
        )
        replicate_counts = np.bincount(replicate_id, minlength=REPLICATE_GROUPS).astype(int)
        window_1d = np.hanning(PATCH_SIZE).astype(np.float32)
        window = window_1d[:, None] * window_1d[None, :]

        for start in range(0, n_locations, chunk_size):
            stop = min(start + chunk_size, n_locations)
            rgb = np.asarray(store["images"][start:stop], dtype=np.uint8)
            batch = stop - start
            flattened = rgb.reshape(batch * len(SCANNERS), PATCH_SIZE, PATCH_SIZE, 3)
            scalar, od, gradient = image_metrics(flattened)
            scalar = scalar.reshape(batch, len(SCANNERS), -1)
            od = od.reshape(batch, len(SCANNERS), PATCH_SIZE, PATCH_SIZE)
            gradient = gradient.reshape(batch, len(SCANNERS), PATCH_SIZE - 2, PATCH_SIZE - 2)
            reference_gradient = gradient[:, 0]
            for scanner_index in range(len(SCANNERS)):
                scalar[:, scanner_index, 7] = rowwise_correlation(
                    gradient[:, scanner_index], reference_gradient
                )
            metrics[start:stop, :, :9] = scalar

            fourier = []
            for scanner_index in range(len(SCANNERS)):
                value = od[:, scanner_index]
                value = value - value.mean(axis=(1, 2), keepdims=True)
                fourier.append(
                    fft.fft2(value * window[None, :, :], axes=(-2, -1), workers=workers)
                )
            reference_fft = fourier[0]
            group_ids = replicate_id[start:stop]
            for replicate in range(REPLICATE_GROUPS):
                selected = group_ids == replicate
                if not selected.any():
                    continue
                for scanner_index, spectrum in enumerate(fourier):
                    power_sum[replicate, scanner_index] += np.sum(
                        np.abs(spectrum[selected]) ** 2, axis=0
                    )
                    cross_sum[replicate, scanner_index] += np.sum(
                        spectrum[selected] * np.conj(reference_fft[selected]), axis=0
                    )
            print(f"[{slide_id}] {stop}/{n_locations}", flush=True)

    if not np.isfinite(metrics).all():
        raise ValueError(f"{slide_id}: non-finite scalar metric")
    if not np.allclose(metrics[:, 0, 7], 1.0, atol=1e-5):
        raise ValueError(f"{slide_id}: AT2 self NCC differs from one")

    location_path = slide_output / "location_metrics.h5"
    temporary_h5 = location_path.with_suffix(".h5.tmp")
    with h5py.File(temporary_h5, "w") as out:
        out.attrs["analysis_version"] = ANALYSIS_VERSION
        out.attrs["slide_id"] = slide_id
        out.attrs["tissue_type"] = str(row["tissue_type"])
        out.attrs["reference_scanner"] = REFERENCE
        out.create_dataset("metrics", data=metrics, compression="gzip", compression_opts=4)
        out.create_dataset("metric_names", data=np.asarray(SCALAR_METRICS, dtype="S32"))
        out.create_dataset("scanner_names", data=np.asarray(SCANNERS, dtype="S8"))
        out.create_dataset("replicate_id", data=replicate_id)
        out.create_dataset("source_index", data=source_index)
        out.create_dataset("phase_coords_xy", data=coords)
    temporary_h5.replace(location_path)

    scalar_rows = []
    for replicate in range(REPLICATE_GROUPS):
        selected = replicate_id == replicate
        for scanner_index, scanner in enumerate(SCANNERS):
            payload = {
                "slide_id": slide_id,
                "tissue_type": str(row["tissue_type"]),
                "replicate": replicate,
                "patches_in_replicate": int(selected.sum()),
                "scanner": scanner,
            }
            payload.update(
                {
                    metric: float(metrics[selected, scanner_index, metric_index].mean())
                    for metric_index, metric in enumerate(SCALAR_METRICS)
                }
            )
            scalar_rows.append(payload)
    write_frame_atomic(
        pd.DataFrame(scalar_rows), slide_output / "replicate_scalar_summary.csv"
    )

    index, valid, radial_counts, frequency_px = frequency_geometry(PATCH_SIZE, SPECTRAL_BINS)
    frequency_um = frequency_px / PROVISIONAL_MPP
    mean_power = power_sum.sum(axis=0) / float(n_locations)
    mean_cross = cross_sum.sum(axis=0) / float(n_locations)
    reference_power = mean_power[0]
    spectra_rows = []
    band_rows = []
    replicate_band_rows = []
    for scanner_index, scanner in enumerate(SCANNERS):
        radial_power = radial_mean(mean_power[scanner_index], index, valid, radial_counts)
        radial_reference = radial_mean(reference_power, index, valid, radial_counts)
        transfer = normalized_transfer(radial_power, radial_reference, frequency_um)
        coherence_2d = np.divide(
            np.abs(mean_cross[scanner_index]) ** 2,
            mean_power[scanner_index] * reference_power,
            out=np.zeros_like(reference_power),
            where=(mean_power[scanner_index] * reference_power) > 1e-20,
        )
        coherence = np.clip(radial_mean(coherence_2d, index, valid, radial_counts), 0.0, 1.0)
        for bin_index in range(SPECTRAL_BINS):
            spectra_rows.append(
                {
                    "slide_id": slide_id,
                    "tissue_type": str(row["tissue_type"]),
                    "scanner": scanner,
                    "frequency_cyc_per_pixel": float(frequency_px[bin_index]),
                    "frequency_cyc_per_um_provisional": float(frequency_um[bin_index]),
                    "relative_transfer": float(transfer[bin_index]),
                    "log2_relative_transfer": float(np.log2(max(transfer[bin_index], 1e-20))),
                    "coherence_to_at2": float(coherence[bin_index]),
                    "radial_power": float(radial_power[bin_index]),
                }
            )
        for band, bounds in BANDS_CYC_PER_UM.items():
            value = geometric_band(transfer, frequency_um, bounds)
            band_rows.append(
                {
                    "slide_id": slide_id,
                    "tissue_type": str(row["tissue_type"]),
                    "scanner": scanner,
                    "band": band,
                    "lower_frequency_cyc_per_um_provisional": bounds[0],
                    "upper_frequency_cyc_per_um_provisional": bounds[1],
                    "relative_transfer": value,
                    "log2_relative_transfer": float(np.log2(value)),
                }
            )

    for replicate in range(REPLICATE_GROUPS):
        count = int(replicate_counts[replicate])
        replicate_power = power_sum[replicate] / float(count)
        radial_reference = radial_mean(replicate_power[0], index, valid, radial_counts)
        for scanner_index, scanner in enumerate(SCANNERS):
            radial_power = radial_mean(replicate_power[scanner_index], index, valid, radial_counts)
            transfer = normalized_transfer(radial_power, radial_reference, frequency_um)
            for band, bounds in BANDS_CYC_PER_UM.items():
                value = geometric_band(transfer, frequency_um, bounds)
                replicate_band_rows.append(
                    {
                        "slide_id": slide_id,
                        "tissue_type": str(row["tissue_type"]),
                        "replicate": replicate,
                        "patches_in_replicate": count,
                        "scanner": scanner,
                        "band": band,
                        "lower_frequency_cyc_per_um_provisional": bounds[0],
                        "upper_frequency_cyc_per_um_provisional": bounds[1],
                        "relative_transfer": value,
                        "log2_relative_transfer": float(np.log2(value)),
                    }
                )
    write_frame_atomic(pd.DataFrame(spectra_rows), spectrum_output / "spectra.csv")
    write_frame_atomic(pd.DataFrame(band_rows), spectrum_output / "band_summary.csv")
    write_frame_atomic(
        pd.DataFrame(replicate_band_rows), spectrum_output / "replicate_band_summary.csv"
    )

    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "pass",
        "completed_utc": utc_now(),
        "slide_id": slide_id,
        "tissue_type": str(row["tissue_type"]),
        "locations": n_locations,
        "replicate_counts": replicate_counts.tolist(),
        "scanners": list(SCANNERS),
        "cohort_sha256": cohort_hash,
        "source_cache": str(source),
        "source_cache_bytes": source.stat().st_size,
        "location_metrics_sha256": sha256(location_path),
        "physical_frequency_status": "provisional",
        "provisional_mpp": PROVISIONAL_MPP,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


def make_blocks(frame: pd.DataFrame, value_column: str) -> tuple[list[dict], int]:
    blocks = []
    replicate_counts = frame.groupby("slide_id")["replicate"].nunique()
    if replicate_counts.nunique() != 1:
        raise ValueError("replicate counts are not balanced across slides")
    n_replicates = int(replicate_counts.iloc[0])
    for tissue, part in frame.groupby("tissue_type", sort=True):
        part = part.sort_values(["slide_id", "replicate"])
        slide_ids = part["slide_id"].drop_duplicates().tolist()
        y = part[value_column].to_numpy(dtype=float)
        labels = part["slide_id"].astype(str).to_numpy()
        design = np.column_stack([(labels == slide).astype(float) for slide in slide_ids])
        if not np.all(design.sum(axis=0) == n_replicates):
            raise ValueError(f"unbalanced tissue block: {tissue}")
        blocks.append(
            {
                "tissue_type": str(tissue),
                "slide_ids": slide_ids,
                "y": y,
                "ones": np.ones(len(y)),
                "slide_design": design,
                "slide_kernel": design @ design.T,
                "tissue_kernel": np.ones((len(y), len(y))),
                "identity": np.eye(len(y)),
            }
        )
    return blocks, n_replicates


def initial_variances(frame: pd.DataFrame, value_column: str, n_replicates: int) -> np.ndarray:
    slide_means = frame.groupby(["tissue_type", "slide_id"])[value_column].mean()
    repeated = frame.set_index(["tissue_type", "slide_id"]).index.map(slide_means)
    residual = float(
        np.square(frame[value_column].to_numpy() - repeated.to_numpy()).sum()
        / max(len(frame) - len(slide_means), 1)
    )
    tissue_means = slide_means.groupby(level="tissue_type").mean()
    within = slide_means - slide_means.index.get_level_values("tissue_type").map(tissue_means)
    slide_variance = max(float(np.var(within, ddof=1)) - residual / n_replicates, 1e-8)
    mean_slides = float(slide_means.groupby(level="tissue_type").size().mean())
    tissue_variance = max(
        float(np.var(tissue_means, ddof=1))
        - slide_variance / mean_slides
        - residual / (n_replicates * mean_slides),
        1e-8,
    )
    return np.asarray([tissue_variance, slide_variance, max(residual, 1e-8)])


def evaluate_reml(
    blocks: list[dict], variances: np.ndarray, include_tissue: bool = True, details: bool = False
) -> dict:
    tissue_variance, slide_variance, residual_variance = variances
    total_n = 0
    log_determinant = one_v_one = one_v_y = y_v_y = 0.0
    fitted_details = []
    for block in blocks:
        covariance = slide_variance * block["slide_kernel"] + residual_variance * block["identity"]
        if include_tissue:
            covariance = covariance + tissue_variance * block["tissue_kernel"]
        factor = linalg.cho_factor(covariance, lower=True, check_finite=False)
        inverse_one = linalg.cho_solve(factor, block["ones"], check_finite=False)
        inverse_y = linalg.cho_solve(factor, block["y"], check_finite=False)
        total_n += len(block["y"])
        log_determinant += float(2.0 * np.log(np.diag(factor[0])).sum())
        one_v_one += float(block["ones"] @ inverse_one)
        one_v_y += float(block["ones"] @ inverse_y)
        y_v_y += float(block["y"] @ inverse_y)
        if details:
            fitted_details.append((block, factor))
    fixed_mean = one_v_y / one_v_one
    negative_log_likelihood = 0.5 * (
        log_determinant
        + np.log(one_v_one)
        + y_v_y
        - one_v_y**2 / one_v_one
        + (total_n - 1) * np.log(2.0 * np.pi)
    )
    result = {
        "negative_log_likelihood": float(negative_log_likelihood),
        "fixed_mean": float(fixed_mean),
        "fixed_se": float(np.sqrt(1.0 / one_v_one)),
        "total_n": total_n,
    }
    if details:
        result["details"] = fitted_details
    return result


def fit_nested_reml(frame: pd.DataFrame, value_column: str = "value") -> dict:
    blocks, n_replicates = make_blocks(frame, value_column)
    start = initial_variances(frame, value_column, n_replicates)
    bounds = [(-22.0, 10.0)] * 3

    def objective(log_variances: np.ndarray) -> float:
        return evaluate_reml(blocks, np.exp(log_variances))["negative_log_likelihood"]

    starts = [
        np.log(start),
        np.log(np.maximum(start * np.asarray([0.25, 2.0, 1.0]), 1e-10)),
        np.log(np.maximum(start * np.asarray([2.0, 0.25, 1.0]), 1e-10)),
    ]
    fits = [
        optimize.minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-8},
        )
        for initial in starts
    ]
    fit = min([item for item in fits if item.success] or fits, key=lambda item: item.fun)
    variances = np.exp(fit.x)
    full = evaluate_reml(blocks, variances, details=True)

    def reduced_objective(log_variances: np.ndarray) -> float:
        values = np.asarray([0.0, np.exp(log_variances[0]), np.exp(log_variances[1])])
        return evaluate_reml(blocks, values, include_tissue=False)["negative_log_likelihood"]

    reduced_fits = [
        optimize.minimize(
            reduced_objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds[1:],
            options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-8},
        )
        for initial in (
            np.log(np.maximum(variances[1:], 1e-10)),
            np.log(np.maximum(variances[1:] * np.asarray([0.25, 2.0]), 1e-10)),
            np.log(np.maximum(variances[1:] * np.asarray([2.0, 0.25]), 1e-10)),
        )
    ]
    reduced = min(
        [item for item in reduced_fits if item.success] or reduced_fits,
        key=lambda item: item.fun,
    )
    likelihood_ratio = max(0.0, 2.0 * (reduced.fun - fit.fun))
    tissue_p = 0.5 * float(stats.chi2.sf(likelihood_ratio, 1)) if likelihood_ratio > 0 else 1.0

    tissue_rows = []
    slide_rows = []
    for block, factor in full["details"]:
        residual = block["y"] - full["fixed_mean"]
        alpha = linalg.cho_solve(factor, residual, check_finite=False)
        inverse_one = linalg.cho_solve(factor, block["ones"], check_finite=False)
        tissue_blup = variances[0] * float(block["ones"] @ alpha)
        tissue_var_conditional = max(
            variances[0] - variances[0] ** 2 * float(block["ones"] @ inverse_one), 0.0
        )
        tissue_rows.append(
            {
                "tissue_type": block["tissue_type"],
                "slides_in_tissue": len(block["slide_ids"]),
                "raw_tissue_deviation": float(block["y"].mean() - full["fixed_mean"]),
                "blup_tissue_deviation": tissue_blup,
                "blup_se_conditional": float(np.sqrt(tissue_var_conditional)),
            }
        )
        for column, slide_id in enumerate(block["slide_ids"]):
            indicator = block["slide_design"][:, column]
            inverse_indicator = linalg.cho_solve(factor, indicator, check_finite=False)
            slide_blup = variances[1] * float(indicator @ alpha)
            slide_var_conditional = max(
                variances[1]
                - variances[1] ** 2 * float(indicator @ inverse_indicator),
                0.0,
            )
            slide_rows.append(
                {
                    "tissue_type": block["tissue_type"],
                    "slide_id": str(slide_id),
                    "blup_slide_within_tissue_deviation": slide_blup,
                    "blup_se_conditional": float(np.sqrt(slide_var_conditional)),
                }
            )
    return {
        "variances": variances,
        "full": full,
        "fit": fit,
        "reduced_fit": reduced,
        "likelihood_ratio": likelihood_ratio,
        "tissue_variance_p": tissue_p,
        "n_replicates": n_replicates,
        "n_tissues": len(blocks),
        "tissue_rows": tissue_rows,
        "slide_rows": slide_rows,
    }


def benjamini_hochberg(values: pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array)
    ranked = array[order]
    adjusted = ranked * len(array) / np.arange(1, len(array) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def build_scalar_contrasts(scalars: pd.DataFrame) -> pd.DataFrame:
    keys = ["slide_id", "tissue_type", "replicate", "patches_in_replicate"]
    reference = scalars[scalars["scanner"] == REFERENCE].set_index(
        ["slide_id", "replicate"]
    )
    rows = []
    for row in scalars[scalars["scanner"] != REFERENCE].itertuples(index=False):
        anchor = reference.loc[(str(row.slide_id), int(row.replicate))]
        for endpoint, specification in MODEL_ENDPOINTS.items():
            scanner_value = float(getattr(row, specification["source"]))
            reference_value = float(anchor[specification["source"]])
            if specification["transform"] == "difference":
                value = scanner_value - reference_value
            elif specification["transform"] == "log2_ratio":
                value = math.log2(max(scanner_value, 1e-12) / max(reference_value, 1e-12))
            elif specification["transform"] == "identity":
                value = scanner_value
            elif specification["transform"] == "one_minus":
                value = 1.0 - scanner_value
            else:
                raise ValueError(specification["transform"])
            rows.append(
                {
                    "slide_id": str(row.slide_id),
                    "tissue_type": str(row.tissue_type),
                    "replicate": int(row.replicate),
                    "patches_in_replicate": int(row.patches_in_replicate),
                    "scanner": str(row.scanner),
                    "endpoint": endpoint,
                    "endpoint_label": specification["label"],
                    "family": specification["family"],
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def fit_all_models(contrasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    components = []
    tissue_rows = []
    slide_rows = []
    for endpoint in contrasts["endpoint"].drop_duplicates():
        for scanner in SCANNERS[1:]:
            frame = contrasts[
                (contrasts["endpoint"] == endpoint) & (contrasts["scanner"] == scanner)
            ].copy()
            result = fit_nested_reml(frame)
            tissue_variance, slide_variance, residual_variance = result["variances"]
            sampling_variance = residual_variance / result["n_replicates"]
            total = tissue_variance + slide_variance + sampling_variance
            degrees = result["n_tissues"] - 1
            critical = float(stats.t.ppf(0.975, degrees))
            mean = result["full"]["fixed_mean"]
            se = result["full"]["fixed_se"]
            t_value = mean / se
            descriptor = frame.iloc[0]
            components.append(
                {
                    "endpoint": endpoint,
                    "endpoint_label": descriptor["endpoint_label"],
                    "family": descriptor["family"],
                    "scanner": scanner,
                    "n_tissues": result["n_tissues"],
                    "n_slides": frame["slide_id"].nunique(),
                    "replicate_groups": result["n_replicates"],
                    "fixed_mean": mean,
                    "fixed_se": se,
                    "fixed_ci95_low": mean - critical * se,
                    "fixed_ci95_high": mean + critical * se,
                    "fixed_t": t_value,
                    "fixed_p": float(2.0 * stats.t.sf(abs(t_value), degrees)),
                    "scanner_by_tissue_variance": tissue_variance,
                    "scanner_by_tissue_sd": float(np.sqrt(tissue_variance)),
                    "scanner_by_slide_within_tissue_variance": slide_variance,
                    "scanner_by_slide_within_tissue_sd": float(np.sqrt(slide_variance)),
                    "replicate_sampling_variance": residual_variance,
                    "sampling_variance_slide_mean": sampling_variance,
                    "tissue_fraction_total_slide_mean_variance": tissue_variance / total,
                    "slide_fraction_total_slide_mean_variance": slide_variance / total,
                    "sampling_fraction_total_slide_mean_variance": sampling_variance / total,
                    "tissue_fraction_between_slide_variance": tissue_variance
                    / max(tissue_variance + slide_variance, 1e-20),
                    "tissue_variance_lrt": result["likelihood_ratio"],
                    "tissue_variance_p_mixture": result["tissue_variance_p"],
                    "optimizer_converged": bool(result["fit"].success),
                    "optimizer_message": str(result["fit"].message),
                    "reml_negative_log_likelihood": float(result["fit"].fun),
                }
            )
            for item in result["tissue_rows"]:
                tissue_rows.append({"endpoint": endpoint, "scanner": scanner, **item})
            for item in result["slide_rows"]:
                slide_rows.append({"endpoint": endpoint, "scanner": scanner, **item})
            print(f"[REML] {endpoint}/{scanner}", flush=True)
    component_frame = pd.DataFrame(components)
    component_frame["fixed_q_bh"] = benjamini_hochberg(component_frame["fixed_p"])
    component_frame["tissue_variance_q_bh"] = benjamini_hochberg(
        component_frame["tissue_variance_p_mixture"]
    )
    return component_frame, pd.DataFrame(tissue_rows), pd.DataFrame(slide_rows)


def figure_scanner_profiles(components: pd.DataFrame, output: Path) -> None:
    scanners = list(SCANNERS[1:])
    colors = dict(zip(scanners, ("#175d72", "#8c5e24", "#7c3aed", "#2f725f", "#a63d40")))
    panels = [
        ("Color", ["delta_lab_l", "delta_lab_a", "delta_lab_b"]),
        ("Contrast / structure", ["log2_lab_l_sd_ratio", "log2_od_sd_ratio", "log2_gradient_rms_ratio"]),
        ("Spatial frequency", ["frequency_low_mid", "frequency_mid", "frequency_high"]),
        ("Paired structure", ["gradient_dissimilarity"]),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for axis, (title, endpoints) in zip(axes.ravel(), panels):
        subset = components[components["endpoint"].isin(endpoints)].copy()
        for scanner_index, scanner in enumerate(scanners):
            scanner_rows = subset[subset["scanner"] == scanner].set_index("endpoint").reindex(endpoints)
            x = np.arange(len(endpoints)) + (scanner_index - 2) * 0.11
            means = scanner_rows["fixed_mean"].to_numpy()
            lower = means - scanner_rows["fixed_ci95_low"].to_numpy()
            upper = scanner_rows["fixed_ci95_high"].to_numpy() - means
            axis.errorbar(
                x,
                means,
                yerr=np.vstack([lower, upper]),
                fmt="o",
                color=colors[scanner],
                label=scanner.upper(),
                capsize=3,
                ms=5,
            )
        axis.axhline(0.0, color="#5a6b70", lw=1, ls="--")
        labels = [
            str(subset[subset["endpoint"] == endpoint]["endpoint_label"].iloc[0])
            for endpoint in endpoints
        ]
        axis.set_xticks(range(len(endpoints)), labels, rotation=18, ha="right")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, ncol=3)
    figure.suptitle(
        "Paired image-level scanner effects (nested-REML fixed means, 95% CI)",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def figure_variance(components: pd.DataFrame, output: Path) -> None:
    endpoints = [
        "delta_lab_l",
        "delta_lab_a",
        "delta_lab_b",
        "log2_od_sd_ratio",
        "log2_gradient_rms_ratio",
        "gradient_dissimilarity",
        "frequency_low_mid",
        "frequency_mid",
        "frequency_high",
    ]
    matrix = (
        components[components["endpoint"].isin(endpoints)]
        .pivot(index="endpoint", columns="scanner", values="tissue_fraction_between_slide_variance")
        .reindex(index=endpoints, columns=SCANNERS[1:])
    )
    qvalues = (
        components[components["endpoint"].isin(endpoints)]
        .pivot(index="endpoint", columns="scanner", values="tissue_variance_q_bh")
        .reindex(index=endpoints, columns=SCANNERS[1:])
    )
    figure, axis = plt.subplots(figsize=(10.5, 7.5), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Purples", vmin=0, vmax=1, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            marker = "*" if qvalues.iloc[row, column] < 0.05 else ""
            axis.text(
                column,
                row,
                f"{100 * value:.0f}%{marker}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "#16252b",
                fontsize=9,
            )
    labels = [
        components.loc[components["endpoint"] == endpoint, "endpoint_label"].iloc[0]
        for endpoint in endpoints
    ]
    axis.set_xticks(range(len(SCANNERS) - 1), [item.upper() for item in SCANNERS[1:]])
    axis.set_yticks(range(len(endpoints)), labels)
    axis.set_title(
        "Tissue share of scanner-effect heterogeneity\n* variance-component LRT, BH q < 0.05",
        loc="left",
        fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.04)
    colorbar.set_label("Tissue fraction of tissue + slide variance")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def figure_tissue_frequency(tissue_slopes: pd.DataFrame, output: Path) -> None:
    subset = tissue_slopes[tissue_slopes["endpoint"] == "frequency_high"]
    matrix = subset.pivot(
        index="scanner", columns="tissue_type", values="blup_tissue_deviation"
    ).reindex(index=SCANNERS[1:])
    if matrix.shape[1] > 1:
        order = leaves_list(linkage(matrix.to_numpy().T, method="average"))
        matrix = matrix.iloc[:, order]
    limit = max(0.05, float(np.nanquantile(np.abs(matrix.to_numpy()), 0.98)))
    figure, axis = plt.subplots(figsize=(16, 5.5), constrained_layout=True)
    image = axis.imshow(
        matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest"
    )
    axis.set_yticks(range(len(SCANNERS) - 1), [item.upper() for item in SCANNERS[1:]])
    axis.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=70, ha="right", fontsize=7)
    axis.set_title(
        "High-frequency scanner × tissue deviations (REML BLUP)", loc="left", fontweight="bold"
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025)
    colorbar.set_label("Deviation from scanner mean (log₂ transfer)")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def figure_pairing_audit(cohort: pd.DataFrame, output: Path) -> None:
    ordered = cohort.sort_values("pair_count").reset_index(drop=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    colors = np.where(ordered["cap_reached"], "#175d72", "#b57627")
    axes[0].bar(np.arange(len(ordered)), ordered["pair_count"], color=colors, width=1.0)
    axes[0].axhline(1000, color="#16252b", ls="--", lw=1)
    axes[0].set(xlabel="Slides ordered by accepted pair count", ylabel="Accepted paired locations")
    axes[0].set_title("A  Retained paired sample per slide", loc="left", fontweight="bold")
    totals = [
        cohort["source_candidate_count"].sum(),
        cohort["processed_candidate_count"].sum(),
        cohort["pair_count"].sum(),
    ]
    axes[1].bar(["Source candidates", "Processed", "Accepted"], totals, color=["#d0d5d3", "#7aa4af", "#175d72"])
    for index, value in enumerate(totals):
        axes[1].text(index, value, f"{int(value):,}", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("Locations")
    axes[1].set_title("B  Selection funnel", loc="left", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=18)
    figure.suptitle("Pairing and selection audit", fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def aggregate(output_root: Path) -> None:
    cohort = pd.read_csv(output_root / "00_contract/cohort.csv", dtype={"slide_id": str})
    summaries = []
    scalar_frames = []
    spectrum_frames = []
    band_frames = []
    replicate_band_frames = []
    for slide_id in cohort["slide_id"]:
        phenotype = output_root / "02_image_phenotypes/shards" / slide_id
        frequency = output_root / "03_frequency/shards" / slide_id
        summary_path = phenotype / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "pass" or summary.get("analysis_version") != ANALYSIS_VERSION:
            raise ValueError(f"invalid shard summary: {summary_path}")
        summaries.append(summary)
        scalar_frames.append(pd.read_csv(phenotype / "replicate_scalar_summary.csv", dtype={"slide_id": str}))
        spectrum_frames.append(pd.read_csv(frequency / "spectra.csv", dtype={"slide_id": str}))
        band_frames.append(pd.read_csv(frequency / "band_summary.csv", dtype={"slide_id": str}))
        replicate_band_frames.append(
            pd.read_csv(frequency / "replicate_band_summary.csv", dtype={"slide_id": str})
        )
    scalars = pd.concat(scalar_frames, ignore_index=True)
    spectra = pd.concat(spectrum_frames, ignore_index=True)
    bands = pd.concat(band_frames, ignore_index=True)
    replicate_bands = pd.concat(replicate_band_frames, ignore_index=True)
    expected_scalar = len(cohort) * len(SCANNERS) * REPLICATE_GROUPS
    expected_spectrum = len(cohort) * len(SCANNERS) * SPECTRAL_BINS
    expected_bands = len(cohort) * len(SCANNERS) * len(BANDS_CYC_PER_UM)
    if len(scalars) != expected_scalar or len(spectra) != expected_spectrum:
        raise ValueError("incomplete scalar or spectrum aggregation")
    if len(bands) != expected_bands or len(replicate_bands) != expected_bands * REPLICATE_GROUPS:
        raise ValueError("incomplete spectral band aggregation")

    write_frame_atomic(scalars, output_root / "02_image_phenotypes/replicate_scalar_summary.csv")
    write_frame_atomic(spectra, output_root / "03_frequency/spectra.csv")
    write_frame_atomic(bands, output_root / "03_frequency/band_summary.csv")
    write_frame_atomic(replicate_bands, output_root / "03_frequency/replicate_band_summary.csv")

    scalar_contrasts = build_scalar_contrasts(scalars)
    frequency_contrasts = replicate_bands[replicate_bands["scanner"] != REFERENCE].copy()
    frequency_contrasts["endpoint"] = "frequency_" + frequency_contrasts["band"].astype(str)
    frequency_contrasts["endpoint_label"] = frequency_contrasts["band"].map(
        {
            "low_mid": "Low–mid frequency transfer",
            "mid": "Mid-frequency transfer",
            "high": "High-frequency transfer",
        }
    )
    frequency_contrasts["family"] = "spatial frequency"
    frequency_contrasts["value"] = frequency_contrasts["log2_relative_transfer"]
    contrast_columns = [
        "slide_id",
        "tissue_type",
        "replicate",
        "patches_in_replicate",
        "scanner",
        "endpoint",
        "endpoint_label",
        "family",
        "value",
    ]
    contrasts = pd.concat(
        [scalar_contrasts[contrast_columns], frequency_contrasts[contrast_columns]],
        ignore_index=True,
    )
    if not np.isfinite(contrasts["value"]).all():
        raise ValueError("non-finite paired contrast")
    write_frame_atomic(contrasts, output_root / "04_lmm/replicate_contrasts.csv")

    components, tissue_slopes, slide_slopes = fit_all_models(contrasts)
    write_frame_atomic(components, output_root / "04_lmm/model_summary.csv")
    write_frame_atomic(tissue_slopes, output_root / "04_lmm/tissue_blups.csv")
    write_frame_atomic(slide_slopes, output_root / "04_lmm/slide_blups.csv")

    figure_pairing_audit(cohort, output_root / "90_report_assets/figure01_pairing_audit.png")
    figure_scanner_profiles(components, output_root / "90_report_assets/figure02_scanner_profiles.png")
    figure_variance(components, output_root / "90_report_assets/figure03_variance_decomposition.png")
    figure_tissue_frequency(tissue_slopes, output_root / "90_report_assets/figure04_tissue_high_frequency.png")

    fixed_significant = components[components["fixed_q_bh"] < 0.05]
    tissue_significant = components[components["tissue_variance_q_bh"] < 0.05]
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "pass",
        "completed_utc": utc_now(),
        "slides": len(cohort),
        "tissues": int(cohort["tissue_type"].nunique()),
        "paired_locations": int(cohort["pair_count"].sum()),
        "scanner_endpoint_models": len(components),
        "optimizer_converged_models": int(components["optimizer_converged"].sum()),
        "fixed_effects_bh_q_lt_0p05": int(len(fixed_significant)),
        "tissue_variance_components_bh_q_lt_0p05": int(len(tissue_significant)),
        "physical_frequency_status": "provisional",
        "primary_outputs": {
            "model_summary": "04_lmm/model_summary.csv",
            "tissue_blups": "04_lmm/tissue_blups.csv",
            "scanner_profiles": "90_report_assets/figure02_scanner_profiles.png",
            "variance_decomposition": "90_report_assets/figure03_variance_decomposition.png",
        },
        "claim_boundary": (
            "Results identify registered acquisition-pipeline differences. Physical-frequency "
            "band labels are provisional pending an explicit effective-MPP field in the cache."
        ),
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--tissue-source", type=Path, default=TISSUE_SOURCE)
    plism_parser = subparsers.add_parser("audit-plism")
    plism_parser.add_argument(
        "--plism-root", type=Path, default=ROOT / "data/PLISM_dataset"
    )
    extract_parser = subparsers.add_parser("extract-slide")
    extract_parser.add_argument("--slide-index", type=int, required=True)
    extract_parser.add_argument("--chunk-size", type=int, default=8)
    extract_parser.add_argument("--workers", type=int, default=4)
    subparsers.add_parser("aggregate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.data_root.resolve(), args.output_root.resolve(), args.tissue_source.resolve())
    elif args.command == "audit-plism":
        audit_plism(args.plism_root, args.output_root.resolve())
    elif args.command == "extract-slide":
        extract_slide(args.slide_index, args.output_root.resolve(), args.chunk_size, args.workers)
    elif args.command == "aggregate":
        aggregate(args.output_root.resolve())
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
