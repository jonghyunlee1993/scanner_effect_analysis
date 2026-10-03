#!/usr/bin/env python3
"""PLISM factorial external stress test for the final image-level study.

The external panel is represented as a 2 x 2 design relative to PanNormal:

* shared scanner model / shared tissue (ID-like; not acquisition-distribution ID),
* novel scanner model / shared tissue,
* shared scanner model / novel tissue, and
* novel scanner model / novel tissue.

PLISM is fully crossed: 46 TMA cores on 13 serial sections were acquired by all
seven scanners.  This program uses AT2 as the within-section, within-location
reference, measures ten image-space endpoints on a common 0.5052 um/px render,
and treats section as the uncertainty unit.  Frozen UNI-v2 embeddings are used
only as a secondary bridge.

Stages
------
prepare
    Freeze the tissue ontology, joint alignment gate, matched locations, and 91
    WSI extraction tasks.
extract-task
    Read one native WSI, render the frozen physical fields of view to the common
    grid, and emit absolute image measurements.  This is the SLURM-array unit.
aggregate
    Form paired AT2 contrasts, estimate the four quadrants and their additive
    interaction, run ontology/focus sensitivities, analyse UNI-v2, and render
    report-facing figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fft


ROOT = Path(__file__).resolve().parent.parent
PLISM_ROOT = ROOT / "data/PLISM_dataset"
OUTPUT_ROOT = ROOT / "outputs/plism_factorial_external_v1"
PAN_AUGMENTATION_ROOT = ROOT / "outputs/augmentation_ood_v1"
ANALYSIS_VERSION = "plism_factorial_external_v1"

REFERENCE = "AT2"
SHARED_TARGET_SCANNERS = ("GT450", "S360", "S60")
NOVEL_TARGET_SCANNERS = ("P", "S210", "SQ")
SCANNERS = (REFERENCE, *SHARED_TARGET_SCANNERS, *NOVEL_TARGET_SCANNERS)

TARGET_MPP = 0.5052
TARGET_PX = 256
PATCH_UM = TARGET_MPP * TARGET_PX
SPECTRAL_BINS = 72
LOCATIONS_PER_CORE_SECTION = 32
ALIGNMENT_RESIDUAL_MAX_UM = 1.0
ALIGNMENT_RESPONSE_MIN = 0.3
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260917

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

BANDS_CYC_PER_UM = {
    "anchor": (0.03, 0.10),
    "low_mid": (0.10, 0.30),
    "mid": (0.30, 0.60),
    "high": (0.60, 0.90),
}

QUADRANT_ORDER = (
    "shared/shared",
    "scanner-OOD",
    "tissue-OOD",
    "double-OOD",
)

# PLISM label -> (PanNormal canonical tissue or blank, mapping tier).
# Tumours are deliberately not collapsed into the corresponding normal organ.
TISSUE_CROSSWALK = {
    "01_penis": ("", "novel"),
    "02_testicle": ("testis", "synonym"),
    "03_ileum": ("small intestine", "substructure"),
    "04_right_colon": ("large intestine (colon)", "substructure"),
    "05_thymus": ("thymus", "exact"),
    "06_adrenal_gland": ("adrenal", "synonym"),
    "07_renal_cortex": ("kidney(cortex/medulla)", "substructure"),
    "08_renal_medulla": ("kidney(cortex/medulla)", "substructure"),
    "09_follicle": ("ovary", "substructure"),
    "10_corpus_luteum": ("ovary", "substructure"),
    "11_uterine_cervix": ("cervix, uterus", "substructure"),
    "12_lymph_node": ("lymph node", "exact"),
    "13_major_salivary_gland": ("salivary gland", "synonym"),
    "14_pancreas": ("pancreas", "exact"),
    "15_spleen": ("spleen", "exact"),
    "16_urinary_bladder": ("urinary system", "substructure"),
    "17_prostate": ("prostate", "exact"),
    "18_liver": ("liver", "exact"),
    "19_jejunum": ("small intestine", "substructure"),
    "20_broncus": ("lung", "substructure"),
    "21_cartilage": ("", "novel"),
    "22_uterine_corpus": ("cervix, uterus", "substructure"),
    "23_left_ventricle": ("heart", "substructure"),
    "24_pituitary_gland": ("pituitary", "synonym"),
    "25_cerebral_cortex": ("brain, cortex", "substructure"),
    "26_thyroid": ("thyroid gland", "synonym"),
    "27_skin": ("skin", "exact"),
    "28_placenta": ("", "novel"),
    "29_lung": ("lung", "exact"),
    "30_myxofibroma": ("", "novel"),
    "31_NEC": ("", "novel"),
    "32_ESCC": ("", "novel"),
    "33_RCCC": ("", "novel"),
    "34_colon_adenocarcionma": ("", "novel"),
    "35_thymoma": ("", "novel"),
    "36_GIST": ("", "novel"),
    "37_metastatic_colorectal_cancer": ("", "novel"),
    "38_EBVGC": ("", "novel"),
    "39_HCC": ("", "novel"),
    "40_collagenous_fiber": ("", "novel"),
    "41_dedifferentiated_liposarcoma": ("", "novel"),
    "42_esophagus": ("esophagus", "exact"),
    "43_stomach": ("stomach", "exact"),
    "44_gallbladder": ("gall bladder", "synonym"),
    "45_skeletal_muscle": ("skeletal muscle", "exact"),
    "46_aorta": ("aorta", "exact"),
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


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def crosswalk_frame() -> pd.DataFrame:
    rows = []
    for tissue, (canonical, tier) in TISSUE_CROSSWALK.items():
        rows.append(
            {
                "plism_tissue": tissue,
                "pannormal_tissue": canonical,
                "mapping_tier": tier,
                "organ_aligned_status": "shared" if tier != "novel" else "novel",
                "strict_status": (
                    "shared" if tier in {"exact", "synonym"} else "novel"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("plism_tissue").reset_index(drop=True)


def _select_locations(block: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Deterministically spread selections over the frozen spatial replicates."""

    if len(block) <= limit:
        return block.sort_values("location")
    chosen = []
    per_replicate = int(math.ceil(limit / 10))
    for _, part in block.groupby("replicate", sort=True):
        part = part.sort_values("location")
        take = min(per_replicate, len(part))
        positions = np.linspace(0, len(part) - 1, take).round().astype(int)
        chosen.append(part.iloc[np.unique(positions)])
    selected = pd.concat(chosen).drop_duplicates("location")
    if len(selected) > limit:
        selected = selected.assign(
            _key=selected["location"].map(
                lambda value: hashlib.sha256(str(int(value)).encode()).hexdigest()
            )
        ).sort_values("_key").head(limit).drop(columns="_key")
    if len(selected) < limit:
        missing = block.loc[~block["location"].isin(selected["location"])].copy()
        missing = missing.assign(
            _key=missing["location"].map(
                lambda value: hashlib.sha256(str(int(value)).encode()).hexdigest()
            )
        ).sort_values("_key").head(limit - len(selected)).drop(columns="_key")
        selected = pd.concat([selected, missing], ignore_index=True)
    return selected.sort_values("location")


def prepare(plism_root: Path, output_root: Path, locations_per_core: int) -> None:
    for relative in (
        "00_contract/task_plans",
        "01_image_metrics/shards",
        "02_factorial",
        "03_uni_bridge",
        "04_figures",
        "logs",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)

    manifest_path = plism_root / "manifest_original.csv"
    alignment_path = plism_root / "alignment/per_location.csv.gz"
    corrections_path = plism_root / "alignment/file_corrections.csv"
    core_map_path = plism_root / "alignment/core_map.csv"
    manifest = pd.read_csv(manifest_path)
    alignment = pd.read_csv(alignment_path)
    alignment["ok"] = alignment["ok"].astype(str).str.lower().eq("true")
    alignment["passes_joint_gate"] = (
        alignment["ok"]
        & (alignment["residual_um"] <= ALIGNMENT_RESIDUAL_MAX_UM)
        & (alignment["response"] >= ALIGNMENT_RESPONSE_MIN)
    )
    scanners_per_location = alignment.groupby(["stain", "location"])[
        "scanner"
    ].transform("nunique")
    joint = alignment.groupby(["stain", "location"])["passes_joint_gate"].transform(
        "all"
    )
    eligible = alignment.loc[
        (scanners_per_location == len(SCANNERS)) & joint
    ].copy()
    reference = eligible.loc[eligible["scanner"] == REFERENCE].copy()

    observed_tissues = set(reference["tissue_type"].astype(str))
    missing_crosswalk = sorted(observed_tissues - set(TISSUE_CROSSWALK))
    if missing_crosswalk:
        raise ValueError(f"tissue crosswalk is incomplete: {missing_crosswalk}")
    selected_parts = []
    for (_, _), block in reference.groupby(["stain", "core"], sort=True):
        selected_parts.append(_select_locations(block, locations_per_core))
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected[
        ["stain", "location", "core", "tissue_type", "replicate"]
    ].sort_values(["stain", "core", "location"])
    selected = selected.merge(
        crosswalk_frame(), left_on="tissue_type", right_on="plism_tissue", how="left",
        validate="many_to_one",
    ).drop(columns="plism_tissue")

    crop_plan = eligible.merge(
        selected[["stain", "location"]],
        on=["stain", "location"],
        how="inner",
        validate="many_to_one",
    )
    expected = len(selected) * len(SCANNERS)
    if len(crop_plan) != expected:
        raise ValueError(f"crop plan is not fully crossed: {len(crop_plan)} != {expected}")

    corrections = pd.read_csv(corrections_path)
    correction_map = {
        (str(row.stain), str(row.scanner)): str(row.actual_file)
        for row in corrections.itertuples(index=False)
    }
    task_rows = []
    for row in manifest.sort_values("task_index").itertuples(index=False):
        stain, scanner = str(row.stain), str(row.scanner)
        task_index = int(row.task_index)
        plan = crop_plan.loc[
            (crop_plan["stain"] == stain) & (crop_plan["scanner"] == scanner)
        ].sort_values("location")
        if plan.empty:
            raise ValueError(f"empty crop plan for {stain}/{scanner}")
        plan_path = output_root / f"00_contract/task_plans/{task_index:03d}.csv"
        write_frame(plan, plan_path)
        task_rows.append(
            {
                "task_index": task_index,
                "stain": stain,
                "scanner": scanner,
                "published_file": str(row.name),
                "actual_file": correction_map.get((stain, scanner), str(row.name)),
                "file_corrected": (stain, scanner) in correction_map,
                "native_mpp_manifest": float(row.native_mpp),
                "locations": len(plan),
                "plan_path": str(plan_path.resolve()),
                "plan_sha256": sha256(plan_path),
            }
        )
    tasks = pd.DataFrame(task_rows)
    if len(tasks) != 91:
        raise ValueError(f"expected 91 WSI tasks, got {len(tasks)}")

    crosswalk_path = output_root / "00_contract/tissue_crosswalk.csv"
    selected_path = output_root / "00_contract/selected_locations.csv"
    tasks_path = output_root / "00_contract/tasks.csv"
    write_frame(crosswalk_frame(), crosswalk_path)
    write_frame(selected, selected_path)
    write_frame(tasks, tasks_path)
    contract = {
        "analysis_version": ANALYSIS_VERSION,
        "created_utc": utc_now(),
        "primary_question": (
            "How does the paired image phenotype change across shared/shared, "
            "scanner-OOD, tissue-OOD, and double-OOD PLISM strata?"
        ),
        "claim_boundary": (
            "Shared/shared is ID-like only with respect to scanner-model and organ-level "
            "tissue labels; every PLISM image remains external in device, site, section, "
            "stain, and acquisition context. Tissue cores are not independent patients."
        ),
        "design": "46 tissue cores x 13 serial sections x 7 scanners, fully crossed",
        "reference_scanner": REFERENCE,
        "shared_target_scanners": list(SHARED_TARGET_SCANNERS),
        "novel_target_scanners": list(NOVEL_TARGET_SCANNERS),
        "tissue_primary": "organ_aligned_status; tumours remain novel",
        "tissue_sensitivity": "strict_status; exact labels and direct synonyms only",
        "stain_role": (
            "nuisance section block; stain and section are confounded, so stability is "
            "assessed by leave-one-section-out rather than a causal stain coefficient"
        ),
        "inference_unit": "section (13 fixed external sections)",
        "primary_measurement_unit": "matched location, aggregated to section x scanner x core",
        "locations_per_core_section_max": locations_per_core,
        "selected_locations": int(len(selected)),
        "crop_tasks": len(tasks),
        "joint_alignment_gate": {
            "all_seven_scanners_required": True,
            "residual_um_max": ALIGNMENT_RESIDUAL_MAX_UM,
            "response_min": ALIGNMENT_RESPONSE_MIN,
        },
        "render": {
            "physical_fov_um": PATCH_UM,
            "target_px": TARGET_PX,
            "target_mpp": TARGET_MPP,
            "kernel": "libvips Lanczos3",
            "status": "matched-grid external stress test; internal effective MPP remains provisional",
        },
        "endpoints": list(ENDPOINTS),
        "endpoint_standardization": (
            "PanNormal target-vector IQR / 1.349 from augmentation_ood_v1"
        ),
        "focus_sensitivity": "repeat after excluding the known out-of-focus HRH/S60 block",
        "sources": {
            "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
            "alignment": {"path": str(alignment_path.resolve()), "sha256": sha256(alignment_path)},
            "corrections": {"path": str(corrections_path.resolve()), "sha256": sha256(corrections_path)},
            "core_map": {"path": str(core_map_path.resolve()), "sha256": sha256(core_map_path)},
            "pannormal_augmentation_contract": {
                "path": str((PAN_AUGMENTATION_ROOT / "00_contract/analysis_contract.json").resolve()),
                "sha256": sha256(PAN_AUGMENTATION_ROOT / "00_contract/analysis_contract.json"),
            },
        },
    }
    contract_path = output_root / "00_contract/analysis_contract.json"
    write_json(contract_path, contract)
    print(
        json.dumps(
            {
                "status": "prepared",
                "tasks": len(tasks),
                "selected_locations": len(selected),
                "organ_aligned_tissues": int(
                    crosswalk_frame().query("organ_aligned_status == 'shared'").shape[0]
                ),
                "novel_tissues": int(
                    crosswalk_frame().query("organ_aligned_status == 'novel'").shape[0]
                ),
                "contract": str(contract_path),
            },
            indent=2,
        )
    )


def srgb_to_lab(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(rgb, dtype=np.float32) / 255.0
    linear = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
    x = 0.4124564 * linear[..., 0] + 0.3575761 * linear[..., 1] + 0.1804375 * linear[..., 2]
    y = 0.2126729 * linear[..., 0] + 0.7151522 * linear[..., 1] + 0.0721750 * linear[..., 2]
    z = 0.0193339 * linear[..., 0] + 0.1191920 * linear[..., 1] + 0.9503041 * linear[..., 2]
    xyz = (x / 0.95047, y, z / 1.08883)

    def transform(component: np.ndarray) -> np.ndarray:
        delta = 6.0 / 29.0
        return np.where(
            component > delta**3,
            np.cbrt(component),
            component / (3.0 * delta**2) + 4.0 / 29.0,
        )

    fx, fy, fz = (transform(component) for component in xyz)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def frequency_geometry(size: int, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    one_d = np.fft.fftfreq(size)
    radius = np.sqrt(one_d[:, None] ** 2 + one_d[None, :] ** 2)
    edges = np.linspace(0.0, 0.5, bins + 1)
    index = np.digitize(radius.ravel(), edges) - 1
    valid = (index >= 0) & (index < bins)
    counts = np.bincount(index[valid], minlength=bins).astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return index, valid, counts, centers


def measure_batch(images: np.ndarray, workers: int) -> list[dict]:
    l_star, a_star, b_star = srgb_to_lab(images)
    od = -np.log((images.astype(np.float32) + 1.0) / 256.0).mean(axis=-1)
    gx = od[:, 1:-1, 2:] - od[:, 1:-1, :-2]
    gy = od[:, 2:, 1:-1] - od[:, :-2, 1:-1]
    window_1d = np.hanning(TARGET_PX).astype(np.float32)
    window = window_1d[:, None] * window_1d[None, :]
    centered = od - od.mean(axis=(1, 2), keepdims=True)
    fourier = fft.fft2(centered * window[None], axes=(-2, -1), workers=workers)
    power = np.abs(fourier) ** 2
    index, valid, counts, frequency_px = frequency_geometry(TARGET_PX, SPECTRAL_BINS)
    frequency_um = frequency_px / TARGET_MPP
    rows = []
    for item in range(len(images)):
        radial = np.bincount(
            index[valid], weights=power[item].ravel()[valid], minlength=SPECTRAL_BINS
        ) / np.maximum(counts, 1.0)
        log2_amplitude = 0.5 * np.log2(np.maximum(radial, 1e-20))
        bands = {}
        for name, (lower, upper) in BANDS_CYC_PER_UM.items():
            keep = (frequency_um >= lower) & (frequency_um < upper)
            if not keep.any():
                raise ValueError(f"empty frequency band {name}")
            bands[f"frequency_{name}_log2_amplitude"] = float(log2_amplitude[keep].mean())
        saturation = ((images[item] <= 1) | (images[item] >= 254)).any(axis=-1)
        rows.append(
            {
                "lab_l_mean": float(l_star[item].mean()),
                "lab_a_mean": float(a_star[item].mean()),
                "lab_b_mean": float(b_star[item].mean()),
                "lab_l_sd": float(l_star[item].std()),
                "mean_od": float(od[item].mean()),
                "od_sd": float(od[item].std()),
                "gradient_rms": float(np.sqrt(np.mean(gx[item] ** 2 + gy[item] ** 2))),
                "saturation_fraction": float(saturation.mean()),
                **bands,
            }
        )
    return rows


def _vips_to_numpy(image) -> np.ndarray:
    if image.hasalpha():
        image = image.flatten(background=255)
    if image.bands > 3:
        image = image.extract_band(0, n=3)
    if image.bands != 3:
        raise ValueError(f"expected RGB image, got {image.bands} bands")
    if image.format != "uchar":
        image = image.cast("uchar")
    return np.ndarray(
        buffer=image.write_to_memory(),
        dtype=np.uint8,
        shape=(image.height, image.width, image.bands),
    ).copy()


def extract_task(task_index: int, plism_root: Path, output_root: Path, workers: int, batch_size: int) -> None:
    import pyvips

    contract_path = output_root / "00_contract/analysis_contract.json"
    contract_hash = sha256(contract_path)
    tasks = pd.read_csv(output_root / "00_contract/tasks.csv")
    matches = tasks.loc[tasks["task_index"] == task_index]
    if len(matches) != 1:
        raise KeyError(f"task {task_index} absent from tasks.csv")
    task = matches.iloc[0]
    destination = output_root / f"01_image_metrics/shards/{task_index:03d}.csv"
    summary_path = destination.with_suffix(".summary.json")
    if destination.exists() and summary_path.exists():
        old = json.loads(summary_path.read_text())
        if (
            old.get("analysis_version") == ANALYSIS_VERSION
            and old.get("contract_sha256") == contract_hash
            and old.get("output_sha256") == sha256(destination)
        ):
            print(json.dumps({"status": "already_complete", "task_index": task_index}))
            return

    plan_path = Path(str(task["plan_path"]))
    if sha256(plan_path) != str(task["plan_sha256"]):
        raise ValueError(f"task {task_index}: plan hash changed")
    plan = pd.read_csv(plan_path)
    source = plism_root / "original_wsi" / str(task["actual_file"])
    image = pyvips.Image.new_from_file(str(source), level=0, access="random")
    if image.hasalpha():
        image = image.flatten(background=255)
    mpp = float(plan["mpp"].iloc[0])
    if not np.allclose(plan["mpp"], mpp):
        raise ValueError(f"task {task_index}: multiple MPP values")
    side = int(round(PATCH_UM / mpp))
    records = []
    skipped = 0
    for start in range(0, len(plan), batch_size):
        batch_plan = plan.iloc[start : start + batch_size]
        arrays = []
        metadata = []
        for row in batch_plan.itertuples(index=False):
            x = int(round(float(row.centre_x))) - side // 2
            y = int(round(float(row.centre_y))) - side // 2
            if x < 0 or y < 0 or x + side > image.width or y + side > image.height:
                skipped += 1
                continue
            patch = image.crop(x, y, side, side)
            if str(row.flip).strip().lower() in {"true", "1"}:
                patch = patch.rot("d180")
            rendered = patch.resize(
                TARGET_PX / patch.width,
                vscale=TARGET_PX / patch.height,
                kernel="lanczos3",
            )
            if rendered.width != TARGET_PX or rendered.height != TARGET_PX:
                rendered = rendered.thumbnail_image(
                    TARGET_PX, height=TARGET_PX, size="force", kernel="lanczos3"
                )
            arrays.append(_vips_to_numpy(rendered))
            metadata.append(row)
        if not arrays:
            continue
        measurements = measure_batch(np.stack(arrays), workers)
        for row, measured in zip(metadata, measurements):
            records.append(
                {
                    "stain": str(row.stain),
                    "scanner": str(row.scanner),
                    "location": int(row.location),
                    "core": int(row.core),
                    "tissue_type": str(row.tissue_type),
                    "replicate": int(row.replicate),
                    "residual_um": float(row.residual_um),
                    "response": float(row.response),
                    "native_mpp": mpp,
                    "native_patch_px": side,
                    **measured,
                }
            )
        print(
            f"[{task_index:03d} {task.stain}/{task.scanner}] "
            f"{min(start + batch_size, len(plan))}/{len(plan)}",
            flush=True,
        )
    frame = pd.DataFrame(records)
    if len(frame) != len(plan):
        raise ValueError(
            f"task {task_index}: {len(frame)} rendered but {len(plan)} planned; skipped={skipped}"
        )
    write_frame(frame, destination)
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "completed_utc": utc_now(),
        "task_index": task_index,
        "stain": str(task["stain"]),
        "scanner": str(task["scanner"]),
        "source": str(source.resolve()),
        "source_bytes": source.stat().st_size,
        "file_corrected": str(task["file_corrected"]).strip().lower() in {"true", "1"},
        "native_mpp": mpp,
        "native_patch_px": side,
        "rendered_patch_px": TARGET_PX,
        "locations": len(frame),
        "contract_sha256": contract_hash,
        "plan_sha256": sha256(plan_path),
        "output_sha256": sha256(destination),
    }
    write_json(summary_path, summary)
    print(json.dumps({"status": "complete", **summary}, indent=2))


def robust_pannormal_scales(root: Path) -> pd.DataFrame:
    arrays = []
    names = None
    for path in sorted((root / "01_reachability/shards").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            current = [item.decode() for item in data["endpoint_names"]]
            if names is None:
                names = current
            elif names != current:
                raise ValueError(f"endpoint mismatch in {path}")
            arrays.append(np.asarray(data["target_vectors"], dtype=float).reshape(-1, len(current)))
    if not arrays or names != list(ENDPOINTS):
        raise ValueError("PanNormal target vectors are missing or use unexpected endpoints")
    values = np.concatenate(arrays, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.std(values, axis=0, ddof=1)
    scale = np.where(scale > 1e-6, scale, fallback)
    return pd.DataFrame(
        {
            "endpoint": names,
            "robust_scale": scale,
            "pannormal_target_rows": len(values),
        }
    )


def build_contrasts(metrics: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    keys = ["stain", "location"]
    reference_columns = {
        "lab_l_mean": "reference_lab_l_mean",
        "lab_a_mean": "reference_lab_a_mean",
        "lab_b_mean": "reference_lab_b_mean",
        "lab_l_sd": "reference_lab_l_sd",
        "mean_od": "reference_mean_od",
        "od_sd": "reference_od_sd",
        "gradient_rms": "reference_gradient_rms",
        "frequency_anchor_log2_amplitude": "reference_frequency_anchor_log2_amplitude",
        "frequency_low_mid_log2_amplitude": "reference_frequency_low_mid_log2_amplitude",
        "frequency_mid_log2_amplitude": "reference_frequency_mid_log2_amplitude",
        "frequency_high_log2_amplitude": "reference_frequency_high_log2_amplitude",
    }
    reference = metrics.loc[metrics["scanner"] == REFERENCE, keys + list(reference_columns)].rename(
        columns=reference_columns
    )
    target = metrics.loc[metrics["scanner"] != REFERENCE].merge(
        reference, on=keys, how="inner", validate="many_to_one"
    )
    target = target.merge(
        selected[
            [
                "stain",
                "location",
                "pannormal_tissue",
                "mapping_tier",
                "organ_aligned_status",
                "strict_status",
            ]
        ],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    target["delta_lab_l"] = target["lab_l_mean"] - target["reference_lab_l_mean"]
    target["delta_lab_a"] = target["lab_a_mean"] - target["reference_lab_a_mean"]
    target["delta_lab_b"] = target["lab_b_mean"] - target["reference_lab_b_mean"]
    for output, value, ref in (
        ("log2_mean_od_ratio", "mean_od", "reference_mean_od"),
        ("log2_lab_l_sd_ratio", "lab_l_sd", "reference_lab_l_sd"),
        ("log2_od_sd_ratio", "od_sd", "reference_od_sd"),
        ("log2_gradient_rms_ratio", "gradient_rms", "reference_gradient_rms"),
    ):
        target[output] = np.log2(
            np.maximum(target[value], 1e-12) / np.maximum(target[ref], 1e-12)
        )
    anchor_delta = (
        target["frequency_anchor_log2_amplitude"]
        - target["reference_frequency_anchor_log2_amplitude"]
    )
    for name in ("low_mid", "mid", "high"):
        target[f"frequency_{name}"] = (
            target[f"frequency_{name}_log2_amplitude"]
            - target[f"reference_frequency_{name}_log2_amplitude"]
            - anchor_delta
        )
    target["scanner_status"] = np.where(
        target["scanner"].isin(SHARED_TARGET_SCANNERS), "shared", "novel"
    )
    return target


def add_quadrant(frame: pd.DataFrame, tissue_column: str) -> pd.DataFrame:
    result = frame.copy()
    tissue_shared = result[tissue_column].eq("shared")
    scanner_shared = result["scanner_status"].eq("shared")
    result["quadrant"] = np.select(
        [
            scanner_shared & tissue_shared,
            ~scanner_shared & tissue_shared,
            scanner_shared & ~tissue_shared,
            ~scanner_shared & ~tissue_shared,
        ],
        QUADRANT_ORDER,
        default="unknown",
    )
    return result


def section_bootstrap(values: pd.DataFrame, columns: list[str], seed: int) -> pd.DataFrame:
    sections = sorted(values["stain"].unique())
    matrix = values.set_index("stain").reindex(sections)[columns].to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("section bootstrap received incomplete values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sections), size=(BOOTSTRAP_REPLICATES, len(sections)))
    draws = matrix[indices].mean(axis=1)
    return pd.DataFrame(
        {
            "metric": columns,
            "estimate": matrix.mean(axis=0),
            "ci_low": np.quantile(draws, 0.025, axis=0),
            "ci_high": np.quantile(draws, 0.975, axis=0),
            "sections": len(sections),
            "positive_sections": (matrix > 0).sum(axis=0),
        }
    )


def summarize_factorial(
    contrasts: pd.DataFrame,
    scales: pd.DataFrame,
    tissue_scheme: str,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tissue_column = "organ_aligned_status" if tissue_scheme == "organ_aligned" else "strict_status"
    frame = contrasts.copy()
    if variant == "exclude_hrh_s60":
        frame = frame.loc[~((frame["stain"] == "HRH") & (frame["scanner"] == "S60"))]
    frame = add_quadrant(frame, tissue_column)
    scale_map = scales.set_index("endpoint")["robust_scale"]
    cell = frame.groupby(
        ["stain", "scanner", "scanner_status", "core", "tissue_type", "quadrant"],
        as_index=False,
    )[list(ENDPOINTS)].mean()
    standardized = cell[list(ENDPOINTS)].to_numpy(float) / scale_map.reindex(ENDPOINTS).to_numpy()
    cell["phenotype_distance"] = np.sqrt(np.mean(standardized**2, axis=1))
    section = (
        cell.groupby(["stain", "quadrant"], as_index=False)["phenotype_distance"]
        .mean()
        .pivot(index="stain", columns="quadrant", values="phenotype_distance")
        .reindex(columns=QUADRANT_ORDER)
        .reset_index()
    )
    section["scanner_ood_increment"] = section["scanner-OOD"] - section["shared/shared"]
    section["tissue_ood_increment"] = section["tissue-OOD"] - section["shared/shared"]
    section["double_ood_increment"] = section["double-OOD"] - section["shared/shared"]
    section["double_excess"] = (
        section["double-OOD"]
        - section["scanner-OOD"]
        - section["tissue-OOD"]
        + section["shared/shared"]
    )
    summary_columns = [*QUADRANT_ORDER, "scanner_ood_increment", "tissue_ood_increment", "double_ood_increment", "double_excess"]
    summary = section_bootstrap(
        section, summary_columns, BOOTSTRAP_SEED + (0 if tissue_scheme == "organ_aligned" else 100) + (0 if variant == "primary" else 10)
    )
    summary.insert(0, "variant", variant)
    summary.insert(0, "tissue_scheme", tissue_scheme)
    contribution_rows = []
    # Keep the contribution view on the same core-section unit used by the
    # factorial estimand; a core with more retained locations must not carry
    # more weight merely because it yielded more patches.
    for quadrant, block in cell.groupby("quadrant", sort=False):
        values = block[list(ENDPOINTS)].abs().mean() / scale_map.reindex(ENDPOINTS)
        for endpoint, value in values.items():
            contribution_rows.append(
                {"quadrant": quadrant, "endpoint": endpoint, "value": float(value)}
            )
    contribution = pd.DataFrame(contribution_rows)
    contribution.insert(0, "variant", variant)
    contribution.insert(0, "tissue_scheme", tissue_scheme)
    cell.insert(0, "variant", variant)
    cell.insert(0, "tissue_scheme", tissue_scheme)
    section.insert(0, "variant", variant)
    section.insert(0, "tissue_scheme", tissue_scheme)
    return cell, section, summary, contribution


def analyse_uni(plism_root: Path, selected: pd.DataFrame) -> pd.DataFrame:
    feature_root = plism_root / "features/registered_AT2/uni_v2"
    rows = []
    for stain, location_block in selected.groupby("stain", sort=True):
        wanted = np.sort(location_block["location"].unique())
        ref_path = feature_root / f"{stain}_{REFERENCE}.h5"
        with h5py.File(ref_path, "r") as handle:
            ref_locations = np.asarray(handle["location"], dtype=int)
            positions = np.searchsorted(ref_locations, wanted)
            valid = (positions < len(ref_locations)) & (
                ref_locations[np.minimum(positions, len(ref_locations) - 1)] == wanted
            )
            wanted = wanted[valid]
            positions = positions[valid]
            if not len(wanted):
                raise ValueError(f"{stain}: no selected locations in UNI reference")
            reference_features = np.asarray(handle["features"][positions], dtype=np.float32)
        reference_norm = np.linalg.norm(reference_features, axis=1)
        location_meta = location_block.set_index("location").reindex(wanted)
        for scanner in (*SHARED_TARGET_SCANNERS, *NOVEL_TARGET_SCANNERS):
            path = feature_root / f"{stain}_{scanner}.h5"
            with h5py.File(path, "r") as handle:
                locations = np.asarray(handle["location"], dtype=int)
                positions = np.searchsorted(locations, wanted)
                valid = (positions < len(locations)) & (
                    locations[np.minimum(positions, len(locations) - 1)] == wanted
                )
                if not valid.all():
                    raise ValueError(f"{stain}/{scanner}: selected locations absent from UNI")
                features = np.asarray(handle["features"][positions], dtype=np.float32)
            cosine = np.sum(features * reference_features, axis=1) / np.maximum(
                np.linalg.norm(features, axis=1) * reference_norm, 1e-12
            )
            for location, value in zip(wanted, cosine):
                meta = location_meta.loc[location]
                rows.append(
                    {
                        "stain": stain,
                        "scanner": scanner,
                        "scanner_status": "shared" if scanner in SHARED_TARGET_SCANNERS else "novel",
                        "location": int(location),
                        "core": int(meta.core),
                        "tissue_type": str(meta.tissue_type),
                        "organ_aligned_status": str(meta.organ_aligned_status),
                        "strict_status": str(meta.strict_status),
                        "cosine_distance": float(1.0 - value),
                    }
                )
    return pd.DataFrame(rows)


def summarize_uni(frame: pd.DataFrame, tissue_scheme: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tissue_column = "organ_aligned_status" if tissue_scheme == "organ_aligned" else "strict_status"
    frame = add_quadrant(frame, tissue_column)
    cell = frame.groupby(
        ["stain", "scanner", "scanner_status", "core", "tissue_type", "quadrant"],
        as_index=False,
    )["cosine_distance"].mean()
    section = (
        cell.groupby(["stain", "quadrant"], as_index=False)["cosine_distance"]
        .mean()
        .pivot(index="stain", columns="quadrant", values="cosine_distance")
        .reindex(columns=QUADRANT_ORDER)
        .reset_index()
    )
    section["scanner_ood_increment"] = section["scanner-OOD"] - section["shared/shared"]
    section["tissue_ood_increment"] = section["tissue-OOD"] - section["shared/shared"]
    section["double_ood_increment"] = section["double-OOD"] - section["shared/shared"]
    section["double_excess"] = section["double-OOD"] - section["scanner-OOD"] - section["tissue-OOD"] + section["shared/shared"]
    summary = section_bootstrap(
        section,
        [*QUADRANT_ORDER, "scanner_ood_increment", "tissue_ood_increment", "double_ood_increment", "double_excess"],
        BOOTSTRAP_SEED + 500 + (0 if tissue_scheme == "organ_aligned" else 100),
    )
    summary.insert(0, "tissue_scheme", tissue_scheme)
    return section, summary


def scanner_summary(frame: pd.DataFrame, value: str, space: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    section = frame.groupby(["stain", "scanner"], as_index=False)[value].mean()
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + (900 if space == "image" else 1000))
    for scanner in (*SHARED_TARGET_SCANNERS, *NOVEL_TARGET_SCANNERS):
        values = (
            section.loc[section["scanner"] == scanner]
            .set_index("stain")
            .reindex(sorted(section["stain"].unique()))[value]
            .to_numpy(float)
        )
        if not np.isfinite(values).all():
            raise ValueError(f"incomplete scanner-section summary for {space}/{scanner}")
        indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
        draws = values[indices].mean(axis=1)
        rows.append(
            {
                "space": space,
                "scanner": scanner,
                "scanner_status": "shared" if scanner in SHARED_TARGET_SCANNERS else "novel",
                "estimate": float(values.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "sections": len(values),
            }
        )
    section.insert(0, "space", space)
    return section, pd.DataFrame(rows)


def shared_scanner_replication(
    contrasts: pd.DataFrame, scales: pd.DataFrame, pannormal_models: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = pd.read_csv(pannormal_models)
    scale_map = scales.set_index("endpoint")["robust_scale"]
    # Balance the external fingerprint over section/core cells before taking
    # the scanner mean, matching the image-space estimand used elsewhere.
    plism_cells = contrasts.groupby(
        ["stain", "scanner", "core"], as_index=False
    )[list(ENDPOINTS)].mean()
    plism = plism_cells.groupby("scanner")[list(ENDPOINTS)].mean()
    rows = []
    summaries = []
    for scanner in SHARED_TARGET_SCANNERS:
        internal = (
            models.loc[
                (models["scanner"] == scanner.lower()) & models["endpoint"].isin(ENDPOINTS),
                ["endpoint", "fixed_mean"],
            ]
            .set_index("endpoint")
            .reindex(ENDPOINTS)
        )
        if internal["fixed_mean"].isna().any():
            raise ValueError(f"missing PanNormal fixed effects for {scanner}")
        x = internal["fixed_mean"].to_numpy(float) / scale_map.reindex(ENDPOINTS).to_numpy()
        y = plism.loc[scanner, list(ENDPOINTS)].to_numpy(float) / scale_map.reindex(ENDPOINTS).to_numpy()
        for endpoint, left, right in zip(ENDPOINTS, x, y):
            rows.append(
                {
                    "scanner": scanner,
                    "endpoint": endpoint,
                    "pannormal_standardized_effect": float(left),
                    "plism_standardized_effect": float(right),
                    "sign_match": bool(np.sign(left) == np.sign(right)),
                }
            )
        summaries.append(
            {
                "scanner": scanner,
                "pearson": float(np.corrcoef(x, y)[0, 1]),
                "spearman": float(pd.Series(x).rank().corr(pd.Series(y).rank())),
                "sign_concordance": float(np.mean(np.sign(x) == np.sign(y))),
                "endpoints": len(ENDPOINTS),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def make_figures(
    image_section: pd.DataFrame,
    image_summary: pd.DataFrame,
    contribution: pd.DataFrame,
    uni_summary: pd.DataFrame,
    scanner_summaries: pd.DataFrame,
    replication: pd.DataFrame,
    replication_summary: pd.DataFrame,
    output_root: Path,
) -> None:
    figure_root = output_root / "04_figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    primary = image_summary.query("tissue_scheme == 'organ_aligned' and variant == 'primary'")
    uni = uni_summary.query("tissue_scheme == 'organ_aligned'")
    image_values = primary.set_index("metric").reindex(QUADRANT_ORDER)["estimate"].to_numpy().reshape(2, 2)
    uni_values = uni.set_index("metric").reindex(QUADRANT_ORDER)["estimate"].to_numpy().reshape(2, 2)
    labels = np.array([["shared/shared", "scanner-OOD"], ["tissue-OOD", "double-OOD"]])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    for axis, matrix, title, fmt in (
        (axes[0], image_values, "Image phenotype distance\n(PanNormal robust-SD units)", ".2f"),
        (axes[1], uni_values, "Frozen UNI-v2 distance\n(1 - paired cosine)", ".3f"),
    ):
        image = axis.imshow(matrix, cmap="YlOrRd", aspect="auto")
        threshold = float((np.nanmin(matrix) + np.nanmax(matrix)) / 2.0)
        for row in range(2):
            for column in range(2):
                text_color = "white" if matrix[row, column] > threshold else "#1c2a2d"
                axis.text(
                    column,
                    row,
                    f"{labels[row, column]}\n{matrix[row, column]:{fmt}}",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color=text_color,
                )
        axis.set_xticks([0, 1], ["Shared model", "Novel model"])
        axis.set_yticks([0, 1], ["Shared tissue", "Novel tissue"])
        axis.set_title(title, loc="left", fontweight="bold")
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.suptitle("PLISM factorial external stress test", x=0.02, ha="left", fontsize=15, fontweight="bold")
    fig.savefig(figure_root / "figure01_factorial_quadrants.png", dpi=190)
    plt.close(fig)

    increment_order = ("scanner_ood_increment", "tissue_ood_increment", "double_ood_increment", "double_excess")
    block = primary.set_index("metric").reindex(increment_order)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)
    x = np.arange(len(block))
    mean = block["estimate"].to_numpy()
    axes[0].errorbar(
        x,
        mean,
        yerr=np.vstack([mean - block["ci_low"].to_numpy(), block["ci_high"].to_numpy() - mean]),
        fmt="o",
        color="#175d72",
        capsize=5,
        lw=2,
    )
    axes[0].axhline(0, color="#68777a", ls="--", lw=1)
    axes[0].set_xticks(x, ["Scanner\nOOD", "Tissue\nOOD", "Double\nOOD", "Beyond\nadditive"])
    axes[0].set_ylabel("Increment from shared/shared")
    axes[0].set_title("Which axis increases the shift?", loc="left", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.2)

    section = image_section.query("tissue_scheme == 'organ_aligned' and variant == 'primary'")
    for quadrant, color in zip(QUADRANT_ORDER, ("#2b6f77", "#d08b2e", "#6f5aa8", "#a63d40")):
        axes[1].plot(np.arange(len(section)), section[quadrant], marker="o", ms=4, lw=1.5, label=quadrant, color=color)
    axes[1].set_xticks(np.arange(len(section)), section["stain"], rotation=55, ha="right")
    axes[1].set_ylabel("Image phenotype distance")
    axes[1].set_title("Does the ordering survive every section?", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(figure_root / "figure02_increments_and_sections.png", dpi=190)
    plt.close(fig)

    block = contribution.query("tissue_scheme == 'organ_aligned' and variant == 'primary'")
    matrix = block.pivot(index="endpoint", columns="quadrant", values="value").reindex(index=ENDPOINTS, columns=QUADRANT_ORDER)
    fig, axis = plt.subplots(figsize=(10.8, 7.2), constrained_layout=True)
    image = axis.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")
    axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(matrix.index)), [name.replace("_", " ") for name in matrix.index])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    axis.set_title("Which image endpoints drive each external quadrant?", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.8)
    colorbar.set_label("Mean absolute contrast (PanNormal robust-SD units)")
    fig.savefig(figure_root / "figure03_endpoint_contributions.png", dpi=190)
    plt.close(fig)

    scanners = [*SHARED_TARGET_SCANNERS, *NOVEL_TARGET_SCANNERS]
    colors = ["#175d72" if scanner in SHARED_TARGET_SCANNERS else "#d08b2e" for scanner in scanners]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.3), constrained_layout=True)
    for axis, space, title, ylabel in (
        (axes[0], "image", "Image phenotype is scanner-specific", "Phenotype distance"),
        (axes[1], "uni", "UNI-v2 shift is also scanner-specific", "1 - paired cosine"),
    ):
        block = scanner_summaries.loc[scanner_summaries["space"] == space].set_index("scanner").reindex(scanners)
        x = np.arange(len(scanners))
        mean = block["estimate"].to_numpy()
        for index, scanner in enumerate(scanners):
            axis.errorbar(
                index,
                mean[index],
                yerr=[[mean[index] - block.iloc[index]["ci_low"]], [block.iloc[index]["ci_high"] - mean[index]]],
                fmt="o",
                color=colors[index],
                capsize=4,
                markersize=7,
            )
        axis.set_xticks(x, scanners)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
        axis.axvline(2.5, color="#aeb8ba", ls="--", lw=1)
        for tick, color in zip(axis.get_xticklabels(), colors):
            tick.set_color(color)
            tick.set_fontweight("bold")
        axis.text(
            1,
            -0.16,
            "shared models",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            color="#175d72",
            fontsize=9,
        )
        axis.text(
            4,
            -0.16,
            "novel models",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            color="#a26316",
            fontsize=9,
        )
    fig.suptitle("OOD status is not an ordered difficulty score", x=0.02, ha="left", fontsize=15, fontweight="bold")
    fig.savefig(figure_root / "figure04_scanner_resolved.png", dpi=190)
    plt.close(fig)

    endpoint_labels = ("L*", "a*", "b*", "mean OD", "L* SD", "OD SD", "gradient", "freq L", "freq M", "freq H")
    all_values = np.concatenate(
        [
            replication["pannormal_standardized_effect"].to_numpy(),
            replication["plism_standardized_effect"].to_numpy(),
        ]
    )
    lower, upper = float(all_values.min()), float(all_values.max())
    padding = 0.08 * (upper - lower)
    endpoint_colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.6), constrained_layout=True)
    for axis, scanner in zip(axes, SHARED_TARGET_SCANNERS):
        block = replication.loc[replication["scanner"] == scanner].set_index("endpoint").reindex(ENDPOINTS)
        stats = replication_summary.set_index("scanner").loc[scanner]
        x = block["pannormal_standardized_effect"].to_numpy()
        y = block["plism_standardized_effect"].to_numpy()
        for index, (label, left, right) in enumerate(zip(endpoint_labels, x, y)):
            axis.scatter(
                left,
                right,
                s=52,
                color=endpoint_colors[index],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
                label=label,
            )
        axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], color="#7f8d90", ls="--", lw=1)
        axis.axhline(0, color="#d4d9da", lw=0.8)
        axis.axvline(0, color="#d4d9da", lw=0.8)
        axis.set_xlim(lower - padding, upper + padding)
        axis.set_ylim(lower - padding, upper + padding)
        axis.set_title(
            f"{scanner}: r={stats.pearson:.2f}, sign={int(round(10 * stats.sign_concordance))}/10",
            loc="left",
            fontweight="bold",
        )
        axis.set_xlabel("PanNormal effect (robust-SD)")
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("PLISM effect (robust-SD)")
    fig.suptitle("Does a shared scanner model keep the same image fingerprint?", x=0.02, ha="left", fontsize=15, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=5,
        frameon=False,
        fontsize=8.5,
    )
    fig.savefig(
        figure_root / "figure05_cross_cohort_fingerprint.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)


def aggregate(plism_root: Path, output_root: Path, pannormal_root: Path) -> None:
    contract_path = output_root / "00_contract/analysis_contract.json"
    contract = json.loads(contract_path.read_text())
    contract_hash = sha256(contract_path)
    tasks = pd.read_csv(output_root / "00_contract/tasks.csv")
    frames = []
    shard_hashes = {}
    for task in tasks.itertuples(index=False):
        path = output_root / f"01_image_metrics/shards/{int(task.task_index):03d}.csv"
        summary_path = path.with_suffix(".summary.json")
        if not path.exists() or not summary_path.exists():
            raise FileNotFoundError(f"missing completed shard {path}")
        summary = json.loads(summary_path.read_text())
        if summary.get("contract_sha256") != contract_hash or summary.get("output_sha256") != sha256(path):
            raise ValueError(f"stale or corrupt shard {path}")
        frames.append(pd.read_csv(path))
        shard_hashes[path.name] = sha256(path)
    metrics = pd.concat(frames, ignore_index=True)
    selected = pd.read_csv(output_root / "00_contract/selected_locations.csv")
    expected_rows = len(selected) * len(SCANNERS)
    if len(metrics) != expected_rows:
        raise ValueError(f"absolute measurement row count {len(metrics)} != {expected_rows}")
    write_frame(metrics, output_root / "01_image_metrics/all_absolute_metrics.csv")

    contrasts = build_contrasts(metrics, selected)
    scales = robust_pannormal_scales(pannormal_root)
    write_frame(scales, output_root / "02_factorial/endpoint_scales.csv")
    write_frame(contrasts, output_root / "02_factorial/location_contrasts.csv")
    cell_parts, section_parts, summary_parts, contribution_parts = [], [], [], []
    for tissue_scheme in ("organ_aligned", "strict"):
        for variant in ("primary", "exclude_hrh_s60"):
            cell, section, summary, contribution = summarize_factorial(
                contrasts, scales, tissue_scheme, variant
            )
            cell_parts.append(cell)
            section_parts.append(section)
            summary_parts.append(summary)
            contribution_parts.append(contribution)
    cells = pd.concat(cell_parts, ignore_index=True)
    sections = pd.concat(section_parts, ignore_index=True)
    summaries = pd.concat(summary_parts, ignore_index=True)
    contributions = pd.concat(contribution_parts, ignore_index=True)
    write_frame(cells, output_root / "02_factorial/core_section_cells.csv")
    write_frame(sections, output_root / "02_factorial/section_sensitivity.csv")
    write_frame(summaries, output_root / "02_factorial/quadrant_summary.csv")
    write_frame(contributions, output_root / "02_factorial/endpoint_contributions.csv")

    uni_location = analyse_uni(plism_root, selected)
    uni_location_count = int(
        uni_location[["stain", "location"]].drop_duplicates().shape[0]
    )
    uni_missing_image_locations = int(len(selected) - uni_location_count)
    write_frame(uni_location, output_root / "03_uni_bridge/location_cosine_distance.csv")
    uni_sections, uni_summaries = [], []
    for tissue_scheme in ("organ_aligned", "strict"):
        section, summary = summarize_uni(uni_location, tissue_scheme)
        section.insert(0, "tissue_scheme", tissue_scheme)
        uni_sections.append(section)
        uni_summaries.append(summary)
    uni_section = pd.concat(uni_sections, ignore_index=True)
    uni_summary = pd.concat(uni_summaries, ignore_index=True)
    write_frame(uni_section, output_root / "03_uni_bridge/section_sensitivity.csv")
    write_frame(uni_summary, output_root / "03_uni_bridge/quadrant_summary.csv")

    image_scanner_section, image_scanner_summary = scanner_summary(
        cells.query("tissue_scheme == 'organ_aligned' and variant == 'primary'"),
        "phenotype_distance",
        "image",
    )
    uni_cells_for_scanner = uni_location.groupby(
        ["stain", "scanner", "scanner_status", "core"], as_index=False
    )["cosine_distance"].mean()
    uni_scanner_section, uni_scanner_summary = scanner_summary(
        uni_cells_for_scanner, "cosine_distance", "uni"
    )
    scanner_sections = pd.concat([image_scanner_section, uni_scanner_section], ignore_index=True)
    scanner_summaries = pd.concat([image_scanner_summary, uni_scanner_summary], ignore_index=True)
    write_frame(scanner_sections, output_root / "02_factorial/scanner_section_summary.csv")
    write_frame(scanner_summaries, output_root / "02_factorial/scanner_summary.csv")
    replication, replication_summary = shared_scanner_replication(
        contrasts,
        scales,
        ROOT / "outputs/final_image_study_v1/04_lmm/model_summary.csv",
    )
    write_frame(replication, output_root / "02_factorial/shared_scanner_replication.csv")
    write_frame(replication_summary, output_root / "02_factorial/shared_scanner_replication_summary.csv")

    make_figures(
        sections,
        summaries,
        contributions,
        uni_summary,
        scanner_summaries,
        replication,
        replication_summary,
        output_root,
    )
    primary = summaries.query("tissue_scheme == 'organ_aligned' and variant == 'primary'").set_index("metric")
    focus = summaries.query("tissue_scheme == 'organ_aligned' and variant == 'exclude_hrh_s60'").set_index("metric")
    uni_primary = uni_summary.query("tissue_scheme == 'organ_aligned'").set_index("metric")
    output_files = [
        output_root / "02_factorial/quadrant_summary.csv",
        output_root / "02_factorial/section_sensitivity.csv",
        output_root / "02_factorial/endpoint_contributions.csv",
        output_root / "03_uni_bridge/quadrant_summary.csv",
        output_root / "02_factorial/scanner_summary.csv",
        output_root / "02_factorial/shared_scanner_replication.csv",
        output_root / "02_factorial/shared_scanner_replication_summary.csv",
        output_root / "04_figures/figure01_factorial_quadrants.png",
        output_root / "04_figures/figure02_increments_and_sections.png",
        output_root / "04_figures/figure03_endpoint_contributions.png",
        output_root / "04_figures/figure04_scanner_resolved.png",
        output_root / "04_figures/figure05_cross_cohort_fingerprint.png",
    ]
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "completed_utc": utc_now(),
        "status": "pass",
        "contract_sha256": contract_hash,
        "sections": int(selected["stain"].nunique()),
        "tissue_cores": int(selected["core"].nunique()),
        "organ_aligned_tissues": int(selected.query("organ_aligned_status == 'shared'")["tissue_type"].nunique()),
        "novel_tissues": int(selected.query("organ_aligned_status == 'novel'")["tissue_type"].nunique()),
        "selected_locations": int(len(selected)),
        "absolute_measurements": int(len(metrics)),
        "paired_target_contrasts": int(len(contrasts)),
        "uni_locations": uni_location_count,
        "uni_missing_image_locations": uni_missing_image_locations,
        "uni_paired_distances": int(len(uni_location)),
        "primary_image_quadrants": {
            name: {
                "estimate": float(primary.loc[name, "estimate"]),
                "ci": [float(primary.loc[name, "ci_low"]), float(primary.loc[name, "ci_high"])],
            }
            for name in QUADRANT_ORDER
        },
        "primary_image_increments": {
            name: {
                "estimate": float(primary.loc[name, "estimate"]),
                "ci": [float(primary.loc[name, "ci_low"]), float(primary.loc[name, "ci_high"])],
                "positive_sections": int(primary.loc[name, "positive_sections"]),
            }
            for name in ("scanner_ood_increment", "tissue_ood_increment", "double_ood_increment", "double_excess")
        },
        "focus_sensitivity_max_absolute_change": float(
            max(abs(float(focus.loc[name, "estimate"] - primary.loc[name, "estimate"])) for name in QUADRANT_ORDER)
        ),
        "uni_quadrants": {
            name: {
                "estimate": float(uni_primary.loc[name, "estimate"]),
                "ci": [float(uni_primary.loc[name, "ci_low"]), float(uni_primary.loc[name, "ci_high"])],
            }
            for name in QUADRANT_ORDER
        },
        "shared_scanner_replication": {
            row.scanner: {
                "pearson": float(row.pearson),
                "spearman": float(row.spearman),
                "sign_concordance": float(row.sign_concordance),
            }
            for row in replication_summary.itertuples(index=False)
        },
        "shards": {"count": len(shard_hashes), "sha256": shard_hashes},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in output_files},
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plism-root", type=Path, default=PLISM_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--locations-per-core", type=int, default=LOCATIONS_PER_CORE_SECTION)
    extract_parser = subparsers.add_parser("extract-task")
    extract_parser.add_argument("--task-index", type=int, required=True)
    extract_parser.add_argument("--workers", type=int, default=4)
    extract_parser.add_argument("--batch-size", type=int, default=16)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--pannormal-root", type=Path, default=PAN_AUGMENTATION_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plism_root = args.plism_root.resolve()
    output_root = args.output_root.resolve()
    if args.command == "prepare":
        prepare(plism_root, output_root, args.locations_per_core)
    elif args.command == "extract-task":
        extract_task(args.task_index, plism_root, output_root, args.workers, args.batch_size)
    elif args.command == "aggregate":
        aggregate(plism_root, output_root, args.pannormal_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
