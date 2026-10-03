#!/usr/bin/env python3
"""Zero-shot PLISM evaluation for frozen GT450-to-AT2 Pix2Pix models.

The module deliberately separates five operations:

``prepare``
    Subsample the already locked PLISM locations without consulting images,
    embeddings, or generated outcomes.  The default is at most four locations
    per physical section and TMA core.
``render-task``
    Render one input WSI on the exact 256 px / 0.5052 um grid used by
    :mod:`plism_factorial_external`.
``evaluate-checkpoint``
    Apply one image-only-selected PanNormal checkpoint without fine-tuning and
    embed raw, generated, and paired real AT2 with the same frozen UNI-v1 used
    for the internal endpoint.
``aggregate``
    Report every CV checkpoint separately, then average checkpoint replicates
    before section-level inference.  Checkpoints are never treated as
    independent biological replicates.
``write-slurm``
    Freeze the five image-selected candidates and emit CPU/GPU batch scripts;
    it never submits them.

GT450 is the required primary input because it is the only PLISM scanner model
that matches a trained GT450-to-AT2 generator.  Other PLISM scanners may be
included as an explicitly exploratory model-boundary stress test.  AT2 is
excluded from closure and the 2x2 pool; it is passed through the generator only
in a separately labelled destructive identity-damage control.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shlex
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd

from .evaluate_images import (
    EDGE_HIGH_THRESHOLD,
    EDGE_LOW_THRESHOLD,
    IMAGE_EVALUATION_VERSION,
    IMAGE_METRICS,
    image_metrics,
)
from .predict import load_validated_generator, normalized_tensor_to_uint8
from .uni import (
    add_plism_model_status,
    cosine_distance_rows,
    embed_frozen_uni,
    embedding_collapse_diagnostics,
    paired_uni_metrics,
)


ROOT = Path(__file__).resolve().parents[2]
PLISM_ROOT = ROOT / "data/PLISM_dataset"
PLISM_ANALYSIS_ROOT = ROOT / "outputs/plism_factorial_external_v1"
PREDECESSOR_UNI_OUTPUT_ROOT = (
    ROOT
    / "outputs/scanner_batch_effect_analysis_2026-09-17"
    / "08_plism_external/pix2pix_gt450_to_at2"
)
OUTPUT_ROOT = (
    ROOT
    / "outputs/scanner_batch_effect_analysis_2026-09-17"
    / "08_plism_external/pix2pix_gt450_to_at2_image_safety"
)
PIX2PIX_RUNS_ROOT = (
    ROOT
    / "outputs/scanner_batch_effect_analysis_2026-09-17"
    / "06_learned_baselines/02_pix2pix/runs/gt450_to_at2"
)

ANALYSIS_VERSION = "pix2pix_gt450_to_at2_plism_external_v2"
MODEL_SOURCE_SCANNER = "GT450"
TARGET_SCANNER = "AT2"
PLISM_INPUT_SCANNERS = ("GT450", "S360", "S60", "P", "S210", "SQ")
TARGET_MPP = 0.5052
TARGET_PX = 256
PATCH_UM = TARGET_MPP * TARGET_PX
DEFAULT_LOCATIONS_PER_CORE_SECTION = 4
DEFAULT_BOOTSTRAP_REPLICATES = 20_000
DEFAULT_SEED = 20260917
UNI2_REPOSITORY = "MahmoodLab/UNI2-h"
UNI2_FILENAME = "pytorch_model.bin"
UNI2_FEATURE_DIM = 1536
UNI1_REPOSITORY = "MahmoodLab/uni"
UNI1_FILENAME = "pytorch_model.bin"
UNI1_FEATURE_DIM = 1024

METRICS = (
    "raw_to_at2_distance",
    "method_to_at2_distance",
    "gain_to_at2",
    "fractional_closure",
    "method_to_raw_distance",
)

IMAGE_SAFETY_GATES = {
    "source_gradient_ncc": {"direction": "minimum", "threshold": 0.90},
    "saturation_fraction": {"direction": "maximum", "threshold": 0.10},
    "invented_edge_fraction": {"direction": "maximum", "threshold": 0.001},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_frame(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _normalize_input_scanners(values: Sequence[str]) -> tuple[str, ...]:
    lookup = {value.casefold(): value for value in PLISM_INPUT_SCANNERS}
    scanners: list[str] = []
    for item in values:
        normalized = str(item).strip()
        if not normalized:
            raise ValueError("input scanner names must not be empty")
        if normalized.casefold() == TARGET_SCANNER.casefold():
            raise ValueError("AT2 input is forbidden: AT2 is the paired target, not a closure input")
        if normalized.casefold() not in lookup:
            raise ValueError(
                f"unsupported PLISM input scanner {item!r}; choose from {PLISM_INPUT_SCANNERS}"
            )
        canonical = lookup[normalized.casefold()]
        if canonical not in scanners:
            scanners.append(canonical)
    if MODEL_SOURCE_SCANNER not in scanners:
        raise ValueError("clean GT450 replication is required before model-boundary scanners")
    return tuple(scanners)


def _positions_for_locations(available: np.ndarray, wanted: np.ndarray, label: str) -> np.ndarray:
    available = np.asarray(available, dtype=np.int64)
    wanted = np.asarray(wanted, dtype=np.int64)
    if available.ndim != 1 or wanted.ndim != 1:
        raise ValueError(f"{label}: locations must be one-dimensional")
    if len(wanted) and not len(available):
        raise ValueError(f"{label}: feature store is empty")
    if len(available) and np.any(available[1:] <= available[:-1]):
        raise ValueError(f"{label}: stored locations are not strictly increasing")
    positions = np.searchsorted(available, wanted)
    valid = positions < len(available)
    valid &= available[np.minimum(positions, max(len(available) - 1, 0))] == wanted
    if not valid.all():
        missing = wanted[~valid][:10].tolist()
        raise ValueError(f"{label}: requested locations are absent: {missing}")
    return positions


def _feature_metadata(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as store:
        if "location" not in store or "features" not in store:
            raise ValueError(f"invalid PLISM UNI2 feature store: {path}")
        metadata = json.loads(str(store.attrs.get("meta", "{}")))
        features = store["features"]
        metadata.update(
            {
                "rows": int(features.shape[0]),
                "feature_dim_observed": int(features.shape[1]),
                "path": str(path.resolve()),
            }
        )
    return metadata


def _available_feature_locations(
    feature_root: Path, stain: str, scanners: Sequence[str]
) -> set[int]:
    common: set[int] | None = None
    for scanner in scanners:
        path = feature_root / f"{stain}_{scanner}.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as store:
            current = set(np.asarray(store["location"], dtype=np.int64).tolist())
        common = current if common is None else common.intersection(current)
    return common or set()


def _select_id_only(block: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Spread selection over sorted locked replicate IDs, never image outcomes."""

    ordered = block.sort_values(["replicate", "location"], kind="stable")
    if len(ordered) <= limit:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, limit).round().astype(int)
    return ordered.iloc[np.unique(positions)]


def prepare_contract(
    *,
    plism_root: str | Path = PLISM_ROOT,
    plism_analysis_root: str | Path = PLISM_ANALYSIS_ROOT,
    output_root: str | Path = OUTPUT_ROOT,
    input_scanners: Sequence[str] = (MODEL_SOURCE_SCANNER,),
    locations_per_core_section: int = DEFAULT_LOCATIONS_PER_CORE_SECTION,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze a PLISM-blind location set and scanner-specific render tasks."""

    if locations_per_core_section < 1:
        raise ValueError("locations_per_core_section must be positive")
    scanners = _normalize_input_scanners(input_scanners)
    plism_root = Path(plism_root).resolve()
    base_root = Path(plism_analysis_root).resolve()
    output = Path(output_root).resolve()
    contract_path = output / "00_contract/analysis_contract.json"
    if contract_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to replace frozen contract: {contract_path}")

    base_contract_path = base_root / "00_contract/analysis_contract.json"
    selected_source_path = base_root / "00_contract/selected_locations.csv"
    tasks_source_path = base_root / "00_contract/tasks.csv"
    for path in (base_contract_path, selected_source_path, tasks_source_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    base_contract = _read_json(base_contract_path)
    if base_contract.get("reference_scanner") != TARGET_SCANNER:
        raise ValueError("PLISM factorial contract does not use AT2 as reference")
    render = base_contract.get("render", {})
    if int(render.get("target_px", -1)) != TARGET_PX or not math.isclose(
        float(render.get("target_mpp", float("nan"))), TARGET_MPP, abs_tol=1e-9
    ):
        raise ValueError("PLISM factorial render grid differs from 256 px / 0.5052 um")

    selected_source = pd.read_csv(selected_source_path)
    required_selected = {
        "stain",
        "location",
        "core",
        "tissue_type",
        "replicate",
        "pannormal_tissue",
        "mapping_tier",
        "organ_aligned_status",
        "strict_status",
    }
    missing = required_selected - set(selected_source.columns)
    if missing:
        raise ValueError(f"locked PLISM locations lack columns: {sorted(missing)}")
    if selected_source[["stain", "location"]].duplicated().any():
        raise ValueError("locked PLISM locations contain duplicate section/location rows")

    feature_root = plism_root / "features/registered_AT2/uni_v2"
    eligible_parts = []
    feature_scanners = (TARGET_SCANNER, *scanners)
    for stain, block in selected_source.groupby("stain", sort=True):
        available = _available_feature_locations(feature_root, str(stain), feature_scanners)
        eligible_parts.append(block.loc[block["location"].astype(int).isin(available)])
    eligible = pd.concat(eligible_parts, ignore_index=True)
    selected_parts = [
        _select_id_only(block, int(locations_per_core_section))
        for _, block in eligible.groupby(["stain", "core"], sort=True)
    ]
    selected = pd.concat(selected_parts, ignore_index=True).sort_values(
        ["stain", "core", "replicate", "location"], kind="stable"
    )
    selected["selection_rule"] = (
        f"up_to_{int(locations_per_core_section)}_linspace_over_locked_replicate_location_ids"
    )
    if selected.empty:
        raise ValueError("no eligible PLISM locations remain")
    if selected.groupby(["stain", "core"]).size().max() > locations_per_core_section:
        raise AssertionError("location cap was not enforced")

    source_tasks = pd.read_csv(tasks_source_path)
    render_rows: list[dict[str, Any]] = []
    task_index = 0
    plan_root = output / "00_contract/render_task_plans"
    render_scanners = (TARGET_SCANNER, *scanners)
    for scanner in render_scanners:
        for stain in sorted(selected["stain"].astype(str).unique()):
            match = source_tasks.loc[
                source_tasks["scanner"].astype(str).eq(scanner)
                & source_tasks["stain"].astype(str).eq(stain)
            ]
            if len(match) != 1:
                raise ValueError(f"expected one frozen PLISM task for {stain}/{scanner}")
            source_task = match.iloc[0]
            source_plan_path = Path(str(source_task["plan_path"]))
            if sha256(source_plan_path) != str(source_task["plan_sha256"]):
                raise ValueError(f"frozen PLISM task plan hash changed: {source_plan_path}")
            plan = pd.read_csv(source_plan_path)
            wanted = selected.loc[selected["stain"].astype(str).eq(stain)]
            plan = plan.merge(
                wanted[
                    [
                        "stain",
                        "location",
                        "pannormal_tissue",
                        "mapping_tier",
                        "organ_aligned_status",
                        "strict_status",
                        "selection_rule",
                    ]
                ],
                on=["stain", "location"],
                how="inner",
                validate="one_to_one",
            ).sort_values("location", kind="stable")
            if len(plan) != len(wanted):
                raise ValueError(f"{stain}/{scanner}: render plan is not complete")
            plan_path = plan_root / f"{task_index:03d}.csv"
            _write_frame(plan, plan_path)
            wsi_path = plism_root / "original_wsi" / str(source_task["actual_file"])
            if not wsi_path.is_file():
                raise FileNotFoundError(wsi_path)
            render_rows.append(
                {
                    "task_index": task_index,
                    "stain": stain,
                    "render_scanner": scanner,
                    "render_role": "paired_target" if scanner == TARGET_SCANNER else "input",
                    "source_factorial_task_index": int(source_task["task_index"]),
                    "source_factorial_plan": str(source_plan_path.resolve()),
                    "source_factorial_plan_sha256": str(source_task["plan_sha256"]),
                    "plan_path": str(plan_path.resolve()),
                    "plan_sha256": sha256(plan_path),
                    "wsi_path": str(wsi_path.resolve()),
                    "wsi_bytes": int(wsi_path.stat().st_size),
                    "locations": int(len(plan)),
                }
            )
            task_index += 1
    render_tasks = pd.DataFrame(render_rows)

    # All historical raw/AT2 vectors must come from exactly one UNI2-h build.
    feature_metadata = []
    encoder_hashes: set[str] = set()
    for stain in sorted(selected["stain"].astype(str).unique()):
        for scanner in feature_scanners:
            path = feature_root / f"{stain}_{scanner}.h5"
            metadata = _feature_metadata(path)
            if metadata.get("encoder_id") != "uni_v2":
                raise ValueError(f"unexpected PLISM encoder in {path}: {metadata.get('encoder_id')}")
            if int(metadata["feature_dim_observed"]) != UNI2_FEATURE_DIM:
                raise ValueError(f"unexpected PLISM feature dimension in {path}")
            if not math.isclose(float(metadata.get("target_mpp", -1)), TARGET_MPP, abs_tol=1e-9):
                raise ValueError(f"feature/render MPP mismatch in {path}")
            if metadata.get("checkpoint_sha256"):
                encoder_hashes.add(str(metadata["checkpoint_sha256"]))
            feature_metadata.append(metadata)
    if len(encoder_hashes) != 1:
        raise ValueError(f"PLISM features do not share one UNI2-h checkpoint: {encoder_hashes}")
    encoder_hash = next(iter(encoder_hashes))
    uni1_checkpoint = _resolve_hf_checkpoint(UNI1_REPOSITORY, UNI1_FILENAME, None)
    uni1_checkpoint_hash = sha256(uni1_checkpoint)

    selected_path = output / "00_contract/selected_locations.csv"
    render_tasks_path = output / "00_contract/render_tasks.csv"
    predecessor_contract_path = PREDECESSOR_UNI_OUTPUT_ROOT / "00_contract/analysis_contract.json"
    predecessor_summary_path = PREDECESSOR_UNI_OUTPUT_ROOT / "03_aggregate/summary.json"
    predecessor_selected_path = PREDECESSOR_UNI_OUTPUT_ROOT / "00_contract/selected_locations.csv"
    predecessor_render_root = PREDECESSOR_UNI_OUTPUT_ROOT / "01_rendered_inputs"
    selection_matches_predecessor = False
    if predecessor_selected_path.is_file():
        predecessor_selected = pd.read_csv(predecessor_selected_path)
        selection_keys = ["stain", "location"]
        if all(column in predecessor_selected.columns for column in selection_keys):
            current_keys = selected[selection_keys].sort_values(selection_keys).reset_index(drop=True)
            previous_keys = (
                predecessor_selected[selection_keys]
                .sort_values(selection_keys)
                .reset_index(drop=True)
            )
            selection_matches_predecessor = current_keys.equals(previous_keys)
    required_predecessor_renders = [
        predecessor_render_root / f"{row.render_scanner}/{row.stain}.h5"
        for row in render_tasks.itertuples(index=False)
    ]
    reuse_predecessor_renders = bool(
        selection_matches_predecessor
        and all(
            path.is_file() and path.with_suffix(".summary.json").is_file()
            for path in required_predecessor_renders
        )
    )
    render_storage_root = (
        predecessor_render_root
        if reuse_predecessor_renders
        else output / "01_rendered_inputs"
    )
    _write_frame(selected, selected_path)
    _write_frame(render_tasks, render_tasks_path)
    contract = {
        "analysis_version": ANALYSIS_VERSION,
        "created_utc": utc_now(),
        "direction": "GT450->AT2",
        "predecessor_uni_v1_evaluation": {
            "root": str(PREDECESSOR_UNI_OUTPUT_ROOT),
            "relationship": (
                "read-only predecessor; this additive image-safety run never overwrites it"
            ),
            "analysis_contract_path": str(predecessor_contract_path),
            "analysis_contract_sha256": (
                sha256(predecessor_contract_path)
                if predecessor_contract_path.is_file()
                else None
            ),
            "aggregate_summary_path": str(predecessor_summary_path),
            "aggregate_summary_sha256": (
                sha256(predecessor_summary_path) if predecessor_summary_path.is_file() else None
            ),
            "uni_reproduction_audit": (
                "when predecessor all_location_metrics.csv.gz is available, aggregate compares "
                "the rerun by checkpoint_fold/source_scanner/stain/location and metric"
            ),
        },
        "primary_input_scanner": MODEL_SOURCE_SCANNER,
        "target_scanner": TARGET_SCANNER,
        "input_scanners": list(scanners),
        "primary_question": (
            "Do five frozen PanNormal GT450-to-AT2 Pix2Pix candidates reduce paired "
            "UNI-v1 distance to freshly rendered real PLISM AT2 without external fine-tuning?"
        ),
        "claim_boundary": (
            "PLISM shared/shared is ID-like only for scanner model and organ-level tissue "
            "membership. It remains external in physical device, site, section, stain, and "
            "acquisition. Non-GT450 inputs are exploratory model-boundary tests."
        ),
        "selection_boundary": (
            "Locations are selected only from the pre-existing locked PLISM factorial set "
            "by section/core and replicate/location IDs; no image, feature value, model "
            "output, or outcome is consulted. No post-generation location exclusion is allowed."
        ),
        "checkpoint_boundary": (
            "Every checkpoint must be the candidate named by an image-only PanNormal "
            "checkpoint_selection.json with uni_opened=false. All five outer-fold candidates "
            "are reported; PLISM outcomes cannot select among them. A diagnostic candidate "
            "that failed a predeclared image-fidelity gate remains diagnostic on PLISM."
        ),
        "inference_unit": "physical serial section (stain; 13 sections)",
        "core_role": "locations average to core first, then cores average within section",
        "cv_role": (
            "five overlapping PanNormal fits are model replicates, not five independent "
            "PLISM datasets; the ensemble is averaged within section before inference"
        ),
        "locations_per_core_section_max": int(locations_per_core_section),
        "selected_locations": int(len(selected)),
        "section_core_cells": int(selected.groupby(["stain", "core"]).ngroups),
        "sections": int(selected["stain"].nunique()),
        "render_tasks": int(len(render_tasks)),
        "input_render_tasks": int(render_tasks["render_role"].eq("input").sum()),
        "paired_target_render_tasks": int(
            render_tasks["render_role"].eq("paired_target").sum()
        ),
        "render": {
            "physical_fov_um": PATCH_UM,
            "target_px": TARGET_PX,
            "target_mpp": TARGET_MPP,
            "kernel": "libvips Lanczos3",
        },
        "render_storage": {
            "root": str(render_storage_root),
            "read_only_predecessor": bool(reuse_predecessor_renders),
            "selection_keys_match_predecessor": bool(selection_matches_predecessor),
            "relationship": (
                "reuse the immutable exact-grid v1 uint8 renders without copying or "
                "overwriting them"
                if reuse_predecessor_renders
                else "write v2 renders inside the additive sibling output root"
            ),
        },
        "primary_encoder": {
            "encoder_id": "uni_v1",
            "repository": UNI1_REPOSITORY,
            "feature_dim": UNI1_FEATURE_DIM,
            "checkpoint_path": str(uni1_checkpoint),
            "checkpoint_sha256": uni1_checkpoint_hash,
            "relationship_to_internal_endpoint": (
                "identical repository, weights, resize/normalization path, and feature dimension"
            ),
            "paired_endpoint_roles": (
                "fresh rendered source, generated output, and fresh rendered paired AT2 are "
                "embedded together; all three endpoint roles therefore use one encoder path"
            ),
        },
        "secondary_uni2": {
            "repository": UNI2_REPOSITORY,
            "feature_dim": UNI2_FEATURE_DIM,
            "checkpoint_sha256": encoder_hash,
            "existing_reference_root": str(feature_root.resolve()),
            "encoder_reproduction_check": (
                "when the optional sensitivity is enabled, freshly embedded source and AT2 "
                "are compared with their exact-location precomputed PLISM UNI2-h vectors"
            ),
            "role": "secondary encoder sensitivity only; never the internal-to-external primary",
        },
        "external_image_safety": {
            "evaluation_version": IMAGE_EVALUATION_VERSION,
            "metrics": list(IMAGE_METRICS),
            "edge_high_threshold": EDGE_HIGH_THRESHOLD,
            "edge_low_threshold": EDGE_LOW_THRESHOLD,
            "definitions": (
                "Exactly scanner_gan.evaluate_images.image_metrics on matched uint8 NHWC "
                "source, generated, and paired real AT2 patches."
            ),
            "gates": IMAGE_SAFETY_GATES,
            "aggregation": (
                "location -> core mean -> physical-section mean; checkpoint models are "
                "averaged within section before across-model inference"
            ),
            "selection_role": (
                "external diagnostic only; PLISM image metrics cannot select or rescue "
                "a frozen checkpoint"
            ),
            "at2_identity_control": (
                "excluded; G(AT2) remains a separate destructive identity-damage control"
            ),
        },
        "model_boundary_design": {
            "scanner_ID_like": "GT450 input only",
            "scanner_OOD": [scanner for scanner in scanners if scanner != MODEL_SOURCE_SCANNER],
            "tissue_ID_like": "organ_aligned_status == shared",
            "tissue_OOD": "organ_aligned_status == novel",
            "AT2_input": (
                "forbidden from closure and 2x2; allowed only for the separate "
                "d(G(AT2),AT2) identity-damage control"
            ),
        },
        "sources": {
            "plism_factorial_contract": {
                "path": str(base_contract_path),
                "sha256": sha256(base_contract_path),
            },
            "plism_factorial_selected_locations": {
                "path": str(selected_source_path),
                "sha256": sha256(selected_source_path),
            },
            "plism_factorial_tasks": {
                "path": str(tasks_source_path),
                "sha256": sha256(tasks_source_path),
            },
        },
        "outputs": {
            "selected_locations": {
                "path": str(selected_path),
                "sha256": sha256(selected_path),
            },
            "render_tasks": {
                "path": str(render_tasks_path),
                "sha256": sha256(render_tasks_path),
            },
        },
    }
    _write_json(contract_path, contract)
    return contract


def _vips_to_numpy(image: Any) -> np.ndarray:
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


def render_task(
    task_index: int,
    *,
    output_root: str | Path = OUTPUT_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render one frozen PLISM WSI task to an immutable image shard."""

    output = Path(output_root).resolve()
    contract_path = output / "00_contract/analysis_contract.json"
    contract = _read_json(contract_path)
    contract_hash = sha256(contract_path)
    if contract.get("analysis_version") != ANALYSIS_VERSION:
        raise ValueError("unsupported PLISM Pix2Pix contract")
    tasks_path = Path(contract["outputs"]["render_tasks"]["path"])
    if sha256(tasks_path) != contract["outputs"]["render_tasks"]["sha256"]:
        raise ValueError("render task table changed after contract freeze")
    tasks = pd.read_csv(tasks_path)
    match = tasks.loc[tasks["task_index"].astype(int).eq(int(task_index))]
    if len(match) != 1:
        raise KeyError(f"unknown render task {task_index}")
    task = match.iloc[0]
    scanner = str(task["render_scanner"])
    render_role = str(task["render_role"])
    stain = str(task["stain"])
    render_storage = contract.get("render_storage", {})
    render_root = Path(render_storage.get("root", output / "01_rendered_inputs"))
    read_only_predecessor = bool(render_storage.get("read_only_predecessor", False))
    destination = render_root / f"{scanner}/{stain}.h5"
    summary_path = destination.with_suffix(".summary.json")
    plan_path = Path(str(task["plan_path"]))
    if sha256(plan_path) != str(task["plan_sha256"]):
        raise ValueError(f"render plan changed: {plan_path}")
    plan = pd.read_csv(plan_path).sort_values("location", kind="stable")
    if read_only_predecessor:
        if not destination.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"read-only predecessor render is missing: {destination}")
        summary = _read_json(summary_path)
        if summary.get("output_sha256") != sha256(destination):
            raise ValueError(f"read-only predecessor render hash mismatch: {destination}")
        with h5py.File(destination, "r") as store:
            stored_locations = np.asarray(store["location"], dtype=np.int64)
            stored_shape = tuple(store["images"].shape)
            stored_dtype = store["images"].dtype
        wanted_locations = plan["location"].to_numpy(np.int64)
        if not np.array_equal(stored_locations, wanted_locations):
            raise ValueError(f"read-only predecessor locations differ: {destination}")
        if stored_shape != (len(plan), TARGET_PX, TARGET_PX, 3) or stored_dtype != np.uint8:
            raise ValueError(f"read-only predecessor image grid differs: {destination}")
        return {
            "status": "reused_read_only_predecessor",
            "analysis_version": ANALYSIS_VERSION,
            "contract_sha256": contract_hash,
            "output_path": str(destination),
            "output_sha256": summary["output_sha256"],
            "locations": int(len(plan)),
            "overwrite_ignored_to_protect_predecessor": bool(overwrite),
        }
    if destination.exists() and summary_path.exists() and not overwrite:
        summary = _read_json(summary_path)
        if (
            summary.get("analysis_version") == ANALYSIS_VERSION
            and summary.get("contract_sha256") == contract_hash
            and summary.get("output_sha256") == sha256(destination)
        ):
            return {"status": "already_complete", **summary}
        raise FileExistsError(f"stale render shard exists: {destination}")

    import pyvips

    wsi_path = Path(str(task["wsi_path"]))
    if int(wsi_path.stat().st_size) != int(task["wsi_bytes"]):
        raise ValueError(f"WSI size changed: {wsi_path}")
    image = pyvips.Image.new_from_file(str(wsi_path), level=0, access="random")
    if image.hasalpha():
        image = image.flatten(background=255)
    mpp_values = pd.to_numeric(plan["mpp"], errors="raise").to_numpy(float)
    if not np.allclose(mpp_values, mpp_values[0]):
        raise ValueError(f"{stain}/{scanner}: multiple native MPP values")
    native_mpp = float(mpp_values[0])
    side = int(round(PATCH_UM / native_mpp))
    rendered = np.empty((len(plan), TARGET_PX, TARGET_PX, 3), dtype=np.uint8)
    for index, row in enumerate(plan.itertuples(index=False)):
        left = int(round(float(row.centre_x))) - side // 2
        top = int(round(float(row.centre_y))) - side // 2
        if left < 0 or top < 0 or left + side > image.width or top + side > image.height:
            raise ValueError(f"{stain}/{scanner}/{row.location}: crop falls outside WSI")
        patch = image.crop(left, top, side, side)
        if str(row.flip).strip().lower() in {"true", "1"}:
            patch = patch.rot("d180")
        patch = patch.resize(
            TARGET_PX / patch.width,
            vscale=TARGET_PX / patch.height,
            kernel="lanczos3",
        )
        if patch.width != TARGET_PX or patch.height != TARGET_PX:
            patch = patch.thumbnail_image(
                TARGET_PX, height=TARGET_PX, size="force", kernel="lanczos3"
            )
        rendered[index] = _vips_to_numpy(patch)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    with h5py.File(temporary, "w") as store:
        store.attrs["analysis_version"] = ANALYSIS_VERSION
        store.attrs["contract_sha256"] = contract_hash
        store.attrs["input_scanner"] = scanner
        store.attrs["render_role"] = render_role
        store.attrs["stain"] = stain
        store.attrs["target_mpp"] = TARGET_MPP
        store.attrs["target_px"] = TARGET_PX
        store.attrs["native_mpp"] = native_mpp
        store.attrs["native_patch_px"] = side
        store.create_dataset(
            "images",
            data=rendered,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(1, TARGET_PX, TARGET_PX, 3),
        )
        for column in ("location", "core", "replicate"):
            store.create_dataset(column, data=plan[column].to_numpy(np.int64))
    temporary.replace(destination)
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "complete",
        "created_utc": utc_now(),
        "contract_sha256": contract_hash,
        "task_index": int(task_index),
        "stain": stain,
        "input_scanner": scanner,
        "render_role": render_role,
        "locations": int(len(plan)),
        "wsi_path": str(wsi_path),
        "wsi_bytes": int(wsi_path.stat().st_size),
        "plan_sha256": sha256(plan_path),
        "output_path": str(destination),
        "output_sha256": sha256(destination),
    }
    _write_json(summary_path, summary)
    return summary


def _validate_image_only_selection(
    checkpoint_path: Path,
    selection_path: Path,
    test_fold: int,
) -> dict[str, Any]:
    selection = _read_json(selection_path)
    if selection.get("selection_version") != "pix2pix_image_only_selection_v1":
        raise ValueError("checkpoint selection is not the locked image-only protocol")
    if bool(selection.get("uni_opened", True)):
        raise ValueError("checkpoint selection reports that UNI was opened")
    if str(selection.get("selection_data")) != "inner-validation image metrics only":
        raise ValueError("checkpoint was not selected only on inner-validation images")
    if str(selection.get("direction", "")).casefold() != "gt450->at2":
        raise ValueError("PLISM evaluation accepts only GT450->AT2 checkpoints")
    if int(selection.get("test_fold", -1)) != int(test_fold):
        raise ValueError("checkpoint selection fold differs from requested fold")
    selected_checkpoint = Path(str(selection.get("selected_checkpoint", ""))).resolve()
    if selected_checkpoint != checkpoint_path.resolve():
        raise ValueError(
            f"supplied checkpoint is not the image-only selected candidate: {selected_checkpoint}"
        )
    return selection


def _resolve_hf_checkpoint(
    repository: str,
    filename: str,
    path: str | Path | None,
) -> Path:
    if path is not None:
        result = Path(path).resolve()
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                f"huggingface_hub is required to locate the cached {repository} checkpoint"
            ) from error
        result = Path(
            hf_hub_download(
                repo_id=repository,
                filename=filename,
                local_files_only=True,
            )
        ).resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def load_frozen_uni1(
    device: Any,
    *,
    checkpoint_path: str | Path | None = None,
    expected_sha256: str | None = None,
) -> tuple[Any, int, Any, Any, Path]:
    """Load the exact 1024-d UNI-v1 evaluator used by the internal endpoint."""

    import timm
    import torch

    checkpoint = _resolve_hf_checkpoint(
        UNI1_REPOSITORY, UNI1_FILENAME, checkpoint_path
    )
    observed_hash = sha256(checkpoint)
    if expected_sha256 and observed_hash != expected_sha256:
        raise ValueError(
            f"UNI-v1 checkpoint hash mismatch: {observed_hash} != {expected_sha256}"
        )
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"incompatible UNI-v1 state: {incompatible}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    return model, 224, mean, std, checkpoint


def load_frozen_uni2(
    device: Any,
    *,
    checkpoint_path: str | Path | None = None,
    expected_sha256: str | None = None,
) -> tuple[Any, int, Any, Any, Path]:
    """Load the exact UNI2-h architecture/checkpoint used by PLISM."""

    import timm
    import torch

    checkpoint = _resolve_hf_checkpoint(
        UNI2_REPOSITORY, UNI2_FILENAME, checkpoint_path
    )
    observed_hash = sha256(checkpoint)
    if expected_sha256 and observed_hash != expected_sha256:
        raise ValueError(
            f"UNI2-h checkpoint hash mismatch: {observed_hash} != {expected_sha256}"
        )
    kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": UNI2_FEATURE_DIM,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model("vit_giant_patch14_224", pretrained=False, **kwargs)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"incompatible UNI2-h state: {incompatible}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    return model, 224, mean, std, checkpoint


def _autocast(device: Any, amp: str):
    import torch

    normalized = amp.strip().casefold()
    if normalized in {"none", "off", "float32"}:
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("mixed precision is supported only on CUDA")
    if normalized in {"auto", "bfloat16", "bf16"}:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if normalized in {"float16", "fp16"}:
        return torch.autocast("cuda", dtype=torch.float16)
    raise ValueError(f"unknown AMP mode {amp!r}")


def _generate_uint8(
    generator: Any,
    images: np.ndarray,
    device: Any,
    *,
    batch_size: int,
    amp: str,
) -> np.ndarray:
    import torch

    outputs = []
    for start in range(0, len(images), batch_size):
        value = torch.from_numpy(images[start : start + batch_size]).permute(0, 3, 1, 2)
        value = value.to(device=device, dtype=torch.float32).div(127.5).sub(1.0)
        with torch.inference_mode(), _autocast(device, amp):
            generated = generator(value)
        outputs.append(normalized_tensor_to_uint8(generated))
    return np.concatenate(outputs, axis=0)


def _embed_uni2(
    model: Any,
    images: np.ndarray,
    size: int,
    mean: Any,
    std: Any,
    device: Any,
    *,
    batch_size: int,
    amp: str,
) -> np.ndarray:
    """Apply the Trident UNI2-h Resize(224)/ImageNet normalization contract."""

    import torch
    import torch.nn.functional as functional

    outputs = []
    for start in range(0, len(images), batch_size):
        value = torch.from_numpy(images[start : start + batch_size]).permute(0, 3, 1, 2)
        value = value.to(device=device, dtype=torch.float32).div(255.0)
        # For a square 256 px input, Resize(224)+CenterCrop(224) is a direct
        # 224x224 resize.  Bilinear is torchvision Resize's historical default.
        value = functional.interpolate(
            value,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        with torch.inference_mode(), _autocast(device, amp):
            feature = model((value - mean) / std)
        feature = functional.normalize(feature.float(), dim=1)
        outputs.append(feature.cpu().numpy().astype(np.float32, copy=False))
    result = np.concatenate(outputs, axis=0)
    if result.shape != (len(images), UNI2_FEATURE_DIM) or not np.isfinite(result).all():
        raise ValueError(f"invalid UNI2-h output shape {result.shape}")
    return result


def _stored_features(
    feature_root: Path,
    stain: str,
    scanner: str,
    locations: np.ndarray,
) -> np.ndarray:
    path = feature_root / f"{stain}_{scanner}.h5"
    with h5py.File(path, "r") as store:
        positions = _positions_for_locations(
            np.asarray(store["location"], dtype=np.int64),
            np.asarray(locations, dtype=np.int64),
            f"{stain}/{scanner}",
        )
        result = np.asarray(store["features"][positions], dtype=np.float32)
    if result.shape != (len(locations), UNI2_FEATURE_DIM):
        raise ValueError(f"invalid stored feature shape in {path}: {result.shape}")
    return result


def _fold_output_root(output: Path, test_fold: int) -> Path:
    return output / f"02_model_evaluations/fold_{int(test_fold)}"


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    test_fold: int,
    *,
    output_root: str | Path = OUTPUT_ROOT,
    selection_path: str | Path | None = None,
    uni_checkpoint_path: str | Path | None = None,
    uni2_sensitivity: bool = False,
    uni2_checkpoint_path: str | Path | None = None,
    generator_batch_size: int = 16,
    uni_batch_size: int = 32,
    device: str = "cuda",
    amp: str = "bfloat16",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate one frozen CV checkpoint on all contract-defined PLISM inputs."""

    import torch

    if generator_batch_size < 1 or uni_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    output = Path(output_root).resolve()
    contract_path = output / "00_contract/analysis_contract.json"
    contract = _read_json(contract_path)
    contract_hash = sha256(contract_path)
    if contract.get("analysis_version") != ANALYSIS_VERSION:
        raise ValueError("unsupported PLISM Pix2Pix contract")
    selected_path = Path(contract["outputs"]["selected_locations"]["path"])
    tasks_path = Path(contract["outputs"]["render_tasks"]["path"])
    if sha256(selected_path) != contract["outputs"]["selected_locations"]["sha256"]:
        raise ValueError("selected PLISM locations changed after contract freeze")
    if sha256(tasks_path) != contract["outputs"]["render_tasks"]["sha256"]:
        raise ValueError("PLISM render tasks changed after contract freeze")

    checkpoint = Path(checkpoint_path).resolve()
    checkpoint_hash = sha256(checkpoint)
    selection = (
        checkpoint.parent.parent / "checkpoint_selection.json"
        if selection_path is None
        else Path(selection_path).resolve()
    )
    selection_payload = _validate_image_only_selection(
        checkpoint, selection, int(test_fold)
    )
    run_manifest_path = checkpoint.parent.parent / "run_manifest.json"
    run_manifest = _read_json(run_manifest_path)
    if run_manifest.get("status") != "complete":
        raise ValueError("PanNormal training run is not complete")
    if run_manifest.get("selection_boundary") != "image-only inner validation; UNI forbidden until lock":
        raise ValueError("PanNormal run lacks the UNI-blind selection boundary")

    fold_root = _fold_output_root(output, int(test_fold))
    manifest_path = fold_root / "evaluation_manifest.json"
    if manifest_path.exists() and not overwrite:
        manifest = _read_json(manifest_path)
        if (
            manifest.get("status") == "complete"
            and manifest.get("checkpoint_sha256") == checkpoint_hash
            and manifest.get("contract_sha256") == contract_hash
            and bool(manifest.get("uni2_sensitivity", False)) == bool(uni2_sensitivity)
        ):
            return {"status": "already_complete", **manifest}
        raise FileExistsError(f"stale fold evaluation exists: {fold_root}")

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    generator, checkpoint_metadata, _ = load_validated_generator(
        checkpoint,
        source_scanner="gt450",
        test_fold=int(test_fold),
        device=resolved_device,
    )
    uni_model, uni_size, uni_mean, uni_std, uni_checkpoint = load_frozen_uni1(
        resolved_device,
        checkpoint_path=uni_checkpoint_path,
        expected_sha256=str(contract["primary_encoder"]["checkpoint_sha256"]),
    )
    uni_checkpoint_hash = sha256(uni_checkpoint)
    if uni2_sensitivity:
        uni2_model, uni2_size, uni2_mean, uni2_std, uni2_checkpoint = load_frozen_uni2(
            resolved_device,
            checkpoint_path=uni2_checkpoint_path,
            expected_sha256=str(contract["secondary_uni2"]["checkpoint_sha256"]),
        )
        uni2_checkpoint_hash: str | None = sha256(uni2_checkpoint)
    else:
        uni2_model = uni2_size = uni2_mean = uni2_std = uni2_checkpoint = None
        uni2_checkpoint_hash = None

    selected = pd.read_csv(selected_path)
    tasks = pd.read_csv(tasks_path)
    render_root = Path(
        contract.get("render_storage", {}).get("root", output / "01_rendered_inputs")
    )
    feature_root = Path(contract["secondary_uni2"]["existing_reference_root"])
    all_metrics: list[pd.DataFrame] = []
    all_image_safety_metrics: list[pd.DataFrame] = []
    all_identity_metrics: list[pd.DataFrame] = []
    shard_rows: list[dict[str, Any]] = []
    expected_location_keys: set[tuple[str, str, int]] = set()

    input_tasks = tasks.loc[tasks["render_role"].astype(str).eq("input")]
    target_tasks = tasks.loc[tasks["render_role"].astype(str).eq("paired_target")]
    target_by_stain = {
        str(row.stain): row for row in target_tasks.itertuples(index=False)
    }
    if set(target_by_stain) != set(selected["stain"].astype(str).unique()):
        raise ValueError("paired AT2 render tasks do not cover every selected section")
    for task in input_tasks.sort_values("task_index").itertuples(index=False):
        scanner = str(task.render_scanner)
        stain = str(task.stain)
        render_path = render_root / f"{scanner}/{stain}.h5"
        render_summary_path = render_path.with_suffix(".summary.json")
        if not render_path.is_file() or not render_summary_path.is_file():
            raise FileNotFoundError(f"render task is incomplete: {render_path}")
        render_summary = _read_json(render_summary_path)
        if render_summary.get("output_sha256") != sha256(render_path):
            raise ValueError(f"render shard hash mismatch: {render_path}")
        with h5py.File(render_path, "r") as store:
            images = np.asarray(store["images"], dtype=np.uint8)
            locations = np.asarray(store["location"], dtype=np.int64)
        target_render_path = render_root / f"{TARGET_SCANNER}/{stain}.h5"
        target_summary_path = target_render_path.with_suffix(".summary.json")
        if not target_render_path.is_file() or not target_summary_path.is_file():
            raise FileNotFoundError(f"paired AT2 render task is incomplete: {target_render_path}")
        target_summary = _read_json(target_summary_path)
        if target_summary.get("output_sha256") != sha256(target_render_path):
            raise ValueError(f"paired AT2 render shard hash mismatch: {target_render_path}")
        with h5py.File(target_render_path, "r") as store:
            target_locations = np.asarray(store["location"], dtype=np.int64)
            at2_images = np.asarray(store["images"], dtype=np.uint8)
        if not np.array_equal(locations, target_locations):
            raise ValueError(f"{stain}/{scanner}: input and AT2 locations differ")
        metadata = (
            selected.loc[selected["stain"].astype(str).eq(stain)]
            .set_index("location")
            .reindex(locations)
            .reset_index()
        )
        if metadata["core"].isna().any():
            raise ValueError(f"{stain}/{scanner}: rendered locations lack frozen metadata")
        for location in locations:
            key = (scanner, stain, int(location))
            if key in expected_location_keys:
                raise ValueError(f"duplicate PLISM evaluation location: {key}")
            expected_location_keys.add(key)

        generated = _generate_uint8(
            generator,
            images,
            resolved_device,
            batch_size=int(generator_batch_size),
            amp=amp,
        )
        identity_generated_images: np.ndarray | None = None
        primary_image_blocks = [images, generated, at2_images]
        if scanner == MODEL_SOURCE_SCANNER:
            # Destructive target-domain control only. It is deliberately kept
            # out of closure and 2x2 pools: does G damage an already-AT2 image?
            identity_generated_images = _generate_uint8(
                generator,
                at2_images,
                resolved_device,
                batch_size=int(generator_batch_size),
                amp=amp,
            )
            primary_image_blocks.append(identity_generated_images)
        primary_features = embed_frozen_uni(
            uni_model,
            np.concatenate(primary_image_blocks, axis=0),
            uni_size,
            uni_mean,
            uni_std,
            resolved_device,
            batch_size=int(uni_batch_size),
            value_range="uint8",
        )
        if primary_features.shape[1] != UNI1_FEATURE_DIM:
            raise ValueError(
                f"primary UNI-v1 feature dimension {primary_features.shape[1]} "
                f"!= {UNI1_FEATURE_DIM}"
            )
        if identity_generated_images is None:
            fresh_source, generated_features, fresh_at2 = np.split(
                primary_features, [len(images), 2 * len(images)]
            )
            identity_generated_features = None
        else:
            (
                fresh_source,
                generated_features,
                fresh_at2,
                identity_generated_features,
            ) = np.split(
                primary_features,
                [len(images), 2 * len(images), 3 * len(images)],
            )

        metric_metadata = metadata.copy()
        metric_metadata["source_scanner"] = scanner
        metric_metadata["slide_id"] = stain
        metric_metadata["fold"] = int(test_fold)
        metric_metadata["location_index"] = metric_metadata["location"].astype(int)
        metric_metadata["shared_tissue"] = metric_metadata["organ_aligned_status"].eq("shared")
        metric_metadata = add_plism_model_status(
            metric_metadata,
            model_source_scanner=MODEL_SOURCE_SCANNER,
        )
        image_safety = metric_metadata.copy()
        image_safety["method"] = "pix2pix"
        image_safety["target_scanner"] = TARGET_SCANNER
        image_safety["checkpoint_fold"] = int(test_fold)
        image_safety["checkpoint_sha256"] = checkpoint_hash
        image_safety["image_evaluation_version"] = IMAGE_EVALUATION_VERSION
        measured_image_safety = image_metrics(images, generated, at2_images)
        for column in IMAGE_METRICS:
            image_safety[column] = measured_image_safety[column].to_numpy()
        metrics = paired_uni_metrics(
            fresh_source,
            generated_features,
            fresh_at2,
            metric_metadata,
            method="pix2pix",
            target_scanner=TARGET_SCANNER,
        )
        metrics["checkpoint_fold"] = int(test_fold)
        metrics["checkpoint_sha256"] = checkpoint_hash
        metrics["primary_encoder"] = "UNI-v1"
        if identity_generated_features is not None:
            identity = metadata[
                [
                    "stain",
                    "location",
                    "core",
                    "tissue_type",
                    "replicate",
                    "organ_aligned_status",
                    "strict_status",
                ]
            ].copy()
            identity["checkpoint_fold"] = int(test_fold)
            identity["checkpoint_sha256"] = checkpoint_hash
            identity["control"] = "AT2_through_GT450_to_AT2_generator"
            identity["input_scanner"] = TARGET_SCANNER
            identity["target_scanner"] = TARGET_SCANNER
            identity["included_in_closure_or_2x2"] = False
            identity["at2_identity_damage_distance"] = cosine_distance_rows(
                identity_generated_features, fresh_at2
            )
            all_identity_metrics.append(identity)

        uni2_values: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if uni2_sensitivity:
            assert uni2_model is not None
            assert uni2_size is not None and uni2_mean is not None and uni2_std is not None
            secondary = _embed_uni2(
                uni2_model,
                np.concatenate((images, generated, at2_images), axis=0),
                int(uni2_size),
                uni2_mean,
                uni2_std,
                resolved_device,
                batch_size=int(uni_batch_size),
                amp=amp,
            )
            uni2_source, uni2_generated, uni2_at2 = np.split(
                secondary, [len(images), 2 * len(images)]
            )
            secondary_metrics = paired_uni_metrics(
                uni2_source,
                uni2_generated,
                uni2_at2,
                metric_metadata,
                method="pix2pix_uni2_sensitivity",
                target_scanner=TARGET_SCANNER,
            )
            for column in METRICS:
                metrics[f"uni2_{column}"] = secondary_metrics[column].to_numpy()
            stored_source = _stored_features(feature_root, stain, scanner, locations)
            stored_at2 = _stored_features(feature_root, stain, TARGET_SCANNER, locations)
            metrics["uni2_fresh_source_to_stored_source_distance"] = cosine_distance_rows(
                uni2_source, stored_source
            )
            metrics["uni2_fresh_at2_to_stored_at2_distance"] = cosine_distance_rows(
                uni2_at2, stored_at2
            )
            uni2_values = (uni2_source, uni2_generated, uni2_at2)

        shard_dir = fold_root / f"feature_shards/{scanner}"
        shard_path = shard_dir / f"{stain}.h5"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = shard_path.with_suffix(shard_path.suffix + f".{os.getpid()}.tmp")
        if temporary.exists():
            temporary.unlink()
        with h5py.File(temporary, "w") as store:
            store.attrs["analysis_version"] = ANALYSIS_VERSION
            store.attrs["contract_sha256"] = contract_hash
            store.attrs["checkpoint_sha256"] = checkpoint_hash
            store.attrs["checkpoint_fold"] = int(test_fold)
            store.attrs["input_scanner"] = scanner
            store.attrs["stain"] = stain
            store.attrs["encoder"] = "UNI-v1"
            store.attrs["encoder_checkpoint_sha256"] = uni_checkpoint_hash
            store.create_dataset("location", data=locations)
            store.create_dataset("fresh_source", data=fresh_source, compression="lzf")
            store.create_dataset("generated", data=generated_features, compression="lzf")
            store.create_dataset("real_at2", data=fresh_at2, compression="lzf")
            if identity_generated_features is not None:
                controls = store.create_group("controls")
                controls.attrs["included_in_closure_or_2x2"] = False
                controls.create_dataset(
                    "at2_through_generator",
                    data=identity_generated_features,
                    compression="lzf",
                )
            if uni2_values is not None:
                group = store.create_group("uni2_sensitivity")
                group.attrs["encoder_checkpoint_sha256"] = str(uni2_checkpoint_hash)
                group.create_dataset("fresh_source", data=uni2_values[0], compression="lzf")
                group.create_dataset("generated", data=uni2_values[1], compression="lzf")
                group.create_dataset("real_at2", data=uni2_values[2], compression="lzf")
        temporary.replace(shard_path)

        metric_path = fold_root / f"location_shards/{scanner}/{stain}.csv.gz"
        image_safety_path = (
            fold_root / f"image_safety_location_shards/{scanner}/{stain}.csv.gz"
        )
        _write_frame(metrics, metric_path)
        _write_frame(image_safety, image_safety_path)
        shard_rows.append(
            {
                "input_scanner": scanner,
                "stain": stain,
                "locations": int(len(locations)),
                "render_path": str(render_path),
                "render_sha256": sha256(render_path),
                "feature_path": str(shard_path),
                "feature_sha256": sha256(shard_path),
                "metric_path": str(metric_path),
                "metric_sha256": sha256(metric_path),
                "image_safety_path": str(image_safety_path),
                "image_safety_sha256": sha256(image_safety_path),
                "uni2_sensitivity": bool(uni2_sensitivity),
            }
        )
        all_metrics.append(metrics)
        all_image_safety_metrics.append(image_safety)

    location_metrics = pd.concat(all_metrics, ignore_index=True)
    expected = len(selected) * len(contract["input_scanners"])
    if len(location_metrics) != expected:
        raise ValueError(f"evaluated {len(location_metrics)} rows, expected {expected}")
    location_path = fold_root / "location_metrics.csv.gz"
    _write_frame(location_metrics, location_path)
    image_safety_metrics = pd.concat(all_image_safety_metrics, ignore_index=True)
    if len(image_safety_metrics) != expected:
        raise ValueError(
            f"image-safety evaluation has {len(image_safety_metrics)} rows, expected {expected}"
        )
    key_columns = ["source_scanner", "stain", "location"]
    uni_keys = pd.MultiIndex.from_frame(location_metrics[key_columns])
    safety_keys = pd.MultiIndex.from_frame(image_safety_metrics[key_columns])
    if not uni_keys.equals(safety_keys):
        raise ValueError("image-safety rows do not align exactly with UNI location rows")
    image_safety_path = fold_root / "image_safety_location_metrics.csv.gz"
    _write_frame(image_safety_metrics, image_safety_path)
    if len(all_identity_metrics) != int(selected["stain"].nunique()):
        raise ValueError("AT2 identity-damage control was not measured once per section")
    identity_metrics = pd.concat(all_identity_metrics, ignore_index=True)
    if len(identity_metrics) != len(selected):
        raise ValueError("AT2 identity-damage control does not cover every selected location")
    identity_path = fold_root / "at2_identity_damage_location_metrics.csv.gz"
    _write_frame(identity_metrics, identity_path)
    collapse_rows = []
    for scanner, positions in location_metrics.groupby("source_scanner", sort=True).indices.items():
        # Read the generated shards back in the same stain/location order used
        # by the metric table.  This also verifies persisted feature integrity.
        generated_blocks = []
        at2_blocks = []
        ordered = location_metrics.iloc[np.asarray(positions)].sort_values(
            ["stain", "location"], kind="stable"
        )
        for stain, block in ordered.groupby("stain", sort=True):
            row = next(
                item for item in shard_rows
                if item["input_scanner"] == scanner and item["stain"] == stain
            )
            with h5py.File(row["feature_path"], "r") as store:
                shard_locations = np.asarray(store["location"], dtype=np.int64)
                shard_features = np.asarray(store["generated"], dtype=np.float32)
                shard_at2 = np.asarray(store["real_at2"], dtype=np.float32)
            wanted = block["location"].to_numpy(np.int64)
            shard_positions = _positions_for_locations(shard_locations, wanted, "persisted generated")
            generated_blocks.append(shard_features[shard_positions])
            at2_blocks.append(shard_at2[shard_positions])
        diagnostics = embedding_collapse_diagnostics(
            np.concatenate(generated_blocks), np.concatenate(at2_blocks)
        )
        collapse_rows.append({"input_scanner": scanner, **diagnostics})
    collapse_path = fold_root / "collapse_diagnostics.csv"
    _write_frame(pd.DataFrame(collapse_rows), collapse_path)

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "complete",
        "created_utc": utc_now(),
        "contract_sha256": contract_hash,
        "checkpoint_fold": int(test_fold),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_metadata": checkpoint_metadata,
        "selection_path": str(selection),
        "selection_sha256": sha256(selection),
        "selection_status": selection_payload.get("selection_status"),
        "analysis_role": (
            "diagnostic_only"
            if selection_payload.get("selection_status")
            != "provisional_image_candidate"
            else "provisional_external_evaluation"
        ),
        "selection_gates": selection_payload.get("gates", {}),
        "selection_selected": selection_payload.get("selected", {}),
        "basic_eligible_checkpoint_count": int(
            selection_payload.get("basic_eligible_checkpoint_count", 0)
        ),
        "run_manifest_path": str(run_manifest_path),
        "run_manifest_sha256": sha256(run_manifest_path),
        "input_scanners": list(contract["input_scanners"]),
        "locations": int(len(location_metrics)),
        "sections": int(location_metrics["stain"].nunique()),
        "primary_encoder": "UNI-v1 (MahmoodLab/uni), 1024-d",
        "primary_encoder_checkpoint_path": str(uni_checkpoint),
        "primary_encoder_checkpoint_sha256": uni_checkpoint_hash,
        "uni2_sensitivity": bool(uni2_sensitivity),
        "uni2_checkpoint_path": (
            str(uni2_checkpoint) if uni2_checkpoint is not None else None
        ),
        "uni2_checkpoint_sha256": uni2_checkpoint_hash,
        "no_finetuning": True,
        "no_outcome_filtering": True,
        "location_metrics": {
            "path": str(location_path),
            "sha256": sha256(location_path),
        },
        "image_safety_metrics": {
            "path": str(image_safety_path),
            "sha256": sha256(image_safety_path),
            "evaluation_version": IMAGE_EVALUATION_VERSION,
            "metrics": list(IMAGE_METRICS),
            "included_in_checkpoint_selection": False,
            "at2_identity_control_included": False,
        },
        "at2_identity_damage_control": {
            "path": str(identity_path),
            "sha256": sha256(identity_path),
            "definition": "d_UNI-v1(G(AT2), AT2)",
            "included_in_closure_or_2x2": False,
        },
        "collapse_diagnostics": {
            "path": str(collapse_path),
            "sha256": sha256(collapse_path),
        },
        "shards": shard_rows,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _bootstrap_mean(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    draws = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _section_tables(
    location: pd.DataFrame,
    *,
    metric_columns: Sequence[str] = METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identifiers = [
        "checkpoint_fold",
        "source_scanner",
        "scanner_status",
        "tissue_status",
        "quadrant",
        "stain",
        "core",
    ]
    metrics = list(metric_columns)
    core = location.groupby(identifiers, as_index=False, dropna=False)[metrics].mean()
    section_identifiers = [item for item in identifiers if item != "core"]
    section = core.groupby(section_identifiers, as_index=False, dropna=False)[metrics].mean()
    return core, section


def _summary_by_model(
    section: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
    metric_columns: Sequence[str] = METRICS,
) -> pd.DataFrame:
    rows = []
    groups = ["checkpoint_fold", "source_scanner", "quadrant"]
    for key, block in section.groupby(groups, sort=True, dropna=False):
        identity = dict(zip(groups, key))
        for metric_index, metric in enumerate(metric_columns):
            estimate, low, high = _bootstrap_mean(
                block[metric].to_numpy(float),
                replicates=bootstrap_replicates,
                seed=seed + 1000 * int(identity["checkpoint_fold"]) + metric_index,
            )
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "sections": int(block["stain"].nunique()),
                    "bootstrap_replicates": int(bootstrap_replicates),
                }
            )
    return pd.DataFrame(rows)


def _summary_across_models(
    section: pd.DataFrame,
    model_summary: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
    metric_columns: Sequence[str] = METRICS,
) -> pd.DataFrame:
    # Average the five model replicates within each physical section first.
    ensemble = section.groupby(
        ["source_scanner", "quadrant", "stain"], as_index=False, dropna=False
    )[list(metric_columns)].mean()
    rows = []
    for key, block in ensemble.groupby(
        ["source_scanner", "quadrant"], sort=True, dropna=False
    ):
        scanner, quadrant = key
        for metric_index, metric in enumerate(metric_columns):
            estimate, low, high = _bootstrap_mean(
                block[metric].to_numpy(float),
                replicates=bootstrap_replicates,
                seed=seed + 10_000 + metric_index,
            )
            model_values = model_summary.loc[
                model_summary["source_scanner"].eq(scanner)
                & model_summary["quadrant"].eq(quadrant)
                & model_summary["metric"].eq(metric),
                "estimate",
            ].to_numpy(float)
            rows.append(
                {
                    "source_scanner": scanner,
                    "quadrant": quadrant,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "sections": int(block["stain"].nunique()),
                    "models": int(len(model_values)),
                    "model_estimate_min": float(np.nanmin(model_values)),
                    "model_estimate_max": float(np.nanmax(model_values)),
                    "model_estimate_sd": float(np.nanstd(model_values, ddof=1)),
                    "models_positive": int(np.sum(model_values > 0)),
                    "models_negative": int(np.sum(model_values < 0)),
                    "model_sign_consistent": bool(
                        np.all(model_values > 0) or np.all(model_values < 0)
                    ),
                    "bootstrap_replicates": int(bootstrap_replicates),
                }
            )
    return pd.DataFrame(rows)


def _image_safety_overall_scanner_summary(
    location: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize image safety without letting dense cores or CV fits dominate."""

    core = location.groupby(
        ["checkpoint_fold", "source_scanner", "stain", "core"],
        as_index=False,
        dropna=False,
    )[list(IMAGE_METRICS)].mean()
    section = core.groupby(
        ["checkpoint_fold", "source_scanner", "stain"],
        as_index=False,
        dropna=False,
    )[list(IMAGE_METRICS)].mean()
    ensemble = section.groupby(
        ["source_scanner", "stain"], as_index=False, dropna=False
    )[list(IMAGE_METRICS)].mean()
    rows = []
    for scanner, block in ensemble.groupby("source_scanner", sort=True, dropna=False):
        model_block = section.loc[section["source_scanner"].eq(scanner)]
        for metric_index, metric in enumerate(IMAGE_METRICS):
            estimate, low, high = _bootstrap_mean(
                block[metric].to_numpy(float),
                replicates=bootstrap_replicates,
                seed=seed + 70_000 + metric_index,
            )
            model_values = model_block.groupby("checkpoint_fold", sort=True)[metric].mean()
            rows.append(
                {
                    "source_scanner": scanner,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "sections": int(block["stain"].nunique()),
                    "models": int(len(model_values)),
                    "model_estimate_min": float(model_values.min()),
                    "model_estimate_max": float(model_values.max()),
                    "model_estimate_sd": float(model_values.std(ddof=1)),
                    "bootstrap_replicates": int(bootstrap_replicates),
                }
            )
    return pd.DataFrame(rows)


def _external_image_safety_gates(overall: pd.DataFrame) -> pd.DataFrame:
    """Apply internal thresholds descriptively; PLISM cannot select checkpoints."""

    rows = []
    for scanner, scanner_block in overall.groupby("source_scanner", sort=True):
        lookup = scanner_block.set_index("metric")
        for metric, gate in IMAGE_SAFETY_GATES.items():
            if metric not in lookup.index:
                raise ValueError(f"missing external image-safety metric: {metric}")
            estimate = float(lookup.loc[metric, "estimate"])
            threshold = float(gate["threshold"])
            direction = str(gate["direction"])
            passed = estimate >= threshold if direction == "minimum" else estimate <= threshold
            rows.append(
                {
                    "source_scanner": scanner,
                    "metric": metric,
                    "direction": direction,
                    "threshold": threshold,
                    "estimate": estimate,
                    "pass": bool(passed),
                    "role": (
                        "clean_GT450_external_replication"
                        if scanner == MODEL_SOURCE_SCANNER
                        else "exploratory_model_boundary"
                    ),
                    "selection_use": "diagnostic_only_never_selects_checkpoint",
                }
            )
    return pd.DataFrame(rows)


def _audit_predecessor_uni(
    current: pd.DataFrame,
    predecessor_root: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Compare the additive rerun with v1 without mutating predecessor files."""

    predecessor_path = Path(predecessor_root) / "03_aggregate/all_location_metrics.csv.gz"
    if not predecessor_path.is_file():
        return (
            {
                "status": "predecessor_location_metrics_not_available",
                "predecessor_path": str(predecessor_path),
                "predecessor_read_only": True,
            },
            None,
        )
    previous = pd.read_csv(predecessor_path)
    keys = ["checkpoint_fold", "source_scanner", "stain", "location"]
    missing_keys = [column for column in keys if column not in previous.columns]
    missing_metrics = [column for column in METRICS if column not in previous.columns]
    if missing_keys or missing_metrics:
        return (
            {
                "status": "predecessor_schema_incompatible",
                "predecessor_path": str(predecessor_path),
                "predecessor_sha256": sha256(predecessor_path),
                "missing_keys": missing_keys,
                "missing_metrics": missing_metrics,
                "predecessor_read_only": True,
            },
            None,
        )
    if current.duplicated(keys).any() or previous.duplicated(keys).any():
        return (
            {
                "status": "duplicate_audit_keys",
                "predecessor_path": str(predecessor_path),
                "predecessor_sha256": sha256(predecessor_path),
                "current_duplicate_keys": int(current.duplicated(keys).sum()),
                "predecessor_duplicate_keys": int(previous.duplicated(keys).sum()),
                "predecessor_read_only": True,
            },
            None,
        )
    merged = current[keys + list(METRICS)].merge(
        previous[keys + list(METRICS)],
        on=keys,
        how="outer",
        suffixes=("_v2", "_v1"),
        indicator=True,
        validate="one_to_one",
    )
    common = merged.loc[merged["_merge"].eq("both")]
    rows = []
    for metric in METRICS:
        current_values = common[f"{metric}_v2"].to_numpy(float)
        previous_values = common[f"{metric}_v1"].to_numpy(float)
        finite = np.isfinite(current_values) & np.isfinite(previous_values)
        differences = np.abs(current_values[finite] - previous_values[finite])
        rows.append(
            {
                "metric": metric,
                "finite_pairs": int(finite.sum()),
                "both_nan": int((np.isnan(current_values) & np.isnan(previous_values)).sum()),
                "mean_absolute_difference": (
                    float(differences.mean()) if len(differences) else np.nan
                ),
                "maximum_absolute_difference": (
                    float(differences.max()) if len(differences) else np.nan
                ),
                "exact_or_both_nan": bool(
                    np.all((current_values == previous_values) | (
                        np.isnan(current_values) & np.isnan(previous_values)
                    ))
                ),
            }
        )
    audit = pd.DataFrame(rows)
    payload = {
        "status": "compared",
        "predecessor_path": str(predecessor_path),
        "predecessor_sha256": sha256(predecessor_path),
        "predecessor_read_only": True,
        "comparison_keys": keys,
        "current_rows": int(len(current)),
        "predecessor_rows": int(len(previous)),
        "overlap_rows": int(len(common)),
        "current_only_rows": int(merged["_merge"].eq("left_only").sum()),
        "predecessor_only_rows": int(merged["_merge"].eq("right_only").sum()),
        "same_key_set": bool(merged["_merge"].eq("both").all()),
        "all_metrics_exact_or_both_nan": bool(audit["exact_or_both_nan"].all()),
    }
    return payload, audit


def _feature_consistency(
    manifests: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for left, right in itertools.combinations(manifests, 2):
        left_lookup = {
            (item["input_scanner"], item["stain"]): item for item in left["shards"]
        }
        right_lookup = {
            (item["input_scanner"], item["stain"]): item for item in right["shards"]
        }
        if set(left_lookup) != set(right_lookup):
            raise ValueError("model manifests do not cover identical PLISM render tasks")
        for key in sorted(left_lookup):
            scanner, stain = key
            with h5py.File(left_lookup[key]["feature_path"], "r") as store:
                left_locations = np.asarray(store["location"], dtype=np.int64)
                left_features = np.asarray(store["generated"], dtype=np.float32)
            with h5py.File(right_lookup[key]["feature_path"], "r") as store:
                right_locations = np.asarray(store["location"], dtype=np.int64)
                right_features = np.asarray(store["generated"], dtype=np.float32)
            if not np.array_equal(left_locations, right_locations):
                raise ValueError(f"model feature locations differ for {scanner}/{stain}")
            distance = cosine_distance_rows(left_features, right_features)
            rows.append(
                {
                    "left_fold": int(left["checkpoint_fold"]),
                    "right_fold": int(right["checkpoint_fold"]),
                    "source_scanner": scanner,
                    "stain": stain,
                    "locations": int(len(distance)),
                    "mean_generated_embedding_cosine": float(1.0 - distance.mean()),
                    "minimum_generated_embedding_cosine": float(1.0 - distance.max()),
                    "mean_generated_embedding_distance": float(distance.mean()),
                }
            )
    return pd.DataFrame(rows)


def aggregate(
    *,
    output_root: str | Path = OUTPUT_ROOT,
    expected_models: int = 5,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Aggregate CV replicates with physical section as the inference unit."""

    if expected_models < 1 or bootstrap_replicates < 1:
        raise ValueError("expected_models and bootstrap_replicates must be positive")
    output = Path(output_root).resolve()
    contract_path = output / "00_contract/analysis_contract.json"
    contract = _read_json(contract_path)
    contract_hash = sha256(contract_path)
    if contract.get("analysis_version") != ANALYSIS_VERSION:
        raise ValueError("unsupported PLISM Pix2Pix contract")
    manifest_paths = sorted((output / "02_model_evaluations").glob("fold_*/evaluation_manifest.json"))
    manifests = [_read_json(path) for path in manifest_paths]
    if len(manifests) != expected_models:
        raise ValueError(f"found {len(manifests)} completed models, expected {expected_models}")
    folds = [int(item["checkpoint_fold"]) for item in manifests]
    selection_statuses = sorted({str(item.get("selection_status", "unknown")) for item in manifests})
    if len(set(folds)) != len(folds):
        raise ValueError(f"duplicate checkpoint folds: {folds}")
    if expected_models == 5 and set(folds) != set(range(5)):
        raise ValueError(f"five-model report requires folds 0..4, found {folds}")

    frames = []
    image_safety_frames = []
    identity_frames = []
    for path, manifest in zip(manifest_paths, manifests):
        if manifest.get("status") != "complete":
            raise ValueError(f"incomplete model manifest: {path}")
        if manifest.get("analysis_version") != ANALYSIS_VERSION:
            raise ValueError(f"model uses another PLISM evaluation version: {path}")
        if manifest.get("contract_sha256") != contract_hash:
            raise ValueError(f"model evaluated under another contract: {path}")
        metrics = manifest["location_metrics"]
        metrics_path = Path(metrics["path"])
        if sha256(metrics_path) != metrics["sha256"]:
            raise ValueError(f"model metric hash mismatch: {metrics_path}")
        frames.append(pd.read_csv(metrics_path))
        image_safety = manifest["image_safety_metrics"]
        if image_safety.get("evaluation_version") != IMAGE_EVALUATION_VERSION:
            raise ValueError(f"model uses another image-safety definition: {path}")
        image_safety_path = Path(image_safety["path"])
        if sha256(image_safety_path) != image_safety["sha256"]:
            raise ValueError(f"image-safety metric hash mismatch: {image_safety_path}")
        image_safety_frames.append(pd.read_csv(image_safety_path))
        control = manifest["at2_identity_damage_control"]
        control_path = Path(control["path"])
        if sha256(control_path) != control["sha256"]:
            raise ValueError(f"AT2 identity-control hash mismatch: {control_path}")
        identity_frames.append(pd.read_csv(control_path))
    location = pd.concat(frames, ignore_index=True)
    expected_rows = int(contract["selected_locations"]) * len(contract["input_scanners"]) * expected_models
    if len(location) != expected_rows:
        raise ValueError(f"aggregate has {len(location)} rows, expected {expected_rows}")
    image_safety_location = pd.concat(image_safety_frames, ignore_index=True)
    if len(image_safety_location) != expected_rows:
        raise ValueError(
            f"aggregate image safety has {len(image_safety_location)} rows, "
            f"expected {expected_rows}"
        )
    aggregate_keys = ["checkpoint_fold", "source_scanner", "stain", "location"]
    uni_keys = location[aggregate_keys].sort_values(aggregate_keys).reset_index(drop=True)
    safety_keys = (
        image_safety_location[aggregate_keys]
        .sort_values(aggregate_keys)
        .reset_index(drop=True)
    )
    if not uni_keys.equals(safety_keys):
        raise ValueError("aggregate UNI and image-safety location keys differ")
    identity_location = pd.concat(identity_frames, ignore_index=True)
    expected_identity_rows = int(contract["selected_locations"]) * expected_models
    if len(identity_location) != expected_identity_rows:
        raise ValueError(
            f"AT2 identity control has {len(identity_location)} rows, "
            f"expected {expected_identity_rows}"
        )
    leaked_identity = identity_location["included_in_closure_or_2x2"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    if leaked_identity.any():
        raise ValueError("AT2 identity-damage control leaked into a closure/2x2 pool")

    result_root = output / "03_aggregate"
    location_path = result_root / "all_location_metrics.csv.gz"
    _write_frame(location, location_path)
    core, section = _section_tables(location)
    core_path = result_root / "core_section_metrics.csv"
    section_path = result_root / "section_metrics.csv"
    _write_frame(core, core_path)
    _write_frame(section, section_path)
    per_model = _summary_by_model(
        section, bootstrap_replicates=bootstrap_replicates, seed=seed
    )
    per_model_path = result_root / "per_model_summary.csv"
    _write_frame(per_model, per_model_path)
    across = _summary_across_models(
        section,
        per_model,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    across_path = result_root / "across_model_summary.csv"
    _write_frame(across, across_path)
    consistency = _feature_consistency(manifests)
    consistency_path = result_root / "model_pair_consistency.csv"
    _write_frame(consistency, consistency_path)

    image_safety_location_path = result_root / "image_safety_location_metrics.csv.gz"
    _write_frame(image_safety_location, image_safety_location_path)
    image_safety_core, image_safety_section = _section_tables(
        image_safety_location,
        metric_columns=IMAGE_METRICS,
    )
    image_safety_core_path = result_root / "image_safety_core_section_metrics.csv"
    image_safety_section_path = result_root / "image_safety_section_metrics.csv"
    _write_frame(image_safety_core, image_safety_core_path)
    _write_frame(image_safety_section, image_safety_section_path)
    image_safety_per_model = _summary_by_model(
        image_safety_section,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 70_000,
        metric_columns=IMAGE_METRICS,
    )
    image_safety_per_model_path = result_root / "image_safety_per_model_summary.csv"
    _write_frame(image_safety_per_model, image_safety_per_model_path)
    image_safety_across = _summary_across_models(
        image_safety_section,
        image_safety_per_model,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 80_000,
        metric_columns=IMAGE_METRICS,
    )
    image_safety_across_path = result_root / "image_safety_across_model_summary.csv"
    _write_frame(image_safety_across, image_safety_across_path)
    image_safety_overall = _image_safety_overall_scanner_summary(
        image_safety_location,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    image_safety_overall_path = result_root / "image_safety_overall_scanner_summary.csv"
    _write_frame(image_safety_overall, image_safety_overall_path)
    image_safety_gates = _external_image_safety_gates(image_safety_overall)
    image_safety_gates_path = result_root / "image_safety_external_gates.csv"
    _write_frame(image_safety_gates, image_safety_gates_path)

    identity_location_path = result_root / "at2_identity_damage_location_metrics.csv.gz"
    _write_frame(identity_location, identity_location_path)
    identity_core = identity_location.groupby(
        ["checkpoint_fold", "stain", "core"], as_index=False
    )["at2_identity_damage_distance"].mean()
    identity_section = identity_core.groupby(
        ["checkpoint_fold", "stain"], as_index=False
    )["at2_identity_damage_distance"].mean()
    identity_section_path = result_root / "at2_identity_damage_section_metrics.csv"
    _write_frame(identity_section, identity_section_path)
    identity_model_rows = []
    for fold, block in identity_section.groupby("checkpoint_fold", sort=True):
        estimate, low, high = _bootstrap_mean(
            block["at2_identity_damage_distance"].to_numpy(float),
            replicates=bootstrap_replicates,
            seed=seed + 30_000 + int(fold),
        )
        identity_model_rows.append(
            {
                "checkpoint_fold": int(fold),
                "metric": "at2_identity_damage_distance",
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "sections": int(block["stain"].nunique()),
                "included_in_closure_or_2x2": False,
            }
        )
    identity_per_model = pd.DataFrame(identity_model_rows)
    identity_per_model_path = result_root / "at2_identity_damage_per_model_summary.csv"
    _write_frame(identity_per_model, identity_per_model_path)
    identity_ensemble_section = identity_section.groupby("stain", as_index=False)[
        "at2_identity_damage_distance"
    ].mean()
    identity_estimate, identity_low, identity_high = _bootstrap_mean(
        identity_ensemble_section["at2_identity_damage_distance"].to_numpy(float),
        replicates=bootstrap_replicates,
        seed=seed + 40_000,
    )
    identity_across = {
        "metric": "at2_identity_damage_distance",
        "definition": "d_UNI-v1(G(AT2), AT2)",
        "estimate": identity_estimate,
        "ci_low": identity_low,
        "ci_high": identity_high,
        "sections": int(identity_ensemble_section["stain"].nunique()),
        "models": len(manifests),
        "model_estimate_min": float(identity_per_model["estimate"].min()),
        "model_estimate_max": float(identity_per_model["estimate"].max()),
        "included_in_closure_or_2x2": False,
    }
    identity_across_path = result_root / "at2_identity_damage_across_models.json"
    _write_json(identity_across_path, identity_across)

    checkpoint_rows = []
    for manifest in manifests:
        selected_candidate = manifest.get("selection_selected", {})
        gates = manifest.get("selection_gates", {})
        checkpoint_rows.append(
            {
                "checkpoint_fold": int(manifest["checkpoint_fold"]),
                "selection_status": str(manifest.get("selection_status", "unknown")),
                "basic_eligible_checkpoint_count": int(
                    manifest.get("basic_eligible_checkpoint_count", 0)
                ),
                "minimum_source_gradient_ncc": gates.get(
                    "minimum_source_gradient_ncc", np.nan
                ),
                "maximum_saturation_fraction": gates.get(
                    "maximum_saturation_fraction", np.nan
                ),
                "validation_source_gradient_ncc": selected_candidate.get(
                    "validation_source_gradient_ncc", np.nan
                ),
                "validation_saturation_fraction": selected_candidate.get(
                    "validation_saturation_fraction", np.nan
                ),
                "validation_l1_normalized": selected_candidate.get(
                    "validation_l1_normalized", np.nan
                ),
                "validation_psnr_db": selected_candidate.get(
                    "validation_psnr_db", np.nan
                ),
                "checkpoint_sha256": manifest["checkpoint_sha256"],
            }
        )
    checkpoint_status = pd.DataFrame(checkpoint_rows).sort_values("checkpoint_fold")
    checkpoint_status_path = result_root / "checkpoint_gate_status.csv"
    _write_frame(checkpoint_status, checkpoint_status_path)

    predecessor_root = contract.get("predecessor_uni_v1_evaluation", {}).get(
        "root", PREDECESSOR_UNI_OUTPUT_ROOT
    )
    predecessor_audit, predecessor_audit_table = _audit_predecessor_uni(
        location, predecessor_root
    )
    predecessor_audit_outputs: dict[str, dict[str, str]] = {}
    if predecessor_audit_table is not None:
        predecessor_audit_table_path = result_root / "predecessor_uni_v1_metric_audit.csv"
        _write_frame(predecessor_audit_table, predecessor_audit_table_path)
        predecessor_audit["metric_comparison"] = {
            "path": str(predecessor_audit_table_path),
            "sha256": sha256(predecessor_audit_table_path),
        }
        predecessor_audit_outputs["predecessor_uni_v1_metric_audit"] = dict(
            predecessor_audit["metric_comparison"]
        )
    predecessor_audit_path = result_root / "predecessor_uni_v1_audit.json"
    _write_json(predecessor_audit_path, predecessor_audit)
    predecessor_audit_outputs["predecessor_uni_v1_audit"] = {
        "path": str(predecessor_audit_path),
        "sha256": sha256(predecessor_audit_path),
    }

    uni2_flags = {bool(item.get("uni2_sensitivity", False)) for item in manifests}
    if len(uni2_flags) != 1:
        raise ValueError("model replicates mix UNI2-h sensitivity modes")
    uni2_enabled = uni2_flags == {True}
    uni2_outputs: dict[str, dict[str, str]] = {}
    if uni2_enabled:
        secondary = location.copy()
        for metric in METRICS:
            secondary[metric] = secondary[f"uni2_{metric}"]
        _, secondary_section = _section_tables(secondary)
        secondary_per_model = _summary_by_model(
            secondary_section,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 50_000,
        )
        secondary_across = _summary_across_models(
            secondary_section,
            secondary_per_model,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 60_000,
        )
        secondary_per_model_path = result_root / "uni2_sensitivity_per_model_summary.csv"
        secondary_across_path = result_root / "uni2_sensitivity_across_model_summary.csv"
        _write_frame(secondary_per_model, secondary_per_model_path)
        _write_frame(secondary_across, secondary_across_path)
        reproduction = location.groupby(
            ["checkpoint_fold", "source_scanner"], as_index=False
        ).agg(
            mean_fresh_source_to_stored_source_distance=(
                "uni2_fresh_source_to_stored_source_distance",
                "mean",
            ),
            max_fresh_source_to_stored_source_distance=(
                "uni2_fresh_source_to_stored_source_distance",
                "max",
            ),
            mean_fresh_at2_to_stored_at2_distance=(
                "uni2_fresh_at2_to_stored_at2_distance",
                "mean",
            ),
            max_fresh_at2_to_stored_at2_distance=(
                "uni2_fresh_at2_to_stored_at2_distance",
                "max",
            ),
        )
        reproduction_path = result_root / "uni2_encoder_reproduction_check.csv"
        _write_frame(reproduction, reproduction_path)
        uni2_outputs = {
            "uni2_sensitivity_per_model_summary": {
                "path": str(secondary_per_model_path),
                "sha256": sha256(secondary_per_model_path),
            },
            "uni2_sensitivity_across_model_summary": {
                "path": str(secondary_across_path),
                "sha256": sha256(secondary_across_path),
            },
            "uni2_encoder_reproduction_check": {
                "path": str(reproduction_path),
                "sha256": sha256(reproduction_path),
            },
        }
    primary = across.loc[
        across["source_scanner"].eq(MODEL_SOURCE_SCANNER)
        & across["metric"].eq("gain_to_at2")
    ]
    primary_image_safety_gates = [
        {
            "metric": str(row.metric),
            "direction": str(row.direction),
            "threshold": float(row.threshold),
            "estimate": float(row.estimate),
            "pass": bool(row.pass_),
            "selection_use": str(row.selection_use),
        }
        for row in image_safety_gates.loc[
            image_safety_gates["source_scanner"].eq(MODEL_SOURCE_SCANNER)
        ]
        .rename(columns={"pass": "pass_"})
        .itertuples(index=False)
    ]
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "status": (
            "diagnostic_complete"
            if any(value != "provisional_image_candidate" for value in selection_statuses)
            else "complete"
        ),
        "created_utc": utc_now(),
        "contract_sha256": contract_hash,
        "checkpoint_folds": sorted(folds),
        "checkpoint_replicates": len(manifests),
        "checkpoint_selection_statuses": selection_statuses,
        "analysis_role": (
            "diagnostic_only"
            if any(value != "provisional_image_candidate" for value in selection_statuses)
            else "provisional_external_evaluation"
        ),
        "physical_sections": int(location["stain"].nunique()),
        "location_rows": int(len(location)),
        "primary_input": MODEL_SOURCE_SCANNER,
        "primary_encoder": "MahmoodLab/uni UNI-v1, 1024-d; fresh raw/generated/AT2",
        "primary_endpoint": "gain_to_at2; positive means reduced paired UNI-v1 distance",
        "primary_gt450_quadrants": primary.to_dict(orient="records"),
        "predecessor_uni_v1_audit": predecessor_audit,
        "external_image_safety": {
            "evaluation_version": IMAGE_EVALUATION_VERSION,
            "metrics": list(IMAGE_METRICS),
            "primary_gt450_descriptive_gates": primary_image_safety_gates,
            "aggregation": (
                "location -> core -> section; checkpoint models averaged within section"
            ),
            "selection_role": (
                "diagnostic only; external outcomes never select checkpoints and cannot "
                "rescue the failed internal source-gradient gate"
            ),
            "at2_identity_control_included": False,
        },
        "at2_identity_damage_control": identity_across,
        "uni2_sensitivity_enabled": uni2_enabled,
        "inference": (
            "location -> core -> section; model replicates averaged inside each section; "
            "percentile bootstrap resamples 13 sections"
        ),
        "interpretation_boundary": (
            "A positive external UNI-v1 gain supports representation-space transfer but "
            "cannot rescue a failed image-fidelity or hallucination gate. Shared/shared "
            "is only scanner-model/tissue-label ID-like, never fully in-distribution. "
            "Any checkpoint labelled diagnostic_candidate_after_basic_fidelity_failure "
            "must remain diagnostic even when its PLISM endpoint is favourable."
        ),
        "outputs": {
            "location_metrics": {"path": str(location_path), "sha256": sha256(location_path)},
            "core_section_metrics": {"path": str(core_path), "sha256": sha256(core_path)},
            "section_metrics": {"path": str(section_path), "sha256": sha256(section_path)},
            "per_model_summary": {"path": str(per_model_path), "sha256": sha256(per_model_path)},
            "across_model_summary": {"path": str(across_path), "sha256": sha256(across_path)},
            "model_pair_consistency": {"path": str(consistency_path), "sha256": sha256(consistency_path)},
            "image_safety_location_metrics": {
                "path": str(image_safety_location_path),
                "sha256": sha256(image_safety_location_path),
            },
            "image_safety_core_section_metrics": {
                "path": str(image_safety_core_path),
                "sha256": sha256(image_safety_core_path),
            },
            "image_safety_section_metrics": {
                "path": str(image_safety_section_path),
                "sha256": sha256(image_safety_section_path),
            },
            "image_safety_per_model_summary": {
                "path": str(image_safety_per_model_path),
                "sha256": sha256(image_safety_per_model_path),
            },
            "image_safety_across_model_summary": {
                "path": str(image_safety_across_path),
                "sha256": sha256(image_safety_across_path),
            },
            "image_safety_overall_scanner_summary": {
                "path": str(image_safety_overall_path),
                "sha256": sha256(image_safety_overall_path),
            },
            "image_safety_external_gates": {
                "path": str(image_safety_gates_path),
                "sha256": sha256(image_safety_gates_path),
            },
            "at2_identity_damage_location_metrics": {
                "path": str(identity_location_path),
                "sha256": sha256(identity_location_path),
            },
            "at2_identity_damage_section_metrics": {
                "path": str(identity_section_path),
                "sha256": sha256(identity_section_path),
            },
            "at2_identity_damage_per_model_summary": {
                "path": str(identity_per_model_path),
                "sha256": sha256(identity_per_model_path),
            },
            "at2_identity_damage_across_models": {
                "path": str(identity_across_path),
                "sha256": sha256(identity_across_path),
            },
            "checkpoint_gate_status": {
                "path": str(checkpoint_status_path),
                "sha256": sha256(checkpoint_status_path),
            },
            **predecessor_audit_outputs,
            **uni2_outputs,
        },
    }
    summary_path = result_root / "summary.json"
    _write_json(summary_path, summary)
    return summary


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def write_slurm_scripts(
    *,
    output_root: str | Path = OUTPUT_ROOT,
    runs_root: str | Path = PIX2PIX_RUNS_ROOT,
) -> dict[str, Any]:
    """Freeze five image-selected candidates and write, but never submit, jobs.

    GPU availability is intentionally not encoded here.  The caller must run
    ``gpu_availability.sh gpuq`` immediately before submitting the evaluation
    array.  The generic ``gpu:1`` request lets that live decision remain with
    the submitter; the primary UNI-v1 path fits a single A100-40G.
    """

    output = Path(output_root).resolve()
    runs = Path(runs_root).resolve()
    rows = []
    for fold in range(5):
        selections = sorted(runs.glob(f"fold_{fold}/seed_*/checkpoint_selection.json"))
        if len(selections) != 1:
            raise ValueError(f"fold {fold}: expected one checkpoint selection, found {len(selections)}")
        selection_path = selections[0]
        selection = _read_json(selection_path)
        checkpoint = Path(str(selection.get("selected_checkpoint", ""))).resolve()
        _validate_image_only_selection(checkpoint, selection_path, fold)
        rows.append(
            {
                "fold": fold,
                "checkpoint": str(checkpoint),
                "selection": str(selection_path),
                "selection_status": str(selection.get("selection_status", "unknown")),
                "validation_source_gradient_ncc": selection.get("selected", {}).get(
                    "validation_source_gradient_ncc", np.nan
                ),
            }
        )
    candidates = pd.DataFrame(rows)
    script_root = output / "00_contract/slurm"
    log_root = output / "logs"
    script_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    candidate_path = output / "00_contract/checkpoint_candidates.csv"
    _write_frame(candidates, candidate_path)

    python_path_value = shlex.quote(str(ROOT / "src"))
    output_value = shlex.quote(str(output))
    candidate_value = shlex.quote(str(candidate_path))
    python_command = "python -m scanner_gan.plism"
    common = f"""source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH={python_path_value}${{PYTHONPATH:+:$PYTHONPATH}}
PLISM_OUTPUT_ROOT={output_value}
mkdir -p "$PLISM_OUTPUT_ROOT/logs"
"""
    prepare_path = script_root / "01_prepare.sbatch"
    _write_text(
        prepare_path,
        f"""#!/bin/bash
#SBATCH --job-name=plism-p2p-prepare
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output={output}/logs/prepare_%j.out

{common}
{python_command} --output-root "$PLISM_OUTPUT_ROOT" prepare --overwrite
""",
    )
    # The clean contract always has 13 AT2 + 13 GT450 render tasks.  In the
    # additive v2 run these normally validate/reuse v1 shards read-only.
    # Optional boundary scanners should regenerate this script after preparing
    # a wider contract, rather than silently expanding the primary run.
    render_path = script_root / "02_render_array.sbatch"
    _write_text(
        render_path,
        f"""#!/bin/bash
#SBATCH --job-name=plism-p2p-render
#SBATCH --partition=defq
#SBATCH --array=0-25%13
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output={output}/logs/render_%A_%a.out

{common}
{python_command} --output-root "$PLISM_OUTPUT_ROOT" render-task --task-index "$SLURM_ARRAY_TASK_ID"
""",
    )
    evaluate_path = script_root / "03_evaluate_array.sbatch"
    _write_text(
        evaluate_path,
        f"""#!/bin/bash
#SBATCH --job-name=plism-p2p-univ1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%5
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output={output}/logs/evaluate_%A_%a.out

{common}
CHECKPOINT_TABLE={candidate_value}
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$CHECKPOINT_TABLE")
IFS=, read -r fold checkpoint selection selection_status validation_ncc <<< "$row"
test "$fold" = "$SLURM_ARRAY_TASK_ID"
{python_command} --output-root "$PLISM_OUTPUT_ROOT" evaluate-checkpoint \
  --test-fold "$fold" --checkpoint "$checkpoint" --selection "$selection" \
  --generator-batch-size 16 --uni-batch-size 32 --overwrite
""",
    )
    aggregate_path = script_root / "04_aggregate.sbatch"
    _write_text(
        aggregate_path,
        f"""#!/bin/bash
#SBATCH --job-name=plism-p2p-aggregate
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output={output}/logs/aggregate_%j.out

{common}
{python_command} --output-root "$PLISM_OUTPUT_ROOT" aggregate --expected-models 5
""",
    )
    submit_path = script_root / "SUBMISSION_ORDER.txt"
    _write_text(
        submit_path,
        f"""These commands are documentation only; this module never submits jobs.

1. Verify controller and capacity:
   getent hosts respublica-01 && scontrol ping
   sinfo -p defq -o \"%.8P %.5D %.15C %.10m %t\"

2. Submit prepare, then render with dependency:
   PREP=$(sbatch --parsable --export=ALL,JOB_CONDA_PREFIX=\"$CONDA_PREFIX\",JOB_WORKDIR=\"$PWD\" {prepare_path})
   RENDER=$(sbatch --parsable --dependency=afterok:$PREP --export=ALL,JOB_CONDA_PREFIX=\"$CONDA_PREFIX\",JOB_WORKDIR=\"$PWD\" {render_path})

3. Immediately before GPU submission, recheck:
   getent hosts respublica-01 && scontrol ping
   bash ~/Scripts/gpu_availability.sh gpuq

4. Submit the five-model GPU array, then aggregate only after success:
   EVAL=$(sbatch --parsable --dependency=afterok:$RENDER --export=ALL,JOB_CONDA_PREFIX=\"$CONDA_PREFIX\",JOB_WORKDIR=\"$PWD\" {evaluate_path})
   sbatch --dependency=afterok:$EVAL --export=ALL,JOB_CONDA_PREFIX=\"$CONDA_PREFIX\",JOB_WORKDIR=\"$PWD\" {aggregate_path}

The default evaluation is primary UNI-v1 only. UNI2-h remains an optional
secondary sensitivity and is not enabled by this formal clean-replication job.
""",
    )
    result = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "scripts_written_not_submitted",
        "created_utc": utc_now(),
        "output_root": str(output),
        "checkpoint_candidates": {
            "path": str(candidate_path),
            "sha256": sha256(candidate_path),
            "rows": len(candidates),
        },
        "scripts": [
            {"path": str(path), "sha256": sha256(path)}
            for path in (prepare_path, render_path, evaluate_path, aggregate_path, submit_path)
        ],
        "jobs_submitted": False,
        "gpu_availability_required_immediately_before_submission": True,
    }
    manifest_path = script_root / "scripts_manifest.json"
    _write_json(manifest_path, result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--plism-root", type=Path, default=PLISM_ROOT)
    prepare.add_argument("--plism-analysis-root", type=Path, default=PLISM_ANALYSIS_ROOT)
    prepare.add_argument(
        "--input-scanner",
        action="append",
        default=None,
        help="Repeat for exploratory boundary inputs; GT450 is always required.",
    )
    prepare.add_argument(
        "--locations-per-core-section",
        type=int,
        default=DEFAULT_LOCATIONS_PER_CORE_SECTION,
    )
    prepare.add_argument("--overwrite", action="store_true")

    render = subparsers.add_parser("render-task")
    render.add_argument("--task-index", type=int, required=True)
    render.add_argument("--overwrite", action="store_true")

    evaluate = subparsers.add_parser("evaluate-checkpoint")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--test-fold", type=int, required=True, choices=range(5))
    evaluate.add_argument("--selection", type=Path)
    evaluate.add_argument("--uni-checkpoint", type=Path)
    evaluate.add_argument(
        "--uni2-sensitivity",
        action="store_true",
        help="Also run fresh UNI2-h as a secondary, non-primary sensitivity.",
    )
    evaluate.add_argument("--uni2-checkpoint", type=Path)
    evaluate.add_argument("--generator-batch-size", type=int, default=16)
    evaluate.add_argument("--uni-batch-size", type=int, default=32)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--amp", default="bfloat16")
    evaluate.add_argument("--overwrite", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--expected-models", type=int, default=5)
    aggregate_parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    aggregate_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    scripts_parser = subparsers.add_parser("write-slurm")
    scripts_parser.add_argument("--runs-root", type=Path, default=PIX2PIX_RUNS_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "prepare":
        result = prepare_contract(
            plism_root=args.plism_root,
            plism_analysis_root=args.plism_analysis_root,
            output_root=args.output_root,
            input_scanners=args.input_scanner or (MODEL_SOURCE_SCANNER,),
            locations_per_core_section=args.locations_per_core_section,
            overwrite=args.overwrite,
        )
    elif args.command == "render-task":
        result = render_task(
            args.task_index,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
    elif args.command == "evaluate-checkpoint":
        result = evaluate_checkpoint(
            args.checkpoint,
            args.test_fold,
            output_root=args.output_root,
            selection_path=args.selection,
            uni_checkpoint_path=args.uni_checkpoint,
            uni2_sensitivity=args.uni2_sensitivity,
            uni2_checkpoint_path=args.uni2_checkpoint,
            generator_batch_size=args.generator_batch_size,
            uni_batch_size=args.uni_batch_size,
            device=args.device,
            amp=args.amp,
            overwrite=args.overwrite,
        )
    elif args.command == "aggregate":
        result = aggregate(
            output_root=args.output_root,
            expected_models=args.expected_models,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
    elif args.command == "write-slurm":
        result = write_slurm_scripts(
            output_root=args.output_root,
            runs_root=args.runs_root,
        )
    else:  # pragma: no cover - argparse enforces the command choices.
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
