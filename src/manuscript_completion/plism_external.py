#!/usr/bin/env python3
"""Leakage-safe conventional correction on the external PLISM cohort.

The workflow is additive and split into independently auditable stages:

``prepare-render``
    Freeze the existing 2,387-location/13-section grid and all 52 immutable
    AT2/GT450/S360/S60 renders. Bind the input-only Vahadane selection survey.
``fit-parameters``
    Fit five direction-specific parameter replicates from exactly three
    PanNormal training folds. Validation, test, and paired PLISM AT2 pixels
    are forbidden. Vahadane uses one globally selected input-only MU setting.
``external-image-task`` / ``external-uni-task`` / ``aggregate``
    Apply frozen parameters and aggregate location -> core -> 13 sections.

The module never edits Panel-A outputs or the immutable existing PLISM renders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from augmentation_ood_study import ENDPOINTS, absolute_measurements, contrasts_from_absolute
from scanner_gan.analytic import (
    _formal_split,
    _lab_moments,
    _paired_frequency_gain,
    _validate_statistics,
    apply_frequency,
    apply_reinhard,
)
from scanner_gan.evaluate_images import image_metrics
from scanner_gan.plism import TARGET_MPP, TARGET_PX, _vips_to_numpy, load_frozen_uni1
from scanner_gan.uni import cosine_distance_rows, embed_frozen_uni


VERSION = "manuscript_plism_external_conventional_v4"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/"
    "05_plism_external_correction"
)
SOURCE_PLISM_CONTRACT = (
    ROOT
    / "outputs/scanner_batch_effect_analysis_2026-09-17/08_plism_external/"
    "pix2pix_gt450_to_at2_image_safety/00_contract/analysis_contract.json"
)
FACTORIAL_ROOT = ROOT / "outputs/plism_factorial_external_v1"
EXISTING_RENDER_ROOT = (
    ROOT
    / "outputs/scanner_batch_effect_analysis_2026-09-17/08_plism_external/"
    "pix2pix_gt450_to_at2/01_rendered_inputs"
)
PLISM_ROOT = ROOT / "data/PLISM_dataset"
SAMPLE_INDEX = (
    ROOT
    / "outputs/scanner_batch_effect_analysis_2026-09-17/06_learned_baselines/"
    "00_contract/sample_index.csv.gz"
)
LOCKED_INDEX = (
    ROOT
    / "outputs/scanner_batch_effect_analysis_2026-09-17/06_learned_baselines/"
    "00_contract/locked_image_evaluation_index.csv.gz"
)
LAB_STATISTICS = (
    ROOT
    / "outputs/scanner_batch_effect_analysis_2026-09-17/02_correction/"
    "lab_sufficient_statistics.csv"
)
FINAL_IMAGE_ROOT = ROOT / "outputs/final_image_study_v1"
SCANNERS = ("gt450", "s360", "s60")
METHODS = ("raw", "reinhard", "macenko", "vahadane", "frequency", "combined")
CORRECTED_METHODS = METHODS[1:]
MISSING_RENDER_SCANNERS = ("S360", "S60")
TARGET_SCANNER = "AT2"
HRH_OOF_SCANNER = "s60"
HRH_OOF_SECTION = "HRH"
PATCH_UM = TARGET_MPP * TARGET_PX
DEFAULT_BOOTSTRAPS = 20_000
V1_CONTRACT = DEFAULT_OUTPUT / "00_contract/analysis_contract.json"
CONTRACT_SUBDIR = "00_contract_v4"
VAHADANE_SELECTION_MANIFEST = (
    ROOT
    / "outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/"
    "02_baseline_benchmark/00_contract/vahadane_selection_manifest_v4.json"
)
EXPECTED_VAHADANE_SELECTION_SHA256 = (
    "fcd1b8eaee27286fe707a77c4078d25f684deaea3081ef896c54a353b600cdf8"
)
RUNTIME_RELATIVE_PATHS = (
    "src/manuscript_completion/plism_external.py",
    "src/manuscript_completion/stain.py",
    "src/augmentation_ood_study.py",
    "src/scanner_gan/analytic.py",
    "src/scanner_gan/evaluate_images.py",
    "src/scanner_gan/plism.py",
    "src/scanner_gan/uni.py",
    "scripts/manuscript_completion_plism_prepare.sbatch",
    "scripts/manuscript_completion_plism_render.sbatch",
    "scripts/manuscript_completion_plism_fit_parameters.sbatch",
    "scripts/manuscript_completion_plism_external_image.sbatch",
    "scripts/manuscript_completion_plism_external_uni.sbatch",
    "scripts/manuscript_completion_plism_internal_image.sbatch",
    "scripts/manuscript_completion_plism_aggregate.sbatch",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*values: object) -> int:
    value = "|".join(map(str, values)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    compression = "gzip" if path.name.endswith(".gz") else None
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def runtime_hashes() -> dict[str, str]:
    """Hash every executable/helper that can affect a frozen PLISM result."""

    result: dict[str, str] = {}
    for relative in RUNTIME_RELATIVE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = sha256(path)
    return result


def _write_runtime_checksum_file(path: Path, hashes: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{hashes[name]}  {name}" for name in sorted(hashes)]
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def _verify_runtime_hashes(contract: Mapping[str, Any]) -> None:
    expected = contract.get("runtime_hashes")
    if not isinstance(expected, dict) or set(expected) != set(RUNTIME_RELATIVE_PATHS):
        raise ValueError("frozen runtime hash set is incomplete")
    observed = runtime_hashes()
    changed = [name for name in sorted(expected) if expected[name] != observed[name]]
    if changed:
        raise RuntimeError(f"live PLISM implementation differs from frozen contract: {changed}")


def fold_roles(sample_index: pd.DataFrame, test_fold: int) -> pd.DataFrame:
    """Assign the same 3-train/1-validation/1-test layout as learned baselines."""

    if not 0 <= int(test_fold) < 5:
        raise ValueError("test_fold must be in 0..4")
    slides = sample_index[["slide_id", "fold"]].copy()
    slides["slide_id"] = slides["slide_id"].astype(str)
    slides["fold"] = slides["fold"].astype(int)
    slides = slides.drop_duplicates()
    if slides["slide_id"].duplicated().any() or set(slides["fold"]) != set(range(5)):
        raise ValueError("sample index does not contain one stable five-fold slide assignment")
    validation = (int(test_fold) + 1) % 5
    slides["split_role"] = np.where(
        slides["fold"].eq(int(test_fold)),
        "test",
        np.where(slides["fold"].eq(validation), "validation", "train"),
    )
    _, provenance = _formal_split(slides)
    if provenance["outer_test_fold"] != int(test_fold) or provenance["validation_fold"] != validation:
        raise AssertionError("split provenance changed")
    return slides


def _validate_render(path: Path, wanted: np.ndarray) -> dict[str, Any]:
    summary_path = path.with_suffix(".summary.json")
    if not path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(path)
    summary = read_json(summary_path)
    observed_hash = sha256(path)
    if summary.get("output_sha256") != observed_hash:
        raise ValueError(f"render hash differs from summary: {path}")
    with h5py.File(path, "r") as store:
        locations = np.asarray(store["location"], dtype=np.int64)
        shape = tuple(store["images"].shape)
        dtype = store["images"].dtype
    if not np.array_equal(locations, wanted):
        raise ValueError(f"render locations differ from locked contract: {path}")
    if shape != (len(wanted), TARGET_PX, TARGET_PX, 3) or dtype != np.uint8:
        raise ValueError(f"render grid differs from locked contract: {path}")
    return {
        "scanner": path.parent.name,
        "stain": path.stem,
        "path": str(path.resolve()),
        "sha256": observed_hash,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256(summary_path),
        "locations": len(wanted),
    }


def _prepare_render_contract_v1_unused(
    output_root: Path = DEFAULT_OUTPUT,
    source_contract_path: Path = SOURCE_PLISM_CONTRACT,
    factorial_root: Path = FACTORIAL_ROOT,
    existing_render_root: Path = EXISTING_RENDER_ROOT,
    plism_root: Path = PLISM_ROOT,
) -> dict[str, Any]:
    """Freeze locations/renders before fitting or observing correction outcomes."""

    output_root = output_root.resolve()
    contract_path = output_root / "00_contract/analysis_contract.json"
    if contract_path.exists():
        contract = read_json(contract_path)
        _verify_runtime_hashes(contract)
        for item in contract["outputs"].values():
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"frozen PLISM contract output changed: {item['path']}")
        return {"status": "already_complete", **contract}

    source_contract = read_json(source_contract_path)
    selected_source = Path(source_contract["outputs"]["selected_locations"]["path"])
    if sha256(selected_source) != source_contract["outputs"]["selected_locations"]["sha256"]:
        raise ValueError("source PLISM selected locations changed")
    selected = pd.read_csv(selected_source)
    if len(selected) != 2387 or selected["stain"].nunique() != 13:
        raise ValueError("PLISM primary panel must be exactly 2,387 locations/13 sections")
    if selected[["stain", "location"]].duplicated().any():
        raise ValueError("duplicate locked PLISM location")

    read_only_rows = []
    for stain, part in selected.groupby("stain", sort=True):
        wanted = np.sort(part["location"].to_numpy(np.int64))
        for scanner in (TARGET_SCANNER, "GT450"):
            read_only_rows.append(
                _validate_render(existing_render_root / scanner / f"{stain}.h5", wanted)
            )
    read_only = pd.DataFrame(read_only_rows)

    factorial_tasks_path = factorial_root / "00_contract/tasks.csv"
    factorial_tasks = pd.read_csv(factorial_tasks_path)
    plan_root = output_root / "00_contract/render_task_plans"
    tasks = []
    for scanner in MISSING_RENDER_SCANNERS:
        for stain, wanted in selected.groupby("stain", sort=True):
            match = factorial_tasks[
                factorial_tasks["scanner"].astype(str).eq(scanner)
                & factorial_tasks["stain"].astype(str).eq(str(stain))
            ]
            if len(match) != 1:
                raise ValueError(f"expected one factorial task for {scanner}/{stain}")
            source_task = match.iloc[0]
            source_plan = Path(str(source_task["plan_path"]))
            if sha256(source_plan) != str(source_task["plan_sha256"]):
                raise ValueError(f"factorial render plan changed: {source_plan}")
            plan = pd.read_csv(source_plan)
            plan = plan.merge(
                wanted[[
                    "stain",
                    "location",
                    "pannormal_tissue",
                    "mapping_tier",
                    "organ_aligned_status",
                    "strict_status",
                    "selection_rule",
                ]],
                on=["stain", "location"],
                how="inner",
                validate="one_to_one",
            ).sort_values("location", kind="stable")
            if len(plan) != len(wanted):
                raise ValueError(f"incomplete exact-grid plan for {scanner}/{stain}")
            task_index = len(tasks)
            plan_path = plan_root / f"{task_index:03d}.csv"
            write_frame(plan_path, plan)
            wsi_path = plism_root / "original_wsi" / str(source_task["actual_file"])
            if not wsi_path.is_file():
                raise FileNotFoundError(wsi_path)
            tasks.append(
                {
                    "task_index": task_index,
                    "scanner": scanner,
                    "stain": str(stain),
                    "locations": len(plan),
                    "plan_path": str(plan_path.resolve()),
                    "plan_sha256": sha256(plan_path),
                    "wsi_path": str(wsi_path.resolve()),
                    "wsi_bytes": wsi_path.stat().st_size,
                }
            )
    render_tasks = pd.DataFrame(tasks)
    if len(render_tasks) != 26 or int(render_tasks["locations"].sum()) != 2 * 2387:
        raise ValueError("missing-render task contract must be 2 scanners x 13 sections")

    selected_path = output_root / "00_contract/selected_locations.csv"
    read_only_path = output_root / "00_contract/read_only_renders.csv"
    tasks_path = output_root / "00_contract/render_tasks.csv"
    runtime_path = output_root / "00_contract/runtime_files.sha256"
    write_frame(selected_path, selected)
    write_frame(read_only_path, read_only)
    write_frame(tasks_path, render_tasks)
    frozen_runtime = runtime_hashes()
    _write_runtime_checksum_file(runtime_path, frozen_runtime)
    contract = {
        "analysis_version": VERSION,
        "status": "render_contract_frozen_before_external_outcomes",
        "created_utc": utc_now(),
        "selected_locations": 2387,
        "sections": 13,
        "section_core_cells": int(selected.groupby(["stain", "core"]).ngroups),
        "source_scanners": [name.upper() for name in SCANNERS],
        "target_scanner": TARGET_SCANNER,
        "methods": list(METHODS),
        "aggregation": "location -> core -> physical section; five fold parameter fits averaged within section before 13-section bootstrap",
        "render": {
            "target_px": TARGET_PX,
            "target_mpp": TARGET_MPP,
            "physical_fov_um": PATCH_UM,
            "kernel": "libvips Lanczos3",
            "immutable_reused_scanners": [TARGET_SCANNER, "GT450"],
            "new_scanners": list(MISSING_RENDER_SCANNERS),
            "existing_root": str(existing_render_root.resolve()),
            "new_root": str((output_root / "01_rendered_inputs").resolve()),
        },
        "fit_boundary": "each fold uses PanNormal split_role=train only (3 folds); validation/test and every PLISM image are excluded",
        "external_target_boundary": "PLISM AT2 is evaluation-only and cannot fit any moment, stain basis, frequency gain, scale, threshold, or model choice",
        "stain_application_boundary": "Macenko/Vahadane estimate the source basis independently within each input source patch as part of the declared transform; target references and every cohort-level parameter are PanNormal-train-only, and paired PLISM AT2 is never used",
        "runtime_hashes": frozen_runtime,
        "sensitivity": {
            "out_of_focus": "exclude HRH only for S60",
            "tissue_status": "shared/novel secondary; section and stain remain confounded",
        },
        "source_contract": {"path": str(source_contract_path.resolve()), "sha256": sha256(source_contract_path)},
        "factorial_tasks": {"path": str(factorial_tasks_path.resolve()), "sha256": sha256(factorial_tasks_path)},
        "outputs": {
            "selected_locations": {"path": str(selected_path), "sha256": sha256(selected_path)},
            "read_only_renders": {"path": str(read_only_path), "sha256": sha256(read_only_path)},
            "render_tasks": {"path": str(tasks_path), "sha256": sha256(tasks_path)},
            "runtime_files": {"path": str(runtime_path), "sha256": sha256(runtime_path)},
        },
    }
    write_json(contract_path, contract)
    return contract


def prepare_render_contract(
    output_root: Path = DEFAULT_OUTPUT,
    *,
    v1_contract_path: Path = V1_CONTRACT,
    vahadane_selection_manifest: Path = VAHADANE_SELECTION_MANIFEST,
) -> dict[str, Any]:
    """Freeze v4 after the source-only Vahadane v4 audit, without opening outcomes."""

    output_root = output_root.resolve()
    contract_path = output_root / CONTRACT_SUBDIR / "analysis_contract.json"
    if contract_path.exists():
        contract = read_json(contract_path)
        _verify_runtime_hashes(contract)
        for item in contract["outputs"].values():
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"frozen PLISM v4 contract output changed: {item['path']}")
        return {"status": "already_complete", **contract}

    v1 = read_json(v1_contract_path)
    if v1.get("status") != "render_contract_frozen_before_external_outcomes":
        raise RuntimeError("v1 render-input contract is not frozen")
    selected_source = Path(v1["outputs"]["selected_locations"]["path"])
    if sha256(selected_source) != v1["outputs"]["selected_locations"]["sha256"]:
        raise RuntimeError("v1 locked location grid changed")
    selected = pd.read_csv(selected_source)
    if len(selected) != 2387 or selected["stain"].nunique() != 13:
        raise ValueError("v4 must retain exactly 2,387 locations/13 sections")

    selection = read_json(vahadane_selection_manifest)
    if sha256(vahadane_selection_manifest) != EXPECTED_VAHADANE_SELECTION_SHA256:
        raise RuntimeError("Vahadane v4 selection manifest hash is not the sealed value")
    selected_configuration = selection.get("selected_configuration")
    required_configuration = {
        "solver": "mu",
        "init": "nndsvdar",
        "alpha_W": 0.001,
        "alpha_H": 0.0,
        "l1_ratio": 1.0,
        "tolerance": 0.001,
        "max_iter": 2000,
        "max_pixels_source": 4096,
        "max_pixels_reference": 65536,
        "shuffle": False,
    }
    if (
        selection.get("status") not in {"selected", "pass"}
        or selection.get("analysis") != "global_source_only_vahadane_contract_selection_v4"
        or not isinstance(selected_configuration, dict)
        or bool(selection.get("target_images_accessed", True))
        or bool(selection.get("target_outcomes_accessed", True))
    ):
        raise RuntimeError("Vahadane selection is not supported by the sealed source-only v4 audit")
    audit = selection.get("locked_source_convergence_audit", {})
    selected_audit = audit.get("results", {}).get("2000", {})
    if (
        audit.get("candidate_max_iterations") != [1000, 2000, 4000]
        or not bool(selected_audit.get("eligible", False))
        or int(selected_audit.get("evaluation_converged", -1)) != 20_600
        or int(selected_audit.get("evaluation_nonconverged", -1)) != 0
        or int(selected_audit.get("sentinel_converged", -1)) != 25
        or float(selected_audit.get("basis_condition_max", math.inf)) >= 50.0
    ):
        raise RuntimeError("sealed source-only Vahadane v4 convergence audit did not pass")
    for key, expected in required_configuration.items():
        observed = selected_configuration.get(key)
        if isinstance(expected, float):
            if not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"unexpected selected Vahadane {key}: {observed}")
        elif observed != expected:
            raise RuntimeError(f"unexpected selected Vahadane {key}: {observed}")
    render_rows: list[dict[str, Any]] = []
    for stain, group in selected.groupby("stain", sort=True):
        wanted = np.sort(group["location"].to_numpy(np.int64))
        for scanner in ("AT2", "GT450", "S360", "S60"):
            path = (
                EXISTING_RENDER_ROOT / scanner / f"{stain}.h5"
                if scanner in {"AT2", "GT450"}
                else output_root / "01_rendered_inputs" / scanner / f"{stain}.h5"
            )
            row = _validate_render(path, wanted)
            row["input_generation_contract_sha256"] = read_json(path.with_suffix(".summary.json")).get(
                "contract_sha256"
            )
            render_rows.append(row)
    renders = pd.DataFrame(render_rows)
    if len(renders) != 52 or int(renders["locations"].sum()) != 4 * 2387:
        raise ValueError("v4 frozen render table must contain 4 scanners x 13 sections")

    contract_root = output_root / CONTRACT_SUBDIR
    selected_path = contract_root / "selected_locations.csv"
    render_path = contract_root / "frozen_renders.csv"
    runtime_path = contract_root / "runtime_files.sha256"
    write_frame(selected_path, selected)
    write_frame(render_path, renders)
    frozen_runtime = runtime_hashes()
    _write_runtime_checksum_file(runtime_path, frozen_runtime)
    contract = {
        "analysis_version": VERSION,
        "status": "v4_frozen_before_target_outcomes",
        "created_utc": utc_now(),
        "selected_locations": 2387,
        "sections": 13,
        "section_core_cells": int(selected.groupby(["stain", "core"]).ngroups),
        "source_scanners": [name.upper() for name in SCANNERS],
        "target_scanner": TARGET_SCANNER,
        "completed_methods": list(METHODS),
        "vahadane_configuration": required_configuration,
        "fit_boundary": "each fold uses PanNormal split_role=train only (3 folds); validation/test and every PLISM target outcome are excluded",
        "external_target_boundary": "PLISM AT2 is evaluation-only and cannot fit any moment, stain basis, frequency gain, scale, threshold, or model choice",
        "stain_application_boundary": "Macenko and Vahadane independently estimate a source basis within each source patch; both AT2 target references are PanNormal-train-only and paired PLISM AT2 is never used",
        "aggregation": "location -> core -> physical section; five parameter folds averaged within section before 13-section bootstrap",
        "sensitivity": {
            "out_of_focus": "exclude HRH only for S60",
            "tissue_status": "shared/novel secondary; section and stain remain confounded",
        },
        "runtime_hashes": frozen_runtime,
        "v1_render_contract": {
            "path": str(v1_contract_path.resolve()),
            "sha256": sha256(v1_contract_path),
            "role": "input generation only; no correction outcome or method selection",
        },
        "vahadane_selection_manifest": {
            "path": str(vahadane_selection_manifest.resolve()),
            "sha256": sha256(vahadane_selection_manifest),
            "target_images_accessed": False,
            "target_outcomes_accessed": False,
        },
        "outputs": {
            "selected_locations": {"path": str(selected_path.resolve()), "sha256": sha256(selected_path)},
            "frozen_renders": {"path": str(render_path.resolve()), "sha256": sha256(render_path)},
            "runtime_files": {"path": str(runtime_path.resolve()), "sha256": sha256(runtime_path)},
            "vahadane_selection": {
                "path": str(vahadane_selection_manifest.resolve()),
                "sha256": sha256(vahadane_selection_manifest),
            },
        },
    }
    write_json(contract_path, contract)
    return contract


def _validated_contract(output_root: Path) -> tuple[dict[str, Any], str]:
    path = output_root / CONTRACT_SUBDIR / "analysis_contract.json"
    contract = read_json(path)
    if contract.get("analysis_version") != VERSION:
        raise ValueError("unsupported PLISM conventional contract")
    if contract.get("status") != "v4_frozen_before_target_outcomes":
        raise ValueError("PLISM v4 contract is not frozen")
    _verify_runtime_hashes(contract)
    for item in contract["outputs"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"frozen contract item changed: {item['path']}")
    return contract, sha256(path)


def render_task(task_index: int, output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_root = output_root.resolve()
    contract, contract_hash = _validated_contract(output_root)
    tasks = pd.read_csv(contract["outputs"]["render_tasks"]["path"])
    match = tasks[tasks["task_index"].astype(int).eq(int(task_index))]
    if len(match) != 1:
        raise KeyError(task_index)
    task = match.iloc[0]
    plan_path = Path(str(task["plan_path"]))
    if sha256(plan_path) != str(task["plan_sha256"]):
        raise ValueError("render plan changed")
    plan = pd.read_csv(plan_path).sort_values("location", kind="stable")
    destination = output_root / "01_rendered_inputs" / str(task["scanner"]) / f"{task['stain']}.h5"
    summary_path = destination.with_suffix(".summary.json")
    if destination.exists() and summary_path.exists():
        old = read_json(summary_path)
        if old.get("contract_sha256") == contract_hash and old.get("output_sha256") == sha256(destination):
            return {"status": "already_complete", **old}
        raise FileExistsError(f"stale render exists: {destination}")
    import pyvips

    wsi_path = Path(str(task["wsi_path"]))
    if wsi_path.stat().st_size != int(task["wsi_bytes"]):
        raise ValueError("source WSI size changed")
    image = pyvips.Image.new_from_file(str(wsi_path), level=0, access="random")
    if image.hasalpha():
        image = image.flatten(background=255)
    mpp = pd.to_numeric(plan["mpp"], errors="raise").to_numpy(float)
    if not np.allclose(mpp, mpp[0]):
        raise ValueError("multiple native MPP values in one render plan")
    native_mpp = float(mpp[0])
    side = int(round(PATCH_UM / native_mpp))
    rendered = np.empty((len(plan), TARGET_PX, TARGET_PX, 3), dtype=np.uint8)
    for index, row in enumerate(plan.itertuples(index=False)):
        left = int(round(float(row.centre_x))) - side // 2
        top = int(round(float(row.centre_y))) - side // 2
        if left < 0 or top < 0 or left + side > image.width or top + side > image.height:
            raise ValueError(f"crop outside WSI: {task['scanner']}/{task['stain']}/{row.location}")
        patch = image.crop(left, top, side, side)
        if str(row.flip).strip().lower() in {"true", "1"}:
            patch = patch.rot("d180")
        patch = patch.resize(TARGET_PX / patch.width, vscale=TARGET_PX / patch.height, kernel="lanczos3")
        if patch.width != TARGET_PX or patch.height != TARGET_PX:
            patch = patch.thumbnail_image(TARGET_PX, height=TARGET_PX, size="force", kernel="lanczos3")
        rendered[index] = _vips_to_numpy(patch)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".h5.{os.getpid()}.tmp")
    with h5py.File(temporary, "w") as store:
        store.attrs.update(
            analysis_version=VERSION,
            contract_sha256=contract_hash,
            scanner=str(task["scanner"]),
            stain=str(task["stain"]),
            target_mpp=TARGET_MPP,
            target_px=TARGET_PX,
            native_mpp=native_mpp,
        )
        store.create_dataset("images", data=rendered, compression="gzip", compression_opts=4, shuffle=True, chunks=(1, TARGET_PX, TARGET_PX, 3))
        for column in ("location", "core", "replicate"):
            store.create_dataset(column, data=plan[column].to_numpy(np.int64))
    temporary.replace(destination)
    summary = {
        "analysis_version": VERSION,
        "status": "complete",
        "created_utc": utc_now(),
        "contract_sha256": contract_hash,
        "task_index": int(task_index),
        "scanner": str(task["scanner"]),
        "stain": str(task["stain"]),
        "locations": len(plan),
        "plan_sha256": sha256(plan_path),
        "wsi_path": str(wsi_path.resolve()),
        "wsi_bytes": wsi_path.stat().st_size,
        "output_path": str(destination),
        "output_sha256": sha256(destination),
    }
    write_json(summary_path, summary)
    return summary


def _endpoint_training_table() -> pd.DataFrame:
    scalar = pd.read_csv(
        FINAL_IMAGE_ROOT / "02_image_phenotypes/replicate_scalar_summary.csv",
        dtype={"slide_id": str},
    )
    scalar = scalar.groupby(["slide_id", "tissue_type", "scanner"], as_index=False).mean(numeric_only=True)
    reference = scalar[scalar["scanner"].eq("at2")].set_index("slide_id")
    rows = []
    for item in scalar[scalar["scanner"].isin(SCANNERS)].itertuples(index=False):
        anchor = reference.loc[str(item.slide_id)]
        rows.append(
            {
                "slide_id": str(item.slide_id),
                "scanner": str(item.scanner),
                "delta_lab_l": float(anchor.lab_l_mean - item.lab_l_mean),
                "delta_lab_a": float(anchor.lab_a_mean - item.lab_a_mean),
                "delta_lab_b": float(anchor.lab_b_mean - item.lab_b_mean),
                "log2_mean_od_ratio": math.log2(max(float(anchor.mean_od), 1e-12) / max(float(item.mean_od), 1e-12)),
                "log2_lab_l_sd_ratio": math.log2(max(float(anchor.lab_l_sd), 1e-12) / max(float(item.lab_l_sd), 1e-12)),
                "log2_od_sd_ratio": math.log2(max(float(anchor.od_sd), 1e-12) / max(float(item.od_sd), 1e-12)),
                "log2_gradient_rms_ratio": math.log2(max(float(anchor.gradient_rms), 1e-12) / max(float(item.gradient_rms), 1e-12)),
            }
        )
    result = pd.DataFrame(rows)
    bands = pd.read_csv(FINAL_IMAGE_ROOT / "03_frequency/band_summary.csv", dtype={"slide_id": str})
    bands = bands[bands["scanner"].isin(SCANNERS)].copy()
    bands["value"] = -bands["log2_relative_transfer"].to_numpy(float)
    wide = bands.pivot(index=["slide_id", "scanner"], columns="band", values="value").reset_index()
    wide = wide.rename(columns={"low_mid": "frequency_low_mid", "mid": "frequency_mid", "high": "frequency_high"})
    return result.merge(wide, on=["slide_id", "scanner"], validate="one_to_one")


def _fit_at2_stain_reference(
    sample_index: pd.DataFrame,
    training_slides: set[str],
    fold: int,
    vahadane_configuration: Mapping[str, Any],
) -> tuple[Any, Any, int]:
    from manuscript_completion.stain import fit_macenko_od, fit_vahadane_od, tissue_od

    rows = sample_index[sample_index["slide_id"].isin(training_slides)].copy()
    arrays = []
    image_count = 0
    for slide_id, group in rows.groupby("slide_id", sort=True):
        ordered = group.sort_values("location_index")
        positions = np.linspace(0, len(ordered) - 1, min(5, len(ordered)), dtype=int)
        selected = ordered.iloc[np.unique(positions)]
        scanner_names = tuple(str(selected["scanner_order"].iloc[0]).split(","))
        scanner_index = scanner_names.index("at2")
        with h5py.File(str(selected["cache_path"].iloc[0]), "r") as store:
            for location in selected["location_index"].astype(int):
                od = tissue_od(np.asarray(store["images"][location, scanner_index], dtype=np.uint8))
                if len(od) > 512:
                    offset = stable_seed("plism-at2", fold, slide_id, location) % len(od)
                    positions = (offset + np.linspace(0, len(od) - 1, 512, dtype=int)) % len(od)
                    od = od[positions]
                if len(od):
                    arrays.append(od)
                image_count += 1
    pixels = np.concatenate(arrays) if arrays else np.empty((0, 3))
    macenko = fit_macenko_od(pixels)
    vahadane = fit_vahadane_od(
        pixels,
        seed=stable_seed("plism-at2-reference", fold),
        max_pixels=int(vahadane_configuration["max_pixels_reference"]),
        alpha=float(vahadane_configuration["alpha_W"]),
        max_iter=int(vahadane_configuration["max_iter"]),
        tolerance=float(vahadane_configuration["tolerance"]),
        solver=str(vahadane_configuration["solver"]),
    )
    if macenko is None or vahadane is None:
        raise RuntimeError("AT2 stain reference fitting failed")
    if not vahadane.converged:
        raise RuntimeError(f"AT2 selected Vahadane reference did not converge ({vahadane.iterations})")
    if not np.isfinite(np.linalg.cond(vahadane.basis)) or np.linalg.cond(vahadane.basis) >= 50.0:
        raise RuntimeError("AT2 selected Vahadane reference failed the basis condition gate")
    return macenko, vahadane, image_count


def fit_parameters(
    vahadane_selection_manifest: Path,
    stain_v4_manifest: Path,
    output_root: Path = DEFAULT_OUTPUT,
    sample_index_path: Path = SAMPLE_INDEX,
    lab_statistics_path: Path = LAB_STATISTICS,
) -> dict[str, Any]:
    """Fit and hash all 15 scanner/fold blocks before external evaluation."""

    from manuscript_completion.stain import parameters_to_json

    output_root = output_root.resolve()
    contract, contract_hash = _validated_contract(output_root)
    gate = read_json(vahadane_selection_manifest)
    selected_raw = gate.get("selected_configuration")
    if (
        gate.get("status") not in {"selected", "pass"}
        or gate.get("analysis") != "global_source_only_vahadane_contract_selection_v4"
        or not isinstance(selected_raw, dict)
        or bool(gate.get("target_images_accessed", True))
        or bool(gate.get("target_outcomes_accessed", True))
    ):
        raise RuntimeError("input-only Vahadane selection manifest is not valid")
    selected_configuration = {
        "solver": str(selected_raw["solver"]),
        "init": str(selected_raw["init"]),
        "alpha_W": float(selected_raw["alpha_W"]),
        "alpha_H": float(selected_raw["alpha_H"]),
        "l1_ratio": float(selected_raw["l1_ratio"]),
        "tolerance": float(selected_raw["tolerance"]),
        "max_iter": int(selected_raw["max_iter"]),
        "max_pixels_source": int(selected_raw["max_pixels_source"]),
        "max_pixels_reference": int(selected_raw["max_pixels_reference"]),
        "shuffle": bool(selected_raw["shuffle"]),
    }
    frozen_selection = contract["vahadane_selection_manifest"]
    selection_sha256 = sha256(vahadane_selection_manifest)
    if (
        selection_sha256 != EXPECTED_VAHADANE_SELECTION_SHA256
        or selection_sha256 != frozen_selection["sha256"]
    ):
        raise RuntimeError("Vahadane selection manifest differs from frozen v4 contract")
    stain_gate = read_json(stain_v4_manifest)
    if stain_gate.get("status") != "pass" or not bool(stain_gate.get("fit_quality_valid", False)):
        raise RuntimeError("formal selected-MU Panel-A v4 stain validation has not passed")
    sample_index = pd.read_csv(sample_index_path, dtype={"slide_id": str})
    statistics = _validate_statistics(pd.read_csv(lab_statistics_path, dtype={"slide_id": str}))
    spectra = pd.read_csv(FINAL_IMAGE_ROOT / "03_frequency/spectra.csv", dtype={"slide_id": str})
    endpoint = _endpoint_training_table()
    parameters = []
    if selected_configuration != contract["vahadane_configuration"]:
        raise RuntimeError("selected Vahadane configuration differs from frozen v4 contract")
    at2_references: dict[int, tuple[Any, Any, int]] = {}
    parameter_root = output_root / "02_parameters"
    for fold in range(5):
        roles = fold_roles(sample_index, fold)
        _, split = _formal_split(roles)
        training_slides = set(split["training_slides"])
        if len(training_slides) not in {61, 62, 63}:
            raise ValueError("unexpected three-fold training slide count")
        at2_references[fold] = _fit_at2_stain_reference(
            sample_index, training_slides, fold, selected_configuration
        )
        target_mean, target_std, _ = _lab_moments(statistics, training_slides, "at2")
        for scanner in SCANNERS:
            source_mean, source_std, _ = _lab_moments(statistics, training_slides, scanner)
            frequency, log_gain, paired = _paired_frequency_gain(
                spectra,
                training_slides,
                scanner,
                "at2",
                smoothing_sigma_bins=1.5,
                log2_gain_clip=(-1.0, 1.0),
            )
            endpoint_train = endpoint[
                endpoint["slide_id"].isin(training_slides) & endpoint["scanner"].eq(scanner)
            ]
            if len(endpoint_train) != len(training_slides):
                raise ValueError("endpoint scales do not cover training slides")
            endpoint_scale = endpoint_train[list(ENDPOINTS)].std(ddof=1).clip(lower=1e-3)
            source_curve = spectra[
                spectra["slide_id"].isin(training_slides) & spectra["scanner"].eq(scanner)
            ].pivot(index="slide_id", columns="frequency_cyc_per_pixel", values="log2_relative_transfer")
            source_curve = source_curve.sort_index(axis=1)
            curve_scale = source_curve.std(axis=0, ddof=1).clip(lower=0.05)
            macenko, vahadane, sampled_images = at2_references[fold]
            payload = {
                "analysis_version": VERSION,
                "status": "frozen",
                "created_utc": utc_now(),
                "contract_sha256": contract_hash,
                "source_scanner": scanner,
                "target_scanner": "at2",
                "outer_test_fold": fold,
                "validation_fold": (fold + 1) % 5,
                "training_folds": split["training_folds"],
                "training_slides": split["training_slides"],
                "training_slide_count": len(training_slides),
                "fit_split_role": "train",
                "external_data_opened_for_fit": False,
                "reinhard": {
                    "source_mean": source_mean,
                    "source_std": source_std,
                    "target_mean": target_mean,
                    "target_std": target_std,
                },
                "frequency": {
                    "frequency_cyc_per_pixel": frequency.tolist(),
                    "log2_amplitude_gain": log_gain.tolist(),
                    "smoothing_sigma_bins": 1.5,
                    "log2_gain_clip": [-1.0, 1.0],
                },
                "stain_target": {
                    "macenko": parameters_to_json(macenko),
                    "vahadane": parameters_to_json(vahadane),
                    "scanner": "at2",
                    "sampled_images": sampled_images,
                },
                "vahadane_configuration": selected_configuration,
                "endpoint_scale": endpoint_scale.to_dict(),
                "radial_curve": {
                    "frequency_cyc_per_pixel": source_curve.columns.to_numpy(float).tolist(),
                    "scale": curve_scale.to_numpy(float).tolist(),
                },
                "fit_provenance": {
                    "sample_index_sha256": sha256(sample_index_path),
                    "lab_statistics_sha256": sha256(lab_statistics_path),
                    "spectra_sha256": sha256(FINAL_IMAGE_ROOT / "03_frequency/spectra.csv"),
                    "vahadane_selection_manifest_path": str(vahadane_selection_manifest.resolve()),
                    "vahadane_selection_manifest_sha256": sha256(vahadane_selection_manifest),
                    "stain_v4_validation_manifest_path": str(stain_v4_manifest.resolve()),
                    "stain_v4_validation_manifest_sha256": sha256(stain_v4_manifest),
                    "stain_code_sha256": sha256(Path(__file__).with_name("stain.py")),
                    "paired_frequency_rows": len(paired),
                },
            }
            path = parameter_root / scanner / f"fold_{fold}.json"
            write_json(path, payload)
            parameters.append(
                {
                    "scanner": scanner,
                    "fold": fold,
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                    "training_slide_count": len(training_slides),
                    "validation_fold": (fold + 1) % 5,
                    "status": "frozen",
                }
            )
    table = pd.DataFrame(parameters)
    if len(table) != 15 or not table.groupby("scanner")["fold"].nunique().eq(5).all():
        raise ValueError("parameter table is incomplete")
    table_path = parameter_root / "parameter_manifest.csv"
    write_frame(table_path, table)
    manifest = {
        "analysis_version": VERSION,
        "status": "frozen_before_external_outcomes",
        "created_utc": utc_now(),
        "contract_sha256": contract_hash,
        "parameter_blocks": len(table),
        "methods": list(METHODS),
        "training_rule": "exactly 3 PanNormal train folds; validation and test excluded",
        "plism_fit_rows": 0,
        "paired_plism_at2_fit_rows": 0,
        "plism_source_patch_application_basis": "Macenko and Vahadane; independently estimated per source patch at transform application, never pooled or selected against AT2",
        "vahadane": {
            "status": "global_input_only_configuration_selected_before_target_outcomes",
            "configuration": selected_configuration,
            "selection_manifest_path": str(vahadane_selection_manifest.resolve()),
            "selection_manifest_sha256": sha256(vahadane_selection_manifest),
            "stain_v4_validation_manifest": {
                "path": str(stain_v4_manifest.resolve()),
                "sha256": sha256(stain_v4_manifest),
                "status": "pass",
            },
        },
        "parameter_manifest": {"path": str(table_path.resolve()), "sha256": sha256(table_path)},
    }
    write_json(parameter_root / "manifest.json", manifest)
    return manifest


def _parameter_blocks(output_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load all frozen parameter blocks after checking their recorded hashes."""

    _, contract_hash = _validated_contract(output_root)
    manifest_path = output_root / "02_parameters/manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "frozen_before_external_outcomes":
        raise RuntimeError("PLISM parameters are not frozen")
    if manifest.get("contract_sha256") != contract_hash:
        raise RuntimeError("PLISM parameter/contract mismatch")
    stain_gate = manifest.get("vahadane", {}).get("stain_v4_validation_manifest", {})
    if (
        not stain_gate
        or not Path(str(stain_gate.get("path", ""))).is_file()
        or sha256(stain_gate["path"]) != stain_gate.get("sha256")
    ):
        raise RuntimeError("formal selected-MU v4 stain validation manifest changed")
    table_path = Path(manifest["parameter_manifest"]["path"])
    if sha256(table_path) != manifest["parameter_manifest"]["sha256"]:
        raise RuntimeError("PLISM parameter manifest changed")
    table = pd.read_csv(table_path)
    if len(table) != 15 or table[["scanner", "fold"]].duplicated().any():
        raise ValueError("PLISM parameter manifest must contain 15 unique blocks")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for item in table.itertuples(index=False):
        path = Path(str(item.path))
        if sha256(path) != str(item.sha256):
            raise RuntimeError(f"frozen parameter changed: {path}")
        payload = read_json(path)
        key = (str(item.scanner), int(item.fold))
        if payload.get("contract_sha256") != contract_hash or payload.get("fit_split_role") != "train":
            raise RuntimeError(f"invalid frozen parameter provenance: {path}")
        if int(payload.get("training_slide_count", 0)) not in {61, 62, 63}:
            raise RuntimeError(f"invalid three-fold training count: {path}")
        result[key] = payload
    if set(result) != {(scanner, fold) for scanner in SCANNERS for fold in range(5)}:
        raise ValueError("PLISM parameter grid is incomplete")
    return result


def _external_task_key(output_root: Path, task_index: int) -> tuple[str, str]:
    contract, _ = _validated_contract(output_root)
    selected = pd.read_csv(contract["outputs"]["selected_locations"]["path"])
    stains = sorted(selected["stain"].astype(str).unique())
    keys = [(scanner, stain) for scanner in SCANNERS for stain in stains]
    if not 0 <= int(task_index) < len(keys):
        raise IndexError(task_index)
    return keys[int(task_index)]


def _render_path(output_root: Path, scanner: str, stain: str) -> Path:
    upper = scanner.upper()
    if upper in {TARGET_SCANNER, "GT450"}:
        return EXISTING_RENDER_ROOT / upper / f"{stain}.h5"
    return output_root / "01_rendered_inputs" / upper / f"{stain}.h5"


def _verify_frozen_render(output_root: Path, scanner: str, stain: str, path: Path) -> None:
    contract, _ = _validated_contract(output_root)
    upper = scanner.upper()
    table = pd.read_csv(contract["outputs"]["frozen_renders"]["path"])
    match = table[
        table["scanner"].astype(str).str.upper().eq(upper)
        & table["stain"].astype(str).eq(stain)
    ]
    if len(match) != 1 or str(match.iloc[0]["sha256"]) != sha256(path):
        raise RuntimeError(f"frozen v4 render changed: {upper}/{stain}")


def _load_render(path: Path, locations: np.ndarray) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as store:
        observed = np.asarray(store["location"], dtype=np.int64)
        if not np.array_equal(observed, locations):
            raise ValueError(f"render locations differ from section contract: {path}")
        images = np.asarray(store["images"], dtype=np.uint8)
    if images.shape != (len(locations), TARGET_PX, TARGET_PX, 3):
        raise ValueError(f"invalid render image shape: {path} {images.shape}")
    return images


def _stain_source_parameters(
    image: np.ndarray, seed: int, vahadane_configuration: Mapping[str, Any]
) -> dict[str, Any]:
    from manuscript_completion.stain import fit_macenko, fit_vahadane

    return {
        "macenko": fit_macenko(image),
        "vahadane": fit_vahadane(
            image,
            seed=seed,
            max_pixels=int(vahadane_configuration["max_pixels_source"]),
            alpha=float(vahadane_configuration["alpha_W"]),
            max_iter=int(vahadane_configuration["max_iter"]),
            tolerance=float(vahadane_configuration["tolerance"]),
            solver=str(vahadane_configuration["solver"]),
        ),
    }


def _apply_stain_from_source(
    image: np.ndarray, source: Any, target: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a target basis while reusing a single deterministic source fit."""

    from manuscript_completion.stain import _concentrations, od_to_rgb, optical_density

    report = {
        "stain_fallback": source is None,
        "stain_preclip_fraction": 0.0,
        "stain_converged": bool(source.converged) if source is not None else False,
        "stain_iterations": int(source.iterations) if source is not None else 0,
        "stain_reconstruction_error": float(source.reconstruction_error) if source is not None else 0.0,
    }
    if source is None:
        return image.copy(), report
    try:
        od = optical_density(image.reshape(-1, 3))
        concentration = _concentrations(od, source.basis)
        scaled = concentration * (target.maximum[None] / np.maximum(source.maximum[None], 1e-4))
        corrected, clipped = od_to_rgb((scaled @ target.basis.T).reshape(image.shape))
        if not np.isfinite(corrected).all():
            raise ValueError("non-finite stain correction")
        report["stain_preclip_fraction"] = float(clipped)
        return corrected, report
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        report["stain_fallback"] = True
        return image.copy(), report


def _normalized_log_power(radial: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(radial, dtype=np.float64), 1e-12)
    value /= np.maximum(value.sum(axis=-1, keepdims=True), 1e-12)
    return np.log2(value)


def _radial_residual_rmse(
    candidate: np.ndarray, target: np.ndarray, parameter: Mapping[str, Any]
) -> float:
    residual = _normalized_log_power(candidate) - _normalized_log_power(target)
    scale = np.asarray(parameter["radial_curve"]["scale"], dtype=float)
    if scale.shape != residual.shape:
        source_grid = np.linspace(0.0, 1.0, len(scale))
        scale = np.interp(np.linspace(0.0, 1.0, len(residual)), source_grid, scale)
    scale = np.maximum(scale, 0.05)
    return float(np.sqrt(np.mean(np.square(residual / scale))))


def _evaluate_candidates(
    *,
    source: np.ndarray,
    target: np.ndarray,
    candidates: Sequence[np.ndarray],
    conditions: Sequence[tuple[str, int]],
    parameters: Mapping[int, Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    workers: int,
) -> list[dict[str, Any]]:
    if len(candidates) != len(conditions) or len(reports) != len(conditions):
        raise ValueError("candidate metadata length mismatch")
    source_scalar, source_radial, _, _ = absolute_measurements(source[None], workers)
    target_scalar, target_radial, _, _ = absolute_measurements(target[None], workers)
    target_vector = contrasts_from_absolute(target_scalar, target_radial, source_scalar, source_radial)[0]
    array = np.stack(candidates)
    scalar, radial, _, _ = absolute_measurements(array, workers)
    generated = contrasts_from_absolute(scalar, radial, source_scalar, source_radial)
    safety = image_metrics(
        np.repeat(source[None], len(array), axis=0),
        array,
        np.repeat(target[None], len(array), axis=0),
    )
    rows: list[dict[str, Any]] = []
    for index, ((method, fold), report) in enumerate(zip(conditions, reports)):
        parameter = parameters[int(fold)]
        scale = np.maximum(
            np.asarray([parameter["endpoint_scale"][name] for name in ENDPOINTS], dtype=float),
            1e-6,
        )
        residual = (generated[index] - target_vector) / scale
        row: dict[str, Any] = {
            **metadata,
            "method": method,
            "parameter_fold": int(fold),
            "distance": float(np.sqrt(np.mean(np.square(residual)))),
            "endpoint_coverage": float(np.mean(np.abs(residual) <= 1.0)),
            "joint_coverage": bool(np.all(np.abs(residual) <= 1.0)),
            "full_radial_residual": _radial_residual_rmse(radial[index], target_radial[0], parameter),
            **{name: float(value) for name, value in safety.iloc[index].items()},
            **dict(report),
        }
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            row[f"target_{endpoint}"] = float(target_vector[endpoint_index])
            row[f"generated_{endpoint}"] = float(generated[index, endpoint_index])
            row[f"scaled_residual_{endpoint}"] = float(residual[endpoint_index])
        rows.append(row)
    return rows


def external_image_task(
    task_index: int,
    output_root: Path = DEFAULT_OUTPUT,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    """Apply all conventional methods to one external scanner/section."""

    from manuscript_completion.stain import parameters_from_json

    output_root = output_root.resolve()
    contract, contract_hash = _validated_contract(output_root)
    scanner, stain = _external_task_key(output_root, task_index)
    blocks = _parameter_blocks(output_root)
    section_parameters = {fold: blocks[(scanner, fold)] for fold in range(5)}
    vahadane_configuration = section_parameters[0]["vahadane_configuration"]
    if any(
        section_parameters[fold]["vahadane_configuration"] != vahadane_configuration
        for fold in range(1, 5)
    ):
        raise RuntimeError("Vahadane configuration differs across parameter folds")
    parameter_hash = hashlib.sha256(
        "|".join(
            sha256(output_root / "02_parameters" / scanner / f"fold_{fold}.json")
            for fold in range(5)
        ).encode()
    ).hexdigest()
    output = output_root / "03_external_image/shards" / scanner / f"{stain}.csv.gz"
    image_output = output_root / "03_external_image/corrected" / scanner / f"{stain}.h5"
    summary_path = output.with_suffix(".summary.json")
    if output.exists() and image_output.exists() and summary_path.exists():
        old = read_json(summary_path)
        if (
            old.get("status") == "complete"
            and old.get("contract_sha256") == contract_hash
            and old.get("parameter_bundle_sha256") == parameter_hash
            and old.get("output_sha256") == sha256(output)
            and old.get("image_output_sha256") == sha256(image_output)
        ):
            return {"status": "already_complete", **old}
        raise FileExistsError(f"stale external image task: {scanner}/{stain}")

    selected = pd.read_csv(contract["outputs"]["selected_locations"]["path"])
    section = selected[selected["stain"].astype(str).eq(stain)].sort_values("location", kind="stable")
    locations = section["location"].to_numpy(np.int64)
    source_path = _render_path(output_root, scanner, stain)
    target_path = _render_path(output_root, "at2", stain)
    _verify_frozen_render(output_root, scanner, stain, source_path)
    _verify_frozen_render(output_root, "at2", stain, target_path)
    source_images = _load_render(source_path, locations)
    target_images = _load_render(target_path, locations)
    corrected_conditions = [(method, fold) for fold in range(5) for method in CORRECTED_METHODS]
    condition_names = [f"{method}:fold_{fold}" for method, fold in corrected_conditions]
    image_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_image = image_output.with_suffix(f".h5.{os.getpid()}.tmp")
    result_rows: list[dict[str, Any]] = []
    with h5py.File(temporary_image, "w") as store:
        store.attrs.update(
            analysis_version=VERSION,
            contract_sha256=contract_hash,
            scanner=scanner,
            stain=stain,
            parameter_bundle_sha256=parameter_hash,
        )
        store.create_dataset("location", data=locations)
        store.create_dataset("condition_names", data=np.asarray(condition_names, dtype="S32"))
        corrected_store = store.create_dataset(
            "images",
            shape=(len(locations), len(corrected_conditions), TARGET_PX, TARGET_PX, 3),
            dtype=np.uint8,
            compression="lzf",
            chunks=(1, 1, TARGET_PX, TARGET_PX, 3),
        )
        for image_index, item in enumerate(section.itertuples(index=False)):
            source = source_images[image_index]
            target = target_images[image_index]
            source_stain = _stain_source_parameters(
                source,
                stable_seed("plism-source", scanner, stain, int(item.location)),
                vahadane_configuration,
            )
            corrected: list[np.ndarray] = []
            reports: list[dict[str, Any]] = []
            for method, fold in corrected_conditions:
                parameter = section_parameters[fold]
                report: dict[str, Any] = {
                    "stain_fallback": False,
                    "stain_preclip_fraction": 0.0,
                    "stain_converged": True,
                    "stain_iterations": 0,
                    "stain_reconstruction_error": 0.0,
                }
                if method == "reinhard":
                    generated = apply_reinhard(source, parameter)
                elif method == "frequency":
                    generated = apply_frequency(source, parameter)
                elif method == "combined":
                    generated = apply_frequency(apply_reinhard(source, parameter), parameter)
                else:
                    target_stain = parameters_from_json(parameter["stain_target"][method])
                    generated, report = _apply_stain_from_source(
                        source, source_stain[method], target_stain
                    )
                corrected.append(generated)
                reports.append(report)
            corrected_store[image_index] = np.stack(corrected)
            common = {
                "scanner": scanner,
                "section": stain,
                "location": int(item.location),
                "core": int(item.core),
                "replicate": int(item.replicate),
                "tissue_type": str(item.pannormal_tissue),
                "mapping_tier": str(item.mapping_tier),
                "organ_aligned_status": str(item.organ_aligned_status),
                "strict_status": str(item.strict_status),
            }
            raw_conditions = [("raw", fold) for fold in range(5)]
            raw_reports = [
                {
                    "stain_fallback": False,
                    "stain_preclip_fraction": 0.0,
                    "stain_converged": True,
                    "stain_iterations": 0,
                    "stain_reconstruction_error": 0.0,
                }
                for _ in range(5)
            ]
            result_rows.extend(
                _evaluate_candidates(
                    source=source,
                    target=target,
                    candidates=[source] * 5 + corrected,
                    conditions=raw_conditions + corrected_conditions,
                    parameters=section_parameters,
                    reports=raw_reports + reports,
                    metadata=common,
                    workers=workers,
                )
            )
            if (image_index + 1) % 10 == 0 or image_index + 1 == len(section):
                print(f"[{scanner}/{stain}] {image_index + 1}/{len(section)}", flush=True)
    temporary_image.replace(image_output)
    frame = pd.DataFrame(result_rows)
    expected = len(section) * len(METHODS) * 5
    if len(frame) != expected or not np.isfinite(frame.select_dtypes("number")).all().all():
        raise ValueError(f"invalid external image shard {scanner}/{stain}: {len(frame)}")
    write_frame(output, frame)
    summary = {
        "analysis_version": VERSION,
        "status": "complete",
        "created_utc": utc_now(),
        "contract_sha256": contract_hash,
        "parameter_bundle_sha256": parameter_hash,
        "task_index": int(task_index),
        "scanner": scanner,
        "section": stain,
        "locations": len(section),
        "rows": len(frame),
        "source_render": {"path": str(source_path.resolve()), "sha256": sha256(source_path)},
        "target_render": {"path": str(target_path.resolve()), "sha256": sha256(target_path)},
        "output_path": str(output.resolve()),
        "output_sha256": sha256(output),
        "image_output_path": str(image_output.resolve()),
        "image_output_sha256": sha256(image_output),
    }
    write_json(summary_path, summary)
    return summary


def external_uni_task(
    task_index: int,
    output_root: Path = DEFAULT_OUTPUT,
    *,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    """Extract frozen UNI-v1 endpoints from one corrected external section."""

    import torch

    output_root = output_root.resolve()
    contract, contract_hash = _validated_contract(output_root)
    scanner, stain = _external_task_key(output_root, task_index)
    image_summary_path = output_root / "03_external_image/shards" / scanner / f"{stain}.csv.summary.json"
    if not image_summary_path.is_file():
        # Path.with_suffix on ``*.csv.gz`` produces ``*.csv.summary.json``.
        raise FileNotFoundError(image_summary_path)
    image_summary = read_json(image_summary_path)
    corrected_path = Path(image_summary["image_output_path"])
    if image_summary.get("image_output_sha256") != sha256(corrected_path):
        raise RuntimeError("corrected external image shard changed")
    output = output_root / "04_external_uni/shards" / scanner / f"{stain}.csv.gz"
    feature_output = output_root / "04_external_uni/features" / scanner / f"{stain}.h5"
    summary_path = output.with_suffix(".summary.json")
    if output.exists() and feature_output.exists() and summary_path.exists():
        old = read_json(summary_path)
        if (
            old.get("status") == "complete"
            and old.get("contract_sha256") == contract_hash
            and old.get("corrected_image_sha256") == sha256(corrected_path)
            and old.get("output_sha256") == sha256(output)
            and old.get("feature_output_sha256") == sha256(feature_output)
        ):
            return {"status": "already_complete", **old}
        raise FileExistsError(f"stale external UNI task: {scanner}/{stain}")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("external UNI-v1 extraction requires CUDA")
    selected = pd.read_csv(contract["outputs"]["selected_locations"]["path"])
    section = selected[selected["stain"].astype(str).eq(stain)].sort_values("location", kind="stable")
    locations = section["location"].to_numpy(np.int64)
    source_path = _render_path(output_root, scanner, stain)
    target_path = _render_path(output_root, "at2", stain)
    _verify_frozen_render(output_root, scanner, stain, source_path)
    _verify_frozen_render(output_root, "at2", stain, target_path)
    source = _load_render(source_path, locations)
    target = _load_render(target_path, locations)
    with h5py.File(corrected_path, "r") as store:
        corrected_locations = np.asarray(store["location"], dtype=np.int64)
        if not np.array_equal(corrected_locations, locations):
            raise ValueError("corrected image locations changed")
        names = [value.decode() for value in store["condition_names"][:]]
        if len(names) != 5 * len(CORRECTED_METHODS):
            raise ValueError("corrected external conditions do not match the frozen method grid")
    resolved = torch.device(device)
    model, size, mean, std, checkpoint = load_frozen_uni1(resolved)
    checkpoint_hash = sha256(checkpoint)
    source_features = embed_frozen_uni(
        model, source, size, mean, std, resolved, batch_size=batch_size, value_range="uint8"
    )
    target_features = embed_frozen_uni(
        model, target, size, mean, std, resolved, batch_size=batch_size, value_range="uint8"
    )
    corrected_features = np.empty((len(locations), len(names), source_features.shape[1]), dtype=np.float32)
    with h5py.File(corrected_path, "r") as store:
        for condition_index, name in enumerate(names):
            images = np.asarray(store["images"][:, condition_index], dtype=np.uint8)
            corrected_features[:, condition_index] = embed_frozen_uni(
                model, images, size, mean, std, resolved,
                batch_size=batch_size, value_range="uint8",
            )
            print(f"[{scanner}/{stain}] UNI {condition_index + 1}/{len(names)} {name}", flush=True)
    source_target = cosine_distance_rows(source_features, target_features)
    records: list[dict[str, Any]] = []
    condition_index = {name: index for index, name in enumerate(names)}
    for row_index, item in enumerate(section.itertuples(index=False)):
        common = {
            "scanner": scanner,
            "section": stain,
            "location": int(item.location),
            "core": int(item.core),
            "replicate": int(item.replicate),
            "tissue_type": str(item.pannormal_tissue),
            "mapping_tier": str(item.mapping_tier),
            "organ_aligned_status": str(item.organ_aligned_status),
            "strict_status": str(item.strict_status),
        }
        raw_distance = float(source_target[row_index])
        records.append(
            {
                **common,
                "method": "raw",
                "parameter_fold": -1,
                "source_target_distance": raw_distance,
                "corrected_target_distance": raw_distance,
                "corrected_source_distance": 0.0,
                "uni_target_gain": 0.0,
                "uni_closure": 0.0,
            }
        )
        for name in names:
            method, fold_text = name.split(":fold_")
            feature = corrected_features[row_index, condition_index[name]][None]
            corrected_target = float(cosine_distance_rows(feature, target_features[row_index][None])[0])
            corrected_source = float(cosine_distance_rows(feature, source_features[row_index][None])[0])
            gain = raw_distance - corrected_target
            records.append(
                {
                    **common,
                    "method": method,
                    "parameter_fold": int(fold_text),
                    "source_target_distance": raw_distance,
                    "corrected_target_distance": corrected_target,
                    "corrected_source_distance": corrected_source,
                    "uni_target_gain": gain,
                    "uni_closure": gain / max(raw_distance, 1e-8),
                }
            )
    frame = pd.DataFrame(records)
    if len(frame) != len(section) * (1 + 5 * len(CORRECTED_METHODS)) or not np.isfinite(frame.select_dtypes("number")).all().all():
        raise ValueError(f"invalid external UNI shard {scanner}/{stain}")
    write_frame(output, frame)
    feature_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = feature_output.with_suffix(f".h5.{os.getpid()}.tmp")
    with h5py.File(temporary, "w") as store:
        store.attrs.update(
            analysis_version=VERSION,
            contract_sha256=contract_hash,
            scanner=scanner,
            stain=stain,
            uni_checkpoint_sha256=checkpoint_hash,
        )
        store.create_dataset("location", data=locations)
        store.create_dataset("condition_names", data=np.asarray(names, dtype="S32"))
        store.create_dataset("source_features", data=source_features, compression="lzf")
        store.create_dataset("target_features", data=target_features, compression="lzf")
        store.create_dataset("corrected_features", data=corrected_features, compression="lzf")
    temporary.replace(feature_output)
    summary = {
        "analysis_version": VERSION,
        "status": "complete",
        "created_utc": utc_now(),
        "contract_sha256": contract_hash,
        "task_index": int(task_index),
        "scanner": scanner,
        "section": stain,
        "locations": len(section),
        "rows": len(frame),
        "uni_checkpoint_path": str(checkpoint.resolve()),
        "uni_checkpoint_sha256": checkpoint_hash,
        "corrected_image_sha256": sha256(corrected_path),
        "output_path": str(output.resolve()),
        "output_sha256": sha256(output),
        "feature_output_path": str(feature_output.resolve()),
        "feature_output_sha256": sha256(feature_output),
    }
    write_json(summary_path, summary)
    return summary


def _internal_task_key(task_index: int, locked_index_path: Path) -> tuple[str, str]:
    locked = pd.read_csv(locked_index_path, dtype={"slide_id": str}, usecols=["slide_id"])
    slide_ids = sorted(locked["slide_id"].unique())
    keys = [(scanner, slide_id) for scanner in SCANNERS for slide_id in slide_ids]
    if not 0 <= int(task_index) < len(keys):
        raise IndexError(task_index)
    return keys[int(task_index)]


def internal_image_task(
    task_index: int,
    output_root: Path = DEFAULT_OUTPUT,
    locked_index_path: Path = LOCKED_INDEX,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    """Evaluate a direction-matched PanNormal outer-test slide comparator."""

    from manuscript_completion.stain import parameters_from_json

    output_root = output_root.resolve()
    _, contract_hash = _validated_contract(output_root)
    scanner, slide_id = _internal_task_key(task_index, locked_index_path)
    blocks = _parameter_blocks(output_root)
    locked = pd.read_csv(locked_index_path, dtype={"slide_id": str})
    rows = locked[locked["slide_id"].eq(slide_id)].sort_values("location_index", kind="stable")
    if len(rows) != 40:
        raise ValueError(f"expected 40 locked internal locations for {slide_id}")
    fold = int(rows["fold"].iloc[0])
    parameter = blocks[(scanner, fold)]
    output = output_root / "05_internal_image/shards" / scanner / f"{slide_id}.csv.gz"
    summary_path = output.with_suffix(".summary.json")
    parameter_path = output_root / "02_parameters" / scanner / f"fold_{fold}.json"
    parameter_hash = sha256(parameter_path)
    if output.exists() and summary_path.exists():
        old = read_json(summary_path)
        if (
            old.get("status") == "complete"
            and old.get("contract_sha256") == contract_hash
            and old.get("parameter_sha256") == parameter_hash
            and old.get("output_sha256") == sha256(output)
        ):
            return {"status": "already_complete", **old}
        raise FileExistsError(f"stale internal image task: {scanner}/{slide_id}")
    scanner_names = tuple(str(rows["scanner_order"].iloc[0]).split(","))
    try:
        source_index = scanner_names.index(scanner)
        target_index = scanner_names.index("at2")
    except ValueError as error:
        raise ValueError(f"scanner order lacks {scanner}/at2 for {slide_id}") from error
    cache_path = Path(str(rows["cache_path"].iloc[0]))
    result_rows: list[dict[str, Any]] = []
    with h5py.File(cache_path, "r") as store:
        for offset, item in enumerate(rows.itertuples(index=False)):
            location = int(item.location_index)
            source = np.asarray(store["images"][location, source_index], dtype=np.uint8)
            target = np.asarray(store["images"][location, target_index], dtype=np.uint8)
            source_stain = _stain_source_parameters(
                source,
                stable_seed("internal-source", scanner, slide_id, location),
                parameter["vahadane_configuration"],
            )
            candidates = [source]
            reports: list[dict[str, Any]] = [
                {
                    "stain_fallback": False,
                    "stain_preclip_fraction": 0.0,
                    "stain_converged": True,
                    "stain_iterations": 0,
                    "stain_reconstruction_error": 0.0,
                }
            ]
            for method in CORRECTED_METHODS:
                report = reports[0].copy()
                if method == "reinhard":
                    generated = apply_reinhard(source, parameter)
                elif method == "frequency":
                    generated = apply_frequency(source, parameter)
                elif method == "combined":
                    generated = apply_frequency(apply_reinhard(source, parameter), parameter)
                else:
                    target_stain = parameters_from_json(parameter["stain_target"][method])
                    generated, report = _apply_stain_from_source(
                        source, source_stain[method], target_stain
                    )
                candidates.append(generated)
                reports.append(report)
            result_rows.extend(
                _evaluate_candidates(
                    source=source,
                    target=target,
                    candidates=candidates,
                    conditions=[(method, fold) for method in METHODS],
                    parameters={fold: parameter},
                    reports=reports,
                    metadata={
                        "scanner": scanner,
                        "slide_id": slide_id,
                        "tissue_type": str(item.tissue_type),
                        "outer_test_fold": fold,
                        "location": location,
                        "source_index": int(item.source_index),
                    },
                    workers=workers,
                )
            )
            if (offset + 1) % 10 == 0:
                print(f"[{scanner}/{slide_id}] {offset + 1}/40", flush=True)
    frame = pd.DataFrame(result_rows)
    if len(frame) != 40 * len(METHODS) or not np.isfinite(frame.select_dtypes("number")).all().all():
        raise ValueError(f"invalid internal image shard {scanner}/{slide_id}")
    write_frame(output, frame)
    summary = {
        "analysis_version": VERSION,
        "status": "complete",
        "created_utc": utc_now(),
        "contract_sha256": contract_hash,
        "task_index": int(task_index),
        "scanner": scanner,
        "slide_id": slide_id,
        "outer_test_fold": fold,
        "rows": len(frame),
        "parameter_sha256": parameter_hash,
        "locked_index_sha256": sha256(locked_index_path),
        "output_path": str(output.resolve()),
        "output_sha256": sha256(output),
    }
    write_json(summary_path, summary)
    return summary


def _bootstrap_mean(
    values: np.ndarray, *, seed: int, replicates: int
) -> tuple[float, float, float]:
    value = np.asarray(values, dtype=float)
    value = value[np.isfinite(value)]
    if not len(value):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(value), size=(int(replicates), len(value)))
    draws = value[indices].mean(axis=1)
    return float(value.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _external_separability(
    output_root: Path,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Leave-one-physical-section-out real-AT2 versus corrected classifier."""

    rows: list[dict[str, Any]] = []
    for scanner in SCANNERS:
        generated: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
        real: list[np.ndarray] = []
        sections: list[str] = []
        for stain in sorted(selected["stain"].astype(str).unique()):
            path = output_root / "04_external_uni/features" / scanner / f"{stain}.h5"
            if not path.is_file():
                raise FileNotFoundError(path)
            with h5py.File(path, "r") as store:
                names = [value.decode() for value in store["condition_names"][:]]
                source = np.asarray(store["source_features"], dtype=np.float32)
                target = np.asarray(store["target_features"], dtype=np.float32)
                corrected = np.asarray(store["corrected_features"], dtype=np.float32)
            lookup = {name: index for index, name in enumerate(names)}
            generated["raw"].append(source)
            for method in CORRECTED_METHODS:
                indices = [lookup[f"{method}:fold_{fold}"] for fold in range(5)]
                average = corrected[:, indices].mean(axis=1)
                average /= np.maximum(np.linalg.norm(average, axis=1, keepdims=True), 1e-12)
                generated[method].append(average.astype(np.float32))
            real.append(target)
            sections.extend([stain] * len(target))
        real_array = np.concatenate(real)
        section_array = np.asarray(sections)
        unique_sections = sorted(set(sections))
        if len(unique_sections) != 13:
            raise ValueError("external separability requires 13 physical sections")
        for method in METHODS:
            generated_array = np.concatenate(generated[method])
            for held_section in unique_sections:
                held = section_array == held_section
                train = ~held
                x_train = np.concatenate([generated_array[train], real_array[train]])
                y_train = np.concatenate(
                    [np.zeros(train.sum(), dtype=int), np.ones(train.sum(), dtype=int)]
                )
                x_test = np.concatenate([generated_array[held], real_array[held]])
                y_test = np.concatenate(
                    [np.zeros(held.sum(), dtype=int), np.ones(held.sum(), dtype=int)]
                )
                classifier = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=0.1,
                        max_iter=1000,
                        solver="liblinear",
                        random_state=stable_seed("plism-separability", scanner, method, held_section),
                    ),
                )
                classifier.fit(x_train, y_train)
                predicted = classifier.predict(x_test)
                rows.append(
                    {
                        "scanner": scanner,
                        "section": held_section,
                        "method": method,
                        "separability_balanced_accuracy": float(
                            balanced_accuracy_score(y_test, predicted)
                        ),
                        "test_locations": int(held.sum()),
                        "classifier": "leave-one-physical-section-out standardized logistic regression C=0.1",
                    }
                )
    return pd.DataFrame(rows)


def _section_table(
    image: pd.DataFrame,
    uni: pd.DataFrame,
    separability: pd.DataFrame,
) -> pd.DataFrame:
    image_metrics_columns = [
        "distance",
        "endpoint_coverage",
        "joint_coverage",
        "full_radial_residual",
        "target_l1_unit",
        "target_psnr_db",
        "target_ssim",
        "source_gradient_ncc",
        "target_gradient_ncc",
        "saturation_fraction",
        "invented_edge_fraction",
        "source_edge_deletion_fraction",
        "stain_fallback",
        "stain_preclip_fraction",
        "stain_converged",
    ]
    uni_metrics_columns = [
        "source_target_distance",
        "corrected_target_distance",
        "corrected_source_distance",
        "uni_target_gain",
        "uni_closure",
    ]
    group = ["scanner", "section", "core", "organ_aligned_status", "method"]
    image_core = image.groupby(group, as_index=False)[image_metrics_columns].mean()
    uni_core = uni.groupby(group, as_index=False)[uni_metrics_columns].mean()
    core = image_core.merge(uni_core, on=group, how="outer", validate="one_to_one")
    primary = core.groupby(["scanner", "section", "method"], as_index=False)[
        image_metrics_columns + uni_metrics_columns
    ].mean()
    primary["organ_aligned_status"] = "all"
    primary["analysis_set"] = "primary_all"
    secondary = core.groupby(
        ["scanner", "section", "organ_aligned_status", "method"], as_index=False
    )[image_metrics_columns + uni_metrics_columns].mean()
    secondary["analysis_set"] = "organ_aligned_" + secondary["organ_aligned_status"].astype(str)
    section = pd.concat([primary, secondary], ignore_index=True, sort=False)
    section = section.merge(
        separability.drop(columns=["classifier"]),
        on=["scanner", "section", "method"],
        how="left",
        validate="many_to_one",
    )
    section.loc[~section["analysis_set"].eq("primary_all"), "separability_balanced_accuracy"] = np.nan
    section.loc[~section["analysis_set"].eq("primary_all"), "test_locations"] = np.nan
    raw = section[section["method"].eq("raw")][
        ["analysis_set", "scanner", "section", "organ_aligned_status", "distance", "full_radial_residual"]
    ].rename(
        columns={
            "distance": "raw_distance",
            "full_radial_residual": "raw_full_radial_residual",
        }
    )
    section = section.merge(
        raw,
        on=["analysis_set", "scanner", "section", "organ_aligned_status"],
        how="left",
        validate="many_to_one",
    )
    section["distance_gain"] = section["raw_distance"] - section["distance"]
    section["radial_gain"] = (
        section["raw_full_radial_residual"] - section["full_radial_residual"]
    )
    section["direction_improved"] = section["distance_gain"] > 0.0
    return section


def _method_summary(
    section: pd.DataFrame,
    *,
    bootstraps: int,
) -> pd.DataFrame:
    metric_columns = [
        "distance",
        "distance_gain",
        "endpoint_coverage",
        "joint_coverage",
        "full_radial_residual",
        "radial_gain",
        "source_gradient_ncc",
        "target_gradient_ncc",
        "saturation_fraction",
        "invented_edge_fraction",
        "source_edge_deletion_fraction",
        "stain_fallback",
        "stain_preclip_fraction",
        "stain_converged",
        "uni_target_gain",
        "uni_closure",
        "separability_balanced_accuracy",
        "direction_improved",
    ]
    primary = section[section["analysis_set"].eq("primary_all")]
    analysis_sets: list[tuple[str, pd.DataFrame]] = [("primary_all", primary)]
    analysis_sets.append(
        (
            "exclude_s60_hrh_oof",
            primary[~(primary["scanner"].eq(HRH_OOF_SCANNER) & primary["section"].eq(HRH_OOF_SECTION))],
        )
    )
    for status in ("shared", "novel"):
        analysis_sets.append(
            (
                f"organ_aligned_{status}",
                section[section["analysis_set"].eq(f"organ_aligned_{status}")],
            )
        )
    rows: list[dict[str, Any]] = []
    for analysis_set, subset in analysis_sets:
        for (scanner, method), group in subset.groupby(["scanner", "method"], sort=True):
            for metric in metric_columns:
                if not np.isfinite(group[metric].to_numpy(float)).any():
                    continue
                estimate, low, high = _bootstrap_mean(
                    group[metric].to_numpy(float),
                    seed=stable_seed("plism-bootstrap", analysis_set, scanner, method, metric),
                    replicates=bootstraps,
                )
                rows.append(
                    {
                        "analysis_set": analysis_set,
                        "scanner": scanner,
                        "method": method,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "sections": int(group["section"].nunique()),
                        "locations": int(group["test_locations"].sum())
                        if metric == "separability_balanced_accuracy"
                        and group["test_locations"].notna().any()
                        else int(group["section"].nunique()),
                        "bootstrap_replicates": int(bootstraps),
                        "cluster_unit": "physical_section",
                    }
                )
    return pd.DataFrame(rows)


def aggregate_results(
    output_root: Path = DEFAULT_OUTPUT,
    *,
    bootstraps: int = DEFAULT_BOOTSTRAPS,
) -> dict[str, Any]:
    """Aggregate external evidence and direction-matched internal retention."""

    output_root = output_root.resolve()
    contract, contract_hash = _validated_contract(output_root)
    parameter_manifest_path = output_root / "02_parameters/manifest.json"
    parameter_manifest = read_json(parameter_manifest_path)
    if parameter_manifest.get("status") != "frozen_before_external_outcomes":
        raise RuntimeError("parameter manifest is incomplete")
    selected = pd.read_csv(contract["outputs"]["selected_locations"]["path"])
    image_paths = sorted((output_root / "03_external_image/shards").glob("*/*.csv.gz"))
    uni_paths = sorted((output_root / "04_external_uni/shards").glob("*/*.csv.gz"))
    internal_paths = sorted((output_root / "05_internal_image/shards").glob("*/*.csv.gz"))
    if len(image_paths) != 39 or len(uni_paths) != 39 or len(internal_paths) != 309:
        raise ValueError(
            f"incomplete PLISM shards: image={len(image_paths)}, uni={len(uni_paths)}, internal={len(internal_paths)}"
        )
    image = pd.concat([pd.read_csv(path) for path in image_paths], ignore_index=True)
    uni = pd.concat([pd.read_csv(path) for path in uni_paths], ignore_index=True)
    internal = pd.concat(
        [pd.read_csv(path, dtype={"slide_id": str}) for path in internal_paths], ignore_index=True
    )
    if image[["scanner", "section", "location", "method", "parameter_fold"]].duplicated().any():
        raise ValueError("duplicate external image result")
    if uni[["scanner", "section", "location", "method", "parameter_fold"]].duplicated().any():
        raise ValueError("duplicate external UNI result")
    if internal[["scanner", "slide_id", "location", "method"]].duplicated().any():
        raise ValueError("duplicate internal image result")
    numeric_frames = (image, uni, internal)
    if any(not np.isfinite(frame.select_dtypes("number")).all().all() for frame in numeric_frames):
        raise ValueError("non-finite PLISM or internal endpoint")
    separability = _external_separability(output_root, selected)
    section = _section_table(image, uni, separability)
    summary = _method_summary(section, bootstraps=bootstraps)

    internal_slide = internal.groupby(["scanner", "slide_id", "method"], as_index=False)[
        ["distance", "full_radial_residual"]
    ].mean()
    internal_raw = internal_slide[internal_slide["method"].eq("raw")][
        ["scanner", "slide_id", "distance", "full_radial_residual"]
    ].rename(columns={"distance": "raw_distance", "full_radial_residual": "raw_radial"})
    internal_slide = internal_slide.merge(
        internal_raw, on=["scanner", "slide_id"], how="left", validate="many_to_one"
    )
    internal_slide["distance_gain"] = internal_slide["raw_distance"] - internal_slide["distance"]
    internal_slide["radial_gain"] = internal_slide["raw_radial"] - internal_slide["full_radial_residual"]
    external_primary = section[section["analysis_set"].eq("primary_all")].groupby(
        ["scanner", "method"], as_index=False
    )[
        ["distance_gain", "radial_gain", "direction_improved"]
    ].mean()
    internal_primary = internal_slide.groupby(["scanner", "method"], as_index=False)[
        ["distance_gain", "radial_gain"]
    ].mean().rename(
        columns={
            "distance_gain": "internal_distance_gain",
            "radial_gain": "internal_radial_gain",
        }
    )
    contrast = external_primary.merge(
        internal_primary, on=["scanner", "method"], validate="one_to_one"
    ).rename(
        columns={
            "distance_gain": "external_distance_gain",
            "radial_gain": "external_radial_gain",
            "direction_improved": "external_section_direction_consistency",
        }
    )
    contrast["distance_gain_retention"] = np.where(
        np.abs(contrast["internal_distance_gain"]) >= 1e-6,
        contrast["external_distance_gain"] / contrast["internal_distance_gain"],
        0.0,
    )
    contrast["radial_gain_retention"] = np.where(
        np.abs(contrast["internal_radial_gain"]) >= 1e-6,
        contrast["external_radial_gain"] / contrast["internal_radial_gain"],
        0.0,
    )
    contrast["distance_external_minus_internal"] = (
        contrast["external_distance_gain"] - contrast["internal_distance_gain"]
    )

    summary_path = output_root / "plism_external_method_summary.csv"
    section_path = output_root / "plism_external_section_metrics.csv"
    contrast_path = output_root / "plism_internal_external_contrast.csv"
    decision_path = output_root / "plism_external_correction_decision.md"
    write_frame(summary_path, summary)
    write_frame(section_path, section)
    write_frame(contrast_path, contrast)
    primary_gain = summary[
        summary["analysis_set"].eq("primary_all")
        & summary["metric"].eq("distance_gain")
        & ~summary["method"].eq("raw")
    ]
    robust = primary_gain[primary_gain["ci_low"] > 0]
    lines = [
        "# PLISM external conventional-correction decision",
        "",
        f"Status: **{'direction-specific external support' if len(robust) else 'external correction inconclusive'}**.",
        "",
        "All cohort-level correction parameters and Macenko/Vahadane target stain references were frozen from exactly three PanNormal training folds before external outcomes were opened. Paired PLISM AT2 was evaluation-only; each source stain basis was estimated independently within its input source patch. Vahadane used one globally selected input-only MU configuration.",
        "",
        f"Across 3 scanner directions × 5 corrected methods, {len(robust)} direction/method cells had a positive 95% physical-section bootstrap interval for ten-endpoint residual gain. This is descriptive across a multi-method panel and is not a multiplicity-adjusted universal-effect claim.",
        "",
        "Section/stain identity is inseparable in this 13-section panel. The S60/HRH exclusion and organ-aligned shared/novel rows are sensitivity analyses, not independent cohorts.",
        "",
        "Real-versus-corrected UNI-v1 separability used leave-one-physical-section-out prediction; 0.5 is chance and larger values indicate residual domain information.",
    ]
    decision_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "analysis_version": VERSION,
        "status": "complete",
        "created_utc": utc_now(),
        "contract_sha256": contract_hash,
        "parameter_manifest_sha256": sha256(parameter_manifest_path),
        "external_locations": int(selected[["stain", "location"]].drop_duplicates().shape[0]),
        "physical_sections": int(selected["stain"].nunique()),
        "scanners": list(SCANNERS),
        "methods": list(METHODS),
        "image_shards": len(image_paths),
        "uni_shards": len(uni_paths),
        "internal_shards": len(internal_paths),
        "bootstrap_replicates": int(bootstraps),
        "aggregation": "location -> core -> physical section; parameter folds averaged within core/section; physical-section bootstrap",
        "multiplicity_boundary": "method/direction results are descriptive; positive cells are not interpreted as a universal multiplicity-adjusted effect",
        "stain_source_application": {
            "fallback_rows": int(image["stain_fallback"].astype(bool).sum()),
            "nonconverged_rows": int((~image["stain_converged"].astype(bool)).sum()),
            "interpretation": "source-patch stain-basis estimation is a declared transform step; Vahadane uses the single global MU configuration selected and validated before PLISM target outcomes",
        },
        "vahadane": {
            "status": "global_input_only_configuration_selected_before_target_outcomes",
            "configuration": contract["vahadane_configuration"],
            "selection_manifest_path": contract["vahadane_selection_manifest"]["path"],
            "selection_manifest_sha256": contract["vahadane_selection_manifest"]["sha256"],
        },
        "outputs": {
            "method_summary": {"path": str(summary_path.resolve()), "sha256": sha256(summary_path)},
            "section_metrics": {"path": str(section_path.resolve()), "sha256": sha256(section_path)},
            "internal_external_contrast": {"path": str(contrast_path.resolve()), "sha256": sha256(contrast_path)},
            "decision": {"path": str(decision_path.resolve()), "sha256": sha256(decision_path)},
        },
        "source_shards_sha256": hashlib.sha256(
            "|".join(sha256(path) for path in image_paths + uni_paths + internal_paths).encode()
        ).hexdigest(),
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-render")
    prepare.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    fit = subparsers.add_parser("fit-parameters")
    fit.add_argument("--vahadane-selection-manifest", type=Path, required=True)
    fit.add_argument("--stain-v4-manifest", type=Path, required=True)
    fit.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    external_image = subparsers.add_parser("external-image-task")
    external_image.add_argument("--task-index", type=int, required=True)
    external_image.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    external_image.add_argument("--workers", type=int, default=1)
    external_uni = subparsers.add_parser("external-uni-task")
    external_uni.add_argument("--task-index", type=int, required=True)
    external_uni.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    external_uni.add_argument("--batch-size", type=int, default=32)
    external_uni.add_argument("--device", default="cuda")
    internal_image = subparsers.add_parser("internal-image-task")
    internal_image.add_argument("--task-index", type=int, required=True)
    internal_image.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    internal_image.add_argument("--workers", type=int, default=1)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    aggregate.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    args = parser.parse_args()
    if args.command == "prepare-render":
        result = prepare_render_contract(args.output_root)
    elif args.command == "fit-parameters":
        result = fit_parameters(
            args.vahadane_selection_manifest, args.stain_v4_manifest, args.output_root
        )
    elif args.command == "external-image-task":
        result = external_image_task(args.task_index, args.output_root, workers=args.workers)
    elif args.command == "external-uni-task":
        result = external_uni_task(
            args.task_index, args.output_root, batch_size=args.batch_size, device=args.device
        )
    elif args.command == "internal-image-task":
        result = internal_image_task(args.task_index, args.output_root, workers=args.workers)
    else:
        result = aggregate_results(args.output_root, bootstraps=args.bootstraps)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
