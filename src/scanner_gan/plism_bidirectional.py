#!/usr/bin/env python3
"""External PLISM validation for full bidirectional scanner translators.

The module applies the ten PanNormal-selected generators without PLISM
fine-tuning or outcome-dependent checkpoint selection:

* five GT450 -> AT2 generators, and
* five AT2 -> GT450 generators.

It reuses the immutable, registered GT450/AT2 PLISM renders and location set
from :mod:`scanner_gan.plism`.  Each direction is evaluated against its paired
same-location target with image metrics and freshly computed frozen UNI-v1
embeddings.  Passing the target through the generator is a separate identity
damage control and is never included in closure.

All PLISM observations remain externally shifted relative to PanNormal.  With
only the clean paired GT450/AT2 inputs, scanner status is source-model shared;
the analysis can estimate shared/shared and tissue-OOD strata, but cannot
populate scanner-OOD or double-OOD cells.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate_images import IMAGE_EVALUATION_VERSION, IMAGE_METRICS, image_metrics
from .plism import (
    TARGET_MPP,
    TARGET_PX,
    _generate_uint8,
    _read_json,
    _write_frame,
    _write_json,
    load_frozen_uni1,
    sha256,
)
from .predict import load_validated_generator
from .train_cyclegan import load_validated_cyclegan_generator
from .uni import (
    add_plism_model_status,
    cosine_distance_rows,
    embed_frozen_uni,
    embedding_collapse_diagnostics,
    paired_uni_metrics,
)


ROOT = Path(__file__).resolve().parents[2]
FULL_EXPERIMENT_ROOT = (
    ROOT
    / "outputs/scanner_batch_effect_analysis_2026-09-17"
    / "06_learned_baselines/09_bidirectional_full_training"
)
OUTPUT_ROOT = FULL_EXPERIMENT_ROOT / "10_plism_external_bidirectional"
RUNS_ROOT = FULL_EXPERIMENT_ROOT / "01_training/runs"
RENDER_CONTRACT_ROOT = (
    ROOT
    / "outputs/scanner_batch_effect_analysis_2026-09-17"
    / "08_plism_external/pix2pix_gt450_to_at2_image_safety"
)

ANALYSIS_VERSION = "pix2pix_bidirectional_full_plism_external_v1"
ANALYSIS_VERSIONS = {
    "pix2pix": ANALYSIS_VERSION,
    "cyclegan": "cyclegan_bidirectional_full_plism_external_v1",
}
SCANNER_TARGET_ANALYSIS_VERSIONS = {
    "pix2pix": "pix2pix_panel_a_scanner_targets_plism_external_v1",
    "cyclegan": "cyclegan_panel_a_scanner_targets_plism_external_v1",
}
DIRECTIONS = (("gt450", "at2"), ("at2", "gt450"))
METRICS = (
    "raw_to_target_distance",
    "method_to_target_distance",
    "gain_to_target",
    "fractional_closure",
    "method_to_raw_distance",
)
IDENTITY_METRICS = (
    "identity_uni_distance",
    *IMAGE_METRICS,
)
FORMAL_IMAGE_GATES = {
    "saturation_fraction": {"direction": "maximum", "threshold": 0.10},
    "invented_edge_fraction": {"direction": "maximum", "threshold": 0.001},
}
DEFAULT_BOOTSTRAP_REPLICATES = 20_000
DEFAULT_SEED = 20260917
DISPLAY = {
    "gt450_to_at2": "GT450 → AT2",
    "at2_to_gt450": "AT2 → GT450",
}


def _normalize_directions(
    directions: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for source, target in directions:
        pair = (str(source).strip().lower(), str(target).strip().lower())
        if not all(pair) or pair[0] == pair[1]:
            raise ValueError(f"invalid scanner direction {source!r}->{target!r}")
        if pair in result:
            raise ValueError(f"duplicate scanner direction {pair[0]}->{pair[1]}")
        result.append(pair)
    if len(result) != 2:
        raise ValueError("PLISM comparison requires exactly two scanner directions")
    return tuple(result)


def _contract_directions(contract: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    values = contract.get("directions")
    if not isinstance(values, list):
        return DIRECTIONS
    return _normalize_directions(
        tuple(
            (
                str(item["source_scanner"]),
                str(item["target_scanner"]),
            )
            for item in values
        )
    )


def _display_direction(source: str, target: str) -> str:
    return f"{source.upper()} → {target.upper()}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def direction_name(source: str, target: str) -> str:
    return f"{source.strip().lower()}_to_{target.strip().lower()}"


def _direction_arrow(source: str, target: str) -> str:
    return f"{source.strip().upper()}->{target.strip().upper()}"


def _normalize_uni_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy AT2-labelled helper columns to target-generic names."""

    mapping = {
        "raw_to_at2_distance": "raw_to_target_distance",
        "method_to_at2_distance": "method_to_target_distance",
        "gain_to_at2": "gain_to_target",
    }
    missing = set(mapping) - set(frame.columns)
    if missing:
        raise ValueError(f"paired UNI metrics lack columns: {sorted(missing)}")
    result = frame.rename(columns=mapping)
    if len(set(result.columns)) != len(result.columns):
        raise ValueError("target-generic UNI renaming created duplicate columns")
    return result


def _validate_full_run_manifest(
    manifest: Mapping[str, Any],
    *,
    source: str,
    target: str,
    fold: int,
    method: str = "pix2pix",
) -> None:
    expected = f"{source}->{target}"
    if str(manifest.get("status", "")).casefold() != "complete":
        raise ValueError(f"{expected}/fold {fold}: training run is not complete")
    method = method.strip().lower()
    if method == "pix2pix":
        if manifest.get("training_version") not in {
            "pix2pix_scanner_reference_v2",
            "manuscript_panel_a_full_training_v1",
        }:
            raise ValueError(f"{expected}/fold {fold}: unsupported Pix2Pix run")
        if str(manifest.get("direction", "")).casefold() != expected:
            raise ValueError(f"{expected}/fold {fold}: run direction differs")
        if manifest.get("selection_boundary") != (
            "image-only inner-validation target L1; UNI forbidden until lock"
        ):
            raise ValueError(
                f"{expected}/fold {fold}: checkpoint selection was not UNI-blind"
            )
    elif method == "cyclegan":
        if manifest.get("training_version") not in {
            "cyclegan_unpaired_bidirectional_v1",
            "manuscript_panel_a_full_training_v1",
        }:
            raise ValueError(f"{expected}/fold {fold}: not a full CycleGAN run")
        observed_domains = {
            item.strip().lower()
            for item in str(manifest.get("direction", "")).split("<->")
        }
        if observed_domains != {source, target}:
            raise ValueError(f"{expected}/fold {fold}: CycleGAN domains differ")
        if "pair-blind" not in str(
            manifest.get("selection_boundary", "")
        ).casefold():
            raise ValueError(
                f"{expected}/fold {fold}: CycleGAN selection was not pair-blind"
            )
    else:
        raise ValueError(f"unsupported method {method!r}")
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{expected}/fold {fold}: run config is absent")
    required = {
        "test_fold": fold,
        "validation_fold": (fold + 1) % 5,
        "max_passes": 200,
        "constant_passes": 100,
    }
    if method == "pix2pix":
        required.update(
            {
                "source_scanner": source,
                "target_scanner": target,
                "selection_policy": "target_l1",
            }
        )
    for key, expected_value in required.items():
        observed = config.get(key)
        if str(observed).casefold() != str(expected_value).casefold():
            raise ValueError(
                f"{expected}/fold {fold}: config {key}={observed!r}, "
                f"expected {expected_value!r}"
            )
    if method == "cyclegan":
        observed_domains = {
            str(config.get("domain_a", "")).strip().lower(),
            str(config.get("domain_b", "")).strip().lower(),
        }
        if observed_domains != {source, target}:
            raise ValueError(
                f"{expected}/fold {fold}: config domains {sorted(observed_domains)} differ"
            )
    if int(manifest.get("completed_passes", -1)) != 200:
        raise ValueError(f"{expected}/fold {fold}: full 200-pass schedule is incomplete")
    score_key = (
        "best_validation_l1"
        if method == "pix2pix"
        else "best_validation_marginal_score"
    )
    if not math.isfinite(float(manifest.get(score_key, float("nan")))):
        raise ValueError(f"{expected}/fold {fold}: invalid validation score")


def _discover_candidates(
    runs_root: Path,
    *,
    method: str = "pix2pix",
    directions: Sequence[tuple[str, str]] = DIRECTIONS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sample_hashes: set[str] = set()
    directions = _normalize_directions(directions)
    for source, target in directions:
        direction = direction_name(source, target)
        for fold in range(5):
            if method == "pix2pix":
                run_directions = (direction,)
            else:
                run_directions = (
                    f"{source}_{target}",
                    f"{target}_{source}",
                )
            manifests = sorted(
                {
                    path.resolve()
                    for run_direction in run_directions
                    for path in runs_root.glob(
                        f"{run_direction}/fold_{fold}/seed_*/run_manifest.json"
                    )
                }
            )
            if len(manifests) != 1:
                raise ValueError(
                    f"{direction}/fold {fold}: expected one full run, found {len(manifests)}"
                )
            manifest_path = manifests[0].resolve()
            manifest = _read_json(manifest_path)
            _validate_full_run_manifest(
                manifest,
                source=source,
                target=target,
                fold=fold,
                method=method,
            )
            checkpoint = Path(str(manifest["best_image_only_checkpoint"])).resolve()
            expected_checkpoint = manifest_path.parent / "checkpoints/best_image_only.pt"
            if checkpoint != expected_checkpoint.resolve():
                raise ValueError(
                    f"{direction}/fold {fold}: best checkpoint path differs from run layout"
                )
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            sample_hash = str(manifest.get("sample_index_sha256", ""))
            if len(sample_hash) != 64:
                raise ValueError(f"{direction}/fold {fold}: invalid sample-index hash")
            sample_hashes.add(sample_hash)
            rows.append(
                {
                    "task_index": len(rows),
                    "direction": direction,
                    "direction_arrow": _direction_arrow(source, target),
                    "source_scanner": source,
                    "target_scanner": target,
                    "fold": fold,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "run_manifest_path": str(manifest_path),
                    "run_manifest_sha256": sha256(manifest_path),
                    "completed_passes": int(manifest["completed_passes"]),
                    "global_step": int(manifest["global_step"]),
                    "best_validation_score": float(
                        manifest[
                            "best_validation_l1"
                            if method == "pix2pix"
                            else "best_validation_marginal_score"
                        ]
                    ),
                    "sample_index_sha256": sample_hash,
                }
            )
    if len(sample_hashes) != 1:
        raise ValueError(f"full runs do not share one sample index: {sorted(sample_hashes)}")
    return pd.DataFrame(rows)


def _validate_render_pair(
    render_root: Path,
    selected: pd.DataFrame,
    stain: str,
    scanners: Sequence[str] = ("AT2", "GT450"),
) -> list[dict[str, Any]]:
    wanted = (
        selected.loc[selected["stain"].astype(str).eq(stain), "location"]
        .astype(np.int64)
        .sort_values(kind="stable")
        .to_numpy()
    )
    rows = []
    observed_locations: dict[str, np.ndarray] = {}
    scanners = tuple(str(item).upper() for item in scanners)
    for scanner in scanners:
        path = (render_root / scanner / f"{stain}.h5").resolve()
        summary_path = path.with_suffix(".summary.json")
        if not path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"missing immutable PLISM render: {path}")
        summary = _read_json(summary_path)
        observed_hash = sha256(path)
        if summary.get("output_sha256") != observed_hash:
            raise ValueError(f"render hash differs from its summary: {path}")
        with h5py.File(path, "r") as store:
            if "images" not in store or "location" not in store:
                raise ValueError(f"invalid render schema: {path}")
            shape = tuple(store["images"].shape)
            dtype = store["images"].dtype
            locations = np.asarray(store["location"], dtype=np.int64)
        if shape != (len(wanted), TARGET_PX, TARGET_PX, 3) or dtype != np.uint8:
            raise ValueError(f"unexpected image grid in {path}: {shape}, {dtype}")
        if not np.array_equal(locations, wanted):
            raise ValueError(f"selected/rendered locations differ for {scanner}/{stain}")
        observed_locations[scanner] = locations
        rows.append(
            {
                "stain": stain,
                "scanner": scanner,
                "path": str(path),
                "sha256": observed_hash,
                "summary_path": str(summary_path.resolve()),
                "summary_sha256": sha256(summary_path),
                "locations": len(locations),
            }
        )
    reference = observed_locations[scanners[0]]
    if any(
        not np.array_equal(reference, observed_locations[scanner])
        for scanner in scanners[1:]
    ):
        raise ValueError(f"paired render locations differ for {stain}")
    return rows


def _validate_registered_render_table(
    source_table: pd.DataFrame,
    selected: pd.DataFrame,
    scanners: Sequence[str],
) -> pd.DataFrame:
    """Validate and subset a registered multi-root PLISM render table."""

    required = {
        "scanner",
        "stain",
        "path",
        "sha256",
        "summary_path",
        "summary_sha256",
        "locations",
    }
    missing = required - set(source_table.columns)
    if missing:
        raise ValueError(f"registered render table lacks columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    scanners = tuple(str(item).upper() for item in scanners)
    for stain in sorted(selected["stain"].astype(str).unique()):
        wanted = (
            selected.loc[selected["stain"].astype(str).eq(stain), "location"]
            .astype(np.int64)
            .sort_values(kind="stable")
            .to_numpy()
        )
        observed_locations: dict[str, np.ndarray] = {}
        for scanner in scanners:
            match = source_table.loc[
                source_table["scanner"].astype(str).str.upper().eq(scanner)
                & source_table["stain"].astype(str).eq(stain)
            ]
            if len(match) != 1:
                raise ValueError(f"expected one registered render for {scanner}/{stain}")
            row = match.iloc[0]
            path = Path(str(row["path"])).resolve()
            summary_path = Path(str(row["summary_path"])).resolve()
            if sha256(path) != str(row["sha256"]):
                raise ValueError(f"registered render hash changed: {path}")
            if sha256(summary_path) != str(row["summary_sha256"]):
                raise ValueError(f"registered render summary hash changed: {summary_path}")
            summary = _read_json(summary_path)
            if summary.get("output_sha256") != str(row["sha256"]):
                raise ValueError(f"render and summary hashes differ: {path}")
            with h5py.File(path, "r") as store:
                if "images" not in store or "location" not in store:
                    raise ValueError(f"invalid render schema: {path}")
                shape = tuple(store["images"].shape)
                dtype = store["images"].dtype
                locations = np.asarray(store["location"], dtype=np.int64)
            if shape != (len(wanted), TARGET_PX, TARGET_PX, 3) or dtype != np.uint8:
                raise ValueError(f"unexpected image grid in {path}: {shape}, {dtype}")
            if not np.array_equal(locations, wanted):
                raise ValueError(f"selected/rendered locations differ for {scanner}/{stain}")
            observed_locations[scanner] = locations
            rows.append({key: row[key] for key in required})
        reference = observed_locations[scanners[0]]
        if any(
            not np.array_equal(reference, observed_locations[scanner])
            for scanner in scanners[1:]
        ):
            raise ValueError(f"paired render locations differ for {stain}")
    return pd.DataFrame(rows).sort_values(["stain", "scanner"], kind="stable")


def prepare_contract(
    *,
    output_root: str | Path = OUTPUT_ROOT,
    runs_root: str | Path = RUNS_ROOT,
    render_contract_root: str | Path = RENDER_CONTRACT_ROOT,
    method: str = "pix2pix",
    directions: Sequence[tuple[str, str]] = DIRECTIONS,
) -> dict[str, Any]:
    """Freeze candidates and read-only PLISM inputs before any external endpoint."""

    output = Path(output_root).resolve()
    runs = Path(runs_root).resolve()
    source_root = Path(render_contract_root).resolve()
    method = method.strip().lower()
    directions = _normalize_directions(directions)
    if method not in ANALYSIS_VERSIONS:
        raise ValueError(f"unsupported method {method!r}")
    contract_path = output / "00_contract/analysis_contract.json"
    if contract_path.exists():
        existing, _ = _validated_contract(output)
        existing_method = str(existing.get("method", "pix2pix")).strip().lower()
        if existing_method != method:
            raise ValueError("existing PLISM contract uses another method")
        return {"status": "already_complete", **existing}

    # Freeze all ten PanNormal-only decisions before opening even structural
    # PLISM render metadata.  The discovery function reads training manifests
    # and checkpoint bytes only.
    candidates = _discover_candidates(runs, method=method, directions=directions)

    source_contract_path = source_root / "00_contract/analysis_contract.json"
    source_contract = _read_json(source_contract_path)
    source_version = str(source_contract.get("analysis_version", ""))
    supported_source_versions = {
        "pix2pix_gt450_to_at2_plism_external_v2",
        "manuscript_plism_external_conventional_v1",
    }
    if source_version not in supported_source_versions:
        raise ValueError("source PLISM contract is not a supported registered-render contract")
    selected_path = Path(source_contract["outputs"]["selected_locations"]["path"])
    selected_hash = sha256(selected_path)
    if selected_hash != source_contract["outputs"]["selected_locations"]["sha256"]:
        raise ValueError("source PLISM selected-location hash changed")
    selected = pd.read_csv(selected_path)
    required = {
        "stain",
        "location",
        "core",
        "tissue_type",
        "replicate",
        "organ_aligned_status",
        "strict_status",
    }
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"selected PLISM metadata lacks columns: {sorted(missing)}")
    if selected[["stain", "location"]].duplicated().any():
        raise ValueError("selected PLISM locations are not unique within section")
    scanners = tuple(
        dict.fromkeys(
            scanner.upper()
            for direction in directions
            for scanner in direction
        )
    )
    if source_version == "pix2pix_gt450_to_at2_plism_external_v2":
        render_root = Path(source_contract["render_storage"]["root"]).resolve()
        render_rows: list[dict[str, Any]] = []
        for stain in sorted(selected["stain"].astype(str).unique()):
            render_rows.extend(
                _validate_render_pair(render_root, selected, stain, scanners=scanners)
            )
        render_table = pd.DataFrame(render_rows)
        encoder_contract = source_contract
    else:
        render_source = source_contract["outputs"]["read_only_renders"]
        render_source_path = Path(str(render_source["path"])).resolve()
        if sha256(render_source_path) != str(render_source["sha256"]):
            raise ValueError("registered PLISM render table hash changed")
        source_render_table = pd.read_csv(render_source_path)
        render_tasks_source = source_contract["outputs"].get("render_tasks")
        if not isinstance(render_tasks_source, Mapping):
            raise ValueError("registered PLISM contract lacks new-scanner render tasks")
        render_tasks_path = Path(str(render_tasks_source["path"])).resolve()
        if sha256(render_tasks_path) != str(render_tasks_source["sha256"]):
            raise ValueError("registered PLISM render-task table hash changed")
        render_tasks = pd.read_csv(render_tasks_path)
        new_render_root = Path(source_contract["render"]["new_root"]).resolve()
        new_rows: list[dict[str, Any]] = []
        for task in render_tasks.itertuples(index=False):
            path = (new_render_root / str(task.scanner) / f"{task.stain}.h5").resolve()
            summary_path = path.with_suffix(".summary.json")
            if not path.is_file() or not summary_path.is_file():
                raise FileNotFoundError(f"missing registered render: {path}")
            new_rows.append(
                {
                    "scanner": str(task.scanner),
                    "stain": str(task.stain),
                    "path": str(path),
                    "sha256": sha256(path),
                    "summary_path": str(summary_path),
                    "summary_sha256": sha256(summary_path),
                    "locations": int(task.locations),
                }
            )
        source_render_table = pd.concat(
            [source_render_table, pd.DataFrame(new_rows)], ignore_index=True
        )
        render_table = _validate_registered_render_table(
            source_render_table, selected, scanners
        )
        encoder_contract_path = Path(source_contract["source_contract"]["path"])
        if sha256(encoder_contract_path) != str(
            source_contract["source_contract"]["sha256"]
        ):
            raise ValueError("encoder source PLISM contract hash changed")
        encoder_contract = _read_json(encoder_contract_path)

    selected_copy = output / "00_contract/selected_locations.csv"
    candidate_path = output / "00_contract/checkpoint_candidates.csv"
    render_path = output / "00_contract/read_only_renders.csv"
    _write_frame(selected, selected_copy)
    _write_frame(candidates, candidate_path)
    _write_frame(render_table, render_path)
    encoder = encoder_contract["primary_encoder"]
    analysis_version = (
        ANALYSIS_VERSIONS[method]
        if directions == DIRECTIONS
        else SCANNER_TARGET_ANALYSIS_VERSIONS[method]
    )
    contract = {
        "analysis_version": analysis_version,
        "method": method,
        "status": "frozen_before_external_evaluation",
        "created_utc": utc_now(),
        "directions": [
            {
                "source_scanner": source.upper(),
                "target_scanner": target.upper(),
                "direction": _direction_arrow(source, target),
                "models": 5,
            }
            for source, target in directions
        ],
        "primary_question": (
            "Do image-only-selected full-training mappings retain their paired image and "
            "frozen-UNI effects on external PLISM, and does target choice change that transfer?"
        ),
        "checkpoint_boundary": (
            "Ten direction-fold Pix2Pix checkpoints were selected only by PanNormal "
            "inner-validation target L1. PLISM images, image metrics, and UNI embeddings "
            "cannot select or tune them."
            if method == "pix2pix"
            else "Five bidirectional CycleGAN checkpoints, instantiated as ten "
            "direction-fold tasks, were selected by pair-blind PanNormal inner-validation "
            "marginal image endpoints. Same-location targets, PLISM, and UNI could not "
            "select or tune them."
        ),
        "external_boundary": (
            "PLISM is external in site, physical device, serial section, stain, and "
            "acquisition. Scanner-model shared is therefore ID-like only, never fully ID."
        ),
        "quadrant_scope": (
            "Clean paired source/target inputs populate shared/shared and tissue-OOD only. "
            "Every observation remains external in site, device, section, stain, and acquisition."
        ),
        "inference_unit": "physical serial section (stain; 13 sections)",
        "aggregation": (
            "location -> TMA core -> physical section; five fold models are averaged "
            "within section before section-level bootstrap"
        ),
        "identity_control": (
            "For each source->target generator, G(target) is compared with unchanged target. "
            "Identity damage is excluded from closure and target-direction gain."
        ),
        "no_finetuning": True,
        "no_outcome_filtering": True,
        "selected_locations": int(len(selected)),
        "sections": int(selected["stain"].nunique()),
        "section_core_cells": int(selected.groupby(["stain", "core"]).ngroups),
        "render": {
            "target_px": TARGET_PX,
            "target_mpp": TARGET_MPP,
            "storage_root": "registered multi-root render table",
            "read_only": True,
            "scanners": list(scanners),
        },
        "primary_encoder": {
            "encoder_id": "uni_v1",
            "repository": encoder["repository"],
            "feature_dim": int(encoder["feature_dim"]),
            "checkpoint_path": str(Path(encoder["checkpoint_path"]).resolve()),
            "checkpoint_sha256": str(encoder["checkpoint_sha256"]),
            "frozen": True,
            "fresh_roles": ["source", "generated", "paired_target", "identity_generated"],
        },
        "image_metrics": {
            "evaluation_version": IMAGE_EVALUATION_VERSION,
            "metrics": list(IMAGE_METRICS),
            "formal_external_gates": FORMAL_IMAGE_GATES,
            "source_gradient_ncc_role": "descriptive_only",
        },
        "source_contract": {
            "path": str(source_contract_path),
            "sha256": sha256(source_contract_path),
        },
        "outputs": {
            "selected_locations": {
                "path": str(selected_copy),
                "sha256": sha256(selected_copy),
            },
            "checkpoint_candidates": {
                "path": str(candidate_path),
                "sha256": sha256(candidate_path),
            },
            "read_only_renders": {
                "path": str(render_path),
                "sha256": sha256(render_path),
            },
        },
    }
    _write_json(contract_path, contract)
    return contract


def _validated_contract(output: Path) -> tuple[dict[str, Any], str]:
    path = output / "00_contract/analysis_contract.json"
    contract = _read_json(path)
    allowed_versions = {
        *ANALYSIS_VERSIONS.values(),
        *SCANNER_TARGET_ANALYSIS_VERSIONS.values(),
    }
    if contract.get("analysis_version") not in allowed_versions:
        raise ValueError("unsupported bidirectional PLISM contract")
    method = str(contract.get("method", "pix2pix")).strip().lower()
    if contract.get("analysis_version") not in {
        ANALYSIS_VERSIONS.get(method),
        SCANNER_TARGET_ANALYSIS_VERSIONS.get(method),
    }:
        raise ValueError("PLISM contract method/version mismatch")
    for item in contract["outputs"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"frozen contract input changed: {item['path']}")
    return contract, sha256(path)


def _load_render(
    render_lookup: pd.DataFrame,
    scanner: str,
    stain: str,
) -> tuple[np.ndarray, np.ndarray, Path]:
    match = render_lookup.loc[
        render_lookup["scanner"].astype(str).str.casefold().eq(scanner.casefold())
        & render_lookup["stain"].astype(str).eq(stain)
    ]
    if len(match) != 1:
        raise ValueError(f"expected one frozen render for {scanner}/{stain}")
    row = match.iloc[0]
    path = Path(str(row["path"]))
    if sha256(path) != str(row["sha256"]):
        raise ValueError(f"frozen render changed: {path}")
    with h5py.File(path, "r") as store:
        images = np.asarray(store["images"], dtype=np.uint8)
        locations = np.asarray(store["location"], dtype=np.int64)
    return images, locations, path


def _candidate_for_task(
    candidates: pd.DataFrame,
    task_index: int,
) -> pd.Series:
    match = candidates.loc[candidates["task_index"].astype(int).eq(int(task_index))]
    if len(match) != 1:
        raise KeyError(f"unknown evaluation task {task_index}")
    return match.iloc[0]


def evaluate_task(
    task_index: int,
    *,
    output_root: str | Path = OUTPUT_ROOT,
    uni_checkpoint_path: str | Path | None = None,
    generator_batch_size: int = 16,
    uni_batch_size: int = 32,
    device: str = "cuda",
    amp: str = "bfloat16",
) -> dict[str, Any]:
    """Evaluate one frozen direction/fold candidate on all paired PLISM sections."""

    import torch

    if generator_batch_size < 1 or uni_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    output = Path(output_root).resolve()
    contract, contract_hash = _validated_contract(output)
    method = str(contract.get("method", "pix2pix")).strip().lower()
    analysis_version = str(contract["analysis_version"])
    candidates = pd.read_csv(contract["outputs"]["checkpoint_candidates"]["path"])
    selected = pd.read_csv(contract["outputs"]["selected_locations"]["path"])
    renders = pd.read_csv(contract["outputs"]["read_only_renders"]["path"])
    candidate = _candidate_for_task(candidates, task_index)
    source = str(candidate["source_scanner"]).casefold()
    target = str(candidate["target_scanner"]).casefold()
    fold = int(candidate["fold"])
    direction = str(candidate["direction"])
    checkpoint = Path(str(candidate["checkpoint_path"])).resolve()
    run_manifest_path = Path(str(candidate["run_manifest_path"])).resolve()
    if sha256(checkpoint) != str(candidate["checkpoint_sha256"]):
        raise ValueError("frozen best checkpoint changed")
    if sha256(run_manifest_path) != str(candidate["run_manifest_sha256"]):
        raise ValueError("frozen training run manifest changed")
    run_manifest = _read_json(run_manifest_path)
    _validate_full_run_manifest(
        run_manifest,
        source=source,
        target=target,
        fold=fold,
        method=method,
    )

    fold_root = output / f"01_model_evaluations/{direction}/fold_{fold}"
    manifest_path = fold_root / "evaluation_manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if (
            manifest.get("status") == "complete"
            and manifest.get("contract_sha256") == contract_hash
            and manifest.get("checkpoint_sha256") == str(candidate["checkpoint_sha256"])
        ):
            return {"status": "already_complete", **manifest}
        raise FileExistsError(f"refusing to replace existing evaluation: {manifest_path}")
    if fold_root.exists() and any(fold_root.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite partial evaluation outputs: {fold_root}"
        )

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    loader = (
        load_validated_generator
        if method == "pix2pix"
        else load_validated_cyclegan_generator
    )
    generator, checkpoint_metadata, _ = loader(
        checkpoint,
        source_scanner=source,
        target_scanner=target,
        test_fold=fold,
        device=resolved_device,
    )
    encoder = contract["primary_encoder"]
    uni_model, uni_size, uni_mean, uni_std, uni_checkpoint = load_frozen_uni1(
        resolved_device,
        checkpoint_path=uni_checkpoint_path or encoder["checkpoint_path"],
        expected_sha256=str(encoder["checkpoint_sha256"]),
    )

    translation_parts: list[pd.DataFrame] = []
    identity_parts: list[pd.DataFrame] = []
    generated_feature_parts: list[np.ndarray] = []
    target_feature_parts: list[np.ndarray] = []
    shard_rows: list[dict[str, Any]] = []
    for stain in sorted(selected["stain"].astype(str).unique()):
        source_images, source_locations, source_path = _load_render(
            renders, source.upper(), stain
        )
        target_images, target_locations, target_path = _load_render(
            renders, target.upper(), stain
        )
        if not np.array_equal(source_locations, target_locations):
            raise ValueError(f"paired locations differ for {direction}/{stain}")
        metadata = (
            selected.loc[selected["stain"].astype(str).eq(stain)]
            .set_index("location")
            .reindex(source_locations)
            .reset_index()
        )
        if metadata["core"].isna().any():
            raise ValueError(f"rendered locations lack metadata for {stain}")

        generated = _generate_uint8(
            generator,
            source_images,
            resolved_device,
            batch_size=generator_batch_size,
            amp=amp,
        )
        identity_generated = _generate_uint8(
            generator,
            target_images,
            resolved_device,
            batch_size=generator_batch_size,
            amp=amp,
        )
        embedded = embed_frozen_uni(
            uni_model,
            np.concatenate(
                (source_images, generated, target_images, identity_generated), axis=0
            ),
            uni_size,
            uni_mean,
            uni_std,
            resolved_device,
            batch_size=uni_batch_size,
            value_range="uint8",
        )
        n = len(source_images)
        source_features, generated_features, target_features, identity_features = np.split(
            embedded, [n, 2 * n, 3 * n]
        )
        generated_feature_parts.append(generated_features)
        target_feature_parts.append(target_features)

        metric_metadata = metadata.copy()
        metric_metadata["source_scanner"] = source.upper()
        metric_metadata["slide_id"] = stain
        metric_metadata["fold"] = fold
        metric_metadata["location_index"] = metric_metadata["location"].astype(int)
        metric_metadata["shared_tissue"] = metric_metadata["organ_aligned_status"].eq(
            "shared"
        )
        metric_metadata = add_plism_model_status(
            metric_metadata,
            model_source_scanner=source.upper(),
        )
        metrics = paired_uni_metrics(
            source_features,
            generated_features,
            target_features,
            metric_metadata,
            method=f"{method}_full",
            target_scanner=target.upper(),
        )
        metrics = _normalize_uni_columns(metrics)
        metrics["direction"] = direction
        metrics["direction_arrow"] = _direction_arrow(source, target)
        metrics["checkpoint_fold"] = fold
        metrics["checkpoint_sha256"] = str(candidate["checkpoint_sha256"])
        measured = image_metrics(source_images, generated, target_images)
        for column in IMAGE_METRICS:
            metrics[column] = measured[column].to_numpy()
        translation_parts.append(metrics)

        identity = metadata.copy()
        identity["direction"] = direction
        identity["direction_arrow"] = _direction_arrow(source, target)
        identity["generator_source_scanner"] = source.upper()
        identity["identity_input_scanner"] = target.upper()
        identity["checkpoint_fold"] = fold
        identity["checkpoint_sha256"] = str(candidate["checkpoint_sha256"])
        identity["included_in_closure"] = False
        identity["identity_uni_distance"] = cosine_distance_rows(
            identity_features, target_features
        )
        identity_images = image_metrics(
            target_images,
            identity_generated,
            target_images,
        )
        for column in IMAGE_METRICS:
            identity[column] = identity_images[column].to_numpy()
        identity_parts.append(identity)

        shard_path = fold_root / f"feature_shards/{stain}.h5"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = shard_path.with_suffix(shard_path.suffix + f".{os.getpid()}.tmp")
        if temporary.exists():
            temporary.unlink()
        with h5py.File(temporary, "w") as store:
            store.attrs["analysis_version"] = analysis_version
            store.attrs["contract_sha256"] = contract_hash
            store.attrs["checkpoint_sha256"] = str(candidate["checkpoint_sha256"])
            store.attrs["direction"] = direction
            store.attrs["stain"] = stain
            store.attrs["encoder_checkpoint_sha256"] = str(
                encoder["checkpoint_sha256"]
            )
            store.create_dataset("location", data=source_locations)
            store.create_dataset("source", data=source_features, compression="lzf")
            store.create_dataset("generated", data=generated_features, compression="lzf")
            store.create_dataset("paired_target", data=target_features, compression="lzf")
            controls = store.create_group("identity_control")
            controls.attrs["included_in_closure"] = False
            controls.create_dataset(
                "generated_from_target", data=identity_features, compression="lzf"
            )
        temporary.replace(shard_path)
        shard_rows.append(
            {
                "stain": stain,
                "locations": n,
                "source_render_path": str(source_path),
                "target_render_path": str(target_path),
                "feature_path": str(shard_path),
                "feature_sha256": sha256(shard_path),
            }
        )

    translation = pd.concat(translation_parts, ignore_index=True)
    identity = pd.concat(identity_parts, ignore_index=True)
    expected = int(contract["selected_locations"])
    if len(translation) != expected or len(identity) != expected:
        raise ValueError(
            f"task has {len(translation)} translation and {len(identity)} identity rows; "
            f"expected {expected}"
        )
    if identity["included_in_closure"].astype(bool).any():
        raise ValueError("identity control leaked into closure")
    translation_path = fold_root / "translation_location_metrics.csv.gz"
    identity_path = fold_root / "identity_location_metrics.csv.gz"
    _write_frame(translation, translation_path)
    _write_frame(identity, identity_path)
    collapse = embedding_collapse_diagnostics(
        np.concatenate(generated_feature_parts),
        np.concatenate(target_feature_parts),
    )
    manifest = {
        "analysis_version": analysis_version,
        "method": method,
        "status": "complete",
        "created_utc": utc_now(),
        "contract_sha256": contract_hash,
        "task_index": int(task_index),
        "direction": direction,
        "source_scanner": source.upper(),
        "target_scanner": target.upper(),
        "checkpoint_fold": fold,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": str(candidate["checkpoint_sha256"]),
        "checkpoint_metadata": checkpoint_metadata,
        "run_manifest_path": str(run_manifest_path),
        "run_manifest_sha256": str(candidate["run_manifest_sha256"]),
        "selection_boundary": contract["checkpoint_boundary"],
        "no_finetuning": True,
        "no_outcome_filtering": True,
        "locations": len(translation),
        "sections": int(translation["stain"].nunique()),
        "primary_encoder_checkpoint_path": str(uni_checkpoint),
        "primary_encoder_checkpoint_sha256": sha256(uni_checkpoint),
        "translation_metrics": {
            "path": str(translation_path),
            "sha256": sha256(translation_path),
        },
        "identity_metrics": {
            "path": str(identity_path),
            "sha256": sha256(identity_path),
            "included_in_closure": False,
        },
        "all_location_collapse_diagnostic": collapse,
        "feature_shards": shard_rows,
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
    draws = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(
        axis=1
    )
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _translation_section_tables(
    location: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [*METRICS, *IMAGE_METRICS]
    identifiers = ["direction", "checkpoint_fold", "quadrant", "stain", "core"]
    core_stratified = location.groupby(
        identifiers, as_index=False, dropna=False
    )[metric_columns].mean()
    core_overall = location.groupby(
        ["direction", "checkpoint_fold", "stain", "core"],
        as_index=False,
        dropna=False,
    )[metric_columns].mean()
    core_overall["quadrant"] = "overall"
    core = pd.concat([core_overall, core_stratified], ignore_index=True)
    section = core.groupby(
        ["direction", "checkpoint_fold", "quadrant", "stain"],
        as_index=False,
        dropna=False,
    )[metric_columns].mean()
    return core, section


def _identity_section_tables(
    location: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    core = location.groupby(
        ["direction", "checkpoint_fold", "stain", "core"],
        as_index=False,
        dropna=False,
    )[list(IDENTITY_METRICS)].mean()
    section = core.groupby(
        ["direction", "checkpoint_fold", "stain"],
        as_index=False,
        dropna=False,
    )[list(IDENTITY_METRICS)].mean()
    return core, section


def _summarize_models_and_sections(
    section: pd.DataFrame,
    *,
    metrics: Sequence[str],
    group_columns: Sequence[str],
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, Any]] = []
    fold_groups = [*group_columns, "checkpoint_fold"]
    for group_index, (key, block) in enumerate(
        section.groupby(fold_groups, sort=True, dropna=False)
    ):
        values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(fold_groups, values))
        for metric_index, metric in enumerate(metrics):
            estimate, low, high = _bootstrap_mean(
                block[metric].to_numpy(float),
                replicates=bootstrap_replicates,
                seed=seed + group_index * 100 + metric_index,
            )
            model_rows.append(
                {
                    **identity,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "sections": int(block["stain"].nunique()),
                }
            )
    per_model = pd.DataFrame(model_rows)

    ensemble = section.groupby(
        [*group_columns, "stain"], as_index=False, dropna=False
    )[list(metrics)].mean()
    ensemble_rows: list[dict[str, Any]] = []
    for group_index, (key, block) in enumerate(
        ensemble.groupby(list(group_columns), sort=True, dropna=False)
    ):
        values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_columns, values))
        for metric_index, metric in enumerate(metrics):
            estimate, low, high = _bootstrap_mean(
                block[metric].to_numpy(float),
                replicates=bootstrap_replicates,
                seed=seed + 50_000 + group_index * 100 + metric_index,
            )
            model_values = per_model.loc[
                np.logical_and.reduce(
                    [per_model[column].eq(value) for column, value in identity.items()]
                )
                & per_model["metric"].eq(metric),
                "estimate",
            ].to_numpy(float)
            ensemble_rows.append(
                {
                    **identity,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "sections": int(block["stain"].nunique()),
                    "models": int(len(model_values)),
                    "model_estimate_min": float(np.min(model_values)),
                    "model_estimate_max": float(np.max(model_values)),
                    "model_estimate_sd": float(np.std(model_values, ddof=1)),
                    "models_positive": int(np.sum(model_values > 0)),
                    "models_negative": int(np.sum(model_values < 0)),
                }
            )
    return per_model, ensemble, pd.DataFrame(ensemble_rows)


def _paired_direction_comparison(
    ensemble_sections: pd.DataFrame,
    *,
    directions: Sequence[tuple[str, str]] = DIRECTIONS,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    directions = _normalize_directions(directions)
    values = ensemble_sections[
        ["direction", "quadrant", "stain", "gain_to_target"]
    ].pivot(index=["quadrant", "stain"], columns="direction", values="gain_to_target")
    expected = {direction_name(*item) for item in directions}
    if set(values.columns) != expected or values.isna().any().any():
        raise ValueError("direction comparison lacks paired section-level gains")
    values = values.reset_index()
    first = direction_name(*directions[0])
    second = direction_name(*directions[1])
    difference_column = f"gain_difference_{second}_minus_{first}"
    values[difference_column] = values[second] - values[first]
    rows = []
    for index, (quadrant, block) in enumerate(values.groupby("quadrant", sort=True)):
        estimate, low, high = _bootstrap_mean(
            block[difference_column].to_numpy(float),
            replicates=bootstrap_replicates,
            seed=seed + 90_000 + index,
        )
        rows.append(
            {
                "quadrant": quadrant,
                "contrast": f"gain({_direction_arrow(*directions[1])}) - gain({_direction_arrow(*directions[0])})",
                "positive_favors": _direction_arrow(*directions[1]),
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "sections": int(block["stain"].nunique()),
                "bootstrap_replicates": bootstrap_replicates,
            }
        )
    return values, pd.DataFrame(rows)


def _external_gate_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    overall = summary.loc[summary["quadrant"].eq("overall")]
    for direction, block in overall.groupby("direction", sort=True):
        lookup = block.set_index("metric")
        for metric, gate in FORMAL_IMAGE_GATES.items():
            if metric not in lookup.index:
                raise ValueError(f"missing external image gate metric {metric}")
            estimate = float(lookup.loc[metric, "estimate"])
            threshold = float(gate["threshold"])
            passed = estimate <= threshold
            rows.append(
                {
                    "direction": direction,
                    "metric": metric,
                    "direction_rule": gate["direction"],
                    "threshold": threshold,
                    "estimate": estimate,
                    "pass": passed,
                    "selection_use": "external_diagnostic_only",
                }
            )
    return pd.DataFrame(rows)


def _summary_metric(
    frame: pd.DataFrame,
    direction: str,
    metric: str,
    *,
    quadrant: str | None = None,
) -> pd.Series:
    selected = frame.loc[
        frame["direction"].eq(direction) & frame["metric"].eq(metric)
    ]
    if quadrant is not None:
        selected = selected.loc[selected["quadrant"].eq(quadrant)]
    if len(selected) != 1:
        raise ValueError(
            f"expected one external summary row for {direction}/{quadrant}/{metric}; "
            f"found {len(selected)}"
        )
    return selected.iloc[0]


def _values_and_errors(rows: Sequence[pd.Series]) -> tuple[np.ndarray, np.ndarray]:
    estimates = np.asarray([float(row["estimate"]) for row in rows])
    lower = np.asarray([float(row["ci_low"]) for row in rows])
    upper = np.asarray([float(row["ci_high"]) for row in rows])
    return estimates, np.vstack((estimates - lower, upper - estimates))


def _save_external_endpoint_figure(
    summary: pd.DataFrame,
    identity_summary: pd.DataFrame,
    path: Path,
    *,
    method_label: str = "Pix2Pix",
    directions: Sequence[tuple[str, str]] = DIRECTIONS,
) -> None:
    direction_pairs = _normalize_directions(directions)
    directions = [direction_name(*item) for item in direction_pairs]
    labels = [_display_direction(*item) for item in direction_pairs]
    colors = ["#315a78", "#b75d3e"]
    x = np.arange(len(directions))
    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    axes = axes.ravel()

    for axis, metric, title in (
        (axes[0], "target_l1_unit", "External target L1 (lower is better)"),
        (axes[1], "target_ssim", "External target SSIM (higher is better)"),
    ):
        rows = [
            _summary_metric(summary, item, metric, quadrant="overall")
            for item in directions
        ]
        estimates, errors = _values_and_errors(rows)
        axis.bar(x, estimates, color=colors)
        axis.errorbar(
            x,
            estimates,
            yerr=errors,
            color="black",
            capsize=4,
            linestyle="none",
        )
        axis.set(title=title, xticks=x, xticklabels=labels)

    raw_gradient = [
        _summary_metric(
            summary,
            item,
            "raw_source_target_gradient_ncc",
            quadrant="overall",
        )
        for item in directions
    ]
    method_gradient = [
        _summary_metric(
            summary, item, "target_gradient_ncc", quadrant="overall"
        )
        for item in directions
    ]
    raw_estimate, raw_error = _values_and_errors(raw_gradient)
    method_estimate, method_error = _values_and_errors(method_gradient)
    width = 0.34
    axes[2].bar(x - width / 2, raw_estimate, width, label="Raw source")
    axes[2].bar(x + width / 2, method_estimate, width, label=method_label)
    axes[2].errorbar(
        x - width / 2,
        raw_estimate,
        yerr=raw_error,
        color="black",
        capsize=3,
        linestyle="none",
    )
    axes[2].errorbar(
        x + width / 2,
        method_estimate,
        yerr=method_error,
        color="black",
        capsize=3,
        linestyle="none",
    )
    axes[2].set(
        title="External gradient agreement with paired target",
        xticks=x,
        xticklabels=labels,
    )
    axes[2].legend(frameon=False)

    invented = [
        _summary_metric(
            summary, item, "invented_edge_fraction", quadrant="overall"
        )
        for item in directions
    ]
    saturation = [
        _summary_metric(summary, item, "saturation_fraction", quadrant="overall")
        for item in directions
    ]
    invented_estimate, invented_error = _values_and_errors(invented)
    saturation_estimate, saturation_error = _values_and_errors(saturation)
    axes[3].bar(
        x - width / 2,
        invented_estimate * 100.0,
        width,
        label="Invented edge",
    )
    axes[3].bar(
        x + width / 2,
        saturation_estimate * 100.0,
        width,
        label="Saturation",
    )
    axes[3].errorbar(
        x - width / 2,
        invented_estimate * 100.0,
        yerr=invented_error * 100.0,
        color="black",
        capsize=3,
        linestyle="none",
    )
    axes[3].errorbar(
        x + width / 2,
        saturation_estimate * 100.0,
        yerr=saturation_error * 100.0,
        color="black",
        capsize=3,
        linestyle="none",
    )
    axes[3].set(
        title="External image-safety fractions (%)\nSaturation gate: 10% (off-scale)",
        xticks=x,
        xticklabels=labels,
    )
    axes[3].axhline(
        0.1,
        color="#a32020",
        linestyle="--",
        linewidth=1.25,
    )
    axes[3].text(
        0.02,
        0.12,
        "dashed: invented-edge gate 0.1%",
        transform=axes[3].get_yaxis_transform(),
        ha="left",
        va="bottom",
        color="#a32020",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
    )
    axes[3].legend(frameon=False)

    raw_uni = [
        _summary_metric(
            summary, item, "raw_to_target_distance", quadrant="overall"
        )
        for item in directions
    ]
    method_uni = [
        _summary_metric(
            summary, item, "method_to_target_distance", quadrant="overall"
        )
        for item in directions
    ]
    raw_uni_estimate, raw_uni_error = _values_and_errors(raw_uni)
    method_uni_estimate, method_uni_error = _values_and_errors(method_uni)
    axes[4].bar(x - width / 2, raw_uni_estimate, width, label="Raw source")
    axes[4].bar(x + width / 2, method_uni_estimate, width, label=method_label)
    axes[4].errorbar(
        x - width / 2,
        raw_uni_estimate,
        yerr=raw_uni_error,
        color="black",
        capsize=3,
        linestyle="none",
    )
    axes[4].errorbar(
        x + width / 2,
        method_uni_estimate,
        yerr=method_uni_error,
        color="black",
        capsize=3,
        linestyle="none",
    )
    axes[4].set(
        title="External frozen-UNI distance (secondary)",
        xticks=x,
        xticklabels=labels,
    )
    axes[4].legend(frameon=False)

    identity = [
        _summary_metric(identity_summary, item, "target_l1_unit")
        for item in directions
    ]
    identity_estimate, identity_error = _values_and_errors(identity)
    axes[5].bar(x, identity_estimate, color=colors)
    axes[5].errorbar(
        x,
        identity_estimate,
        yerr=identity_error,
        color="black",
        capsize=4,
        linestyle="none",
    )
    axes[5].set(
        title="External target-input identity L1",
        xticks=x,
        xticklabels=labels,
    )
    for axis in axes:
        axis.tick_params(axis="x", rotation=12)
    figure.suptitle(f"Frozen full {method_label} models on paired PLISM scanner targets")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def aggregate(
    *,
    output_root: str | Path = OUTPUT_ROOT,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Aggregate model replicates inside each physical PLISM section."""

    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    output = Path(output_root).resolve()
    contract, contract_hash = _validated_contract(output)
    method = str(contract.get("method", "pix2pix")).strip().lower()
    analysis_version = str(contract["analysis_version"])
    directions = _contract_directions(contract)
    direction_names = [direction_name(*item) for item in directions]
    expected_tasks = len(directions) * 5
    method_label = "Pix2Pix" if method == "pix2pix" else "CycleGAN"
    completed_path = output / "02_aggregate/summary.json"
    if completed_path.exists():
        completed = _read_json(completed_path)
        if (
            completed.get("status") == "complete"
            and completed.get("analysis_version") == analysis_version
            and completed.get("contract_sha256") == contract_hash
        ):
            for item in completed.get("outputs", {}).values():
                if sha256(item["path"]) != item["sha256"]:
                    raise ValueError(f"completed aggregate output changed: {item['path']}")
            return {"status": "already_complete", **completed}
        raise FileExistsError(f"refusing to replace aggregate: {completed_path}")
    result_root = output / "02_aggregate"
    if result_root.exists() and any(result_root.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite partial aggregate outputs: {result_root}"
        )
    manifest_paths = sorted(
        (output / "01_model_evaluations").glob("*_to_*/fold_*/evaluation_manifest.json")
    )
    if len(manifest_paths) != expected_tasks:
        raise ValueError(
            f"found {len(manifest_paths)} evaluations; expected {expected_tasks}"
        )
    manifests = [_read_json(path) for path in manifest_paths]
    observed_tasks = {int(item["task_index"]) for item in manifests}
    if observed_tasks != set(range(expected_tasks)):
        raise ValueError(
            f"evaluation tasks differ from 0..{expected_tasks - 1}: {sorted(observed_tasks)}"
        )
    translation_parts = []
    identity_parts = []
    for path, manifest in zip(manifest_paths, manifests):
        if manifest.get("status") != "complete":
            raise ValueError(f"incomplete evaluation: {path}")
        if manifest.get("analysis_version") != analysis_version:
            raise ValueError(f"wrong analysis version: {path}")
        if manifest.get("contract_sha256") != contract_hash:
            raise ValueError(f"evaluation used another contract: {path}")
        for key, destination in (
            ("translation_metrics", translation_parts),
            ("identity_metrics", identity_parts),
        ):
            payload = manifest[key]
            metric_path = Path(payload["path"])
            if sha256(metric_path) != payload["sha256"]:
                raise ValueError(f"evaluation metrics changed: {metric_path}")
            destination.append(pd.read_csv(metric_path))
    translation = pd.concat(translation_parts, ignore_index=True)
    identity = pd.concat(identity_parts, ignore_index=True)
    expected = int(contract["selected_locations"]) * expected_tasks
    if len(translation) != expected or len(identity) != expected:
        raise ValueError("aggregate row count differs from frozen contract")
    if identity["included_in_closure"].astype(str).str.casefold().isin(
        {"true", "1", "yes"}
    ).any():
        raise ValueError("identity control leaked into closure")

    translation_path = result_root / "translation_location_metrics.csv.gz"
    identity_path = result_root / "identity_location_metrics.csv.gz"
    _write_frame(translation, translation_path)
    _write_frame(identity, identity_path)

    core, section = _translation_section_tables(translation)
    core_path = result_root / "translation_core_metrics.csv"
    section_path = result_root / "translation_section_metrics.csv"
    _write_frame(core, core_path)
    _write_frame(section, section_path)
    per_model, ensemble_section, summary = _summarize_models_and_sections(
        section,
        metrics=(*METRICS, *IMAGE_METRICS),
        group_columns=("direction", "quadrant"),
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    per_model_path = result_root / "translation_per_model_summary.csv"
    ensemble_section_path = result_root / "translation_ensemble_section_metrics.csv"
    summary_path = result_root / "translation_ensemble_summary.csv"
    _write_frame(per_model, per_model_path)
    _write_frame(ensemble_section, ensemble_section_path)
    _write_frame(summary, summary_path)

    identity_core, identity_section = _identity_section_tables(identity)
    identity_core_path = result_root / "identity_core_metrics.csv"
    identity_section_path = result_root / "identity_section_metrics.csv"
    _write_frame(identity_core, identity_core_path)
    _write_frame(identity_section, identity_section_path)
    identity_per_model, identity_ensemble_section, identity_summary = (
        _summarize_models_and_sections(
            identity_section,
            metrics=IDENTITY_METRICS,
            group_columns=("direction",),
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 100_000,
        )
    )
    identity_per_model_path = result_root / "identity_per_model_summary.csv"
    identity_ensemble_section_path = result_root / "identity_ensemble_section_metrics.csv"
    identity_summary_path = result_root / "identity_ensemble_summary.csv"
    _write_frame(identity_per_model, identity_per_model_path)
    _write_frame(identity_ensemble_section, identity_ensemble_section_path)
    _write_frame(identity_summary, identity_summary_path)

    paired, contrast = _paired_direction_comparison(
        ensemble_section,
        directions=directions,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    paired_path = result_root / "paired_direction_section_gain.csv"
    contrast_path = result_root / "paired_direction_comparison.csv"
    _write_frame(paired, paired_path)
    _write_frame(contrast, contrast_path)
    gates = _external_gate_table(summary)
    gate_path = result_root / "external_image_safety_gates.csv"
    _write_frame(gates, gate_path)
    figure_path = result_root / "bidirectional_plism_endpoints.png"
    _save_external_endpoint_figure(
        summary,
        identity_summary,
        figure_path,
        method_label=method_label,
        directions=directions,
    )

    contrast_overall = contrast.loc[contrast["quadrant"].eq("overall")].iloc[0]
    narrative_lines = [
        f"# Full {method_label} PLISM external scanner-target validation",
        "",
        f"내부 PanNormal validation으로 고정한 {method_label} 모델을 PLISM에 재학습 없이 적용했다. Image-level fidelity와 safety를 먼저 보고 frozen UNI-v1은 보조 bridge로 해석한다.",
        "",
    ]
    for direction, direction_pair in zip(direction_names, directions):
        l1 = _summary_metric(summary, direction, "target_l1_unit", quadrant="overall")
        ssim = _summary_metric(summary, direction, "target_ssim", quadrant="overall")
        invented = _summary_metric(
            summary, direction, "invented_edge_fraction", quadrant="overall"
        )
        saturation = _summary_metric(
            summary, direction, "saturation_fraction", quadrant="overall"
        )
        gain = _summary_metric(summary, direction, "gain_to_target", quadrant="overall")
        identity_l1 = _summary_metric(identity_summary, direction, "target_l1_unit")
        identity_uni = _summary_metric(
            identity_summary, direction, "identity_uni_distance"
        )
        gate_pass = bool(gates.loc[gates["direction"].eq(direction), "pass"].all())
        narrative_lines.extend(
            [
                f"## {_display_direction(*direction_pair)}",
                "",
                f"- external image target L1 / SSIM: {l1['estimate']:.4f} / {ssim['estimate']:.4f}",
                f"- invented-edge / saturation: {invented['estimate']:.6f} / {saturation['estimate']:.6f}",
                f"- available external image-safety gates: {'PASS' if gate_pass else 'FAIL'}",
                f"- frozen-UNI gain: {gain['estimate']:.4f} (95% CI {gain['ci_low']:.4f}, {gain['ci_high']:.4f})",
                f"- target-input identity image L1 / UNI displacement: {identity_l1['estimate']:.4f} / {identity_uni['estimate']:.4f}",
                "",
            ]
        )
    if all(
        _summary_metric(summary, direction, "gain_to_target", quadrant="overall")[
            "estimate"
        ]
        < 0
        for direction in direction_names
    ):
        direction_interpretation = (
            "두 조건 모두 외부 UNI 거리를 악화시켰다. 양의 contrast는 두 번째 조건의 "
            "악화가 상대적으로 작았다는 뜻이지 성공을 뜻하지 않는다."
        )
    else:
        direction_interpretation = (
            "양수면 두 번째로 열거한 scanner target의 외부 UNI gain이 더 크다는 뜻이다."
        )
    contrast_label = str(contrast_overall["contrast"])
    narrative_lines.extend(
        [
            "## Direction contrast",
            "",
            f"{contrast_label}는 "
            f"{contrast_overall['estimate']:.4f} "
            f"(95% CI {contrast_overall['ci_low']:.4f}, {contrast_overall['ci_high']:.4f})이다. "
            + direction_interpretation,
            "",
            "PLISM 결과는 checkpoint 선택이나 tuning에 사용하지 않았으며, 모든 결과는 외부 site/device/section/stain/acquisition 검증이다.",
        ]
    )
    narrative_path = result_root / "narrative_ko.md"
    narrative_path.write_text("\n".join(narrative_lines) + "\n")

    output_paths = {
        "translation_location_metrics": translation_path,
        "identity_location_metrics": identity_path,
        "translation_core_metrics": core_path,
        "translation_section_metrics": section_path,
        "translation_per_model_summary": per_model_path,
        "translation_ensemble_section_metrics": ensemble_section_path,
        "translation_ensemble_summary": summary_path,
        "identity_core_metrics": identity_core_path,
        "identity_section_metrics": identity_section_path,
        "identity_per_model_summary": identity_per_model_path,
        "identity_ensemble_section_metrics": identity_ensemble_section_path,
        "identity_ensemble_summary": identity_summary_path,
        "paired_direction_section_gain": paired_path,
        "paired_direction_comparison": contrast_path,
        "external_image_safety_gates": gate_path,
        "bidirectional_plism_endpoints": figure_path,
        "narrative_ko": narrative_path,
    }
    result = {
        "analysis_version": analysis_version,
        "method": method,
        "status": "complete",
        "created_utc": utc_now(),
        "contract_sha256": contract_hash,
        "checkpoint_replicates_per_direction": 5,
        "physical_sections": int(translation["stain"].nunique()),
        "location_rows": int(len(translation)),
        "directions": direction_names,
        "primary_endpoint": (
            "gain_to_target; positive means reduced same-location frozen-UNI distance"
        ),
        "direction_contrast": (
            f"{contrast_label}; positive favors {_direction_arrow(*directions[1])}"
        ),
        "inference": contract["aggregation"],
        "quadrant_scope": contract["quadrant_scope"],
        "identity_control_included_in_closure": False,
        "external_results_used_for_selection": False,
        "outputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in output_paths.items()
        },
    }
    result_path = result_root / "summary.json"
    _write_json(result_path, result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    prepare.add_argument("--method", choices=tuple(ANALYSIS_VERSIONS), default="pix2pix")
    prepare.add_argument(
        "--render-contract-root", type=Path, default=RENDER_CONTRACT_ROOT
    )
    prepare.add_argument(
        "--direction",
        action="append",
        default=[],
        metavar="SOURCE:TARGET",
        help=(
            "scanner direction to freeze; pass exactly twice. "
            "Defaults to GT450:AT2 and AT2:GT450"
        ),
    )

    evaluate = subparsers.add_parser("evaluate-task")
    evaluate.add_argument("--task-index", type=int, required=True, choices=range(10))
    evaluate.add_argument("--uni-checkpoint", type=Path)
    evaluate.add_argument("--generator-batch-size", type=int, default=16)
    evaluate.add_argument("--uni-batch-size", type=int, default=32)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--amp", default="bfloat16")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    aggregate_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "prepare":
        directions = DIRECTIONS
        if args.direction:
            parsed = []
            for item in args.direction:
                parts = str(item).split(":")
                if len(parts) != 2:
                    raise ValueError(f"invalid --direction {item!r}; use SOURCE:TARGET")
                parsed.append((parts[0], parts[1]))
            directions = _normalize_directions(tuple(parsed))
        result = prepare_contract(
            output_root=args.output_root,
            runs_root=args.runs_root,
            render_contract_root=args.render_contract_root,
            method=args.method,
            directions=directions,
        )
    elif args.command == "evaluate-task":
        result = evaluate_task(
            args.task_index,
            output_root=args.output_root,
            uni_checkpoint_path=args.uni_checkpoint,
            generator_batch_size=args.generator_batch_size,
            uni_batch_size=args.uni_batch_size,
            device=args.device,
            amp=args.amp,
        )
    elif args.command == "aggregate":
        result = aggregate(
            output_root=args.output_root,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
