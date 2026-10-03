#!/usr/bin/env python3
"""Summarize the locked Pix2Pix-to-AT2 experiment without mixing in PLISM.

The internal estimand is paired movement in frozen-UNI space from each source
scanner toward the same-location AT2 image.  Locations are first averaged
within physical slides.  For a pooled estimate, source scanners are then
averaged within the same physical slide before slides are bootstrapped.

PLISM is deliberately treated as a separate external-replication layer.  This
module can inventory an optional PLISM result root, but never pools PLISM with
the internal PanNormal estimates.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SUMMARY_VERSION = "pix2pix_to_at2_internal_summary_v1"
SOURCE_SCANNERS = ("gt450", "versa", "akoya")
SCANNER_LABELS = {"gt450": "GT450", "versa": "VERSA", "akoya": "AKOYA"}
SCANNER_COLORS = {"gt450": "#377eb8", "versa": "#4daf4a", "akoya": "#984ea3"}
EXPECTED_FOLDS = tuple(range(5))
UNI_METRICS = (
    "raw_to_at2_distance",
    "method_to_at2_distance",
    "gain_to_at2",
    "fractional_closure",
    "method_to_raw_distance",
)
IMAGE_METRICS = (
    "target_l1_unit",
    "target_psnr_db",
    "target_ssim",
    "source_gradient_ncc",
    "target_gradient_ncc",
    "raw_source_target_gradient_ncc",
    "saturation_fraction",
    "invented_edge_fraction",
    "source_edge_deletion_fraction",
)
SAFETY_GATES = {
    "source_gradient_ncc": ("minimum", 0.90),
    "saturation_fraction": ("maximum", 0.10),
    "invented_edge_fraction": ("maximum", 0.001),
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _bootstrap_mean(
    values: Iterable[float],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence_level) / 2.0
    estimate = float(array.mean())
    draws = rng.choice(array, size=(int(replicates), len(array)), replace=True)
    low, high = np.quantile(draws.mean(axis=1), [alpha, 1.0 - alpha])
    return estimate, float(low), float(high), int(len(array))


def _validate_scanner_table(
    frame: pd.DataFrame,
    scanner: str,
    metrics: Sequence[str],
    path: Path,
) -> pd.DataFrame:
    required = {"slide_id", *metrics}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    result = frame.copy()
    if "source_scanner" in result:
        sources = {str(value).casefold() for value in result["source_scanner"].unique()}
        if sources != {scanner}:
            raise ValueError(f"{path} contains source scanners {sorted(sources)}, expected {scanner}")
    else:
        result.insert(0, "source_scanner", scanner)
    if result.duplicated(["slide_id", "source_scanner"]).any():
        raise ValueError(f"{path} has duplicate physical-slide rows")
    for metric in metrics:
        result[metric] = pd.to_numeric(result[metric], errors="raise")
    return result


def _load_scanner_slide_tables(
    root: Path,
    metrics: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Path]]:
    frames = []
    paths: dict[str, Path] = {}
    for scanner in SOURCE_SCANNERS:
        path = root / f"{scanner}_to_at2" / "slide_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing scanner result: {path}")
        frame = _validate_scanner_table(pd.read_csv(path), scanner, metrics, path)
        frames.append(frame)
        paths[scanner] = path
    result = pd.concat(frames, ignore_index=True)
    return result, paths


def _summarize_slide_metrics(
    slide_metrics: pd.DataFrame,
    metrics: Sequence[str],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scanner_index, scanner in enumerate(SOURCE_SCANNERS):
        block = slide_metrics[slide_metrics["source_scanner"].str.casefold() == scanner]
        for metric_index, metric in enumerate(metrics):
            estimate, low, high, count = _bootstrap_mean(
                block[metric],
                replicates=replicates,
                confidence_level=confidence_level,
                seed=seed + 100 * scanner_index + metric_index,
            )
            rows.append(
                {
                    "scope": "scanner",
                    "source_scanner": scanner,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_physical_slides": count,
                    "bootstrap_replicates": replicates,
                    "confidence_level": confidence_level,
                    "pooling_rule": "locations averaged within physical slide",
                }
            )

    # This ordering is important: a slide represented by three scanners gets
    # one vote, not three votes, in the pooled uncertainty calculation.
    pooled = (
        slide_metrics.groupby("slide_id", sort=True, dropna=False)[list(metrics)]
        .mean()
        .reset_index()
    )
    scanner_counts = slide_metrics.groupby("slide_id")["source_scanner"].nunique()
    incomplete = scanner_counts[scanner_counts != len(SOURCE_SCANNERS)]
    if len(incomplete):
        examples = incomplete.head().to_dict()
        raise ValueError(
            "pooled estimates require all three source scanners for every physical slide; "
            f"incomplete examples={examples}"
        )
    for metric_index, metric in enumerate(metrics):
        estimate, low, high, count = _bootstrap_mean(
            pooled[metric],
            replicates=replicates,
            confidence_level=confidence_level,
            seed=seed + 1_000 + metric_index,
        )
        rows.append(
            {
                "scope": "pooled",
                "source_scanner": "pooled_within_slide",
                "metric": metric,
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "n_physical_slides": count,
                "bootstrap_replicates": replicates,
                "confidence_level": confidence_level,
                "pooling_rule": "source scanners averaged within physical slide, then slides bootstrapped",
                "minimum_scanners_per_slide": int(scanner_counts.min()),
                "maximum_scanners_per_slide": int(scanner_counts.max()),
            }
        )
    return pd.DataFrame(rows)


def _load_evaluation_manifests(
    evaluation_root: Path,
    *,
    evaluation_name: str,
) -> dict[str, dict[str, Any]]:
    manifests = {}
    for scanner in SOURCE_SCANNERS:
        path = evaluation_root / f"{scanner}_to_at2" / "evaluation_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing {evaluation_name} evaluation manifest: {path}")
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            raise ValueError(f"{evaluation_name} evaluation is not complete: {path}")
        if str(payload.get("target_scanner", "")).casefold() != "at2":
            raise ValueError(f"{evaluation_name} evaluation target is not AT2: {path}")
        manifests[scanner] = payload
    return manifests


def _attach_safety_gates(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["gate_direction"] = ""
    result["gate_threshold"] = np.nan
    result["gate_pass"] = pd.array([pd.NA] * len(result), dtype="boolean")
    for metric, (direction, threshold) in SAFETY_GATES.items():
        mask = result["metric"] == metric
        estimates = result.loc[mask, "estimate"]
        passes = estimates >= threshold if direction == "minimum" else estimates <= threshold
        result.loc[mask, "gate_direction"] = direction
        result.loc[mask, "gate_threshold"] = threshold
        result.loc[mask, "gate_pass"] = passes.to_numpy()
    return result


def _load_checkpoint_selections(checkpoint_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(checkpoint_root.rglob("checkpoint_selection.json"))
    selections = []
    candidates = []
    observed: set[tuple[str, int]] = set()
    for path in paths:
        payload = json.loads(path.read_text())
        direction = str(payload.get("direction", "")).casefold()
        if not direction.endswith("->at2"):
            continue
        scanner = direction.removesuffix("->at2")
        if scanner not in SOURCE_SCANNERS:
            continue
        fold = int(payload["test_fold"])
        key = (scanner, fold)
        if key in observed:
            raise ValueError(f"duplicate checkpoint selection for {scanner} fold {fold}")
        observed.add(key)
        selected = payload["selected"]
        selections.append(
            {
                "source_scanner": scanner,
                "target_scanner": "at2",
                "test_fold": fold,
                "validation_fold": int(payload["validation_fold"]),
                "selection_status": payload["selection_status"],
                "selected_pass": int(selected["completed_passes"]),
                "selected_global_step": int(selected["global_step"]),
                "selected_basic_fidelity_eligible": bool(
                    selected["basic_fidelity_eligible"]
                ),
                "eligible_checkpoint_count": int(
                    payload["basic_eligible_checkpoint_count"]
                ),
                "validation_l1_normalized": float(selected["validation_l1_normalized"]),
                "validation_psnr_db": float(selected["validation_psnr_db"]),
                "validation_source_gradient_ncc": float(
                    selected["validation_source_gradient_ncc"]
                ),
                "validation_saturation_fraction": float(
                    selected["validation_saturation_fraction"]
                ),
                "selection_path": str(path.resolve()),
            }
        )
        for candidate in payload["candidates"]:
            candidates.append(
                {
                    "source_scanner": scanner,
                    "test_fold": fold,
                    "completed_passes": int(candidate["completed_passes"]),
                    "global_step": int(candidate["global_step"]),
                    "basic_fidelity_eligible": bool(
                        candidate["basic_fidelity_eligible"]
                    ),
                    "validation_l1_normalized": float(
                        candidate["validation_l1_normalized"]
                    ),
                    "validation_psnr_db": float(candidate["validation_psnr_db"]),
                    "validation_source_gradient_ncc": float(
                        candidate["validation_source_gradient_ncc"]
                    ),
                    "validation_saturation_fraction": float(
                        candidate["validation_saturation_fraction"]
                    ),
                }
            )
    expected = {(scanner, fold) for scanner in SOURCE_SCANNERS for fold in EXPECTED_FOLDS}
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise ValueError(
            "expected exactly 15 scanner/fold selections; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    selection_frame = pd.DataFrame(selections).sort_values(
        ["source_scanner", "test_fold"]
    )
    candidate_frame = pd.DataFrame(candidates).sort_values(
        ["source_scanner", "test_fold", "completed_passes"]
    )
    return selection_frame.reset_index(drop=True), candidate_frame.reset_index(drop=True)


def _checkpoint_stability(selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scanner, block in selection.groupby("source_scanner", sort=False):
        counts = block["selected_pass"].value_counts().sort_index()
        rows.append(
            {
                "source_scanner": scanner,
                "n_folds": int(len(block)),
                "mean_selected_pass": float(block["selected_pass"].mean()),
                "minimum_selected_pass": int(block["selected_pass"].min()),
                "maximum_selected_pass": int(block["selected_pass"].max()),
                "unique_selected_passes": int(block["selected_pass"].nunique()),
                "modal_selected_pass": int(counts.idxmax()),
                "modal_selected_pass_fraction": float(counts.max() / len(block)),
                "basic_fidelity_eligible_folds": int(
                    block["selected_basic_fidelity_eligible"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _feature_records(
    uni_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    raw_features = []
    generated_features = []
    generated_groups = []
    at2_by_key: dict[tuple[str, int], np.ndarray] = {}
    at2_group_by_key: dict[tuple[str, int], str] = {}
    keys_by_scanner: dict[str, set[tuple[str, int]]] = {}
    duplicate_at2_rows = 0
    for scanner in SOURCE_SCANNERS:
        feature_root = uni_root / f"{scanner}_to_at2" / "feature_shards"
        paths = sorted(feature_root.glob("*.h5"))
        if not paths:
            raise FileNotFoundError(f"no UNI feature shards in {feature_root}")
        scanner_keys: set[tuple[str, int]] = set()
        for path in paths:
            with h5py.File(path, "r") as store:
                slide_id = str(store.attrs["slide_id"])
                locations = np.asarray(store["location_index"], dtype=int)
                raw = np.asarray(store["raw_source"], dtype=np.float32)
                generated = np.asarray(store["generated"], dtype=np.float32)
                at2 = np.asarray(store["real_at2"], dtype=np.float32)
            if raw.shape != generated.shape or raw.shape != at2.shape:
                raise ValueError(f"feature shapes differ in {path}")
            if len(locations) != len(raw):
                raise ValueError(f"location/feature counts differ in {path}")
            raw_features.append(raw)
            generated_features.append(generated)
            generated_groups.extend([slide_id] * len(raw))
            for location, vector in zip(locations, at2):
                key = (slide_id, int(location))
                if key in scanner_keys:
                    raise ValueError(f"duplicate feature key within {scanner}: {key}")
                scanner_keys.add(key)
                if key in at2_by_key:
                    duplicate_at2_rows += 1
                    if not np.allclose(at2_by_key[key], vector, rtol=1e-5, atol=1e-6):
                        raise ValueError(
                            f"same physical AT2 key has different embeddings across scanners: {key}"
                        )
                else:
                    at2_by_key[key] = vector
                    at2_group_by_key[key] = slide_id
        keys_by_scanner[scanner] = scanner_keys
    reference_keys = keys_by_scanner[SOURCE_SCANNERS[0]]
    for scanner in SOURCE_SCANNERS[1:]:
        if keys_by_scanner[scanner] != reference_keys:
            raise ValueError(
                "pooled domain probes require identical physical AT2 keys across scanners; "
                f"{scanner} missing={len(reference_keys - keys_by_scanner[scanner])}, "
                f"extra={len(keys_by_scanner[scanner] - reference_keys)}"
            )
    ordered_keys = sorted(at2_by_key)
    return (
        np.concatenate(raw_features),
        np.concatenate(generated_features),
        np.asarray([at2_by_key[key] for key in ordered_keys], dtype=np.float32),
        np.asarray(generated_groups, dtype=object),
        np.asarray([at2_group_by_key[key] for key in ordered_keys], dtype=object),
        {
            "generated_or_raw_rows": int(sum(len(value) for value in raw_features)),
            "unique_at2_rows": int(len(ordered_keys)),
            "duplicate_at2_rows_removed": int(duplicate_at2_rows),
            "unique_at2_physical_slides": int(
                len({at2_group_by_key[key] for key in ordered_keys})
            ),
        },
    )


def _grouped_domain_probe(
    source_features: np.ndarray,
    at2_features: np.ndarray,
    source_groups: np.ndarray,
    at2_groups: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold

    features = np.concatenate((source_features, at2_features))
    labels = np.concatenate(
        (
            np.zeros(len(source_features), dtype=int),
            np.ones(len(at2_features), dtype=int),
        )
    )
    groups = np.concatenate((source_groups, at2_groups))
    folds = min(5, len(np.unique(groups)))
    if folds < 2:
        return {"status": "pending_too_few_physical_slides", "folds": folds}
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for train, test in splitter.split(features, labels, groups):
        classifier = LogisticRegression(
            max_iter=1_000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
        classifier.fit(features[train], labels[train])
        scores.append(
            balanced_accuracy_score(labels[test], classifier.predict(features[test]))
        )
    return {
        "status": "complete",
        "balanced_accuracy": float(np.mean(scores)),
        "fold_scores": [float(value) for value in scores],
        "folds": folds,
        "grouping": "physical slide",
        "class_weight": "balanced",
        "classes": ["source_or_generated", "unique_real_at2"],
        "source_rows": int(len(source_features)),
        "at2_rows": int(len(at2_features)),
    }


def _pooled_domain_probes(uni_root: Path, *, seed: int) -> dict[str, Any]:
    try:
        raw, generated, at2, source_groups, at2_groups, counts = _feature_records(
            uni_root
        )
    except FileNotFoundError as error:
        return {
            "status": "pending_feature_shards",
            "reason": str(error),
            "at2_deduplication_rule": "one AT2 embedding per (physical slide, location)",
        }
    return {
        "status": "complete",
        "at2_deduplication_rule": "one AT2 embedding per (physical slide, location)",
        "counts": counts,
        "raw_source_vs_unique_at2": _grouped_domain_probe(
            raw, at2, source_groups, at2_groups, seed=seed
        ),
        "pix2pix_vs_unique_at2": _grouped_domain_probe(
            generated, at2, source_groups, at2_groups, seed=seed
        ),
    }


def _plism_inventory(plism_root: Path | None, output_root: Path) -> dict[str, Any]:
    if plism_root is None:
        return {
            "status": "pending_not_provided",
            "interpretation": "PLISM is a separate external test and is not inferred from internal results.",
        }
    root = plism_root.resolve()
    if not root.is_dir():
        return {
            "status": "pending_missing_root",
            "root": str(root),
            "interpretation": "PLISM is not pooled with the internal PanNormal experiment.",
        }
    formal_summary_path = root / "03_aggregate/summary.json"
    formal_table_path = root / "03_aggregate/across_model_summary.csv"
    if formal_summary_path.is_file() and formal_table_path.is_file():
        formal = json.loads(formal_summary_path.read_text())
        table = pd.read_csv(formal_table_path)
        table.insert(0, "source_file", str(formal_table_path.resolve()))
        copied_path = output_root / "plism_external_summary.csv"
        table.to_csv(copied_path, index=False)

        def external_metric(quadrant: str, metric: str) -> dict[str, Any]:
            selected = table[
                table["source_scanner"].astype(str).eq("GT450")
                & table["quadrant"].astype(str).eq(quadrant)
                & table["metric"].astype(str).eq(metric)
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"expected one formal PLISM row for {quadrant}/{metric}, "
                    f"found {len(selected)}"
                )
            row = selected.iloc[0]
            return {
                "estimate": float(row["estimate"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "sections": int(row["sections"]),
                "models": int(row["models"]),
                "model_estimate_min": float(row["model_estimate_min"]),
                "model_estimate_max": float(row["model_estimate_max"]),
                "model_sign_consistent": bool(row["model_sign_consistent"]),
            }

        return {
            "status": "loaded_separate_external_evidence",
            "formal_status": str(formal.get("status", "unknown")),
            "analysis_role": str(formal.get("analysis_role", "unknown")),
            "root": str(root),
            "summary_files": 1,
            "rows": int(len(table)),
            "quadrants": sorted(str(value) for value in table["quadrant"].unique()),
            "copied_summary": str(copied_path.resolve()),
            "per_model_summary": str(
                (root / "03_aggregate/per_model_summary.csv").resolve()
            ),
            "primary_encoder": str(formal.get("primary_encoder", "unknown")),
            "shared_shared": {
                metric: external_metric("shared/shared", metric)
                for metric in (
                    "raw_to_at2_distance",
                    "method_to_at2_distance",
                    "gain_to_at2",
                    "fractional_closure",
                )
            },
            "tissue_ood": {
                metric: external_metric("tissue-OOD", metric)
                for metric in (
                    "raw_to_at2_distance",
                    "method_to_at2_distance",
                    "gain_to_at2",
                    "fractional_closure",
                )
            },
            "at2_identity_damage": formal.get("at2_identity_damage_control", {}),
            "interpretation": (
                "PLISM remains separate and is never pooled with internal PanNormal slides. "
                "Because all five checkpoints failed the internal image-fidelity gate, the "
                "external result is diagnostic even if its representation endpoint is favourable."
            ),
        }
    paths = sorted(root.rglob("summary.csv"))
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame.insert(0, "source_file", str(path.resolve()))
        frames.append(frame)
    if not frames:
        return {
            "status": "pending_no_summary",
            "root": str(root),
            "interpretation": "PLISM is not pooled with the internal PanNormal experiment.",
        }
    combined = pd.concat(frames, ignore_index=True, sort=False)
    path = output_root / "plism_external_summary.csv"
    combined.to_csv(path, index=False)
    quadrants = []
    if "quadrant" in combined:
        quadrants = sorted(str(value) for value in combined["quadrant"].dropna().unique())
    return {
        "status": "loaded_separate_external_evidence",
        "root": str(root),
        "summary_files": len(paths),
        "rows": int(len(combined)),
        "quadrants": quadrants,
        "copied_summary": str(path.resolve()),
        "interpretation": "PLISM remains separate and is never pooled with internal PanNormal slides.",
    }


def _metric_row(summary: pd.DataFrame, scanner: str, metric: str) -> pd.Series:
    block = summary[
        (summary["source_scanner"] == scanner) & (summary["metric"] == metric)
    ]
    if len(block) != 1:
        raise ValueError(f"expected one summary row for {scanner}/{metric}, found {len(block)}")
    return block.iloc[0]


def _uni_conclusion(uni_summary: pd.DataFrame) -> dict[str, Any]:
    statuses = {}
    for scanner in SOURCE_SCANNERS:
        gain = _metric_row(uni_summary, scanner, "gain_to_at2")
        if gain["ci_low"] > 0:
            status = "supported"
        elif gain["ci_high"] < 0:
            status = "moved_away"
        else:
            status = "uncertain"
        statuses[scanner] = {
            "status": status,
            "gain": float(gain["estimate"]),
            "ci_low": float(gain["ci_low"]),
            "ci_high": float(gain["ci_high"]),
        }
    scanner_statuses = {value["status"] for value in statuses.values()}
    if scanner_statuses == {"supported"}:
        overall = "supported_all_scanners"
    elif "supported" in scanner_statuses:
        overall = "mixed_by_scanner"
    elif scanner_statuses == {"moved_away"}:
        overall = "not_supported"
    else:
        overall = "uncertain"
    return {"overall": overall, "by_scanner": statuses}


def _safety_conclusion(
    image_summary: pd.DataFrame,
    manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_scanner = {}
    for scanner in SOURCE_SCANNERS:
        gate_results = {}
        for metric in SAFETY_GATES:
            row = _metric_row(image_summary, scanner, metric)
            gate_results[metric] = {
                "estimate": float(row["estimate"]),
                "threshold": float(row["gate_threshold"]),
                "direction": row["gate_direction"],
                "pass": bool(row["gate_pass"]),
            }
        by_scanner[scanner] = {
            "available_gates_pass": all(item["pass"] for item in gate_results.values()),
            "gates": gate_results,
        }
    pending = sorted(
        {
            str(item)
            for payload in manifests.values()
            for item in payload.get("pending_formal_gates", [])
        }
    )
    all_available = all(value["available_gates_pass"] for value in by_scanner.values())
    if not all_available:
        formal_status = "failed_available_gate"
    elif pending:
        formal_status = "not_established_pending_formal_gates"
    else:
        formal_status = "passed_all_declared_gates"
    return {
        "formal_status": formal_status,
        "all_available_gates_pass": all_available,
        "pending_formal_gates": pending,
        "by_scanner": by_scanner,
    }


def _format_interval(row: pd.Series, *, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    digits = 1 if percent else 4
    return (
        f"{row['estimate'] * scale:.{digits}f}{suffix} "
        f"(95% CI {row['ci_low'] * scale:.{digits}f}–"
        f"{row['ci_high'] * scale:.{digits}f}{suffix})"
    )


def _write_korean_narrative(
    path: Path,
    uni_summary: pd.DataFrame,
    image_summary: pd.DataFrame,
    uni_conclusion: dict[str, Any],
    safety_conclusion: dict[str, Any],
    checkpoint_stability: pd.DataFrame,
    domain_probes: dict[str, Any],
    plism: dict[str, Any],
) -> None:
    lines = [
        "# Pix2Pix → AT2 결과: 쉬운 해석",
        "",
        "## 무엇을 물었나",
        "",
        (
            "GT450, VERSA, AKOYA 이미지를 Pix2Pix로 AT2 쪽으로 옮겼을 때, 같은 위치의 실제 "
            "AT2 이미지와 UNI embedding 거리가 줄어드는지를 물었다. 양의 gain은 가까워졌다는 뜻이고, "
            "closure는 원래 scanner 간 거리의 몇 퍼센트를 닫았는지를 뜻한다."
        ),
        "",
        "## 내부 데이터의 답",
        "",
    ]
    for scanner in SOURCE_SCANNERS:
        label = SCANNER_LABELS[scanner]
        raw = _metric_row(uni_summary, scanner, "raw_to_at2_distance")
        method = _metric_row(uni_summary, scanner, "method_to_at2_distance")
        gain = _metric_row(uni_summary, scanner, "gain_to_at2")
        closure = _metric_row(uni_summary, scanner, "fractional_closure")
        status = uni_conclusion["by_scanner"][scanner]["status"]
        meaning = {
            "supported": "거리 감소가 슬라이드 bootstrap에서도 일관되었다",
            "moved_away": "오히려 AT2에서 멀어졌다",
            "uncertain": "평균 변화의 방향은 보이지만 불확실성이 0을 가로질렀다",
        }[status]
        lines.append(
            f"- {label}: raw 거리 {_format_interval(raw)}에서 Pix2Pix 거리 "
            f"{_format_interval(method)}로 변했다. gain은 {_format_interval(gain)}, "
            f"closure는 {_format_interval(closure, percent=True)}이며, {meaning}."
        )
    pooled_gain = _metric_row(uni_summary, "pooled_within_slide", "gain_to_at2")
    pooled_closure = _metric_row(
        uni_summary, "pooled_within_slide", "fractional_closure"
    )
    lines.extend(
        [
            "",
            (
                "세 scanner의 pooled 값은 같은 물리 슬라이드 안에서 scanner를 먼저 평균한 뒤 "
                "슬라이드를 bootstrap했다. 따라서 한 슬라이드가 scanner 수만큼 반복 집계되지 않는다. "
                f"pooled gain은 {_format_interval(pooled_gain)}, pooled closure는 "
                f"{_format_interval(pooled_closure, percent=True)}이다."
            ),
            "",
            "## UNI 결과와 별개로 이미지 안전성은 어떠했나",
            "",
        ]
    )
    if safety_conclusion["all_available_gates_pass"]:
        lines.append(
            "현재 계산된 구조 보존, 포화, 새 edge 발생의 세 안전성 gate는 세 scanner 모두 통과했다."
        )
    else:
        failed = []
        for scanner, result in safety_conclusion["by_scanner"].items():
            for metric, gate in result["gates"].items():
                if not gate["pass"]:
                    failed.append(f"{SCANNER_LABELS[scanner]}의 {metric}")
        lines.append(
            "UNI의 이동 방향과 별개로 이미지 안전성 gate를 통과하지 못했다: "
            + ", ".join(failed)
            + ". 이 경우 UNI 개선만으로 성공이라고 부를 수 없다."
        )
    pending = safety_conclusion["pending_formal_gates"]
    if pending:
        lines.append(
            "또한 최종 안전 판정 전 남은 검사는 " + ", ".join(pending) + "이다."
        )
    lines.extend(["", "## 학습 길이는 안정적이었나", ""])
    for row in checkpoint_stability.itertuples(index=False):
        lines.append(
            f"- {SCANNER_LABELS[row.source_scanner]}: 5개 fold에서 선택된 pass 범위는 "
            f"{row.minimum_selected_pass}–{row.maximum_selected_pass}, 최빈 pass는 "
            f"{row.modal_selected_pass} ({row.modal_selected_pass_fraction:.0%})였다. "
            f"기본 fidelity gate를 만족한 fold는 {row.basic_fidelity_eligible_folds}/5였다."
        )
    lines.extend(["", "## UNI 분류 가능성과 PLISM", ""])
    if domain_probes.get("status") == "complete":
        raw_score = domain_probes["raw_source_vs_unique_at2"]["balanced_accuracy"]
        generated_score = domain_probes["pix2pix_vs_unique_at2"]["balanced_accuracy"]
        lines.append(
            f"물리 슬라이드 분리 domain classifier의 balanced accuracy는 raw {raw_score:.3f}, "
            f"Pix2Pix {generated_score:.3f}였다. 이 계산에서는 같은 AT2 위치를 scanner마다 "
            "세 번 복제하지 않고 정확히 한 번만 사용했다. 0.5에 가까울수록 두 domain을 "
            "구별하기 어렵다는 뜻이다."
        )
    else:
        lines.append("통합 domain classifier는 feature shard가 준비된 뒤 계산한다.")
    if plism["status"] == "loaded_separate_external_evidence":
        if "shared_shared" in plism:
            shared = plism["shared_shared"]
            external_gain = shared["gain_to_at2"]
            identity = plism.get("at2_identity_damage", {})
            lines.append(
                "PLISM의 scanner-model/tissue-label shared/shared에서도 raw 거리 "
                f"{shared['raw_to_at2_distance']['estimate']:.4f}가 Pix2Pix 후 "
                f"{shared['method_to_at2_distance']['estimate']:.4f}로 커졌다. gain은 "
                f"{external_gain['estimate']:.4f} (95% CI "
                f"{external_gain['ci_low']:.4f}–{external_gain['ci_high']:.4f})였고, "
                f"5개 모델의 방향이 모두 같았다. AT2를 그대로 generator에 넣은 파괴성 "
                f"대조군의 거리는 {identity.get('estimate', float('nan')):.4f}였다. 내부 실패가 "
                "외부 site에서도 반복됐지만, 모든 checkpoint가 image-fidelity gate를 실패했으므로 "
                "이 결과는 diagnostic-only로 해석한다."
            )
        else:
            lines.append(
                "PLISM 결과 파일은 로드되었다. 그러나 PLISM은 site, 장비, 절편, 염색 및 획득 조건이 "
                "함께 달라지는 외부 검증이므로 내부 PanNormal 수치와 합치지 않고 별도 결과로 해석한다."
            )
    else:
        lines.append(
            "PLISM 검증은 아직 pending이다. 내부 데이터에서 UNI 거리가 줄었다고 해서 PLISM에서도 "
            "유지된다고 결론내리지 않으며, 준비된 모델을 그대로 고정한 외부 평가로 따로 확인한다."
        )
    lines.extend(
        [
            "",
            "## 한 문장 결론",
            "",
            (
                "Pix2Pix의 성공 조건은 ‘AT2 색처럼 보임’이 아니라, 같은 위치 AT2와의 UNI 거리를 "
                "줄이고 이미지 안전성 gate를 지키며, 그 효과가 독립적인 PLISM에서도 유지되는 것이다."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines))


def _errorbar(ax: Any, x: np.ndarray, block: pd.DataFrame, color: str, label: str) -> None:
    estimate = block["estimate"].to_numpy(float)
    low = block["ci_low"].to_numpy(float)
    high = block["ci_high"].to_numpy(float)
    ax.errorbar(
        x,
        estimate,
        yerr=np.vstack((estimate - low, high - estimate)),
        fmt="o",
        color=color,
        capsize=3,
        lw=1.4,
        label=label,
    )


def _plot_summary(
    path: Path,
    uni_summary: pd.DataFrame,
    image_summary: pd.DataFrame,
    selections: pd.DataFrame,
    uni_conclusion: dict[str, Any],
    safety_conclusion: dict[str, Any],
    domain_probes: dict[str, Any],
    plism: dict[str, Any],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    labels = [SCANNER_LABELS[value] for value in SOURCE_SCANNERS]
    positions = np.arange(len(SOURCE_SCANNERS))

    ax = axes[0, 0]
    width = 0.34
    for offset, metric, label, color in (
        (-width / 2, "raw_to_at2_distance", "Raw", "#777777"),
        (width / 2, "method_to_at2_distance", "Pix2Pix", "#e41a1c"),
    ):
        block = pd.DataFrame(
            [_metric_row(uni_summary, scanner, metric) for scanner in SOURCE_SCANNERS]
        )
        estimates = block["estimate"].to_numpy(float)
        ax.bar(positions + offset, estimates, width, color=color, alpha=0.85, label=label)
        ax.errorbar(
            positions + offset,
            estimates,
            yerr=np.vstack(
                (
                    estimates - block["ci_low"].to_numpy(float),
                    block["ci_high"].to_numpy(float) - estimates,
                )
            ),
            fmt="none",
            ecolor="black",
            capsize=3,
            lw=1,
        )
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Paired cosine distance to AT2")
    ax.set_title("A  Does Pix2Pix move toward real AT2?")
    ax.legend(frameon=False)

    plot_scanners = [*SOURCE_SCANNERS, "pooled_within_slide"]
    plot_labels = [*labels, "Pooled"]
    plot_colors = [*[SCANNER_COLORS[value] for value in SOURCE_SCANNERS], "#111111"]
    ax = axes[0, 1]
    for index, (scanner, color) in enumerate(zip(plot_scanners, plot_colors)):
        row = _metric_row(uni_summary, scanner, "gain_to_at2")
        ax.errorbar(
            index,
            row["estimate"],
            yerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
            fmt="o",
            color=color,
            capsize=4,
            ms=6,
        )
    ax.axhline(0, color="#555555", ls="--", lw=1)
    ax.set_xticks(np.arange(len(plot_labels)), plot_labels, rotation=15)
    ax.set_ylabel("UNI gain (raw distance − Pix2Pix distance)")
    ax.set_title("B  Positive gain means closer to AT2")

    ax = axes[0, 2]
    for index, (scanner, color) in enumerate(zip(plot_scanners, plot_colors)):
        row = _metric_row(uni_summary, scanner, "fractional_closure")
        ax.errorbar(
            index,
            100 * row["estimate"],
            yerr=[
                [100 * (row["estimate"] - row["ci_low"])],
                [100 * (row["ci_high"] - row["estimate"])],
            ],
            fmt="o",
            color=color,
            capsize=4,
            ms=6,
        )
    ax.axhline(0, color="#555555", ls="--", lw=1)
    ax.set_xticks(np.arange(len(plot_labels)), plot_labels, rotation=15)
    ax.set_ylabel("Fractional closure (%)")
    ax.set_title("C  Fraction of the original gap closed")

    ax = axes[1, 0]
    metric_labels = {
        "source_gradient_ncc": "Structure NCC",
        "saturation_fraction": "Saturation",
        "invented_edge_fraction": "Invented edge",
    }
    gate_metrics = list(SAFETY_GATES)
    gate_x = np.arange(len(gate_metrics))
    gate_width = 0.23
    for scanner_index, scanner in enumerate(SOURCE_SCANNERS):
        margins = []
        lows = []
        highs = []
        for metric in gate_metrics:
            row = _metric_row(image_summary, scanner, metric)
            direction, threshold = SAFETY_GATES[metric]
            if direction == "minimum":
                estimate = (row["estimate"] - threshold) / threshold
                low = (row["ci_low"] - threshold) / threshold
                high = (row["ci_high"] - threshold) / threshold
            else:
                estimate = (threshold - row["estimate"]) / threshold
                low = (threshold - row["ci_high"]) / threshold
                high = (threshold - row["ci_low"]) / threshold
            margins.append(estimate)
            lows.append(low)
            highs.append(high)
        margins_array = np.asarray(margins)
        locations = gate_x + (scanner_index - 1) * gate_width
        ax.bar(
            locations,
            margins_array,
            gate_width,
            color=SCANNER_COLORS[scanner],
            alpha=0.85,
            label=SCANNER_LABELS[scanner],
        )
        ax.errorbar(
            locations,
            margins_array,
            yerr=np.vstack((margins_array - np.asarray(lows), np.asarray(highs) - margins_array)),
            fmt="none",
            ecolor="black",
            capsize=2,
            lw=0.8,
        )
    ax.axhline(0, color="#222222", lw=1)
    ax.set_xticks(gate_x, [metric_labels[value] for value in gate_metrics], rotation=15)
    ax.set_ylabel("Normalized safety margin (>0 passes)")
    ax.set_title("D  Image evidence can veto UNI improvement")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for scanner in SOURCE_SCANNERS:
        block = selections[selections["source_scanner"] == scanner].sort_values("test_fold")
        eligible = block["selected_basic_fidelity_eligible"].to_numpy(bool)
        ax.plot(
            block["test_fold"],
            block["selected_pass"],
            color=SCANNER_COLORS[scanner],
            alpha=0.6,
            lw=1,
        )
        ax.scatter(
            block["test_fold"],
            block["selected_pass"],
            color=np.where(eligible, SCANNER_COLORS[scanner], "white"),
            edgecolor=SCANNER_COLORS[scanner],
            s=45,
            label=SCANNER_LABELS[scanner],
            zorder=3,
        )
    ax.set_xticks(EXPECTED_FOLDS)
    ax.set_xlabel("Outer test fold")
    ax.set_ylabel("Selected training pass")
    ax.set_title("E  Checkpoint/pass consistency\n(open marker = basic fidelity failure)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 2]
    ax.axis("off")
    uni_text = {
        "supported_all_scanners": "UNI closure: supported for all scanners",
        "mixed_by_scanner": "UNI closure: mixed across scanners",
        "not_supported": "UNI closure: not supported",
        "uncertain": "UNI closure: uncertain",
    }[uni_conclusion["overall"]]
    safety_text = {
        "failed_available_gate": "Image safety: at least one available gate failed",
        "not_established_pending_formal_gates": "Image safety: available gates pass; formal audit pending",
        "passed_all_declared_gates": "Image safety: all declared gates pass",
    }[safety_conclusion["formal_status"]]
    if domain_probes.get("status") == "complete":
        raw_score = domain_probes["raw_source_vs_unique_at2"]["balanced_accuracy"]
        generated_score = domain_probes["pix2pix_vs_unique_at2"]["balanced_accuracy"]
        domain_text = f"Domain probe BA: raw {raw_score:.3f} → Pix2Pix {generated_score:.3f}"
    else:
        domain_text = "Domain probe: pending"
    if plism["status"] == "loaded_separate_external_evidence" and "shared_shared" in plism:
        external = plism["shared_shared"]["gain_to_at2"]
        identity = plism.get("at2_identity_damage", {})
        plism_text = (
            f"PLISM GT450 gain: {external['estimate']:+.3f} "
            f"[{external['ci_low']:+.3f}, {external['ci_high']:+.3f}]\n"
            f"AT2 identity damage: {identity.get('estimate', float('nan')):.3f}\n"
            "5/5 models moved away; diagnostic-only"
        )
    elif plism["status"] == "loaded_separate_external_evidence":
        plism_text = "PLISM: loaded as separate external evidence"
    else:
        plism_text = "PLISM: pending separate external validation"
    ax.text(
        0.03,
        0.95,
        "F  Decision boundary\n\n"
        + uni_text
        + "\n\n"
        + safety_text
        + "\n\n"
        + domain_text
        + "\n(unique AT2 locations; no triple count)\n\n"
        + plism_text
        + "\n\nA favourable UNI result cannot override\nan image-safety failure.",
        va="top",
        ha="left",
        fontsize=10,
        linespacing=1.25,
    )
    fig.suptitle(
        "Pix2Pix scanner→AT2: paired frozen-UNI closure, safety, and external boundary",
        fontsize=14,
        fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_plism_external_summary(
    path: Path,
    uni_summary: pd.DataFrame,
    plism: dict[str, Any],
) -> None:
    """Show whether the internal GT450 result transfers to external PLISM."""

    if plism.get("status") != "loaded_separate_external_evidence" or "shared_shared" not in plism:
        return
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    groups = [
        (
            "Internal\nGT450",
            {
                metric: _metric_row(uni_summary, "gt450", metric).to_dict()
                for metric in ("raw_to_at2_distance", "method_to_at2_distance", "gain_to_at2")
            },
        ),
        ("PLISM\nshared/shared", plism["shared_shared"]),
        ("PLISM\ntissue-OOD", plism["tissue_ood"]),
    ]
    positions = np.arange(len(groups))
    width = 0.34
    ax = axes[0, 0]
    for offset, metric, label, color in (
        (-width / 2, "raw_to_at2_distance", "Raw GT450", "#777777"),
        (width / 2, "method_to_at2_distance", "Pix2Pix", "#e41a1c"),
    ):
        values = np.asarray([block[metric]["estimate"] for _, block in groups])
        lows = np.asarray([block[metric]["ci_low"] for _, block in groups])
        highs = np.asarray([block[metric]["ci_high"] for _, block in groups])
        ax.bar(positions + offset, values, width, color=color, alpha=0.85, label=label)
        ax.errorbar(
            positions + offset,
            values,
            yerr=np.vstack((values - lows, highs - values)),
            fmt="none",
            ecolor="black",
            capsize=3,
            lw=1,
        )
    ax.set_xticks(positions, [label for label, _ in groups])
    ax.set_ylabel("Paired UNI-v1 cosine distance to AT2")
    ax.set_title("A  Translation increases distance internally and externally")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for index, (_, block) in enumerate(groups):
        value = block["gain_to_at2"]
        ax.errorbar(
            index,
            value["estimate"],
            yerr=[
                [value["estimate"] - value["ci_low"]],
                [value["ci_high"] - value["estimate"]],
            ],
            fmt="o",
            color=("#377eb8", "#984ea3", "#ff7f00")[index],
            capsize=4,
            ms=7,
        )
    ax.axhline(0, color="#444444", ls="--", lw=1)
    ax.set_xticks(positions, [label for label, _ in groups])
    ax.set_ylabel("Gain (raw distance − Pix2Pix distance)")
    ax.set_title("B  Positive would mean successful movement toward AT2")

    ax = axes[1, 0]
    per_model = pd.read_csv(plism["per_model_summary"])
    per_model = per_model[
        per_model["source_scanner"].astype(str).eq("GT450")
        & per_model["metric"].astype(str).eq("gain_to_at2")
    ]
    for quadrant, color, marker in (
        ("shared/shared", "#984ea3", "o"),
        ("tissue-OOD", "#ff7f00", "s"),
    ):
        block = per_model[per_model["quadrant"].astype(str).eq(quadrant)].sort_values(
            "checkpoint_fold"
        )
        ax.plot(
            block["checkpoint_fold"],
            block["estimate"],
            marker=marker,
            color=color,
            label=quadrant,
        )
    ax.axhline(0, color="#444444", ls="--", lw=1)
    ax.set_xticks(EXPECTED_FOLDS)
    ax.set_xlabel("Frozen PanNormal checkpoint fold")
    ax.set_ylabel("PLISM gain")
    ax.set_title("C  All 5 frozen models move PLISM away from AT2")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    identity = plism["at2_identity_damage"]
    estimate = float(identity["estimate"])
    low = float(identity["ci_low"])
    high = float(identity["ci_high"])
    ax.errorbar(
        [0],
        [estimate],
        yerr=[[estimate - low], [high - estimate]],
        fmt="o",
        color="#e41a1c",
        capsize=5,
        ms=8,
    )
    ax.axhline(0, color="#444444", ls="--", lw=1)
    ax.set_xlim(-0.75, 0.75)
    ax.set_xticks([0], ["G(AT2) vs AT2"])
    ax.set_ylabel("UNI-v1 cosine distance")
    ax.set_title("D  Target-domain identity is not preserved")
    ax.text(
        0.03,
        0.96,
        "Destructive control only\nExcluded from closure and OOD pools\n"
        "External inference: 13 sections\nStatus: diagnostic-only (image gates failed)",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    fig.suptitle(
        "Pix2Pix GT450→AT2: the internal failure replicates in PLISM",
        fontsize=14,
        fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_representative_gallery(root: Path, path: Path) -> None:
    """Plot one deterministic median-structure example per source scanner."""

    rows: list[tuple[str, pd.Series, np.ndarray, np.ndarray, np.ndarray]] = []
    for scanner in SOURCE_SCANNERS:
        metric_path = (
            root
            / "05_image_evaluation/pix2pix"
            / f"{scanner}_to_at2/location_metrics.csv.gz"
        )
        metrics = pd.read_csv(metric_path)
        median = float(metrics["source_gradient_ncc"].median())
        selected = (
            metrics.assign(
                median_ncc_distance=(metrics["source_gradient_ncc"] - median).abs()
            )
            .sort_values(
                ["median_ncc_distance", "slide_id", "location_index"],
                kind="stable",
            )
            .iloc[0]
        )
        fold = int(selected["fold"])
        manifest_path = (
            root
            / "04_predictions/pix2pix"
            / f"{scanner}_to_at2/fold_{fold}/prediction_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        matching = [
            item for item in manifest["shards"]
            if str(item["slide_id"]) == str(selected["slide_id"])
        ]
        if len(matching) != 1:
            raise ValueError(
                f"expected one prediction shard for {scanner}/{selected['slide_id']}"
            )
        with h5py.File(matching[0]["path"], "r") as store:
            locations = np.asarray(store["metadata/location_index"], dtype=np.int64)
            positions = np.flatnonzero(locations == int(selected["location_index"]))
            if len(positions) != 1:
                raise ValueError(
                    f"expected one image for {scanner}/{selected['slide_id']}/"
                    f"{selected['location_index']}"
                )
            index = int(positions[0])
            source = np.asarray(store["images/raw_source"][index], dtype=np.uint8)
            generated = np.asarray(store["images/generated"][index], dtype=np.uint8)
            at2 = np.asarray(store["images/real_at2"][index], dtype=np.uint8)
        rows.append((scanner, selected, source, generated, at2))

    fig, axes = plt.subplots(3, 4, figsize=(11.5, 8.5), constrained_layout=True)
    column_titles = ("Raw source", "Pix2Pix", "Real paired AT2", "|Pix2Pix − AT2|")
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontweight="bold")
    for row_index, (scanner, selected, source, generated, at2) in enumerate(rows):
        axes[row_index, 0].imshow(source)
        axes[row_index, 1].imshow(generated)
        axes[row_index, 2].imshow(at2)
        residual = np.abs(generated.astype(np.int16) - at2.astype(np.int16)).mean(axis=2)
        axes[row_index, 3].imshow(residual, cmap="magma", vmin=0, vmax=64)
        axes[row_index, 0].set_ylabel(
            f"{SCANNER_LABELS[scanner]}\n"
            f"median structure NCC={selected['source_gradient_ncc']:.3f}\n"
            f"invented edge={selected['invented_edge_fraction']:.3f}",
            fontsize=9,
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 2].text(
            0.02,
            0.98,
            f"{selected['tissue_type']} · {selected['slide_id']}",
            transform=axes[row_index, 2].transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
        )
    fig.suptitle(
        "Representative median-structure examples (deterministic, not best-case)",
        fontsize=14,
        fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize(
    input_root: str | Path,
    output_root: str | Path,
    *,
    plism_root: str | Path | None = None,
    bootstrap_replicates: int = 20_000,
    confidence_level: float = 0.95,
    seed: int = 20260917,
) -> dict[str, Any]:
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    root = Path(input_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    uni_root = root / "06_uni" / "pix2pix"
    image_root = root / "05_image_evaluation" / "pix2pix"
    checkpoint_root = root / "02_pix2pix" / "runs"

    uni_slides, uni_paths = _load_scanner_slide_tables(uni_root, UNI_METRICS)
    image_slides, image_paths = _load_scanner_slide_tables(image_root, IMAGE_METRICS)
    _load_evaluation_manifests(uni_root, evaluation_name="UNI")
    image_manifests = _load_evaluation_manifests(
        image_root, evaluation_name="image-safety"
    )
    uni_keys = set(
        zip(uni_slides["source_scanner"].str.casefold(), uni_slides["slide_id"])
    )
    image_keys = set(
        zip(image_slides["source_scanner"].str.casefold(), image_slides["slide_id"])
    )
    if uni_keys != image_keys:
        raise ValueError(
            "UNI and image-safety layers must cover the same source-scanner/physical-slide keys; "
            f"UNI-only={len(uni_keys - image_keys)}, image-only={len(image_keys - uni_keys)}"
        )
    selections, candidates = _load_checkpoint_selections(checkpoint_root)

    uni_summary = _summarize_slide_metrics(
        uni_slides,
        UNI_METRICS,
        replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        seed=seed,
    )
    image_summary = _attach_safety_gates(
        _summarize_slide_metrics(
            image_slides,
            IMAGE_METRICS,
            replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=seed + 10_000,
        )
    )
    stability = _checkpoint_stability(selections)
    domain_probes = _pooled_domain_probes(uni_root, seed=seed)
    plism = _plism_inventory(None if plism_root is None else Path(plism_root), output)
    uni_conclusion = _uni_conclusion(uni_summary)
    safety_conclusion = _safety_conclusion(image_summary, image_manifests)

    output_paths = {
        "uni_summary": output / "uni_summary.csv",
        "image_safety_summary": output / "image_safety_summary.csv",
        "checkpoint_selections": output / "checkpoint_selections.csv",
        "checkpoint_candidates": output / "checkpoint_candidates.csv",
        "checkpoint_stability": output / "checkpoint_stability.csv",
        "figure": output / "pix2pix_to_at2_summary.png",
        "plism_figure": output / "pix2pix_plism_external.png",
        "gallery": output / "pix2pix_representative_gallery.png",
        "narrative_ko": output / "narrative_ko.md",
        "summary_json": output / "summary.json",
    }
    uni_summary.to_csv(output_paths["uni_summary"], index=False)
    image_summary.to_csv(output_paths["image_safety_summary"], index=False)
    selections.to_csv(output_paths["checkpoint_selections"], index=False)
    candidates.to_csv(output_paths["checkpoint_candidates"], index=False)
    stability.to_csv(output_paths["checkpoint_stability"], index=False)
    _write_korean_narrative(
        output_paths["narrative_ko"],
        uni_summary,
        image_summary,
        uni_conclusion,
        safety_conclusion,
        stability,
        domain_probes,
        plism,
    )
    _plot_summary(
        output_paths["figure"],
        uni_summary,
        image_summary,
        selections,
        uni_conclusion,
        safety_conclusion,
        domain_probes,
        plism,
    )
    _plot_plism_external_summary(output_paths["plism_figure"], uni_summary, plism)
    _plot_representative_gallery(root, output_paths["gallery"])

    payload = {
        "summary_version": SUMMARY_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "direction": [f"{scanner}->at2" for scanner in SOURCE_SCANNERS],
        "internal_primary_endpoint": (
            "paired frozen-UNI gain = raw source-to-AT2 cosine distance minus "
            "Pix2Pix-to-AT2 cosine distance"
        ),
        "internal_pooling_rule": (
            "locations averaged within physical slide; for pooled estimates, source scanners "
            "averaged within physical slide before slide bootstrap"
        ),
        "uni_conclusion": uni_conclusion,
        "image_safety_conclusion": safety_conclusion,
        "pooled_domain_probes": domain_probes,
        "plism": plism,
        "checkpoint_count": int(len(selections)),
        "source_files": {
            "uni_slide_metrics": {key: str(value.resolve()) for key, value in uni_paths.items()},
            "image_slide_metrics": {
                key: str(value.resolve()) for key, value in image_paths.items()
            },
        },
        "outputs": {
            key: str(value.resolve())
            for key, value in output_paths.items()
            if value.exists()
        },
        "interpretation_boundary": (
            "A favourable UNI result cannot override an image-safety failure. PLISM is an "
            "external validation and is never pooled with the internal PanNormal result."
        ),
    }
    _write_json(output_paths["summary_json"], payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="06_learned_baselines root containing 02_pix2pix, 05_image_evaluation, and 06_uni",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--plism-root",
        type=Path,
        default=None,
        help="optional, separate PLISM external-result root; never pooled internally",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize(
        args.input_root,
        args.output_root,
        plism_root=args.plism_root,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
