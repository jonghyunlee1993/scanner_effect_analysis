"""Aggregate frequency perturbation experiments at slide/tissue level."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from final_image_study import fit_nested_reml
from manuscript_completion.contract import (
    ANALYSIS_ROOT,
    OUTPUT_ROOT,
    SCANNERS,
    UNI_V1_CHECKPOINT_SHA256,
    sha256,
    utc_now,
)
from manuscript_completion.perturbation import (
    ANALYSIS_VERSION,
    METRIC_NAMES,
    RMS_BAND_MEDIAN_RATIO_MAX,
    RMS_MEDIAN_RELATIVE_ERROR_MAX,
    RMS_P90_RELATIVE_ERROR_MAX,
    _calibration_path,
    _contract_path,
    expected_condition_names,
    verify_implementation_manifest,
)


BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 20260917
PROBE_C_GRID = (0.01, 0.1, 1.0)


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _load(
    mode: str,
    output_root: Path,
    implementation_manifest_sha256: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    relative = {
        "matched": "04_frequency_mechanism/02_matched_information_removal/shards",
        "sensitivity": "04_frequency_mechanism/03_uni_band_sensitivity/shards",
    }[mode]
    root = output_root / relative
    summaries = sorted(root.glob("*.summary.json"))
    cohort = pd.read_csv(
        ANALYSIS_ROOT / "00_contract/cohort.csv", dtype={"slide_id": str}
    )
    if len(cohort) != 103 or cohort["slide_id"].duplicated().any():
        raise ValueError("locked cohort must contain 103 unique physical slides")
    cohort = cohort.set_index("slide_id")
    summary_slides = {path.name.removesuffix(".summary.json") for path in summaries}
    if summary_slides != set(cohort.index):
        raise ValueError(
            f"{mode}: summary slide set mismatch; "
            f"missing={sorted(set(cohort.index)-summary_slides)}, "
            f"extra={sorted(summary_slides-set(cohort.index))}"
        )
    contract_path = _contract_path(output_root, mode)
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    contract = json.loads(contract_path.read_text())
    contract_hash = sha256(contract_path)
    if (
        contract.get("status") != "frozen_before_uni_outcomes"
        or contract.get("analysis_version") != ANALYSIS_VERSION
    ):
        raise ValueError(f"{mode}: invalid v2 contract")
    expected_conditions = expected_condition_names(mode)
    if contract.get("expected_conditions") != expected_conditions:
        raise ValueError(f"{mode}: contract condition schema mismatch")
    dose_calibration_hash = None
    if mode == "sensitivity":
        calibration_path = _calibration_path(output_root)
        calibration = json.loads(calibration_path.read_text())
        dose_calibration_hash = sha256(calibration_path)
        if (
            calibration.get("status") != "frozen_before_uni_outcomes"
            or calibration.get("contract_sha256") != contract_hash
        ):
            raise ValueError("sensitivity dose calibration is stale or not frozen")
    metadata = []
    features = []
    metrics = []
    seen_slides: set[str] = set()
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())
        filename_slide = summary_path.name.removesuffix(".summary.json")
        if (
            summary.get("status") != "pass"
            or summary.get("analysis_version") != ANALYSIS_VERSION
            or summary.get("mode") != mode
            or str(summary.get("slide_id")) != filename_slide
            or summary.get("contract_sha256") != contract_hash
            or summary.get("dose_calibration_sha256") != dose_calibration_hash
            or summary.get("implementation_manifest_sha256")
            != implementation_manifest_sha256
            or summary.get("uni_v1_checkpoint_sha256") != UNI_V1_CHECKPOINT_SHA256
        ):
            raise ValueError(f"non-pass shard: {summary_path}")
        shard = summary_path.with_name(summary_path.name.replace(".summary.json", ".h5"))
        if not shard.is_file() or summary.get("output_sha256") != sha256(shard):
            raise ValueError(f"hash mismatch: {shard}")
        with h5py.File(shard, "r") as store:
            required = {
                "features",
                "image_metrics",
                "condition_names",
                "metric_names",
                "scanner_names",
                "location_index",
                "source_index",
                "replicate_id",
            }
            if not required.issubset(store.keys()):
                raise ValueError(f"incomplete H5 schema: {shard}")
            condition_names = _decode(store["condition_names"][:])
            scanner_names = _decode(store["scanner_names"][:])
            metric_names = _decode(store["metric_names"][:])
            if scanner_names != list(SCANNERS) or metric_names != list(METRIC_NAMES):
                raise ValueError(f"schema mismatch: {shard}")
            if condition_names != expected_conditions:
                raise ValueError(f"condition mismatch: {shard}")
            block_features = np.asarray(store["features"], dtype=np.float32)
            block_metrics = np.asarray(store["image_metrics"], dtype=np.float32)
            locations = np.asarray(store["location_index"], dtype=int)
            source_indices = np.asarray(store["source_index"], dtype=int)
            replicates = np.asarray(store["replicate_id"], dtype=int)
            slide_id = str(store.attrs["slide_id"])
            tissue_type = str(store.attrs["tissue_type"])
            fold = int(store.attrs["fold"])
            attributes = {
                "analysis_version": str(store.attrs.get("analysis_version", "")),
                "mode": str(store.attrs.get("mode", "")),
                "contract_sha256": str(store.attrs.get("contract_sha256", "")),
                "dose_calibration_sha256": str(
                    store.attrs.get("dose_calibration_sha256", "")
                ),
                "implementation_manifest_sha256": str(
                    store.attrs.get("implementation_manifest_sha256", "")
                ),
                "uni_v1_checkpoint_sha256": str(
                    store.attrs.get("uni_v1_checkpoint_sha256", "")
                ),
            }
        expected_feature_shape = (20, 6, len(condition_names), 1024)
        expected_metric_shape = (20, 6, len(condition_names), len(METRIC_NAMES))
        if block_features.shape != expected_feature_shape:
            raise ValueError(f"unexpected feature shape: {shard}: {block_features.shape}")
        if block_metrics.shape != expected_metric_shape:
            raise ValueError(f"unexpected image-metric shape: {shard}: {block_metrics.shape}")
        if (
            locations.shape != (20,)
            or source_indices.shape != (20,)
            or replicates.shape != (20,)
            or len(np.unique(locations)) != 20
            or len(np.unique(source_indices)) != 20
        ):
            raise ValueError(f"location/source-index schema mismatch: {slide_id}")
        if not np.array_equal(np.bincount(replicates, minlength=10), np.full(10, 2)):
            raise ValueError(f"replicate coverage mismatch: {slide_id}")
        if not np.isfinite(block_features).all() or not np.isfinite(block_metrics).all():
            raise ValueError(f"non-finite shard values: {slide_id}")
        maximum_norm_error = float(
            np.max(np.abs(np.linalg.norm(block_features, axis=-1) - 1.0))
        )
        if maximum_norm_error > 5e-4:
            raise ValueError(f"non-unit UNI embeddings: {slide_id}: {maximum_norm_error}")
        cohort_row = cohort.loc[slide_id]
        if (
            slide_id != filename_slide
            or slide_id in seen_slides
            or tissue_type != str(cohort_row.tissue_type)
            or fold != int(cohort_row.fold)
            or attributes["analysis_version"] != ANALYSIS_VERSION
            or attributes["mode"] != mode
            or attributes["contract_sha256"] != contract_hash
            or attributes["dose_calibration_sha256"] != (dose_calibration_hash or "")
            or attributes["implementation_manifest_sha256"]
            != implementation_manifest_sha256
            or attributes["uni_v1_checkpoint_sha256"] != UNI_V1_CHECKPOINT_SHA256
        ):
            raise ValueError(f"cohort/attribute mismatch: {slide_id}")
        seen_slides.add(slide_id)
        for location_offset, (location, replicate) in enumerate(zip(locations, replicates)):
            for scanner_index, scanner in enumerate(SCANNERS):
                metadata.append(
                    {
                        "slide_id": slide_id,
                        "tissue_type": tissue_type,
                        "fold": fold,
                        "location_index": int(location),
                        "replicate": int(replicate),
                        "scanner": scanner,
                        "location_offset": location_offset,
                        "scanner_index": scanner_index,
                    }
                )
        features.append(block_features.reshape(-1, len(condition_names), block_features.shape[-1]))
        metrics.append(block_metrics.reshape(-1, len(condition_names), block_metrics.shape[-1]))
    return (
        pd.DataFrame(metadata),
        np.concatenate(features),
        np.concatenate(metrics),
        expected_conditions,
    )


def _bootstrap_mean(values: pd.DataFrame, value: str, seed: int) -> tuple[float, float, float]:
    slide = values.groupby("slide_id", sort=False)[value].mean().to_numpy(float)
    if not len(slide):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = slide[rng.integers(0, len(slide), size=(BOOTSTRAP_REPLICATES, len(slide)))].mean(axis=1)
    return float(slide.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _paired_bootstrap(
    frame: pd.DataFrame,
    condition: str,
    value: str,
    seed: int,
) -> tuple[float, float, float]:
    table = frame.pivot_table(index="slide_id", columns="condition", values=value, aggfunc="mean")
    delta = (table[condition] - table["raw"]).dropna().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta)))].mean(axis=1)
    return float(delta.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _balanced_accuracy_inference(
    probes: pd.DataFrame,
    probe: str,
    condition: str,
    seed: int,
) -> dict[str, Any]:
    """Return class-balanced OOF accuracy with a physical-slide cluster CI.

    When every class is observed on every slide (scanner probe), the same slide
    bootstrap weights are shared across classes.  For tissue labels, slides are
    nested within class, so slides are resampled within each observed tissue and
    tissue recalls are then averaged with equal weight.
    """
    selected = probes[(probes.probe == probe) & (probes.condition == condition)].copy()
    if selected.empty:
        raise ValueError(f"no probe rows for {probe}/{condition}")
    per_slide = (
        selected.groupby(["truth", "slide_id"], as_index=False)["correct"].mean()
    )
    class_names = sorted(per_slide["truth"].unique())
    class_values = {
        name: per_slide.loc[per_slide.truth == name, ["slide_id", "correct"]]
        .set_index("slide_id")["correct"]
        .sort_index()
        for name in class_names
    }
    estimate = float(np.mean([value.mean() for value in class_values.values()]))
    sklearn_estimate = float(
        balanced_accuracy_score(selected["truth"], selected["predicted"])
    )
    if not np.isclose(estimate, sklearn_estimate, atol=1e-12):
        raise AssertionError("balanced-accuracy sufficient statistic mismatch")

    raw = probes[(probes.probe == probe) & (probes.condition == "raw")]
    raw_per_slide = (
        raw.groupby(["truth", "slide_id"], as_index=False)["correct"].mean()
    )
    paired_values: dict[str, pd.DataFrame] = {}
    for name in class_names:
        current = class_values[name].rename("current")
        anchor = (
            raw_per_slide.loc[raw_per_slide.truth == name, ["slide_id", "correct"]]
            .set_index("slide_id")["correct"]
            .rename("raw")
        )
        paired = pd.concat([current, anchor], axis=1, join="inner").dropna().sort_index()
        if len(paired) != len(current):
            raise ValueError(f"unpaired raw probe predictions for {probe}/{condition}/{name}")
        paired_values[name] = paired

    rng = np.random.default_rng(seed)
    slide_sets = [tuple(value.index) for value in class_values.values()]
    shared_slides = bool(slide_sets) and all(value == slide_sets[0] for value in slide_sets)
    if shared_slides:
        slide_count = len(slide_sets[0])
        indices = rng.integers(0, slide_count, size=(BOOTSTRAP_REPLICATES, slide_count))
        current_draws = []
        delta_draws = []
        for name in class_names:
            vector = class_values[name].to_numpy(float)
            paired = paired_values[name]
            current_draws.append(vector[indices].mean(axis=1))
            delta_draws.append(
                (paired["current"].to_numpy() - paired["raw"].to_numpy())[indices].mean(axis=1)
            )
    else:
        current_draws = []
        delta_draws = []
        for name in class_names:
            vector = class_values[name].to_numpy(float)
            paired = paired_values[name]
            indices = rng.integers(0, len(vector), size=(BOOTSTRAP_REPLICATES, len(vector)))
            current_draws.append(vector[indices].mean(axis=1))
            delta = paired["current"].to_numpy() - paired["raw"].to_numpy()
            delta_draws.append(delta[indices].mean(axis=1))
    bootstrap = np.mean(current_draws, axis=0)
    delta_bootstrap = np.mean(delta_draws, axis=0)
    delta_point = float(
        np.mean(
            [
                (value["current"] - value["raw"]).mean()
                for value in paired_values.values()
            ]
        )
    )
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "delta_from_raw": 0.0 if condition == "raw" else delta_point,
        "delta_ci_low": 0.0
        if condition == "raw"
        else float(np.quantile(delta_bootstrap, 0.025)),
        "delta_ci_high": 0.0
        if condition == "raw"
        else float(np.quantile(delta_bootstrap, 0.975)),
        "evaluated_rows": len(selected),
        "evaluated_slides": int(selected.slide_id.nunique()),
        "evaluated_classes": len(class_names),
        "estimand": "macro_recall_over_oof_seen_classes",
        "bootstrap": (
            "joint_physical_slide_cluster"
            if shared_slides
            else "physical_slides_resampled_within_class_then_macro_averaged"
        ),
    }


def _select_c(x: np.ndarray, y: np.ndarray, folds: np.ndarray, outer: int) -> float:
    available = sorted(set(folds.tolist()) - {outer})
    scores = []
    for c in PROBE_C_GRID:
        fold_scores = []
        for inner in available:
            train = (folds != outer) & (folds != inner)
            test = (folds == inner) & np.isin(y, np.unique(y[train]))
            if len(np.unique(y[train])) < 2 or not test.any():
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c, max_iter=3000, class_weight="balanced"),
            )
            model.fit(x[train], y[train])
            fold_scores.append(balanced_accuracy_score(y[test], model.predict(x[test])))
        scores.append((float(np.mean(fold_scores)) if fold_scores else -np.inf, c))
    return max(scores)[1]


def _probe_predictions(
    centroids: pd.DataFrame,
    matrix: np.ndarray,
    label: str,
    condition: str,
) -> pd.DataFrame:
    y = centroids[label].astype(str).to_numpy()
    folds = centroids["fold"].to_numpy(int)
    rows = []
    for outer in sorted(np.unique(folds)):
        train = folds != outer
        test = (folds == outer) & np.isin(y, np.unique(y[train]))
        c = _select_c(matrix, y, folds, int(outer))
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=3000, class_weight="balanced"),
        )
        model.fit(matrix[train], y[train])
        predicted = model.predict(matrix[test])
        for index, guess in zip(np.flatnonzero(test), predicted):
            item = centroids.iloc[index]
            rows.append(
                {
                    "condition": condition,
                    "probe": label,
                    "slide_id": item.slide_id,
                    "scanner": item.scanner,
                    "fold": int(outer),
                    "truth": y[index],
                    "predicted": str(guess),
                    "correct": int(y[index] == guess),
                    "selected_c": c,
                }
            )
    return pd.DataFrame(rows)


def _effective_rank(matrix: np.ndarray) -> tuple[float, float]:
    centered = matrix.astype(np.float64) - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    eigenvalues = np.square(singular) / max(len(centered) - 1, 1)
    trace = float(eigenvalues.sum())
    probability = eigenvalues / max(trace, 1e-15)
    probability = probability[probability > 0]
    effective = float(np.exp(-(probability * np.log(probability)).sum()))
    return trace, effective


def aggregate_matched(output_root: Path) -> dict[str, Any]:
    implementation = verify_implementation_manifest()
    metadata, features, metrics, conditions = _load(
        "matched", output_root, str(implementation["sha256"])
    )
    contract_path = _contract_path(output_root, "matched")
    metric_index = {name: index for index, name in enumerate(METRIC_NAMES)}
    rows = []
    for condition_index, condition in enumerate(conditions):
        condition_features = features[:, condition_index]
        condition_metrics = metrics[:, condition_index]
        for (slide_id, location_index), indexes in metadata.groupby(
            ["slide_id", "location_index"], sort=False
        ).indices.items():
            ordered = metadata.iloc[indexes].sort_values("scanner_index")
            positions = ordered.index.to_numpy()
            at2_feature = condition_features[positions[0]]
            for scanner_offset, scanner in enumerate(SCANNERS[1:], start=1):
                target_position = positions[scanner_offset]
                rows.append(
                    {
                        "slide_id": slide_id,
                        "tissue_type": ordered.iloc[0].tissue_type,
                        "fold": int(ordered.iloc[0].fold),
                        "location_index": int(location_index),
                        "replicate": int(ordered.iloc[0].replicate),
                        "scanner": scanner,
                        "condition": condition,
                        "scanner_cosine_distance": float(
                            1.0 - np.dot(at2_feature, condition_features[target_position])
                        ),
                        "at2_perturbation_rms_od": float(
                            condition_metrics[positions[0], metric_index["perturbation_rms_od"]]
                        ),
                        "target_perturbation_rms_od": float(
                            condition_metrics[target_position, metric_index["perturbation_rms_od"]]
                        ),
                        "at2_gradient_ncc": float(
                            condition_metrics[positions[0], metric_index["gradient_ncc_to_raw"]]
                        ),
                        "target_gradient_ncc": float(
                            condition_metrics[target_position, metric_index["gradient_ncc_to_raw"]]
                        ),
                    }
                )
    location = pd.DataFrame(rows)
    base = output_root / "04_frequency_mechanism/02_matched_information_removal"
    write_frame(location, base / "matched_filter_location_metrics.csv.gz")
    slide = (
        location.groupby(["slide_id", "tissue_type", "fold", "scanner", "condition"], as_index=False)
        .mean(numeric_only=True)
    )
    write_frame(slide, base / "matched_filter_slide_summary.csv")

    # A direct group index avoids relying on tuple comparison semantics in numpy.
    group_indices = metadata.groupby(["slide_id", "scanner"], sort=False).indices
    probe_frames = []
    retention_rows = []
    for condition_index, condition in enumerate(conditions):
        matrix = np.stack(
            [features[indexes, condition_index].mean(axis=0) for indexes in group_indices.values()]
        )
        ordered_meta = pd.DataFrame(
            [
                {
                    "slide_id": key[0],
                    "scanner": key[1],
                    "tissue_type": metadata.iloc[indexes[0]].tissue_type,
                    "fold": int(metadata.iloc[indexes[0]].fold),
                }
                for key, indexes in group_indices.items()
            ]
        )
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        trace, effective = _effective_rank(matrix)
        retention_rows.append(
            {
                "condition": condition,
                "metric": "embedding_covariance_trace",
                "estimate": trace,
            }
        )
        retention_rows.append(
            {
                "condition": condition,
                "metric": "embedding_effective_rank",
                "estimate": effective,
            }
        )
        for label in ("scanner", "tissue_type"):
            probe_frames.append(_probe_predictions(ordered_meta, matrix, label, condition))
    probes = pd.concat(probe_frames, ignore_index=True)
    write_frame(probes, base / "information_probe_predictions.csv.gz")
    for condition in conditions:
        for probe in ("scanner", "tissue_type"):
            inference = _balanced_accuracy_inference(
                probes,
                probe,
                condition,
                _stable_seed(BOOTSTRAP_SEED, "probe", probe, condition),
            )
            retention_rows.append(
                {
                    "condition": condition,
                    "metric": f"{probe}_balanced_accuracy",
                    **inference,
                }
            )
    distance_pooled = slide.groupby(["slide_id", "condition"], as_index=False)[
        "scanner_cosine_distance"
    ].mean()
    for condition in conditions:
        subset = distance_pooled[distance_pooled.condition == condition]
        estimate, low, high = _bootstrap_mean(
            subset, "scanner_cosine_distance", BOOTSTRAP_SEED + conditions.index(condition)
        )
        delta, delta_low, delta_high = (0.0, 0.0, 0.0) if condition == "raw" else _paired_bootstrap(
            distance_pooled,
            condition,
            "scanner_cosine_distance",
            BOOTSTRAP_SEED + 100 + conditions.index(condition),
        )
        retention_rows.append(
            {
                "condition": condition,
                "metric": "pooled_scanner_cosine_distance",
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "delta_from_raw": delta,
                "delta_ci_low": delta_low,
                "delta_ci_high": delta_high,
            }
        )
    retention = pd.DataFrame(retention_rows)
    write_frame(retention, base / "information_retention_summary.csv")

    gap = retention[retention.metric == "pooled_scanner_cosine_distance"].copy()
    reduced = gap[(gap.condition != "raw") & (gap.delta_ci_high < 0)]
    tissue = retention[retention.metric == "tissue_type_balanced_accuracy"].set_index("condition")
    raw_tissue = float(tissue.loc["raw", "estimate"])
    decisions = []
    for row in reduced.itertuples(index=False):
        tissue_change = float(tissue.loc[row.condition, "estimate"] - raw_tissue)
        decisions.append((row.condition, tissue_change))
    if not len(reduced):
        conclusion = "No tested matched information-removal condition reduced the pooled scanner gap with a slide-bootstrap CI below zero."
    elif any(change < 0 for _, change in decisions):
        conclusion = "At least one scanner-gap reduction coincided with lower held-out tissue discriminability, supporting apparent alignment through information loss for that condition."
    else:
        conclusion = "Scanner-gap reduction was observed without a lower point estimate of tissue discriminability; covariance, effective rank, fidelity, and scanner-specific results remain part of the claim gate."
    decision_text = (
        "# Matched information-removal decision\n\n"
        + conclusion
        + "\n\nThis decision is restricted to frozen UNI-v1 and the declared symmetric filters. "
        "It does not reuse the source-only destructive-low-pass result.\n"
    )
    (base / "apparent_alignment_decision.md").write_text(decision_text)
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "pass",
        "completed_utc": utc_now(),
        "slides": int(metadata.slide_id.nunique()),
        "conditions": conditions,
        "locations": int(metadata[["slide_id", "location_index"]].drop_duplicates().shape[0]),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "contract": str(contract_path.resolve()),
        "contract_sha256": sha256(contract_path),
        "implementation_manifest": str(implementation["path"]),
        "implementation_manifest_sha256": str(implementation["sha256"]),
        "probe_estimand": "macro recall over out-of-fold seen classes",
        "outputs": {
            path.name: sha256(path)
            for path in (
                base / "matched_filter_location_metrics.csv.gz",
                base / "matched_filter_slide_summary.csv",
                base / "information_probe_predictions.csv.gz",
                base / "information_retention_summary.csv",
                base / "apparent_alignment_decision.md",
            )
        },
    }
    write_json(manifest, base / "manifest.json")
    return manifest


def _condition_fields(condition: str) -> dict[str, Any]:
    if condition == "raw":
        return {"intervention": "raw", "band": "raw", "dose_fraction": 0.0}
    parts = condition.split("_")
    dose_fraction = int(parts[-1][1:]) / 100.0
    if parts[0] == "phase":
        band = "_".join(parts[1:-1])
        intervention = "phase"
    else:
        intervention = f"amplitude_{parts[1]}"
        band = "_".join(parts[2:-1])
    return {"intervention": intervention, "band": band, "dose_fraction": dose_fraction}


def _achieved_rms_equivalence(analysis: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    rows: list[dict[str, Any]] = []
    amplitude = analysis[analysis.intervention.str.startswith("amplitude_")].copy()
    amplitude["relative_error"] = np.abs(
        amplitude["achieved_rms_od"] - amplitude["target_rms_od"]
    ) / np.maximum(amplitude["target_rms_od"], 1e-12)
    for scanner_scope in ("pooled", *SCANNERS):
        selected = amplitude if scanner_scope == "pooled" else amplitude[amplitude.scanner == scanner_scope]
        for (intervention, dose, band), group in selected.groupby(
            ["intervention", "dose_fraction", "band"], sort=False
        ):
            median_error = float(group["relative_error"].median())
            p90_error = float(group["relative_error"].quantile(0.90))
            pooled_gate = scanner_scope == "pooled"
            rows.append(
                {
                    "record_type": "target_attainment",
                    "intervention": intervention,
                    "dose_fraction": dose,
                    "band": band,
                    "scanner": scanner_scope,
                    "n_images": len(group),
                    "target_rms_od": float(group["target_rms_od"].median()),
                    "achieved_rms_od_median": float(group["achieved_rms_od"].median()),
                    "relative_error_median": median_error,
                    "relative_error_p90": p90_error,
                    "band_median_max_to_min_ratio": np.nan,
                    "used_in_primary_gate": pooled_gate,
                    "pass": bool(
                        median_error <= RMS_MEDIAN_RELATIVE_ERROR_MAX
                        and p90_error <= RMS_P90_RELATIVE_ERROR_MAX
                    ),
                }
            )
        for (intervention, dose), group in selected.groupby(
            ["intervention", "dose_fraction"], sort=False
        ):
            medians = group.groupby("band")["achieved_rms_od"].median()
            ratio = float(medians.max() / max(medians.min(), 1e-12))
            pooled_gate = scanner_scope == "pooled"
            rows.append(
                {
                    "record_type": "between_band_equivalence",
                    "intervention": intervention,
                    "dose_fraction": dose,
                    "band": "all",
                    "scanner": scanner_scope,
                    "n_images": len(group),
                    "target_rms_od": float(group["target_rms_od"].median()),
                    "achieved_rms_od_median": float(group["achieved_rms_od"].median()),
                    "relative_error_median": np.nan,
                    "relative_error_p90": np.nan,
                    "band_median_max_to_min_ratio": ratio,
                    "used_in_primary_gate": pooled_gate,
                    "pass": bool(ratio <= RMS_BAND_MEDIAN_RATIO_MAX),
                }
            )
    result = pd.DataFrame(rows)
    gate = result[result.used_in_primary_gate]
    return result, bool(len(gate) and gate["pass"].all())


def aggregate_sensitivity(output_root: Path) -> dict[str, Any]:
    implementation = verify_implementation_manifest()
    shard_implementation_sha256 = str(
        implementation["payload"].get(
            "source_shard_implementation_manifest_sha256", implementation["sha256"]
        )
    )
    metadata, features, metrics, conditions = _load(
        "sensitivity", output_root, shard_implementation_sha256
    )
    metric_index = {name: index for index, name in enumerate(METRIC_NAMES)}
    raw_index = conditions.index("raw")
    calibration_path = _calibration_path(output_root)
    calibration = json.loads(calibration_path.read_text())
    contract_path = _contract_path(output_root, "sensitivity")
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != "frozen_before_uni_outcomes":
        raise ValueError("band perturbation contract was not frozen pre-outcome")
    minimum_coverage = contract["gradient_stratum_minimum_coverage"]
    minimum_slides = int(minimum_coverage["physical_slides"])
    minimum_tissues = int(minimum_coverage["tissues"])
    rows = []
    for condition_index, condition in enumerate(conditions):
        fields = _condition_fields(condition)
        displacement = 1.0 - np.sum(
            features[:, raw_index] * features[:, condition_index], axis=1
        )
        for index, item in metadata.iterrows():
            target_rms = (
                0.0
                if condition == "raw"
                else float(calibration["folds"][str(int(item.fold))]["base_rms_od"])
                * float(fields["dose_fraction"])
            )
            rows.append(
                {
                    "slide_id": item.slide_id,
                    "tissue_type": item.tissue_type,
                    "fold": int(item.fold),
                    "location_index": int(item.location_index),
                    "replicate": int(item.replicate),
                    "scanner": item.scanner,
                    "condition": condition,
                    **fields,
                    "uni_cosine_displacement": float(displacement[index]),
                    "achieved_rms_od": float(
                        metrics[index, condition_index, metric_index["perturbation_rms_od"]]
                    ),
                    "target_rms_od": target_rms,
                    "gradient_energy": float(
                        metrics[index, raw_index, metric_index["gradient_energy"]]
                    ),
                    "baseline_band_power": float(
                        metrics[
                            index,
                            raw_index,
                            metric_index.get(f"band_power_{fields['band']}", metric_index["band_power_high"]),
                        ]
                    ),
                }
            )
    location = pd.DataFrame(rows)
    base = output_root / "04_frequency_mechanism/03_uni_band_sensitivity"
    write_frame(location, base / "uni_band_dose_response.csv")

    raw_context = location[location.condition == "raw"][
        ["slide_id", "scanner", "location_index", "gradient_energy"]
    ].copy()
    raw_context["gradient_quartile"] = raw_context.groupby("scanner", group_keys=False)[
        "gradient_energy"
    ].transform(lambda value: pd.qcut(value, 4, labels=False, duplicates="raise"))
    raw_context["gradient_quartile"] = raw_context["gradient_quartile"].astype(int)
    analysis = location[location.condition != "raw"].merge(
        raw_context[["slide_id", "scanner", "location_index", "gradient_quartile"]],
        on=["slide_id", "scanner", "location_index"],
        validate="many_to_one",
    )
    rms_summary, rms_gate_pass = _achieved_rms_equivalence(analysis)
    write_frame(rms_summary, base / "achieved_rms_equivalence.csv")

    model_rows = []
    for (intervention, dose), group in analysis.groupby(["intervention", "dose_fraction"]):
        table = group.pivot_table(
            index=[
                "slide_id",
                "tissue_type",
                "replicate",
                "scanner",
                "location_index",
                "gradient_quartile",
            ],
            columns="band",
            values="uni_cosine_displacement",
            aggfunc="mean",
        ).dropna().reset_index()
        for comparator in ("low_mid", "mid"):
            table["value"] = table["high"] - table[comparator]
            for scanner_scope in ("pooled", *SCANNERS):
                scanner_selected = (
                    table if scanner_scope == "pooled" else table[table.scanner == scanner_scope]
                )
                for gradient_scope in ("pooled", "Q1", "Q2", "Q3", "Q4"):
                    gradient_selected = (
                        scanner_selected
                        if gradient_scope == "pooled"
                        else scanner_selected[
                            scanner_selected.gradient_quartile == int(gradient_scope[1:]) - 1
                        ]
                    )
                    contrast = (
                        gradient_selected.groupby(
                            ["slide_id", "tissue_type", "replicate"], as_index=False
                        )["value"].mean()
                    )
                    slides = int(contrast.slide_id.nunique())
                    tissues = int(contrast.tissue_type.nunique())
                    coverage_pass = bool(
                        slides >= minimum_slides and tissues >= minimum_tissues
                    )
                    if not coverage_pass:
                        model_rows.append(
                            {
                                "intervention": intervention,
                                "dose_fraction": dose,
                                "scanner": scanner_scope,
                                "gradient_quartile": gradient_scope,
                                "contrast": f"high_minus_{comparator}",
                                "estimate": np.nan,
                                "se": np.nan,
                                "ci_low": np.nan,
                                "ci_high": np.nan,
                                "tissue_variance": np.nan,
                                "slide_variance": np.nan,
                                "residual_variance": np.nan,
                                "tissues": tissues,
                                "slides": slides,
                                "replicate_rows": len(contrast),
                                "coverage_pass": False,
                                "fit_converged": False,
                                "analysis_status": "insufficient_coverage",
                            }
                        )
                        continue
                    fit = fit_nested_reml(contrast, "value")
                    estimate = float(fit["full"]["fixed_mean"])
                    se = float(fit["full"]["fixed_se"])
                    critical = float(stats.t.ppf(0.975, max(int(fit["n_tissues"]) - 1, 1)))
                    model_rows.append(
                        {
                            "intervention": intervention,
                            "dose_fraction": dose,
                            "scanner": scanner_scope,
                            "gradient_quartile": gradient_scope,
                            "contrast": f"high_minus_{comparator}",
                            "estimate": estimate,
                            "se": se,
                            "ci_low": estimate - critical * se,
                            "ci_high": estimate + critical * se,
                            "tissue_variance": float(fit["variances"][0]),
                            "slide_variance": float(fit["variances"][1]),
                            "residual_variance": float(fit["variances"][2]),
                            "tissues": int(fit["n_tissues"]),
                            "slides": slides,
                            "replicate_rows": len(contrast),
                            "coverage_pass": True,
                            "fit_converged": bool(fit["fit"].success),
                            "analysis_status": "fit",
                        }
                    )
    mixed = pd.DataFrame(model_rows)
    write_frame(mixed, base / "uni_band_mixed_model.csv")

    amplitude = mixed[
        mixed.intervention.isin(["amplitude_attenuation", "amplitude_gain"])
        & mixed.scanner.eq("pooled")
        & mixed.gradient_quartile.eq("pooled")
        & mixed.coverage_pass
    ]
    primary_model_pass = bool(len(amplitude) and (amplitude.ci_low > 0).all())
    primary_pass = bool(primary_model_pass and rms_gate_pass)
    scanner_support = mixed[
        mixed.intervention.isin(["amplitude_attenuation", "amplitude_gain"])
        & ~mixed.scanner.eq("pooled")
        & mixed.gradient_quartile.eq("pooled")
        & mixed.contrast.eq("high_minus_low_mid")
        & mixed.coverage_pass
    ]
    supported_scanners = int(
        scanner_support.groupby("scanner")["estimate"].apply(lambda value: bool((value > 0).all())).sum()
    )
    if not rms_gate_pass:
        conclusion = "The achieved OD-RMS equivalence gate failed, so no preferential-frequency claim is made regardless of the UNI mixed-model estimates."
    elif primary_pass and supported_scanners >= 5:
        conclusion = "At equal within-image OD-RMS, high-band amplitude perturbations produced larger frozen UNI-v1 displacement than low/mid perturbations in the pooled nested model and at least five scanners."
    else:
        conclusion = "The predeclared evidence gate for preferential high-frequency sensitivity was not met; band-, scanner-, and phase-specific estimates are reported without a general PFM claim."
    decision = (
        "# UNI frequency-sensitivity decision\n\n"
        + conclusion
        + "\n\nThe conclusion is encoder-specific to frozen UNI-v1. Phase results are interpreted as local-structure sensitivity, not amplitude sensitivity.\n"
    )
    (base / "uni_frequency_sensitivity_decision.md").write_text(decision)
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "pass",
        "completed_utc": utc_now(),
        "slides": int(metadata.slide_id.nunique()),
        "conditions": conditions,
        "primary_model_pass": primary_model_pass,
        "achieved_rms_gate_pass": rms_gate_pass,
        "primary_pass": primary_pass,
        "supported_scanners": supported_scanners,
        "implementation_manifest": str(implementation["path"]),
        "implementation_manifest_sha256": str(implementation["sha256"]),
        "outputs": {
            path.name: sha256(path)
            for path in (
                contract_path,
                calibration_path,
                base / "uni_band_dose_response.csv",
                base / "achieved_rms_equivalence.csv",
                base / "uni_band_mixed_model.csv",
                base / "uni_frequency_sensitivity_decision.md",
            )
        },
    }
    write_json(manifest, base / "manifest.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("mode", choices=("matched", "sensitivity"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = (
        aggregate_matched(args.output_root)
        if args.mode == "matched"
        else aggregate_sensitivity(args.output_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
