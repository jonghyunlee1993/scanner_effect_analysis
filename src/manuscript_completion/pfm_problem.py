#!/usr/bin/env python3
"""Define the frozen-UNI scanner problem on the locked 103-slide cohort.

This module only reads the existing raw UNI embeddings.  It audits every
shard, quantifies same-location AT2-to-scanner distances, evaluates a
six-class scanner probe with physical-slide held-out folds, and places the
scanner distances on a slide-balanced biological-distance scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCANNERS = ("at2", "versa", "akoya", "gt450", "s360", "s60")
TARGET_SCANNERS = SCANNERS[1:]
RAW_CONDITIONS = ("source",) + tuple(f"target:{name}" for name in TARGET_SCANNERS)
DEFAULT_INPUT_ROOT = Path("outputs/scanner_batch_effect_analysis_2026-09-17/03_uni/shards")
DEFAULT_COHORT = Path("outputs/scanner_batch_effect_analysis_2026-09-17/00_contract/cohort.csv")
DEFAULT_FOLDS = Path("outputs/augmentation_ood_v1/00_contract/slide_folds.csv")
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/scanner_batch_effect_analysis_2026-09-17/"
    "12_manuscript_completion/01_pfm_problem_definition"
)
DEFAULT_ALPHAS = (1e-5, 1e-4, 1e-3)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_records(records: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(records):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
    return float(low), float(high)


def multinomial_slide_weights(
    slide_count: int, bootstrap_count: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(
        slide_count,
        np.full(slide_count, 1.0 / slide_count),
        size=bootstrap_count,
    ).astype(np.float64)


def bootstrap_mean(
    values: Sequence[float], bootstrap_count: int, seed: int
) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all() or len(vector) < 2:
        raise ValueError("bootstrap values must be a finite one-dimensional sample")
    weights = multinomial_slide_weights(len(vector), bootstrap_count, seed)
    estimates = weights @ vector / len(vector)
    low, high = percentile_interval(estimates)
    return float(vector.mean()), low, high


def _decode(values: np.ndarray) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values]


def load_contract(cohort_path: Path, fold_path: Path) -> pd.DataFrame:
    cohort = pd.read_csv(cohort_path, dtype={"slide_id": str})
    folds = pd.read_csv(fold_path, dtype={"slide_id": str})
    required = {"slide_id", "tissue_type"}
    if not required.issubset(cohort.columns):
        raise ValueError(f"cohort lacks columns: {sorted(required - set(cohort.columns))}")
    if set(folds.columns) != {"slide_id", "fold"}:
        raise ValueError("fold manifest must contain exactly slide_id and fold")
    if cohort["slide_id"].duplicated().any() or folds["slide_id"].duplicated().any():
        raise ValueError("slide IDs must be unique in cohort and fold manifests")
    if set(cohort["slide_id"]) != set(folds["slide_id"]):
        raise ValueError("cohort and fold slide populations differ")
    if "fold" in cohort.columns:
        check = cohort[["slide_id", "fold"]].merge(
            folds, on="slide_id", suffixes=("_cohort", "_locked"), validate="one_to_one"
        )
        if not np.array_equal(check["fold_cohort"], check["fold_locked"]):
            raise ValueError("cohort folds disagree with locked slide folds")
        cohort = cohort.drop(columns="fold")
    cohort = cohort.merge(folds, on="slide_id", validate="one_to_one")
    if sorted(cohort["fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("exactly the locked folds 0--4 are required")
    return cohort.sort_values("slide_id", kind="stable").reset_index(drop=True)


def audit_and_load_raw_embeddings(
    cohort: pd.DataFrame, shard_root: Path
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Audit all shards and return [slide, location, scanner, feature] embeddings."""
    expected_slides = cohort["slide_id"].tolist()
    present_h5 = {path.stem for path in shard_root.glob("*.h5")}
    present_summaries = {
        path.name.removesuffix(".summary.json") for path in shard_root.glob("*.summary.json")
    }
    if present_h5 != set(expected_slides):
        raise ValueError(
            f"H5 slide set mismatch; missing={sorted(set(expected_slides)-present_h5)}, "
            f"extra={sorted(present_h5-set(expected_slides))}"
        )
    if present_summaries != set(expected_slides):
        raise ValueError("summary JSON slide set does not match cohort")

    embeddings: list[np.ndarray] = []
    locations: list[np.ndarray] = []
    audit_rows: list[dict] = []
    reference_shape: tuple[int, int] | None = None
    reference_conditions: list[str] | None = None
    for row in cohort.itertuples(index=False):
        slide_id = str(row.slide_id)
        shard_path = shard_root / f"{slide_id}.h5"
        summary_path = shard_root / f"{slide_id}.summary.json"
        shard_hash = sha256_file(shard_path)
        summary_hash = sha256_file(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "pass":
            raise ValueError(f"{slide_id}: shard summary is not pass")
        if summary.get("output_sha256") != shard_hash:
            raise ValueError(f"{slide_id}: H5 hash disagrees with shard summary")
        with h5py.File(shard_path, "r") as store:
            required_datasets = {"features", "condition_names", "location_index", "source_index"}
            if not required_datasets.issubset(store.keys()):
                raise ValueError(f"{slide_id}: incomplete H5 schema")
            conditions = _decode(np.asarray(store["condition_names"]))
            missing = sorted(set(RAW_CONDITIONS) - set(conditions))
            if missing:
                raise ValueError(f"{slide_id}: missing raw conditions {missing}")
            if len(set(conditions)) != len(conditions):
                raise ValueError(f"{slide_id}: duplicate condition names")
            condition_index = {name: offset for offset, name in enumerate(conditions)}
            indices = [condition_index[name] for name in RAW_CONDITIONS]
            features = np.asarray(store["features"][:, indices, :], dtype=np.float32)
            location_index = np.asarray(store["location_index"], dtype=np.int64)
            source_index = np.asarray(store["source_index"], dtype=np.int64)
            attr_slide = str(store.attrs.get("slide_id", ""))
            attr_tissue = str(store.attrs.get("tissue_type", ""))
            analysis_id = str(store.attrs.get("analysis_id", ""))
        if attr_slide != slide_id or attr_tissue != str(row.tissue_type):
            raise ValueError(f"{slide_id}: H5 attributes disagree with cohort")
        if features.ndim != 3 or features.shape[1] != len(SCANNERS):
            raise ValueError(f"{slide_id}: unexpected raw feature shape {features.shape}")
        if len(location_index) != features.shape[0] or len(source_index) != features.shape[0]:
            raise ValueError(f"{slide_id}: feature/metadata length mismatch")
        if len(np.unique(location_index)) != len(location_index):
            raise ValueError(f"{slide_id}: duplicate location_index")
        if len(np.unique(source_index)) != len(source_index):
            raise ValueError(f"{slide_id}: duplicate source_index")
        if not np.isfinite(features).all():
            raise ValueError(f"{slide_id}: non-finite raw embedding")
        norms = np.linalg.norm(features, axis=-1)
        max_norm_error = float(np.max(np.abs(norms - 1.0)))
        if max_norm_error > 5e-4:
            raise ValueError(f"{slide_id}: embeddings are not unit normalized")
        if reference_shape is None:
            reference_shape = (features.shape[0], features.shape[2])
            reference_conditions = conditions
        if (features.shape[0], features.shape[2]) != reference_shape:
            raise ValueError(f"{slide_id}: inconsistent location count or feature dimension")
        if conditions != reference_conditions:
            raise ValueError(f"{slide_id}: inconsistent condition schema/order")
        if int(summary.get("locations", -1)) != features.shape[0]:
            raise ValueError(f"{slide_id}: summary location count mismatch")
        if int(summary.get("feature_dim", -1)) != features.shape[2]:
            raise ValueError(f"{slide_id}: summary feature dimension mismatch")
        embeddings.append(features)
        locations.append(location_index)
        audit_rows.append(
            {
                "slide_id": slide_id,
                "tissue_type": str(row.tissue_type),
                "fold": int(row.fold),
                "analysis_id": analysis_id,
                "locations": int(features.shape[0]),
                "raw_scanners": int(features.shape[1]),
                "feature_dim": int(features.shape[2]),
                "finite": True,
                "max_unit_norm_error": max_norm_error,
                "h5_bytes": shard_path.stat().st_size,
                "h5_sha256": shard_hash,
                "summary_sha256": summary_hash,
            }
        )
    embedding_array = np.stack(embeddings)
    location_array = np.stack(locations)
    if embedding_array.shape[:3] != (len(cohort), 20, 6):
        raise ValueError(f"expected 103-like x 20 x 6 raw panel, got {embedding_array.shape}")
    return embedding_array, location_array, pd.DataFrame(audit_rows)


def paired_distance_tables(
    embeddings: np.ndarray, cohort: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    source = embeddings[:, :, 0, :]
    for scanner_index, scanner in enumerate(TARGET_SCANNERS, start=1):
        distance = 1.0 - np.sum(source * embeddings[:, :, scanner_index, :], axis=-1)
        for slide_offset, cohort_row in enumerate(cohort.itertuples(index=False)):
            values = distance[slide_offset]
            rows.append(
                {
                    "slide_id": str(cohort_row.slide_id),
                    "tissue_type": str(cohort_row.tissue_type),
                    "fold": int(cohort_row.fold),
                    "reference_scanner": "at2",
                    "scanner": scanner,
                    "n_locations": len(values),
                    "mean_cosine_distance": float(values.mean()),
                    "median_cosine_distance": float(np.median(values)),
                    "q1_cosine_distance": float(np.quantile(values, 0.25)),
                    "q3_cosine_distance": float(np.quantile(values, 0.75)),
                    "min_cosine_distance": float(values.min()),
                    "max_cosine_distance": float(values.max()),
                }
            )
    by_slide = pd.DataFrame(rows)
    summary_rows: list[dict] = []
    for scanner, frame in by_slide.groupby("scanner", sort=False):
        values = frame["mean_cosine_distance"].to_numpy()
        estimate, low, high = bootstrap_mean(
            values, 20_000, stable_seed("paired-distance", scanner)
        )
        summary_rows.append(
            {
                "reference_scanner": "at2",
                "scanner": scanner,
                "n_slides": len(frame),
                "n_locations": int(frame["n_locations"].sum()),
                "slide_mean": estimate,
                "slide_mean_ci_low": low,
                "slide_mean_ci_high": high,
                "slide_median": float(np.median(values)),
                "slide_q1": float(np.quantile(values, 0.25)),
                "slide_q3": float(np.quantile(values, 0.75)),
                "bootstrap_unit": "physical_slide",
                "bootstrap_replicates": 20_000,
            }
        )
    return by_slide, pd.DataFrame(summary_rows)


def make_classifier_frame(
    embeddings: np.ndarray, location_indices: np.ndarray, cohort: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    slide_count, location_count, scanner_count, feature_dim = embeddings.shape
    matrix = embeddings.reshape(slide_count * location_count * scanner_count, feature_dim)
    rows: list[dict] = []
    for slide_offset, row in enumerate(cohort.itertuples(index=False)):
        for location_offset in range(location_count):
            for scanner_index, scanner in enumerate(SCANNERS):
                rows.append(
                    {
                        "slide_id": str(row.slide_id),
                        "tissue_type": str(row.tissue_type),
                        "fold": int(row.fold),
                        "location_index": int(location_indices[slide_offset, location_offset]),
                        "scanner": scanner,
                        "scanner_index": scanner_index,
                    }
                )
    metadata = pd.DataFrame(rows)
    if len(metadata) != len(matrix):
        raise AssertionError("classifier feature/metadata order mismatch")
    return matrix, metadata


def _classifier(alpha: float, seed: int, workers: int) -> object:
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            class_weight="balanced",
            max_iter=1_000,
            tol=1e-4,
            average=True,
            random_state=seed,
            n_jobs=max(1, workers),
        ),
    )


def nested_scanner_classifier(
    matrix: np.ndarray,
    metadata: pd.DataFrame,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    seed: int = 20260917,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a six-class classifier using locked outer and inner slide folds."""
    folds = metadata["fold"].to_numpy(dtype=int)
    labels = metadata["scanner_index"].to_numpy(dtype=int)
    all_predictions: list[pd.DataFrame] = []
    tuning_rows: list[dict] = []
    for outer_fold in sorted(np.unique(folds)):
        outer_train = folds != outer_fold
        outer_test = folds == outer_fold
        candidate_scores: dict[float, float] = {}
        for alpha in alphas:
            inner_scores = []
            for inner_fold in sorted(np.unique(folds[outer_train])):
                inner_train = outer_train & (folds != inner_fold)
                inner_test = outer_train & (folds == inner_fold)
                model = _classifier(
                    float(alpha), stable_seed(seed, outer_fold, inner_fold, alpha), workers
                )
                model.fit(matrix[inner_train], labels[inner_train])
                guess = model.predict(matrix[inner_test])
                score = balanced_accuracy_score(labels[inner_test], guess)
                inner_scores.append(float(score))
                tuning_rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "inner_fold": int(inner_fold),
                        "alpha": float(alpha),
                        "balanced_accuracy": float(score),
                        "selected": False,
                    }
                )
            candidate_scores[float(alpha)] = float(np.mean(inner_scores))
        selected_alpha = max(candidate_scores, key=lambda a: (candidate_scores[a], a))
        for item in tuning_rows:
            if item["outer_fold"] == outer_fold and item["alpha"] == selected_alpha:
                item["selected"] = True
        model = _classifier(
            selected_alpha, stable_seed(seed, "outer", outer_fold, selected_alpha), workers
        )
        model.fit(matrix[outer_train], labels[outer_train])
        predicted = model.predict(matrix[outer_test]).astype(int)
        decision = model.decision_function(matrix[outer_test])
        classes = model.named_steps["sgdclassifier"].classes_.astype(int)
        aligned = np.zeros((outer_test.sum(), len(SCANNERS)), dtype=np.float64)
        aligned[:, classes] = decision
        if not np.isfinite(aligned).all():
            raise ValueError(f"outer fold {outer_fold}: non-finite classifier score")
        frame = metadata.loc[
            outer_test, ["slide_id", "tissue_type", "fold", "location_index", "scanner"]
        ].copy()
        frame["true_scanner"] = frame.pop("scanner")
        frame["predicted_scanner"] = [SCANNERS[index] for index in predicted]
        frame["correct"] = (predicted == labels[outer_test]).astype(int)
        frame["selected_alpha"] = selected_alpha
        for scanner_index, scanner in enumerate(SCANNERS):
            frame[f"score_{scanner}"] = aligned[:, scanner_index]
        all_predictions.append(frame)
    predictions = pd.concat(all_predictions, ignore_index=True)
    expected = metadata[
        ["slide_id", "fold", "location_index", "scanner"]
    ].rename(columns={"scanner": "true_scanner"})
    observed = predictions[["slide_id", "fold", "location_index", "true_scanner"]]
    if len(observed) != len(expected) or observed.duplicated().any():
        raise ValueError("outer classifier predictions are incomplete or duplicated")
    if set(map(tuple, observed.itertuples(index=False, name=None))) != set(
        map(tuple, expected.itertuples(index=False, name=None))
    ):
        raise ValueError("outer classifier prediction keys differ from input keys")
    return predictions, pd.DataFrame(tuning_rows)


def macro_ovr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Macro one-vs-rest AUROC from unrestricted class decision scores."""
    values = [
        roc_auc_score(labels == scanner_index, scores[:, scanner_index])
        for scanner_index in range(len(SCANNERS))
    ]
    return float(np.mean(values))


def classifier_cluster_bootstrap(
    predictions: pd.DataFrame, bootstrap_count: int, seed: int
) -> dict[str, float]:
    """Exact physical-slide cluster bootstrap for BA and macro one-vs-rest AUC."""
    slide_names = sorted(predictions["slide_id"].unique())
    slide_lookup = {name: offset for offset, name in enumerate(slide_names)}
    slide_index = predictions["slide_id"].map(slide_lookup).to_numpy(dtype=int)
    label_lookup = {name: offset for offset, name in enumerate(SCANNERS)}
    labels = predictions["true_scanner"].map(label_lookup).to_numpy(dtype=int)
    guesses = predictions["predicted_scanner"].map(label_lookup).to_numpy(dtype=int)
    scores = predictions[[f"score_{name}" for name in SCANNERS]].to_numpy()
    weights = multinomial_slide_weights(len(slide_names), bootstrap_count, seed)

    recall_by_slide_class = np.empty((len(slide_names), len(SCANNERS)), dtype=np.float64)
    for slide in range(len(slide_names)):
        for scanner in range(len(SCANNERS)):
            selected = (slide_index == slide) & (labels == scanner)
            if selected.sum() == 0:
                raise ValueError("every slide must contain every scanner class")
            recall_by_slide_class[slide, scanner] = np.mean(guesses[selected] == scanner)
    ba_by_slide = recall_by_slide_class.mean(axis=1)
    ba_boot = weights @ ba_by_slide / len(slide_names)
    ba_point = float(balanced_accuracy_score(labels, guesses))
    ba_low, ba_high = percentile_interval(ba_boot)

    contribution_matrices = []
    positive_count = np.zeros((len(SCANNERS), len(slide_names)), dtype=int)
    negative_count = np.zeros_like(positive_count)
    for scanner in range(len(SCANNERS)):
        score = scores[:, scanner]
        matrix = np.zeros((len(slide_names), len(slide_names)), dtype=np.float64)
        for positive_slide in range(len(slide_names)):
            positive_scores = score[(labels == scanner) & (slide_index == positive_slide)]
            positive_count[scanner, positive_slide] = len(positive_scores)
            for negative_slide in range(len(slide_names)):
                negative_scores = np.sort(
                    score[(labels != scanner) & (slide_index == negative_slide)]
                )
                negative_count[scanner, negative_slide] = len(negative_scores)
                left = np.searchsorted(negative_scores, positive_scores, side="left")
                right = np.searchsorted(negative_scores, positive_scores, side="right")
                matrix[positive_slide, negative_slide] = float(
                    np.sum(left + 0.5 * (right - left))
                )
        contribution_matrices.append(matrix)
    if not (
        np.all(positive_count == positive_count[0, 0])
        and np.all(negative_count == negative_count[0, 0])
    ):
        raise ValueError("fast macro-AUROC bootstrap requires a balanced six-scanner panel")
    mean_contribution = np.mean(contribution_matrices, axis=0)
    numerator = np.sum((weights @ mean_contribution) * weights, axis=1)
    positive_per_slide = int(positive_count[0, 0])
    negative_per_slide = int(negative_count[0, 0])
    weighted_slide_count = weights.sum(axis=1)
    auc_boot = numerator / (
        positive_per_slide * negative_per_slide * np.square(weighted_slide_count)
    )
    auc_point = macro_ovr_auc(labels, scores)
    matrix_point = float(
        mean_contribution.sum()
        / (positive_per_slide * negative_per_slide * len(slide_names) ** 2)
    )
    if not np.isclose(auc_point, matrix_point, atol=1e-12):
        raise AssertionError("cluster-AUROC sufficient statistic disagrees with sklearn")
    auc_low, auc_high = percentile_interval(auc_boot)
    return {
        "n_slides": len(slide_names),
        "n_predictions": len(predictions),
        "balanced_accuracy": ba_point,
        "balanced_accuracy_ci_low": ba_low,
        "balanced_accuracy_ci_high": ba_high,
        "macro_ovr_auroc": auc_point,
        "macro_ovr_auroc_ci_low": auc_low,
        "macro_ovr_auroc_ci_high": auc_high,
        "bootstrap_unit": "physical_slide",
        "bootstrap_replicates": bootstrap_count,
    }


def classifier_summary_table(
    predictions: pd.DataFrame, bootstrap_count: int, seed: int
) -> pd.DataFrame:
    rows: list[dict] = []
    for fold, frame in predictions.groupby("fold", sort=True):
        labels = pd.Categorical(frame["true_scanner"], categories=SCANNERS).codes
        guesses = pd.Categorical(frame["predicted_scanner"], categories=SCANNERS).codes
        scores = frame[[f"score_{name}" for name in SCANNERS]].to_numpy()
        rows.append(
            {
                "scope": "outer_fold",
                "fold": int(fold),
                "n_slides": frame["slide_id"].nunique(),
                "n_predictions": len(frame),
                "balanced_accuracy": balanced_accuracy_score(labels, guesses),
                "macro_ovr_auroc": macro_ovr_auc(labels, scores),
            }
        )
    overall = classifier_cluster_bootstrap(predictions, bootstrap_count, seed)
    rows.append({"scope": "all_oof", "fold": "all", **overall})
    return pd.DataFrame(rows)


def biological_distance_by_slide(
    embeddings: np.ndarray, cohort: pd.DataFrame
) -> pd.DataFrame:
    """Create one equally weighted biological-distance score per physical slide.

    Embeddings are averaged over locations without re-normalization.  Therefore
    ``1 - dot(mean_a, mean_b)`` is exactly the mean of all 20x20 location-pair
    cosine distances.  Different-tissue scores first average slides within each
    comparator tissue and then average tissues, preventing large tissues from
    dominating the biological scale.
    """
    centroids = embeddings.mean(axis=1)
    tissues = cohort["tissue_type"].astype(str).to_numpy()
    tissue_names = sorted(np.unique(tissues))
    rows: list[dict] = []
    for slide_index, row in enumerate(cohort.itertuples(index=False)):
        same_indices = np.flatnonzero(
            (tissues == str(row.tissue_type)) & (np.arange(len(cohort)) != slide_index)
        )
        other_tissues = [name for name in tissue_names if name != str(row.tissue_type)]
        for scanner_index, scanner in enumerate(SCANNERS):
            focal = centroids[slide_index, scanner_index]
            if len(same_indices):
                same_values = 1.0 - centroids[same_indices, scanner_index] @ focal
                same_distance = float(same_values.mean())
            else:
                same_distance = float("nan")
            tissue_values = []
            partner_slides = 0
            for tissue in other_tissues:
                indices = np.flatnonzero(tissues == tissue)
                partner_slides += len(indices)
                tissue_values.append(
                    float(np.mean(1.0 - centroids[indices, scanner_index] @ focal))
                )
            rows.append(
                {
                    "slide_id": str(row.slide_id),
                    "tissue_type": str(row.tissue_type),
                    "fold": int(row.fold),
                    "scanner": scanner,
                    "same_tissue_distance": same_distance,
                    "same_tissue_partner_slides": len(same_indices),
                    "between_tissue_distance": float(np.mean(tissue_values)),
                    "between_tissue_partner_tissues": len(other_tissues),
                    "between_tissue_partner_slides": partner_slides,
                }
            )
    return pd.DataFrame(rows)


def biology_summary_table(
    biology: pd.DataFrame,
    paired_by_slide: pd.DataFrame,
    bootstrap_count: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for scanner, frame in biology.groupby("scanner", sort=False):
        for relation, column in (
            ("same_tissue_different_slide", "same_tissue_distance"),
            ("between_tissue", "between_tissue_distance"),
        ):
            values = frame[column].dropna().to_numpy(dtype=float)
            estimate, low, high = bootstrap_mean(
                values, bootstrap_count, stable_seed(seed, scanner, relation)
            )
            rows.append(
                {
                    "record_type": "biology_distance",
                    "scanner": scanner,
                    "comparator_scanner": scanner,
                    "biology_relation": relation,
                    "n_slides": len(values),
                    "mean_distance": estimate,
                    "mean_distance_ci_low": low,
                    "mean_distance_ci_high": high,
                    "median_distance": float(np.median(values)),
                    "q1_distance": float(np.quantile(values, 0.25)),
                    "q3_distance": float(np.quantile(values, 0.75)),
                    "bootstrap_unit": "physical_slide",
                    "bootstrap_replicates": bootstrap_count,
                }
            )

    slide_order = sorted(biology["slide_id"].unique())
    for target_scanner, paired in paired_by_slide.groupby("scanner", sort=False):
        paired_vector = (
            paired.set_index("slide_id").loc[slide_order, "mean_cosine_distance"].to_numpy()
        )
        for comparator in ("at2", target_scanner):
            biological_vector = (
                biology[biology["scanner"] == comparator]
                .set_index("slide_id")
                .loc[slide_order, "between_tissue_distance"]
                .to_numpy()
            )
            weights = multinomial_slide_weights(
                len(slide_order),
                bootstrap_count,
                stable_seed(seed, "ratio", target_scanner, comparator),
            )
            paired_boot = weights @ paired_vector / len(slide_order)
            biology_boot = weights @ biological_vector / len(slide_order)
            ratio_boot = paired_boot / biology_boot
            percentile_boot = np.empty(bootstrap_count, dtype=np.float64)
            for index in range(bootstrap_count):
                threshold = paired_boot[index]
                percentile_boot[index] = 100.0 * np.sum(
                    weights[index] * (biological_vector <= threshold)
                ) / len(slide_order)
            ratio_low, ratio_high = percentile_interval(ratio_boot)
            percentile_low, percentile_high = percentile_interval(percentile_boot)
            paired_mean = float(paired_vector.mean())
            biology_mean = float(biological_vector.mean())
            rows.append(
                {
                    "record_type": "scanner_vs_biology",
                    "scanner": target_scanner,
                    "comparator_scanner": comparator,
                    "biology_relation": "between_tissue",
                    "n_slides": len(slide_order),
                    "mean_distance": biology_mean,
                    "paired_scanner_distance": paired_mean,
                    "scanner_to_biology_ratio": paired_mean / biology_mean,
                    "scanner_to_biology_ratio_ci_low": ratio_low,
                    "scanner_to_biology_ratio_ci_high": ratio_high,
                    "scanner_distance_percentile_of_slide_biology": 100.0
                    * np.mean(biological_vector <= paired_mean),
                    "scanner_distance_percentile_ci_low": percentile_low,
                    "scanner_distance_percentile_ci_high": percentile_high,
                    "bootstrap_unit": "physical_slide",
                    "bootstrap_replicates": bootstrap_count,
                }
            )
    return pd.DataFrame(rows)


def _git_value(args: list[str]) -> str | bool | None:
    try:
        result = subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True, timeout=30
        )
        output = result.stdout.strip()
        return output
    except (OSError, subprocess.SubprocessError):
        return None


def validate_outputs(output_root: Path, expected_predictions: int, expected_slides: int) -> dict:
    required = {
        "input_shard_audit.csv": expected_slides,
        "scanner_paired_distance_by_slide.csv": expected_slides * len(TARGET_SCANNERS),
        "scanner_paired_distance_summary.csv": len(TARGET_SCANNERS),
        "scanner_classifier_predictions.csv": expected_predictions,
        "scanner_classifier_summary.csv": 6,
        "scanner_classifier_tuning.csv": 5 * 4 * len(DEFAULT_ALPHAS),
        "scanner_vs_biology_distance_by_slide.csv": expected_slides * len(SCANNERS),
        "scanner_vs_biology_distance_summary.csv": len(SCANNERS) * 2
        + len(TARGET_SCANNERS) * 2,
    }
    checks = {}
    for name, row_count in required.items():
        path = output_root / name
        frame = pd.read_csv(path)
        if len(frame) != row_count:
            raise ValueError(f"{name}: expected {row_count} rows, got {len(frame)}")
        numeric = frame.select_dtypes(include=[np.number])
        # same-tissue is intentionally missing only for the singleton tissue.
        if name != "scanner_vs_biology_distance_by_slide.csv" and np.isinf(numeric).any().any():
            raise ValueError(f"{name}: infinite numeric output")
        checks[name] = {"rows": len(frame), "sha256": sha256_file(path)}
    return checks


def run_analysis(args: argparse.Namespace) -> None:
    started = utc_now()
    cohort = load_contract(args.cohort, args.folds)
    if len(cohort) != 103:
        raise ValueError(f"locked analysis requires 103 slides, got {len(cohort)}")
    embeddings, locations, audit = audit_and_load_raw_embeddings(cohort, args.shard_root)
    write_csv(args.output_root / "input_shard_audit.csv", audit)

    paired_by_slide, paired_summary = paired_distance_tables(embeddings, cohort)
    write_csv(args.output_root / "scanner_paired_distance_by_slide.csv", paired_by_slide)
    write_csv(args.output_root / "scanner_paired_distance_summary.csv", paired_summary)

    matrix, metadata = make_classifier_frame(embeddings, locations, cohort)
    predictions, tuning = nested_scanner_classifier(
        matrix,
        metadata,
        alphas=args.alphas,
        seed=args.seed,
        workers=args.workers,
    )
    classifier_summary = classifier_summary_table(
        predictions, args.bootstraps, stable_seed(args.seed, "classifier")
    )
    write_csv(args.output_root / "scanner_classifier_predictions.csv", predictions)
    write_csv(args.output_root / "scanner_classifier_summary.csv", classifier_summary)
    write_csv(args.output_root / "scanner_classifier_tuning.csv", tuning)

    biology = biological_distance_by_slide(embeddings, cohort)
    biology_summary = biology_summary_table(
        biology, paired_by_slide, args.bootstraps, stable_seed(args.seed, "biology")
    )
    write_csv(args.output_root / "scanner_vs_biology_distance_by_slide.csv", biology)
    write_csv(args.output_root / "scanner_vs_biology_distance_summary.csv", biology_summary)

    expected_predictions = len(cohort) * embeddings.shape[1] * len(SCANNERS)
    output_checks = validate_outputs(args.output_root, expected_predictions, len(cohort))
    raw_hashes = [(str(row.slide_id), row.h5_sha256) for row in audit.itertuples(index=False)]
    source_path = Path(__file__).resolve()
    package_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "h5py": h5py.__version__,
        "scikit_learn": sklearn.__version__,
    }
    manifest = {
        "analysis": "PFM problem definition",
        "status": "pass",
        "started_utc": started,
        "completed_utc": utc_now(),
        "contract": {
            "slides": len(cohort),
            "tissues": int(cohort["tissue_type"].nunique()),
            "scanners": list(SCANNERS),
            "locations_per_slide": int(embeddings.shape[1]),
            "feature_dim": int(embeddings.shape[-1]),
            "reference_scanner": "at2",
            "outer_cv": "locked five-fold physical-slide split",
            "inner_cv": "remaining four locked slide folds",
            "classifier": "standardized averaged SGD multinomial log-loss linear probe",
            "classifier_alpha_grid": list(map(float, args.alphas)),
            "bootstrap_unit": "physical_slide",
            "bootstrap_replicates": args.bootstraps,
            "seed": args.seed,
            "same_tissue_rule": (
                "per-focal-slide mean of all 20x20 location pairs to every other slide "
                "of the same tissue; singleton tissue excluded"
            ),
            "between_tissue_rule": (
                "per-focal-slide mean of all 20x20 location pairs, first equally averaging "
                "partner slides within tissue and then equally averaging other tissues"
            ),
        },
        "encoder_audit": {
            "status": "pass_with_provenance_limitation",
            "raw_conditions_uniform_across_shards": True,
            "all_embeddings_finite_and_unit_normalized": True,
            "single_analysis_id": sorted(audit["analysis_id"].unique().tolist()),
            "checkpoint_sha256_present_in_shards": False,
            "preprocessing_manifest_present_in_shards": False,
            "limitation": (
                "The existing shards prove a uniform schema and joint extraction of all six "
                "raw scanners, but do not persist a model-artifact checksum or preprocessing "
                "payload; encoder identity cannot be independently reconstructed from H5 alone."
            ),
        },
        "inputs": {
            "shard_root": str(args.shard_root.resolve()),
            "shard_count": len(audit),
            "raw_shard_set_sha256": sha256_records(raw_hashes),
            "cohort": str(args.cohort.resolve()),
            "cohort_sha256": sha256_file(args.cohort),
            "folds": str(args.folds.resolve()),
            "folds_sha256": sha256_file(args.folds),
            "source_module": str(source_path),
            "source_module_sha256": sha256_file(source_path),
        },
        "environment": {
            "executable": sys.executable,
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "hostname": platform.node(),
            "package_versions": package_versions,
            "git_commit": _git_value(["rev-parse", "HEAD"]),
            "git_dirty": bool(_git_value(["status", "--porcelain"])),
            "argv": sys.argv,
        },
        "outputs": output_checks,
        "classifier_all_oof": classifier_summary[
            classifier_summary["scope"] == "all_oof"
        ].iloc[0].dropna().to_dict(),
    }
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstraps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstraps != 20_000:
        raise ValueError("locked manuscript analysis requires exactly 20,000 bootstraps")
    if tuple(args.alphas) != DEFAULT_ALPHAS:
        raise ValueError(f"locked alpha grid is {DEFAULT_ALPHAS}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_analysis(args)


if __name__ == "__main__":
    main()
