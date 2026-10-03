#!/usr/bin/env python3
"""Open frozen UNI after translated images are locked and evaluate target closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from .uni import (
    aggregate_physical_slides,
    embed_frozen_uni,
    embedding_collapse_diagnostics,
    load_frozen_uni,
    paired_uni_metrics,
    summarize_physical_slides,
)


UNI_EVALUATION_VERSION = "scanner_translation_frozen_uni_reference_v3"
REFERENCE_UNI_METRICS = (
    "raw_to_target_distance",
    "method_to_target_distance",
    "gain_to_target",
    "fractional_closure",
    "method_to_raw_distance",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_prediction_manifests(
    prediction_dirs: Sequence[str | Path],
) -> tuple[str, str, str, str, str, list[tuple[Path, dict[str, Any]]]]:
    values = []
    for directory in prediction_dirs:
        root = Path(directory).resolve()
        path = root / "prediction_manifest.json"
        manifest = json.loads(path.read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"prediction manifest is not complete: {path}")
        if "UNI" in str(manifest.get("selection_boundary", "")) and not str(
            manifest["selection_boundary"]
        ).endswith("no UNI computation here"):
            raise ValueError(f"unexpected selection boundary: {path}")
        values.append((path, manifest))
    if not values:
        raise ValueError("at least one prediction directory is required")
    sources = {str(value[1]["source_scanner"]).lower() for value in values}
    targets = {str(value[1]["target_scanner"]).lower() for value in values}
    roles = {str(value[1].get("analysis_role", "")) for value in values}
    inputs = {
        str(value[1].get("inference_input_scanner", "")).lower()
        for value in values
    }
    methods = {
        str(value[1].get("method", "pix2pix")).strip().lower()
        for value in values
    }
    if (
        len(sources) != 1
        or len(targets) != 1
        or len(roles) != 1
        or len(inputs) != 1
        or len(methods) != 1
    ):
        raise ValueError("prediction manifests mix scanner directions or analysis roles")
    folds = [int(value[1]["test_fold"]) for value in values]
    if len(folds) != len(set(folds)):
        raise ValueError(f"duplicate outer folds: {folds}")
    source = sources.pop()
    target = targets.pop()
    role = roles.pop()
    inference_input = inputs.pop()
    method = methods.pop()
    if method not in {"pix2pix", "cyclegan"}:
        raise ValueError(f"unsupported prediction method {method!r}")
    expected_input = {
        "primary_translation": source,
        "target_identity_damage": target,
    }.get(role)
    if expected_input is None or inference_input != expected_input:
        raise ValueError(
            f"invalid analysis role/input contract: role={role!r}, "
            f"input={inference_input!r}, expected={expected_input!r}"
        )
    return (
        source,
        target,
        role,
        inference_input,
        method,
        sorted(values, key=lambda value: int(value[1]["test_fold"])),
    )


def _domain_probe(
    generated: np.ndarray,
    target: np.ndarray,
    slides: np.ndarray,
    *,
    seed: int = 20260917,
) -> dict[str, Any]:
    features = np.concatenate((generated, target), axis=0)
    labels = np.concatenate(
        (np.zeros(len(generated), dtype=int), np.ones(len(target), dtype=int))
    )
    groups = np.concatenate((slides, slides), axis=0)
    unique_slides = np.unique(slides)
    folds = min(5, len(unique_slides))
    if folds < 2:
        return {"balanced_accuracy": float("nan"), "folds": 0}
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for train, test in splitter.split(features, labels, groups):
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
        classifier.fit(features[train], labels[train])
        scores.append(
            balanced_accuracy_score(labels[test], classifier.predict(features[test]))
        )
    return {
        "balanced_accuracy": float(np.mean(scores)),
        "fold_scores": [float(value) for value in scores],
        "folds": folds,
        "grouping": "physical slide",
        "classes": ["generated", "real_target"],
    }


def evaluate(
    prediction_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    batch_size: int = 64,
    bootstrap_replicates: int = 20_000,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("frozen UNI evaluation requires CUDA")
    (
        source_scanner,
        target_scanner,
        analysis_role,
        inference_input_scanner,
        method,
        manifests,
    ) = _read_prediction_manifests(prediction_dirs)
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    model, size, mean, std = load_frozen_uni(device)

    all_metrics = []
    all_raw_features = []
    all_generated_features = []
    all_target_features = []
    all_slides = []
    shard_summaries = []
    seen_keys: set[tuple[str, int]] = set()

    for manifest_path, manifest in manifests:
        fold = int(manifest["test_fold"])
        for shard in manifest["shards"]:
            shard_path = Path(shard["path"])
            if sha256(shard_path) != shard["sha256"]:
                raise ValueError(f"prediction shard hash changed: {shard_path}")
            with h5py.File(shard_path, "r") as store:
                source = np.asarray(store["images/raw_source"], dtype=np.uint8)
                generated = np.asarray(store["images/generated"], dtype=np.uint8)
                target_key = (
                    "images/real_target"
                    if "images/real_target" in store
                    else "images/real_at2"
                )
                target = np.asarray(store[target_key], dtype=np.uint8)
                locations = np.asarray(store["metadata/location_index"], dtype=np.int64)
                slide_id = str(store.attrs["slide_id"])
                tissue = str(store.attrs["tissue_type"])
            if source.shape != generated.shape or source.shape != target.shape:
                raise ValueError(f"image shapes differ in {shard_path}")
            if source.shape != (40, 256, 256, 3):
                raise ValueError(
                    f"expected 40 locked 256px images in {shard_path}, got {source.shape}"
                )
            for location in locations:
                key = (slide_id, int(location))
                if key in seen_keys:
                    raise ValueError(
                        f"duplicate outer-test location across folds: {key}"
                    )
                seen_keys.add(key)

            images = np.concatenate((source, generated, target), axis=0)
            features = embed_frozen_uni(
                model,
                images,
                size,
                mean,
                std,
                device,
                batch_size=batch_size,
                value_range="uint8",
            )
            n = len(source)
            raw_features, generated_features, target_features = np.split(
                features, (n, 2 * n)
            )
            metadata = pd.DataFrame(
                {
                    "source_scanner": source_scanner,
                    "analysis_role": analysis_role,
                    "inference_input_scanner": inference_input_scanner,
                    "slide_id": slide_id,
                    "tissue_type": tissue,
                    "fold": fold,
                    "location_index": locations,
                    "prediction_shard": str(shard_path),
                    "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                }
            )
            metrics = paired_uni_metrics(
                raw_features,
                generated_features,
                target_features,
                metadata,
                method=method,
                target_scanner=target_scanner.upper(),
            )
            metrics = metrics.rename(
                columns={
                    "raw_to_at2_distance": "raw_to_target_distance",
                    "method_to_at2_distance": "method_to_target_distance",
                    "gain_to_at2": "gain_to_target",
                }
            )
            metrics_path = output_root / "location_shards" / f"{slide_id}.csv.gz"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(metrics_path, index=False, compression="gzip")

            feature_path = output_root / "feature_shards" / f"{slide_id}.h5"
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = feature_path.with_suffix(
                feature_path.suffix + f".{os.getpid()}.tmp"
            )
            with h5py.File(temporary, "w") as store:
                store.attrs["evaluation_version"] = UNI_EVALUATION_VERSION
                store.attrs["encoder_id"] = "uni_v1"
                store.attrs["source_scanner"] = source_scanner
                store.attrs["target_scanner"] = target_scanner
                store.attrs["slide_id"] = slide_id
                store.attrs["fold"] = fold
                store.create_dataset("location_index", data=locations)
                store.create_dataset("raw_source", data=raw_features, compression="lzf")
                store.create_dataset(
                    "generated", data=generated_features, compression="lzf"
                )
                store.create_dataset(
                    "real_target", data=target_features, compression="lzf"
                )
                if target_scanner == "at2":
                    store["real_at2"] = store["real_target"]
            temporary.replace(feature_path)

            all_metrics.append(metrics)
            all_raw_features.append(raw_features)
            all_generated_features.append(generated_features)
            all_target_features.append(target_features)
            all_slides.extend([slide_id] * n)
            shard_summaries.append(
                {
                    "slide_id": slide_id,
                    "fold": fold,
                    "locations": n,
                    "prediction_path": str(shard_path),
                    "prediction_sha256": shard["sha256"],
                    "feature_path": str(feature_path),
                    "feature_sha256": sha256(feature_path),
                    "metric_path": str(metrics_path),
                    "metric_sha256": sha256(metrics_path),
                }
            )

    location_metrics = pd.concat(all_metrics, ignore_index=True)
    expected_slides = sum(int(manifest["slides"]) for _, manifest in manifests)
    if location_metrics["slide_id"].nunique() != expected_slides:
        raise ValueError("UNI output slide count does not match prediction manifests")
    location_path = output_root / "location_metrics.csv.gz"
    location_metrics.to_csv(location_path, index=False, compression="gzip")
    slide_metrics = aggregate_physical_slides(
        location_metrics, metric_columns=REFERENCE_UNI_METRICS
    )
    slide_path = output_root / "slide_metrics.csv"
    slide_metrics.to_csv(slide_path, index=False)
    summary_frame = summarize_physical_slides(
        slide_metrics,
        metric_columns=REFERENCE_UNI_METRICS,
        bootstrap_replicates=bootstrap_replicates,
    )
    summary_path = output_root / "summary.csv"
    summary_frame.to_csv(summary_path, index=False)

    raw_features = np.concatenate(all_raw_features)
    generated_features = np.concatenate(all_generated_features)
    target_features = np.concatenate(all_target_features)
    collapse = {
        "raw_source": embedding_collapse_diagnostics(raw_features, target_features),
        method: embedding_collapse_diagnostics(generated_features, target_features),
        "real_target": embedding_collapse_diagnostics(target_features),
    }
    domain_probe = _domain_probe(
        generated_features,
        target_features,
        np.asarray(all_slides, dtype=object),
    )
    manifest_payload = {
        "evaluation_version": UNI_EVALUATION_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": "MahmoodLab/uni (frozen UNI-v1, 1024-d)",
        "encoder_id": "uni_v1",
        "source_scanner": source_scanner,
        "target_scanner": target_scanner,
        "analysis_role": analysis_role,
        "inference_input_scanner": inference_input_scanner,
        "method": method,
        "outer_folds": [int(value[1]["test_fold"]) for value in manifests],
        "locations": int(len(location_metrics)),
        "physical_slides": int(location_metrics["slide_id"].nunique()),
        "primary_endpoint": (
            "gain_to_target = raw_to_target_distance - method_to_target_distance"
        ),
        "fractional_closure_guard": "undefined when raw_to_target_distance <= 1e-6",
        "prediction_manifests": [
            {"path": str(path), "sha256": sha256(path)} for path, _ in manifests
        ],
        "outputs": {
            "location_metrics": {
                "path": str(location_path),
                "sha256": sha256(location_path),
            },
            "slide_metrics": {"path": str(slide_path), "sha256": sha256(slide_path)},
            "summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
        },
        "collapse_diagnostics": collapse,
        "real_target_vs_generated_probe": domain_probe,
        "shards": shard_summaries,
        "interpretation_boundary": (
            "UNI was opened only after image-only checkpoint selection and locked prediction. "
            "A favourable UNI result cannot override an image-fidelity or hallucination failure."
        ),
    }
    write_json(output_root / "evaluation_manifest.json", manifest_payload)
    return manifest_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    args = parser.parse_args()
    result = evaluate(
        args.prediction_dir,
        args.output_dir,
        batch_size=args.batch_size,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
