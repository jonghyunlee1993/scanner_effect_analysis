#!/usr/bin/env python3
"""Exploratory held-out tissue probe on already stored AKOYA UNI embeddings."""

from pathlib import Path
import hashlib
import json

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/scanner_batch_effect_analysis_2026-09-17"
OUT = Path(__file__).resolve().parent / "results/frequency_pfm_audit"
ARMS = ["source", "target:akoya", "reinhard:akoya", "combined:akoya"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def effective_rank(x: np.ndarray) -> float:
    singular = np.linalg.svd(x.astype(np.float64) - x.mean(axis=0), compute_uv=False)
    p = singular**2
    p = p / p.sum()
    return float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort_path = BASE / "00_contract/cohort.csv"
    cohort = pd.read_csv(cohort_path, dtype={"slide_id": str}).sort_values("slide_id")
    slides = []
    files = []
    for row in cohort.itertuples(index=False):
        path = BASE / "03_uni/shards" / f"{row.slide_id}.h5"
        files.append(path)
        with h5py.File(path) as handle:
            names = [name.decode() if isinstance(name, bytes) else str(name)
                     for name in handle["condition_names"][:]]
            indexes = [names.index(name) for name in ARMS]
            features = np.asarray(handle["features"][:, indexes, :], dtype=np.float32)
            if features.shape != (20, 4, 1024):
                raise ValueError((row.slide_id, features.shape))
            centroids = norm(features.mean(axis=0))
            slides.append(centroids)
    matrix = np.stack(slides)
    labels = cohort.tissue_type.astype(str).to_numpy()
    folds = cohort.fold.to_numpy(int)
    records = []
    for outer in sorted(np.unique(folds)):
        train = folds != outer
        seen = np.unique(labels[train])
        test = (folds == outer) & np.isin(labels, seen)
        test_indices = np.flatnonzero(test)
        for train_arm in ARMS:
            training = matrix[train, ARMS.index(train_arm)]
            class_centroids = norm(np.stack([training[labels[train] == label].mean(axis=0) for label in seen]))
            for test_arm in ARMS:
                if train_arm != test_arm and train_arm != "target:akoya":
                    continue
                scores = matrix[test, ARMS.index(test_arm)] @ class_centroids.T
                guesses = seen[scores.argmax(axis=1)]
                for index, guess in zip(test_indices, guesses):
                    records.append({"slide_id": cohort.iloc[index].slide_id,
                                    "tissue_type": labels[index], "fold": int(outer),
                                    "train_arm": train_arm, "test_arm": test_arm,
                                    "correct": int(guess == labels[index])})
    predictions = pd.DataFrame(records)
    predictions.to_csv(OUT / "akoya_tissue_probe_predictions.csv", index=False)
    summary = []
    for (train_arm, test_arm), group in predictions.groupby(["train_arm", "test_arm"]):
        by_tissue = group.groupby("tissue_type").correct.mean()
        summary.append({"train_arm": train_arm, "test_arm": test_arm,
                        "macro_recall": float(by_tissue.mean()),
                        "slide_accuracy": float(group.correct.mean()),
                        "evaluated_slides": int(len(group)),
                        "evaluated_tissues": int(len(by_tissue))})
    pd.DataFrame(summary).to_csv(OUT / "akoya_tissue_probe_summary.csv", index=False)
    pd.DataFrame({"arm": ARMS,
                  "slide_centroid_effective_rank": [effective_rank(matrix[:, j]) for j in range(4)]}
                ).to_csv(OUT / "akoya_embedding_rank.csv", index=False)
    outputs = ["akoya_tissue_probe_predictions.csv", "akoya_tissue_probe_summary.csv", "akoya_embedding_rank.csv"]
    manifest = {"purpose": "exploratory tissue retention check for current AKOYA UNI-v1 arms",
                "method": "cosine nearest class centroid; outer physical-slide folds; train-target transfer and within-arm probes",
                "n_slides": len(cohort), "n_arms": len(ARMS),
                "cohort_sha256": sha256(cohort_path),
                "code_sha256": sha256(Path(__file__)),
                "output_sha256": {name: sha256(OUT / name) for name in outputs},
                "h5_file_count": len(files)}
    (OUT / "retention_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
