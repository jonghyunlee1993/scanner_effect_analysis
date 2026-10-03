#!/usr/bin/env python3
"""Same-tissue slide retrieval for the four existing AKOYA UNI arms."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ARMS = ("source", "target:akoya", "reinhard:akoya", "combined:akoya")
CONTRASTS = (
    ("combined_minus_reinhard", "combined:akoya", "reinhard:akoya"),
    ("combined_minus_source", "combined:akoya", "source"),
    ("combined_minus_target", "combined:akoya", "target:akoya"),
)


def normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def bootstrap(values: np.ndarray, rng: np.random.Generator, replicates: int) -> tuple[float, float]:
    indexes = rng.integers(0, len(values), size=(replicates, len(values)))
    samples = values[indexes].mean(axis=1)
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort, dtype={"slide_id": str}).sort_values("slide_id")
    if len(cohort) != 103 or cohort.slide_id.duplicated().any():
        raise ValueError("expected 103 unique slides")
    tensors = []
    for slide in cohort.slide_id:
        with h5py.File(args.shards / f"{slide}.h5") as handle:
            names = [x.decode() if isinstance(x, bytes) else str(x)
                     for x in handle["condition_names"][:]]
            index = [names.index(arm) for arm in ARMS]
            features = np.asarray(handle["features"][:, index, :], dtype=np.float32)
        if features.shape != (20, 4, 1024):
            raise ValueError((slide, features.shape))
        tensors.append(normalize(features.mean(axis=0)))
    matrix = np.stack(tensors)
    tissues = cohort.tissue_type.astype(str).to_numpy()
    counts = cohort.tissue_type.value_counts()
    evaluable = np.asarray([counts[tissue] > 1 for tissue in tissues])
    if evaluable.sum() != 102:
        raise ValueError("expected 102 evaluable slides")

    records = []
    for j, arm in enumerate(ARMS):
        scores = matrix[:, j, :] @ matrix[:, j, :].T
        np.fill_diagonal(scores, -np.inf)
        neighbours = scores.argmax(axis=1)
        for i, neighbour in enumerate(neighbours):
            if evaluable[i]:
                records.append({"slide_id": cohort.slide_id.iloc[i], "tissue_type": tissues[i],
                                "arm": arm, "nearest_slide": cohort.slide_id.iloc[neighbour],
                                "correct": int(tissues[i] == tissues[neighbour])})
    predictions = pd.DataFrame(records)
    wide = predictions.pivot(index=["slide_id", "tissue_type"], columns="arm", values="correct")
    if wide.isna().any().any() or len(wide) != 102 or wide.index.get_level_values("tissue_type").nunique() != 36:
        raise ValueError("incomplete paired retrieval outcomes")

    rng = np.random.default_rng(args.seed)
    summary = []
    for arm in ARMS:
        values = wide[arm]
        summary.append({"arm": arm, "macro_recall": float(values.groupby(level="tissue_type").mean().mean()),
                        "slide_accuracy": float(values.mean()), "n_slides": len(values), "n_tissues": 36})
    contrasts = []
    for label, treatment, comparator in CONTRASTS:
        slide_delta = (wide[treatment] - wide[comparator]).to_numpy(float)
        tissue_delta = (wide[treatment] - wide[comparator]).groupby(level="tissue_type").mean().to_numpy(float)
        for metric, values, unit in (
            ("macro_recall_difference", tissue_delta, "tissue_type"),
            ("slide_accuracy_difference", slide_delta, "physical_slide"),
        ):
            low, high = bootstrap(values, rng, args.replicates)
            contrasts.append({"contrast": label, "metric": metric,
                              "estimate": float(values.mean()), "ci_low": low, "ci_high": high,
                              "n_slides": len(wide), "n_tissues": 36,
                              "bootstrap_unit": unit, "bootstrap_replicates": args.replicates,
                              "bootstrap_seed": args.seed})

    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "akoya_tissue_retrieval_predictions.csv", index=False)
    pd.DataFrame(summary).to_csv(args.output / "akoya_tissue_retrieval_summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(args.output / "akoya_tissue_retrieval_contrasts.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(contrasts).to_string(index=False))


if __name__ == "__main__":
    main()
