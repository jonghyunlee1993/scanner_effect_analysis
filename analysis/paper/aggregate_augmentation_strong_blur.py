#!/usr/bin/env python3
"""Audit extreme blur convergence and loss of image gradients."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
AUGMENTATION = PROJECT / "outputs/augmentation_ood_v1"
COMMON = Path(__file__).resolve().parent / "results/augmentation_common_uni_space"
OUTPUT = Path(__file__).resolve().parent / "results/augmentation_strong_blur"
SCANNERS = ("at2", "versa", "akoya", "gt450", "s360", "s60")
SIGMAS = (0., 3., 6., 12., 24., 48.)


def main() -> None:
    cohort = pd.read_csv(AUGMENTATION / "00_contract/cohort.csv", dtype={"slide_id": str})
    features, details, locations = [], [], []
    for slide_id in cohort.slide_id:
        path = OUTPUT / "shards" / f"{slide_id}.h5"
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as store:
            loc = np.asarray(store["location_index"], dtype=int)
            found = tuple(np.asarray(store["sigmas"], dtype=float))
            f = np.asarray(store["features"], dtype=np.float32)
            d = np.asarray(store["detail_metrics"], dtype=np.float32)
        if f.shape != (3, 6, 6, 1024) or d.shape != (3, 6, 6, 2) or not np.allclose(found, SIGMAS):
            raise ValueError(f"incomplete strong blur shard: {slide_id}")
        features.append(f)
        details.append(d)
        locations.append(loc)
    features = np.stack(features)
    details = np.stack(details)
    raw = features[:, :, :, 0]
    center = raw.mean(axis=2, keepdims=True)
    locked = np.load(COMMON / "paired_centered_blur_sharp_pca.npz")
    components = np.asarray(locked["components"][:2], dtype=np.float32)
    mean = np.asarray(locked["mean"], dtype=np.float32)
    transformed = (features - center[:, :, :, None, :] - mean) @ components.T
    if transformed.shape != (103, 3, 6, 6, 2):
        raise ValueError("unexpected locked UNI projection")
    slide_rows = []
    for slide_index, slide in enumerate(cohort.itertuples(index=False)):
        for scanner_index, scanner in enumerate(SCANNERS):
            for step, sigma in enumerate(SIGMAS):
                x, y = transformed[slide_index, :, scanner_index, step].mean(axis=0)
                slide_rows.append({
                    "slide_id": str(slide.slide_id),
                    "tissue_type": str(slide.tissue_type),
                    "scanner": scanner,
                    "step": step,
                    "sigma": sigma,
                    "pc1": x,
                    "pc2": y,
                    "gradient_energy_ratio": float(details[slide_index, :, scanner_index, step, 0].mean()),
                    "gradient_ncc": float(details[slide_index, :, scanner_index, step, 1].mean()),
                })
    slides = pd.DataFrame(slide_rows)
    slides.to_csv(OUTPUT / "slide_positions.csv", index=False)
    centroids = slides.groupby(["scanner", "step", "sigma"], as_index=False)[["pc1", "pc2"]].mean()
    centroids.to_csv(OUTPUT / "centroid_positions.csv", index=False)

    pair_rows = []
    for left, right in combinations(range(6), 2):
        baseline = 1 - np.einsum("sld,sld->sl", raw[:, :, left], raw[:, :, right])
        raw_slide = baseline.mean(axis=1)
        for step, sigma in enumerate(SIGMAS):
            distance = 1 - np.einsum(
                "sld,sld->sl", features[:, :, left, step], features[:, :, right, step]
            )
            changed = distance.mean(axis=1)
            reduction = 100 * (raw_slide - changed) / raw_slide
            pair_rows.append({
                "scanner_a": SCANNERS[left],
                "scanner_b": SCANNERS[right],
                "sigma": sigma,
                "raw_gap_mean": float(raw_slide.mean()),
                "changed_gap_mean": float(changed.mean()),
                "mean_gap_ratio": float(changed.mean() / raw_slide.mean()),
                "median_slide_gap_reduction_pct": float(np.median(reduction)),
                "positive_slide_count": int((reduction > 0).sum()),
            })
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUTPUT / "full_uni_paired_gaps.csv", index=False)

    tissue = cohort.tissue_type.astype(str).to_numpy()
    counts = pd.Series(tissue).value_counts()
    evaluable = np.asarray([counts[value] > 1 for value in tissue], dtype=bool)
    retrieval_rows = []
    for scanner_index, scanner in enumerate(SCANNERS):
        for step, sigma in enumerate(SIGMAS):
            vectors = features[:, :, scanner_index, step].mean(axis=1)
            vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            similarities = vectors @ vectors.T
            np.fill_diagonal(similarities, -np.inf)
            nearest = similarities.argmax(axis=1)
            retrieval_rows.append({
                "scanner": scanner,
                "sigma": sigma,
                "evaluable_slides": int(evaluable.sum()),
                "top1_same_tissue_rate": float(np.mean(tissue[nearest[evaluable]] == tissue[evaluable])),
                "measure_note": "Slide-centroid nearest neighbour among other physical slides; tissues with one slide excluded from queries",
            })
    pd.DataFrame(retrieval_rows).to_csv(OUTPUT / "tissue_retrieval.csv", index=False)

    summaries = []
    base_xy = centroids[centroids.sigma.eq(0)].set_index("scanner")[["pc1", "pc2"]].loc[list(SCANNERS)].to_numpy(float)
    baseline_2d = np.asarray([
        np.linalg.norm(base_xy[i] - base_xy[j]) for i, j in combinations(range(6), 2)
    ])
    for step, sigma in enumerate(SIGMAS):
        xy = centroids[centroids.sigma.eq(sigma)].set_index("scanner")[["pc1", "pc2"]].loc[list(SCANNERS)].to_numpy(float)
        distances_2d = np.asarray([
            np.linalg.norm(xy[i] - xy[j]) for i, j in combinations(range(6), 2)
        ])
        pair = pairs[pairs.sigma.eq(sigma)]
        detail = slides[slides.sigma.eq(sigma)]
        summaries.append({
            "sigma": sigma,
            "median_2d_centroid_gap_ratio": float(np.median(distances_2d / baseline_2d)),
            "mean_2d_centroid_gap_ratio": float(distances_2d.mean() / baseline_2d.mean()),
            "median_full_uni_gap_ratio_across_pairs": float(pair.mean_gap_ratio.median()),
            "mean_full_uni_gap_ratio_across_pairs": float(pair.changed_gap_mean.sum() / pair.raw_gap_mean.sum()),
            "median_gradient_energy_retained_pct": float(100 * detail.gradient_energy_ratio.median()),
            "median_source_gradient_ncc": float(detail.gradient_ncc.median()),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    best = summary[summary.sigma.gt(0)].sort_values("median_full_uni_gap_ratio_across_pairs").iloc[0]
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "slides": 103,
        "tissue_types": int(cohort.tissue_type.nunique()),
        "locations_per_slide": 3,
        "scanners": list(SCANNERS),
        "sigmas": list(SIGMAS),
        "pca_basis": "locked paired-centered blur/sharpen basis fitted before adding sigma > 3",
        "best_full_uni_convergence_sigma": float(best.sigma),
        "interpretation_limit": "Loss of grayscale gradient energy does not prove loss of all tissue or clinical information",
    }, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
