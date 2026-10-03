#!/usr/bin/env python3
"""Fit one paired scanner PCA and audit six-scanner augmentation paths."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


PROJECT = Path(__file__).resolve().parents[2]
AUGMENTATION = PROJECT / "outputs/augmentation_ood_v1"
OUTPUT = Path(__file__).resolve().parent / "results/augmentation_common_uni_space"
SCANNERS = ("at2", "versa", "akoya", "gt450", "s360", "s60")
OPERATIONS = ("Gaussian blur", "Unsharp mask", "Gamma")
STRENGTHS = ((0., .5, 1., 2., 3.), (0., .25, .5, 1., 2.), (1., .9, .75, .6, .5))


def main() -> None:
    cohort = pd.read_csv(AUGMENTATION / "00_contract/cohort.csv", dtype={"slide_id": str})
    raw_rows = []
    augmented_rows = []
    for slide_id in cohort.slide_id:
        path = OUTPUT / "shards" / f"{slide_id}.h5"
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as store:
            scanner_names = tuple(value.decode() for value in store["scanner_names"][:])
            operation_names = tuple(value.decode() for value in store["operation_names"][:])
            raw = np.asarray(store["raw_features"], dtype=np.float32)
            augmented = np.asarray(store["augmented_features"], dtype=np.float32)
        if scanner_names != SCANNERS or operation_names != OPERATIONS:
            raise ValueError(f"scanner/operation order changed: {slide_id}")
        if raw.shape != (3, 6, 1024) or augmented.shape != (3, 6, 3, 5, 1024):
            raise ValueError(f"incomplete feature shard: {slide_id}")
        if not np.allclose(augmented[:, :, :, 0], raw[:, :, None, :], atol=1e-6):
            raise ValueError(f"augmentation trajectory has inconsistent source: {slide_id}")
        raw_rows.append(raw)
        augmented_rows.append(augmented)
    raw = np.stack(raw_rows)
    augmented = np.stack(augmented_rows)
    center = raw.mean(axis=2, keepdims=True)
    residual_raw = raw - center
    total_pca = PCA(n_components=5, svd_solver="randomized", random_state=20260924)
    total_pca.fit(residual_raw.reshape(-1, 1024))
    # A scanner-only basis captures scanner centroids but misses almost all
    # augmentation displacement. The display basis is therefore fitted to the
    # 90 scanner-by-operation-by-strength population means, then frozen before
    # projecting any individual slide. Retain scanner-only and total-variation
    # PCA diagnostics so a 2D crossing is never treated as full-D agreement.
    real_scanner_means = residual_raw.mean(axis=(0, 1))
    scanner_mean_pca = PCA(n_components=5, svd_solver="full")
    scanner_mean_pca.fit(real_scanner_means)
    residual_aug = augmented - center[:, :, :, None, None, :]
    mean_augmented_states = residual_aug.mean(axis=(0, 1)).reshape(-1, 1024)
    pca = PCA(n_components=5, svd_solver="full")
    pca.fit(mean_augmented_states)
    projected = pca.transform(residual_aug.reshape(-1, 1024))[:, :2]
    projected = projected.reshape(103, 3, 6, 3, 5, 2)
    delta = augmented - augmented[:, :, :, :, 0:1]
    delta_2d = np.tensordot(delta, pca.components_[:2].T, axes=([-1], [0]))
    delta_scanner_span = np.tensordot(delta, scanner_mean_pca.components_.T, axes=([-1], [0]))
    energy = np.maximum(np.square(delta).sum(axis=-1), 1e-12)
    captured = np.square(delta_2d).sum(axis=-1) / np.maximum(
        energy, 1e-12
    )
    captured_scanner_span = np.square(delta_scanner_span).sum(axis=-1) / energy
    capture_rows = []
    for scanner_index, scanner in enumerate(SCANNERS):
        for operation_index, operation in enumerate(OPERATIONS):
            for step in range(1, 5):
                slide_capture = captured[:, :, scanner_index, operation_index, step].mean(axis=1)
                slide_scanner_span = captured_scanner_span[:, :, scanner_index, operation_index, step].mean(axis=1)
                capture_rows.append({
                    "scanner": scanner,
                    "operation": operation,
                    "step": step,
                    "strength": STRENGTHS[operation_index][step],
                    "median_2d_augmentation_energy_fraction": float(np.median(slide_capture)),
                    "mean_2d_augmentation_energy_fraction": float(np.mean(slide_capture)),
                    "median_scanner_mean_5d_span_energy_fraction": float(np.median(slide_scanner_span)),
                })
    pd.DataFrame(capture_rows).to_csv(OUTPUT / "projection_capture.csv", index=False)
    records = []
    for slide_index, slide in enumerate(cohort.itertuples(index=False)):
        for scanner_index, scanner in enumerate(SCANNERS):
            for operation_index, operation in enumerate(OPERATIONS):
                for step, strength in enumerate(STRENGTHS[operation_index]):
                    xy = projected[slide_index, :, scanner_index, operation_index, step].mean(axis=0)
                    records.append({
                        "slide_id": str(slide.slide_id),
                        "tissue_type": str(slide.tissue_type),
                        "scanner": scanner,
                        "operation": operation,
                        "step": step,
                        "strength": strength,
                        "pc1": float(xy[0]),
                        "pc2": float(xy[1]),
                    })
    slides = pd.DataFrame(records)
    if len(slides) != 103 * 6 * 3 * 5:
        raise ValueError("incomplete slide-level projected trajectories")
    slides.to_csv(OUTPUT / "slide_positions.csv", index=False)
    centroids = slides.groupby(["scanner", "operation", "step", "strength"], as_index=False)[["pc1", "pc2"]].mean()
    centroids.to_csv(OUTPUT / "centroid_trajectories.csv", index=False)

    pair_rows = []
    for i, j in combinations(range(6), 2):
        baseline = 1 - np.einsum("sld,sld->sl", raw[:, :, i], raw[:, :, j])
        baseline_slide = baseline.mean(axis=1)
        for operation_index, operation in enumerate(OPERATIONS):
            for step, strength in enumerate(STRENGTHS[operation_index]):
                left = augmented[:, :, i, operation_index, step]
                right = augmented[:, :, j, operation_index, step]
                changed = 1 - np.einsum("sld,sld->sl", left, right)
                changed_slide = changed.mean(axis=1)
                reduction = 100 * (baseline_slide - changed_slide) / baseline_slide
                pair_rows.append({
                    "scanner_a": SCANNERS[i],
                    "scanner_b": SCANNERS[j],
                    "operation": operation,
                    "step": step,
                    "strength": strength,
                    "raw_cosine_distance_mean": float(baseline_slide.mean()),
                    "changed_cosine_distance_mean": float(changed_slide.mean()),
                    "median_paired_gap_reduction_pct": float(np.median(reduction)),
                    "mean_paired_gap_reduction_pct": float(reduction.mean()),
                    "positive_slide_count": int((reduction > 0).sum()),
                    "n_slides": 103,
                })
    pair_gaps = pd.DataFrame(pair_rows)
    pair_gaps.to_csv(OUTPUT / "paired_full_uni_gaps.csv", index=False)

    straight_rows = []
    for scanner_index, scanner in enumerate(SCANNERS):
        for operation_index, operation in enumerate(OPERATIONS):
            trajectory = augmented[:, :, scanner_index, operation_index]
            arc = np.linalg.norm(np.diff(trajectory, axis=2), axis=-1).sum(axis=2)
            chord = np.linalg.norm(trajectory[:, :, -1] - trajectory[:, :, 0], axis=-1)
            straightness = (chord / np.maximum(arc, 1e-9)).mean(axis=1)
            straight_rows.append({
                "scanner": scanner,
                "operation": operation,
                "median_full_uni_path_straightness": float(np.median(straightness)),
                "q25": float(np.quantile(straightness, .25)),
                "q75": float(np.quantile(straightness, .75)),
            })
    pd.DataFrame(straight_rows).to_csv(OUTPUT / "path_straightness.csv", index=False)
    manifest = {
        "slides": 103,
        "tissue_types": int(cohort.tissue_type.nunique()),
        "locations_per_slide": 3,
        "scanners": list(SCANNERS),
        "operations": list(OPERATIONS),
        "pca_fit": "90 scanner-by-augmentation-by-strength population mean UNI signatures after within-location centering; individual slides only projected",
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "first_two_explained_mean_trajectory_variation": float(pca.explained_variance_ratio_[:2].sum()),
        "first_two_explained_between_scanner_means_by_scanner_only_pca": float(scanner_mean_pca.explained_variance_ratio_[:2].sum()),
        "first_two_explained_total_paired_variation": float(total_pca.explained_variance_ratio_[:2].sum()),
        "full_dimension_check": "same-operation, same-strength paired cosine distances between every scanner pair",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    np.savez_compressed(OUTPUT / "locked_scanner_pca.npz",
                        components=pca.components_.astype(np.float32),
                        mean=pca.mean_.astype(np.float32),
                        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32))
    print(json.dumps(manifest, indent=2), flush=True)
    print(centroids.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
