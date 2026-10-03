#!/usr/bin/env python3
"""Compare raw and paired-centred UNI displays for blur/sharpen trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


PROJECT = Path(__file__).resolve().parents[2]
AUGMENTATION = PROJECT / "outputs/augmentation_ood_v1"
OUTPUT = Path(__file__).resolve().parent / "results/augmentation_common_uni_space"
SCANNERS = ("at2", "versa", "akoya", "gt450", "s360", "s60")
OPERATIONS = ("Gaussian blur", "Unsharp mask")
STRENGTHS = ((0., .5, 1., 2., 3.), (0., .25, .5, 1., 2.))
ALL_OPERATIONS = ("Gaussian blur", "Unsharp mask", "Gamma")
ALL_STRENGTHS = (*STRENGTHS, (1., .9, .75, .6, .5))
COLORS = {
    "at2": "#525252", "versa": "#7552a3", "akoya": "#ce4470",
    "gt450": "#2086b0", "s360": "#e19a23", "s60": "#23896c",
}
MARKERS = {"Gaussian blur": "o", "Unsharp mask": "s"}
LABEL_OFFSETS = {
    "at2": (-30, 9), "versa": (6, -15), "akoya": (-46, -15),
    "gt450": (6, 8), "s360": (6, -15), "s60": (6, 8),
}


def plot_one(centroids: pd.DataFrame, path: Path, explained: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9.3, 7.3), dpi=170)
    for scanner in SCANNERS:
        for operation in OPERATIONS:
            group = centroids[
                centroids.scanner.eq(scanner) & centroids.operation.eq(operation)
            ].sort_values("step")
            x, y = group.pc1.to_numpy(float), group.pc2.to_numpy(float)
            ax.plot(x, y, color=COLORS[scanner], linewidth=1.1, alpha=.37, zorder=2)
            for step in range(1, 5):
                ax.scatter(x[step], y[step], marker=MARKERS[operation],
                           s=(36, 50, 70, 93)[step - 1], color=COLORS[scanner],
                           alpha=(.42, .59, .78, 1)[step - 1],
                           edgecolor="white", linewidth=.55, zorder=3 + step)
            ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]),
                        arrowprops={"arrowstyle": "->", "lw": 1.1,
                                    "alpha": .6, "color": COLORS[scanner]}, zorder=7)
        start = centroids[
            centroids.scanner.eq(scanner) & centroids.operation.eq("Gaussian blur")
            & centroids.step.eq(0)
        ].iloc[0]
        ax.scatter(start.pc1, start.pc2, marker="X", s=140,
                   color=COLORS[scanner], edgecolor="black", linewidth=.7, zorder=9)
        ax.annotate(scanner.upper(), (start.pc1, start.pc2),
                    xytext=LABEL_OFFSETS[scanner], textcoords="offset points",
                    color=COLORS[scanner], fontsize=10, weight="bold", zorder=10)
    ax.axhline(0, color=".84", linewidth=.7, zorder=0)
    ax.axvline(0, color=".84", linewidth=.7, zorder=0)
    ax.grid(color=".92", linewidth=.6, zorder=0)
    ax.set_xlabel(f"UNI PC1 ({100 * explained[0]:.1f}%)")
    ax.set_ylabel(f"UNI PC2 ({100 * explained[1]:.1f}%)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color=".45", markerfacecolor=".45",
               markersize=8, label="Blur"),
        Line2D([0], [0], marker="s", color=".45", markerfacecolor=".45",
               markersize=8, label="Sharpen"),
    ], loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    cohort = pd.read_csv(AUGMENTATION / "00_contract/cohort.csv", dtype={"slide_id": str})
    raw_parts, aug_parts = [], []
    for slide_id in cohort.slide_id:
        with h5py.File(OUTPUT / "shards" / f"{slide_id}.h5", "r") as store:
            raw_parts.append(np.asarray(store["raw_features"], dtype=np.float32))
            aug_parts.append(np.asarray(store["augmented_features"], dtype=np.float32))
    raw = np.stack(raw_parts)
    all_augmented = np.stack(aug_parts)
    augmented = all_augmented[:, :, :, :2]
    if raw.shape != (103, 3, 6, 1024) or all_augmented.shape != (103, 3, 6, 3, 5, 1024):
        raise ValueError("incomplete features")
    paired_mean = raw.mean(axis=2, keepdims=True)
    centered = augmented - paired_mean[:, :, :, None, None, :]
    mean_states = centered.mean(axis=(0, 1)).reshape(60, 1024)
    models = {
        "paired_centered": PCA(n_components=5, svd_solver="full").fit(mean_states),
        "raw_uni": PCA(n_components=5, svd_solver="randomized", random_state=20260924).fit(raw.reshape(-1, 1024)),
    }
    np.savez_compressed(
        OUTPUT / "paired_centered_blur_sharp_pca.npz",
        components=models["paired_centered"].components_.astype(np.float32),
        mean=models["paired_centered"].mean_.astype(np.float32),
        explained_variance_ratio=models["paired_centered"].explained_variance_ratio_.astype(np.float32),
    )
    diagnostics = {}
    for name, pca in models.items():
        values = centered if name == "paired_centered" else augmented
        xy = pca.transform(values.reshape(-1, 1024))[:, :2].reshape(103, 3, 6, 2, 5, 2)
        records = []
        for slide_index, slide in enumerate(cohort.itertuples(index=False)):
            for scanner_index, scanner in enumerate(SCANNERS):
                for operation_index, operation in enumerate(OPERATIONS):
                    for step, strength in enumerate(STRENGTHS[operation_index]):
                        point = xy[slide_index, :, scanner_index, operation_index, step].mean(axis=0)
                        records.append({
                            "slide_id": str(slide.slide_id), "scanner": scanner,
                            "operation": operation, "step": step, "strength": strength,
                            "pc1": point[0], "pc2": point[1],
                        })
        slides = pd.DataFrame(records)
        centroids = slides.groupby(["scanner", "operation", "step", "strength"], as_index=False)[["pc1", "pc2"]].mean()
        slides.to_csv(OUTPUT / f"{name}_blur_sharp_slide_positions.csv", index=False)
        centroids.to_csv(OUTPUT / f"{name}_blur_sharp_centroids.csv", index=False)
        plot_one(centroids, OUTPUT / f"{name}_blur_sharp_scatter", pca.explained_variance_ratio_)
        raw_xy = xy[:, :, :, 0, 0].reshape(-1, 6, 2)
        scanner_mean = raw_xy.mean(axis=0)
        grand_mean = scanner_mean.mean(axis=0)
        between = np.square(scanner_mean - grand_mean).sum(axis=1).mean()
        within = np.square(raw_xy - scanner_mean[None]).sum(axis=2).mean()
        diagnostics[name] = {
            "pca_fit": "60 augmented scanner mean states after paired-centering" if name == "paired_centered" else "1854 real uncentered scanner/location embeddings",
            "first_two_explained_variance_fraction": float(pca.explained_variance_ratio_[:2].sum()),
            "scanner_centroid_variance_to_within_scanner_variance_ratio_in_2d": float(between / within),
            "between_scanner_centroid_variance_2d": float(between),
            "within_scanner_variance_2d": float(within),
        }
    # Compare how far the fixed one-parameter sweeps move the full embedding.
    movement = 1 - np.einsum("slcofd,slcd->slcof", all_augmented, raw)
    movement_rows = []
    for operation_index, operation in enumerate(ALL_OPERATIONS):
        for step in range(1, 5):
            per_slide = movement[:, :, :, operation_index, step].mean(axis=(1, 2))
            movement_rows.append({
                "operation": operation,
                "strength": ALL_STRENGTHS[operation_index][step],
                "step": step,
                "median_full_uni_cosine_distance_from_source": float(np.median(per_slide)),
                "q25": float(np.quantile(per_slide, .25)),
                "q75": float(np.quantile(per_slide, .75)),
            })
    pd.DataFrame(movement_rows).to_csv(OUTPUT / "all_operator_full_uni_displacement.csv", index=False)
    delta = (all_augmented - raw[:, :, :, None, None, :]).mean(axis=(0, 1))
    alignment_rows = []
    for operation_index, operation in enumerate(ALL_OPERATIONS):
        for step in range(1, 5):
            directions = delta[:, operation_index, step]
            directions = directions / np.maximum(
                np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
            )
            similarity = directions @ directions.T
            pair_values = similarity[np.triu_indices(6, k=1)]
            alignment_rows.append({
                "comparison": "same_operation_across_scanners",
                "operation": operation,
                "strength": ALL_STRENGTHS[operation_index][step],
                "scanner": "all_pairs",
                "median_direction_cosine": float(np.median(pair_values)),
                "min_direction_cosine": float(pair_values.min()),
                "max_direction_cosine": float(pair_values.max()),
            })
    for blur_step, sharpen_step in ((2, 2), (3, 3), (4, 4)):
        for scanner_index, scanner in enumerate(SCANNERS):
            left = delta[scanner_index, 0, blur_step]
            right = delta[scanner_index, 1, sharpen_step]
            alignment_rows.append({
                "comparison": "blur_vs_sharpen_within_scanner",
                "operation": "blur/sharpen",
                "strength": f"sigma={STRENGTHS[0][blur_step]};alpha={STRENGTHS[1][sharpen_step]}",
                "scanner": scanner,
                "median_direction_cosine": float(left @ right / (
                    np.linalg.norm(left) * np.linalg.norm(right)
                )),
                "min_direction_cosine": np.nan,
                "max_direction_cosine": np.nan,
            })
    pd.DataFrame(alignment_rows).to_csv(OUTPUT / "augmentation_direction_alignment.csv", index=False)
    (OUTPUT / "blur_sharp_space_comparison.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
