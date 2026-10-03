#!/usr/bin/env python3
"""Locked UNI scatter with matched original, blur, and sharpen examples."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea, VPacker
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
AUGMENTATION = PROJECT / "outputs/augmentation_ood_v1"
COMMON = Path(__file__).resolve().parent / "results/augmentation_common_uni_space"
OUTPUT = Path(__file__).resolve().parent / "results/augmentation_strong_blur"
WRITE_DIR = Path(os.environ.get("PAPER_PLOT_STAGING_DIR", str(OUTPUT)))
SCANNERS = ("at2", "versa", "akoya", "gt450", "s360", "s60")
COLORS = {
    "at2": "#525252", "versa": "#7552a3", "akoya": "#ce4470",
    "gt450": "#2086b0", "s360": "#e19a23", "s60": "#23896c",
}
LABEL_OFFSETS = {
    "at2": (-30, 9), "versa": (6, -15), "akoya": (-45, -13),
    "gt450": (6, 8), "s360": (6, -15), "s60": (6, 8),
}
BLUR_LEVELS = (0., .5, 1., 3., 6.)
FINAL_SIGMA = 6.
STRONG_SIGMAS = (0., 3., 6., 12., 24., 48.)


def choose_representative_image(centroids: pd.DataFrame) -> tuple[np.ndarray, dict]:
    locked = np.load(COMMON / "paired_centered_blur_sharp_pca.npz")
    components = np.asarray(locked["components"][:2], dtype=np.float32)
    center = np.asarray(locked["mean"], dtype=np.float32)
    target = centroids[
        centroids.scanner.eq("at2") & centroids.sigma.eq(FINAL_SIGMA)
    ][["pc1", "pc2"]].iloc[0].to_numpy(float)
    cohort = pd.read_csv(AUGMENTATION / "00_contract/cohort.csv", dtype={"slide_id": str})
    cohort_index = cohort.set_index("slide_id")
    candidates = []
    for slide_id in cohort.slide_id:
        with h5py.File(OUTPUT / "shards" / f"{slide_id}.h5", "r") as store:
            locations = np.asarray(store["location_index"], dtype=int)
            raw = np.asarray(store["features"][:, :, 0], dtype=np.float32)
            final = np.asarray(store["features"][:, 0, STRONG_SIGMAS.index(FINAL_SIGMA)], dtype=np.float32)
        paired_center = raw.mean(axis=1)
        projected = (final - paired_center - center) @ components.T
        for index, point in enumerate(projected):
            candidates.append((float(np.linalg.norm(point - target)), str(slide_id), int(locations[index])))
    nearest = sorted(candidates)[:20]
    # Among points close to the plotted centroid, retain visible original
    # morphology so the loss of structure in the inset is legible.
    scored = []
    for distance, slide_id, location in nearest:
        with h5py.File(str(cohort_index.loc[slide_id, "cache_path"]), "r") as store:
            source = np.asarray(store["images"][location, 0], dtype=np.uint8)
        gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gy, gx = np.gradient(gray)
        energy = float(np.square(gx).mean() + np.square(gy).mean())
        scored.append((energy, distance, slide_id, location, source))
    _, distance, slide_id, location, source = max(scored, key=lambda x: x[0])
    metadata = {
        "scanner": "AT2",
        "sigma": FINAL_SIGMA,
        "slide_id": slide_id,
        "location_index": location,
        "tissue_type": str(cohort_index.loc[slide_id, "tissue_type"]),
        "distance_to_at2_final_centroid_in_display": distance,
        "selection": "highest source gradient energy among 20 images closest to the AT2 sigma-6 2D centroid",
    }
    return source, metadata


def main() -> None:
    strong_blur = pd.read_csv(OUTPUT / "centroid_positions.csv")
    original_blur = pd.read_csv(COMMON / "paired_centered_blur_sharp_centroids.csv")
    original_blur = original_blur[original_blur.operation.eq("Gaussian blur")].rename(columns={"strength": "sigma"})
    blur = pd.concat([
        original_blur[original_blur.sigma.isin(BLUR_LEVELS[:-1])][["scanner", "sigma", "pc1", "pc2"]],
        strong_blur[strong_blur.sigma.eq(FINAL_SIGMA)][["scanner", "sigma", "pc1", "pc2"]],
    ], ignore_index=True)
    if len(blur) != 6 * len(BLUR_LEVELS):
        raise ValueError("incomplete five-level blur trajectory")
    sharpen = pd.read_csv(COMMON / "paired_centered_blur_sharp_centroids.csv")
    sharpen = sharpen[sharpen.operation.eq("Unsharp mask")]
    locked = np.load(COMMON / "paired_centered_blur_sharp_pca.npz")
    variance = np.asarray(locked["explained_variance_ratio"])
    source, selection = choose_representative_image(blur)
    blurred = cv2.GaussianBlur(source, (0, 0), FINAL_SIGMA)
    rgb = source.astype(np.float32) / 255.0
    lowpass = cv2.GaussianBlur(rgb, (0, 0), 1.0)
    sharpened = np.clip(np.rint(np.clip(rgb + 2.0 * (rgb - lowpass), 0, 1) * 255), 0, 255).astype(np.uint8)
    WRITE_DIR.mkdir(parents=True, exist_ok=True)
    (WRITE_DIR / "inset_image_selection.json").write_text(json.dumps(selection, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(10.2, 7.6), dpi=180)
    for scanner in SCANNERS:
        color = COLORS[scanner]
        track = blur[blur.scanner.eq(scanner) & blur.sigma.isin(BLUR_LEVELS)].sort_values("sigma")
        x, y = track.pc1.to_numpy(float), track.pc2.to_numpy(float)
        ax.plot(x, y, color=color, linewidth=1.15, alpha=.40, zorder=2)
        for step in range(1, 5):
            ax.scatter(x[step], y[step], marker="o", s=(35, 51, 70, 100)[step - 1],
                       color=color, alpha=(.43, .60, .78, 1.)[step - 1],
                       edgecolor="white", linewidth=.6, zorder=3 + step)
        ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]),
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1., "alpha": .55}, zorder=6)
        other = sharpen[sharpen.scanner.eq(scanner)].sort_values("step")
        sx, sy = other.pc1.to_numpy(float), other.pc2.to_numpy(float)
        ax.plot(sx, sy, color=color, linewidth=1.1, alpha=.36, zorder=2)
        for step in range(1, 5):
            ax.scatter(sx[step], sy[step], marker="s", s=(34, 49, 69, 93)[step - 1],
                       color=color, alpha=(.43, .60, .78, 1.)[step - 1],
                       edgecolor="white", linewidth=.55, zorder=3 + step)
        ax.annotate("", xy=(sx[-1], sy[-1]), xytext=(sx[-2], sy[-2]),
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1., "alpha": .55}, zorder=6)
        ax.scatter(x[0], y[0], marker="X", s=145, color=color,
                   edgecolor="black", linewidth=.7, zorder=10)
        ax.annotate(scanner.upper(), (x[0], y[0]), xytext=LABEL_OFFSETS[scanner],
                    textcoords="offset points", color=color, fontsize=10,
                    weight="bold", zorder=11)

    at2_blur = blur[blur.scanner.eq("at2")].set_index("sigma")
    at2_sharp = sharpen[sharpen.scanner.eq("at2")].sort_values("step").iloc[-1]
    for image, label, point, image_center in (
        (source, "Original AT2", at2_blur.loc[0.], (.40, -.23)),
        (blurred, "Blur σ = 6", at2_blur.loc[FINAL_SIGMA], (-.41, -.23)),
        (sharpened, "Sharpen α = 2", at2_sharp, (.40, .17)),
    ):
        content = VPacker(children=[
            TextArea(label, textprops={"size": 9, "weight": "bold", "color": "#303030"}),
            OffsetImage(image, zoom=.43),
        ], align="center", pad=0, sep=4)
        inset = AnnotationBbox(
            content, (point.pc1, point.pc2), xybox=image_center,
            xycoords="data", boxcoords="data",
            arrowprops={"arrowstyle": "->", "color": ".30", "lw": 1.05,
                        "shrinkA": 3, "shrinkB": 4, "mutation_scale": 10},
            frameon=True,
            bboxprops={"boxstyle": "round,pad=.16", "fc": "white", "ec": ".38", "lw": .85},
            zorder=14,
        )
        ax.add_artist(inset)
    ax.axhline(0, color=".85", lw=.7, zorder=0)
    ax.axvline(0, color=".85", lw=.7, zorder=0)
    ax.grid(color=".92", lw=.6, zorder=0)
    ax.set_xlim(-.55, .54)
    ax.set_ylim(-.44, .35)
    ax.set_xlabel(f"UNI PC1 ({100 * variance[0]:.1f}%)")
    ax.set_ylabel(f"UNI PC2 ({100 * variance[1]:.1f}%)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color=".45", markerfacecolor=".45", markersize=8,
               label="Blur: σ = 0, .5, 1, 3, 6"),
        Line2D([0], [0], marker="s", color=".45", markerfacecolor=".45", markersize=8,
               label="Sharpen: α = 0, .25, .5, 1, 2"),
    ], loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(WRITE_DIR / "blur_sharp_extreme_uni_scatter.png", dpi=220)
    fig.savefig(WRITE_DIR / "blur_sharp_extreme_uni_scatter.pdf")
    paper_figures = Path(os.environ.get("PAPER_FIGURES_DIR", str(PROJECT / "00_manuscript/figures")))
    paper_figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_figures / "fig_augmentation_uni_trajectories.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
