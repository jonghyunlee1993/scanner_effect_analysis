#!/usr/bin/env python3
"""Figure 3 on set20: UNI PCA trajectories of six scanners under blur and sharpening.

Same design as `analysis/paper/plot_augmentation_strong_blur.py` (paper figure
`fig_augmentation_uni_trajectories.png`), drawn from the RV01 set20 outputs
(`analysis/revision/results/blur_sharpen_set20/pca_centroids.csv`, `pca_set20.npz`; 103 slides x
20 locations, paired-centred PCA). Displayed strengths as in the paper figure: blur sigma = 0, .5,
1, 3, 6 and unsharp alpha = 0, .25, .5, 1, 2 (sigma = 2 is in the PCA fit but, as in the paper,
not drawn). Scanner colours follow `analysis/paper/plot_scanner_common_image_phenotype.py`.

Insets: the paper's example AT2 image (slide 2-8_6, location_index 20, brain/cerebellum; see
`analysis/paper/results/augmentation_strong_blur/inset_image_selection.json`), which is one of
the set20 locations; blur and sharpening use `prenorm.augmentation.transform`. Its position in
the set20 display is recorded in `results/manuscript_figures/augmentation_inset_selection.json`.

Writes `00_manuscript/figures/fig_augmentation_uni_trajectories_set20.{png,pdf}`.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea, VPacker  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from prenorm.augmentation import transform  # noqa: E402

RV01 = PROJECT / "analysis/revision/results/blur_sharpen_set20"
COHORT = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17/00_contract/cohort.csv"
SET20 = PROJECT / "analysis/revision/results/location_sets/set20.csv"
PAPER_SELECTION = PROJECT / "analysis/paper/results/augmentation_strong_blur/inset_image_selection.json"
OUTPUT = Path(__file__).resolve().parent / "results/manuscript_figures"
FIGURES = Path(os.environ.get("PAPER_FIGURES_DIR", str(PROJECT / "00_manuscript/figures")))
NAME = "fig_augmentation_uni_trajectories_set20"
SCANNERS = ("at2", "versa", "akoya", "gt450", "s360", "s60")
COLORS = {"at2": "#525252", "versa": "#5c88bd", "akoya": "#bf5d5d",
          "gt450": "#ce9444", "s360": "#78a26d", "s60": "#9379b3"}
LABEL_OFFSETS = {"at2": (-30, 9), "versa": (8, 7), "akoya": (-45, -13),
                 "gt450": (8, -13), "s360": (6, -15), "s60": (-35, -6)}
BLUR_DISPLAY = (0., .5, 1., 3., 6.)
SHARPEN_DISPLAY = (0., .25, .5, 1., 2.)
FINAL_SIGMA, FINAL_ALPHA = 6., 2.
SIZES_BLUR, SIZES_SHARP = (35, 51, 70, 100), (34, 49, 69, 93)
ALPHAS = (.43, .60, .78, 1.)


def load_inset() -> tuple[np.ndarray, dict]:
    paper = json.loads(PAPER_SELECTION.read_text())
    slide_id, location = paper["slide_id"], int(paper["location_index"])
    set20 = pd.read_csv(SET20, dtype={"slide_id": str})
    if not set20[set20.slide_id.eq(slide_id)].location_index.eq(location).any():
        raise ValueError("paper inset location is not in set20")
    cohort = pd.read_csv(COHORT, dtype={"slide_id": str}).set_index("slide_id")
    with h5py.File(str(cohort.loc[slide_id, "cache_path"]), "r") as store:
        scanners = tuple(v.decode().lower() for v in store["scanner_names"][:])
        source = np.asarray(store["images"][location, scanners.index("at2")], dtype=np.uint8)
    # Position of this location's sigma = 6 state in the set20 display (paired-centred, as RV01).
    pca = np.load(RV01 / "pca_set20.npz")
    with h5py.File(RV01 / "shards" / f"{slide_id}.h5", "r") as store:
        index = int(np.flatnonzero(np.asarray(store["location_index"]) == location)[0])
        sigmas = tuple(float(v) for v in store["blur_sigmas"][:])
        raw = np.asarray(store["raw_features"][index], dtype=np.float64)
        final = np.asarray(store["blur_features"][index, 0, sigmas.index(FINAL_SIGMA)], dtype=np.float64)
    point = (final - raw.mean(axis=0) - pca["mean"]) @ pca["components"][:2].T
    centroids = pd.read_csv(RV01 / "pca_centroids.csv")
    target = centroids[centroids.scanner.eq("at2") & centroids.operation.eq("Gaussian blur")
                       & centroids.strength.eq(FINAL_SIGMA)][["pc1", "pc2"]].iloc[0].to_numpy(float)
    metadata = {"scanner": "AT2", "slide_id": slide_id, "location_index": location,
                "tissue_type": paper["tissue_type"],
                "source": "paper inset location (analysis/paper/results/augmentation_strong_blur/inset_image_selection.json)",
                "sigma6_position_set20_display": [float(v) for v in point],
                "at2_sigma6_centroid_set20_display": [float(v) for v in target],
                "distance_to_at2_sigma6_centroid_set20_display": float(np.linalg.norm(point - target))}
    return source, metadata


def main() -> None:
    centroids = pd.read_csv(RV01 / "pca_centroids.csv")
    blur = centroids[centroids.operation.eq("Gaussian blur") & centroids.strength.isin(BLUR_DISPLAY)]
    sharpen = centroids[centroids.operation.eq("Unsharp mask") & centroids.strength.isin(SHARPEN_DISPLAY)]
    if len(blur) != 6 * len(BLUR_DISPLAY) or len(sharpen) != 6 * len(SHARPEN_DISPLAY):
        raise ValueError("incomplete trajectories")
    variance = np.asarray(np.load(RV01 / "pca_set20.npz")["explained_variance_ratio"])
    source, selection = load_inset()
    blurred = transform(source, "blur_sigma", FINAL_SIGMA)
    sharpened = transform(source, "sharpen_amount", FINAL_ALPHA)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "augmentation_inset_selection.json").write_text(json.dumps(selection, indent=2) + "\n")

    plt.rcParams.update({"pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10.2, 7.6), dpi=180)
    for scanner in SCANNERS:
        color = COLORS[scanner]
        for frame, marker, sizes, width in ((blur, "o", SIZES_BLUR, .6), (sharpen, "s", SIZES_SHARP, .55)):
            track = frame[frame.scanner.eq(scanner)].sort_values("strength")
            x, y = track.pc1.to_numpy(float), track.pc2.to_numpy(float)
            ax.plot(x, y, color=color, linewidth=1.15, alpha=.40, zorder=2)
            for step in range(1, 5):
                ax.scatter(x[step], y[step], marker=marker, s=sizes[step - 1], color=color,
                           alpha=ALPHAS[step - 1], edgecolor="white", linewidth=width, zorder=3 + step)
            ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]),
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 1., "alpha": .55}, zorder=6)
        start = blur[blur.scanner.eq(scanner) & blur.strength.eq(0.)].iloc[0]
        ax.scatter(start.pc1, start.pc2, marker="X", s=145, color=color, edgecolor="black",
                   linewidth=.7, zorder=10)
        ax.annotate(scanner.upper(), (start.pc1, start.pc2), xytext=LABEL_OFFSETS[scanner],
                    textcoords="offset points", color=color, fontsize=10, weight="bold", zorder=11)

    at2_blur = blur[blur.scanner.eq("at2")].set_index("strength")
    at2_sharp = sharpen[sharpen.scanner.eq("at2")].set_index("strength").loc[FINAL_ALPHA]
    insets = []
    for image, label, point, image_center in (
        (source, "Original AT2", at2_blur.loc[0.], (.40, -.23)),
        (blurred, "Blur σ = 6", at2_blur.loc[FINAL_SIGMA], (-.41, -.23)),
        (sharpened, "Sharpen α = 2", at2_sharp, (.40, .17)),
    ):
        content = VPacker(children=[
            TextArea(label, textprops={"size": 9, "weight": "bold", "color": "#303030"}),
            OffsetImage(image, zoom=.43),
        ], align="center", pad=0, sep=4)
        box = AnnotationBbox(content, image_center, xycoords="data", frameon=True,
                             bboxprops={"boxstyle": "round,pad=.16", "fc": "white", "ec": ".38", "lw": .85},
                             zorder=14)
        ax.add_artist(box)
        insets.append((box, np.array([point.pc1, point.pc2], dtype=float)))
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
               label="Blur: σ = 0, 0.5, 1, 3, 6"),
        Line2D([0], [0], marker="s", color=".45", markerfacecolor=".45", markersize=8,
               label="Sharpen: α = 0, 0.25, 0.5, 1, 2"),
    ], loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    # Inset arrows are drawn as ordinary annotations from the box edge: with matplotlib 3.9 an
    # explicit AnnotationBbox mutation_scale is dpi-scaled twice, inflating the PNG arrowheads.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_data = ax.transData.inverted()
    for box, target in insets:
        extent = box.get_window_extent(renderer)
        (x0, y0), (x1, y1) = to_data.transform([[extent.x0, extent.y0], [extent.x1, extent.y1]])
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        direction = target - center
        with np.errstate(divide="ignore"):
            scale = min((x1 - x0) / 2 / abs(direction[0]), (y1 - y0) / 2 / abs(direction[1]))
        start = center + scale * direction
        ax.annotate("", xy=tuple(target), xytext=tuple(start), xycoords="data", textcoords="data",
                    arrowprops={"arrowstyle": "->", "color": ".30", "lw": 1.05, "shrinkA": 3,
                                "shrinkB": 7, "mutation_scale": 10}, zorder=9.5)  # under the X markers
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{NAME}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
