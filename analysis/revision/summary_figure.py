#!/usr/bin/env python3
"""RV07: image alignment against representation alignment, and representation alignment against
target detectability.

A. PanNormal, every image-level correction x target scanner: reduction in image residual from raw
   (%, the 10 measured image properties) against the representation gain in each of the four PFMs.
   The four PFMs of one correction x scanner share the same image, so they sit on one vertical bar;
   the bar's length shows how differently the PFMs respond to the same image.
B. Every correction x PFM, PanNormal and PLISM: representation gain against best-probe
   detectability of the target scanner (RV04b).

Representation gain = target gain / the PFM's raw between-tissue distance (both pooled over
directions for B), so that PFMs with different distance scales share one axis. Spearman
correlations between image and representation improvement across the 35 correction x scanner
cells are reported per PFM (descriptive). No new embedding or bootstrap computation.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

RESULTS = Path(__file__).resolve().parent / "results"
CELLS = RESULTS / "three_axis_evaluation/cell_statistics.csv"
DETECT = RESULTS / "detectability_within_fold/detectability.csv"
OUTPUT = RESULTS / "summary_figure"
SCANNERS = ("versa", "akoya", "gt450", "s360", "s60")
PFMS = ("uni_v1", "uni2", "virchow2", "hoptimus1")
PFM_LABELS = {"uni_v1": "UNI", "uni2": "UNI2-h", "virchow2": "Virchow2", "hoptimus1": "H-optimus-1"}
PFM_MARKERS = {"uni_v1": "o", "uni2": "s", "virchow2": "^", "hoptimus1": "D"}
IMAGE_METHODS = ("reinhard", "macenko", "vahadane", "pix2pix", "cyclegan", "frequency", "combined")
FEATURE_METHODS = ("ridge", "combat", "ols")
METHOD_LABELS = {"reinhard": "Reinhard", "macenko": "Macenko", "vahadane": "Vahadane", "pix2pix": "Pix2Pix",
                 "cyclegan": "CycleGAN", "frequency": "Frequency", "combined": "Color + frequency",
                 "ridge": "Ridge affine", "combat": "ComBat", "ols": "Affine OLS", "raw": "Raw"}
# Categorical slots 1-7 of the dataviz reference palette (validated: all checks pass on light);
# dark marker edges give the low-contrast slots relief.
METHOD_COLORS = {"reinhard": "#2a78d6", "macenko": "#eb6834", "vahadane": "#1baf7a", "pix2pix": "#eda100",
                 "cyclegan": "#e87ba4", "frequency": "#008300", "combined": "#4a3aa7"}
LEVEL_COLORS = {"raw": "#8a8a8a", "image": "#1f6fb4", "feature": "#c8553d"}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"


def style_axis(ax) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def statistic(cells: pd.DataFrame, name: str) -> pd.Series:
    frame = cells[cells.statistic == name]
    return frame.set_index(["dataset", "pfm", "method", "scanner"]).estimate


def build_cells() -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = pd.read_csv(CELLS)
    gain = statistic(cells, "target_gain")
    between = statistic(cells, "between_tissue_distance")
    residual = statistic(cells, "image_image_residual")

    rows = []
    for method in IMAGE_METHODS:
        for scanner in SCANNERS:
            raw_residual = residual[("pannormal", "uni_v1", "raw", scanner)]
            image_gain = 100 * (raw_residual - residual[("pannormal", "uni_v1", method, scanner)]) / raw_residual
            for pfm in PFMS:
                scale = between[("pannormal", pfm, "raw", "pooled")]
                rows.append({"method": method, "scanner": scanner, "pfm": pfm,
                             "image_residual_reduction_pct": image_gain,
                             "representation_gain_pct": 100 * gain[("pannormal", pfm, method, scanner)] / scale})
    panel_a = pd.DataFrame(rows)

    detect = pd.read_csv(DETECT)
    detect = detect[detect.statistic.isin(["linear", "mlp", "knn"])]
    best = detect.groupby(["dataset", "pfm", "method"]).estimate.max()
    rows = []
    for dataset in ("pannormal", "plism"):
        for pfm in PFMS:
            scale = between[(dataset, pfm, "raw", "pooled")]
            for method in ("raw",) + IMAGE_METHODS + FEATURE_METHODS:
                key = (dataset, pfm, method, "pooled")
                if key not in gain.index or (dataset, pfm, method) not in best.index:
                    continue
                level = "raw" if method == "raw" else ("feature" if method in FEATURE_METHODS else "image")
                rows.append({"dataset": dataset, "pfm": pfm, "method": method, "level": level,
                             "representation_gain_pct": 100 * gain[key] / scale,
                             "detectability_best": best[(dataset, pfm, method)]})
    return panel_a, pd.DataFrame(rows)


def correlations(panel_a: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pfm in PFMS:
        frame = panel_a[panel_a.pfm == pfm]
        rho, p = spearmanr(frame.image_residual_reduction_pct, frame.representation_gain_pct)
        rows.append({"pfm": pfm, "n_cells": len(frame), "spearman_rho": rho, "p_value": p})
    spread = panel_a.groupby(["method", "scanner"]).representation_gain_pct.agg(lambda x: x.max() - x.min())
    sign = panel_a.groupby(["method", "scanner"]).representation_gain_pct.agg(lambda x: (x > 0).any() and (x < 0).any())
    rows.append({"pfm": "all (cells with opposite signs across PFMs)", "n_cells": int(sign.sum()),
                 "spearman_rho": np.nan, "p_value": np.nan})
    rows.append({"pfm": "all (median PFM range, pct points)", "n_cells": len(spread),
                 "spearman_rho": float(spread.median()), "p_value": np.nan})
    return pd.DataFrame(rows)


def draw(panel_a: pd.DataFrame, panel_b: pd.DataFrame, corr: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "legend.fontsize": 8})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.6, 5.4), gridspec_kw={"width_ratios": [1.35, 1]})

    style_axis(ax_a)
    ax_a.axhline(0, color=INK, linewidth=0.8, zorder=1)
    ax_a.axvline(0, color=MUTED, linewidth=0.6, linestyle=(0, (3, 3)), zorder=1)
    for (method, scanner), frame in panel_a.groupby(["method", "scanner"]):
        x = frame.image_residual_reduction_pct.iloc[0]
        color = METHOD_COLORS[method]
        ax_a.vlines(x, frame.representation_gain_pct.min(), frame.representation_gain_pct.max(),
                    color=color, linewidth=1.6, alpha=0.55, zorder=2)
        for row in frame.itertuples():
            ax_a.scatter(x, row.representation_gain_pct, s=34, marker=PFM_MARKERS[row.pfm], color=color,
                         edgecolor=INK, linewidth=0.5, zorder=3)
    for method, scanner, text, dx, dy in (("combined", "akoya", "Color + frequency, AKOYA", -4, 6),
                                          ("pix2pix", "gt450", "Pix2Pix, GT450", -4, -6),
                                          ("macenko", "akoya", "Macenko, AKOYA", 4, -6)):
        frame = panel_a[(panel_a.method == method) & (panel_a.scanner == scanner)]
        x = frame.image_residual_reduction_pct.iloc[0]
        y = frame.representation_gain_pct.max() if dy > 0 else frame.representation_gain_pct.min()
        ax_a.annotate(text, (x, y), xytext=(x + dx, y + dy), fontsize=8, color=INK,
                      ha="right" if dx < 0 else "left", va="center",
                      arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6})
    rho_text = "\n".join(f"{PFM_LABELS[r.pfm]}: ρ = {r.spearman_rho:+.2f}"
                         for r in corr[corr.pfm.isin(PFMS)].itertuples())
    ax_a.text(0.02, 0.03, "Spearman, image vs representation\n(35 correction × scanner cells)\n" + rho_text,
              transform=ax_a.transAxes, fontsize=8, color=INK, va="bottom",
              bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": GRID})
    ax_a.set_xlabel("Image residual reduction from raw (%)\n(closer to the target in 10 measured image properties →)")
    ax_a.set_ylabel("Representation gain (% of between-tissue distance)\n(closer to the real target →)")
    method_handles = [Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=METHOD_COLORS[m],
                             markeredgecolor=INK, markeredgewidth=0.5, label=METHOD_LABELS[m]) for m in IMAGE_METHODS]
    pfm_handles = [Line2D([], [], marker=PFM_MARKERS[p], linestyle="", markersize=6, markerfacecolor="white",
                          markeredgecolor=INK, label=PFM_LABELS[p]) for p in PFMS]
    first = ax_a.legend(handles=method_handles, title="Correction", loc="upper left", frameon=False,
                        title_fontsize=8.5)
    ax_a.add_artist(first)
    ax_a.legend(handles=pfm_handles, title="PFM", loc="upper left", bbox_to_anchor=(0.25, 1.0), frameon=False,
                title_fontsize=8.5)
    ax_a.text(-0.1, 1.02, "A", transform=ax_a.transAxes, fontsize=13, fontweight="bold", va="bottom")

    style_axis(ax_b)
    ax_b.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax_b.axvline(0, color=MUTED, linewidth=0.6, linestyle=(0, (3, 3)), zorder=1)
    for row in panel_b.itertuples():
        color = LEVEL_COLORS[row.level]
        filled = row.dataset == "pannormal"
        ax_b.scatter(row.representation_gain_pct, row.detectability_best, s=36, marker=PFM_MARKERS[row.pfm],
                     facecolor=color if filled else "white", edgecolor=color if not filled else INK,
                     linewidth=1.1 if not filled else 0.5, zorder=3)
    ax_b.set_ylim(0.45, 1.03)
    ax_b.set_xlim(panel_b.representation_gain_pct.min() - 4, panel_b.representation_gain_pct.max() + 3)
    ax_b.text(ax_b.get_xlim()[0], 0.5, " not detectable", color=MUTED, fontsize=8, va="bottom")
    ax_b.set_xlabel("Representation gain (% of between-tissue distance)")
    ax_b.set_ylabel("Target-scanner detectability (best of three probes)")
    level_handles = [Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=LEVEL_COLORS[k],
                            markeredgecolor=INK, markeredgewidth=0.5, label=label)
                     for k, label in (("raw", "Raw"), ("image", "Image-level"), ("feature", "Feature-level"))]
    data_handles = [Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor="#555555",
                           markeredgecolor=INK, label="PanNormal"),
                    Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor="white",
                           markeredgecolor="#555555", markeredgewidth=1.1, label="PLISM (fitted in PanNormal)")]
    ax_b.legend(handles=level_handles + data_handles, loc="center left", bbox_to_anchor=(0.0, 0.42), frameon=False)
    feature = panel_b[(panel_b.dataset == "pannormal") & (panel_b.level == "feature")]
    ax_b.text(feature.representation_gain_pct.median(), feature.detectability_best.min() - 0.02,
              "Feature-level, PanNormal", fontsize=8, color=INK, ha="center", va="top")
    ax_b.text(ax_b.get_xlim()[0] + 1, 1.012, "PLISM: every method", fontsize=8, color=INK, va="bottom")
    ax_b.text(-0.12, 1.02, "B", transform=ax_b.transAxes, fontsize=13, fontweight="bold", va="bottom")

    fig.tight_layout(w_pad=3)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"fig_image_vs_representation.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel_a, panel_b = build_cells()
    corr = correlations(panel_a)
    panel_a.to_csv(OUTPUT / "panel_a_cells.csv", index=False)
    panel_b.to_csv(OUTPUT / "panel_b_cells.csv", index=False)
    corr.to_csv(OUTPUT / "correlations.csv", index=False)
    draw(panel_a, panel_b, corr)

    lines = ["# RV07 image alignment vs representation alignment", "",
             "Representation gain = target gain / raw between-tissue distance of the PFM (pooled).",
             "Image residual reduction = (raw − corrected) / raw image residual per scanner (PanNormal).", "",
             "| PFM | cells | Spearman ρ | p |", "| --- | --- | --- | --- |"]
    for row in corr.itertuples():
        lines.append(f"| {row.pfm} | {row.n_cells} | {row.spearman_rho:.3f} | {row.p_value:.3g} |")
    b = panel_b.copy()
    summary = b.groupby(["dataset", "level"]).agg(gain_min=("representation_gain_pct", "min"),
                                                  gain_max=("representation_gain_pct", "max"),
                                                  detect_min=("detectability_best", "min"),
                                                  detect_max=("detectability_best", "max")).round(3)
    lines += ["", "Panel B ranges by dataset and correction level:", "",
              "| dataset | level | gain min | gain max | detectability min | detectability max |",
              "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {d} | {lvl} | {r.gain_min} | {r.gain_max} | {r.detect_min} | {r.detect_max} |"
              for (d, lvl), r in summary.iterrows()]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUTPUT / "qc.json").write_text(json.dumps({"panel_a_rows": len(panel_a), "panel_b_rows": len(panel_b),
                                                "panel_a_missing": int(panel_a.isna().any(axis=1).sum())}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
