#!/usr/bin/env python3
"""Manuscript figures from RV14: frequency sensitivity against scanner robustness.

Main figure: panels A and B of `results/sensitivity_robustness/fig_sensitivity_robustness`,
re-rendered from the saved RV14 outputs (`model_table.csv`, `correlations.csv`) with the same
encodings, offsets and correlation boxes (`sensitivity_robustness.scatter`). Panel C (rho by band)
is omitted; those correlations are reported in the Supplement.

Supplementary figure: `results/sensitivity_robustness/fig_tissue_vs_scanner_supp` with the same
data and encodings, American spelling and the bold panel letters used by the other figures.
No new statistics.

Writes `00_manuscript/figures/fig_sensitivity_robustness.{png,pdf}` and
`fig_tissue_vs_scanner_supp.{png,pdf}`.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(HERE))
import sensitivity_robustness as rv14  # noqa: E402

FIGURES = Path(os.environ.get("PAPER_FIGURES_DIR", str(PROJECT / "00_manuscript/figures")))
NAME = "fig_sensitivity_robustness"
X = "normalized_shift_high_d0.25"
XLABEL = "High-frequency sensitivity\n(shift per equal OD change / between-tissue distance)"


def main() -> None:
    table = pd.read_csv(rv14.OUTPUT / "model_table.csv").set_index("model")
    corr = pd.read_csv(rv14.OUTPUT / "correlations.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(10.2, 4.4))
    grid = fig.add_gridspec(1, 2, wspace=0.26)

    ax = fig.add_subplot(grid[0])
    rv14.scatter(ax, table, X, "robustness_index", rv14.lookup(corr, X, "robustness_index"),
                 {"hoptimus1": (6, -7, "left"), "seal_conch_pre": (6, -8, "left"), "uni2": (8, -9, "left"),
                  "exaonepath_raw": (8, 2, "left"), "plip": (6, 6, "left")})
    ax.axhline(rv14.RI_CHANCE, color=rv14.MUTED, linestyle=(0, (3, 3)), linewidth=0.8)
    ax.text(ax.get_xlim()[0], rv14.RI_CHANCE, " chance", fontsize=7, color=rv14.MUTED, ha="left", va="bottom")
    ax.set(xlabel=XLABEL, ylabel="PathoROB robustness index (higher = more robust)")
    ax.set_title("A", loc="left", fontsize=11, fontweight="bold")

    ax = fig.add_subplot(grid[1])
    rv14.scatter(ax, table, X, "normalized_distance", rv14.lookup(corr, X, "normalized_distance"),
                 {"hoptimus1": (0, 14, "center"), "uni2": (-6, 8, "right"), "seal_uni2": (6, -8, "left"),
                  "exaonepath_raw": (8, -6, "left"), "exaonepath": (-8, 0, "right"), "virchow2": (8, -5, "left"),
                  "conch_pre": (8, -9, "left"), "seal_conch_pre": (-6, 24, "center"), "dinov2": (12, 12, "left")})
    ax.set(xlabel=XLABEL, ylabel="Normalized scanner distance (lower = more robust)")
    ax.set_title("B", loc="left", fontsize=11, fontweight="bold")

    handles = [plt.Line2D([], [], marker="o", linestyle="None", markerfacecolor=rv14.BLUE, markeredgecolor="white",
                          markersize=8, label="Image-only pretraining"),
               plt.Line2D([], [], marker="o", linestyle="None", markerfacecolor=rv14.ORANGE, markeredgecolor="white",
                          markersize=8, label="Image + text or ST anchor"),
               plt.Line2D([], [], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor=rv14.GREY,
                          markersize=8, label="Reference (not in ρ)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.09))

    for ax in fig.axes:  # typographic minus in the correlation boxes
        for text in ax.texts:
            text.set_text(re.sub(r"(?<![A-Za-z])-(?=\d)", "\u2212", text.get_text()))
    save(fig, NAME)
    supplementary(table)


def supplementary(table: pd.DataFrame) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.62})
    offsets = {"conch_pre": (-10, -8, "right"), "seal_conch_pre": (-10, 8, "right"), "uni2": (6, 6, "left"),
               "hoptimus1": (2, 16, "right"), "virchow2": (8, -2, "left"),
               "seal_uni2": (6, -7, "left"), "exaonepath_raw": (6, -8, "left")}
    for name in rv14.WSI + rv14.REFERENCE:
        row = table.loc[name]
        reference = name in rv14.REFERENCE
        color = rv14.GREY if reference else (rv14.ORANGE if name in rv14.ANCHORED else rv14.BLUE)
        left.scatter(row["tissue_retrieval"], row["normalized_distance"], s=64, zorder=3,
                     facecolor="white" if reference else color, edgecolor=color if reference else "white",
                     linewidth=1.6 if reference else 1.0)
        dx, dy, ha = offsets.get(name, (6, 4, "left"))
        weight = "bold" if name in ("plip", "conch_pre") else "normal"
        arrow = {"arrowstyle": "-", "color": rv14.MUTED, "linewidth": 0.6} if max(abs(dx), abs(dy)) > 9 else None
        left.annotate(rv14.LABELS[name], (row["tissue_retrieval"], row["normalized_distance"]), xytext=(dx, dy),
                      textcoords="offset points", fontsize=8, fontweight=weight,
                      color=rv14.MUTED if reference else rv14.INK, ha=ha, va="center", arrowprops=arrow)
    left.set(xlabel="Tissue information (same-tissue retrieval, macro recall)",
             ylabel="Normalized scanner distance")
    rv14.style_axis(left)
    order = table.sort_values("robustness_index", ascending=False).index.tolist()
    positions = np.arange(len(order))
    right.barh(positions - 0.2, table.loc[order, "so_rate"], height=0.38, color=rv14.BLUE,
               label="Same tissue, other scanner (SO)")
    right.barh(positions + 0.2, table.loc[order, "os_rate"], height=0.38, color=rv14.ORANGE,
               label="Other tissue, same scanner (OS)")
    right.set_yticks(positions, [f"{rv14.LABELS[n].replace(chr(10), ' ')}  (RI {table.loc[n, 'robustness_index']:.2f})"
                                 for n in order])
    right.invert_yaxis()
    right.set(xlabel="Fraction of the k nearest neighbors")
    right.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    rv14.style_axis(right)
    for ax, letter in ((left, "A"), (right, "B")):
        ax.set_title(letter, loc="left", fontsize=11, fontweight="bold")
    save(fig, "fig_tissue_vs_scanner_supp")


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {FIGURES / name}.png/.pdf", flush=True)


if __name__ == "__main__":
    main()
