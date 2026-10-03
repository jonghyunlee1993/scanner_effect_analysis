#!/usr/bin/env python3
"""Figure 2 composite: scanner phenotype map (A) and frequency transfer curves (B, C).

A. Colour-high-frequency map, redrawn from the data and encoding of
   `analysis/paper/plot_scanner_common_image_phenotype.py` (PanNormal tissue means, PLISM core
   means, scanner means, same-scanner PanNormal -> PLISM arrows, label offsets); fonts scaled
   to the composite, axes styled as B/C, the Pan VERSA label moved off the PLISM S360 marker,
   and a leader line added from "AT2 reference" to the AT2 origin.
B, C. `panel_curves` and `panel_tissues` from `analysis/revision/frequency_transfer_figures.py`
   (imported, so content is identical to RV12 `fig_transfer_curves.png`); only text sizes are
   unified across panels and the "AT2 (reference)" note in B is moved off the curves. In C, the
   highlighted tissues are, for AKOYA and for GT450, the tissue types with the lowest and the
   highest high-band log2 transfer (`band_summary.csv`, mean over slides) among tissue types with
   at least two slides.

Writes `00_manuscript/figures/fig_scanner_phenotype_transfer.{png,pdf}` and the supplementary
heatmap `fig_tissue_scanner_heatmap_supp.{png,pdf}` (RV12 heatmap without its in-figure title).
No new statistics.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import frequency_transfer_figures as rv12  # noqa: E402

PHENOTYPE = PROJECT / "analysis/paper/results/scanner_color_frequency_space"
FIGURES = Path(os.environ.get("PAPER_FIGURES_DIR", str(PROJECT / "00_manuscript/figures")))
NAME = "fig_scanner_phenotype_transfer"
METRICS = ["delta_e76", "frequency_high"]
# Colours of plot_scanner_common_image_phenotype.py (PLISM-only scanners included).
COLORS = {
    "versa": "#5c88bd", "akoya": "#bf5d5d", "gt450": "#ce9444",
    "s360": "#78a26d", "s60": "#9379b3", "p": "#877f73",
    "s210": "#4c9f9a", "sq": "#c37aab",
}
# Darker shade of the GT450 colour for its highlighted tissue curves in C.
GT450_DARK = "#7a4c10"
# Text sizes shared by all panels (points).
LABEL_SIZE, TICK_SIZE, TEXT_SIZE, NAME_SIZE, LETTER_SIZE = 10.5, 9.5, 9, 10, 15
# (offset in points, ha, va) of each scanner-mean label in panel A: the paper renderer's offsets,
# except Pan VERSA, whose original offset (-25, 22) covered the PLISM S360 marker (it now sits
# below-left of its marker, under the S60 arrow), and Pan S60, nudged right off the S360 arrow.
PAN_LABELS = {"akoya": ((8, 12), "left", "baseline"), "gt450": ((8, 8), "left", "baseline"),
              "s360": ((50, -20), "left", "baseline"), "s60": ((18, -17), "left", "baseline"),
              "versa": ((2, -18), "right", "top")}
PLISM_LABELS = {"gt450": ((8, -20), "left", "baseline"), "s360": ((18, 20), "left", "baseline"),
                "s60": ((-58, -36), "left", "baseline"), "p": ((-20, -58), "left", "baseline"),
                "s210": ((-60, 31), "left", "baseline"), "sq": ((-55, 14), "left", "baseline")}


def panel_phenotype(ax) -> None:
    pan = pd.read_csv(PHENOTYPE / "pannormal_tissue_positions.csv")
    plism = pd.read_csv(PHENOTYPE / "plism_core_positions.csv")
    pan_means = pan.groupby("scanner")[METRICS].mean()
    plism_means = plism.groupby("scanner")[METRICS].mean()
    for frame, marker, size, alpha in ((pan, "o", 15, .24), (plism, "^", 17, .21)):
        for scanner, group in frame.groupby("scanner"):
            ax.scatter(group.delta_e76, group.frequency_high, s=size, marker=marker,
                       color=COLORS[scanner], alpha=alpha, linewidths=0, zorder=2)
    for scanner in ("gt450", "s360", "s60"):
        start, end = pan_means.loc[scanner], plism_means.loc[scanner]
        ax.annotate("", xy=(end.delta_e76, end.frequency_high),
                    xytext=(start.delta_e76, start.frequency_high),
                    arrowprops={"arrowstyle": "->", "color": COLORS[scanner], "lw": 1.5,
                                "linestyle": "--", "alpha": .85}, zorder=3)
    for name, means, marker, size, labels in (("Pan", pan_means, "o", 95, PAN_LABELS),
                                              ("PLISM", plism_means, "^", 105, PLISM_LABELS)):
        for scanner, point in means.iterrows():
            ax.scatter(point.delta_e76, point.frequency_high, s=size, marker=marker,
                       facecolor=COLORS[scanner], edgecolor="white", linewidth=1.2, zorder=5)
            offset, ha, va = labels[scanner]
            text = ax.annotate(f"{name} {scanner.upper()}", (point.delta_e76, point.frequency_high),
                               xytext=offset, textcoords="offset points", ha=ha, va=va,
                               fontsize=NAME_SIZE, fontweight="medium", color=COLORS[scanner], zorder=7)
            text.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
    ax.scatter([0], [0], marker="+", s=60, color="black", linewidth=1.4, zorder=6)
    ax.annotate("AT2 reference", (0, 0), xytext=(0.2, -1.0), textcoords="data", fontsize=TEXT_SIZE,
                color=rv12.INK, ha="left", va="center",
                arrowprops={"arrowstyle": "-", "color": rv12.MUTED, "lw": .6, "shrinkA": 2, "shrinkB": 5,
                             "relpos": (0.03, 1.0)})
    ax.axhline(0, color=".78", linewidth=.8, zorder=0)
    ax.set(xlim=(-1, 41), ylim=(-2.8, 1.2), xlabel="Color difference from AT2 (CIELAB ΔE76)",
           ylabel="High-band log₂ amplitude ratio to AT2")
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="None", markerfacecolor=".45", markeredgecolor="white",
               markersize=8, label="PanNormal tissue mean"),
        Line2D([], [], marker="^", linestyle="None", markerfacecolor=".45", markeredgecolor="white",
               markersize=8, label="PLISM core mean"),
        Line2D([], [], linestyle="--", color=".45", marker=">", markersize=4.5,
               label="Same scanner: PanNormal → PLISM"),
    ], loc="lower left", frameon=True, framealpha=.94, edgecolor=rv12.GRID, fontsize=TEXT_SIZE)
    rv12.style_axis(ax)


def extreme_tissues() -> tuple:
    """Lowest and highest high-band transfer per scanner, among tissue types with >= 2 slides."""
    bands = pd.read_csv(rv12.FREQUENCY / "band_summary.csv", dtype={"slide_id": str})
    high = bands[bands.band == "high"]
    slides = high.groupby("tissue_type").slide_id.nunique()
    high = high[high.tissue_type.isin(slides[slides >= 2].index)]
    means = high.groupby(["scanner", "tissue_type"]).log2_relative_transfer.mean()
    highlights = []
    for scanner, color in (("akoya", rv12.INK), ("gt450", GT450_DARK)):
        ordered = means.loc[scanner].sort_values()
        highlights += [(scanner, ordered.index[0], "-", color), (scanner, ordered.index[-1], (0, (4, 2)), color)]
        print(f"{scanner}: lowest {ordered.index[0]} {ordered.iloc[0]:+.3f}, "
              f"highest {ordered.index[-1]} {ordered.iloc[-1]:+.3f}", flush=True)
    return tuple(highlights)


def unify_text(ax) -> None:
    """Scale the RV12 panel annotations (hard-coded at 8 pt) to the shared text size."""
    for text in ax.texts:
        text.set_fontsize(TEXT_SIZE)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(TEXT_SIZE)


def letter(fig, ax, label: str, x: float) -> None:
    top = ax.get_position().y1
    fig.text(x, top + 0.012, label, fontsize=LETTER_SIZE, fontweight="bold", ha="left", va="bottom")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.labelsize": LABEL_SIZE,
                         "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE, "pdf.fonttype": 42})
    spectra = pd.read_csv(rv12.FREQUENCY / "spectra.csv", dtype={"slide_id": str}).rename(
        columns={"frequency_cyc_per_um_provisional": "frequency"})

    fig = plt.figure(figsize=(10.5, 10.4))
    outer = GridSpec(2, 1, figure=fig, height_ratios=(1.45, 1), hspace=0.26,
                     left=0.085, right=0.915, top=0.96, bottom=0.058)
    ax_a = fig.add_subplot(outer[0])
    bottom = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.2)
    ax_b = fig.add_subplot(bottom[0])
    ax_c = fig.add_subplot(bottom[1], sharey=ax_b)

    panel_phenotype(ax_a)
    rv12.panel_curves(ax_b, spectra)
    rv12.panel_tissues(ax_c, spectra, highlights=extreme_tissues())
    for text in ax_c.texts:
        text.set_text(text.get_text().replace("eye(cornea)", "eye (cornea)"))
    for ax in (ax_b, ax_c):
        unify_text(ax)
    # The RV12 note sits on the low-frequency curves; place it on the zero line at the right end,
    # where only the AT2 line passes between GT450 (above) and VERSA (below).
    for text in ax_b.texts:
        if text.get_text() == "AT2 (reference)":
            text.set_position((0.975, 0.07))
            text.set_horizontalalignment("right")
    ax_b.set_ylabel("log₂ amplitude transfer to AT2")
    ax_c.tick_params(labelleft=False)

    letter(fig, ax_a, "A", 0.012)
    letter(fig, ax_b, "B", 0.012)
    letter(fig, ax_c, "C", ax_c.get_position().x0 - 0.035)

    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{NAME}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {FIGURES / NAME}.png/.pdf", flush=True)

    # Supplementary Fig. S1: the RV12 tissue x scanner heatmap, without the in-figure title (the
    # caption carries it).
    heatmap = rv12.heatmaps(pd.read_csv(rv12.EFFECTS), show_title=False)
    for suffix in ("png", "pdf"):
        heatmap.savefig(FIGURES / f"fig_tissue_scanner_heatmap_supp.{suffix}", dpi=300, bbox_inches="tight",
                        facecolor="white")
    plt.close(heatmap)
    print(f"wrote {FIGURES / 'fig_tissue_scanner_heatmap_supp'}.png/.pdf", flush=True)


if __name__ == "__main__":
    main()
