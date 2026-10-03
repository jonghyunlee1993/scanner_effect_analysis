"""Plot Table 4-consistent scanner-level UNI incremental target gains."""

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "outputs/table4_color_frequency_crossencoder_2026-09-25/summary.csv"
OUTPUT = Path(os.environ.get("PAPER_FIGURES_DIR", str(PROJECT / "00_manuscript/figures"))) / "fig_frequency_incremental_uni_gain.png"

with SOURCE.open(newline="") as handle:
    rows = {
        (row["dataset"], row["scanner"]): row
        for row in csv.DictReader(handle)
        if row["model"] == "uni_v1" and row["metric"] == "incremental_gain"
    }

groups = [
    ("PanNormal", ["all", "akoya", "versa", "s360", "s60", "gt450"]),
    ("PLISM", ["all", "s60", "s360", "gt450"]),
]
colors = ["#27658d", "#b55c28"]

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharex=True)
for ax, (title, keys), color in zip(axes, groups, colors):
    y = np.arange(len(keys))[::-1]
    for pos, key in zip(y, keys):
        row = rows[(title, key)]
        mean = float(row["mean"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        pooled = key == "all"
        ax.errorbar(
            mean,
            pos,
            xerr=[[mean - low], [high - mean]],
            fmt="D" if pooled else "o",
            markersize=5.8 if pooled else 5.0,
            color="black" if pooled else color,
            ecolor="black" if pooled else color,
            elinewidth=1.45,
            capsize=2.5,
            zorder=3,
        )
    ax.axvline(0, color="0.45", linewidth=0.9, zorder=1)
    ax.set_title(title, fontsize=13, loc="left", pad=9)
    ax.set_yticks(y)
    ax.set_yticklabels(["All" if key == "all" else key.upper() for key in keys])
    ax.set_xlim(-0.025, 0.025)
    ax.set_xticks([-0.02, 0, 0.02])
    ax.tick_params(axis="both", labelsize=11, length=3)
    ax.grid(axis="x", color="0.89", linewidth=0.6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("0.6")

fig.supxlabel("Additional decrease in UNI distance", fontsize=12, y=0.035)
fig.subplots_adjust(left=0.14, right=0.98, top=0.87, bottom=0.24, wspace=0.36)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", pad_inches=0.06)
plt.close(fig)
