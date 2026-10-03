#!/usr/bin/env python3
"""Band-sensitivity figures on set20 (UNI, UNI2-h, Virchow2, H-optimus-1).

Main figure: one panel per PFM, side by side, with the cosine displacement against dose for the
low-mid, mid and high bands (both signs, all scanners; slide means, bootstrap 95% CI). Each panel
starts at zero so that the relative weighting of the bands can be compared across PFMs whose
distance scales differ.

Supplementary figure: one panel per PFM with the target-direction gain over Reinhard by scanner
and band at dose 0.25, dashed scanner separators.

Same encodings as the former per-PFM figures (`scripts/frequency/plot_bandwise_uni.py`,
`scripts/frequency/review_bandwise_crosspfm_aggregate.py`), now regrouped by panel type. Values are
read from RV02 `analysis/revision/results/band_manipulation_set20/summary_statistics.csv`
(`set == "set20"`: 103 slides x 20 locations, five scanners equally weighted, 2,000 slide bootstrap
resamples, seed 20260924). No new statistics.

Writes `00_manuscript/figures/fig_band_sensitivity_pfms.{png,pdf}` and
`fig_band_target_gain_supp.{png,pdf}`.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
SUMMARY = PROJECT / "analysis/revision/results/band_manipulation_set20/summary_statistics.csv"
FIGURES = Path(os.environ.get("PAPER_FIGURES_DIR", str(PROJECT / "00_manuscript/figures")))
BANDS = ("low_mid", "mid", "high")
BAND_LABELS = {"low_mid": "Low–mid", "mid": "Mid", "high": "High"}
SCANNERS = ("versa", "akoya", "gt450", "s360", "s60")
COLORS = {"low_mid": "#5A7DA5", "mid": "#50A698", "high": "#D46B43"}
DOSES = (0.25, 0.50)
MODELS = {"uni_v1": "UNI", "uni2": "UNI2-h", "virchow2": "Virchow2", "hoptimus1": "H-optimus-1"}
FONT_SIZE = 11


def pick(summary: pd.DataFrame, **query) -> pd.DataFrame:
    mask = np.ones(len(summary), dtype=bool)
    for key, value in query.items():
        mask &= summary[key].eq(value).to_numpy()
    return summary[mask]


def letter(ax, label: str) -> None:
    ax.text(-0.02, 1.04, label, transform=ax.transAxes, fontsize=FONT_SIZE + 4, fontweight="bold",
            ha="right", va="bottom")


def band_legend(**kwargs):
    handles = [Line2D([], [], color=COLORS[b], marker="o", markersize=5, label=BAND_LABELS[b]) for b in BANDS]
    return dict(handles=handles, frameon=False, title="Band", title_fontsize=FONT_SIZE - 1,
                fontsize=FONT_SIZE - 1, **kwargs)


def displacement(summary: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, len(MODELS), figsize=(10.2, 3.2))
    figure.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.2, wspace=0.45)
    for ax, (model, name), label in zip(axes, MODELS.items(), "ABCD"):
        top = 0.0
        for band in BANDS:
            rows = pick(summary, model=model, statistic=f"shift_{band}", subset="both_signs",
                        scanner="all").sort_values("dose_fraction")
            if tuple(rows.dose_fraction) != DOSES:
                raise ValueError(f"{model} {band}: doses {tuple(rows.dose_fraction)}")
            means = rows["mean"].to_numpy()
            ax.errorbar(rows.dose_fraction, means,
                        yerr=(means - rows.ci_low.to_numpy(), rows.ci_high.to_numpy() - means),
                        color=COLORS[band], marker="o", markersize=4.5, capsize=2.5, linewidth=1.4)
            top = max(top, float(rows.ci_high.max()))
        ax.set_ylim(0, top * 1.12)
        ax.set_xlim(0.2, 0.55)
        ax.set_xticks(DOSES, ["0.25", "0.50"])
        ax.set_title(name, fontsize=FONT_SIZE + 1, pad=6)
        ax.grid(axis="y", alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        letter(ax, label)
    axes[0].set_ylabel("Cosine displacement")
    axes[0].legend(**band_legend(loc="upper left"))
    figure.supxlabel("Dose (fraction of the weakest band's OD RMS)", fontsize=FONT_SIZE, y=0.02)
    save(figure, "fig_band_sensitivity_pfms")


def target_gain(summary: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.6, 6.6))
    figure.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.08, hspace=0.42, wspace=0.3)
    for ax, (model, name), label in zip(axes.flat, MODELS.items(), "ABCD"):
        for boundary in np.arange(len(SCANNERS) - 1) + 0.5:
            ax.axvline(boundary, color="#C8CDD3", linestyle="--", linewidth=0.8, zorder=0)
        for scanner_index, scanner in enumerate(SCANNERS):
            for band_index, band in enumerate(BANDS):
                rows = pick(summary, model=model, statistic=f"target_gain_{band}", dose_fraction=0.25,
                            subset="target_direction", scanner=scanner)
                if len(rows) != 1:
                    raise ValueError(f"{model} {scanner} {band}: {len(rows)} rows")
                row = rows.iloc[0]
                x = scanner_index + (band_index - 1) * 0.21
                ax.errorbar(x, row["mean"], yerr=[[row["mean"] - row.ci_low], [row.ci_high - row["mean"]]],
                            fmt="o", color=COLORS[band], capsize=2, markersize=4.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlim(-0.5, len(SCANNERS) - 0.5)
        ax.set_xticks(range(len(SCANNERS)), [s.upper() for s in SCANNERS])
        ax.set_title(name, fontsize=FONT_SIZE + 1, pad=6)
        ax.grid(axis="y", alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        letter(ax, label)
    for ax in axes[:, 0]:
        ax.set_ylabel("Target gain over Reinhard")
    figure.legend(**band_legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0)))
    save(figure, "fig_band_target_gain_supp")


def save(figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"{name}.{suffix}", dpi=300, facecolor="white")
    plt.close(figure)
    print(f"wrote {FIGURES / name}.png/.pdf", flush=True)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT_SIZE, "pdf.fonttype": 42})
    summary = pd.read_csv(SUMMARY)
    summary = summary[summary["set"].eq("set20")]
    if summary.slides.ne(103).any():
        raise ValueError("set20 summary is not over 103 slides")
    displacement(summary)
    target_gain(summary)


if __name__ == "__main__":
    main()
