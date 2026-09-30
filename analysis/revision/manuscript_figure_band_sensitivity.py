#!/usr/bin/env python3
"""Figure 5 and Supplementary band figures on set20 (UNI v1, UNI2-h, Virchow2, H-optimus-1).

Same design as `scripts/frequency/plot_bandwise_uni.py` (UNI) and
`scripts/frequency/review_bandwise_crosspfm_aggregate.py` (other PFMs):
A. cosine displacement against dose for the low-mid, mid and high bands (both signs, all
   scanners; slide means, bootstrap 95% CI);
B. target-direction gain over Reinhard by scanner and band at dose 0.25, dashed scanner separators.
Values are read from RV02 `analysis/revision/results/band_manipulation_set20/summary_statistics.csv`
(`set == "set20"`: 103 slides x 20 locations, five scanners equally weighted, 2,000 slide
bootstrap resamples, seed 20260924). Panel letters are drawn here (previously added in PowerPoint).

Writes `00_manuscript/figures/fig_uni_band_sensitivity_set20.{png,pdf}` and
`fig_{uni2,virchow2,hoptimus1}_band_sensitivity_set20_supp.{png,pdf}`.
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

PROJECT = Path(__file__).resolve().parents[2]
SUMMARY = PROJECT / "analysis/revision/results/band_manipulation_set20/summary_statistics.csv"
FIGURES = Path(os.environ.get("PAPER_FIGURES_DIR", str(PROJECT / "00_manuscript/figures")))
BANDS = ("low_mid", "mid", "high")
BAND_LABELS = {"low_mid": "Low–mid", "mid": "Mid", "high": "High"}
SCANNERS = ("versa", "akoya", "gt450", "s360", "s60")
COLORS = {"low_mid": "#5A7DA5", "mid": "#50A698", "high": "#D46B43"}
DOSES = (0.25, 0.50)
# model -> (axis label, output name, base font size of the paper renderer)
MODELS = {
    "uni_v1": ("UNI", "fig_uni_band_sensitivity_set20", 12),
    "uni2": ("UNI2-h", "fig_uni2_band_sensitivity_set20_supp", 11),
    "virchow2": ("Virchow2", "fig_virchow2_band_sensitivity_set20_supp", 11),
    "hoptimus1": ("H-optimus-1", "fig_hoptimus1_band_sensitivity_set20_supp", 11),
}


def pick(summary: pd.DataFrame, **query) -> pd.DataFrame:
    mask = np.ones(len(summary), dtype=bool)
    for key, value in query.items():
        mask &= summary[key].eq(value).to_numpy()
    return summary[mask]


def render(summary: pd.DataFrame, model: str) -> None:
    label, name, font_size = MODELS[model]
    plt.rcParams.update({"font.size": font_size, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.3), constrained_layout=True)
    figure.get_layout_engine().set(w_pad=0.06)

    for band in BANDS:
        rows = pick(summary, model=model, statistic=f"shift_{band}", subset="both_signs",
                    scanner="all").sort_values("dose_fraction")
        if tuple(rows.dose_fraction) != DOSES:
            raise ValueError(f"{model} {band}: doses {tuple(rows.dose_fraction)}")
        means = rows["mean"].to_numpy()
        axes[0].errorbar(rows.dose_fraction, means,
                         yerr=(means - rows.ci_low.to_numpy(), rows.ci_high.to_numpy() - means),
                         color=COLORS[band], marker="o", capsize=3, label=BAND_LABELS[band])
    axes[0].set(xlabel="Dose (fraction of weakest band RMS)", ylabel=f"{label} cosine displacement")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(axis="y", alpha=0.2)

    for boundary in np.arange(len(SCANNERS) - 1) + 0.5:
        axes[1].axvline(boundary, color="#C8CDD3", linestyle="--", linewidth=0.8, zorder=0)
    for scanner_index, scanner in enumerate(SCANNERS):
        for band_index, band in enumerate(BANDS):
            rows = pick(summary, model=model, statistic=f"target_gain_{band}", dose_fraction=0.25,
                        subset="target_direction", scanner=scanner)
            if len(rows) != 1:
                raise ValueError(f"{model} {scanner} {band}: {len(rows)} rows")
            row = rows.iloc[0]
            x = scanner_index + (band_index - 1) * 0.21
            axes[1].errorbar(x, row["mean"], yerr=[[row["mean"] - row.ci_low], [row.ci_high - row["mean"]]],
                             fmt="o", color=COLORS[band], capsize=2, markersize=5)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlim(-0.5, len(SCANNERS) - 0.5)
    axes[1].set_xticks(range(len(SCANNERS)), [s.upper() for s in SCANNERS], rotation=30)
    axes[1].set(ylabel=f"{label} target gain over Reinhard")
    axes[1].grid(axis="y", alpha=0.2)

    for ax, letter in zip(axes, "AB"):
        ax.set_title(letter, loc="left", fontsize=font_size + 6, fontweight="bold", x=-0.2, y=1.0, pad=6)

    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"{name}.{suffix}", dpi=300, facecolor="white")
    plt.close(figure)
    print(f"wrote {FIGURES / name}.png/.pdf", flush=True)


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    summary = summary[summary["set"].eq("set20")]
    if summary.slides.ne(103).any():
        raise ValueError("set20 summary is not over 103 slides")
    for model in MODELS:
        render(summary, model)


if __name__ == "__main__":
    main()
