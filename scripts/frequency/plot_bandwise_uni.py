#!/usr/bin/env python3
"""Create a compact research figure from the completed bandwise UNI experiment."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("outputs/bandwise_uni_experiment")
MANUSCRIPT_FIGURE = Path(os.environ.get("PAPER_FIGURES_DIR", "00_manuscript/figures")) / "fig_uni_band_sensitivity.png"
BANDS = ("low_mid", "mid", "high")
SCANNERS = ("versa", "akoya", "gt450", "s360", "s60")
COLORS = {"low_mid": "#5A7DA5", "mid": "#50A698", "high": "#D46B43"}


def interval(values: np.ndarray, seed: int = 20260924) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
    return float(values.mean()), *np.quantile(sample, [0.025, 0.975]).tolist()


def main() -> None:
    paths = [ROOT / f"fold_{fold}.csv.gz" for fold in range(5)]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("all five folds are required")
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.3), constrained_layout=True)

    for band in BANDS:
        means, lower, upper = [], [], []
        for dose in (0.25, 0.50):
            selected = frame[(frame.band == band) & (frame.dose_fraction == dose)]
            per_slide = selected.groupby("slide_id").embedding_displacement.mean().to_numpy()
            mean, lo, hi = interval(per_slide)
            means.append(mean)
            lower.append(mean - lo)
            upper.append(hi - mean)
        axes[0].errorbar((0.25, 0.50), means, yerr=(lower, upper),
                         color=COLORS[band], marker="o", capsize=3,
                         label={"low_mid": "Low–mid", "mid": "Mid", "high": "High"}[band])
    axes[0].set(xlabel="Dose (fraction of weakest band RMS)",
                ylabel="UNI cosine displacement")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(axis="y", alpha=0.2)

    for boundary in np.arange(len(SCANNERS) - 1) + 0.5:
        axes[1].axvline(boundary, color="#C8CDD3", linestyle="--", linewidth=0.8, zorder=0)
    for scanner_index, scanner in enumerate(SCANNERS):
        for band_index, band in enumerate(BANDS):
            selected = frame[(frame.scanner == scanner) & (frame.band == band)
                             & (frame.dose_fraction == 0.25) & (frame.target_like)]
            per_slide = selected.groupby("slide_id").target_gain.mean().to_numpy()
            mean, lo, hi = interval(per_slide)
            xpos = scanner_index + (band_index - 1) * 0.21
            axes[1].errorbar(xpos, mean, yerr=[[mean - lo], [hi - mean]], fmt="o",
                             color=COLORS[band], capsize=2, markersize=5)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlim(-0.5, len(SCANNERS) - 0.5)
    axes[1].set_xticks(range(len(SCANNERS)), [x.upper() for x in SCANNERS], rotation=30)
    axes[1].set(ylabel="UNI target gain over Reinhard")
    axes[1].grid(axis="y", alpha=0.2)
    MANUSCRIPT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    if "PAPER_FIGURES_DIR" not in os.environ:
        figure.savefig(ROOT / "bandwise_uni.png", dpi=220)
    figure.savefig(MANUSCRIPT_FIGURE, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()
