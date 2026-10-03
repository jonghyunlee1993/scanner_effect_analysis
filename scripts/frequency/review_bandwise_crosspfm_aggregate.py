#!/usr/bin/env python3
"""Audit and plot Figure 1-equivalent panels for additional PFMs."""

from __future__ import annotations

import argparse

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/bandwise_crosspfm_2026-09-25"
FIGURES = Path(os.environ.get("PAPER_FIGURES_DIR", str(ROOT / "00_manuscript/figures")))
MODELS = ("uni2", "virchow2", "hoptimus1")
LABELS = {"uni2": "UNI2-h", "virchow2": "Virchow2",
          "hoptimus1": "H-optimus-1"}
BANDS = ("low_mid", "mid", "high")
SCANNERS = ("versa", "akoya", "gt450", "s360", "s60")
COLORS = {"low_mid": "#5A7DA5", "mid": "#50A698", "high": "#D46B43"}


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), (3000, len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def read_and_audit() -> pd.DataFrame:
    paths = [OUTPUT / "shards" / model / f"fold_{fold}.csv.gz"
             for model in MODELS for fold in range(5)]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError([str(path) for path in paths if not path.is_file()])
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    key = ["model", "slide_id", "location_index", "scanner", "band",
           "dose_fraction", "sign"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate perturbation rows")
    for model in MODELS:
        part = frame[frame.model == model]
        if len(part) != 103 * 3 * len(SCANNERS) * len(BANDS) * 2 * 2:
            raise ValueError(f"{model}: incomplete rows ({len(part)})")
        if part.slide_id.nunique() != 103 or not part.groupby("slide_id").location_index.nunique().eq(3).all():
            raise ValueError(f"{model}: incomplete slide coverage")
        if not np.isfinite(part[["embedding_displacement", "target_gain",
                                 "target_rms_od", "achieved_rms_od"]].to_numpy()).all():
            raise ValueError(f"{model}: nonfinite values")
    frame["dose_relative_error"] = (
        frame.achieved_rms_od / frame.target_rms_od - 1.0
    ).abs()
    dose = frame.groupby(["model", "band", "dose_fraction", "sign"], as_index=False).dose_relative_error.agg(
        median="median", maximum="max"
    )
    if (dose["median"] > 0.02).any():
        raise ValueError("median dose error exceeded 2%")
    dose.to_csv(OUTPUT / "dose_audit.csv", index=False)
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model in MODELS:
        part = frame[frame.model == model]
        for dose in (0.25, 0.50):
            for band in BANDS:
                selected = part[(part.dose_fraction == dose) & (part.band == band)]
                per_slide = selected.groupby("slide_id").embedding_displacement.mean().to_numpy()
                mean, low, high = bootstrap(per_slide, 20260925)
                records.append(dict(model=model, metric="embedding_displacement",
                                    scanner="all", band=band, dose_fraction=dose,
                                    mean=mean, ci_low=low, ci_high=high, slides=len(per_slide)))
        for scanner in SCANNERS:
            for band in BANDS:
                selected = part[(part.scanner == scanner) & (part.band == band)
                                & (part.dose_fraction == 0.25) & (part.target_like)]
                per_slide = selected.groupby("slide_id").target_gain.mean().to_numpy()
                mean, low, high = bootstrap(per_slide, 20260926)
                records.append(dict(model=model, metric="target_gain",
                                    scanner=scanner, band=band, dose_fraction=0.25,
                                    mean=mean, ci_low=low, ci_high=high, slides=len(per_slide)))
    summary = pd.DataFrame(records)
    summary.to_csv(OUTPUT / "figure_summary.csv", index=False)

    wide = frame.pivot(index=["model", "slide_id", "location_index", "scanner",
                               "dose_fraction", "sign"],
                       columns="band", values="embedding_displacement")
    contrasts = []
    for model in MODELS:
        for dose in (0.25, 0.50):
            selected = wide.xs((model, dose), level=("model", "dose_fraction"))
            for comparator in ("low_mid", "mid"):
                per_slide = (selected["high"] - selected[comparator]).groupby(level="slide_id").mean().to_numpy()
                mean, low, high = bootstrap(per_slide, 20260927)
                contrasts.append(dict(model=model, contrast=f"high_minus_{comparator}",
                                      dose_fraction=dose, mean=mean, ci_low=low,
                                      ci_high=high, slides=len(per_slide)))
    pd.DataFrame(contrasts).to_csv(OUTPUT / "sensitivity_contrasts.csv", index=False)
    return summary


def plot(summary: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42})
    FIGURES.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.3), constrained_layout=True)
        panel = summary[summary.model == model]
        for band in BANDS:
            selected = panel[(panel.metric == "embedding_displacement") &
                             (panel.band == band)].sort_values("dose_fraction")
            means = selected["mean"].to_numpy()
            axes[0].errorbar(selected.dose_fraction, means,
                             yerr=(means - selected.ci_low.to_numpy(),
                                   selected.ci_high.to_numpy() - means),
                             color=COLORS[band], marker="o", capsize=3,
                             label={"low_mid": "Low–mid", "mid": "Mid", "high": "High"}[band])
        axes[0].set(xlabel="Dose (fraction of weakest band RMS)",
                    ylabel=f"{LABELS[model]} cosine displacement")
        axes[0].legend(frameon=False, loc="lower right")
        axes[0].grid(axis="y", alpha=0.2)
        for boundary in np.arange(len(SCANNERS) - 1) + 0.5:
            axes[1].axvline(boundary, color="#C8CDD3", linestyle="--",
                            linewidth=0.8, zorder=0)
        for scanner_index, scanner in enumerate(SCANNERS):
            for band_index, band in enumerate(BANDS):
                row = panel[(panel.metric == "target_gain") &
                            (panel.scanner == scanner) & (panel.band == band)].iloc[0]
                x = scanner_index + (band_index - 1) * 0.21
                axes[1].errorbar(x, row["mean"],
                                 yerr=[[row["mean"] - row.ci_low],
                                       [row.ci_high - row["mean"]]],
                                 fmt="o", color=COLORS[band], capsize=2,
                                 markersize=5)
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_xlim(-0.5, len(SCANNERS) - 0.5)
        axes[1].set_xticks(range(len(SCANNERS)), [x.upper() for x in SCANNERS], rotation=30)
        axes[1].set(ylabel=f"{LABELS[model]} target gain over Reinhard")
        axes[1].grid(axis="y", alpha=0.2)
        figure.savefig(FIGURES / f"fig_{model}_band_sensitivity_supp.png", dpi=300)
        if "PAPER_FIGURES_DIR" not in os.environ:
            figure.savefig(OUTPUT / f"{model}_band_sensitivity.png", dpi=180)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="render from the frozen figure summary")
    args = parser.parse_args()
    if args.plot_only:
        plot(pd.read_csv(OUTPUT / "figure_summary.csv"))
        return
    frame = read_and_audit()
    summary = summarize(frame)
    plot(summary)
    print(json.dumps({"rows": len(frame), "models": list(MODELS),
                      "slides_per_model": 103,
                      "figure_summary": str(OUTPUT / "figure_summary.csv")}), flush=True)


if __name__ == "__main__":
    main()
