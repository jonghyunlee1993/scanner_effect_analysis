#!/usr/bin/env python3
"""RV12: frequency transfer curves and the tissue x scanner map.

A. log2 amplitude transfer to AT2 against spatial frequency, one curve per scanner (mean of
   tissue-type means; ribbon = interquartile range across tissues), analysis bands shaded and
   the AT2 Nyquist frequency marked.
B. every tissue's curve for AKOYA and GT450, with aorta and liver highlighted.
C. tissue x scanner heatmaps of the tissue-specific scanner effects m(t, s) from RV11.
Supplementary: coherence to AT2 against frequency.

Also checks that band averages of the curves reproduce `band_summary.csv`. No new statistics.
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
from matplotlib.colors import TwoSlopeNorm

PROJECT = Path(__file__).resolve().parents[2]
FREQUENCY = PROJECT / "outputs/final_image_study_v1/03_frequency"
EFFECTS = Path(__file__).resolve().parent / "results/scanner_tissue_interaction/tissue_scanner_effects.csv"
OUTPUT = Path(__file__).resolve().parent / "results/frequency_transfer_figures"
SCANNERS = ("versa", "akoya", "gt450", "s360", "s60")
LABELS = {"versa": "VERSA", "akoya": "AKOYA", "gt450": "GT450", "s360": "S360", "s60": "S60", "at2": "AT2"}
# Paper colours (analysis/paper/plot_scanner_common_image_phenotype.py); line style is the
# secondary encoding for the GT450/S360 pair, which the colours alone do not separate.
COLORS = {"versa": "#5c88bd", "akoya": "#bf5d5d", "gt450": "#ce9444", "s360": "#78a26d", "s60": "#9379b3"}
STYLES = {"versa": "-", "akoya": "-", "gt450": "-", "s360": (0, (5, 2)), "s60": (0, (1.5, 1.5))}
BANDS = (("low–mid", 0.10, 0.30), ("mid", 0.30, 0.60), ("high", 0.60, 0.90))
NYQUIST = 1 / (2 * 0.5052)
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
HEATMAP_MEASURES = (("delta_lab_a", "Δ a* (green → red)"), ("log2_od_sd_ratio", "log₂ OD contrast ratio"),
                    ("frequency_low_mid", "log₂ low–mid transfer"), ("frequency_high", "log₂ high transfer"))


def style_axis(ax) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def shade_bands(ax, label_y: float) -> None:
    for index, (name, low, high) in enumerate(BANDS):
        ax.axvspan(low, high, color="#f1f1f1" if index % 2 == 0 else "#e8e8e8", zorder=0, linewidth=0)
        ax.text((low + high) / 2, label_y, name, ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.axvline(NYQUIST, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))


def tissue_means(spectra: pd.DataFrame, value: str) -> pd.DataFrame:
    return (spectra.groupby(["scanner", "tissue_type", "frequency"], as_index=False)[value].mean())


def band_check(spectra: pd.DataFrame) -> dict:
    bands = pd.read_csv(FREQUENCY / "band_summary.csv", dtype={"slide_id": str})
    result = {}
    for name, low, high in BANDS:
        key = {"low–mid": "low_mid", "mid": "mid", "high": "high"}[name]
        inside = spectra[(spectra.frequency >= low) & (spectra.frequency < high)]
        mean_log = inside.groupby(["slide_id", "scanner"]).log2_relative_transfer.mean()
        amp = inside.assign(amplitude=np.sqrt(inside.radial_power))
        at2 = amp[amp.scanner == "at2"].set_index(["slide_id", "frequency"]).amplitude
        amp = amp[amp.scanner != "at2"].copy()
        amp["at2"] = at2.reindex(pd.MultiIndex.from_frame(amp[["slide_id", "frequency"]])).to_numpy()
        ratio = amp.groupby(["slide_id", "scanner"]).apply(lambda g: np.log2(g.amplitude.sum() / g.at2.sum()))
        reference = bands[bands.band == key].set_index(["slide_id", "scanner"]).log2_relative_transfer
        joined = pd.DataFrame({"reference": reference, "mean_log2": mean_log, "log2_amplitude_ratio": ratio}).dropna()
        result[key] = {"n": int(len(joined)),
                       "max_abs_diff_mean_log2": float((joined.mean_log2 - joined.reference).abs().max()),
                       "max_abs_diff_log2_amplitude_ratio": float((joined.log2_amplitude_ratio - joined.reference).abs().max()),
                       "corr_mean_log2": float(joined[["mean_log2", "reference"]].corr().iloc[0, 1])}
    return result


def panel_curves(ax, spectra: pd.DataFrame) -> None:
    means = tissue_means(spectra[spectra.scanner != "at2"], "log2_relative_transfer")
    summary = means.groupby(["scanner", "frequency"]).log2_relative_transfer.agg(
        mean="mean", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)).reset_index()
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.text(0.015, 0.06, "AT2 (reference)", fontsize=8, color=INK, va="bottom")
    shade_bands(ax, 1.25)
    ends = []
    for scanner in SCANNERS:
        part = summary[summary.scanner == scanner]
        ax.fill_between(part.frequency, part.q25, part.q75, color=COLORS[scanner], alpha=0.18, linewidth=0)
        ax.plot(part.frequency, part["mean"], color=COLORS[scanner], linestyle=STYLES[scanner], linewidth=2,
                label=LABELS[scanner])
        ends.append((float(part["mean"].iloc[-1]), scanner))
    # Direct labels at the right end, nudged apart so they do not collide.
    placed = []
    for y, scanner in sorted(ends):
        y_label = y if not placed else max(y, placed[-1] + 0.14)
        placed.append(y_label)
        ax.text(1.005, y_label, LABELS[scanner], fontsize=8, color=INK, va="center", ha="left",
                transform=ax.get_yaxis_transform())
    ax.set(xlim=(0, 1.0), ylim=(-2.9, 1.45), xlabel="Spatial frequency (cycles/µm)",
           ylabel="log₂ amplitude transfer to AT2")
    ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)
    style_axis(ax)


def panel_tissues(ax, spectra: pd.DataFrame,
                  highlights=(("akoya", "aorta", "-", INK), ("akoya", "liver", (0, (4, 2)), INK))) -> None:
    """Every tissue curve for AKOYA and GT450; ``highlights`` = (scanner, tissue, linestyle, colour)."""
    means = tissue_means(spectra[spectra.scanner.isin(["akoya", "gt450"])], "log2_relative_transfer")
    ax.axhline(0, color=INK, linewidth=0.9)
    shade_bands(ax, 1.25)
    for scanner in ("akoya", "gt450"):
        for tissue, part in means[means.scanner == scanner].groupby("tissue_type"):
            ax.plot(part.frequency, part.log2_relative_transfer, color=COLORS[scanner], linewidth=0.7, alpha=0.35)
    for scanner, tissue, style, color in highlights:
        part = means[(means.scanner == scanner) & (means.tissue_type == tissue)]
        ax.plot(part.frequency, part.log2_relative_transfer, color=color, linewidth=1.6, linestyle=style)
        ax.text(1.005, float(part.log2_relative_transfer.iloc[-1]), f"{LABELS[scanner]},\n{tissue}", fontsize=8,
                color=color, va="center", ha="left", transform=ax.get_yaxis_transform())
    ax.text(0.03, 0.55, "GT450: 37 tissues", fontsize=8, color=INK)
    ax.text(0.03, -1.9, "AKOYA: 37 tissues", fontsize=8, color=INK)
    ax.set(xlim=(0, 1.0), ylim=(-2.9, 1.45), xlabel="Spatial frequency (cycles/µm)")
    style_axis(ax)


def heatmaps(effects: pd.DataFrame, show_title: bool = True) -> plt.Figure:
    order = (effects[(effects.endpoint == "frequency_high") & (effects.scanner == "akoya")]
             .sort_values("effect").tissue_type.tolist())
    fig, axes = plt.subplots(1, len(HEATMAP_MEASURES), figsize=(12, 9.5), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, (endpoint, title) in zip(axes, HEATMAP_MEASURES):
        wide = (effects[effects.endpoint == endpoint].pivot(index="tissue_type", columns="scanner", values="effect")
                .reindex(index=order, columns=list(SCANNERS)))
        limit = float(np.nanquantile(np.abs(wide.to_numpy()), 0.98))
        image = ax.imshow(wide.to_numpy(), aspect="auto", cmap="RdBu_r",
                          norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit), interpolation="nearest")
        ax.set_xticks(range(len(SCANNERS)), [LABELS[s] for s in SCANNERS], rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9, color=INK)
        ax.tick_params(length=0, labelcolor=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.035, pad=0.09)
        bar.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK)
        bar.outline.set_visible(False)
    axes[0].set_yticks(range(len(order)), [t.replace("(", " (") for t in order], fontsize=7)
    if show_title:
        fig.suptitle("Tissue-specific scanner effects relative to AT2 (rows ordered by AKOYA high-frequency transfer)",
                     fontsize=10, color=INK, y=0.93)
    return fig


def coherence(spectra: pd.DataFrame) -> plt.Figure:
    means = tissue_means(spectra[spectra.scanner != "at2"], "coherence_to_at2")
    summary = means.groupby(["scanner", "frequency"]).coherence_to_at2.mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 4))
    shade_bands(ax, 1.005)
    for scanner in SCANNERS:
        part = summary[summary.scanner == scanner]
        ax.plot(part.frequency, part.coherence_to_at2, color=COLORS[scanner], linestyle=STYLES[scanner],
                linewidth=2, label=LABELS[scanner])
    ax.set(xlim=(0, 1.0), ylim=(0, 1.05), xlabel="Spatial frequency (cycles/µm)", ylabel="Coherence with AT2")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style_axis(ax)
    fig.tight_layout()
    return fig


def save(fig: plt.Figure, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{name}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 10, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    spectra = pd.read_csv(FREQUENCY / "spectra.csv", dtype={"slide_id": str}).rename(
        columns={"frequency_cyc_per_um_provisional": "frequency"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True, gridspec_kw={"wspace": 0.2})
    panel_curves(left, spectra)
    panel_tissues(right, spectra)
    left.set_title("A  Scanner transfer curves", loc="left", fontsize=10)
    right.set_title("B  Tissue curves: AKOYA and GT450", loc="left", fontsize=10)
    save(fig, "fig_transfer_curves")
    save(heatmaps(pd.read_csv(EFFECTS)), "fig_tissue_scanner_heatmap")
    save(coherence(spectra), "fig_coherence_supp")
    check = band_check(spectra)
    (OUTPUT / "band_check.json").write_text(json.dumps(check, indent=2) + "\n")
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
