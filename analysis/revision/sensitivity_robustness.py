#!/usr/bin/env python3
"""RV14: perturbation sensitivity and scanner robustness across PFMs.

For every model, RV13's ``model_draws`` gives each statistic on the shared 2,000 slide
resamples. Across the eight WSI-pretrained variants, this script correlates the normalized
representation shift caused by equal-OD band manipulations (low-mid, mid, high; and the
high / low-mid ratio) with robustness to real scanner differences (PathoROB RI, normalized
scanner distance, best-probe detectability):

* Spearman rho across models, with a 95% CI from recomputing rho in every slide resample;
* exact permutation p over the 8! model orderings of the point estimates;
* partial Spearman controlling for between-tissue distance.

Figures: the main figure (sensitivity vs RI and vs normalized scanner distance; rho by band and
for the ratio) and a supplementary figure (tissue information vs scanner sensitivity for all
variants, and the SO / OS neighbour rates behind RI).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from anchor_frequency_diversity_metrics import model_draws  # noqa: E402
from corrected_embedding_reader import bootstrap_weights, load_cohort  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
TASKS = RESULTS / "anchor_frequency_diversity/tasks"
OUTPUT = RESULTS / "sensitivity_robustness"
WSI = ("uni_v1", "uni2", "virchow2", "hoptimus1", "exaonepath", "seal_uni2", "conch_pre", "seal_conch_pre")
REFERENCE = ("plip", "dinov2", "exaonepath_raw")
LABELS = {"uni_v1": "UNI", "uni2": "UNI2-h", "virchow2": "Virchow2", "hoptimus1": "H-optimus-1",
          "exaonepath": "EXAONEPath", "seal_uni2": "SEAL-UNI2", "conch_pre": "CONCH",
          "seal_conch_pre": "SEAL-CONCH", "plip": "PLIP", "dinov2": "DINOv2",
          "exaonepath_raw": "EXAONEPath\n(no Macenko)"}
ANCHORED = {"seal_uni2", "conch_pre", "seal_conch_pre", "plip"}
PREDICTORS = {"normalized_shift_low_mid_d0.25": "Low–mid", "normalized_shift_mid_d0.25": "Mid",
              "normalized_shift_high_d0.25": "High", "high_over_low_mid_d0.25": "High / low–mid ratio",
              "normalized_shift_low_mid_d0.5": "Low–mid (dose 0.50)", "normalized_shift_mid_d0.5": "Mid (dose 0.50)",
              "normalized_shift_high_d0.5": "High (dose 0.50)"}
OUTCOMES = {"robustness_index": "PathoROB RI", "normalized_distance": "Normalized scanner distance",
            "detectability_best": "Best-probe detectability"}
# Two-group palette validated with the dataviz validator (CVD and normal-vision separation pass).
BLUE, ORANGE, GREY = "#1f6fb4", "#c8553d", "#8a8a8a"
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
RI_CHANCE = 0.089


def spearman_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman rho for each row pair ([R, n] arrays)."""
    rx = stats.rankdata(x, axis=1)
    ry = stats.rankdata(y, axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    return (rx * ry).sum(axis=1) / np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))


def permutation_p(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    observed = abs(stats.pearsonr(rx, ry).statistic)
    perms = np.array(list(itertools.permutations(ry)))
    rxc = rx - rx.mean()
    pc = perms - perms.mean(axis=1, keepdims=True)
    rho = (pc @ rxc) / np.sqrt((pc ** 2).sum(axis=1) * (rxc ** 2).sum())
    return float((np.abs(rho) >= observed - 1e-12).mean())


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    r = np.column_stack([stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)])
    design = np.column_stack([np.ones(len(z)), r[:, 2]])
    resid = [r[:, i] - design @ np.linalg.lstsq(design, r[:, i], rcond=None)[0] for i in (0, 1)]
    return float(stats.pearsonr(*resid).statistic)


def summarize(draws: np.ndarray) -> tuple[float, float, float]:
    finite = draws[1:][np.isfinite(draws[1:])]
    return float(draws[0]), float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def neighbour_rates(name: str) -> tuple[float, float]:
    ri = np.load(TASKS / name / "ri.npz")
    k = int(ri["k_opt"])
    n = ri["so"].shape[0] * k
    return float(ri["so"][:, :k].sum() / n), float(ri["os"][:, :k].sum() / n)


def style_axis(ax) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def scatter(ax, table: pd.DataFrame, x: str, y: str, corr: dict, offsets: dict) -> None:
    for name in WSI + REFERENCE:
        row = table.loc[name]
        reference = name in REFERENCE
        color = GREY if reference else (ORANGE if name in ANCHORED else BLUE)
        ax.errorbar(row[x], row[y], xerr=[[row[x] - row[x + "_lo"]], [row[x + "_hi"] - row[x]]],
                    yerr=[[row[y] - row[y + "_lo"]], [row[y + "_hi"] - row[y]]], fmt="none",
                    ecolor=color, elinewidth=0.8, alpha=0.6, zorder=2)
        ax.scatter(row[x], row[y], s=64, zorder=3, marker="o",
                   facecolor="white" if reference else color, edgecolor=color if reference else "white",
                   linewidth=1.6 if reference else 1.0)
        dx, dy, ha = offsets.get(name, (6, 4, "left"))
        arrow = {"arrowstyle": "-", "color": MUTED, "linewidth": 0.6} if max(abs(dx), abs(dy)) > 10 else None
        ax.annotate(LABELS[name], (row[x], row[y]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, color=MUTED if reference else INK, ha=ha, va="center", arrowprops=arrow)
    text = f"8 WSI-pretrained models\nρ = {corr['rho']:+.2f} [{corr['lo']:+.2f}, {corr['hi']:+.2f}]\npermutation p = {corr['p_perm']:.3f}"
    top = y == "robustness_index"
    ax.text(0.97, 0.97 if top else 0.03, text, transform=ax.transAxes, ha="right", va="top" if top else "bottom",
            fontsize=8, color=INK, bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.4"})
    style_axis(ax)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    weights = bootstrap_weights("pannormal", len(load_cohort()))
    draws = {}
    for name in WSI + REFERENCE:
        values, _ = model_draws(name, weights)
        draws[name] = values
        print(f"loaded {name}", flush=True)

    statistics = sorted({*PREDICTORS, *OUTCOMES, "between_tissue_distance", "tissue_retrieval"})
    rows = []
    for name in WSI + REFERENCE:
        row = {"model": name, "label": LABELS[name], "in_correlation": name in WSI,
               "group": "reference" if name in REFERENCE else ("anchored" if name in ANCHORED else "image-only")}
        for statistic in statistics:
            estimate, lo, hi = summarize(draws[name][statistic])
            row.update({statistic: estimate, statistic + "_lo": lo, statistic + "_hi": hi})
        row["so_rate"], row["os_rate"] = neighbour_rates(name)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(OUTPUT / "model_table.csv")

    correlations = []
    btd = table.loc[list(WSI), "between_tissue_distance"].to_numpy()
    for predictor, plabel in PREDICTORS.items():
        for outcome, olabel in OUTCOMES.items():
            x = np.stack([draws[name][predictor] for name in WSI], axis=1)
            y = np.stack([draws[name][outcome] for name in WSI], axis=1)
            rho = spearman_rows(x, y)
            estimate, lo, hi = summarize(rho)
            correlations.append({"predictor": predictor, "predictor_label": plabel, "outcome": outcome,
                                 "outcome_label": olabel, "rho": estimate, "lo": lo, "hi": hi,
                                 "p_perm": permutation_p(x[0], y[0]),
                                 "partial_rho_given_between_tissue": partial_spearman(x[0], y[0], btd)})
    corr = pd.DataFrame(correlations)
    corr.to_csv(OUTPUT / "correlations.csv", index=False)

    main_figure(table, corr)
    supplementary_figure(table)
    write_summary(table, corr)


def lookup(corr: pd.DataFrame, predictor: str, outcome: str) -> dict:
    return corr[(corr.predictor == predictor) & (corr.outcome == outcome)].iloc[0].to_dict()


def main_figure(table: pd.DataFrame, corr: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig = plt.figure(figsize=(13.5, 4.4))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.32)
    x = "normalized_shift_high_d0.25"
    ax = fig.add_subplot(grid[0])
    scatter(ax, table, x, "robustness_index", lookup(corr, x, "robustness_index"),
            {"hoptimus1": (6, -7, "left"), "seal_conch_pre": (6, -8, "left"), "uni2": (8, -9, "left"),
             "exaonepath_raw": (8, 2, "left"), "plip": (6, 6, "left")})
    ax.axhline(RI_CHANCE, color=MUTED, linestyle=(0, (3, 3)), linewidth=0.8)
    ax.text(ax.get_xlim()[0], RI_CHANCE, " chance", fontsize=7, color=MUTED, ha="left", va="bottom")
    ax.set(xlabel="High-frequency sensitivity\n(shift per equal OD change / between-tissue distance)",
           ylabel="PathoROB robustness index (higher = more robust)")
    ax.set_title("A", loc="left", fontsize=11, fontweight="bold")
    ax = fig.add_subplot(grid[1])
    scatter(ax, table, x, "normalized_distance", lookup(corr, x, "normalized_distance"),
            {"hoptimus1": (0, 14, "center"), "uni2": (-6, 8, "right"), "seal_uni2": (6, -8, "left"),
             "exaonepath_raw": (8, -6, "left"), "exaonepath": (-8, 0, "right"), "virchow2": (8, -5, "left"),
             "conch_pre": (8, -9, "left"), "seal_conch_pre": (-6, 24, "center"), "dinov2": (12, 12, "left")})
    ax.set(xlabel="High-frequency sensitivity\n(shift per equal OD change / between-tissue distance)",
           ylabel="Normalized scanner distance (lower = more robust)")
    ax.set_title("B", loc="left", fontsize=11, fontweight="bold")
    ax = fig.add_subplot(grid[2])
    order = ["normalized_shift_low_mid_d0.25", "normalized_shift_mid_d0.25", "normalized_shift_high_d0.25",
             "high_over_low_mid_d0.25"]
    for index, predictor in enumerate(order):
        row = lookup(corr, predictor, "robustness_index")
        color = MUTED if predictor.startswith("high_over") else BLUE
        ax.errorbar(row["rho"], index, xerr=[[row["rho"] - row["lo"]], [row["hi"] - row["rho"]]], fmt="o",
                    color=color, markersize=7, elinewidth=1.4, capsize=3)
        ax.text(0.98, index, f"{row['rho']:+.2f}", transform=ax.get_yaxis_transform(), fontsize=8,
                color=INK, ha="right", va="center")
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(range(len(order)), ["Low–mid\nsensitivity", "Mid\nsensitivity", "High\nsensitivity",
                                      "High / low–mid\nratio"])
    ax.invert_yaxis()
    ax.set(xlim=(-1.05, 1.05), xlabel="Spearman ρ with robustness index\n(8 WSI-pretrained models)")
    ax.set_title("C", loc="left", fontsize=11, fontweight="bold")
    style_axis(ax)
    handles = [plt.Line2D([], [], marker="o", linestyle="None", markerfacecolor=BLUE, markeredgecolor="white",
                          markersize=8, label="Image-only pretraining"),
               plt.Line2D([], [], marker="o", linestyle="None", markerfacecolor=ORANGE, markeredgecolor="white",
                          markersize=8, label="Image + text or ST anchor"),
               plt.Line2D([], [], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor=GREY,
                          markersize=8, label="Reference (not in ρ)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.4, -0.06))
    save(fig, "fig_sensitivity_robustness")


def supplementary_figure(table: pd.DataFrame) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.62})
    offsets = {"conch_pre": (-10, -8, "right"), "seal_conch_pre": (-10, 8, "right"), "uni2": (6, 6, "left"),
               "hoptimus1": (2, 16, "right"), "virchow2": (8, -2, "left"),
               "seal_uni2": (6, -7, "left"), "exaonepath_raw": (6, -8, "left")}
    for name in WSI + REFERENCE:
        row = table.loc[name]
        reference = name in REFERENCE
        color = GREY if reference else (ORANGE if name in ANCHORED else BLUE)
        left.scatter(row["tissue_retrieval"], row["normalized_distance"], s=64, zorder=3,
                     facecolor="white" if reference else color, edgecolor=color if reference else "white",
                     linewidth=1.6 if reference else 1.0)
        dx, dy, ha = offsets.get(name, (6, 4, "left"))
        weight = "bold" if name in ("plip", "conch_pre") else "normal"
        arrow = {"arrowstyle": "-", "color": MUTED, "linewidth": 0.6} if max(abs(dx), abs(dy)) > 9 else None
        left.annotate(LABELS[name], (row["tissue_retrieval"], row["normalized_distance"]), xytext=(dx, dy),
                      textcoords="offset points", fontsize=8, fontweight=weight,
                      color=MUTED if reference else INK, ha=ha, va="center", arrowprops=arrow)
    left.set(xlabel="Tissue information (same-tissue retrieval, macro recall)",
             ylabel="Scanner sensitivity (normalized scanner distance)")
    left.set_title("A  Tissue information vs scanner sensitivity", loc="left", fontsize=10)
    style_axis(left)
    order = table.sort_values("robustness_index", ascending=False).index.tolist()
    positions = np.arange(len(order))
    right.barh(positions - 0.2, table.loc[order, "so_rate"], height=0.38, color=BLUE,
               label="Same tissue, other scanner (SO)")
    right.barh(positions + 0.2, table.loc[order, "os_rate"], height=0.38, color=ORANGE,
               label="Other tissue, same scanner (OS)")
    right.set_yticks(positions, [f"{LABELS[n].replace(chr(10), ' ')}  (RI {table.loc[n, 'robustness_index']:.2f})"
                                 for n in order])
    right.invert_yaxis()
    right.set(xlabel="Fraction of the k nearest neighbours")
    right.set_title("B  Neighbours behind the robustness index", loc="left", fontsize=10)
    right.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    style_axis(right)
    save(fig, "fig_tissue_vs_scanner_supp")


def save(fig, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{name}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(table: pd.DataFrame, corr: pd.DataFrame) -> None:
    lines = ["# RV14 perturbation sensitivity and scanner robustness", "",
             "Eight WSI-pretrained variants; Spearman ρ with 95% CI from 2,000 shared slide resamples; exact "
             "permutation p over 8! model orderings; partial ρ controls for between-tissue distance.", "",
             "| Predictor | Outcome | ρ [95% CI] | permutation p | partial ρ (between-tissue distance) |",
             "| --- | --- | --- | --- | --- |"]
    for row in corr.itertuples():
        lines.append(f"| {row.predictor_label} | {row.outcome_label} | {row.rho:+.2f} [{row.lo:+.2f}, {row.hi:+.2f}] | "
                     f"{row.p_perm:.4f} | {row.partial_rho_given_between_tissue:+.2f} |")
    lines += ["", "## Models", "", "| Model | Group | High-frequency sensitivity | RI | Normalized distance | "
              "Tissue retrieval | SO rate | OS rate |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, row in table.iterrows():
        lines.append(f"| {row.label} | {row.group} | {row['normalized_shift_high_d0.25']:.4f} | "
                     f"{row.robustness_index:.3f} | {row.normalized_distance:.3f} | {row.tissue_retrieval:.3f} | "
                     f"{row.so_rate:.3f} | {row.os_rate:.3f} |")
    lines += ["", "Figures: `fig_sensitivity_robustness` (main candidate), `fig_tissue_vs_scanner_supp`.", ""]
    (OUTPUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
