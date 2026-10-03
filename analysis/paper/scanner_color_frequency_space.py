#!/usr/bin/env python3
"""Map paired scanner image phenotypes onto descriptive color/frequency axes."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/scanner_batch_effect_analysis_2026-09-17"
OUT = Path(__file__).resolve().parent / "results/scanner_color_frequency_space"
SCANNERS = ["at2", "versa", "akoya", "gt450", "s360", "s60"]
COLOR = ["delta_lab_l", "delta_lab_a", "delta_lab_b", "log2_mean_od_ratio"]
FREQUENCY = ["frequency_low_mid", "frequency_mid", "frequency_high"]


def scalar_axis(data: pd.DataFrame, columns: list[str], anchor: str) -> tuple[pd.DataFrame, dict]:
    block = data[columns].to_numpy(float)
    scales = block.std(axis=0, ddof=1)
    if (scales <= 0).any():
        raise ValueError("Zero scale")
    means = data.groupby("target_scanner")[columns].mean().reindex(SCANNERS).fillna(0.0)
    centroid = means.to_numpy(float) / scales
    pca = PCA().fit(centroid)
    slide_pca = PCA().fit(block / scales)
    direction = pca.components_[0].copy()
    if direction[columns.index(anchor)] < 0:
        direction *= -1
    scores = (centroid - centroid[0]) @ direction
    return (pd.DataFrame({"scanner": SCANNERS, "score": scores}),
            {"variables": columns, "scales": scales.tolist(),
             "pc1_loadings": direction.tolist(),
             "centroid_pc1_variance_fraction": float(pca.explained_variance_ratio_[0]),
             "centroid_pc2_variance_fraction": float(pca.explained_variance_ratio_[1]),
             "slide_pc1_variance_fraction": float(slide_pca.explained_variance_ratio_[0])})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = BASE / "01_scanner_atlas/slide_vectors.csv"
    frame = pd.read_csv(path, dtype={"slide_id": str})
    frame = frame[frame.source_scanner.eq("at2")].copy()
    if len(frame) != 515 or frame.slide_id.nunique() != 103:
        raise ValueError("Expected 103 slides x 5 targets")
    frame["delta_e76"] = np.sqrt((frame[["delta_lab_l", "delta_lab_a", "delta_lab_b"]]**2).sum(axis=1))
    reference = frame.drop_duplicates("slide_id").copy()
    reference["target_scanner"] = "at2"
    for column in COLOR + FREQUENCY + ["delta_e76"]:
        reference[column] = 0.0
    all_rows = pd.concat([frame, reference], ignore_index=True)
    color_score, color_info = scalar_axis(all_rows, COLOR, "delta_lab_l")
    frequency_score, frequency_info = scalar_axis(all_rows, FREQUENCY, "frequency_high")
    coordinates = (all_rows.groupby("target_scanner", as_index=False)
                   [["delta_e76", "frequency_low_mid", "frequency_mid", "frequency_high"]].mean()
                   .rename(columns={"target_scanner": "scanner"}))
    coordinates = coordinates.merge(color_score.rename(columns={"score": "color_pc1"}), on="scanner")
    coordinates = coordinates.merge(frequency_score.rename(columns={"score": "frequency_pc1"}), on="scanner")
    coordinates = coordinates.set_index("scanner").reindex(SCANNERS).reset_index()

    # Resample physical slides jointly across scanners; fixed axes describe uncertainty in position.
    slides = sorted(frame.slide_id.unique())
    cube = np.stack([all_rows[all_rows.target_scanner.eq(scanner)].set_index("slide_id")
                     .reindex(slides)[COLOR + FREQUENCY + ["delta_e76"]].to_numpy(float)
                     for scanner in SCANNERS], axis=1)
    rng = np.random.default_rng(20260924)
    draws = cube[rng.integers(0, len(slides), size=(5000, len(slides)))].mean(axis=1)
    for domain, info, name in [(COLOR, color_info, "color_pc1"),
                               (FREQUENCY, frequency_info, "frequency_pc1")]:
        indices = [COLOR.index(c) if c in COLOR else len(COLOR) + FREQUENCY.index(c) for c in domain]
        mean_scores = draws[:, :, indices] / np.asarray(info["scales"])
        projected = ((mean_scores - mean_scores[:, :1]) @ np.asarray(info["pc1_loadings"]))
        low, high = np.quantile(projected, [0.025, 0.975], axis=0)
        coordinates[f"{name}_ci_low"] = low
        coordinates[f"{name}_ci_high"] = high
    high = draws[:, :, len(COLOR) + FREQUENCY.index("frequency_high")]
    de = draws[:, :, -1]
    for name, values in [("frequency_high", high), ("delta_e76", de)]:
        low, high_ci = np.quantile(values, [0.025, 0.975], axis=0)
        coordinates[f"{name}_ci_low"] = low
        coordinates[f"{name}_ci_high"] = high_ci
    coordinates.to_csv(OUT / "scanner_coordinates.csv", index=False)

    # The relationship to UNI is descriptive: one exposure and one response per slide.
    uni = pd.read_csv(BASE / "03_uni/location_metrics.csv",
                      usecols=["slide_id", "scanner", "arm", "target_gain"], dtype={"slide_id": str})
    uni = uni[uni.arm.isin(["reinhard", "combined"])].groupby(
        ["slide_id", "scanner", "arm"], as_index=False).target_gain.mean()
    uni = uni.pivot(index=["slide_id", "scanner"], columns="arm", values="target_gain").reset_index()
    uni["incremental_gain"] = uni.combined - uni.reinhard
    pair = frame.merge(uni, left_on=["slide_id", "target_scanner"],
                       right_on=["slide_id", "scanner"], validate="one_to_one")
    relation = []
    for scanner, group in pair.groupby("scanner"):
        overall = pearsonr(group.frequency_high, group.incremental_gain)
        color_only = pearsonr(group.delta_e76, group.incremental_gain)
        color_design = np.column_stack([np.ones(len(group)), group[COLOR].to_numpy(float)])
        high_resid = group.frequency_high.to_numpy(float) - color_design @ np.linalg.lstsq(
            color_design, group.frequency_high.to_numpy(float), rcond=None)[0]
        gain_resid = group.incremental_gain.to_numpy(float) - color_design @ np.linalg.lstsq(
            color_design, group.incremental_gain.to_numpy(float), rcond=None)[0]
        color_adjusted = pearsonr(high_resid, gain_resid)
        centered = group[["frequency_high", "incremental_gain"]] - group.groupby("tissue_type")[
            ["frequency_high", "incremental_gain"]].transform("mean")
        within = pearsonr(centered.frequency_high, centered.incremental_gain)
        relation.append({"scanner": scanner, "n_slides": len(group),
                         "pearson_r": overall.statistic, "pearson_p": overall.pvalue,
                         "delta_e76_gain_r": color_only.statistic,
                         "high_frequency_gain_r_after_color_adjustment": color_adjusted.statistic,
                         "within_tissue_r": within.statistic,
                         "within_tissue_p_naive": within.pvalue,
                         "note": "observational; within-tissue p ignores tissue clustering"})
    pd.DataFrame(relation).to_csv(OUT / "frequency_uni_association.csv", index=False)

    colors = {"at2": "#333333", "versa": "#5c88bd", "akoya": "#bf5d5d",
              "gt450": "#ce9444", "s360": "#78a26d", "s60": "#9379b3"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for scanner, group in frame.groupby("target_scanner"):
        axes[1].scatter(group.delta_e76, group.frequency_high, s=8,
                        color=colors[scanner], alpha=.12, linewidths=0)
    for row in coordinates.itertuples(index=False):
        offset = ((5, -18) if row.scanner == "s360" else
                  (5, -13) if row.scanner == "s60" else (5, 5))
        axes[0].errorbar(row.color_pc1, row.frequency_pc1,
                         xerr=[[row.color_pc1-row.color_pc1_ci_low], [row.color_pc1_ci_high-row.color_pc1]],
                         yerr=[[row.frequency_pc1-row.frequency_pc1_ci_low],
                               [row.frequency_pc1_ci_high-row.frequency_pc1]],
                         fmt="o", color=colors[row.scanner], capsize=2)
        axes[0].annotate(row.scanner.upper(), (row.color_pc1, row.frequency_pc1),
                         xytext=offset, textcoords="offset points", fontsize=8)
        axes[1].errorbar(row.delta_e76, row.frequency_high,
                         xerr=[[row.delta_e76-row.delta_e76_ci_low], [row.delta_e76_ci_high-row.delta_e76]],
                         yerr=[[row.frequency_high-row.frequency_high_ci_low],
                               [row.frequency_high_ci_high-row.frequency_high]],
                         fmt="o", color=colors[row.scanner], capsize=2)
        axes[1].annotate(row.scanner.upper(), (row.delta_e76, row.frequency_high),
                         xytext=offset, textcoords="offset points", fontsize=8)
    axes[0].axhline(0, color="0.8", linewidth=.6)
    axes[0].axvline(0, color="0.8", linewidth=.6)
    axes[0].set(xlabel="Color PC1 relative to AT2", ylabel="Frequency PC1 relative to AT2",
                title="Multivariate scanner phenotype")
    axes[1].axhline(0, color="0.8", linewidth=.6)
    axes[1].set(xlabel="Mean paired CIELAB ΔE76 from AT2",
                ylabel="High-band log2 amplitude ratio to AT2",
                title="Interpretable two-axis view")
    fig.savefig(OUT / "scanner_color_frequency_space.png", dpi=220)
    fig.savefig(OUT / "scanner_color_frequency_space.pdf")
    plt.close(fig)
    (OUT / "axis_summary.json").write_text(json.dumps(
        {"n_slides": 103, "basis": "AT2-referenced scanner-slide image contrasts on the common grid",
         "color": color_info, "frequency": frequency_info,
         "interpretation": "Descriptive scanner mean positions, not intrinsic optical hardware specifications"},
        indent=2) + "\n")
    print((OUT / "axis_summary.json").read_text())


if __name__ == "__main__":
    main()
