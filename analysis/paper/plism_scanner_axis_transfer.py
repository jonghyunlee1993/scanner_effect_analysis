#!/usr/bin/env python3
"""Project PLISM image contrasts onto axes fixed from PanNormal."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results/scanner_color_frequency_space"
PLISM = ROOT / "outputs/plism_factorial_external_v1/02_factorial/core_section_cells.csv"
COLOR = ["delta_lab_l", "delta_lab_a", "delta_lab_b", "log2_mean_od_ratio"]
FREQUENCY = ["frequency_low_mid", "frequency_mid", "frequency_high"]
SCANNERS = ["GT450", "P", "S210", "S360", "S60", "SQ"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project(frame: pd.DataFrame, axes: dict) -> pd.DataFrame:
    result = frame.copy()
    for key, columns in [("color", COLOR), ("frequency", FREQUENCY)]:
        scales = np.asarray(axes[key]["scales"], float)
        direction = np.asarray(axes[key]["pc1_loadings"], float)
        result[f"{key}_pc1"] = (result[columns].to_numpy(float) / scales) @ direction
    result["delta_e76"] = np.sqrt((result[COLOR[:3]]**2).sum(axis=1))
    return result


def aggregate_sections(cells: pd.DataFrame, axes: dict, expected_sections: int) -> pd.DataFrame:
    per_section = cells.groupby(["stain", "scanner"], as_index=False)[COLOR + FREQUENCY].mean()
    counts = cells.groupby(["stain", "scanner"]).core.nunique()
    if (counts.min() < 45 or per_section.stain.nunique() != expected_sections
            or per_section.scanner.nunique() != 6):
        raise ValueError("Incomplete PLISM section x scanner cells")
    return project(per_section, axes)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    axes_path = OUT / "axis_summary.json"
    axes = json.loads(axes_path.read_text())
    cells = pd.read_csv(PLISM)
    cells = cells[(cells.tissue_scheme == "organ_aligned") & (cells.variant == "primary")]
    sections = aggregate_sections(cells, axes, 13)
    sections.to_csv(OUT / "plism_section_projection.csv", index=False)
    pan = pd.read_csv(OUT / "scanner_coordinates.csv").set_index("scanner")
    rng = np.random.default_rng(20260924)
    rows = []
    for scanner in SCANNERS:
        group = sections[sections.scanner.eq(scanner)].set_index("stain")
        if len(group) != 13:
            raise ValueError((scanner, len(group)))
        row = {"scanner": scanner, "n_sections": len(group),
               "pan_scanner_shared": scanner.lower() in pan.index}
        for metric in ["color_pc1", "frequency_pc1", "delta_e76", "frequency_high"]:
            values = group[metric].to_numpy(float)
            draws = values[rng.integers(0, 13, (20000, 13))].mean(axis=1)
            row[metric] = float(values.mean())
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = map(
                float, np.quantile(draws, [0.025, 0.975]))
            if scanner.lower() in pan.index:
                row[f"pan_{metric}"] = float(pan.loc[scanner.lower(), metric])
                row[f"external_minus_pan_{metric}"] = row[metric] - row[f"pan_{metric}"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "plism_transfer_summary.csv", index=False)

    # Assess whether PanNormal's fixed one-dimensional directions capture PLISM variation.
    axis_transfer = {}
    for family, columns in [("color", COLOR), ("frequency", FREQUENCY)]:
        scale = np.asarray(axes[family]["scales"], float)
        direction = np.asarray(axes[family]["pc1_loadings"], float)
        for label, matrix in [("section_scanner", sections[columns].to_numpy(float)),
                              ("scanner_means", sections.groupby("scanner")[columns].mean().to_numpy(float))]:
            centered = matrix / scale - (matrix / scale).mean(axis=0)
            fraction = float(np.sum((centered @ direction)**2) / np.sum(centered**2))
            axis_transfer[f"{family}_{label}_variance_fraction"] = fraction
    (OUT / "plism_axis_transfer_diagnostics.json").write_text(json.dumps(axis_transfer, indent=2) + "\n")

    no_hrh = cells[cells.stain.ne("HRH")]
    sensitivity = aggregate_sections(no_hrh, axes, 12)
    sensitivity.groupby("scanner", as_index=False)[
        ["color_pc1", "frequency_pc1", "delta_e76", "frequency_high"]
    ].mean().to_csv(OUT / "plism_exclude_hrh_sensitivity.csv", index=False)

    palette = {"at2": "#333333", "versa": "#5c88bd", "akoya": "#bf5d5d",
               "gt450": "#ce9444", "s360": "#78a26d", "s60": "#9379b3",
               "p": "#877f73", "s210": "#4c9f9a", "sq": "#c37aab"}
    fig, axis = plt.subplots(figsize=(7.5, 5.7), constrained_layout=True)
    for row in pan.itertuples():
        axis.scatter(row.color_pc1, row.frequency_pc1, s=70,
                     facecolor="white", edgecolor=palette[row.Index], linewidth=1.8, zorder=5)
        axis.annotate(f"Pan {row.Index.upper()}", (row.color_pc1, row.frequency_pc1),
                      xytext=(5, 5), textcoords="offset points", fontsize=7.5)
    for scanner, group in sections.groupby("scanner"):
        key = scanner.lower()
        axis.scatter(group.color_pc1, group.frequency_pc1, s=15,
                     color=palette[key], alpha=.16, marker="^", linewidth=0)
        item = summary[summary.scanner.eq(scanner)].iloc[0]
        axis.errorbar(item.color_pc1, item.frequency_pc1,
                      xerr=[[item.color_pc1-item.color_pc1_ci_low],
                            [item.color_pc1_ci_high-item.color_pc1]],
                      yerr=[[item.frequency_pc1-item.frequency_pc1_ci_low],
                            [item.frequency_pc1_ci_high-item.frequency_pc1]],
                      fmt="^", color=palette[key], markersize=7, capsize=2, zorder=6)
        axis.annotate(f"PLISM {scanner}", (item.color_pc1, item.frequency_pc1),
                      xytext=(5, -12 if scanner in ("S60", "SQ") else 6),
                      textcoords="offset points", fontsize=7.5)
        if key in pan.index:
            axis.annotate("", xy=(item.color_pc1, item.frequency_pc1),
                          xytext=(float(pan.loc[key, "color_pc1"]),
                                  float(pan.loc[key, "frequency_pc1"])),
                          arrowprops={"arrowstyle": "->", "color": palette[key],
                                      "linewidth": 1.1, "alpha": .8})
    axis.axhline(0, color="0.82", linewidth=.7)
    axis.axvline(0, color="0.82", linewidth=.7)
    axis.set(xlabel="PanNormal color PC1 relative to AT2",
             ylabel="PanNormal frequency PC1 relative to AT2",
             title="Frozen PanNormal axes projected onto PLISM")
    fig.savefig(OUT / "plism_axis_transfer.png", dpi=220)
    fig.savefig(OUT / "plism_axis_transfer.pdf")
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7.5, 5.7), constrained_layout=True)
    for row in pan.itertuples():
        axis.scatter(row.delta_e76, row.frequency_high, s=70,
                     facecolor="white", edgecolor=palette[row.Index], linewidth=1.8, zorder=5)
        pan_offset = {"versa": (-47, 10), "s60": (5, 7),
                      "s360": (5, -18)}.get(row.Index, (5, 5))
        axis.annotate(f"Pan {row.Index.upper()}", (row.delta_e76, row.frequency_high),
                      xytext=pan_offset, textcoords="offset points", fontsize=7.5)
    for scanner, group in sections.groupby("scanner"):
        key = scanner.lower()
        axis.scatter(group.delta_e76, group.frequency_high, s=15,
                     color=palette[key], alpha=.16, marker="^", linewidth=0)
        item = summary[summary.scanner.eq(scanner)].iloc[0]
        axis.errorbar(item.delta_e76, item.frequency_high,
                      xerr=[[item.delta_e76-item.delta_e76_ci_low],
                            [item.delta_e76_ci_high-item.delta_e76]],
                      yerr=[[item.frequency_high-item.frequency_high_ci_low],
                            [item.frequency_high_ci_high-item.frequency_high]],
                      fmt="^", color=palette[key], markersize=7, capsize=2, zorder=6)
        axis.annotate(f"PLISM {scanner}", (item.delta_e76, item.frequency_high),
                      xytext=(5, -16 if scanner == "P" else
                              -12 if scanner in ("S60", "SQ") else 6),
                      textcoords="offset points", fontsize=7.5)
        if key in pan.index:
            axis.annotate("", xy=(item.delta_e76, item.frequency_high),
                          xytext=(float(pan.loc[key, "delta_e76"]),
                                  float(pan.loc[key, "frequency_high"])),
                          arrowprops={"arrowstyle": "->", "color": palette[key],
                                      "linewidth": 1.1, "alpha": .8})
    axis.axhline(0, color="0.82", linewidth=.7)
    axis.set(xlabel="Mean paired CIELAB ΔE76 from each dataset's AT2",
             ylabel="High-band log2 amplitude ratio to each dataset's AT2",
             title="All scanners in directly measured coordinates")
    fig.savefig(OUT / "all_scanners_direct_space.png", dpi=220)
    fig.savefig(OUT / "all_scanners_direct_space.pdf")
    plt.close(fig)
    manifest = {"source": str(PLISM), "source_sha256": sha256(PLISM),
                "pan_axis_sha256": sha256(axes_path),
                "design": "PanNormal scales/loadings frozen; PLISM core -> section x scanner -> section bootstrap",
                "sections": 13, "shared_scanners": ["GT450", "S360", "S60"],
                "note": "External device, site, section, stain and tissue context differ; this is an axis transfer test."}
    (OUT / "plism_transfer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
