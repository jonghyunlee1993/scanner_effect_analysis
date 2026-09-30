#!/usr/bin/env python3
"""RV-P0d (i): restore the producer of the Fig. 2 tissue/core positions.

Recovered producer: `.Trash/2026-09-26_paper_refactor/unused_analysis/analysis/paper/
scanner_tissue_space.py` (run on 2026-09-24 as SLURM job 23553153 from
`00_manuscript/analysis/`, later relocated to `analysis/paper/` and then moved to `.Trash`).
The byte-code cache of the run survives at `.Trash/2026-09-26_paper_refactor/generated/
analysis/paper/__pycache__/scanner_tissue_space.cpython-313.pyc` and records the same
source size and mtime (12,499 bytes, 2026-09-24 16:54:52).

`rank_summary`, `variance_partition` and `main` below are copied verbatim; only the
module constant `OUT` differs, so regenerated files are written under
`results/provenance_restoration/scanner_tissue_space/` and nothing frozen is touched.
`verify()` then compares every regenerated file with the frozen file in
`analysis/paper/results/scanner_color_frequency_space/` and with the evidence lock.
"""

import hashlib
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results/provenance_restoration/scanner_tissue_space"
FROZEN = ROOT / "analysis/paper/results/scanner_color_frequency_space"
EVIDENCE_LOCK = ROOT / "analysis/paper/evidence_hashes.json"
PAN = ROOT / "outputs/scanner_batch_effect_analysis_2026-09-17/01_scanner_atlas/slide_vectors.csv"
PLISM = ROOT / "outputs/plism_factorial_external_v1/02_factorial/core_section_cells.csv"
UNI = ROOT / "outputs/scanner_batch_effect_analysis_2026-09-17/03_uni/location_metrics.csv"
SCANNERS = ["versa", "akoya", "gt450", "s360", "s60"]
PALETTE = {"versa": "#5c88bd", "akoya": "#bf5d5d", "gt450": "#ce9444",
           "s360": "#78a26d", "s60": "#9379b3", "p": "#877f73",
           "s210": "#4c9f9a", "sq": "#c37aab"}
METRICS = ["delta_e76", "frequency_high"]


def rank_summary(grouped: pd.DataFrame, group_col: str) -> dict:
    means = grouped.groupby("scanner")[METRICS].mean()
    result = {"units": int(grouped[group_col].nunique()), "scanner_mean_order": {},
              "mean_spearman_rank_rho": {}, "median_spearman_rank_rho": {},
              "extreme_scanner_counts": {}}
    for metric in METRICS:
        result["scanner_mean_order"][metric] = means[metric].sort_values().index.tolist()
        pivot = grouped.pivot(index=group_col, columns="scanner", values=metric)
        reference = means.loc[pivot.columns, metric].to_numpy(float)
        rho = [spearmanr(row, reference).statistic for row in pivot.to_numpy(float)]
        result["mean_spearman_rank_rho"][metric] = float(np.mean(rho))
        result["median_spearman_rank_rho"][metric] = float(np.median(rho))
        result["extreme_scanner_counts"][metric] = {
            "minimum": pivot.idxmin(axis=1).value_counts().to_dict(),
            "maximum": pivot.idxmax(axis=1).value_counts().to_dict()}
    return result


def variance_partition(pan: pd.DataFrame, metric: str) -> dict:
    """Exact weighted decomposition for complete slide x scanner blocks."""
    y = pan[metric].to_numpy(float)
    grand = y.mean()
    scan = pan.groupby("scanner")[metric].transform("mean").to_numpy(float)
    tissue = pan.groupby("tissue_type")[metric].transform("mean").to_numpy(float)
    tissue_scan = pan.groupby(["tissue_type", "scanner"])[metric].transform("mean").to_numpy(float)
    parts = {"scanner": scan - grand, "tissue": tissue - grand,
             "scanner_by_tissue": tissue_scan - scan - tissue + grand,
             "within_tissue_slide": y - tissue_scan}
    ss = {name: float(np.sum(value**2)) for name, value in parts.items()}
    total = float(np.sum((y - grand)**2))
    if not np.isclose(sum(ss.values()), total):
        raise ValueError("Nonorthogonal decomposition")
    return {"sum_of_squares": ss, "fraction": {name: value / total for name, value in ss.items()}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pan = pd.read_csv(PAN, dtype={"slide_id": str})
    pan = pan[pan.source_scanner.eq("at2")].copy().rename(columns={"target_scanner": "scanner"})
    pan = pan[pan.scanner.isin(SCANNERS)]
    pan["delta_e76"] = np.linalg.norm(pan[["delta_lab_l", "delta_lab_a", "delta_lab_b"]].to_numpy(float), axis=1)
    if len(pan) != 103 * 5 or pan.groupby("slide_id").scanner.nunique().ne(5).any():
        raise ValueError("Incomplete PanNormal slide x scanner design")
    tissue = pan.groupby(["tissue_type", "scanner"], as_index=False).agg(
        n_slides=("slide_id", "nunique"), delta_e76=("delta_e76", "mean"),
        frequency_high=("frequency_high", "mean"))
    tissue.to_csv(OUT / "pannormal_tissue_positions.csv", index=False)

    uni = pd.read_csv(UNI, usecols=["slide_id", "scanner", "arm", "target_gain"],
                      dtype={"slide_id": str})
    uni = uni[uni.arm.isin(["reinhard", "combined"])].groupby(
        ["slide_id", "scanner", "arm"], as_index=False).target_gain.mean()
    uni = uni.pivot(index=["slide_id", "scanner"], columns="arm", values="target_gain").reset_index()
    uni["incremental_uni_gain"] = uni.combined - uni.reinhard
    paired_uni = pan[["slide_id", "tissue_type", "scanner", "frequency_high", "delta_e76"]].merge(
        uni[["slide_id", "scanner", "incremental_uni_gain"]],
        on=["slide_id", "scanner"], validate="one_to_one")
    tissue_uni = paired_uni.groupby(["tissue_type", "scanner"], as_index=False).agg(
        n_slides=("slide_id", "nunique"), frequency_high=("frequency_high", "mean"),
        delta_e76=("delta_e76", "mean"),
        incremental_uni_gain=("incremental_uni_gain", "mean"))
    tissue_uni.to_csv(OUT / "pannormal_tissue_uni_gain.csv", index=False)
    uni_summary = {}
    for scanner, group in tissue_uni.groupby("scanner"):
        uni_summary[scanner] = {
            "positive_tissue_means": int(group.incremental_uni_gain.gt(0).sum()),
            "n_tissues": int(len(group)),
            "mean_gain_across_tissues": float(group.incremental_uni_gain.mean()),
            "spearman_frequency_vs_gain_across_tissues": float(
                spearmanr(group.frequency_high, group.incremental_uni_gain).statistic)}

    cells = pd.read_csv(PLISM)
    cells = cells[cells.tissue_scheme.eq("organ_aligned") & cells.variant.eq("primary")].copy()
    cells["scanner"] = cells.scanner.str.lower()
    cells["delta_e76"] = np.linalg.norm(
        cells[["delta_lab_l", "delta_lab_a", "delta_lab_b"]].to_numpy(float), axis=1)
    core = cells.groupby(["core", "tissue_type", "scanner"], as_index=False).agg(
        n_stains=("stain", "nunique"), delta_e76=("delta_e76", "mean"),
        frequency_high=("frequency_high", "mean"))
    if core.core.nunique() != 46 or core.groupby("core").scanner.nunique().ne(6).any():
        raise ValueError("Incomplete PLISM core x scanner design")
    core.to_csv(OUT / "plism_core_positions.csv", index=False)

    result = {"coordinate_definition": "Paired AT2-reference CIELAB DeltaE76 and high-band log2 amplitude on the common grid; no trained embedding",
              "pannormal": rank_summary(tissue, "tissue_type"),
              "plism": rank_summary(core, "core"),
              "tissue_level_uni_gain": uni_summary,
              "pannormal_variance_fraction": {metric: variance_partition(pan, metric)["fraction"] for metric in METRICS},
              "limitations": ["PanNormal has only 1-5 physical slides per tissue type.",
                              "PLISM has one core per tissue label, repeatedly imaged across 13 stained sections.",
                              "PLISM has no AKOYA; dataset-specific AT2 references do not align acquisition context.",
                              "Rank summaries are descriptive, not inferential tests."]}
    (OUT / "tissue_space_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1), sharex=True, sharey=True, constrained_layout=True)
    for ax, grouped, title, count_label in [
        (axes[0], tissue, "PanNormal: tissue means", "37 tissue types; 1–5 slides each"),
        (axes[1], core, "PLISM: core means", "46 cores; 12–13 sections each")]:
        for scanner, item in grouped.groupby("scanner"):
            x, y = item.delta_e76.to_numpy(float), item.frequency_high.to_numpy(float)
            ax.scatter(x, y, s=14, alpha=.28, color=PALETTE[scanner], linewidths=0)
            ax.scatter(x.mean(), y.mean(), s=75, marker="X", color=PALETTE[scanner],
                       edgecolor="white", linewidth=.7, zorder=5)
            offsets = ({"versa": (-34, 15), "s60": (7, -14), "s360": (4, -17)}
                       if title.startswith("PanNormal") else
                       {"sq": (-12, 14), "s60": (7, -16), "p": (4, -14),
                        "s210": (4, 6), "s360": (4, -15)})
            ax.annotate(scanner.upper(), (x.mean(), y.mean()),
                        xytext=offsets.get(scanner, (3, 5)), textcoords="offset points", fontsize=8)
        ax.scatter([0], [0], s=38, marker="+", color="black", linewidths=1.2, zorder=4)
        ax.annotate("AT2", (0, 0), xytext=(3, 4), textcoords="offset points", fontsize=8)
        ax.axhline(0, color=".85", linewidth=.7, zorder=0)
        ax.set(title=title, xlabel="Paired color distance from AT2 (CIELAB ΔE76)")
        ax.text(.02, .02, count_label, transform=ax.transAxes, fontsize=7.5, color=".35")
    axes[0].set_ylabel("High-band log2 amplitude ratio to AT2")
    fig.savefig(OUT / "scanner_tissue_space.png", dpi=220)
    fig.savefig(OUT / "scanner_tissue_space.pdf")
    plt.close(fig)

    # Place both datasets in the same physical coordinates. Shared-scanner arrows
    # make context shifts visible while the small points retain tissue spread.
    fig, ax = plt.subplots(figsize=(9.8, 6.7), constrained_layout=True)
    pan_mean = tissue.groupby("scanner")[METRICS].mean()
    plism_mean = core.groupby("scanner")[METRICS].mean()
    for scanner, group in tissue.groupby("scanner"):
        ax.scatter(group.delta_e76, group.frequency_high, s=17, marker="o",
                   color=PALETTE[scanner], alpha=.24, linewidths=0, zorder=2)
    for scanner, group in core.groupby("scanner"):
        ax.scatter(group.delta_e76, group.frequency_high, s=19, marker="^",
                   color=PALETTE[scanner], alpha=.21, linewidths=0, zorder=2)
    for scanner in ["gt450", "s360", "s60"]:
        start = pan_mean.loc[scanner]
        end = plism_mean.loc[scanner]
        ax.annotate("", xy=(end.delta_e76, end.frequency_high),
                    xytext=(start.delta_e76, start.frequency_high),
                    arrowprops={"arrowstyle": "->", "color": PALETTE[scanner],
                                "lw": 1.6, "linestyle": "--", "alpha": .85}, zorder=3)
    for scanner, point in pan_mean.iterrows():
        ax.scatter(point.delta_e76, point.frequency_high, s=120, marker="o",
                   facecolor=PALETTE[scanner], edgecolor="white", linewidth=1.3, zorder=5)
    for scanner, point in plism_mean.iterrows():
        ax.scatter(point.delta_e76, point.frequency_high, s=130, marker="^",
                   facecolor=PALETTE[scanner], edgecolor="white", linewidth=1.3, zorder=5)
    pan_offsets = {"akoya": (8, 7), "gt450": (8, 8), "s360": (10, -19),
                   "s60": (-35, -18), "versa": (-40, 14)}
    plism_offsets = {"gt450": (7, -17), "s360": (8, 9), "s60": (-45, -16),
                     "p": (9, -19), "s210": (8, 12), "sq": (-32, 14)}
    for dataset, means, offsets in [("Pan", pan_mean, pan_offsets),
                                    ("PLISM", plism_mean, plism_offsets)]:
        for scanner, point in means.iterrows():
            label = ax.annotate(f"{dataset} {scanner.upper()}",
                                (point.delta_e76, point.frequency_high),
                                xytext=offsets[scanner], textcoords="offset points",
                                fontsize=8.5, fontweight="medium", color=PALETTE[scanner], zorder=7)
            label.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white")])
    ax.scatter([0], [0], marker="+", s=70, color="black", linewidth=1.4, zorder=6)
    ax.annotate("AT2 reference", (0, 0), xytext=(6, -13), textcoords="offset points", fontsize=8)
    ax.axhline(0, color=".78", linewidth=.8, zorder=0)
    ax.set(xlim=(-1, 41), ylim=(-2.8, 1.2),
           xlabel="Paired color distance from each dataset's AT2 (CIELAB ΔE76)",
           ylabel="High-band log2 amplitude ratio to each dataset's AT2",
           title="Scanner and tissue variation in one measured space")
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="None", markerfacecolor=".45",
               markeredgecolor="white", markersize=9, label="PanNormal tissue mean"),
        Line2D([], [], marker="^", linestyle="None", markerfacecolor=".45",
               markeredgecolor="white", markersize=9, label="PLISM core mean"),
        Line2D([], [], linestyle="--", color=".45", marker=">", markersize=5,
               label="Same scanner model: PanNormal → PLISM")],
        loc="lower left", frameon=True, framealpha=.94, fontsize=8)
    fig.savefig(OUT / "scanner_tissue_space_combined.png", dpi=240)
    fig.savefig(OUT / "scanner_tissue_space_combined.pdf")
    plt.close(fig)
    print(json.dumps(result, indent=2))




def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict:
    """Compare regenerated files with the frozen files; nothing frozen is written."""
    lock = json.loads(EVIDENCE_LOCK.read_text())["files"] if EVIDENCE_LOCK.is_file() else {}
    rows = []
    for new in sorted(OUT.iterdir()):
        if new.is_dir() or new.name in {"verification.csv", "verification.json"}:
            continue
        frozen = FROZEN / new.name
        row = {"file": new.name, "frozen_path": str(frozen.relative_to(ROOT)),
               "regenerated_sha256": _sha256(new),
               "frozen_sha256": _sha256(frozen) if frozen.is_file() else "missing"}
        row["byte_identical"] = row["regenerated_sha256"] == row["frozen_sha256"]
        locked = lock.get(row["frozen_path"], {}).get("sha256")
        row["evidence_lock_sha256"] = locked or ""
        row["frozen_matches_lock"] = (locked == row["frozen_sha256"]) if locked else ""
        row["max_abs_numeric_difference"] = ""
        if not row["byte_identical"] and frozen.is_file():
            if new.suffix == ".csv":
                a, b = pd.read_csv(new), pd.read_csv(frozen)
                if list(a.columns) == list(b.columns) and len(a) == len(b):
                    numeric = a.select_dtypes("number").columns
                    text_equal = a.drop(columns=numeric).astype(str).equals(
                        b.drop(columns=numeric).astype(str))
                    row["max_abs_numeric_difference"] = float(np.nanmax(np.abs(
                        a[numeric].to_numpy(float) - b[numeric].to_numpy(float)))) \
                        if text_equal else "labels differ"
                else:
                    row["max_abs_numeric_difference"] = "shape differs"
            elif new.suffix == ".png":
                a, b = plt.imread(new), plt.imread(frozen)
                row["max_abs_numeric_difference"] = (
                    float(np.abs(a.astype(float) - b.astype(float)).max())
                    if a.shape == b.shape else f"shape {a.shape} vs {b.shape}")
            elif new.suffix == ".pdf":
                row["max_abs_numeric_difference"] = "not compared (PDF embeds creation date)"
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "verification.csv", index=False)
    key = ["pannormal_tissue_positions.csv", "plism_core_positions.csv"]
    result = {
        "producer": ".Trash/2026-09-26_paper_refactor/unused_analysis/analysis/paper/scanner_tissue_space.py",
        "original_job": "23553153 (sbatch analysis/run_scanner_tissue_space.sbatch, workdir 00_manuscript, 2026-09-24 16:55)",
        "fig2_inputs_byte_identical": bool(table.set_index("file").loc[key, "byte_identical"].all()),
        "files_compared": int(len(table)),
        "files_byte_identical": int(table.byte_identical.sum()),
        "comparison_rule": "byte-identical SHA-256; otherwise numeric max |difference| (CSV) or pixel max |difference| (PNG, 0-1 scale)",
    }
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(table.to_string(index=False))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
    verify()
