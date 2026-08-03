"""Build journal-layout Main Figures 1--3 from locked E0--E3 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


SCANNERS = ("gt450", "versa", "akoya", "s60", "s360")
LABELS = {
    "at2": "AT2",
    "gt450": "GT450",
    "versa": "VERSA",
    "akoya": "AKOYA",
    "s60": "S60",
    "s360": "S360",
}
COLORS = {
    "at2": "#6B7280",
    "gt450": "#F59E0B",
    "versa": "#10B981",
    "akoya": "#8B5CF6",
    "s60": "#0EA5E9",
    "s360": "#EC4899",
}
BANDS = ("low_mid", "mid", "high")

SOURCES = {
    "geometry": "outputs/e0_native_geometry_final/summary.json",
    "grid": "outputs/e0_native_aa_grid/audit/summary.json",
    "fallback_actions": "outputs/e0_native_fallback_actions/actions.csv",
    "alias": "outputs/e0_alias_audit_final_geometry/scanner_profile_summary.csv",
    "background": "outputs/e0d_same_chain_background/same_chain_snr_summary.csv",
    "noise_sensitivity": "outputs/e0d_same_chain_background/noise_floor_band_sensitivity_summary.csv",
    "e1_summary": "outputs/e1_native_aa_spectral_109/summary.json",
    "spectra": "outputs/e1_native_aa_spectral_109/slide_spectra.csv",
    "tissue_variance": "outputs/e1_native_aa_spectral_109/tissue_random_slope_variance_components.csv",
    "tissue_slopes": "outputs/e1_native_aa_spectral_109/tissue_random_slopes.csv",
    "e3": "outputs/e3_native_aa_scalar_reducibility/scanner_summary.csv",
}

PDF_METADATA = {
    "Creator": "prenorm build_e0_e3_main_figures.py",
    "CreationDate": None,
    "ModDate": None,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="outputs/e0_e3_main_figures")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize_alias_profiles(frame: pd.DataFrame):
    required = {
        "scanner",
        "transform_profile",
        "pipeline",
        "sinusoid_alias_to_inband_ratio",
        "white_noise_alias_to_inband_ratio_max",
        "alias_power_ratio_limit",
        "alias_gate_pass",
    }
    require(required.issubset(frame.columns), "alias table schema differs")
    result = frame.copy()
    result["worst_alias_ratio"] = result[
        ["sinusoid_alias_to_inband_ratio", "white_noise_alias_to_inband_ratio_max"]
    ].max(axis=1)
    require(
        set(result["pipeline"]) == {"original_bicubic", "explicit_aa_lanczos3"},
        "alias pipeline set differs",
    )
    require(len(result) == 12, "alias profile count differs")
    explicit = result[result["pipeline"].eq("explicit_aa_lanczos3")]
    require(explicit["alias_gate_pass"].astype(str).str.lower().eq("true").all(), "explicit-AA gate failed")
    require(explicit["worst_alias_ratio"].max() <= 0.05, "explicit-AA alias ratio exceeds 0.05")
    return result


def select_background_high(frame: pd.DataFrame):
    result = frame[frame["band"].eq("high")].copy()
    expected = {"at2", *SCANNERS}
    require(set(result["scanner"]) == expected, "background scanner set differs")
    require(result["negative_after_subtraction_fraction"].eq(0).all(), "negative subtraction found")
    return result.set_index("scanner").loc[["at2", *SCANNERS]].reset_index()


def summarize_geometry_recovery(actions: pd.DataFrame, cells_per_scanner: int = 109):
    require({"scanner", "action"}.issubset(actions.columns), "fallback action schema differs")
    counts = actions.groupby("scanner").size().reindex(SCANNERS, fill_value=0)
    require(int(counts.sum()) == 61, "primary geometry failure count differs")
    require((counts <= cells_per_scanner).all(), "invalid primary geometry failure count")
    return pd.DataFrame(
        {
            "scanner": SCANNERS,
            "primary_pass_fraction": (cells_per_scanner - counts.to_numpy()) / cells_per_scanner,
            "final_pass_fraction": np.ones(len(SCANNERS)),
            "primary_failures": counts.to_numpy(),
        }
    )


def configure_style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_panel_title(axis, letter, title):
    axis.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=8)


def draw_design(axis):
    axis.set_axis_off()
    boxes = [
        (0.76, "109 physical slides\n37 normal tissue types", "#E0F2FE"),
        (0.53, "6 scanners\nsame physical section", "#DCFCE7"),
        (0.30, "100 matched locations/slide\n10,900 six-scanner tuples", "#FEF3C7"),
        (0.07, "Native WSI → explicit AA\n65,400 analysis patches", "#F3E8FF"),
    ]
    for index, (y, text, color) in enumerate(boxes):
        patch = FancyBboxPatch(
            (0.09, y),
            0.82,
            0.14,
            boxstyle="round,pad=0.015,rounding_size=0.015",
            transform=axis.transAxes,
            facecolor=color,
            edgecolor="#334155",
            linewidth=1.0,
        )
        axis.add_patch(patch)
        axis.text(0.5, y + 0.07, text, ha="center", va="center", transform=axis.transAxes)
        if index < len(boxes) - 1:
            axis.annotate(
                "",
                xy=(0.5, y - 0.055),
                xytext=(0.5, y - 0.005),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": "#475569", "lw": 1.2},
            )
    add_panel_title(axis, "A", "Paired native-WSI study design")


def draw_degeneracy(axis):
    axis.set_axis_off()
    centers = [(0.17, 0.68), (0.31, 0.70), (0.22, 0.49)]
    colors = (COLORS["gt450"], COLORS["versa"], COLORS["akoya"])
    offsets = np.array([[-0.018, 0.0], [0.014, 0.012], [0.006, -0.018], [-0.012, -0.014]])
    for (center_x, center_y), color in zip(centers, colors):
        axis.scatter(center_x + offsets[:, 0], center_y + offsets[:, 1], s=24, color=color, transform=axis.transAxes)
    axis.text(0.24, 0.84, "Scanner-separable\nrepresentations", ha="center", va="center", transform=axis.transAxes)
    axis.annotate(
        "minimize scanner\nseparability only",
        xy=(0.69, 0.60),
        xytext=(0.43, 0.60),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        arrowprops={"arrowstyle": "-|>", "color": "#475569", "lw": 1.3},
        fontsize=8,
    )
    for color in colors:
        axis.scatter([0.78], [0.60], s=58, color=color, alpha=0.75, transform=axis.transAxes)
    collapsed = FancyBboxPatch(
        (0.67, 0.42),
        0.23,
        0.36,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        transform=axis.transAxes,
        facecolor="#FEE2E2",
        edgecolor="#B91C1C",
        linewidth=1.0,
        zorder=-1,
    )
    axis.add_patch(collapsed)
    axis.text(0.785, 0.48, "constant / collapsed\ninvariance ↑; content = 0", ha="center", va="center", transform=axis.transAxes, fontsize=8)
    axis.text(
        0.50,
        0.17,
        "Therefore every correction must pass paired-content and collapse guardrails.",
        ha="center",
        va="center",
        transform=axis.transAxes,
        fontsize=8.3,
        color="#7F1D1D",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFF7ED", "edgecolor": "#FDBA74"},
    )
    add_panel_title(axis, "B", "Why invariance alone is non-identifying")


def make_figure1(geometry, grid, actions, alias, background, noise, output: Path, dpi: int):
    figure, axes = plt.subplots(2, 3, figsize=(16.2, 8.2), constrained_layout=True)
    draw_design(axes[0, 0])
    draw_degeneracy(axes[0, 1])

    axis = axes[0, 2]
    recovery = summarize_geometry_recovery(actions)
    x = np.arange(len(SCANNERS))
    width = 0.36
    axis.bar(
        x - width / 2,
        100 * recovery["primary_pass_fraction"],
        width=width,
        color="#94A3B8",
        label="Initial native geometry",
    )
    axis.bar(
        x + width / 2,
        100 * recovery["final_pass_fraction"],
        width=width,
        color="#2563EB",
        label="Final outcome-blind route",
    )
    axis.axhline(100, color="#475569", lw=0.8)
    axis.set_ylim(60, 103)
    axis.set_xticks(x, [LABELS[value] for value in SCANNERS], rotation=22)
    axis.set_ylabel("Passing scanner–slide cells (%)")
    axis.legend(frameon=False, fontsize=7, loc="lower right")
    worst_index = int(recovery["primary_failures"].to_numpy().argmax())
    axis.text(
        worst_index - width / 2,
        100 * recovery.loc[worst_index, "primary_pass_fraction"] + 1.5,
        f"{int(recovery.loc[worst_index, 'primary_failures'])} failed",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    add_panel_title(axis, "C", "Outcome-blind geometry recovery")

    axis = axes[1, 0]
    counts = [
        ("Locations", geometry["locations_passing"], geometry["location_rows_expected"]),
        ("Scanner–slide cells", geometry["cells_passing"], geometry["cells_expected"]),
        ("Six-scanner tuples", geometry["six_scanner_tuples_passing"], geometry["six_scanner_tuples_expected"]),
        ("Native-AA patches", grid["patches_observed"], grid["patches_expected"]),
    ]
    values = [100 * passed / expected for _, passed, expected in counts]
    positions = np.arange(len(counts))
    axis.barh(positions, values, color=["#2563EB", "#0EA5E9", "#14B8A6", "#8B5CF6"])
    axis.set_yticks(positions, [label for label, _, _ in counts])
    axis.invert_yaxis()
    axis.set_xlim(0, 105)
    axis.set_xlabel("Final gate retention (%)")
    axis.axvline(100, color="#475569", lw=0.8)
    for position, (label, passed, expected) in enumerate(counts):
        axis.text(99, position, f"{passed:,}/{expected:,}", ha="right", va="center", color="white", fontweight="bold")
    add_panel_title(axis, "D", "Final geometry and render gates")

    axis = axes[1, 1]
    summary = summarize_alias_profiles(alias)
    profile_order = [f"{scanner}\n{profile}" for scanner in ("gt450", "versa") for profile in ("q05", "q50", "q95")]
    x = np.arange(len(profile_order))
    for pipeline, offset, marker, color, label in (
        ("original_bicubic", -0.08, "o", "#64748B", "Historical bicubic"),
        ("explicit_aa_lanczos3", 0.08, "D", "#2563EB", "Explicit AA"),
    ):
        selected = summary[summary["pipeline"].eq(pipeline)].copy()
        selected["key"] = selected["scanner"] + "\n" + selected["transform_profile"]
        selected = selected.set_index("key").loc[profile_order]
        axis.scatter(x + offset, selected["worst_alias_ratio"], s=38, marker=marker, color=color, label=label, zorder=3)
    axis.axhline(0.05, color="#DC2626", linestyle="--", lw=1.1, label="Prespecified 0.05 gate")
    axis.set_yscale("log")
    axis.set_ylim(0.008, 2.5)
    axis.set_xticks(x, [value.replace("gt450", "GT450").replace("versa", "VERSA") for value in profile_order])
    axis.set_ylabel("Worst alias / true in-band power")
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.grid(axis="y", which="both", alpha=0.18)
    add_panel_title(axis, "E", "Final-transform alias audit")

    axis = axes[1, 2]
    high = select_background_high(background)
    x = np.arange(len(high))
    medians = high["background_fraction_median"].to_numpy()
    q95 = high["background_fraction_q95"].to_numpy()
    axis.errorbar(
        x,
        medians,
        yerr=np.vstack([np.zeros_like(medians), q95 - medians]),
        fmt="o",
        markersize=6,
        capsize=3,
        color="#0F766E",
        ecolor="#5EEAD4",
        linewidth=1.5,
    )
    axis.set_yscale("log")
    axis.set_ylim(2e-4, 1e-1)
    axis.set_xticks(x, [LABELS[value] for value in high["scanner"]], rotation=20)
    axis.set_ylabel("Background / tissue power\nmedian to q95")
    axis.grid(axis="y", which="both", alpha=0.18)
    primary_noise = noise[
        noise["band"].eq("high")
        & noise["minimum_snr_threshold"].eq(1.0)
        & ~noise["scanner"].eq("at2")
    ]
    max_delta = primary_noise["median_delta_corrected_minus_raw_log2"].abs().max()
    axis.text(
        0.02,
        0.97,
        f"47,088/47,088 subtracted bins positive\nmax |median ΔERT| = {max_delta:.4f} log₂",
        transform=axis.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CBD5E1"},
    )
    add_panel_title(axis, "F", "Same-chain operational background floor")

    figure.savefig(output / "figure1_study_estimator_validity.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(
        output / "figure1_study_estimator_validity.pdf",
        bbox_inches="tight",
        metadata=PDF_METADATA,
    )
    plt.close(figure)


def make_figure2(spectra, e1_summary, e3, output: Path, dpi: int):
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 4.6), constrained_layout=True)

    axis = axes[0]
    selected = spectra[
        spectra["scanner"].isin(SCANNERS)
        & spectra["frequency_cyc_per_um"].between(0.03, 0.95)
    ]
    grouped = (
        selected.groupby(["scanner", "frequency_cyc_per_um"], observed=True)["log2_relative_transfer"]
        .agg(median="median", q10=lambda values: values.quantile(0.1), q90=lambda values: values.quantile(0.9))
        .reset_index()
    )
    for scanner in SCANNERS:
        rows = grouped[grouped["scanner"].eq(scanner)]
        frequency = rows["frequency_cyc_per_um"].to_numpy()
        axis.plot(frequency, rows["median"], color=COLORS[scanner], lw=2.0, label=LABELS[scanner])
        axis.fill_between(frequency, rows["q10"], rows["q90"], color=COLORS[scanner], alpha=0.13, linewidth=0)
    axis.axhline(0, color="#64748B", linestyle="--", lw=0.9)
    axis.axvspan(0.60, 0.90, color="#E2E8F0", alpha=0.45, zorder=-2)
    axis.set_xlim(0.03, 0.95)
    axis.set_xlabel("Spatial frequency (cycles/µm)")
    axis.set_ylabel("Relative transfer to AT2 (log₂)")
    axis.legend(frameon=False, ncol=2)
    add_panel_title(axis, "A", "Cohort ERT: median and 10–90% of slides")

    axis = axes[1]
    rows = {row["scanner"]: row for row in e1_summary["high_band_results"]}
    x = np.arange(len(SCANNERS))
    estimate = np.array([rows[scanner]["fixed_mean_log2_transfer"] for scanner in SCANNERS])
    lower = np.array([rows[scanner]["fixed_ci95_low"] for scanner in SCANNERS])
    upper = np.array([rows[scanner]["fixed_ci95_high"] for scanner in SCANNERS])
    for index, scanner in enumerate(SCANNERS):
        axis.errorbar(
            index,
            estimate[index],
            yerr=[[estimate[index] - lower[index]], [upper[index] - estimate[index]]],
            fmt="o",
            color=COLORS[scanner],
            capsize=4,
            markersize=7,
        )
        fold = rows[scanner]["fixed_fold_transfer"]
        offset = 0.10 if estimate[index] >= 0 else -0.14
        axis.text(index, estimate[index] + offset, f"{fold:.3f}×", ha="center", va="center", fontsize=8)
    axis.axhline(0, color="#64748B", linestyle="--", lw=0.9)
    axis.set_xticks(x, [LABELS[value] for value in SCANNERS], rotation=25)
    axis.set_ylabel("High-band fixed effect (log₂, 95% CI)")
    add_panel_title(axis, "B", "Audited high-band scanner effects")

    axis = axes[2]
    table = e3.set_index("scanner").loc[list(SCANNERS)]
    estimate = table["excess_off_axis_rms_median"].to_numpy()
    lower = table["excess_off_axis_rms_median_ci95_low"].to_numpy()
    upper = table["excess_off_axis_rms_median_ci95_high"].to_numpy()
    for index, scanner in enumerate(SCANNERS):
        axis.errorbar(
            index,
            estimate[index],
            yerr=[[estimate[index] - lower[index]], [upper[index] - estimate[index]]],
            fmt="o",
            color=COLORS[scanner],
            capsize=4,
            markersize=7,
        )
    axis.axhline(0, color="#DC2626", linestyle="--", lw=1.0)
    axis.set_ylim(0, max(upper) * 1.28)
    axis.set_xticks(x, [LABELS[value] for value in SCANNERS], rotation=25)
    axis.set_ylabel("Excess off-axis RMS (log₂, 95% bootstrap CI)")
    boundary_scanner = table["gain_boundary_fraction"].idxmax()
    axis.text(
        0.03,
        0.96,
        f"{LABELS[boundary_scanner]}: "
        f"{100 * table.loc[boundary_scanner, 'gain_boundary_fraction']:.1f}% at gain bound",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#CBD5E1"},
    )
    add_panel_title(axis, "C", "LOSO scalar blur–sharpen lack of fit")

    figure.savefig(output / "figure2_audited_scanner_spectrum.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(
        output / "figure2_audited_scanner_spectrum.pdf",
        bbox_inches="tight",
        metadata=PDF_METADATA,
    )
    plt.close(figure)


def make_figure3(variance, slopes, output: Path, dpi: int):
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), constrained_layout=True)

    high = variance[variance["band"].eq("high")].set_index("scanner").loc[list(SCANNERS)]
    axis = axes[0]
    x = np.arange(len(SCANNERS))
    tissue_values = high["scanner_by_tissue_variance"].to_numpy()
    slide_values = high["scanner_by_slide_within_tissue_variance"].to_numpy()
    sampling_values = high["sampling_variance_100patch"].to_numpy()
    axis.bar(x, tissue_values, color="#8B5CF6", label="scanner × tissue")
    axis.bar(x, slide_values, bottom=tissue_values, color="#38BDF8", label="scanner × slide | tissue")
    axis.bar(
        x,
        sampling_values,
        bottom=tissue_values + slide_values,
        color="#CBD5E1",
        label="100-patch sampling",
    )
    axis.set_xticks(x, [LABELS[value] for value in SCANNERS], rotation=25)
    axis.set_ylabel("Variance (log₂ transfer²)")
    axis.legend(frameon=False, fontsize=7)
    add_panel_title(axis, "A", "High-band nested variance decomposition")

    axis = axes[1]
    fraction = variance.pivot(index="band", columns="scanner", values="tissue_fraction_between_slide_variance")
    qvalues = variance.pivot(index="band", columns="scanner", values="tissue_variance_q_bh")
    fraction = fraction.loc[list(BANDS), list(SCANNERS)]
    qvalues = qvalues.loc[list(BANDS), list(SCANNERS)]
    image = axis.imshow(fraction.to_numpy(), cmap="Purples", vmin=0, vmax=1, aspect="auto")
    for row in range(len(BANDS)):
        for column in range(len(SCANNERS)):
            value = fraction.iloc[row, column]
            star = "*" if qvalues.iloc[row, column] < 0.05 else ""
            color = "white" if value > 0.55 else "#111827"
            axis.text(column, row, f"{100 * value:.0f}%{star}", ha="center", va="center", color=color, fontsize=8)
    axis.set_xticks(x, [LABELS[value] for value in SCANNERS], rotation=25)
    axis.set_yticks(np.arange(len(BANDS)), ["low–mid", "mid", "high"])
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("Tissue share")
    axis.text(0, -0.24, "* BH q < 0.05", transform=axis.transAxes, fontsize=7, color="#475569")
    add_panel_title(axis, "B", "Tissue share of between-slide variance")

    axis = axes[2]
    high_slopes = slopes[slopes["band"].eq("high")].copy()
    tissues = sorted(high_slopes["tissue_type"].unique())
    matrix = (
        high_slopes.pivot(index="scanner", columns="tissue_type", values="blup_tissue_slope")
        .loc[list(SCANNERS), tissues]
        .to_numpy()
    )
    limit = max(0.3, float(np.nanmax(np.abs(matrix))))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_yticks(np.arange(len(SCANNERS)), [LABELS[value] for value in SCANNERS])
    axis.set_xticks(np.arange(len(tissues)), tissues, rotation=75, ha="right", fontsize=5.7)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("Tissue BLUP deviation (log₂)")
    add_panel_title(axis, "C", "High-band tissue-specific scanner slopes")

    figure.savefig(output / "figure3_scanner_content_structure.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(
        output / "figure3_scanner_content_structure.pdf",
        bbox_inches="tight",
        metadata=PDF_METADATA,
    )
    plt.close(figure)


def run(repo_root: Path, output: Path, dpi: int):
    source_paths = {name: repo_root / relative for name, relative in SOURCES.items()}
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    require(not missing, f"missing source artifacts: {missing}")
    geometry = json.loads(source_paths["geometry"].read_text())
    grid = json.loads(source_paths["grid"].read_text())
    e1_summary = json.loads(source_paths["e1_summary"].read_text())
    alias = pd.read_csv(source_paths["alias"])
    actions = pd.read_csv(source_paths["fallback_actions"], dtype={"slide_id": str})
    background = pd.read_csv(source_paths["background"])
    noise = pd.read_csv(source_paths["noise_sensitivity"])
    spectra = pd.read_csv(source_paths["spectra"], dtype={"slide_id": str})
    variance = pd.read_csv(source_paths["tissue_variance"])
    slopes = pd.read_csv(source_paths["tissue_slopes"])
    e3 = pd.read_csv(source_paths["e3"])

    require(geometry.get("cohort_gate_pass") is True, "geometry gate failed")
    require(grid.get("grid_gate_pass") is True, "native-AA grid gate failed")
    require(spectra["slide_id"].nunique() == 109, "spectrum slide count differs")
    require(set(spectra["scanner"]) == {"at2", *SCANNERS}, "spectrum scanner set differs")
    require(slopes["tissue_type"].nunique() == 37, "tissue count differs")

    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    make_figure1(geometry, grid, actions, alias, background, noise, output, dpi)
    make_figure2(spectra, e1_summary, e3, output, dpi)
    make_figure3(variance, slopes, output, dpi)

    outputs = sorted(output.glob("figure[123]_*.png")) + sorted(output.glob("figure[123]_*.pdf"))
    require(len(outputs) == 6, "expected three PNG/PDF figure pairs")
    summary = {
        "analysis": "e0_e3_main_figure_build",
        "source_artifacts": {
            name: {"path": SOURCES[name], "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "figures": [
            {"path": str(path.relative_to(repo_root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in outputs
        ],
        "figure_1_complete": True,
        "figure_2_complete": True,
        "figure_3_complete": True,
        "e4_e7_outcomes_used": False,
        "build_pass": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    summary = run(repo_root, output, args.dpi)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
