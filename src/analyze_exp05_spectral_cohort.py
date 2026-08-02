"""Aggregate Exp-05 slide shards and estimate scanner-by-slide variation.

The balanced five-replicate design permits a closed-form random-intercept
variance decomposition for each scanner and frequency band:

    y_slide,replicate = scanner fixed mean + slide-specific scanner slope
                        + patch-sampling error.

The slide-specific term is the scanner-by-slide interaction.  Tissue is not
modeled here because the registered-pair registry contains no tissue label;
all tables retain ``slide_id`` so that labels can be merged later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_exp05_spectral_pilot import (
    BANDS,
    SCANNERS,
    SCANNER_COLORS,
    render_blur_sharpen_bridge,
    save_figure,
)


TABLE_FILES = {
    "spectra": "slide_spectra.csv",
    "bands": "slide_band_summary.csv",
    "replicate_bands": "slide_band_replicates.csv",
    "interventions": "exp02_hf_gain_spectra.csv",
    "projection": "scanner_hf_gain_projection.csv",
    "patches": "selected_patches.csv",
    "alignment": "patch_alignment.csv",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards",
        default="outputs/exp05_spectral_cohort_109/shards",
    )
    parser.add_argument(
        "--output",
        default="outputs/exp05_spectral_cohort_109",
    )
    parser.add_argument("--expected-slides", type=int, default=109)
    parser.add_argument("--expected-patches", type=int, default=100)
    parser.add_argument("--expected-replicates", type=int, default=5)
    return parser.parse_args(argv)


def read_shards(shard_root: Path, expected_slides: int):
    shard_dirs = sorted(
        path for path in shard_root.iterdir() if path.is_dir()
    )
    if len(shard_dirs) != expected_slides:
        raise ValueError(
            f"expected {expected_slides} shard directories, got {len(shard_dirs)}"
        )
    tables = {}
    for name, filename in TABLE_FILES.items():
        paths = [path / filename for path in shard_dirs]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{name}: {len(missing)} missing shard tables; first={missing[0]}"
            )
        tables[name] = pd.concat(
            [pd.read_csv(path, dtype={"slide_id": str}) for path in paths],
            ignore_index=True,
        )
    return tables


def validate_tables(tables, expected_slides, expected_patches, expected_replicates):
    expected_ids = set(tables["spectra"]["slide_id"].astype(str))
    if len(expected_ids) != expected_slides:
        raise AssertionError(
            f"expected {expected_slides} slide IDs, got {len(expected_ids)}"
        )
    for name, frame in tables.items():
        found = set(frame["slide_id"].astype(str))
        if found != expected_ids:
            raise AssertionError(f"{name} has inconsistent slide IDs")
        numeric = frame.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy()).all():
            raise AssertionError(f"{name} contains non-finite numeric values")
    patch_counts = tables["patches"].groupby("slide_id").size()
    if not (patch_counts == expected_patches).all():
        raise AssertionError("not every slide has the requested patch count")
    replicate_counts = (
        tables["replicate_bands"]
        .groupby(["slide_id", "scanner", "band"])["replicate"]
        .nunique()
    )
    if not (replicate_counts == expected_replicates).all():
        raise AssertionError("replicate-band cells are incomplete")
    at2 = tables["replicate_bands"].query("scanner == 'at2'")[
        "log2_relative_transfer"
    ].to_numpy()
    if not np.allclose(at2, 0.0, atol=1e-10):
        raise AssertionError("AT2 replicate self-transfer is not zero")


def variance_components(replicates: pd.DataFrame):
    """Balanced one-way REML/method-of-moments estimates per scanner-band."""
    rows = []
    random_rows = []
    for (band, scanner), frame in replicates.query(
        "scanner != 'at2'"
    ).groupby(["band", "scanner"], sort=True):
        value = "log2_relative_transfer"
        counts = frame.groupby("slide_id")[value].size()
        if counts.nunique() != 1:
            raise ValueError(f"unbalanced replicate count: {band}/{scanner}")
        n_replicates = int(counts.iloc[0])
        slide_means = frame.groupby("slide_id", sort=True)[value].mean()
        n_slides = len(slide_means)
        fixed_mean = float(slide_means.mean())

        with_mean = frame["slide_id"].map(slide_means)
        within_ss = float(np.square(frame[value] - with_mean).sum())
        within_df = n_slides * (n_replicates - 1)
        residual_variance = within_ss / within_df
        between_ms = (
            n_replicates
            * float(np.square(slide_means - fixed_mean).sum())
            / (n_slides - 1)
        )
        slide_variance = max(
            (between_ms - residual_variance) / n_replicates, 0.0
        )
        observed_slide_mean_variance = float(slide_means.var(ddof=1))
        standard_error = np.sqrt(observed_slide_mean_variance / n_slides)
        t_value = fixed_mean / standard_error if standard_error > 0 else np.nan
        p_value = (
            float(2 * stats.t.sf(abs(t_value), df=n_slides - 1))
            if np.isfinite(t_value)
            else np.nan
        )
        critical = float(stats.t.ppf(0.975, df=n_slides - 1))
        ci_low = fixed_mean - critical * standard_error
        ci_high = fixed_mean + critical * standard_error
        final_sampling_variance = residual_variance / n_replicates
        reliability = (
            slide_variance / (slide_variance + final_sampling_variance)
            if slide_variance + final_sampling_variance > 0
            else 0.0
        )
        rows.append({
            "band": band,
            "scanner": scanner,
            "n_slides": n_slides,
            "replicate_groups": n_replicates,
            "patches_per_replicate": int(frame["patches_in_replicate"].median()),
            "fixed_mean_log2_transfer": fixed_mean,
            "fixed_fold_transfer": float(2.0 ** fixed_mean),
            "fixed_se": float(standard_error),
            "fixed_ci95_low": float(ci_low),
            "fixed_ci95_high": float(ci_high),
            "t_vs_at2": float(t_value),
            "p_vs_at2": p_value,
            "scanner_by_slide_variance": float(slide_variance),
            "scanner_by_slide_sd": float(np.sqrt(slide_variance)),
            "replicate_sampling_variance": float(residual_variance),
            "replicate_sampling_sd": float(np.sqrt(residual_variance)),
            "sampling_variance_100patch": float(final_sampling_variance),
            "sampling_sd_100patch": float(np.sqrt(final_sampling_variance)),
            "slide_slope_reliability": float(reliability),
            "slide_fraction_of_100patch_variance": float(reliability),
        })
        for slide_id, slide_mean in slide_means.items():
            raw_deviation = float(slide_mean - fixed_mean)
            random_rows.append({
                "slide_id": str(slide_id),
                "band": band,
                "scanner": scanner,
                "slide_mean_log2_transfer": float(slide_mean),
                "fixed_mean_log2_transfer": fixed_mean,
                "raw_slide_slope": raw_deviation,
                "blup_slide_slope": float(reliability * raw_deviation),
                "reliability": float(reliability),
            })
    return pd.DataFrame(rows), pd.DataFrame(random_rows)


def random_slope_correlations(replicates: pd.DataFrame):
    """Estimate multivariate random-slope covariance after sampling correction."""
    rows = []
    matrices = {}
    for band, frame in replicates.query(
        "scanner != 'at2'"
    ).groupby("band", sort=True):
        wide = frame.pivot(
            index=["slide_id", "replicate"],
            columns="scanner",
            values="log2_relative_transfer",
        ).reindex(columns=SCANNERS[1:])
        slide_means = wide.groupby(level="slide_id").mean()
        repeated_means = slide_means.loc[
            wide.index.get_level_values("slide_id")
        ].to_numpy()
        residual = wide.to_numpy() - repeated_means
        n_slides = len(slide_means)
        n_replicates = wide.index.get_level_values("replicate").nunique()
        within_covariance = (
            residual.T @ residual / (n_slides * (n_replicates - 1))
        )
        observed_mean_covariance = np.cov(
            slide_means.to_numpy(), rowvar=False, ddof=1
        )
        random_slope_covariance = (
            observed_mean_covariance - within_covariance / n_replicates
        )
        diagonal = np.diag(random_slope_covariance)
        if np.any(diagonal <= 0):
            raise ValueError(f"non-positive random-slope variance in {band}")
        correlation_values = random_slope_covariance / np.sqrt(
            np.outer(diagonal, diagonal)
        )
        correlation = pd.DataFrame(
            correlation_values,
            index=SCANNERS[1:],
            columns=SCANNERS[1:],
        )
        matrices[band] = correlation
        for row_index, scanner_a in enumerate(correlation.index):
            for column_index, scanner_b in enumerate(correlation.columns):
                rows.append({
                    "band": band,
                    "scanner_a": scanner_a,
                    "scanner_b": scanner_b,
                    "random_slope_covariance": float(
                        random_slope_covariance[row_index, column_index]
                    ),
                    "within_replicate_covariance": float(
                        within_covariance[row_index, column_index]
                    ),
                    "correlation": float(correlation.loc[scanner_a, scanner_b]),
                })
    return pd.DataFrame(rows), matrices


def render_cohort_overview(spectra, bands, output_dir):
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.1))

    axis = axes[0]
    for scanner in SCANNERS[1:]:
        selected = spectra[spectra["scanner"] == scanner]
        aggregate = (
            selected.groupby("frequency_cyc_per_um")["log2_relative_transfer"]
            .agg(
                median="median",
                q10=lambda value: value.quantile(0.10),
                q90=lambda value: value.quantile(0.90),
            )
            .reset_index()
        )
        x = aggregate["frequency_cyc_per_um"].to_numpy()
        axis.plot(
            x,
            aggregate["median"],
            color=SCANNER_COLORS[scanner],
            lw=2.2,
            label=scanner.upper(),
        )
        axis.fill_between(
            x,
            aggregate["q10"],
            aggregate["q90"],
            color=SCANNER_COLORS[scanner],
            alpha=0.13,
        )
    axis.axhline(0, color="#6B7280", lw=1, ls="--")
    axis.set(
        xlim=(0.03, 0.95),
        xlabel="Spatial frequency (cycles/µm)",
        ylabel="Relative transfer to AT2 (log₂)",
        title="A  Cohort curves: median and 10–90% slides",
    )
    axis.legend(frameon=False, ncol=2, fontsize=8)

    axis = axes[1]
    high = bands.query("band == 'high' and scanner != 'at2'")
    values = [
        high.loc[high["scanner"] == scanner, "log2_relative_transfer"].to_numpy()
        for scanner in SCANNERS[1:]
    ]
    violins = axis.violinplot(values, showextrema=False, showmedians=False)
    for body, scanner in zip(violins["bodies"], SCANNERS[1:]):
        body.set_facecolor(SCANNER_COLORS[scanner])
        body.set_edgecolor("none")
        body.set_alpha(0.25)
    jitter_rng = np.random.default_rng(20260731)
    for position, (scanner, value) in enumerate(zip(SCANNERS[1:], values), start=1):
        axis.scatter(
            position + jitter_rng.normal(0, 0.035, len(value)),
            value,
            s=8,
            color=SCANNER_COLORS[scanner],
            alpha=0.28,
            linewidth=0,
        )
        axis.plot(
            [position - 0.16, position + 0.16],
            [np.mean(value), np.mean(value)],
            color=SCANNER_COLORS[scanner],
            lw=3,
        )
    axis.axhline(0, color="#6B7280", lw=1, ls="--")
    axis.set_xticks(range(1, len(SCANNERS)), [s.upper() for s in SCANNERS[1:]])
    axis.tick_params(axis="x", rotation=30)
    axis.set(
        ylabel="0.60–0.90 cycles/µm (log₂ transfer)",
        title="B  Scanner main effects and slide spread",
    )

    axis = axes[2]
    matrix = high.pivot(
        index="scanner", columns="slide_id", values="log2_relative_transfer"
    ).reindex(index=SCANNERS[1:])
    order = leaves_list(linkage(matrix.to_numpy().T, method="average"))
    matrix = matrix.iloc[:, order]
    limit = max(1.0, float(np.nanquantile(np.abs(matrix.to_numpy()), 0.98)))
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    axis.set_yticks(range(len(SCANNERS) - 1), [s.upper() for s in SCANNERS[1:]])
    tick_positions = np.linspace(0, matrix.shape[1] - 1, 8).astype(int)
    axis.set_xticks(tick_positions, [matrix.columns[i] for i in tick_positions], rotation=45, ha="right", fontsize=7)
    axis.set_title("C  109 slide signatures (unsupervised order)")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("High-band log₂ transfer")

    figure.suptitle(
        "Paired-scanner effective transfer · 109 slides × 100 patches",
        fontsize=15,
        y=1.03,
    )
    figure.tight_layout()
    return save_figure(figure, output_dir, "figure1_cohort_spectral_overview")


def render_mixed_effects(components, matrices, output_dir):
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.1))
    band_order = list(BANDS)

    axis = axes[0]
    base = np.arange(len(SCANNERS) - 1)
    offsets = np.linspace(-0.22, 0.22, len(band_order))
    band_colors = {"low_mid": "#4C78A8", "mid": "#72B7B2", "high": "#E45756"}
    for offset, band in zip(offsets, band_order):
        frame = components[components["band"] == band].set_index("scanner").reindex(SCANNERS[1:])
        mean = frame["fixed_mean_log2_transfer"].to_numpy()
        lower = mean - frame["fixed_ci95_low"].to_numpy()
        upper = frame["fixed_ci95_high"].to_numpy() - mean
        axis.errorbar(
            base + offset,
            mean,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            ms=5,
            capsize=2,
            color=band_colors[band],
            label=band.replace("_", "–"),
        )
    axis.axhline(0, color="#6B7280", lw=1, ls="--")
    axis.set_xticks(base, [s.upper() for s in SCANNERS[1:]], rotation=30)
    axis.set(ylabel="Fixed effect (log₂ transfer to AT2)", title="A  Scanner fixed effects (95% CI)")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    high = components[components["band"] == "high"].set_index("scanner").reindex(SCANNERS[1:])
    slide_var = high["scanner_by_slide_variance"].to_numpy()
    sampling_var = high["sampling_variance_100patch"].to_numpy()
    axis.bar(base, slide_var, color=[SCANNER_COLORS[s] for s in SCANNERS[1:]], alpha=0.78, label="scanner × slide")
    axis.bar(base, sampling_var, bottom=slide_var, color="#D1D5DB", label="100-patch sampling")
    axis.set_xticks(base, [s.upper() for s in SCANNERS[1:]], rotation=30)
    axis.set(ylabel="Variance (log₂ transfer²)", title="B  High-band variance decomposition")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[2]
    correlation = matrices["high"].reindex(index=SCANNERS[1:], columns=SCANNERS[1:])
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    for row in range(len(correlation)):
        for column in range(len(correlation)):
            value = correlation.iloc[row, column]
            axis.text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.55 else "black",
            )
    labels = [s.upper() for s in SCANNERS[1:]]
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("C  Sampling-corrected random-slope correlation")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson r across slides")

    figure.suptitle(
        "Balanced random-slope decomposition · five 20-patch replicates per slide",
        fontsize=15,
        y=1.03,
    )
    figure.tight_layout()
    return save_figure(figure, output_dir, "figure2_mixed_effects")


def main(argv=None):
    args = parse_args(argv)
    shard_root = Path(args.shards)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = read_shards(shard_root, args.expected_slides)
    validate_tables(
        tables,
        args.expected_slides,
        args.expected_patches,
        args.expected_replicates,
    )
    combined_paths = {}
    for name, frame in tables.items():
        path = output_dir / TABLE_FILES[name]
        frame.to_csv(path, index=False)
        combined_paths[name] = str(path)

    components, random_slopes = variance_components(tables["replicate_bands"])
    correlations, matrices = random_slope_correlations(
        tables["replicate_bands"]
    )
    component_path = output_dir / "mixed_effect_variance_components.csv"
    slope_path = output_dir / "slide_random_slopes.csv"
    correlation_path = output_dir / "random_slope_correlations.csv"
    components.to_csv(component_path, index=False)
    random_slopes.to_csv(slope_path, index=False)
    correlations.to_csv(correlation_path, index=False)

    scanner_summary = (
        tables["projection"]
        .groupby("scanner")
        .agg(
            equivalent_hf_gain_mean=("equivalent_hf_gain", "mean"),
            equivalent_hf_gain_sd=("equivalent_hf_gain", "std"),
            off_axis_rms_log2_mean=("off_axis_rms_log2", "mean"),
            off_axis_rms_log2_sd=("off_axis_rms_log2", "std"),
            mid_log2_transfer_mean=("mid_log2_transfer", "mean"),
            high_log2_transfer_mean=("high_log2_transfer", "mean"),
        )
        .reset_index()
    )
    scanner_summary_path = output_dir / "scanner_summary.csv"
    scanner_summary.to_csv(scanner_summary_path, index=False)

    figures = []
    figures.extend(
        render_cohort_overview(tables["spectra"], tables["bands"], output_dir)
    )
    figures.extend(render_mixed_effects(components, matrices, output_dir))
    figures.extend(
        render_blur_sharpen_bridge(
            tables["spectra"],
            tables["interventions"],
            tables["projection"],
            output_dir,
        )
    )

    high = components[components["band"] == "high"].copy()
    summary = {
        "analysis": "balanced_scanner_fixed_and_slide_random_slope_decomposition",
        "absolute_mtf_claim": False,
        "slides": args.expected_slides,
        "patches_per_slide": args.expected_patches,
        "replicate_design": {
            "groups": args.expected_replicates,
            "patches_per_group": args.expected_patches // args.expected_replicates,
            "purpose": (
                "separate scanner-by-slide variance and covariance from "
                "patch-sampling error"
            ),
        },
        "model": (
            "per scanner and band: log2_transfer = fixed_scanner_mean + "
            "slide_specific_scanner_slope + replicate_sampling_error"
        ),
        "tissue_model": {
            "included": False,
            "reason": "registered_pair_summary.csv has no tissue/organ column",
            "next_model_after_metadata_merge": (
                "scanner * tissue fixed effects + scanner-by-slide random slopes; "
                "slide nested within tissue"
            ),
        },
        "tables": {
            **combined_paths,
            "variance_components": str(component_path),
            "random_slopes": str(slope_path),
            "random_slope_correlations": str(correlation_path),
            "scanner_summary": str(scanner_summary_path),
        },
        "figures": figures,
        "high_band_results": high.to_dict("records"),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    display_columns = [
        "scanner",
        "fixed_mean_log2_transfer",
        "fixed_fold_transfer",
        "fixed_ci95_low",
        "fixed_ci95_high",
        "scanner_by_slide_sd",
        "sampling_sd_100patch",
        "slide_slope_reliability",
        "p_vs_at2",
    ]
    print("[exp05 cohort] high-band mixed-effect summary", flush=True)
    print(high[display_columns].to_string(index=False), flush=True)
    print(f"[exp05 cohort] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
