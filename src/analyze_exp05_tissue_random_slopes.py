"""Nested tissue/slide random-slope analysis for Exp-05.

For every scanner (relative to AT2) and frequency band, fit

    y_tissue,slide,replicate = fixed scanner mean
                               + tissue-specific scanner slope
                               + slide-within-tissue scanner slope
                               + patch-sampling error.

This is the diagonal-covariance form of

    0 + scanner + (0 + scanner | tissue_type)
                + (0 + scanner | tissue_type:slide_id)

fit by profiled REML.  Five disjoint 20-patch spectrum estimates per slide
identify sampling error separately from the two biological/process levels.
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
from scipy import linalg, optimize, stats
from scipy.cluster.hierarchy import leaves_list, linkage

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_exp05_spectral_pilot import (
    BANDS,
    SCANNERS,
    save_figure,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort-output",
        default="outputs/exp05_spectral_cohort_109",
    )
    parser.add_argument(
        "--annotation",
        default=(
            "/mnt/isilon/oldridge_lab/batch_effects/metadata/"
            "pan_normal_annotation.csv"
        ),
    )
    parser.add_argument("--expected-slides", type=int, default=109)
    return parser.parse_args(argv)


def load_annotated_replicates(cohort_output: Path, annotation_path: Path):
    replicates = pd.read_csv(
        cohort_output / "slide_band_replicates.csv",
        dtype={"slide_id": str},
    )
    annotation = pd.read_csv(
        annotation_path,
        dtype=str,
        encoding="utf-8-sig",
    ).rename(
        columns={"AnonSlideID": "slide_id", "Tissue Type": "tissue_type"}
    )
    required = {"slide_id", "tissue_type"}
    if not required.issubset(annotation.columns):
        raise ValueError(
            f"annotation must contain {sorted(required)} after renaming; "
            f"got {annotation.columns.tolist()}"
        )
    if annotation["slide_id"].duplicated().any():
        raise ValueError("annotation contains duplicated slide IDs")
    annotation = annotation[["slide_id", "tissue_type"]].copy()
    merged = replicates.merge(
        annotation,
        on="slide_id",
        how="left",
        validate="many_to_one",
    )
    missing = sorted(
        merged.loc[merged["tissue_type"].isna(), "slide_id"].unique()
    )
    if missing:
        raise ValueError(f"slides missing tissue annotation: {missing}")
    return merged, annotation


def make_blocks(frame: pd.DataFrame):
    value = "log2_relative_transfer"
    blocks = []
    expected_replicates = frame.groupby("slide_id")["replicate"].nunique()
    if expected_replicates.nunique() != 1:
        raise ValueError("replicate counts are not balanced across slides")
    n_replicates = int(expected_replicates.iloc[0])
    for tissue_type, tissue_frame in frame.groupby("tissue_type", sort=True):
        tissue_frame = tissue_frame.sort_values(["slide_id", "replicate"])
        slide_ids = tissue_frame["slide_id"].drop_duplicates().tolist()
        y = tissue_frame[value].to_numpy(dtype=float)
        slide_labels = tissue_frame["slide_id"].astype(str).to_numpy()
        slide_design = np.column_stack(
            [(slide_labels == slide_id).astype(float) for slide_id in slide_ids]
        )
        if not np.all(slide_design.sum(axis=0) == n_replicates):
            raise ValueError(f"unbalanced tissue block: {tissue_type}")
        blocks.append({
            "tissue_type": str(tissue_type),
            "slide_ids": slide_ids,
            "y": y,
            "ones": np.ones(len(y), dtype=float),
            "slide_design": slide_design,
            "slide_kernel": slide_design @ slide_design.T,
            "tissue_kernel": np.ones((len(y), len(y)), dtype=float),
            "identity": np.eye(len(y), dtype=float),
        })
    return blocks, n_replicates


def initial_variances(frame: pd.DataFrame, n_replicates: int):
    value = "log2_relative_transfer"
    slide_means = frame.groupby(["tissue_type", "slide_id"])[value].mean()
    repeated_slide_mean = frame.set_index(["tissue_type", "slide_id"]).index.map(
        slide_means
    )
    residual_variance = float(
        np.square(frame[value].to_numpy() - repeated_slide_mean.to_numpy()).sum()
        / (len(frame) - len(slide_means))
    )
    tissue_means = slide_means.groupby(level="tissue_type").mean()
    within_tissue = slide_means - slide_means.index.get_level_values(
        "tissue_type"
    ).map(tissue_means)
    slide_variance = max(
        float(np.var(within_tissue, ddof=1)) - residual_variance / n_replicates,
        1e-6,
    )
    mean_slides_per_tissue = float(
        slide_means.groupby(level="tissue_type").size().mean()
    )
    tissue_variance = max(
        float(np.var(tissue_means, ddof=1))
        - slide_variance / mean_slides_per_tissue
        - residual_variance / (n_replicates * mean_slides_per_tissue),
        1e-6,
    )
    return np.asarray(
        [tissue_variance, slide_variance, max(residual_variance, 1e-6)]
    )


def evaluate_reml(blocks, variances, include_tissue=True, return_details=False):
    tissue_variance, slide_variance, residual_variance = variances
    total_n = 0
    log_determinant = 0.0
    one_v_one = 0.0
    one_v_y = 0.0
    y_v_y = 0.0
    details = []
    for block in blocks:
        covariance = (
            slide_variance * block["slide_kernel"]
            + residual_variance * block["identity"]
        )
        if include_tissue:
            covariance = covariance + tissue_variance * block["tissue_kernel"]
        factor = linalg.cho_factor(covariance, lower=True, check_finite=False)
        inverse_one = linalg.cho_solve(
            factor, block["ones"], check_finite=False
        )
        inverse_y = linalg.cho_solve(factor, block["y"], check_finite=False)
        total_n += len(block["y"])
        log_determinant += float(
            2.0 * np.log(np.diag(factor[0])).sum()
        )
        one_v_one += float(block["ones"] @ inverse_one)
        one_v_y += float(block["ones"] @ inverse_y)
        y_v_y += float(block["y"] @ inverse_y)
        if return_details:
            details.append((block, factor))
    fixed_mean = one_v_y / one_v_one
    quadratic = y_v_y - one_v_y**2 / one_v_one
    negative_log_likelihood = 0.5 * (
        log_determinant
        + np.log(one_v_one)
        + quadratic
        + (total_n - 1) * np.log(2.0 * np.pi)
    )
    result = {
        "negative_log_likelihood": float(negative_log_likelihood),
        "fixed_mean": float(fixed_mean),
        "fixed_se": float(np.sqrt(1.0 / one_v_one)),
        "total_n": total_n,
    }
    if return_details:
        result["details"] = details
    return result


def fit_nested_reml(frame: pd.DataFrame):
    blocks, n_replicates = make_blocks(frame)
    start = initial_variances(frame, n_replicates)
    bounds = [(-18.0, 5.0)] * 3

    def objective(log_variances):
        variances = np.exp(log_variances)
        return evaluate_reml(blocks, variances)["negative_log_likelihood"]

    starts = [
        np.log(start),
        np.log(np.maximum(start * np.asarray([0.25, 2.0, 1.0]), 1e-8)),
        np.log(np.maximum(start * np.asarray([2.0, 0.25, 1.0]), 1e-8)),
    ]
    fits = [
        optimize.minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-8},
        )
        for initial in starts
    ]
    converged_fits = [result for result in fits if result.success]
    fit = min(converged_fits or fits, key=lambda result: result.fun)
    variances = np.exp(fit.x)
    full = evaluate_reml(blocks, variances, return_details=True)

    def reduced_objective(log_variances):
        reduced_variances = np.asarray(
            [0.0, np.exp(log_variances[0]), np.exp(log_variances[1])]
        )
        return evaluate_reml(
            blocks, reduced_variances, include_tissue=False
        )["negative_log_likelihood"]

    reduced_start = np.maximum(variances[1:], 1e-8)
    reduced_starts = [
        np.log(reduced_start),
        np.log(np.maximum(reduced_start * np.asarray([0.25, 2.0]), 1e-8)),
        np.log(np.maximum(reduced_start * np.asarray([2.0, 0.25]), 1e-8)),
    ]
    reduced_fits = [
        optimize.minimize(
            reduced_objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds[1:],
            options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-8},
        )
        for initial in reduced_starts
    ]
    converged_reduced_fits = [
        result for result in reduced_fits if result.success
    ]
    reduced_fit = min(
        converged_reduced_fits or reduced_fits,
        key=lambda result: result.fun,
    )
    likelihood_ratio = max(0.0, 2.0 * (reduced_fit.fun - fit.fun))
    tissue_variance_p = (
        0.5 * float(stats.chi2.sf(likelihood_ratio, df=1))
        if likelihood_ratio > 0
        else 1.0
    )

    fixed_mean = full["fixed_mean"]
    tissue_rows = []
    slide_rows = []
    for block, factor in full["details"]:
        residual = block["y"] - fixed_mean
        alpha = linalg.cho_solve(factor, residual, check_finite=False)
        inverse_one = linalg.cho_solve(
            factor, block["ones"], check_finite=False
        )
        tissue_blup = variances[0] * float(block["ones"] @ alpha)
        tissue_conditional_variance = max(
            variances[0]
            - variances[0] ** 2 * float(block["ones"] @ inverse_one),
            0.0,
        )
        raw_tissue_mean = float(block["y"].mean() - fixed_mean)
        tissue_rows.append({
            "tissue_type": block["tissue_type"],
            "slides_in_tissue": len(block["slide_ids"]),
            "raw_tissue_slope": raw_tissue_mean,
            "blup_tissue_slope": tissue_blup,
            "blup_se_conditional": float(np.sqrt(tissue_conditional_variance)),
        })
        for column, slide_id in enumerate(block["slide_ids"]):
            indicator = block["slide_design"][:, column]
            inverse_indicator = linalg.cho_solve(
                factor, indicator, check_finite=False
            )
            slide_blup = variances[1] * float(indicator @ alpha)
            slide_conditional_variance = max(
                variances[1]
                - variances[1] ** 2 * float(indicator @ inverse_indicator),
                0.0,
            )
            slide_rows.append({
                "tissue_type": block["tissue_type"],
                "slide_id": str(slide_id),
                "blup_slide_within_tissue_slope": slide_blup,
                "blup_se_conditional": float(np.sqrt(slide_conditional_variance)),
            })
    return {
        "variances": variances,
        "full": full,
        "fit": fit,
        "reduced_fit": reduced_fit,
        "likelihood_ratio": likelihood_ratio,
        "tissue_variance_p": tissue_variance_p,
        "n_replicates": n_replicates,
        "n_tissues": len(blocks),
        "tissue_rows": tissue_rows,
        "slide_rows": slide_rows,
    }


def benjamini_hochberg(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def fit_all_models(replicates: pd.DataFrame):
    component_rows = []
    tissue_rows = []
    slide_rows = []
    for band in BANDS:
        for scanner in SCANNERS[1:]:
            frame = replicates[
                (replicates["band"] == band)
                & (replicates["scanner"] == scanner)
            ].copy()
            result = fit_nested_reml(frame)
            tissue_variance, slide_variance, residual_variance = result[
                "variances"
            ]
            sampling_variance = residual_variance / result["n_replicates"]
            final_variance = tissue_variance + slide_variance + sampling_variance
            df = result["n_tissues"] - 1
            critical = float(stats.t.ppf(0.975, df=df))
            fixed_mean = result["full"]["fixed_mean"]
            fixed_se = result["full"]["fixed_se"]
            fixed_t = fixed_mean / fixed_se
            component_rows.append({
                "band": band,
                "scanner": scanner,
                "n_tissues": result["n_tissues"],
                "n_slides": frame["slide_id"].nunique(),
                "replicate_groups": result["n_replicates"],
                "fixed_mean_log2_transfer": fixed_mean,
                "fixed_fold_transfer": float(2.0**fixed_mean),
                "fixed_se": fixed_se,
                "fixed_ci95_low": fixed_mean - critical * fixed_se,
                "fixed_ci95_high": fixed_mean + critical * fixed_se,
                "fixed_t": fixed_t,
                "fixed_p": float(2.0 * stats.t.sf(abs(fixed_t), df=df)),
                "scanner_by_tissue_variance": tissue_variance,
                "scanner_by_tissue_sd": float(np.sqrt(tissue_variance)),
                "scanner_by_slide_within_tissue_variance": slide_variance,
                "scanner_by_slide_within_tissue_sd": float(np.sqrt(slide_variance)),
                "replicate_sampling_variance": residual_variance,
                "sampling_variance_100patch": sampling_variance,
                "sampling_sd_100patch": float(np.sqrt(sampling_variance)),
                "tissue_fraction_total_100patch_variance": (
                    tissue_variance / final_variance
                ),
                "slide_fraction_total_100patch_variance": (
                    slide_variance / final_variance
                ),
                "sampling_fraction_total_100patch_variance": (
                    sampling_variance / final_variance
                ),
                "tissue_fraction_between_slide_variance": (
                    tissue_variance / (tissue_variance + slide_variance)
                ),
                "tissue_variance_lrt": result["likelihood_ratio"],
                "tissue_variance_p_mixture": result["tissue_variance_p"],
                "optimizer_converged": bool(result["fit"].success),
                "optimizer_message": str(result["fit"].message),
                "reml_negative_log_likelihood": result["fit"].fun,
            })
            for row in result["tissue_rows"]:
                tissue_rows.append({"band": band, "scanner": scanner, **row})
            for row in result["slide_rows"]:
                slide_rows.append({"band": band, "scanner": scanner, **row})
            print(
                f"[tissue REML] {band}/{scanner}: "
                f"tissue_sd={np.sqrt(tissue_variance):.3f}, "
                f"slide_sd={np.sqrt(slide_variance):.3f}, "
                f"p={result['tissue_variance_p']:.3g}",
                flush=True,
            )
    components = pd.DataFrame(component_rows)
    components["tissue_variance_q_bh"] = benjamini_hochberg(
        components["tissue_variance_p_mixture"]
    )
    return components, pd.DataFrame(tissue_rows), pd.DataFrame(slide_rows)


def render_tissue_random_slopes(components, tissue_slopes, output_dir):
    figure, axes = plt.subplots(1, 3, figsize=(21, 5.8))
    high = components[components["band"] == "high"].set_index("scanner").reindex(
        SCANNERS[1:]
    )
    x = np.arange(len(SCANNERS) - 1)

    axis = axes[0]
    tissue_var = high["scanner_by_tissue_variance"].to_numpy()
    slide_var = high["scanner_by_slide_within_tissue_variance"].to_numpy()
    sampling_var = high["sampling_variance_100patch"].to_numpy()
    axis.bar(x, tissue_var, color="#7C3AED", alpha=0.82, label="scanner × tissue")
    axis.bar(
        x,
        slide_var,
        bottom=tissue_var,
        color="#38BDF8",
        alpha=0.82,
        label="scanner × slide | tissue",
    )
    axis.bar(
        x,
        sampling_var,
        bottom=tissue_var + slide_var,
        color="#D1D5DB",
        label="100-patch sampling",
    )
    axis.set_xticks(x, [s.upper() for s in SCANNERS[1:]], rotation=30)
    axis.set(
        ylabel="Variance (log₂ transfer²)",
        title="A  High-band nested variance decomposition",
    )
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    fraction = (
        components.pivot(
            index="band",
            columns="scanner",
            values="tissue_fraction_between_slide_variance",
        )
        .reindex(index=list(BANDS), columns=SCANNERS[1:])
    )
    image = axis.imshow(fraction, cmap="Purples", vmin=0, vmax=1, aspect="auto")
    q_values = (
        components.pivot(
            index="band", columns="scanner", values="tissue_variance_q_bh"
        ).reindex(index=list(BANDS), columns=SCANNERS[1:])
    )
    for row in range(fraction.shape[0]):
        for column in range(fraction.shape[1]):
            value = fraction.iloc[row, column]
            marker = "*" if q_values.iloc[row, column] < 0.05 else ""
            axis.text(
                column,
                row,
                f"{100 * value:.0f}%{marker}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=9,
            )
    axis.set_xticks(range(len(SCANNERS) - 1), [s.upper() for s in SCANNERS[1:]], rotation=30)
    axis.set_yticks(range(len(BANDS)), [b.replace("_", "–") for b in BANDS])
    axis.set_title("B  Tissue share of between-slide variance")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Tissue variance fraction")
    axis.text(
        0.0,
        -0.23,
        "* tissue variance LRT, BH q<0.05",
        transform=axis.transAxes,
        fontsize=8,
        color="#4B5563",
    )

    axis = axes[2]
    high_slopes = tissue_slopes[tissue_slopes["band"] == "high"]
    matrix = high_slopes.pivot(
        index="scanner", columns="tissue_type", values="blup_tissue_slope"
    ).reindex(index=SCANNERS[1:])
    order = leaves_list(linkage(matrix.to_numpy().T, method="average"))
    matrix = matrix.iloc[:, order]
    limit = max(0.15, float(np.quantile(np.abs(matrix.to_numpy()), 0.98)))
    image = axis.imshow(
        matrix,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_yticks(range(len(SCANNERS) - 1), [s.upper() for s in SCANNERS[1:]])
    axis.set_xticks(
        range(matrix.shape[1]),
        matrix.columns,
        rotation=75,
        ha="right",
        fontsize=6.5,
    )
    axis.set_title("C  High-band tissue-specific scanner slopes (BLUP)")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Deviation from scanner mean (log₂)")

    figure.suptitle(
        "Scanner effects decomposed across 37 tissue types and 109 slides",
        fontsize=15,
        y=1.03,
    )
    figure.tight_layout()
    return save_figure(figure, output_dir, "figure4_tissue_random_slopes")


def main(argv=None):
    args = parse_args(argv)
    cohort_output = Path(args.cohort_output)
    annotation_path = Path(args.annotation)
    replicates, annotation = load_annotated_replicates(
        cohort_output, annotation_path
    )
    if replicates["slide_id"].nunique() != args.expected_slides:
        raise AssertionError(
            f"expected {args.expected_slides} annotated slides, got "
            f"{replicates['slide_id'].nunique()}"
        )
    if replicates["tissue_type"].nunique() < 20:
        raise AssertionError("too few tissue levels for random-slope analysis")

    annotated_path = cohort_output / "slide_band_replicates_annotated.csv"
    annotation_used_path = cohort_output / "slide_tissue_annotation_used.csv"
    replicates.to_csv(annotated_path, index=False)
    used_annotation = (
        replicates[["slide_id", "tissue_type"]]
        .drop_duplicates()
        .sort_values("slide_id")
    )
    used_annotation.to_csv(annotation_used_path, index=False)

    components, tissue_slopes, slide_slopes = fit_all_models(replicates)
    if not components["optimizer_converged"].all():
        failed = components.loc[
            ~components["optimizer_converged"], ["band", "scanner", "optimizer_message"]
        ]
        raise RuntimeError(f"nested REML optimizer failures:\n{failed}")

    component_path = cohort_output / "tissue_random_slope_variance_components.csv"
    tissue_slope_path = cohort_output / "tissue_random_slopes.csv"
    slide_slope_path = cohort_output / "slide_within_tissue_random_slopes.csv"
    components.to_csv(component_path, index=False)
    tissue_slopes.to_csv(tissue_slope_path, index=False)
    slide_slopes.to_csv(slide_slope_path, index=False)
    figures = render_tissue_random_slopes(
        components, tissue_slopes, cohort_output
    )

    high = components[components["band"] == "high"].copy()
    summary = {
        "analysis": "nested_tissue_and_slide_scanner_random_slopes",
        "annotation": str(annotation_path),
        "model": (
            "per band, diagonal scanner-slope covariance: 0 + scanner + "
            "(0 + scanner | tissue_type) + "
            "(0 + scanner | tissue_type:slide_id)"
        ),
        "estimation": "profiled REML with five disjoint 20-patch replicates",
        "slides": int(replicates["slide_id"].nunique()),
        "tissues": int(replicates["tissue_type"].nunique()),
        "tissue_slide_counts": (
            used_annotation.groupby("tissue_type")["slide_id"].nunique().to_dict()
        ),
        "tables": {
            "annotated_replicates": str(annotated_path),
            "annotation_used": str(annotation_used_path),
            "variance_components": str(component_path),
            "tissue_random_slopes": str(tissue_slope_path),
            "slide_within_tissue_random_slopes": str(slide_slope_path),
        },
        "figures": figures,
        "high_band_results": high.to_dict("records"),
    }
    with (cohort_output / "tissue_random_slope_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    columns = [
        "scanner",
        "fixed_mean_log2_transfer",
        "scanner_by_tissue_sd",
        "scanner_by_slide_within_tissue_sd",
        "sampling_sd_100patch",
        "tissue_fraction_between_slide_variance",
        "tissue_variance_p_mixture",
        "tissue_variance_q_bh",
    ]
    print("[tissue REML] high-band summary", flush=True)
    print(high[columns].to_string(index=False), flush=True)
    print(f"[tissue REML] wrote {cohort_output}", flush=True)


if __name__ == "__main__":
    main()
