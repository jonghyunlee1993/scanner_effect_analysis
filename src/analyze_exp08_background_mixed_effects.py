"""Refit Exp-05 nested mixed models with Exp-07 background covariates.

For each non-reference scanner and tissue-frequency band, compare:

M0: intercept + tissue random slope + slide-within-tissue random slope
M1: M0 + scanner-centred background physics covariates
M2: M1 + strict-glass acceptance rate (QC sensitivity analysis)

The tissue outcome is already a scanner/AT2 log2 transfer. Background
features are therefore first contrasted against AT2 within slide and then
centred and standardized within scanner. REML estimates variance components;
ML compares fixed-effect specifications. Five-fold grouped prediction holds
out complete slides and retains only tissue BLUPs learned from training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_exp05_tissue_random_slopes import (
    benjamini_hochberg,
    initial_variances,
    load_annotated_replicates,
)
from run_exp05_spectral_pilot import BANDS, SCANNERS, save_figure


MAIN_FEATURES = (
    "bg_matched_nps_z",
    "bg_nps_tilt_z",
    "bg_od_luminance_z",
    "bg_od_rg_z",
    "bg_od_brg_z",
)
SENSITIVITY_FEATURES = MAIN_FEATURES + ("bg_acceptance_logit_z",)

FEATURE_LABELS = {
    "bg_matched_nps_z": "Matched-band log₂ NPS",
    "bg_nps_tilt_z": "NPS high–low tilt",
    "bg_od_luminance_z": "Background OD luminance",
    "bg_od_rg_z": "Background OD R–G",
    "bg_od_brg_z": "Background OD B–RG",
    "bg_acceptance_logit_z": "Glass-QC acceptance",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort-output", default="outputs/exp05_spectral_cohort_109"
    )
    parser.add_argument(
        "--background-aggregate",
        default="outputs/exp07_raw_background_109x6/aggregate",
    )
    parser.add_argument(
        "--annotation",
        default=(
            "/mnt/isilon/oldridge_lab/batch_effects/metadata/"
            "pan_normal_annotation.csv"
        ),
    )
    parser.add_argument(
        "--output", default="outputs/exp08_background_mixed_effects"
    )
    parser.add_argument("--expected-slides", type=int, default=109)
    parser.add_argument("--scanners", nargs="+", default=list(SCANNERS[1:]))
    parser.add_argument("--bands", nargs="+", default=list(BANDS))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def geometric_band_nps(spectra):
    plane = spectra[spectra["detrend"] == "plane"].copy()
    rows = []
    for band, (lower, upper) in BANDS.items():
        selected = plane[
            (plane["frequency_cyc_per_um"] >= lower)
            & (plane["frequency_cyc_per_um"] < upper)
        ]
        values = (
            selected.groupby(["slide_id", "scanner"])["nps_od2_um2"]
            .apply(
                lambda item: float(
                    np.exp(np.mean(np.log(np.maximum(item.to_numpy(), 1e-30))))
                )
            )
            .rename("background_nps")
            .reset_index()
        )
        values["band"] = band
        values["log2_background_nps"] = np.log2(values["background_nps"])
        rows.append(values)
    return pd.concat(rows, ignore_index=True)


def contrast_against_at2(frame, value_columns):
    reference = frame[frame["scanner"] == "at2"][
        ["slide_id", *value_columns]
    ].rename(columns={column: f"{column}_at2" for column in value_columns})
    merged = frame.merge(reference, on="slide_id", validate="many_to_one")
    for column in value_columns:
        merged[f"{column}_contrast"] = (
            merged[column] - merged[f"{column}_at2"]
        )
    return merged


def standardize_within(frame, group_columns, columns):
    result = frame.copy()
    for column in columns:
        grouped = result.groupby(group_columns)[column]
        mean = grouped.transform("mean")
        standard_deviation = grouped.transform("std")
        if (standard_deviation <= 1e-12).any():
            bad = result.loc[standard_deviation <= 1e-12, group_columns]
            raise ValueError(f"zero feature variance for {column}: {bad.head()}")
        result[f"{column}_centered"] = result[column] - mean
        result[f"{column}_z"] = (result[column] - mean) / standard_deviation
    return result


def assign_slide_folds(annotation, folds):
    rows = []
    for tissue_type, frame in annotation.groupby("tissue_type", sort=True):
        slide_ids = sorted(frame["slide_id"].astype(str).unique())
        digest = hashlib.sha256(str(tissue_type).encode("utf-8")).digest()
        offset = int.from_bytes(digest[:2], "little") % folds
        for index, slide_id in enumerate(slide_ids):
            rows.append(
                {
                    "slide_id": slide_id,
                    "cv_fold": (index + offset) % folds,
                }
            )
    result = pd.DataFrame(rows)
    if result["slide_id"].duplicated().any():
        raise ValueError("duplicated fold assignment")
    return result


def build_background_features(background_dir, annotation, folds):
    moments = pd.read_csv(
        background_dir / "background_moments.csv", dtype={"slide_id": str}
    )
    metadata = pd.read_csv(
        background_dir / "cell_metadata.csv", dtype={"slide_id": str}
    )
    spectra = pd.read_csv(
        background_dir / "raw_glass_nps_spectra.csv.gz",
        dtype={"slide_id": str},
    )
    expected = annotation["slide_id"].astype(str).nunique() * len(SCANNERS)
    if len(moments) != expected or len(metadata) != expected:
        raise ValueError((len(moments), len(metadata), expected))

    moments = moments.merge(
        metadata[["slide_id", "scanner", "acceptance_rate"]],
        on=["slide_id", "scanner"],
        validate="one_to_one",
    )
    moments["od_luminance"] = moments[
        ["od10_mean_r", "od10_mean_g", "od10_mean_b"]
    ].mean(axis=1)
    moments["od_rg"] = moments["od10_mean_r"] - moments["od10_mean_g"]
    moments["od_brg"] = moments["od10_mean_b"] - 0.5 * (
        moments["od10_mean_r"] + moments["od10_mean_g"]
    )
    clipped_rate = moments["acceptance_rate"].clip(0.005, 0.995)
    moments["acceptance_logit"] = np.log(clipped_rate / (1.0 - clipped_rate))
    moment_values = [
        "od_luminance",
        "od_rg",
        "od_brg",
        "acceptance_logit",
    ]
    moment_contrast = contrast_against_at2(moments, moment_values)
    moment_contrast = moment_contrast[moment_contrast["scanner"] != "at2"]
    contrast_columns = [f"{column}_contrast" for column in moment_values]
    moment_contrast = standardize_within(
        moment_contrast, ["scanner"], contrast_columns
    )
    moment_contrast = moment_contrast.rename(
        columns={
            "od_luminance_contrast_z": "bg_od_luminance_z",
            "od_rg_contrast_z": "bg_od_rg_z",
            "od_brg_contrast_z": "bg_od_brg_z",
            "acceptance_logit_contrast_z": "bg_acceptance_logit_z",
        }
    )

    nps = geometric_band_nps(spectra)
    reference = nps[nps["scanner"] == "at2"][
        ["slide_id", "band", "log2_background_nps"]
    ].rename(columns={"log2_background_nps": "log2_background_nps_at2"})
    nps = nps.merge(reference, on=["slide_id", "band"], validate="many_to_one")
    nps["log2_nps_ratio"] = (
        nps["log2_background_nps"] - nps["log2_background_nps_at2"]
    )
    nps = nps[nps["scanner"] != "at2"].copy()
    nps = standardize_within(nps, ["scanner", "band"], ["log2_nps_ratio"])
    nps = nps.rename(columns={"log2_nps_ratio_z": "bg_matched_nps_z"})

    wide = nps.pivot(
        index=["slide_id", "scanner"], columns="band", values="log2_nps_ratio"
    ).reset_index()
    wide["nps_tilt"] = wide["high"] - wide["low_mid"]
    wide = standardize_within(wide, ["scanner"], ["nps_tilt"])
    wide = wide.rename(columns={"nps_tilt_z": "bg_nps_tilt_z"})

    feature_columns = [
        "slide_id",
        "scanner",
        "bg_od_luminance_z",
        "bg_od_rg_z",
        "bg_od_brg_z",
        "bg_acceptance_logit_z",
        *[f"{column}_contrast" for column in moment_values],
    ]
    features = nps[
        [
            "slide_id",
            "scanner",
            "band",
            "log2_nps_ratio",
            "log2_nps_ratio_centered",
            "bg_matched_nps_z",
        ]
    ].merge(
        wide[["slide_id", "scanner", "nps_tilt", "nps_tilt_centered", "bg_nps_tilt_z"]],
        on=["slide_id", "scanner"],
        validate="many_to_one",
    ).merge(
        moment_contrast[feature_columns],
        on=["slide_id", "scanner"],
        validate="many_to_one",
    )
    features = features.merge(
        annotation[["slide_id", "tissue_type"]],
        on="slide_id",
        validate="many_to_one",
    ).merge(
        assign_slide_folds(annotation, folds),
        on="slide_id",
        validate="many_to_one",
    )
    if features[list(SENSITIVITY_FEATURES)].isna().any().any():
        raise ValueError("missing standardized background features")
    return features


def fixed_design(frame, feature_names):
    columns = [np.ones(len(frame), dtype=float)]
    columns.extend(frame[name].to_numpy(dtype=float) for name in feature_names)
    return np.column_stack(columns)


def make_blocks(frame, feature_names):
    blocks = []
    replicate_counts = frame.groupby("slide_id")["replicate"].nunique()
    if replicate_counts.nunique() != 1:
        raise ValueError("replicate counts are not balanced")
    n_replicates = int(replicate_counts.iloc[0])
    for tissue_type, tissue_frame in frame.groupby("tissue_type", sort=True):
        tissue_frame = tissue_frame.sort_values(["slide_id", "replicate"])
        slide_ids = tissue_frame["slide_id"].drop_duplicates().tolist()
        labels = tissue_frame["slide_id"].astype(str).to_numpy()
        slide_design = np.column_stack(
            [(labels == slide_id).astype(float) for slide_id in slide_ids]
        )
        blocks.append(
            {
                "tissue_type": str(tissue_type),
                "slide_ids": slide_ids,
                "y": tissue_frame["log2_relative_transfer"].to_numpy(dtype=float),
                "x": fixed_design(tissue_frame, feature_names),
                "ones": np.ones(len(tissue_frame), dtype=float),
                "slide_design": slide_design,
                "slide_kernel": slide_design @ slide_design.T,
                "tissue_kernel": np.ones((len(tissue_frame), len(tissue_frame))),
                "identity": np.eye(len(tissue_frame)),
            }
        )
    return blocks, n_replicates


def evaluate_lmm(blocks, variances, reml=True, return_details=False):
    tissue_variance, slide_variance, residual_variance = variances
    total_n = 0
    log_determinant = 0.0
    xt_v_x = None
    xt_v_y = None
    y_v_y = 0.0
    details = []
    for block in blocks:
        covariance = (
            tissue_variance * block["tissue_kernel"]
            + slide_variance * block["slide_kernel"]
            + residual_variance * block["identity"]
        )
        factor = linalg.cho_factor(covariance, lower=True, check_finite=False)
        inverse_x = linalg.cho_solve(factor, block["x"], check_finite=False)
        inverse_y = linalg.cho_solve(factor, block["y"], check_finite=False)
        current_xt_v_x = block["x"].T @ inverse_x
        current_xt_v_y = block["x"].T @ inverse_y
        xt_v_x = current_xt_v_x if xt_v_x is None else xt_v_x + current_xt_v_x
        xt_v_y = current_xt_v_y if xt_v_y is None else xt_v_y + current_xt_v_y
        y_v_y += float(block["y"] @ inverse_y)
        total_n += len(block["y"])
        log_determinant += float(2.0 * np.log(np.diag(factor[0])).sum())
        if return_details:
            details.append((block, factor))
    fixed_covariance = linalg.inv(xt_v_x, check_finite=False)
    beta = fixed_covariance @ xt_v_y
    quadratic = y_v_y - float(xt_v_y @ beta)
    parameters = len(beta)
    negative_log_likelihood = 0.5 * (
        log_determinant
        + quadratic
        + total_n * np.log(2.0 * np.pi)
    )
    if reml:
        sign, log_fixed_determinant = np.linalg.slogdet(xt_v_x)
        if sign <= 0:
            raise np.linalg.LinAlgError("fixed design is not positive definite")
        negative_log_likelihood += 0.5 * (
            log_fixed_determinant - parameters * np.log(2.0 * np.pi)
        )
    result = {
        "negative_log_likelihood": float(negative_log_likelihood),
        "beta": beta,
        "fixed_covariance": fixed_covariance,
        "total_n": total_n,
        "fixed_parameters": parameters,
    }
    if return_details:
        result["details"] = details
    return result


def fit_lmm(
    frame,
    feature_names,
    reml=True,
    start_variances=None,
    multiple_starts=True,
):
    blocks, n_replicates = make_blocks(frame, feature_names)
    if start_variances is None:
        start_variances = initial_variances(frame, n_replicates)
    start_variances = np.maximum(np.asarray(start_variances, dtype=float), 1e-8)
    bounds = [(-18.0, 5.0)] * 3

    def objective(log_variances):
        return evaluate_lmm(
            blocks, np.exp(log_variances), reml=reml
        )["negative_log_likelihood"]

    starts = [np.log(start_variances)]
    if multiple_starts:
        starts.extend(
            [
                np.log(np.maximum(start_variances * [0.25, 2.0, 1.0], 1e-8)),
                np.log(np.maximum(start_variances * [2.0, 0.25, 1.0], 1e-8)),
            ]
        )
    fits = [
        optimize.minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
        )
        for start in starts
    ]
    fit = min(fits, key=lambda value: value.fun)
    variances = np.exp(fit.x)
    evaluated = evaluate_lmm(
        blocks, variances, reml=reml, return_details=True
    )
    return {
        "blocks": blocks,
        "n_replicates": n_replicates,
        "fit": fit,
        "variances": variances,
        **evaluated,
    }


def tissue_blups(fit):
    result = {}
    beta = fit["beta"]
    tissue_variance = fit["variances"][0]
    for block, factor in fit["details"]:
        residual = block["y"] - block["x"] @ beta
        alpha = linalg.cho_solve(factor, residual, check_finite=False)
        result[block["tissue_type"]] = tissue_variance * float(
            block["ones"] @ alpha
        )
    return result


def fixed_effect_rows(fit, names, model, scanner, band):
    standard_errors = np.sqrt(np.diag(fit["fixed_covariance"]))
    rows = []
    for name, coefficient, standard_error in zip(
        ("intercept", *names), fit["beta"], standard_errors
    ):
        z_value = coefficient / standard_error
        rows.append(
            {
                "scanner": scanner,
                "band": band,
                "model": model,
                "feature": name,
                "coefficient": coefficient,
                "standard_error": standard_error,
                "z_value": z_value,
                "p_value_normal": 2.0 * stats.norm.sf(abs(z_value)),
            }
        )
    return rows


def model_record(name, fit, scanner, band):
    tissue_variance, slide_variance, residual_variance = fit["variances"]
    return {
        "scanner": scanner,
        "band": band,
        "model": name,
        "tissue_variance": tissue_variance,
        "slide_within_tissue_variance": slide_variance,
        "replicate_sampling_variance": residual_variance,
        "sampling_variance_100patch": residual_variance / fit["n_replicates"],
        "reml_negative_log_likelihood": fit["negative_log_likelihood"],
        "optimizer_converged": bool(fit["fit"].success),
        "optimizer_message": str(fit["fit"].message),
    }


def cross_validate(frame, feature_names, folds, full_m0, full_m1):
    rows = []
    for fold in sorted(frame["cv_fold"].unique()):
        train = frame[frame["cv_fold"] != fold].copy()
        test = frame[frame["cv_fold"] == fold].copy()
        if test.empty:
            continue
        fitted = {}
        for name, features, full_fit in (
            ("M0", (), full_m0),
            ("M1", feature_names, full_m1),
        ):
            fitted[name] = fit_lmm(
                train,
                features,
                reml=True,
                start_variances=full_fit["variances"],
                multiple_starts=False,
            )
        test_cells = (
            test.groupby(["slide_id", "tissue_type", "cv_fold"], as_index=False)
            .agg(
                observed=("log2_relative_transfer", "mean"),
                **{feature: (feature, "first") for feature in feature_names},
            )
        )
        for name, features in (("M0", ()), ("M1", feature_names)):
            fit = fitted[name]
            blups = tissue_blups(fit)
            design = fixed_design(test_cells, features)
            prediction = design @ fit["beta"]
            prediction += test_cells["tissue_type"].map(blups).fillna(0.0).to_numpy()
            for item, predicted in zip(test_cells.itertuples(index=False), prediction):
                rows.append(
                    {
                        "fold": int(fold),
                        "slide_id": str(item.slide_id),
                        "tissue_type": str(item.tissue_type),
                        "model": name,
                        "observed": float(item.observed),
                        "predicted": float(predicted),
                        "error": float(item.observed - predicted),
                        "optimizer_converged": bool(fit["fit"].success),
                    }
                )
    return rows


def fit_one_configuration(scanner, band, frame, cv_folds):
    frame = frame[
        (frame["scanner"] == scanner) & (frame["band"] == band)
    ].copy()
    if frame["slide_id"].nunique() != 109:
        raise ValueError((scanner, band, frame["slide_id"].nunique()))
    slide_design = frame.drop_duplicates("slide_id")[[*SENSITIVITY_FEATURES]]
    condition_number = float(
        np.linalg.cond(
            np.column_stack(
                [np.ones(len(slide_design)), slide_design[list(MAIN_FEATURES)]]
            )
        )
    )

    m0_reml = fit_lmm(frame, (), reml=True)
    m1_reml = fit_lmm(
        frame,
        MAIN_FEATURES,
        reml=True,
        start_variances=m0_reml["variances"],
    )
    m2_reml = fit_lmm(
        frame,
        SENSITIVITY_FEATURES,
        reml=True,
        start_variances=m1_reml["variances"],
    )
    m0_ml = fit_lmm(
        frame,
        (),
        reml=False,
        start_variances=m0_reml["variances"],
        multiple_starts=True,
    )
    m1_ml = fit_lmm(
        frame,
        MAIN_FEATURES,
        reml=False,
        start_variances=m1_reml["variances"],
        multiple_starts=True,
    )
    m2_ml = fit_lmm(
        frame,
        SENSITIVITY_FEATURES,
        reml=False,
        start_variances=m2_reml["variances"],
        multiple_starts=True,
    )

    records = [
        model_record("M0", m0_reml, scanner, band),
        model_record("M1", m1_reml, scanner, band),
        model_record("M2", m2_reml, scanner, band),
    ]
    fixed = []
    fixed.extend(fixed_effect_rows(m0_reml, (), "M0", scanner, band))
    fixed.extend(fixed_effect_rows(m1_reml, MAIN_FEATURES, "M1", scanner, band))
    fixed.extend(
        fixed_effect_rows(m2_reml, SENSITIVITY_FEATURES, "M2", scanner, band)
    )

    def information_criterion(fit):
        parameters = fit["fixed_parameters"] + 3
        return {
            "ml_nll": fit["negative_log_likelihood"],
            "aic": 2 * fit["negative_log_likelihood"] + 2 * parameters,
            "bic": 2 * fit["negative_log_likelihood"]
            + np.log(fit["total_n"]) * parameters,
        }

    info0 = information_criterion(m0_ml)
    info1 = information_criterion(m1_ml)
    info2 = information_criterion(m2_ml)
    likelihood_ratio = max(0.0, 2.0 * (info0["ml_nll"] - info1["ml_nll"]))
    sensitivity_ratio = max(
        0.0, 2.0 * (info1["ml_nll"] - info2["ml_nll"])
    )
    comparison = {
        "scanner": scanner,
        "band": band,
        "n_slides": frame["slide_id"].nunique(),
        "n_tissues": frame["tissue_type"].nunique(),
        "fixed_design_condition_number": condition_number,
        "m0_ml_nll": info0["ml_nll"],
        "m1_ml_nll": info1["ml_nll"],
        "m2_ml_nll": info2["ml_nll"],
        "m1_vs_m0_lrt": likelihood_ratio,
        "m1_vs_m0_df": len(MAIN_FEATURES),
        "m1_vs_m0_p": stats.chi2.sf(likelihood_ratio, len(MAIN_FEATURES)),
        "m2_vs_m1_acceptance_lrt": sensitivity_ratio,
        "m2_vs_m1_acceptance_p": stats.chi2.sf(sensitivity_ratio, 1),
        "delta_aic_m1_minus_m0": info1["aic"] - info0["aic"],
        "delta_bic_m1_minus_m0": info1["bic"] - info0["bic"],
        "delta_aic_m2_minus_m1": info2["aic"] - info1["aic"],
        "m0_tissue_variance": m0_reml["variances"][0],
        "m1_tissue_variance": m1_reml["variances"][0],
        "m2_tissue_variance": m2_reml["variances"][0],
        "m0_slide_variance": m0_reml["variances"][1],
        "m1_slide_variance": m1_reml["variances"][1],
        "m2_slide_variance": m2_reml["variances"][1],
        "m0_sampling_variance": m0_reml["variances"][2],
        "m1_sampling_variance": m1_reml["variances"][2],
        "m2_sampling_variance": m2_reml["variances"][2],
        "tissue_variance_reduction_m1": 1.0
        - m1_reml["variances"][0] / m0_reml["variances"][0],
        "slide_variance_reduction_m1": 1.0
        - m1_reml["variances"][1] / m0_reml["variances"][1],
        "between_slide_variance_reduction_m1": 1.0
        - (m1_reml["variances"][0] + m1_reml["variances"][1])
        / (m0_reml["variances"][0] + m0_reml["variances"][1]),
        "all_reml_converged": all(
            value["fit"].success for value in (m0_reml, m1_reml, m2_reml)
        ),
        "m0_ml_converged": bool(m0_ml["fit"].success),
        "m1_ml_converged": bool(m1_ml["fit"].success),
        "m2_ml_converged": bool(m2_ml["fit"].success),
        "m0_ml_message": str(m0_ml["fit"].message),
        "m1_ml_message": str(m1_ml["fit"].message),
        "m2_ml_message": str(m2_ml["fit"].message),
        "all_ml_converged": all(
            value["fit"].success for value in (m0_ml, m1_ml, m2_ml)
        ),
    }
    cv_rows = cross_validate(
        frame, MAIN_FEATURES, cv_folds, m0_reml, m1_reml
    )
    for row in cv_rows:
        row["scanner"] = scanner
        row["band"] = band
    return {
        "comparison": comparison,
        "models": records,
        "fixed": fixed,
        "cv": cv_rows,
    }


def heatmap(axis, frame, value, scanners, bands, title, formatter, cmap, center=None):
    matrix = (
        frame.pivot(index="band", columns="scanner", values=value)
        .reindex(index=bands, columns=scanners)
    )
    array = matrix.to_numpy(dtype=float)
    if center is None:
        image = axis.imshow(array, cmap=cmap, aspect="auto")
    else:
        limit = max(float(np.nanquantile(np.abs(array - center), 0.98)), 1e-6)
        image = axis.imshow(
            array,
            cmap=cmap,
            aspect="auto",
            vmin=center - limit,
            vmax=center + limit,
        )
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            axis.text(
                column,
                row,
                formatter(array[row, column]),
                ha="center",
                va="center",
                fontsize=8,
            )
    axis.set_xticks(range(len(scanners)), [value.upper() for value in scanners], rotation=30)
    axis.set_yticks(range(len(bands)), [value.replace("_", "–") for value in bands])
    axis.set_title(title)
    return image


def render_comparison(comparison, cv_summary, scanners, bands, output):
    merged = comparison.merge(cv_summary, on=["scanner", "band"], validate="one_to_one")
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    image = heatmap(
        axes[0, 0],
        merged,
        "slide_variance_reduction_m1",
        scanners,
        bands,
        "A  Slide-within-tissue variance explained by background",
        lambda value: f"{100 * value:+.0f}%",
        "RdBu_r",
        center=0,
    )
    figure.colorbar(image, ax=axes[0, 0], label="Variance reduction")
    image = heatmap(
        axes[0, 1],
        merged,
        "tissue_variance_reduction_m1",
        scanners,
        bands,
        "B  Tissue-level variance change",
        lambda value: f"{100 * value:+.0f}%",
        "RdBu_r",
        center=0,
    )
    figure.colorbar(image, ax=axes[0, 1], label="Variance reduction")
    image = heatmap(
        axes[1, 0],
        merged,
        "delta_aic_m1_minus_m0",
        scanners,
        bands,
        "C  ML model comparison (negative favours background)",
        lambda value: f"{value:+.1f}",
        "RdBu_r",
        center=0,
    )
    figure.colorbar(image, ax=axes[1, 0], label="ΔAIC M1−M0")
    image = heatmap(
        axes[1, 1],
        merged,
        "cv_rmse_improvement_fraction",
        scanners,
        bands,
        "D  Held-out-slide RMSE improvement",
        lambda value: f"{100 * value:+.1f}%",
        "RdBu_r",
        center=0,
    )
    figure.colorbar(image, ax=axes[1, 1], label="CV RMSE improvement")
    figure.suptitle(
        "Does scanner-centred background physics explain tissue transfer?",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(figure, output, "figure1_background_mixed_model_comparison")


def render_fixed_effects(fixed, scanners, bands, output):
    selected = fixed[
        (fixed["model"] == "M1") & (fixed["feature"] != "intercept")
    ].copy()
    rows = []
    for band in bands:
        for scanner in scanners:
            rows.append(f"{band.replace('_', '–')} · {scanner.upper()}")
    selected["row"] = selected.apply(
        lambda item: f"{item.band.replace('_', '–')} · {item.scanner.upper()}",
        axis=1,
    )
    matrix = selected.pivot(index="row", columns="feature", values="coefficient").reindex(
        index=rows, columns=MAIN_FEATURES
    )
    p_values = selected.pivot(index="row", columns="feature", values="p_value_normal").reindex(
        index=rows, columns=MAIN_FEATURES
    )
    limit = max(float(np.nanquantile(np.abs(matrix.to_numpy()), 0.98)), 0.05)
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(
        matrix,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            marker = "*" if p_values.iloc[row, column] < 0.05 else ""
            axis.text(
                column,
                row,
                f"{matrix.iloc[row, column]:+.2f}{marker}",
                ha="center",
                va="center",
                fontsize=8,
            )
    axis.set_xticks(
        range(len(MAIN_FEATURES)),
        [FEATURE_LABELS[value] for value in MAIN_FEATURES],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(range(len(rows)), rows)
    axis.set_title("Standardized background fixed effects on log₂ tissue transfer")
    figure.colorbar(image, ax=axis, label="Coefficient per 1 SD background contrast")
    figure.text(
        0.5,
        0.01,
        "* nominal normal-approximation p<0.05; model-level ML tests are primary",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return save_figure(figure, output, "figure2_background_fixed_effects")


def main(argv=None):
    args = parse_args(argv)
    cohort_output = Path(args.cohort_output)
    background_dir = Path(args.background_aggregate)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scanners = [value for value in args.scanners if value != "at2"]
    invalid_scanners = sorted(set(scanners) - set(SCANNERS[1:]))
    invalid_bands = sorted(set(args.bands) - set(BANDS))
    if invalid_scanners or invalid_bands:
        raise ValueError((invalid_scanners, invalid_bands))

    replicates, annotation = load_annotated_replicates(
        cohort_output, Path(args.annotation)
    )
    annotation = annotation[
        annotation["slide_id"].isin(replicates["slide_id"].astype(str).unique())
    ].copy()
    if replicates["slide_id"].nunique() != args.expected_slides:
        raise ValueError(replicates["slide_id"].nunique())
    features = build_background_features(background_dir, annotation, args.cv_folds)
    joined = replicates.merge(
        features,
        on=["slide_id", "scanner", "band", "tissue_type"],
        validate="many_to_one",
    )
    expected_rows = args.expected_slides * len(SCANNERS[1:]) * len(BANDS) * 5
    if len(joined) != expected_rows:
        raise ValueError((len(joined), expected_rows))
    features.to_csv(output / "background_features.csv", index=False)
    joined.to_csv(output / "model_data.csv.gz", index=False, compression="gzip")

    tasks = []
    for band in args.bands:
        for scanner in scanners:
            tasks.append((scanner, band))
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = {
            executor.submit(
                fit_one_configuration,
                scanner,
                band,
                joined[
                    (joined["scanner"] == scanner) & (joined["band"] == band)
                ].copy(),
                args.cv_folds,
            ): (scanner, band)
            for scanner, band in tasks
        }
        for future in as_completed(futures):
            scanner, band = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"[exp08] {band}/{scanner}: "
                f"slide reduction={100 * result['comparison']['slide_variance_reduction_m1']:+.1f}%, "
                f"ΔAIC={result['comparison']['delta_aic_m1_minus_m0']:+.1f}",
                flush=True,
            )

    comparison = pd.DataFrame([item["comparison"] for item in results])
    models = pd.DataFrame([row for item in results for row in item["models"]])
    fixed = pd.DataFrame([row for item in results for row in item["fixed"]])
    cv = pd.DataFrame([row for item in results for row in item["cv"]])
    comparison["m1_vs_m0_q_bh"] = benjamini_hochberg(comparison["m1_vs_m0_p"])
    fixed["q_bh_all_background_terms"] = np.nan
    selected = (fixed["model"] == "M1") & (fixed["feature"] != "intercept")
    fixed.loc[selected, "q_bh_all_background_terms"] = benjamini_hochberg(
        fixed.loc[selected, "p_value_normal"]
    )
    cv_summary_rows = []
    for (scanner, band), frame in cv.groupby(["scanner", "band"]):
        wide = frame.pivot(
            index="slide_id", columns="model", values=["observed", "predicted", "error"]
        )
        observed = wide[("observed", "M0")].to_numpy()
        error0 = wide[("error", "M0")].to_numpy()
        error1 = wide[("error", "M1")].to_numpy()
        rmse0 = float(np.sqrt(np.mean(np.square(error0))))
        rmse1 = float(np.sqrt(np.mean(np.square(error1))))
        mae0 = float(np.mean(np.abs(error0)))
        mae1 = float(np.mean(np.abs(error1)))
        cv_summary_rows.append(
            {
                "scanner": scanner,
                "band": band,
                "slides": len(observed),
                "cv_rmse_m0": rmse0,
                "cv_rmse_m1": rmse1,
                "cv_rmse_improvement_fraction": 1.0 - rmse1 / rmse0,
                "cv_mae_m0": mae0,
                "cv_mae_m1": mae1,
                "cv_mae_improvement_fraction": 1.0 - mae1 / mae0,
                "paired_abs_error_wilcoxon_p": float(
                    stats.wilcoxon(np.abs(error0), np.abs(error1)).pvalue
                ),
                "all_cv_optimizers_converged": bool(
                    frame["optimizer_converged"].all()
                ),
            }
        )
    cv_summary = pd.DataFrame(cv_summary_rows)
    cv_summary["cv_rmse_improvement_q_bh"] = benjamini_hochberg(
        cv_summary["paired_abs_error_wilcoxon_p"]
    )

    if not comparison[["all_reml_converged", "all_ml_converged"]].all().all():
        failed = comparison[
            ~(comparison["all_reml_converged"] & comparison["all_ml_converged"])
        ]
        raise RuntimeError(f"full model convergence failures:\n{failed}")
    if not cv_summary["all_cv_optimizers_converged"].all():
        failed = cv_summary[~cv_summary["all_cv_optimizers_converged"]]
        raise RuntimeError(f"CV convergence failures:\n{failed}")

    comparison.to_csv(output / "model_comparison.csv", index=False)
    models.to_csv(output / "variance_components.csv", index=False)
    fixed.to_csv(output / "fixed_effects.csv", index=False)
    cv.to_csv(output / "cv_predictions.csv", index=False)
    cv_summary.to_csv(output / "cv_summary.csv", index=False)
    figures = []
    figures.extend(render_comparison(comparison, cv_summary, scanners, args.bands, output))
    figures.extend(render_fixed_effects(fixed, scanners, args.bands, output))

    merged = comparison.merge(cv_summary, on=["scanner", "band"])
    with (output / "summary.json").open("w") as handle:
        json.dump(
            {
                "analysis": "background_augmented_nested_mixed_effects",
                "models": {
                    "M0": "intercept + tissue + slide-within-tissue random slopes",
                    "M1": "M0 + five scanner-centred background physics covariates",
                    "M2": "M1 + glass-QC acceptance sensitivity covariate",
                },
                "fixed_model_comparison": "maximum likelihood",
                "variance_components": "restricted maximum likelihood",
                "prediction": f"{args.cv_folds}-fold grouped held-out-slide CV with training tissue BLUP",
                "slides": args.expected_slides,
                "tissues": int(joined["tissue_type"].nunique()),
                "scanners": scanners,
                "bands": args.bands,
                "main_features": list(MAIN_FEATURES),
                "figures": figures,
                "model_comparison": merged.to_dict("records"),
            },
            handle,
            indent=2,
        )
    print("[exp08] model comparison", flush=True)
    print(
        merged[
            [
                "band",
                "scanner",
                "slide_variance_reduction_m1",
                "tissue_variance_reduction_m1",
                "delta_aic_m1_minus_m0",
                "m1_vs_m0_p",
                "cv_rmse_improvement_fraction",
            ]
        ].sort_values(["band", "scanner"]).to_string(index=False),
        flush=True,
    )
    print(f"[exp08] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
