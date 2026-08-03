"""Evaluate the frozen six-variable baseline-spectrum correction predictor."""

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
from scipy import stats

from analyze_e7_tissue_probe import bootstrap_weights, tissue_balanced_estimate
from e5_comparator_population import FEATURE_CONDITIONS, IMAGE_CONDITIONS, SCANNERS
from fetch_e0_pfm_checkpoints import sha256


METHODS = (*IMAGE_CONDITIONS, *FEATURE_CONDITIONS)
BANDS = {
    "low_mid": (0.10, 0.30),
    "mid": (0.30, 0.60),
    "high": (0.60, 0.90),
}
PREDICTORS = (
    "at2_log2_power_low_mid",
    "at2_log2_power_mid",
    "at2_log2_power_high",
    "scanner_transfer_sd_low_mid",
    "scanner_transfer_sd_mid",
    "scanner_transfer_sd_high",
)
RIDGE_ALPHA = 1.0
BOOTSTRAP_SEED = 20_260_805
MODEL_LABELS = {
    "resnet50": "ResNet50",
    "uni_v1": "UNI v1",
    "conch_v1": "CONCH v1",
    "virchow2": "Virchow2",
}
METHOD_LABELS = {
    "reinhard_lab": "Reinhard",
    "paired_od_affine": "Paired OD",
    "frequency_calibration": "Frequency",
    "coral": "CORAL",
    "orthogonal_procrustes": "Procrustes",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectra", default="outputs/e1_native_aa_spectral_109/slide_spectra.csv")
    parser.add_argument("--bands", default="outputs/e1_native_aa_spectral_109/slide_band_summary.csv")
    parser.add_argument("--geometry", default="outputs/e0_native_geometry_final/native_geometry_manifest.csv")
    parser.add_argument("--e5-invariance", default="outputs/e5_comparator_frontier/slide_invariance.csv")
    parser.add_argument("--contract", default="docs/e6_baseline_spectrum_predictor_contract.md")
    parser.add_argument("--output", default="outputs/e6_baseline_spectrum_predictor")
    return parser.parse_args()


def build_predictors(spectra: pd.DataFrame, bands: pd.DataFrame, annotation: pd.DataFrame):
    at2 = spectra[spectra["scanner"] == "at2"].copy()
    rows = []
    for slide_id, frame in at2.groupby("slide_id", sort=True):
        row = {"slide_id": str(slide_id)}
        for band, (lower, upper) in BANDS.items():
            selected = frame[
                (frame["frequency_cyc_per_um"] >= lower)
                & (frame["frequency_cyc_per_um"] < upper)
            ]["radial_power"].to_numpy(dtype=float)
            if len(selected) < 2 or np.any(selected <= 0):
                raise ValueError(f"{slide_id}/{band}: invalid AT2 radial-power support")
            row[f"at2_log2_power_{band}"] = float(np.log2(selected).mean())
        rows.append(row)
    predictors = pd.DataFrame(rows)
    source = bands[bands["scanner"].isin(SCANNERS[1:])]
    spread = (
        source.groupby(["slide_id", "band"])["log2_relative_transfer"]
        .std(ddof=0)
        .unstack("band")
        .rename(columns=lambda value: f"scanner_transfer_sd_{value}")
        .reset_index()
    )
    predictors = predictors.merge(spread, on="slide_id", validate="one_to_one")
    predictors = predictors.merge(annotation, on="slide_id", validate="one_to_one")
    if (
        len(predictors) != 109
        or predictors["tissue_type"].nunique() != 37
        or not set(PREDICTORS).issubset(predictors)
        or not np.isfinite(predictors[list(PREDICTORS)]).all().all()
    ):
        raise ValueError("baseline-spectrum predictor population mismatch")
    predictors["slides_in_tissue"] = predictors["tissue_type"].map(
        predictors.groupby("tissue_type")["slide_id"].nunique()
    )
    return predictors


def build_responses(invariance: pd.DataFrame, predictors: pd.DataFrame):
    raw = invariance[invariance["condition"] == "raw"][
        ["encoder_id", "slide_id", "scanner_centroid_rms"]
    ].rename(columns={"scanner_centroid_rms": "raw_radius"})
    actual = invariance[invariance["condition"].isin(METHODS)].rename(
        columns={"scanner_centroid_rms": "method_radius"}
    )
    response = actual.merge(
        raw, on=["encoder_id", "slide_id"], validate="many_to_one"
    )
    response["radius_benefit"] = response["raw_radius"] - response["method_radius"]
    response = response.merge(predictors, on="slide_id", validate="many_to_one")
    if len(response) != 2_180:
        raise ValueError(f"expected 2,180 PFM/method/slide responses, got {len(response)}")
    return response


def ridge_fit_predict(x_train, y_train, x_test, alpha=RIDGE_ALPHA):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0, ddof=0)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("nonpositive training predictor scale")
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    intercept = float(y_train.mean())
    coefficients = np.linalg.solve(
        train.T @ train + alpha * np.eye(train.shape[1]),
        train.T @ (y_train - intercept),
    )
    return intercept + test @ coefficients, coefficients, mean, scale, intercept


def crossfit_cell(frame: pd.DataFrame):
    predictions = np.empty(len(frame), dtype=float)
    null_predictions = np.empty(len(frame), dtype=float)
    x = frame[list(PREDICTORS)].to_numpy(dtype=float)
    y = frame["radius_benefit"].to_numpy(dtype=float)
    tissues = frame["tissue_type"].to_numpy()
    for tissue in sorted(set(tissues)):
        test = tissues == tissue
        train = ~test
        prediction, _, _, _, intercept = ridge_fit_predict(
            x[train], y[train], x[test]
        )
        predictions[test] = prediction
        null_predictions[test] = intercept
    _, coefficients, mean, scale, intercept = ridge_fit_predict(x, y, x)
    return predictions, null_predictions, coefficients, mean, scale, intercept


def evaluate_predictions(frame: pd.DataFrame, plan, analysis_set: str):
    selected, tissues, slide_ids, weights = plan
    ordered = selected[["slide_id", "tissue_type"]].merge(
        frame,
        on=["slide_id", "tissue_type"],
        validate="one_to_one",
    )
    if ordered["slide_id"].tolist() != slide_ids:
        raise ValueError("predictor evaluation slide order mismatch")
    observed = ordered["radius_benefit"].to_numpy(dtype=float)
    predicted = ordered["predicted_radius_benefit"].to_numpy(dtype=float)
    null = ordered["null_predicted_radius_benefit"].to_numpy(dtype=float)
    model_squared = np.square(observed - predicted)
    null_squared = np.square(observed - null)
    model_absolute = np.abs(observed - predicted)
    null_absolute = np.abs(observed - null)
    labels = ordered["tissue_type"].to_numpy()
    mse_model = tissue_balanced_estimate(model_squared, labels)
    mse_null = tissue_balanced_estimate(null_squared, labels)
    mae_model = tissue_balanced_estimate(model_absolute, labels)
    mae_null = tissue_balanced_estimate(null_absolute, labels)
    bootstrap_mse_model = weights @ model_squared
    bootstrap_mse_null = weights @ null_squared
    bootstrap_r2 = 1.0 - bootstrap_mse_model / bootstrap_mse_null
    bootstrap_mae_delta = weights @ (null_absolute - model_absolute)
    r2 = 1.0 - mse_model / mse_null
    correlation = float(stats.pearsonr(observed, predicted).statistic)
    return {
        "analysis_set": analysis_set,
        "tissues": len(tissues),
        "slides": len(slide_ids),
        "predictive_r2_vs_crossfit_null": r2,
        "predictive_r2_ci95_lower": float(np.quantile(bootstrap_r2, 0.025)),
        "predictive_r2_ci95_upper": float(np.quantile(bootstrap_r2, 0.975)),
        "model_mae": mae_model,
        "null_mae": mae_null,
        "mae_reduction": mae_null - mae_model,
        "mae_reduction_ci95_lower": float(np.quantile(bootstrap_mae_delta, 0.025)),
        "mae_reduction_ci95_upper": float(np.quantile(bootstrap_mae_delta, 0.975)),
        "pearson_observed_predicted": correlation,
    }


def render_figure(metrics: pd.DataFrame, coefficients: pd.DataFrame, output: Path):
    full = metrics[metrics["analysis_set"] == "full_37_tissues"]
    r2 = full.pivot(
        index="encoder_id", columns="method", values="predictive_r2_vs_crossfit_null"
    ).reindex(index=MODEL_LABELS, columns=METHODS)
    mae = full.pivot(
        index="encoder_id", columns="method", values="mae_reduction"
    ).reindex(index=MODEL_LABELS, columns=METHODS)
    coefficient_mean = (
        coefficients.groupby("predictor")["standardized_ridge_coefficient"]
        .mean()
        .reindex(PREDICTORS)
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.3))
    limit = max(0.05, float(np.nanmax(np.abs(r2.to_numpy()))))
    image = axes[0].imshow(r2, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(r2.shape[0]):
        for column in range(r2.shape[1]):
            axes[0].text(column, row, f"{r2.iloc[row, column]:+.2f}", ha="center", va="center", fontsize=8)
    axes[0].set_title("A  Tissue-held-out predictive R²")
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    limit = max(1e-4, float(np.nanmax(np.abs(mae.to_numpy()))))
    image = axes[1].imshow(mae, cmap="PiYG", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(mae.shape[0]):
        for column in range(mae.shape[1]):
            axes[1].text(column, row, f"{mae.iloc[row, column]:+.4f}", ha="center", va="center", fontsize=8)
    axes[1].set_title("B  Absolute-error reduction vs null")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].barh(range(len(coefficient_mean)), coefficient_mean, color="#2563EB")
    axes[2].axvline(0, color="#6B7280", linewidth=0.8)
    axes[2].set_yticks(range(len(coefficient_mean)), [value.replace("_", " ") for value in coefficient_mean.index], fontsize=8)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Mean standardized ridge coefficient")
    axes[2].set_title("C  Descriptive coefficient direction")
    for axis in axes[:2]:
        axis.set_xticks(range(len(METHODS)), [METHOD_LABELS[m] for m in METHODS], rotation=35, ha="right")
        axis.set_yticks(range(len(MODEL_LABELS)), [MODEL_LABELS[m] for m in MODEL_LABELS])
    figure.suptitle("Baseline-spectrum prediction of correction response", fontsize=14, y=1.02)
    figure.tight_layout()
    paths = {}
    for suffix in ("png", "pdf"):
        path = output / f"figure_s_baseline_spectrum_prediction.{suffix}"
        figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths[suffix] = str(path)
    plt.close(figure)
    return paths


def main():
    args = parse_args()
    contract_path = Path(args.contract)
    if "**Status:** FROZEN" not in contract_path.read_text():
        raise RuntimeError("baseline-spectrum predictor contract is not frozen")
    geometry = pd.read_csv(args.geometry, dtype={"slide_id": str})
    annotation = geometry[["slide_id", "tissue_type"]].drop_duplicates()
    if len(annotation) != 109 or annotation["tissue_type"].nunique() != 37:
        raise ValueError("predictor tissue annotation mismatch")
    spectra_path = Path(args.spectra)
    bands_path = Path(args.bands)
    invariance_path = Path(args.e5_invariance)
    predictors = build_predictors(
        pd.read_csv(spectra_path, dtype={"slide_id": str}),
        pd.read_csv(bands_path, dtype={"slide_id": str}),
        annotation,
    )
    response = build_responses(
        pd.read_csv(invariance_path, dtype={"slide_id": str}), predictors
    )
    prediction_rows = []
    coefficient_rows = []
    for (encoder_id, method), frame in response.groupby(
        ["encoder_id", "condition"], sort=True
    ):
        frame = frame.sort_values("slide_id").reset_index(drop=True)
        predicted, null, coefficients, means, scales, intercept = crossfit_cell(frame)
        for index, row in frame.iterrows():
            prediction_rows.append({
                "encoder_id": encoder_id,
                "method": method,
                "slide_id": row["slide_id"],
                "tissue_type": row["tissue_type"],
                "slides_in_tissue": int(row["slides_in_tissue"]),
                "radius_benefit": row["radius_benefit"],
                "predicted_radius_benefit": predicted[index],
                "null_predicted_radius_benefit": null[index],
            })
        for predictor, coefficient, mean, scale in zip(
            PREDICTORS, coefficients, means, scales
        ):
            coefficient_rows.append({
                "encoder_id": encoder_id,
                "method": method,
                "predictor": predictor,
                "standardized_ridge_coefficient": coefficient,
                "predictor_population_mean": mean,
                "predictor_population_sd": scale,
                "response_population_intercept": intercept,
            })
    predictions = pd.DataFrame(prediction_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    slide_frame = predictors[["slide_id", "tissue_type", "slides_in_tissue"]]
    plans = {
        "full_37_tissues": bootstrap_weights(slide_frame, 1, BOOTSTRAP_SEED),
        "min3_sensitivity": bootstrap_weights(slide_frame, 3, BOOTSTRAP_SEED + 1),
    }
    metric_rows = []
    for (encoder_id, method), frame in predictions.groupby(
        ["encoder_id", "method"], sort=True
    ):
        for analysis_set, plan in plans.items():
            metric_rows.append({
                "encoder_id": encoder_id,
                "method": method,
                **evaluate_predictions(frame, plan, analysis_set),
            })
    metrics = pd.DataFrame(metric_rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    table_paths = {
        "baseline_predictors.csv": output / "baseline_predictors.csv",
        "crossfit_predictions.csv": output / "crossfit_predictions.csv",
        "prediction_metrics.csv": output / "prediction_metrics.csv",
        "full_population_coefficients.csv": output / "full_population_coefficients.csv",
    }
    predictors.to_csv(table_paths["baseline_predictors.csv"], index=False)
    predictions.to_csv(table_paths["crossfit_predictions.csv"], index=False)
    metrics.to_csv(table_paths["prediction_metrics.csv"], index=False)
    coefficients.to_csv(table_paths["full_population_coefficients.csv"], index=False)
    temporary = output / f".bootstrap_weights.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        temporary,
        full_slide_ids=np.asarray(plans["full_37_tissues"][2]),
        full_weights=plans["full_37_tissues"][3],
        min3_slide_ids=np.asarray(plans["min3_sensitivity"][2]),
        min3_weights=plans["min3_sensitivity"][3],
    )
    bootstrap_path = output / "hierarchical_bootstrap_weights.npz"
    os.replace(temporary, bootstrap_path)
    figures = render_figure(metrics, coefficients, output)
    expected_rows = {
        "baseline_predictors.csv": 109,
        "crossfit_predictions.csv": 2_180,
        "prediction_metrics.csv": 40,
        "full_population_coefficients.csv": 120,
    }
    observed_rows = {
        filename: len(pd.read_csv(path)) for filename, path in table_paths.items()
    }
    gate = bool(
        observed_rows == expected_rows
        and np.isfinite(metrics.select_dtypes(include=[np.number])).all().all()
        and np.isfinite(predictions.select_dtypes(include=[np.number])).all().all()
    )
    summary = {
        "analysis": "e6_secondary_baseline_spectrum_predictor",
        "placement": "post-E6 descriptive secondary; no outcome-driven predictor selection",
        "predictors": list(PREDICTORS),
        "ridge_alpha": RIDGE_ALPHA,
        "cross_validation": "exact 37-fold leave-one-tissue-type-out",
        "bootstrap_replicates": 5_000,
        "bootstrap_seed_full": BOOTSTRAP_SEED,
        "bootstrap_seed_min3": BOOTSTRAP_SEED + 1,
        "expected_rows": expected_rows,
        "observed_rows": observed_rows,
        "contract_sha256": sha256(contract_path),
        "spectra_sha256": sha256(spectra_path),
        "bands_sha256": sha256(bands_path),
        "e5_invariance_sha256": sha256(invariance_path),
        "bootstrap_weights_sha256": sha256(bootstrap_path),
        "tables": {
            filename: {"path": str(path), "sha256": sha256(path)}
            for filename, path in table_paths.items()
        },
        "figures": {
            Path(path).name: {"path": path, "sha256": sha256(Path(path))}
            for path in figures.values()
        },
        "analysis_gate_pass": gate,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
