"""Leave-one-slide-out test of the prespecified scalar blur/sharpen manifold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_exp05_spectral_pilot import SCANNERS


DENSE_GAINS = np.linspace(0.0, 2.0, 401)
FIT_BAND = (0.10, 0.90)
BOOTSTRAP_SEED = 20260802


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="outputs/e1_native_aa_spectral_109")
    parser.add_argument("--output", default="outputs/e3_native_aa_scalar_reducibility")
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def dense_manifold(gains: np.ndarray, curves: np.ndarray):
    return np.stack(
        [np.interp(DENSE_GAINS, gains, curves[:, column]) for column in range(curves.shape[1])],
        axis=1,
    )


def project_curve(observed: np.ndarray, manifold: np.ndarray, fit_mask: np.ndarray):
    error = np.sqrt(np.mean((manifold[:, fit_mask] - observed[None, fit_mask]) ** 2, axis=1))
    best = int(np.argmin(error))
    return best, float(error[best])


def interpolate_curve(gain: float, gains: np.ndarray, curves: np.ndarray):
    return np.asarray(
        [np.interp(gain, gains, curves[:, column]) for column in range(curves.shape[1])]
    )


def build_arrays(spectra: pd.DataFrame, interventions: pd.DataFrame):
    slides = sorted(spectra["slide_id"].astype(str).unique())
    gains = np.asarray(sorted(interventions["effective_hf_gain"].unique()), dtype=float)
    first = spectra[
        spectra["slide_id"].astype(str).eq(slides[0]) & spectra["scanner"].eq("at2")
    ].sort_values("frequency_cyc_per_um")
    frequency = first["frequency_cyc_per_um"].to_numpy(float)
    scanner_curves = np.empty((len(slides), len(SCANNERS), len(frequency)), dtype=float)
    scalar_curves = np.empty((len(slides), len(gains), len(frequency)), dtype=float)
    for slide_index, slide_id in enumerate(slides):
        for scanner_index, scanner in enumerate(SCANNERS):
            frame = spectra[
                spectra["slide_id"].astype(str).eq(slide_id)
                & spectra["scanner"].eq(scanner)
            ].sort_values("frequency_cyc_per_um")
            if not np.array_equal(frame["frequency_cyc_per_um"].to_numpy(float), frequency):
                raise ValueError(f"{slide_id}/{scanner}: frequency grid differs")
            scanner_curves[slide_index, scanner_index] = frame[
                "log2_relative_transfer"
            ].to_numpy(float)
        for gain_index, gain in enumerate(gains):
            frame = interventions[
                interventions["slide_id"].astype(str).eq(slide_id)
                & np.isclose(interventions["effective_hf_gain"], gain)
            ].sort_values("frequency_cyc_per_um")
            scalar_curves[slide_index, gain_index] = frame[
                "log2_relative_transfer"
            ].to_numpy(float)
    return slides, gains, frequency, scanner_curves, scalar_curves


def leave_one_slide_out(spectra: pd.DataFrame, interventions: pd.DataFrame):
    slides, gains, frequency, scanner_curves, scalar_curves = build_arrays(
        spectra, interventions
    )
    fit_mask = (frequency >= FIT_BAND[0]) & (frequency < FIT_BAND[1])
    scalar_sum = scalar_curves.sum(axis=0)
    rows = []
    for slide_index, slide_id in enumerate(slides):
        train_mean = (scalar_sum - scalar_curves[slide_index]) / (len(slides) - 1)
        train_dense = dense_manifold(gains, train_mean)
        test_scalar = scalar_curves[slide_index]
        for scanner_index, scanner in enumerate(SCANNERS[1:], start=1):
            best, scanner_residual = project_curve(
                scanner_curves[slide_index, scanner_index], train_dense, fit_mask
            )
            equivalent_gain = float(DENSE_GAINS[best])
            matched_scalar = interpolate_curve(equivalent_gain, gains, test_scalar)
            null_best, null_residual = project_curve(matched_scalar, train_dense, fit_mask)
            rows.append(
                {
                    "slide_id": slide_id,
                    "scanner": scanner,
                    "equivalent_hf_gain": equivalent_gain,
                    "gain_at_boundary": bool(best in {0, len(DENSE_GAINS) - 1}),
                    "scanner_off_axis_rms_log2": scanner_residual,
                    "matched_scalar_equivalent_gain": float(DENSE_GAINS[null_best]),
                    "matched_scalar_null_rms_log2": null_residual,
                    "excess_off_axis_rms_log2": scanner_residual - null_residual,
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != 109 * 5 or not np.isfinite(result.select_dtypes(include=[np.number])).all().all():
        raise ValueError("LOSO scalar projection is incomplete or non-finite")
    return result


def bootstrap_summary(rows: pd.DataFrame, n_bootstrap: int):
    if n_bootstrap < 100:
        raise ValueError("at least 100 slide bootstrap replicates are required")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries = []
    draws = []
    for scanner in SCANNERS[1:]:
        frame = rows[rows["scanner"].eq(scanner)].sort_values("slide_id")
        values = frame["excess_off_axis_rms_log2"].to_numpy(float)
        indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
        bootstrap_median = np.median(values[indices], axis=1)
        bootstrap_mean = np.mean(values[indices], axis=1)
        for replicate in range(n_bootstrap):
            draws.append(
                {
                    "scanner": scanner,
                    "bootstrap_replicate": replicate,
                    "mean_excess_off_axis_rms_log2": float(bootstrap_mean[replicate]),
                    "median_excess_off_axis_rms_log2": float(bootstrap_median[replicate]),
                }
            )
        ci_low, ci_high = np.quantile(bootstrap_median, [0.025, 0.975])
        summaries.append(
            {
                "scanner": scanner,
                "slides": len(frame),
                "equivalent_gain_mean": float(frame["equivalent_hf_gain"].mean()),
                "equivalent_gain_median": float(frame["equivalent_hf_gain"].median()),
                "gain_boundary_fraction": float(frame["gain_at_boundary"].mean()),
                "scanner_off_axis_rms_median": float(frame["scanner_off_axis_rms_log2"].median()),
                "matched_scalar_null_rms_median": float(frame["matched_scalar_null_rms_log2"].median()),
                "excess_off_axis_rms_mean": float(values.mean()),
                "excess_off_axis_rms_median": float(np.median(values)),
                "excess_off_axis_rms_median_ci95_low": float(ci_low),
                "excess_off_axis_rms_median_ci95_high": float(ci_high),
                "scalar_family_rejected": bool(ci_low > 0.0),
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(draws)


def main():
    args = parse_args()
    cohort = Path(args.cohort)
    spectra = pd.read_csv(cohort / "slide_spectra.csv", dtype={"slide_id": str})
    interventions = pd.read_csv(
        cohort / "exp02_hf_gain_spectra.csv", dtype={"slide_id": str}
    )
    rows = leave_one_slide_out(spectra, interventions)
    summary, draws = bootstrap_summary(rows, args.bootstrap)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "loso_slide_projection.csv", index=False)
    summary.to_csv(output / "scanner_summary.csv", index=False)
    draws.to_csv(output / "slide_bootstrap.csv", index=False)
    run_summary = {
        "analysis": "e3_scalar_blur_sharpen_reducibility_loso",
        "slides": 109,
        "scanners": list(SCANNERS[1:]),
        "fit_band_cycles_per_um": list(FIT_BAND),
        "manifold": "AT2 FixedLaplacianPyramid detail gain 0.0..2.0",
        "cross_validation": "leave one physical slide out; train manifold mean uses the other 108 slides",
        "null": "held-out slide scalar intervention interpolated at the scanner-equivalent gain",
        "bootstrap_unit": "physical slide",
        "bootstrap_replicates": args.bootstrap,
        "rejection_rule": "lower 95% slide-bootstrap CI of median excess off-axis RMS > 0",
        "all_scanners_reject_scalar_family": bool(summary["scalar_family_rejected"].all()),
    }
    (output / "summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(run_summary, indent=2))
    if not summary["scalar_family_rejected"].all():
        raise SystemExit(2)


if __name__ == "__main__":
    main()
