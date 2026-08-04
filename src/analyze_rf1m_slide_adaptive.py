"""Test whether a slide's own band energy predicts the gain it needs.

RF1M fits one gain per scanner and band from pooled training energy.  The
slide-adaptive proposal replaces the pooled denominator with the slide's own
source energy, which is measurable at deployment.  Both are special cases of

    log E_target(i) ~ a + b * log E_source(i)

with `b = 0` reproducing RF1M and `b = 1` reproducing per-slide normalization.

Applying gain `g` multiplies a band's energy by `g**2`, so the held-out error of
each estimator is just the error of its predicted target log-energy.  The test is
therefore a cross-fitted regression comparison and reads no PFM endpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from build_rf1m_slide_band_energy import ANALYSIS as ENERGY_ANALYSIS
from e5_comparator_population import FOVS, SCANNERS
from e5_reinhard_residual_frequency import RF1_FOLDS
from fetch_e0_pfm_checkpoints import sha256
from rf1m_combined import RF1M_SIGMAS, RF1M_VERSION


BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260803
ESTIMATORS = ("pooled", "per_slide", "regression")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy", default="outputs/rf1m_slide_adaptive/energy")
    parser.add_argument("--output", default="outputs/rf1m_slide_adaptive/result")
    return parser.parse_args()


def fit_log_energy_line(source_log: np.ndarray, target_log: np.ndarray) -> tuple[float, float]:
    """Least-squares intercept and slope of target log-energy on source log-energy."""
    x = np.asarray(source_log, dtype=np.float64)
    y = np.asarray(target_log, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 3:
        raise ValueError("log-energy regression needs matching vectors of length >= 3")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("log energies must be finite")
    variance = float(np.var(x))
    if variance <= 0:
        return float(y.mean()), 0.0
    slope = float(np.cov(x, y, ddof=0)[0, 1] / variance)
    return float(y.mean() - slope * x.mean()), slope


def predicted_target_log(
    estimator: str,
    source_log: np.ndarray,
    train_source_log: np.ndarray,
    train_target_log: np.ndarray,
) -> np.ndarray:
    """Predicted held-out target log band energy under each estimator.

    `pooled` is RF1M: the gain is the ratio of pooled training energies, so the
    prediction moves the held-out slide by a constant offset.  `per_slide`
    replaces the pooled source denominator with the slide's own energy, which
    predicts the pooled training target regardless of the slide.
    """
    source_log = np.asarray(source_log, dtype=np.float64)
    pooled_source = float(np.log(np.exp(train_source_log).mean()))
    pooled_target = float(np.log(np.exp(train_target_log).mean()))
    if estimator == "pooled":
        return source_log + (pooled_target - pooled_source)
    if estimator == "per_slide":
        return np.full_like(source_log, pooled_target)
    if estimator == "regression":
        intercept, slope = fit_log_energy_line(train_source_log, train_target_log)
        return intercept + slope * source_log
    raise ValueError(f"unknown estimator: {estimator}")


def load_energy(root: Path, fov: int, target: str = "at2"):
    """Load one band-energy file, accepting the pre-multi-target filename."""
    path = root / f"{target}_fov_{fov}.npz"
    if not path.exists():
        path = root / f"fov_{fov}.npz"
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    if not (
        summary.get("analysis") == ENERGY_ANALYSIS
        and summary.get("rf1m_version") == RF1M_VERSION
        and summary.get("pfm_feature_access") is False
        and summary.get("fov") == fov
        and summary.get("target", "at2") == target
        and summary.get("output_sha256") == sha256(path)
        and summary.get("energy_gate_pass") is True
    ):
        raise RuntimeError(f"invalid slide band-energy output: {path}")
    with np.load(path) as source:
        return {name: source[name] for name in source.files}, summary


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(BOOTSTRAP_SEED)

    rows = []
    hashes = {}
    for fov in FOVS:
        values, summary = load_energy(Path(args.energy), fov)
        hashes[f"fov_{fov}"] = summary["output_sha256"]
        fold_of_slide = values["fold_of_slide"]
        target_log = np.log(values["target_band_energy"])
        source_log = np.log(values["source_band_energy"])
        for scanner_index, scanner in enumerate(SCANNERS[1:]):
            for band_index, sigma in enumerate(RF1M_SIGMAS):
                errors = {name: np.empty(len(fold_of_slide)) for name in ESTIMATORS}
                slopes = []
                for fold in range(RF1_FOLDS):
                    heldout = fold_of_slide == fold
                    train = ~heldout
                    train_source = source_log[train, scanner_index, fold, band_index]
                    train_target = target_log[train, band_index]
                    heldout_source = source_log[heldout, scanner_index, fold, band_index]
                    heldout_target = target_log[heldout, band_index]
                    slopes.append(fit_log_energy_line(train_source, train_target)[1])
                    for name in ESTIMATORS:
                        errors[name][heldout] = (
                            predicted_target_log(
                                name, heldout_source, train_source, train_target
                            )
                            - heldout_target
                        )
                row = {
                    "fov": fov,
                    "scanner": scanner,
                    "sigma_px": sigma,
                    "slides": int(len(fold_of_slide)),
                    "mean_cv_slope": float(np.mean(slopes)),
                    "min_cv_slope": float(np.min(slopes)),
                    "max_cv_slope": float(np.max(slopes)),
                }
                for name in ESTIMATORS:
                    row[f"{name}_rmse"] = float(np.sqrt((errors[name] ** 2).mean()))
                for name in ("per_slide", "regression"):
                    row[f"{name}_vs_pooled_percent"] = 100.0 * (
                        row[f"{name}_rmse"] / row["pooled_rmse"] - 1.0
                    )
                # Slide-blocked bootstrap of the paired squared-error difference.
                paired = errors["regression"] ** 2 - errors["pooled"] ** 2
                index = generator.integers(0, len(paired), size=(BOOTSTRAP_REPLICATES, len(paired)))
                replicates = paired[index].mean(axis=1)
                row["regression_minus_pooled_mse_ci_low"] = float(np.quantile(replicates, 0.025))
                row["regression_minus_pooled_mse_ci_high"] = float(np.quantile(replicates, 0.975))
                row["regression_better"] = bool(row["regression_minus_pooled_mse_ci_high"] < 0)
                rows.append(row)

    table_path = output / "slide_adaptive_comparison.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    slopes = np.asarray([row["mean_cv_slope"] for row in rows])
    improvement = np.asarray([row["regression_vs_pooled_percent"] for row in rows])
    per_slide = np.asarray([row["per_slide_vs_pooled_percent"] for row in rows])
    better = int(sum(row["regression_better"] for row in rows))
    result = {
        "analysis": "rf1m_slide_adaptive_feasibility",
        "rf1m_version": RF1M_VERSION,
        "status": "IMAGE_ONLY_FEASIBILITY_TEST",
        "outcome_access": False,
        "pfm_feature_access": False,
        "question": (
            "Does a slide's own post-Reinhard band energy predict the band energy its "
            "correction must reach, better than the pooled training ratio RF1M uses?"
        ),
        "model": "log E_target(i) ~ a + b log E_source(i); b=0 is RF1M, b=1 is per-slide",
        "validation": "exact five-fold cross-fitting on the frozen RF1 physical-slide folds",
        "cells": len(rows),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "mean_cv_slope": float(slopes.mean()),
        "min_cv_slope": float(slopes.min()),
        "max_cv_slope": float(slopes.max()),
        "cells_with_slope_above_half": int((slopes > 0.5).sum()),
        "mean_regression_vs_pooled_percent": float(improvement.mean()),
        "mean_per_slide_vs_pooled_percent": float(per_slide.mean()),
        "cells_regression_better_by_ci": better,
        "energy_sha256": hashes,
        "artifacts": {table_path.name: sha256(table_path)},
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
