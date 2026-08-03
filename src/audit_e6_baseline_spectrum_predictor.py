"""Audit and lock the descriptive E6 baseline-spectrum predictor results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_e0_pfm_checkpoints import sha256


EXPECTED_ROWS = {
    "baseline_predictors.csv": 109,
    "crossfit_predictions.csv": 2_180,
    "prediction_metrics.csv": 40,
    "full_population_coefficients.csv": 120,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/e6_baseline_spectrum_predictor")
    parser.add_argument("--contract", default="docs/e6_baseline_spectrum_predictor_contract.md")
    parser.add_argument("--e6-lock", default="outputs/e6_heterogeneity_results_lock/summary.json")
    parser.add_argument("--output", default="outputs/e6_baseline_spectrum_predictor_lock")
    return parser.parse_args()


def main():
    args = parse_args()
    results = Path(args.results)
    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text())
    contract_path = Path(args.contract)
    e6_lock_path = Path(args.e6_lock)
    if summary.get("analysis_gate_pass") is not True:
        raise RuntimeError("baseline-spectrum predictor analysis gate has not passed")
    if summary.get("contract_sha256") != sha256(contract_path):
        raise ValueError("baseline-spectrum predictor contract hash mismatch")
    if json.loads(e6_lock_path.read_text()).get("result_lock_pass") is not True:
        raise RuntimeError("E6 heterogeneity result lock has not passed")
    frames = {}
    manifest_rows = []
    for filename, expected_rows in EXPECTED_ROWS.items():
        path = results / filename
        frame = pd.read_csv(path, dtype={"slide_id": str})
        frames[filename] = frame
        if (
            len(frame) != expected_rows
            or not np.isfinite(frame.select_dtypes(include=[np.number])).all().all()
            or summary["tables"][filename]["sha256"] != sha256(path)
        ):
            raise ValueError(f"{filename}: row/finiteness/hash audit failed")
        manifest_rows.append({
            "artifact": str(path),
            "rows": len(frame),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    predictions = frames["crossfit_predictions.csv"]
    for (_, _), frame in predictions.groupby(["encoder_id", "method"]):
        for tissue, heldout in frame.groupby("tissue_type"):
            training_mean = frame.loc[
                frame["tissue_type"] != tissue, "radius_benefit"
            ].mean()
            if not np.allclose(
                heldout["null_predicted_radius_benefit"], training_mean, atol=1e-12
            ):
                raise ValueError("leave-one-tissue-out null prediction audit failed")
    metrics = frames["prediction_metrics.csv"]
    if (
        metrics.groupby("analysis_set")["encoder_id"].nunique().ne(4).any()
        or metrics.groupby("analysis_set")["method"].nunique().ne(5).any()
    ):
        raise ValueError("predictor metric grid mismatch")
    bootstrap_path = results / "hierarchical_bootstrap_weights.npz"
    if summary.get("bootstrap_weights_sha256") != sha256(bootstrap_path):
        raise ValueError("predictor bootstrap hash mismatch")
    with np.load(bootstrap_path) as source:
        if (
            source["full_weights"].shape != (5_000, 109)
            or source["min3_weights"].shape != (5_000, 98)
            or not np.allclose(source["full_weights"].sum(axis=1), 1.0, atol=1e-6)
            or not np.allclose(source["min3_weights"].sum(axis=1), 1.0, atol=1e-6)
        ):
            raise ValueError("predictor bootstrap population mismatch")
    manifest_rows.append({
        "artifact": str(bootstrap_path),
        "rows": 10_000,
        "bytes": bootstrap_path.stat().st_size,
        "sha256": sha256(bootstrap_path),
    })
    for filename in (
        "figure_s_baseline_spectrum_prediction.png",
        "figure_s_baseline_spectrum_prediction.pdf",
    ):
        path = results / filename
        if path.stat().st_size <= 10_000 or summary["figures"][filename]["sha256"] != sha256(path):
            raise ValueError(f"{filename}: figure audit failed")
        manifest_rows.append({
            "artifact": str(path),
            "rows": 1,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest_rows.append({
        "artifact": str(summary_path),
        "rows": 1,
        "bytes": summary_path.stat().st_size,
        "sha256": sha256(summary_path),
    })
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output / "artifact_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    full = metrics[metrics["analysis_set"] == "full_37_tissues"].sort_values(
        ["encoder_id", "method"]
    )
    result = {
        "analysis": "e6_baseline_spectrum_predictor_result_lock",
        "placement": "post-E6 descriptive secondary; no morphology or causal claim",
        "cells": 20,
        "crossfit_predictions": 2_180,
        "full_results": full.to_dict("records"),
        "contract_sha256": sha256(contract_path),
        "e6_heterogeneity_result_lock_sha256": sha256(e6_lock_path),
        "analysis_summary_sha256": sha256(summary_path),
        "artifacts": len(manifest),
        "artifact_bytes": int(manifest["bytes"].sum()),
        "artifact_manifest": str(manifest_path.resolve()),
        "result_lock_pass": True,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
