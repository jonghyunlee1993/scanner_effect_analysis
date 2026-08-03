"""Independently audit and lock the exact E6 LOTO transfer frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_e0_pfm_checkpoints import sha256


EXPECTED_ROWS = {
    "slide_invariance.csv": 2_616,
    "slide_content_margin.csv": 13_080,
    "slide_collapse.csv": 39_240,
    "slide_image_clipping.csv": 6_540,
    "endpoint_summary.csv": 24,
    "content_scanner_summary.csv": 120,
    "collapse_summary.csv": 360,
    "common_pfm_summary.csv": 6,
    "decision_comparison.csv": 4,
    "image_clipping_summary.csv": 60,
    "loso_loto_endpoint_comparison.csv": 20,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/e6_loto_frontier")
    parser.add_argument("--population-audit", default="outputs/e6_loto_features/audit/summary.json")
    parser.add_argument("--e5-lock", default="outputs/e5_comparator_results_lock/summary.json")
    parser.add_argument("--heterogeneity-lock", default="outputs/e6_heterogeneity_results_lock/summary.json")
    parser.add_argument("--e6-contract", default="docs/e6_heterogeneity_execution_contract.md")
    parser.add_argument("--output", default="outputs/e6_loto_results_lock")
    return parser.parse_args()


def main():
    args = parse_args()
    results = Path(args.results)
    analysis_summary_path = results / "summary.json"
    population_audit_path = Path(args.population_audit)
    e5_lock_path = Path(args.e5_lock)
    heterogeneity_lock_path = Path(args.heterogeneity_lock)
    e6_contract_path = Path(args.e6_contract)
    summary = json.loads(analysis_summary_path.read_text())
    population = json.loads(population_audit_path.read_text())
    e5_lock = json.loads(e5_lock_path.read_text())
    heterogeneity_lock = json.loads(heterogeneity_lock_path.read_text())
    if summary.get("analysis_gate_pass") is not True:
        raise RuntimeError("E6 LOTO frontier gate has not passed")
    if population.get("audit_pass") is not True or population.get("total_features_observed") != 1_308_000:
        raise RuntimeError("E6 LOTO feature-population audit has not passed")
    if e5_lock.get("result_lock_pass") is not True or heterogeneity_lock.get("result_lock_pass") is not True:
        raise RuntimeError("upstream E5/E6 result lock has not passed")
    if summary.get("population_audit_sha256") != sha256(population_audit_path):
        raise ValueError("LOTO population-audit hash mismatch")
    if summary.get("e6_contract_sha256") != sha256(e6_contract_path):
        raise ValueError("E6 contract hash mismatch")

    manifest_rows = []
    frames = {}
    for filename, expected_rows in EXPECTED_ROWS.items():
        path = results / filename
        frame = pd.read_csv(path, dtype={"slide_id": str})
        frames[filename] = frame
        numeric = frame.select_dtypes(include=[np.number])
        allowed_nan = {
            "endpoint_summary.csv": {
                "fidelity_constrained_rr",
                "safe_improved_rr_rank",
            },
            "decision_comparison.csv": {"fidelity_constrained_best_rr"},
            "loso_loto_endpoint_comparison.csv": {
                "safe_improved_rr_rank_loso",
                "safe_improved_rr_rank_loto",
            },
        }.get(filename, set())
        required_numeric = numeric.drop(
            columns=[column for column in allowed_nan if column in numeric],
            errors="ignore",
        )
        if (
            len(frame) != expected_rows
            or np.isinf(numeric.to_numpy()).any()
            or not np.isfinite(required_numeric).all().all()
        ):
            raise ValueError(f"{filename}: row/finiteness audit failed")
        manifest_rows.append({
            "artifact": str(path),
            "rows": len(frame),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    for filename in (
        "figure_s_loto_transfer_sensitivity.png",
        "figure_s_loto_transfer_sensitivity.pdf",
    ):
        path = results / filename
        if path.stat().st_size <= 10_000:
            raise ValueError(f"{filename}: implausibly small figure")
        manifest_rows.append({
            "artifact": str(path),
            "rows": 1,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    endpoints = frames["endpoint_summary.csv"]
    actual = endpoints[endpoints["condition"] != "raw"]
    if len(actual) != 20 or actual.groupby("encoder_id")["condition"].nunique().ne(5).any():
        raise ValueError("LOTO endpoint grid mismatch")
    common = frames["common_pfm_summary.csv"]
    recalculated_common_safe = sorted(
        condition
        for condition, frame in actual.groupby("condition")
        if frame["safe_for_pfm"].astype(bool).all()
    )
    reported_common_safe = sorted(
        condition
        for condition in summary["common_safe_methods"]
        if condition != "raw"
    )
    if recalculated_common_safe != reported_common_safe:
        raise ValueError("LOTO common-safe audit failed")
    if not np.array_equal(
        actual["safe_and_invariance_improved"].astype(bool).to_numpy(),
        (
            actual["safe_for_pfm"].astype(bool)
            & actual["invariance_improved"].astype(bool)
        ).to_numpy(),
    ):
        raise ValueError("LOTO safe-and-improved decision mismatch")
    comparison = frames["loso_loto_endpoint_comparison.csv"]
    if int(comparison["safe_decision_changed"].sum()) != summary["safe_decision_changes_from_loso"]:
        raise ValueError("LOSO/LOTO safe-decision comparison mismatch")
    if int(comparison["safe_improved_decision_changed"].sum()) != summary["safe_improved_decision_changes_from_loso"]:
        raise ValueError("LOSO/LOTO safe-improved comparison mismatch")

    manifest_rows.extend([
        {
            "artifact": str(analysis_summary_path),
            "rows": 1,
            "bytes": analysis_summary_path.stat().st_size,
            "sha256": sha256(analysis_summary_path),
        },
        {
            "artifact": str(population_audit_path),
            "rows": 1,
            "bytes": population_audit_path.stat().st_size,
            "sha256": sha256(population_audit_path),
        },
    ])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output / "artifact_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    result = {
        "analysis": "e6_exact_loto_transfer_result_lock",
        "placement": "secondary transfer sensitivity; E5 LOSO remains primary",
        "folds": 37,
        "slides": 109,
        "population_features": population["total_features_observed"],
        "common_safe_methods": recalculated_common_safe,
        "common_safe_and_improved_methods": sorted(
            condition
            for condition, frame in actual.groupby("condition")
            if frame["safe_and_invariance_improved"].astype(bool).all()
        ),
        "safe_decision_changes_from_loso": int(comparison["safe_decision_changed"].sum()),
        "safe_improved_decision_changes_from_loso": int(
            comparison["safe_improved_decision_changed"].sum()
        ),
        "endpoint_results": actual.sort_values(["encoder_id", "condition"]).to_dict("records"),
        "population_audit_sha256": sha256(population_audit_path),
        "e5_primary_result_lock_sha256": sha256(e5_lock_path),
        "e6_heterogeneity_result_lock_sha256": sha256(heterogeneity_lock_path),
        "e6_contract_sha256": sha256(e6_contract_path),
        "analysis_summary_sha256": sha256(analysis_summary_path),
        "artifacts": len(manifest),
        "artifact_bytes": int(manifest["bytes"].sum()),
        "artifact_manifest": str(manifest_path.resolve()),
        "result_lock_pass": True,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
