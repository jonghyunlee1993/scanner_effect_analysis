"""Lock and audit frozen E4 control-frontier results and Figure 4 provenance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from e4_control_population import CONTROL_VERSION, condition_names
from fetch_e0_pfm_checkpoints import sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", default="docs/e4_e7_decision_record.md")
    parser.add_argument(
        "--feature-audit", default="outputs/e4_control_features/audit/summary.json"
    )
    parser.add_argument("--frontier", default="outputs/e4_control_frontier")
    parser.add_argument("--figure", default="outputs/e4_control_figure")
    parser.add_argument("--output", default="outputs/e4_control_results_lock")
    return parser.parse_args()


def validate_endpoint_logic(endpoints: pd.DataFrame):
    failures = []
    expected_conditions = {"raw", *condition_names()}
    if len(endpoints) != 40:
        failures.append("endpoint_row_count")
    if set(endpoints["condition"]) != expected_conditions:
        failures.append("endpoint_conditions")
    if endpoints["encoder_id"].nunique() != 4:
        failures.append("endpoint_pfms")
    safe = endpoints["content_noninferiority_pass"] & endpoints[
        "collapse_every_scanner_pass"
    ]
    if not (safe == endpoints["safe_for_pfm"]).all():
        failures.append("safe_logic")
    safe_improved = safe & endpoints["invariance_improved"]
    if not (safe_improved == endpoints["safe_and_invariance_improved"]).all():
        failures.append("safe_improved_logic")
    return failures


def main():
    args = parse_args()
    decision_path = Path(args.decision)
    feature_audit_path = Path(args.feature_audit)
    frontier_root = Path(args.frontier)
    figure_root = Path(args.figure)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    feature_audit = json.loads(feature_audit_path.read_text())
    frontier_summary_path = frontier_root / "summary.json"
    frontier_summary = json.loads(frontier_summary_path.read_text())
    figure_summary_path = figure_root / "summary.json"
    figure_summary = json.loads(figure_summary_path.read_text())
    endpoint_path = frontier_root / "endpoint_summary.csv"
    endpoints = pd.read_csv(endpoint_path)
    content_scanner = pd.read_csv(frontier_root / "content_scanner_summary.csv")
    collapse_summary = pd.read_csv(frontier_root / "collapse_summary.csv")
    common = pd.read_csv(frontier_root / "common_pfm_summary.csv")
    slide_invariance = pd.read_csv(frontier_root / "slide_invariance.csv")
    slide_content = pd.read_csv(frontier_root / "slide_content_margin.csv")
    slide_collapse = pd.read_csv(frontier_root / "slide_collapse.csv")

    failures = validate_endpoint_logic(endpoints)
    if feature_audit.get("audit_pass") is not True:
        failures.append("feature_population_audit")
    if frontier_summary.get("analysis_complete") is not True:
        failures.append("frontier_complete")
    if frontier_summary.get("decision_record_sha256") != sha256(decision_path):
        failures.append("decision_record_hash")
    if figure_summary.get("figure_gate_pass") is not True:
        failures.append("figure_gate")
    if figure_summary.get("endpoint_summary_sha256") != sha256(endpoint_path):
        failures.append("figure_endpoint_hash")
    expected_rows = {
        "content_scanner_summary": (len(content_scanner), 200),
        "collapse_summary": (len(collapse_summary), 600),
        "common_pfm_summary": (len(common), 10),
        "slide_invariance": (len(slide_invariance), 4360),
        "slide_content_margin": (len(slide_content), 21800),
        "slide_collapse": (len(slide_collapse), 65400),
    }
    for name, (observed, expected) in expected_rows.items():
        if observed != expected:
            failures.append(f"{name}_rows")

    oracle = endpoints[endpoints["condition"] == "registered_loo_hf_mean_0p25"]
    if not (
        len(oracle) == 4
        and oracle["safe_for_pfm"].all()
        and oracle["invariance_improved"].all()
    ):
        failures.append("paired_oracle_four_pfm")
    for negative in ("hf_retention_0p00", "global_train_hf_mean_1p00"):
        rows = endpoints[endpoints["condition"] == negative]
        if not (len(rows) == 4 and rows["invariance_improved"].all() and (~rows["safe_for_pfm"]).all()):
            failures.append(f"negative_control_logic:{negative}")

    artifacts = [
        decision_path,
        feature_audit_path,
        frontier_summary_path,
        endpoint_path,
        frontier_root / "content_scanner_summary.csv",
        frontier_root / "collapse_summary.csv",
        frontier_root / "common_pfm_summary.csv",
        frontier_root / "slide_invariance.csv",
        frontier_root / "slide_content_margin.csv",
        frontier_root / "slide_collapse.csv",
        figure_summary_path,
        figure_root / "figure4_control_bounded_frontier.png",
        figure_root / "figure4_control_bounded_frontier.pdf",
    ]
    manifest_rows = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in artifacts
    ]
    with (output_root / "artifact_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(manifest_rows)

    lock_pass = not failures
    summary = {
        "analysis": "e4_control_results_lock",
        "control_version": CONTROL_VERSION,
        "artifacts": len(manifest_rows),
        "endpoint_rows": len(endpoints),
        "slide_result_rows": len(slide_invariance) + len(slide_content) + len(slide_collapse),
        "paired_oracle_safe_and_improved_pfms": int(
            oracle["safe_and_invariance_improved"].sum()
        ),
        "common_safe_and_improved_conditions": common.loc[
            common["safe_and_improved_all_four_pfms"], "condition"
        ].tolist(),
        "failures": failures,
        "result_lock_pass": lock_pass,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not lock_pass:
        raise RuntimeError("E4 control result lock failed")


if __name__ == "__main__":
    main()

