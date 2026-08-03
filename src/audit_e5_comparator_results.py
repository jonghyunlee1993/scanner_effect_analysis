"""Lock E5 comparator inputs, populations, endpoint tables and Main Figure 5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from e5_comparator_population import E5_VERSION, FEATURE_CONDITIONS, IMAGE_CONDITIONS
from fetch_e0_pfm_checkpoints import sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", default="docs/e4_e7_decision_record.md")
    parser.add_argument("--execution-contract", default="docs/e5_comparator_execution_contract.md")
    parser.add_argument("--stability", default="outputs/e5_input_stability/summary.json")
    parser.add_argument("--image-statistics", default="outputs/e5_image_statistics")
    parser.add_argument("--feature-statistics", default="outputs/e5_feature_statistics")
    parser.add_argument("--population-audit", default="outputs/e5_comparator_features/audit")
    parser.add_argument("--frontier", default="outputs/e5_comparator_frontier")
    parser.add_argument("--figure", default="outputs/e5_comparator_figure")
    parser.add_argument("--output", default="outputs/e5_comparator_results_lock")
    return parser.parse_args()


def require(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def main():
    args = parse_args()
    decision = Path(args.decision)
    execution = Path(args.execution_contract)
    stability_path = Path(args.stability)
    stability = json.loads(stability_path.read_text())
    require(stability.get("stability_gate_pass") is True, "E5 stability gate failed")
    require(stability.get("outcome_access") is False, "stability audit accessed outcomes")
    require(stability.get("execution_contract_sha256") == sha256(execution), "execution-contract hash drift")
    require(stability.get("decision_record_sha256") == sha256(decision), "decision-record hash drift")

    population_root = Path(args.population_audit)
    population_summary_path = population_root / "summary.json"
    population = json.loads(population_summary_path.read_text())
    require(population.get("audit_pass") is True, "E5 comparator population audit failed")
    require(population.get("total_features_observed") == 1_308_000, "E5 comparator count mismatch")
    require(population.get("stability_manifest_sha256") == sha256(stability_path), "population stability hash drift")

    frontier_root = Path(args.frontier)
    frontier_summary_path = frontier_root / "summary.json"
    frontier = json.loads(frontier_summary_path.read_text())
    require(frontier.get("analysis_gate_pass") is True, "E5 frontier analysis failed")
    require(frontier.get("endpoint_rows") == 24, "E5 endpoint row count mismatch")
    require(frontier.get("slide_invariance_rows") == 2_616, "E5 invariance row count mismatch")
    require(frontier.get("population_audit_sha256") == sha256(population_summary_path), "frontier audit hash drift")
    require(frontier.get("decision_record_sha256") == sha256(decision), "frontier decision hash drift")
    require(frontier.get("execution_contract_sha256") == sha256(execution), "frontier contract hash drift")

    endpoints = pd.read_csv(frontier_root / "endpoint_summary.csv")
    content = pd.read_csv(frontier_root / "slide_content_margin.csv")
    collapse = pd.read_csv(frontier_root / "slide_collapse.csv")
    common = pd.read_csv(frontier_root / "common_pfm_summary.csv")
    decisions = pd.read_csv(frontier_root / "decision_comparison.csv")
    require(len(endpoints) == 24, "endpoint table is incomplete")
    require(len(content) == 13_080, "content table is incomplete")
    require(len(collapse) == 39_240, "collapse table is incomplete")
    require(len(common) == 6 and len(decisions) == 4, "E5 summary table is incomplete")
    require(set(endpoints["condition"]) == {"raw", *IMAGE_CONDITIONS, *FEATURE_CONDITIONS}, "E5 method set drift")

    figure_root = Path(args.figure)
    figure_summary_path = figure_root / "summary.json"
    figure = json.loads(figure_summary_path.read_text())
    require(figure.get("figure_gate_pass") is True, "Main Figure 5 gate failed")
    require(figure.get("frontier_summary_sha256") == sha256(frontier_summary_path), "figure/frontier hash drift")
    png = figure_root / "figure5_actual_correction_frontier.png"
    pdf = figure_root / "figure5_actual_correction_frontier.pdf"
    require(figure.get("png_sha256") == sha256(png), "Figure 5 PNG hash drift")
    require(figure.get("pdf_sha256") == sha256(pdf), "Figure 5 PDF hash drift")

    paths = [
        decision,
        execution,
        stability_path,
        population_summary_path,
        population_root / "shard_audit.csv",
        frontier_summary_path,
        *(frontier_root / filename for filename in (
            "slide_invariance.csv",
            "slide_content_margin.csv",
            "slide_collapse.csv",
            "slide_image_clipping.csv",
            "endpoint_summary.csv",
            "content_scanner_summary.csv",
            "collapse_summary.csv",
            "common_pfm_summary.csv",
            "decision_comparison.csv",
            "image_clipping_summary.csv",
        )),
        figure_summary_path,
        png,
        pdf,
    ]
    for fov in (224, 256, 512):
        paths.extend(
            [
                Path(args.image_statistics) / f"fov_{fov}.npz",
                Path(args.image_statistics) / f"fov_{fov}.summary.json",
            ]
        )
    for model in ("resnet50", "uni_v1", "conch_v1", "virchow2"):
        paths.extend(
            [
                Path(args.feature_statistics) / f"{model}.h5",
                Path(args.feature_statistics) / f"{model}.summary.json",
            ]
        )
    missing = [str(path) for path in paths if not path.exists()]
    require(not missing, f"missing E5 lock artifacts: {missing}")
    artifact_rows = [
        {
            "path": str(path.resolve()),
            "bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }
        for path in paths
    ]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(artifact_rows).to_csv(output / "artifact_manifest.csv", index=False)
    safe = endpoints[endpoints["condition"] != "raw"].groupby("encoder_id")["safe_for_pfm"].sum()
    improved = endpoints[endpoints["condition"] != "raw"].groupby("encoder_id")["safe_and_invariance_improved"].sum()
    lock = {
        "analysis": "e5_comparator_results_lock",
        "e5_version": E5_VERSION,
        "decision_record_sha256": sha256(decision),
        "execution_contract_sha256": sha256(execution),
        "stability_manifest_sha256": sha256(stability_path),
        "selected_coral_shrinkage": stability["selected_coral_shrinkage"],
        "selected_frequency_gain_cap": stability["selected_frequency_gain_cap"],
        "population_features": population["total_features_observed"],
        "slide_result_rows": int(len(pd.read_csv(frontier_root / "slide_invariance.csv")) + len(content) + len(collapse)),
        "endpoint_rows": len(endpoints),
        "safe_method_counts_by_pfm": {key: int(value) for key, value in safe.items()},
        "safe_and_improved_counts_by_pfm": {key: int(value) for key, value in improved.items()},
        "common_safe_methods": common[
            (common["condition"] != "raw")
            & common["common_safe_across_four_pfms"]
        ]["condition"].tolist(),
        "common_safe_and_improved_methods": common[
            (common["condition"] != "raw")
            & common["safe_and_improved_all_four_pfms"]
        ]["condition"].tolist(),
        "decision_differs_pfms": int(decisions["decision_differs"].sum()),
        "invariance_improved_but_unsafe_cells": int(
            frontier["invariance_improved_but_unsafe_cells"]
        ),
        "invariance_improved_but_unsafe_method_pfm": frontier[
            "invariance_improved_but_unsafe_method_pfm"
        ],
        "artifacts": len(artifact_rows),
        "artifact_bytes": int(sum(row["bytes"] for row in artifact_rows)),
        "artifact_manifest": str((output / "artifact_manifest.csv").resolve()),
        "result_lock_pass": True,
    }
    (output / "summary.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
