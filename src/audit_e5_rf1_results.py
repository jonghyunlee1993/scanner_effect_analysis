"""Audit and lock the complete post-core E5-RF1 result population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from e5_reinhard_residual_frequency import RF1_CONDITION, RF1_VERSION
from fetch_e0_pfm_checkpoints import sha256


FRONTIER_ROWS = {
    "slide_invariance.csv": 872,
    "slide_content_margin.csv": 4_360,
    "slide_collapse.csv": 13_080,
    "slide_image_gamut.csv": 2_180,
    "endpoint_summary.csv": 4,
    "content_scanner_summary.csv": 20,
    "collapse_summary.csv": 60,
    "image_gamut_summary.csv": 20,
    "comparison_to_locked_e5.csv": 12,
    "incremental_vs_reinhard.csv": 4,
}
TISSUE_ROWS = {
    "slide_source_probe.csv": 4_320,
    "slide_at2_probe.csv": 432,
    "endpoint_summary.csv": 64,
    "scanner_endpoint_summary.csv": 320,
    "at2_endpoint_summary.csv": 32,
    "tissue_class_sizes.csv": 37,
    "incremental_vs_reinhard.csv": 32,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="docs/e5_reinhard_residual_frequency_contract.md")
    parser.add_argument("--results-doc", default="docs/e5_reinhard_residual_frequency_results.md")
    parser.add_argument("--stability", default="outputs/e5_rf1_input_stability")
    parser.add_argument("--feature-audit", default="outputs/e5_rf1_features/audit/summary.json")
    parser.add_argument("--visual", default="outputs/e5_rf1_visual_audit")
    parser.add_argument("--frontier", default="outputs/e5_rf1_frontier")
    parser.add_argument("--tissue", default="outputs/e5_rf1_tissue_probe")
    parser.add_argument("--core-lock", default="outputs/pannormal_core_results_lock/summary.json")
    parser.add_argument("--output", default="outputs/e5_rf1_results_lock")
    return parser.parse_args()


def audited_table(path: Path, expected_rows: int):
    frame = pd.read_csv(path, dtype={"slide_id": str})
    numeric = frame.select_dtypes(include=[np.number])
    if len(frame) != expected_rows or not np.isfinite(numeric).all().all():
        raise ValueError(f"{path}: row/finiteness audit failed")
    return frame


def main():
    args = parse_args()
    contract_path = Path(args.contract)
    contract_sha = sha256(contract_path)
    if "**Status:** PRE-OUTCOME FROZEN" not in contract_path.read_text():
        raise RuntimeError("E5-RF1 contract is not frozen")
    results_doc_path = Path(args.results_doc)
    if "**Status:** RESULT-LOCKED post-core extension" not in results_doc_path.read_text():
        raise RuntimeError("E5-RF1 result document is not marked result-locked")
    core_lock_path = Path(args.core_lock)
    core_lock = json.loads(core_lock_path.read_text())
    if not (
        core_lock.get("analysis") == "pannormal_core_results_lock"
        and core_lock.get("core_result_lock_pass") is True
    ):
        raise RuntimeError("PanNormal core result lock has not passed")

    stability_root = Path(args.stability)
    stability_path = stability_root / "summary.json"
    stability = json.loads(stability_path.read_text())
    if not (
        stability.get("analysis") == "e5_rf1_input_stability"
        and stability.get("rf1_version") == RF1_VERSION
        and stability.get("outcome_access") is False
        and stability.get("selected_gain_cap") is not None
        and stability.get("execution_contract_sha256") == contract_sha
        and stability.get("stability_gate_pass") is True
    ):
        raise RuntimeError("E5-RF1 input stability lock has not passed")
    candidate_path = stability_root / "gain_cap_candidates.csv"
    candidates = audited_table(candidate_path, 195)
    selected_rows = candidates[
        np.isclose(candidates["gain_cap"], float(stability["selected_gain_cap"]))
    ]
    if len(selected_rows) != 15 or not selected_rows["cell_gate_pass"].astype(bool).all():
        raise ValueError("selected E5-RF1 cap does not pass every FOV/scanner cell")

    visual_root = Path(args.visual)
    visual_path = visual_root / "summary.json"
    visual = json.loads(visual_path.read_text())
    if not (
        visual.get("analysis") == "e5_rf1_visual_audit"
        and visual.get("rf1_version") == RF1_VERSION
        and visual.get("selected_gain_cap") == stability["selected_gain_cap"]
        and visual.get("input_stability_sha256") == sha256(stability_path)
        and visual.get("examples") == 5
        and visual.get("cells") == 15
        and visual.get("visual_gate_pass") is True
    ):
        raise RuntimeError("E5-RF1 visual audit has not passed")
    visual_artifacts = [
        visual_root / "sample_manifest.csv",
        visual_root / "selected_cap_spectrum_cells.csv",
        visual_root / "paired_patch_audit.png",
        visual_root / "paired_patch_audit.pdf",
    ]
    if any(not path.exists() or path.stat().st_size == 0 for path in visual_artifacts):
        raise ValueError("E5-RF1 visual audit artifact is missing or empty")
    for path in visual_artifacts[-2:]:
        if visual["figures"][path.name]["sha256"] != sha256(path):
            raise ValueError(f"{path.name}: visual audit hash mismatch")

    feature_audit_path = Path(args.feature_audit)
    feature_audit = json.loads(feature_audit_path.read_text())
    if not (
        feature_audit.get("analysis") == "e5_rf1_feature_population_audit"
        and feature_audit.get("rf1_version") == RF1_VERSION
        and feature_audit.get("shards_passing") == 436
        and feature_audit.get("features_observed") == 261_600
        and feature_audit.get("stability_manifest_sha256") == sha256(stability_path)
        and feature_audit.get("audit_pass") is True
    ):
        raise RuntimeError("E5-RF1 feature population lock has not passed")

    frontier_root = Path(args.frontier)
    frontier_path = frontier_root / "summary.json"
    frontier = json.loads(frontier_path.read_text())
    if not (
        frontier.get("analysis") == "e5_rf1_frontier"
        and frontier.get("rf1_version") == RF1_VERSION
        and frontier.get("condition") == RF1_CONDITION
        and frontier.get("selected_gain_cap") == stability["selected_gain_cap"]
        and frontier.get("population_audit_sha256") == sha256(feature_audit_path)
        and frontier.get("input_stability_sha256") == sha256(stability_path)
        and frontier.get("execution_contract_sha256") == contract_sha
        and frontier.get("analysis_gate_pass") is True
    ):
        raise RuntimeError("E5-RF1 frontier lock has not passed")
    frontier_frames = {
        filename: audited_table(frontier_root / filename, rows)
        for filename, rows in FRONTIER_ROWS.items()
    }
    endpoints = frontier_frames["endpoint_summary.csv"]
    if set(endpoints["condition"]) != {RF1_CONDITION} or endpoints["encoder_id"].nunique() != 4:
        raise ValueError("E5-RF1 endpoint identity audit failed")

    tissue_root = Path(args.tissue)
    tissue_path = tissue_root / "summary.json"
    tissue = json.loads(tissue_path.read_text())
    if not (
        tissue.get("analysis") == "e5_rf1_grouped_tissue_probe"
        and tissue.get("rf1_version") == RF1_VERSION
        and tissue.get("condition") == RF1_CONDITION
        and tissue.get("rf1_contract_sha256") == contract_sha
        and tissue.get("analysis_gate_pass") is True
    ):
        raise RuntimeError("E5-RF1 tissue-probe lock has not passed")
    tissue_frames = {
        filename: audited_table(tissue_root / filename, rows)
        for filename, rows in TISSUE_ROWS.items()
    }
    for filename, frame in tissue_frames.items():
        if tissue["tables"][filename]["rows"] != len(frame):
            raise ValueError(f"{filename}: tissue summary row mismatch")
        if tissue["tables"][filename]["sha256"] != sha256(tissue_root / filename):
            raise ValueError(f"{filename}: tissue summary hash mismatch")
    tissue_endpoints = tissue_frames["endpoint_summary.csv"]
    raw_tissue = tissue_endpoints[tissue_endpoints["condition"] == "raw"]
    if not np.allclose(
        raw_tissue[["delta_from_raw", "delta_ci95_lower", "delta_ci95_upper"]], 0.0,
        atol=1e-12,
    ):
        raise ValueError("RF1 tissue raw contrast is not identically zero")
    weights_path = tissue_root / "hierarchical_bootstrap_weights.npz"
    if tissue.get("bootstrap_weights_sha256") != sha256(weights_path):
        raise ValueError("RF1 tissue bootstrap hash mismatch")
    with np.load(weights_path) as source:
        if (
            source["full_weights"].shape != (5_000, 108)
            or source["min3_weights"].shape != (5_000, 98)
            or not np.allclose(source["full_weights"].sum(axis=1), 1.0, atol=1e-6)
            or not np.allclose(source["min3_weights"].sum(axis=1), 1.0, atol=1e-6)
        ):
            raise ValueError("RF1 tissue bootstrap population mismatch")

    artifacts = [
        contract_path,
        results_doc_path,
        stability_path,
        candidate_path,
        visual_path,
        *visual_artifacts,
        feature_audit_path,
        frontier_path,
        *(frontier_root / filename for filename in FRONTIER_ROWS),
        tissue_path,
        *(tissue_root / filename for filename in TISSUE_ROWS),
        weights_path,
    ]
    manifest = pd.DataFrame(
        [
            {
                "artifact": str(path.resolve()),
                "bytes": int(path.stat().st_size),
                "sha256": sha256(path),
            }
            for path in artifacts
        ]
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "artifact_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    result = {
        "analysis": "e5_rf1_result_lock",
        "rf1_version": RF1_VERSION,
        "condition": RF1_CONDITION,
        "interpretation": "post-core analytic method-development extension",
        "selected_gain_cap": stability["selected_gain_cap"],
        "pfm_endpoints": endpoints.to_dict("records"),
        "common_safe": frontier["common_safe"],
        "common_safe_and_improved": frontier["common_safe_and_improved"],
        "tissue_interpretation": "coarse secondary tissue-type evidence only",
        "rf1_contract_sha256": contract_sha,
        "results_document_sha256": sha256(results_doc_path),
        "core_result_lock_sha256": sha256(core_lock_path),
        "input_stability_sha256": sha256(stability_path),
        "visual_audit_sha256": sha256(visual_path),
        "feature_population_audit_sha256": sha256(feature_audit_path),
        "frontier_summary_sha256": sha256(frontier_path),
        "tissue_summary_sha256": sha256(tissue_path),
        "artifacts": len(manifest),
        "artifact_bytes": int(manifest["bytes"].sum()),
        "artifact_manifest": str(manifest_path.resolve()),
        "result_lock_pass": True,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
