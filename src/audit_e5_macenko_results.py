"""Lock the completed Supplement-only Macenko population results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_e0_pfm_checkpoints import sha256


EXPECTED_ROWS = {
    "slide_invariance.csv": 872,
    "slide_content_margin.csv": 4_360,
    "slide_collapse.csv": 13_080,
    "slide_clipping_fallback.csv": 2_180,
    "endpoint_summary.csv": 4,
    "content_scanner_summary.csv": 20,
    "collapse_summary.csv": 60,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/e5_macenko_supplement")
    parser.add_argument(
        "--population-audit",
        default="outputs/e5_macenko_features/audit/summary.json",
    )
    parser.add_argument(
        "--primary-lock", default="outputs/e5_comparator_results_lock/summary.json"
    )
    parser.add_argument(
        "--contract", default="docs/e5_macenko_supplement_contract.md"
    )
    parser.add_argument("--output", default="outputs/e5_macenko_supplement_lock")
    return parser.parse_args()


def main():
    args = parse_args()
    results = Path(args.results)
    population_audit_path = Path(args.population_audit)
    primary_lock_path = Path(args.primary_lock)
    contract_path = Path(args.contract)
    analysis_summary_path = results / "summary.json"
    analysis_summary = json.loads(analysis_summary_path.read_text())
    population_audit = json.loads(population_audit_path.read_text())
    primary_lock = json.loads(primary_lock_path.read_text())
    if analysis_summary.get("analysis_gate_pass") is not True:
        raise RuntimeError("Macenko frontier gate has not passed")
    if population_audit.get("audit_pass") is not True:
        raise RuntimeError("Macenko population audit has not passed")
    if primary_lock.get("result_lock_pass") is not True:
        raise RuntimeError("E5 primary result lock has not passed")
    if analysis_summary.get("population_audit_sha256") != sha256(population_audit_path):
        raise ValueError("Macenko population-audit hash mismatch")
    if analysis_summary.get("supplement_contract_sha256") != sha256(contract_path):
        raise ValueError("Macenko supplement-contract hash mismatch")

    manifest_rows = []
    frames = {}
    for filename, expected_rows in EXPECTED_ROWS.items():
        path = results / filename
        frame = pd.read_csv(path, dtype={"slide_id": str})
        frames[filename] = frame
        numeric = frame.select_dtypes(include=[np.number])
        finite = bool(np.isfinite(numeric).all().all())
        if len(frame) != expected_rows or not finite:
            raise ValueError(
                f"{filename}: expected {expected_rows} finite rows, got {len(frame)}"
            )
        manifest_rows.append({
            "artifact": str(path),
            "rows": len(frame),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    endpoints = frames["endpoint_summary.csv"]
    if (
        endpoints["encoder_id"].nunique() != 4
        or set(endpoints["condition"]) != {"macenko_supplement"}
        or int(endpoints["safe_for_pfm"].sum()) != 1
        or int(endpoints["safe_and_invariance_improved"].sum()) != 1
    ):
        raise ValueError("Macenko endpoint decision population mismatch")
    collapse = frames["collapse_summary.csv"]
    if not collapse["gate_pass"].astype(bool).all():
        raise ValueError("Macenko collapse summary disagrees with locked endpoint grid")
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
    model_rows = endpoints.sort_values("encoder_id").to_dict("records")
    summary = {
        "analysis": "e5_macenko_supplement_result_lock",
        "placement": "Supplement-only; excluded from primary E5 ranking and claims",
        "models": 4,
        "slides": 109,
        "population_features": population_audit["features_observed"],
        "fallback_patches": population_audit["fallback_patches"],
        "safe_pfms": int(endpoints["safe_for_pfm"].sum()),
        "safe_and_improved_pfms": int(
            endpoints["safe_and_invariance_improved"].sum()
        ),
        "common_safe_across_four_pfms": bool(endpoints["safe_for_pfm"].all()),
        "safe_and_improved_all_four_pfms": bool(
            endpoints["safe_and_invariance_improved"].all()
        ),
        "endpoint_results": model_rows,
        "primary_e5_result_lock_sha256": sha256(primary_lock_path),
        "supplement_contract_sha256": sha256(contract_path),
        "analysis_summary_sha256": sha256(analysis_summary_path),
        "population_audit_sha256": sha256(population_audit_path),
        "artifacts": len(manifest),
        "artifact_bytes": int(manifest["bytes"].sum()),
        "artifact_manifest": str(manifest_path.resolve()),
        "result_lock_pass": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
