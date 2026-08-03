"""Audit and lock E7 grouped tissue-probe outputs and Figure 6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_e0_pfm_checkpoints import sha256


EXPECTED_ROWS = {
    "slide_source_probe.csv": 25_920,
    "slide_at2_probe.csv": 432,
    "endpoint_summary.csv": 384,
    "scanner_endpoint_summary.csv": 1_920,
    "loto_loso_transfer_summary.csv": 160,
    "at2_endpoint_summary.csv": 32,
    "tissue_class_sizes.csv": 37,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/e7_tissue_probe")
    parser.add_argument("--contract", default="docs/e7_tissue_probe_execution_contract.md")
    parser.add_argument("--e5-lock", default="outputs/e5_comparator_results_lock/summary.json")
    parser.add_argument("--e6-loto-lock", default="outputs/e6_loto_results_lock/summary.json")
    parser.add_argument("--output", default="outputs/e7_tissue_probe_results_lock")
    return parser.parse_args()


def main():
    args = parse_args()
    results = Path(args.results)
    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text())
    contract_path = Path(args.contract)
    e5_lock_path = Path(args.e5_lock)
    loto_lock_path = Path(args.e6_loto_lock)
    if summary.get("analysis_gate_pass") is not True:
        raise RuntimeError("E7 tissue-probe analysis gate has not passed")
    if summary.get("e7_contract_sha256") != sha256(contract_path):
        raise ValueError("E7 contract hash mismatch")
    if json.loads(e5_lock_path.read_text()).get("result_lock_pass") is not True:
        raise RuntimeError("E5 primary result lock has not passed")
    if json.loads(loto_lock_path.read_text()).get("result_lock_pass") is not True:
        raise RuntimeError("E6 LOTO result lock has not passed")

    frames = {}
    manifest_rows = []
    for filename, expected_rows in EXPECTED_ROWS.items():
        path = results / filename
        frame = pd.read_csv(path, dtype={"slide_id": str})
        frames[filename] = frame
        numeric = frame.select_dtypes(include=[np.number])
        if len(frame) != expected_rows or not np.isfinite(numeric).all().all():
            raise ValueError(f"{filename}: row/finiteness audit failed")
        if summary["tables"][filename]["sha256"] != sha256(path):
            raise ValueError(f"{filename}: summary hash mismatch")
        manifest_rows.append({
            "artifact": str(path),
            "rows": len(frame),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    for filename in (
        "figure6_content_tissue_evidence.png",
        "figure6_content_tissue_evidence.pdf",
    ):
        path = results / filename
        if path.stat().st_size <= 10_000:
            raise ValueError(f"{filename}: implausibly small Figure 6 artifact")
        if summary["figures"][filename]["sha256"] != sha256(path):
            raise ValueError(f"{filename}: summary hash mismatch")
        manifest_rows.append({
            "artifact": str(path),
            "rows": 1,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    bootstrap_path = results / "hierarchical_bootstrap_weights.npz"
    if summary.get("bootstrap_weights_sha256") != sha256(bootstrap_path):
        raise ValueError("E7 bootstrap-weight hash mismatch")
    with np.load(bootstrap_path) as source:
        if (
            source["full_weights"].shape != (5_000, 108)
            or source["min3_weights"].shape != (5_000, 98)
            or not np.allclose(source["full_weights"].sum(axis=1), 1.0, atol=1e-6)
            or not np.allclose(source["min3_weights"].sum(axis=1), 1.0, atol=1e-6)
        ):
            raise ValueError("E7 hierarchical bootstrap population mismatch")
    manifest_rows.append({
        "artifact": str(bootstrap_path),
        "rows": 10_000,
        "bytes": bootstrap_path.stat().st_size,
        "sha256": sha256(bootstrap_path),
    })

    classes = frames["tissue_class_sizes.csv"]
    if (
        classes["slides"].sum() != 109
        or classes["evaluable_probe"].astype(bool).sum() != 36
        or classes["included_min3_sensitivity"].astype(bool).sum() != 31
        or classes.loc[classes["included_min3_sensitivity"].astype(bool), "slides"].sum() != 98
    ):
        raise ValueError("E7 tissue class-size audit failed")
    endpoints = frames["endpoint_summary.csv"]
    raw = endpoints[endpoints["condition"] == "raw"]
    if not np.allclose(raw[["delta_from_raw", "delta_ci95_lower", "delta_ci95_upper"]], 0.0):
        raise ValueError("E7 raw contrast is not identically zero")
    if not np.allclose(
        endpoints["delta_from_raw"],
        endpoints["estimate"] - endpoints["raw_estimate"],
        atol=1e-12,
    ):
        raise ValueError("E7 paired point contrast audit failed")
    source = frames["slide_source_probe.csv"]
    metric_columns = [
        "top1_accuracy",
        "top5_accuracy",
        "correct_tissue_margin",
        "centroid_profile_agreement",
    ]
    raw_loso = source[
        (source["condition"] == "raw")
        & (source["correction_population"] == "loso")
    ].sort_values(["encoder_id", "slide_id", "scanner"])
    raw_loto = source[
        (source["condition"] == "raw")
        & (source["correction_population"] == "loto")
    ].sort_values(["encoder_id", "slide_id", "scanner"])
    if not np.array_equal(raw_loso[metric_columns].to_numpy(), raw_loto[metric_columns].to_numpy()):
        raise ValueError("E7 raw LOSO/LOTO rows differ")

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
    full_top1 = endpoints[
        (endpoints["analysis_set"] == "full_evaluable")
        & (endpoints["metric"] == "top1_accuracy")
    ].sort_values(["correction_population", "encoder_id", "condition"])
    result = {
        "analysis": "e7_grouped_tissue_probe_result_lock",
        "interpretation": "coarse secondary tissue-type evidence; not biological or clinical validation",
        "full_evaluable": {"tissues": 36, "slides": 108},
        "min3_sensitivity": {"tissues": 31, "slides": 98},
        "bootstrap_replicates": 5_000,
        "full_top1_results": full_top1.to_dict("records"),
        "e7_contract_sha256": sha256(contract_path),
        "e5_result_lock_sha256": sha256(e5_lock_path),
        "e6_loto_result_lock_sha256": sha256(loto_lock_path),
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
