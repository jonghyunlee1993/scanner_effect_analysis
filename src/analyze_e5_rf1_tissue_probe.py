"""Summarize grouped tissue-type secondary endpoints for E5-RF1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_e7_tissue_probe import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    METRICS,
    bootstrap_weights,
    summarize_vector,
)
from e5_comparator_population import SCANNERS
from e5_reinhard_residual_frequency import RF1_CONDITION, RF1_VERSION
from e6_loto_population import load_tissue_annotation
from fetch_e0_pfm_checkpoints import sha256


CONDITIONS = ("raw", RF1_CONDITION)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--rows", default="outputs/e5_rf1_tissue_probe_rows")
    parser.add_argument("--geometry", default="outputs/e0_native_geometry_final/native_geometry_manifest.csv")
    parser.add_argument("--rf1-contract", default="docs/e5_reinhard_residual_frequency_contract.md")
    parser.add_argument("--e7-contract", default="docs/e7_tissue_probe_execution_contract.md")
    parser.add_argument("--locked-e7", default="outputs/e7_tissue_probe/slide_source_probe.csv")
    parser.add_argument("--output", default="outputs/e5_rf1_tissue_probe")
    return parser.parse_args()


def paired_summary(condition_frame: pd.DataFrame, raw_frame: pd.DataFrame, metric: str, plan):
    point, lower, upper, _ = summarize_vector(condition_frame, metric, plan)
    raw_point, _, _, _ = summarize_vector(raw_frame, metric, plan)
    paired = condition_frame[["slide_id", "tissue_type", metric]].merge(
        raw_frame[["slide_id", metric]],
        on="slide_id",
        suffixes=("_condition", "_raw"),
        validate="one_to_one",
    )
    paired[metric] = paired[f"{metric}_condition"] - paired[f"{metric}_raw"]
    delta, delta_lower, delta_upper, _ = summarize_vector(paired, metric, plan)
    return point, lower, upper, raw_point, delta, delta_lower, delta_upper


def summarize_source(source: pd.DataFrame, plans: dict):
    endpoints = []
    scanners = []
    for analysis_set, plan in plans.items():
        for model_id in sorted(source["encoder_id"].unique()):
            model = source[source["encoder_id"] == model_id]
            pooled = (
                model.groupby(
                    ["condition", "slide_id", "tissue_type", "slides_in_tissue"],
                    as_index=False,
                )[list(METRICS)]
                .mean()
            )
            raw_pooled = pooled[pooled["condition"] == "raw"]
            for condition in CONDITIONS:
                condition_pooled = pooled[pooled["condition"] == condition]
                for metric in METRICS:
                    values = paired_summary(condition_pooled, raw_pooled, metric, plan)
                    endpoints.append(
                        {
                            "analysis_set": analysis_set,
                            "encoder_id": model_id,
                            "condition": condition,
                            "metric": metric,
                            "tissues": len(plan[1]),
                            "slides": len(plan[2]),
                            "estimate": values[0],
                            "ci95_lower": values[1],
                            "ci95_upper": values[2],
                            "raw_estimate": values[3],
                            "delta_from_raw": values[4],
                            "delta_ci95_lower": values[5],
                            "delta_ci95_upper": values[6],
                        }
                    )
            for scanner in SCANNERS[1:]:
                scanner_frame = model[model["scanner"] == scanner]
                raw_scanner = scanner_frame[scanner_frame["condition"] == "raw"]
                for condition in CONDITIONS:
                    condition_frame = scanner_frame[scanner_frame["condition"] == condition]
                    for metric in METRICS:
                        values = paired_summary(condition_frame, raw_scanner, metric, plan)
                        scanners.append(
                            {
                                "analysis_set": analysis_set,
                                "encoder_id": model_id,
                                "condition": condition,
                                "scanner": scanner,
                                "metric": metric,
                                "tissues": len(plan[1]),
                                "slides": len(plan[2]),
                                "estimate": values[0],
                                "ci95_lower": values[1],
                                "ci95_upper": values[2],
                                "raw_estimate": values[3],
                                "delta_from_raw": values[4],
                                "delta_ci95_lower": values[5],
                                "delta_ci95_upper": values[6],
                            }
                        )
    return pd.DataFrame(endpoints), pd.DataFrame(scanners)


def summarize_at2(at2: pd.DataFrame, plans: dict):
    rows = []
    for analysis_set, plan in plans.items():
        for model_id in sorted(at2["encoder_id"].unique()):
            frame = at2[at2["encoder_id"] == model_id]
            for metric in METRICS:
                point, lower, upper, _ = summarize_vector(frame, metric, plan)
                rows.append(
                    {
                        "analysis_set": analysis_set,
                        "encoder_id": model_id,
                        "metric": metric,
                        "tissues": len(plan[1]),
                        "slides": len(plan[2]),
                        "estimate": point,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                    }
                )
    return pd.DataFrame(rows)


def incremental_vs_reinhard(source: pd.DataFrame, locked_path: Path, plans: dict):
    locked = pd.read_csv(locked_path, dtype={"slide_id": str})
    locked = locked[
        (locked["correction_population"] == "loso")
        & (locked["condition"] == "reinhard_lab")
    ]
    rows = []
    for analysis_set, plan in plans.items():
        for model_id in sorted(source["encoder_id"].unique()):
            rf1_model = source[
                (source["encoder_id"] == model_id)
                & (source["condition"] == RF1_CONDITION)
            ]
            reinhard_model = locked[locked["encoder_id"] == model_id]
            for metric in METRICS:
                rf1 = (
                    rf1_model.groupby(["slide_id", "tissue_type"], as_index=False)[metric]
                    .mean()
                    .rename(columns={metric: "rf1"})
                )
                reinhard = (
                    reinhard_model.groupby(["slide_id", "tissue_type"], as_index=False)[metric]
                    .mean()
                    .rename(columns={metric: "reinhard"})
                )
                paired = rf1.merge(
                    reinhard,
                    on=["slide_id", "tissue_type"],
                    validate="one_to_one",
                )
                paired[metric] = paired["rf1"] - paired["reinhard"]
                point, lower, upper, _ = summarize_vector(paired, metric, plan)
                rows.append(
                    {
                        "analysis_set": analysis_set,
                        "encoder_id": model_id,
                        "metric": metric,
                        "tissues": len(plan[1]),
                        "slides": len(plan[2]),
                        "rf1_minus_reinhard": point,
                        "difference_ci95_lower": lower,
                        "difference_ci95_upper": upper,
                    }
                )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text())
    rf1_contract_path = Path(args.rf1_contract)
    e7_contract_path = Path(args.e7_contract)
    source_frames = []
    at2_frames = []
    input_hashes = {}
    for model in contract["models"]:
        model_id = model["encoder_id"]
        root = Path(args.rows) / model_id
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text())
        source_path = root / "slide_source_probe.csv"
        at2_path = root / "slide_at2_probe.csv"
        if not (
            summary.get("analysis") == "e5_rf1_grouped_tissue_probe_rows"
            and summary.get("rf1_version") == RF1_VERSION
            and summary.get("source_rows") == 1_080
            and summary.get("at2_rows") == 108
            and summary.get("rf1_contract_sha256") == sha256(rf1_contract_path)
            and summary.get("e7_contract_sha256") == sha256(e7_contract_path)
            and summary.get("source_sha256") == sha256(source_path)
            and summary.get("at2_sha256") == sha256(at2_path)
            and summary.get("row_gate_pass") is True
        ):
            raise RuntimeError(f"{model_id}: RF1 tissue row gate has not passed")
        source_frames.append(pd.read_csv(source_path, dtype={"slide_id": str}))
        at2_frames.append(pd.read_csv(at2_path, dtype={"slide_id": str}))
        input_hashes[model_id] = sha256(summary_path)

    source = pd.concat(source_frames, ignore_index=True)
    at2 = pd.concat(at2_frames, ignore_index=True)
    tissue_by_slide, tissues = load_tissue_annotation(Path(args.geometry))
    sizes = pd.DataFrame(
        {
            "tissue_type": tissues,
            "slides": [sum(value == tissue for value in tissue_by_slide.values()) for tissue in tissues],
        }
    )
    sizes["evaluable_probe"] = sizes["slides"] >= 2
    sizes["included_min3_sensitivity"] = sizes["slides"] >= 3
    base = source[
        (source["encoder_id"] == contract["models"][0]["encoder_id"])
        & (source["condition"] == "raw")
        & (source["scanner"] == SCANNERS[1])
    ][["slide_id", "tissue_type", "slides_in_tissue"]]
    plans = {
        "full_evaluable": bootstrap_weights(base, 2, BOOTSTRAP_SEED),
        "min3_sensitivity": bootstrap_weights(base, 3, BOOTSTRAP_SEED + 1),
    }
    endpoints, scanner_endpoints = summarize_source(source, plans)
    at2_endpoints = summarize_at2(at2, plans)
    incremental = incremental_vs_reinhard(source, Path(args.locked_e7), plans)

    if (
        len(source) != 4_320
        or len(at2) != 432
        or len(endpoints) != 64
        or len(scanner_endpoints) != 320
        or len(at2_endpoints) != 32
        or len(incremental) != 32
    ):
        raise RuntimeError("RF1 tissue-probe analysis population mismatch")
    numeric = pd.concat(
        [
            source.select_dtypes(include=[np.number]).stack(),
            endpoints.select_dtypes(include=[np.number]).stack(),
            scanner_endpoints.select_dtypes(include=[np.number]).stack(),
            at2_endpoints.select_dtypes(include=[np.number]).stack(),
            incremental.select_dtypes(include=[np.number]).stack(),
        ],
        ignore_index=True,
    )
    if not np.isfinite(numeric).all():
        raise RuntimeError("RF1 tissue-probe analysis contains non-finite values")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "slide_source_probe.csv": source,
        "slide_at2_probe.csv": at2,
        "endpoint_summary.csv": endpoints,
        "scanner_endpoint_summary.csv": scanner_endpoints,
        "at2_endpoint_summary.csv": at2_endpoints,
        "tissue_class_sizes.csv": sizes,
        "incremental_vs_reinhard.csv": incremental,
    }
    table_manifest = {}
    for filename, frame in tables.items():
        path = output / filename
        frame.to_csv(path, index=False)
        table_manifest[filename] = {"rows": len(frame), "sha256": sha256(path)}
    weights_path = output / "hierarchical_bootstrap_weights.npz"
    np.savez_compressed(
        weights_path,
        full_slide_ids=np.asarray(plans["full_evaluable"][2]),
        full_weights=plans["full_evaluable"][3],
        min3_slide_ids=np.asarray(plans["min3_sensitivity"][2]),
        min3_weights=plans["min3_sensitivity"][3],
    )
    summary = {
        "analysis": "e5_rf1_grouped_tissue_probe",
        "rf1_version": RF1_VERSION,
        "condition": RF1_CONDITION,
        "models": [model["encoder_id"] for model in contract["models"]],
        "full_evaluable": {"tissues": 36, "slides": 108},
        "min3_sensitivity": {"tissues": 31, "slides": 98},
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed_full": BOOTSTRAP_SEED,
        "bootstrap_seed_min3": BOOTSTRAP_SEED + 1,
        "tables": table_manifest,
        "bootstrap_weights_sha256": sha256(weights_path),
        "row_summary_sha256": input_hashes,
        "rf1_contract_sha256": sha256(rf1_contract_path),
        "e7_contract_sha256": sha256(e7_contract_path),
        "locked_e7_source_sha256": sha256(Path(args.locked_e7)),
        "analysis_gate_pass": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(
        endpoints[
            (endpoints["analysis_set"] == "full_evaluable")
            & (endpoints["condition"] == RF1_CONDITION)
            & (endpoints["metric"] == "top1_accuracy")
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
