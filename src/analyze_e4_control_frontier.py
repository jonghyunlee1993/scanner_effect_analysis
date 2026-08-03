"""Compute the frozen E4 four-PFM control frontier after population audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from e4_control_population import CONTROL_VERSION, SCANNERS, condition_names
from e4_primary_metrics import (
    COLLAPSE_METRICS,
    bootstrap_mean_ci,
    collapse_metric_values,
    l2_normalize,
    scanner_centroid_rms,
    unmatched_q95,
)
from fetch_e0_pfm_checkpoints import sha256


BOOTSTRAP_SEED = 20260803
BOOTSTRAP_REPLICATES = 5000
RAW_RADIUS_EPSILON = 0.01
CONTENT_NONINFERIORITY_MARGIN = -0.02
COLLAPSE_POINT_THRESHOLD = 0.90
COLLAPSE_LOWER_THRESHOLD = 0.85


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json"
    )
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--controls", default="outputs/e4_control_features")
    parser.add_argument(
        "--control-audit", default="outputs/e4_control_features/audit/summary.json"
    )
    parser.add_argument("--decision", default="docs/e4_e7_decision_record.md")
    parser.add_argument("--output", default="outputs/e4_control_frontier")
    return parser.parse_args()


def bootstrap_indices(slides: int) -> np.ndarray:
    if slides != 109:
        raise ValueError(f"frozen E4 bootstrap requires 109 slides, got {slides}")
    return np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, slides, size=(BOOTSTRAP_REPLICATES, slides)
    )


def slide_rows(model: dict, raw_root: Path, control_root: Path):
    model_id = model["encoder_id"]
    paths = sorted((control_root / model_id / "shards").glob("*.h5"))
    if len(paths) != 109:
        raise ValueError(f"{model_id}: expected 109 audited control shards, got {len(paths)}")
    invariance_rows = []
    content_rows = []
    collapse_rows = []
    conditions = ("raw", *condition_names())
    for slide_index, control_path in enumerate(paths):
        slide_id = control_path.stem
        raw_path = raw_root / model_id / "shards" / control_path.name
        with h5py.File(raw_path, "r") as raw_source:
            raw = np.asarray(raw_source["features"][:], dtype=np.float32)
            raw_locations = raw_source["location_id"][:]
        with h5py.File(control_path, "r") as control_source:
            controls = np.asarray(control_source["features"][:], dtype=np.float32)
            control_locations = control_source["location_id"][:]
            stored_conditions = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in control_source["condition"][:]
            ]
        if raw.shape != (6, 100, int(model["feature_dim"])):
            raise ValueError(f"{model_id}/{slide_id}: invalid raw feature shape")
        if controls.shape != (9, 6, 100, int(model["feature_dim"])):
            raise ValueError(f"{model_id}/{slide_id}: invalid control feature shape")
        if stored_conditions != list(condition_names()) or not np.array_equal(
            raw_locations, control_locations
        ):
            raise ValueError(f"{model_id}/{slide_id}: identity mismatch")

        raw_unit = l2_normalize(raw)
        raw_at2 = raw_unit[0]
        q95 = unmatched_q95(raw_at2)
        raw_collapse = {
            scanner_index: collapse_metric_values(raw[scanner_index])
            for scanner_index in range(1, 6)
        }
        for condition_index, condition in enumerate(conditions):
            features = raw if condition == "raw" else controls[condition_index - 1]
            unit = raw_unit if condition == "raw" else l2_normalize(features)
            invariance_rows.append(
                {
                    "encoder_id": model_id,
                    "slide_id": slide_id,
                    "condition": condition,
                    "scanner_centroid_rms": scanner_centroid_rms(features),
                }
            )
            margins = np.sum(unit[1:] * raw_at2[None], axis=-1) - q95[None]
            for scanner_index, scanner in enumerate(SCANNERS[1:], start=1):
                content_rows.append(
                    {
                        "encoder_id": model_id,
                        "slide_id": slide_id,
                        "condition": condition,
                        "scanner": scanner,
                        "content_margin": float(margins[scanner_index - 1].mean()),
                    }
                )
                values = (
                    raw_collapse[scanner_index]
                    if condition == "raw"
                    else collapse_metric_values(features[scanner_index])
                )
                for metric in COLLAPSE_METRICS:
                    raw_value = raw_collapse[scanner_index][metric]
                    if raw_value <= 0:
                        raise ValueError(
                            f"{model_id}/{slide_id}/{scanner}: nonpositive raw {metric}"
                        )
                    collapse_rows.append(
                        {
                            "encoder_id": model_id,
                            "slide_id": slide_id,
                            "condition": condition,
                            "scanner": scanner,
                            "metric": metric,
                            "raw_value": raw_value,
                            "condition_value": values[metric],
                            "ratio": values[metric] / raw_value,
                        }
                    )
        print(f"[{slide_index + 1}/109] {model_id} {slide_id}", flush=True)
    return invariance_rows, content_rows, collapse_rows


def summarize(invariance: pd.DataFrame, content: pd.DataFrame, collapse: pd.DataFrame):
    indices = bootstrap_indices(109)
    endpoint_rows = []
    content_scanner_rows = []
    collapse_summary_rows = []
    for model_id in sorted(invariance["encoder_id"].unique()):
        model_invariance = invariance[invariance["encoder_id"] == model_id]
        model_content = content[content["encoder_id"] == model_id]
        model_collapse = collapse[collapse["encoder_id"] == model_id]
        raw_radius = (
            model_invariance[model_invariance["condition"] == "raw"]
            .sort_values("slide_id")["scanner_centroid_rms"]
            .to_numpy()
        )
        raw_content = (
            model_content[model_content["condition"] == "raw"]
            .groupby("slide_id", sort=True)["content_margin"]
            .mean()
            .to_numpy()
        )
        for condition in ("raw", *condition_names()):
            radius = (
                model_invariance[model_invariance["condition"] == condition]
                .sort_values("slide_id")["scanner_centroid_rms"]
                .to_numpy()
            )
            radius_point, radius_lower, radius_upper = bootstrap_mean_ci(radius, indices)
            difference_point, difference_lower, difference_upper = bootstrap_mean_ci(
                raw_radius - radius, indices
            )
            raw_radius_point = float(raw_radius.mean())
            relative_reduction = (
                1.0 - radius_point / raw_radius_point
                if raw_radius_point >= RAW_RADIUS_EPSILON
                else np.nan
            )

            condition_content_frame = model_content[
                model_content["condition"] == condition
            ]
            pooled_content = (
                condition_content_frame.groupby("slide_id", sort=True)["content_margin"]
                .mean()
                .to_numpy()
            )
            content_point, content_lower, content_upper = bootstrap_mean_ci(
                pooled_content, indices
            )
            delta_point, delta_lower, delta_upper = bootstrap_mean_ci(
                pooled_content - raw_content, indices
            )
            content_pass = bool(delta_lower >= CONTENT_NONINFERIORITY_MARGIN)

            for scanner in SCANNERS[1:]:
                scanner_values = (
                    condition_content_frame[condition_content_frame["scanner"] == scanner]
                    .sort_values("slide_id")["content_margin"]
                    .to_numpy()
                )
                raw_scanner_values = (
                    model_content[
                        (model_content["condition"] == "raw")
                        & (model_content["scanner"] == scanner)
                    ]
                    .sort_values("slide_id")["content_margin"]
                    .to_numpy()
                )
                point, lower, upper = bootstrap_mean_ci(scanner_values, indices)
                d_point, d_lower, d_upper = bootstrap_mean_ci(
                    scanner_values - raw_scanner_values, indices
                )
                content_scanner_rows.append(
                    {
                        "encoder_id": model_id,
                        "condition": condition,
                        "scanner": scanner,
                        "content_margin": point,
                        "content_margin_ci_lower": lower,
                        "content_margin_ci_upper": upper,
                        "delta_content_margin": d_point,
                        "delta_ci_lower": d_lower,
                        "delta_ci_upper": d_upper,
                    }
                )

            collapse_passes = []
            for scanner in SCANNERS[1:]:
                for metric in COLLAPSE_METRICS:
                    ratios = (
                        model_collapse[
                            (model_collapse["condition"] == condition)
                            & (model_collapse["scanner"] == scanner)
                            & (model_collapse["metric"] == metric)
                        ]
                        .sort_values("slide_id")["ratio"]
                        .to_numpy()
                    )
                    point, lower, upper = bootstrap_mean_ci(ratios, indices)
                    gate = bool(
                        point >= COLLAPSE_POINT_THRESHOLD
                        and lower >= COLLAPSE_LOWER_THRESHOLD
                    )
                    collapse_passes.append(gate)
                    collapse_summary_rows.append(
                        {
                            "encoder_id": model_id,
                            "condition": condition,
                            "scanner": scanner,
                            "metric": metric,
                            "ratio": point,
                            "ratio_ci_lower": lower,
                            "ratio_ci_upper": upper,
                            "gate_pass": gate,
                        }
                    )
            collapse_pass = bool(all(collapse_passes))
            invariance_improved = bool(difference_lower > 0)
            safe = bool(content_pass and collapse_pass)
            endpoint_rows.append(
                {
                    "encoder_id": model_id,
                    "condition": condition,
                    "scanner_centroid_rms": radius_point,
                    "radius_ci_lower": radius_lower,
                    "radius_ci_upper": radius_upper,
                    "raw_minus_condition": difference_point,
                    "difference_ci_lower": difference_lower,
                    "difference_ci_upper": difference_upper,
                    "relative_radius_reduction": relative_reduction,
                    "raw_radius_epsilon_pass": raw_radius_point >= RAW_RADIUS_EPSILON,
                    "invariance_improved": invariance_improved,
                    "content_margin": content_point,
                    "content_margin_ci_lower": content_lower,
                    "content_margin_ci_upper": content_upper,
                    "delta_content_margin": delta_point,
                    "delta_content_ci_lower": delta_lower,
                    "delta_content_ci_upper": delta_upper,
                    "content_noninferiority_pass": content_pass,
                    "collapse_every_scanner_pass": collapse_pass,
                    "safe_for_pfm": safe,
                    "safe_and_invariance_improved": safe and invariance_improved,
                }
            )
    endpoints = pd.DataFrame(endpoint_rows)
    common_rows = []
    for condition, frame in endpoints.groupby("condition", sort=False):
        common_rows.append(
            {
                "condition": condition,
                "pfms": len(frame),
                "common_safe_across_four_pfms": bool(frame["safe_for_pfm"].all()),
                "invariance_improved_all_four_pfms": bool(frame["invariance_improved"].all()),
                "safe_and_improved_all_four_pfms": bool(
                    frame["safe_and_invariance_improved"].all()
                ),
            }
        )
    return (
        endpoints,
        pd.DataFrame(content_scanner_rows),
        pd.DataFrame(collapse_summary_rows),
        pd.DataFrame(common_rows),
    )


def main():
    args = parse_args()
    audit_path = Path(args.control_audit)
    audit = json.loads(audit_path.read_text())
    if audit.get("audit_pass") is not True or audit.get("features_observed") != 2_354_400:
        raise RuntimeError("E4 control population audit has not passed")
    decision_path = Path(args.decision)
    decision_text = decision_path.read_text()
    if "**Status:** FROZEN" not in decision_text or "Approval text: `전부 승인`" not in decision_text:
        raise RuntimeError("E4--E7 decision record is not frozen")

    contract = json.loads(Path(args.contract).read_text())
    invariance_rows = []
    content_rows = []
    collapse_rows = []
    for model in contract["models"]:
        local = slide_rows(model, Path(args.raw), Path(args.controls))
        invariance_rows.extend(local[0])
        content_rows.extend(local[1])
        collapse_rows.extend(local[2])

    invariance = pd.DataFrame(invariance_rows)
    content = pd.DataFrame(content_rows)
    collapse = pd.DataFrame(collapse_rows)
    endpoints, content_scanner, collapse_summary, common = summarize(
        invariance, content, collapse
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    invariance.to_csv(output / "slide_invariance.csv", index=False)
    content.to_csv(output / "slide_content_margin.csv", index=False)
    collapse.to_csv(output / "slide_collapse.csv", index=False)
    endpoints.to_csv(output / "endpoint_summary.csv", index=False)
    content_scanner.to_csv(output / "content_scanner_summary.csv", index=False)
    collapse_summary.to_csv(output / "collapse_summary.csv", index=False)
    common.to_csv(output / "common_pfm_summary.csv", index=False)
    summary = {
        "analysis": "e4_control_frontier",
        "control_version": CONTROL_VERSION,
        "decision_record": str(decision_path.resolve()),
        "decision_record_sha256": sha256(decision_path),
        "control_audit": str(audit_path.resolve()),
        "control_audit_sha256": sha256(audit_path),
        "pfms": len(contract["models"]),
        "slides": 109,
        "conditions_including_raw": 10,
        "bootstrap_unit": "physical_slide",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "raw_radius_epsilon": RAW_RADIUS_EPSILON,
        "content_noninferiority_margin": CONTENT_NONINFERIORITY_MARGIN,
        "collapse_point_threshold": COLLAPSE_POINT_THRESHOLD,
        "collapse_lower_ci_threshold": COLLAPSE_LOWER_THRESHOLD,
        "collapse_hard_gate": "every_non_AT2_source_scanner_and_all_three_metrics",
        "slide_invariance_rows": len(invariance),
        "slide_content_rows": len(content),
        "slide_collapse_rows": len(collapse),
        "endpoint_rows": len(endpoints),
        "analysis_complete": bool(
            len(invariance) == 4 * 10 * 109
            and len(content) == 4 * 10 * 109 * 5
            and len(collapse) == 4 * 10 * 109 * 5 * 3
            and len(endpoints) == 4 * 10
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["analysis_complete"]:
        raise RuntimeError("E4 control frontier is incomplete")


if __name__ == "__main__":
    main()

