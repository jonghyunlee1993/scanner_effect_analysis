"""Score the frozen five-method E5 comparator frontier after population audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from analyze_e4_control_frontier import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COLLAPSE_LOWER_THRESHOLD,
    COLLAPSE_POINT_THRESHOLD,
    CONTENT_NONINFERIORITY_MARGIN,
    RAW_RADIUS_EPSILON,
    bootstrap_indices,
)
from e4_primary_metrics import (
    COLLAPSE_METRICS,
    bootstrap_mean_ci,
    collapse_metric_values,
    l2_normalize,
    scanner_centroid_rms,
    unmatched_q95,
)
from e5_comparator_population import E5_VERSION, FEATURE_CONDITIONS, IMAGE_CONDITIONS, SCANNERS
from fetch_e0_pfm_checkpoints import sha256


CONDITIONS = ("raw", *IMAGE_CONDITIONS, *FEATURE_CONDITIONS)
METHOD_FAMILY = {
    "raw": "raw",
    **{condition: "image" for condition in IMAGE_CONDITIONS},
    **{condition: "feature" for condition in FEATURE_CONDITIONS},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--image", default="outputs/e5_image_features")
    parser.add_argument("--feature", default="outputs/e5_feature_harmonization")
    parser.add_argument("--audit", default="outputs/e5_comparator_features/audit/summary.json")
    parser.add_argument("--decision", default="docs/e4_e7_decision_record.md")
    parser.add_argument("--execution-contract", default="docs/e5_comparator_execution_contract.md")
    parser.add_argument("--output", default="outputs/e5_comparator_frontier")
    return parser.parse_args()


def slide_rows(model: dict, raw_root: Path, image_root: Path, feature_root: Path):
    model_id = model["encoder_id"]
    image_paths = sorted((image_root / model_id / "shards").glob("*.h5"))
    feature_paths = sorted((feature_root / model_id / "shards").glob("*.h5"))
    if len(image_paths) != 109 or [p.name for p in image_paths] != [p.name for p in feature_paths]:
        raise ValueError(f"{model_id}: incomplete/mismatched E5 shard population")
    invariance_rows = []
    content_rows = []
    collapse_rows = []
    clipping_rows = []
    for slide_index, (image_path, feature_path) in enumerate(zip(image_paths, feature_paths)):
        slide_id = image_path.stem
        raw_path = raw_root / model_id / "shards" / image_path.name
        with h5py.File(raw_path, "r") as source:
            raw = np.asarray(source["features"][:], dtype=np.float32)
            raw_locations = source["location_id"][:]
        with h5py.File(image_path, "r") as source:
            image = np.asarray(source["features"][:], dtype=np.float32)
            image_locations = source["location_id"][:]
            image_conditions = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in source["condition"][:]
            ]
            clip_fraction = np.asarray(source["preclip_range_fraction"][:], dtype=np.float32)
            clip_mae = np.asarray(source["clip_pixel_mae"][:], dtype=np.float32)
        with h5py.File(feature_path, "r") as source:
            feature = np.asarray(source["features"][:], dtype=np.float32)
            feature_locations = source["location_id"][:]
            feature_conditions = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in source["condition"][:]
            ]
        dimension = int(model["feature_dim"])
        if raw.shape != (6, 100, dimension) or image.shape != (3, 6, 100, dimension) or feature.shape != (2, 6, 100, dimension):
            raise ValueError(f"{model_id}/{slide_id}: invalid E5 feature shape")
        if image_conditions != list(IMAGE_CONDITIONS) or feature_conditions != list(FEATURE_CONDITIONS):
            raise ValueError(f"{model_id}/{slide_id}: condition order mismatch")
        if not (
            np.array_equal(raw_locations, image_locations)
            and np.array_equal(raw_locations, feature_locations)
        ):
            raise ValueError(f"{model_id}/{slide_id}: location identity mismatch")

        condition_features = {"raw": raw}
        condition_features.update({name: image[index] for index, name in enumerate(IMAGE_CONDITIONS)})
        condition_features.update({name: feature[index] for index, name in enumerate(FEATURE_CONDITIONS)})
        raw_unit = l2_normalize(raw)
        raw_at2 = raw_unit[0]
        q95 = unmatched_q95(raw_at2)
        raw_collapse = {
            scanner_index: collapse_metric_values(raw[scanner_index])
            for scanner_index in range(1, 6)
        }
        for condition in CONDITIONS:
            values = condition_features[condition]
            unit = raw_unit if condition == "raw" else l2_normalize(values)
            invariance_rows.append(
                {
                    "encoder_id": model_id,
                    "slide_id": slide_id,
                    "condition": condition,
                    "method_family": METHOD_FAMILY[condition],
                    "scanner_centroid_rms": scanner_centroid_rms(values),
                }
            )
            margins = np.sum(unit[1:] * raw_at2[None], axis=-1) - q95[None]
            for scanner_index, scanner in enumerate(SCANNERS[1:], start=1):
                content_rows.append(
                    {
                        "encoder_id": model_id,
                        "slide_id": slide_id,
                        "condition": condition,
                        "method_family": METHOD_FAMILY[condition],
                        "scanner": scanner,
                        "content_margin": float(margins[scanner_index - 1].mean()),
                    }
                )
                metrics = raw_collapse[scanner_index] if condition == "raw" else collapse_metric_values(values[scanner_index])
                for metric in COLLAPSE_METRICS:
                    collapse_rows.append(
                        {
                            "encoder_id": model_id,
                            "slide_id": slide_id,
                            "condition": condition,
                            "method_family": METHOD_FAMILY[condition],
                            "scanner": scanner,
                            "metric": metric,
                            "raw_value": raw_collapse[scanner_index][metric],
                            "condition_value": metrics[metric],
                            "ratio": metrics[metric] / raw_collapse[scanner_index][metric],
                        }
                    )
        for condition_index, condition in enumerate(IMAGE_CONDITIONS):
            for scanner_index, scanner in enumerate(SCANNERS[1:], start=1):
                clipping_rows.append(
                    {
                        "encoder_id": model_id,
                        "slide_id": slide_id,
                        "condition": condition,
                        "scanner": scanner,
                        "preclip_range_fraction": float(clip_fraction[condition_index, scanner_index].mean()),
                        "clip_pixel_mae": float(clip_mae[condition_index, scanner_index].mean()),
                    }
                )
        print(f"[{slide_index + 1}/109] {model_id} {slide_id}", flush=True)
    return invariance_rows, content_rows, collapse_rows, clipping_rows


def summarize(invariance: pd.DataFrame, content: pd.DataFrame, collapse: pd.DataFrame):
    indices = bootstrap_indices(109)
    endpoint_rows = []
    content_scanner_rows = []
    collapse_summary_rows = []
    for model_id in sorted(invariance["encoder_id"].unique()):
        model_invariance = invariance[invariance["encoder_id"] == model_id]
        model_content = content[content["encoder_id"] == model_id]
        model_collapse = collapse[collapse["encoder_id"] == model_id]
        raw_radius = model_invariance[model_invariance["condition"] == "raw"].sort_values("slide_id")["scanner_centroid_rms"].to_numpy()
        raw_content = model_content[model_content["condition"] == "raw"].groupby("slide_id", sort=True)["content_margin"].mean().to_numpy()
        for condition in CONDITIONS:
            radius = model_invariance[model_invariance["condition"] == condition].sort_values("slide_id")["scanner_centroid_rms"].to_numpy()
            radius_point, radius_lower, radius_upper = bootstrap_mean_ci(radius, indices)
            difference_point, difference_lower, difference_upper = bootstrap_mean_ci(raw_radius - radius, indices)
            raw_radius_point = float(raw_radius.mean())
            relative_reduction = 1.0 - radius_point / raw_radius_point if raw_radius_point >= RAW_RADIUS_EPSILON else np.nan
            condition_content = model_content[model_content["condition"] == condition]
            pooled = condition_content.groupby("slide_id", sort=True)["content_margin"].mean().to_numpy()
            content_point, content_lower, content_upper = bootstrap_mean_ci(pooled, indices)
            delta_point, delta_lower, delta_upper = bootstrap_mean_ci(pooled - raw_content, indices)
            content_pass = bool(delta_lower >= CONTENT_NONINFERIORITY_MARGIN)
            for scanner in SCANNERS[1:]:
                scanner_values = condition_content[condition_content["scanner"] == scanner].sort_values("slide_id")["content_margin"].to_numpy()
                raw_scanner = model_content[(model_content["condition"] == "raw") & (model_content["scanner"] == scanner)].sort_values("slide_id")["content_margin"].to_numpy()
                point, lower, upper = bootstrap_mean_ci(scanner_values, indices)
                d_point, d_lower, d_upper = bootstrap_mean_ci(scanner_values - raw_scanner, indices)
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
                    ratios = model_collapse[(model_collapse["condition"] == condition) & (model_collapse["scanner"] == scanner) & (model_collapse["metric"] == metric)].sort_values("slide_id")["ratio"].to_numpy()
                    point, lower, upper = bootstrap_mean_ci(ratios, indices)
                    gate = bool(point >= COLLAPSE_POINT_THRESHOLD and lower >= COLLAPSE_LOWER_THRESHOLD)
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
                    "method_family": METHOD_FAMILY[condition],
                    "scanner_centroid_rms": radius_point,
                    "radius_ci_lower": radius_lower,
                    "radius_ci_upper": radius_upper,
                    "raw_minus_condition": difference_point,
                    "difference_ci_lower": difference_lower,
                    "difference_ci_upper": difference_upper,
                    "relative_radius_reduction": relative_reduction,
                    "fidelity_constrained_rr": relative_reduction if safe else np.nan,
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
                    "invariance_improved_but_unsafe": invariance_improved and not safe,
                }
            )
    endpoints = pd.DataFrame(endpoint_rows)
    endpoints["unconstrained_rr_rank"] = endpoints.groupby("encoder_id")["relative_radius_reduction"].rank(method="min", ascending=False).astype(int)
    endpoints["safe_improved_rr_rank"] = endpoints["relative_radius_reduction"].where(endpoints["safe_and_invariance_improved"]).groupby(endpoints["encoder_id"]).rank(method="min", ascending=False)
    common = []
    for condition, frame in endpoints.groupby("condition", sort=False):
        common.append(
            {
                "condition": condition,
                "method_family": METHOD_FAMILY[condition],
                "pfms": len(frame),
                "common_safe_across_four_pfms": bool(frame["safe_for_pfm"].all()),
                "invariance_improved_all_four_pfms": bool(frame["invariance_improved"].all()),
                "safe_and_improved_all_four_pfms": bool(frame["safe_and_invariance_improved"].all()),
            }
        )
    decisions = []
    for model_id, frame in endpoints[endpoints["condition"] != "raw"].groupby("encoder_id"):
        unconstrained = frame.sort_values(["relative_radius_reduction", "condition"], ascending=[False, True]).iloc[0]
        safe = frame[frame["safe_and_invariance_improved"]].sort_values(["relative_radius_reduction", "condition"], ascending=[False, True])
        unsafe_positive = frame[
            frame["invariance_improved"] & ~frame["safe_for_pfm"]
        ].sort_values("condition")
        constrained = None if safe.empty else safe.iloc[0]
        decisions.append(
            {
                "encoder_id": model_id,
                "unconstrained_best": unconstrained["condition"],
                "unconstrained_best_rr": unconstrained["relative_radius_reduction"],
                "unconstrained_best_safe": bool(unconstrained["safe_for_pfm"]),
                "fidelity_constrained_best": None if constrained is None else constrained["condition"],
                "fidelity_constrained_best_rr": np.nan if constrained is None else constrained["relative_radius_reduction"],
                "decision_differs": bool(constrained is None or unconstrained["condition"] != constrained["condition"]),
                "invariance_improved_but_unsafe_count": len(unsafe_positive),
                "invariance_improved_but_unsafe_methods": "|".join(
                    unsafe_positive["condition"].tolist()
                ),
            }
        )
    return endpoints, pd.DataFrame(content_scanner_rows), pd.DataFrame(collapse_summary_rows), pd.DataFrame(common), pd.DataFrame(decisions)


def summarize_clipping(frame: pd.DataFrame):
    rows = []
    indices = bootstrap_indices(109)
    for keys, selected in frame.groupby(["encoder_id", "condition", "scanner"], sort=True):
        ordered = selected.sort_values("slide_id")
        clip = bootstrap_mean_ci(ordered["preclip_range_fraction"].to_numpy(), indices)
        mae = bootstrap_mean_ci(ordered["clip_pixel_mae"].to_numpy(), indices)
        rows.append(
            {
                "encoder_id": keys[0],
                "condition": keys[1],
                "scanner": keys[2],
                "preclip_range_fraction": clip[0],
                "preclip_ci_lower": clip[1],
                "preclip_ci_upper": clip[2],
                "clip_pixel_mae": mae[0],
                "clip_mae_ci_lower": mae[1],
                "clip_mae_ci_upper": mae[2],
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    audit_path = Path(args.audit)
    audit = json.loads(audit_path.read_text())
    if not (
        audit.get("audit_pass") is True
        and audit.get("total_features_observed") == 1_308_000
    ):
        raise RuntimeError("E5 comparator population audit has not passed")
    decision_path = Path(args.decision)
    if "**Status:** FROZEN" not in decision_path.read_text():
        raise RuntimeError("E4--E7 decision record is not frozen")
    execution_path = Path(args.execution_contract)
    if "**Status:** PRE-OUTCOME FROZEN" not in execution_path.read_text():
        raise RuntimeError("E5 execution contract is not frozen")
    contract = json.loads(Path(args.contract).read_text())
    invariance_rows = []
    content_rows = []
    collapse_rows = []
    clipping_rows = []
    for model in contract["models"]:
        local = slide_rows(model, Path(args.raw), Path(args.image), Path(args.feature))
        invariance_rows.extend(local[0])
        content_rows.extend(local[1])
        collapse_rows.extend(local[2])
        clipping_rows.extend(local[3])
    invariance = pd.DataFrame(invariance_rows)
    content = pd.DataFrame(content_rows)
    collapse = pd.DataFrame(collapse_rows)
    clipping = pd.DataFrame(clipping_rows)
    endpoints, content_scanner, collapse_summary, common, decisions = summarize(invariance, content, collapse)
    clipping_summary = summarize_clipping(clipping)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "slide_invariance.csv": invariance,
        "slide_content_margin.csv": content,
        "slide_collapse.csv": collapse,
        "slide_image_clipping.csv": clipping,
        "endpoint_summary.csv": endpoints,
        "content_scanner_summary.csv": content_scanner,
        "collapse_summary.csv": collapse_summary,
        "common_pfm_summary.csv": common,
        "decision_comparison.csv": decisions,
        "image_clipping_summary.csv": clipping_summary,
    }
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False)
    summary = {
        "analysis": "e5_comparator_frontier",
        "e5_version": E5_VERSION,
        "pfms": 4,
        "slides": 109,
        "conditions_including_raw": len(CONDITIONS),
        "image_conditions": list(IMAGE_CONDITIONS),
        "feature_conditions": list(FEATURE_CONDITIONS),
        "bootstrap_unit": "physical_slide",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "raw_radius_epsilon": RAW_RADIUS_EPSILON,
        "content_noninferiority_margin": CONTENT_NONINFERIORITY_MARGIN,
        "collapse_point_threshold": COLLAPSE_POINT_THRESHOLD,
        "collapse_lower_ci_threshold": COLLAPSE_LOWER_THRESHOLD,
        "slide_invariance_rows": len(invariance),
        "slide_content_rows": len(content),
        "slide_collapse_rows": len(collapse),
        "endpoint_rows": len(endpoints),
        "common_safe_methods": common[common["common_safe_across_four_pfms"]]["condition"].tolist(),
        "common_safe_and_improved_methods": common[common["safe_and_improved_all_four_pfms"]]["condition"].tolist(),
        "decision_differs_pfms": int(decisions["decision_differs"].sum()),
        "invariance_improved_but_unsafe_cells": int(
            endpoints["invariance_improved_but_unsafe"].sum()
        ),
        "invariance_improved_but_unsafe_method_pfm": endpoints[
            endpoints["invariance_improved_but_unsafe"]
        ][["encoder_id", "condition"]].to_dict("records"),
        "population_audit": str(audit_path.resolve()),
        "population_audit_sha256": sha256(audit_path),
        "decision_record_sha256": sha256(decision_path),
        "execution_contract_sha256": sha256(execution_path),
        "analysis_gate_pass": bool(len(endpoints) == 24 and len(invariance) == 2_616),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["analysis_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
