"""Score the post-core E5-RF1 condition with the frozen E5 fidelity gates."""

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
from e5_comparator_population import SCANNERS
from e5_reinhard_residual_frequency import RF1_CONDITION, RF1_VERSION
from extract_e5_rf1_features import METRICS
from fetch_e0_pfm_checkpoints import sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--features", default="outputs/e5_rf1_features")
    parser.add_argument("--audit", default="outputs/e5_rf1_features/audit/summary.json")
    parser.add_argument("--stability", default="outputs/e5_rf1_input_stability/summary.json")
    parser.add_argument("--locked-e5", default="outputs/e5_comparator_frontier/endpoint_summary.csv")
    parser.add_argument("--locked-e5-root", default="outputs/e5_comparator_frontier")
    parser.add_argument(
        "--execution-contract", default="docs/e5_reinhard_residual_frequency_contract.md"
    )
    parser.add_argument("--output", default="outputs/e5_rf1_frontier")
    return parser.parse_args()


def slide_rows(model: dict, raw_root: Path, feature_root: Path):
    model_id = model["encoder_id"]
    paths = sorted((feature_root / model_id / "shards").glob("*.h5"))
    if len(paths) != 109:
        raise ValueError(f"{model_id}: expected 109 E5-RF1 shards, got {len(paths)}")
    invariance_rows = []
    content_rows = []
    collapse_rows = []
    image_rows = []
    for index, path in enumerate(paths):
        slide_id = path.stem
        raw_path = raw_root / model_id / "shards" / path.name
        with h5py.File(raw_path, "r") as source:
            raw = np.asarray(source["features"][:], dtype=np.float32)
        with h5py.File(path, "r") as source:
            hybrid = np.asarray(source["features"][0], dtype=np.float32)
            metric = {name: np.asarray(source[name][0], dtype=np.float32) for name in METRICS}
        if raw.shape != hybrid.shape or raw.shape != (6, 100, int(model["feature_dim"])):
            raise ValueError(f"{model_id}/{slide_id}: feature shape mismatch")
        raw_unit = l2_normalize(raw)
        hybrid_unit = l2_normalize(hybrid)
        raw_at2 = raw_unit[0]
        q95 = unmatched_q95(raw_at2)
        raw_collapse = {
            scanner_index: collapse_metric_values(raw[scanner_index])
            for scanner_index in range(1, len(SCANNERS))
        }
        for condition, values, unit in (
            ("raw", raw, raw_unit),
            (RF1_CONDITION, hybrid, hybrid_unit),
        ):
            invariance_rows.append(
                {
                    "encoder_id": model_id,
                    "slide_id": slide_id,
                    "condition": condition,
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
                        "scanner": scanner,
                        "content_margin": float(margins[scanner_index - 1].mean()),
                    }
                )
                values_for_collapse = (
                    raw_collapse[scanner_index]
                    if condition == "raw"
                    else collapse_metric_values(values[scanner_index])
                )
                for collapse_metric in COLLAPSE_METRICS:
                    collapse_rows.append(
                        {
                            "encoder_id": model_id,
                            "slide_id": slide_id,
                            "condition": condition,
                            "scanner": scanner,
                            "metric": collapse_metric,
                            "raw_value": raw_collapse[scanner_index][collapse_metric],
                            "condition_value": values_for_collapse[collapse_metric],
                            "ratio": values_for_collapse[collapse_metric]
                            / raw_collapse[scanner_index][collapse_metric],
                        }
                    )
        for scanner_index, scanner in enumerate(SCANNERS[1:], start=1):
            row = {
                "encoder_id": model_id,
                "slide_id": slide_id,
                "scanner": scanner,
            }
            row.update(
                {
                    name: float(metric[name][scanner_index].mean())
                    for name in METRICS
                }
            )
            image_rows.append(row)
        print(f"[{index + 1}/109] {model_id} {slide_id}", flush=True)
    return invariance_rows, content_rows, collapse_rows, image_rows


def summarize(invariance: pd.DataFrame, content: pd.DataFrame, collapse: pd.DataFrame):
    indices = bootstrap_indices(109)
    endpoint_rows = []
    scanner_rows = []
    collapse_rows = []
    for model_id in sorted(invariance["encoder_id"].unique()):
        model_invariance = invariance[invariance["encoder_id"] == model_id]
        model_content = content[content["encoder_id"] == model_id]
        model_collapse = collapse[collapse["encoder_id"] == model_id]
        raw_radius = model_invariance[model_invariance["condition"] == "raw"].sort_values(
            "slide_id"
        )["scanner_centroid_rms"].to_numpy()
        corrected_radius = model_invariance[
            model_invariance["condition"] == RF1_CONDITION
        ].sort_values("slide_id")["scanner_centroid_rms"].to_numpy()
        radius_point, radius_lower, radius_upper = bootstrap_mean_ci(corrected_radius, indices)
        difference = bootstrap_mean_ci(raw_radius - corrected_radius, indices)
        raw_radius_point = float(raw_radius.mean())
        rr = 1.0 - radius_point / raw_radius_point if raw_radius_point >= RAW_RADIUS_EPSILON else np.nan

        raw_content = (
            model_content[model_content["condition"] == "raw"]
            .groupby("slide_id", sort=True)["content_margin"]
            .mean()
            .to_numpy()
        )
        corrected_content = model_content[model_content["condition"] == RF1_CONDITION]
        pooled = corrected_content.groupby("slide_id", sort=True)["content_margin"].mean().to_numpy()
        content_point, content_lower, content_upper = bootstrap_mean_ci(pooled, indices)
        delta = bootstrap_mean_ci(pooled - raw_content, indices)
        content_pass = bool(delta[1] >= CONTENT_NONINFERIORITY_MARGIN)

        collapse_passes = []
        for scanner in SCANNERS[1:]:
            scanner_values = corrected_content[corrected_content["scanner"] == scanner].sort_values(
                "slide_id"
            )["content_margin"].to_numpy()
            raw_values = model_content[
                (model_content["condition"] == "raw") & (model_content["scanner"] == scanner)
            ].sort_values("slide_id")["content_margin"].to_numpy()
            point = bootstrap_mean_ci(scanner_values, indices)
            scanner_delta = bootstrap_mean_ci(scanner_values - raw_values, indices)
            scanner_rows.append(
                {
                    "encoder_id": model_id,
                    "condition": RF1_CONDITION,
                    "scanner": scanner,
                    "content_margin": point[0],
                    "content_margin_ci_lower": point[1],
                    "content_margin_ci_upper": point[2],
                    "delta_content_margin": scanner_delta[0],
                    "delta_ci_lower": scanner_delta[1],
                    "delta_ci_upper": scanner_delta[2],
                }
            )
            for metric in COLLAPSE_METRICS:
                ratios = model_collapse[
                    (model_collapse["condition"] == RF1_CONDITION)
                    & (model_collapse["scanner"] == scanner)
                    & (model_collapse["metric"] == metric)
                ].sort_values("slide_id")["ratio"].to_numpy()
                values = bootstrap_mean_ci(ratios, indices)
                gate = bool(
                    values[0] >= COLLAPSE_POINT_THRESHOLD
                    and values[1] >= COLLAPSE_LOWER_THRESHOLD
                )
                collapse_passes.append(gate)
                collapse_rows.append(
                    {
                        "encoder_id": model_id,
                        "condition": RF1_CONDITION,
                        "scanner": scanner,
                        "metric": metric,
                        "ratio": values[0],
                        "ratio_ci_lower": values[1],
                        "ratio_ci_upper": values[2],
                        "gate_pass": gate,
                    }
                )
        collapse_pass = bool(all(collapse_passes))
        improved = bool(difference[1] > 0)
        safe = bool(content_pass and collapse_pass)
        endpoint_rows.append(
            {
                "encoder_id": model_id,
                "condition": RF1_CONDITION,
                "scanner_centroid_rms": radius_point,
                "radius_ci_lower": radius_lower,
                "radius_ci_upper": radius_upper,
                "raw_minus_condition": difference[0],
                "difference_ci_lower": difference[1],
                "difference_ci_upper": difference[2],
                "relative_radius_reduction": rr,
                "raw_radius_epsilon_pass": raw_radius_point >= RAW_RADIUS_EPSILON,
                "invariance_improved": improved,
                "content_margin": content_point,
                "content_margin_ci_lower": content_lower,
                "content_margin_ci_upper": content_upper,
                "delta_content_margin": delta[0],
                "delta_content_ci_lower": delta[1],
                "delta_content_ci_upper": delta[2],
                "content_noninferiority_pass": content_pass,
                "collapse_every_scanner_pass": collapse_pass,
                "safe_for_pfm": safe,
                "safe_and_invariance_improved": safe and improved,
                "invariance_improved_but_unsafe": improved and not safe,
            }
        )
    return pd.DataFrame(endpoint_rows), pd.DataFrame(scanner_rows), pd.DataFrame(collapse_rows)


def summarize_image(frame: pd.DataFrame):
    rows = []
    indices = bootstrap_indices(109)
    for keys, selected in frame.groupby(["encoder_id", "scanner"], sort=True):
        ordered = selected.sort_values("slide_id")
        row = {"encoder_id": keys[0], "scanner": keys[1]}
        for name in METRICS:
            point, lower, upper = bootstrap_mean_ci(ordered[name].to_numpy(), indices)
            row[name] = point
            row[f"{name}_ci_lower"] = lower
            row[f"{name}_ci_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def incremental_vs_reinhard(
    invariance: pd.DataFrame,
    content: pd.DataFrame,
    locked_root: Path,
):
    locked_invariance = pd.read_csv(locked_root / "slide_invariance.csv", dtype={"slide_id": str})
    locked_content = pd.read_csv(locked_root / "slide_content_margin.csv", dtype={"slide_id": str})
    locked_invariance = locked_invariance[locked_invariance["condition"] == "reinhard_lab"]
    locked_content = locked_content[locked_content["condition"] == "reinhard_lab"]
    rows = []
    indices = bootstrap_indices(109)
    for model_id in sorted(invariance["encoder_id"].unique()):
        hybrid_radius = invariance[
            (invariance["encoder_id"] == model_id)
            & (invariance["condition"] == RF1_CONDITION)
        ][["slide_id", "scanner_centroid_rms"]].rename(
            columns={"scanner_centroid_rms": "rf1_radius"}
        )
        reinhard_radius = locked_invariance[
            locked_invariance["encoder_id"] == model_id
        ][["slide_id", "scanner_centroid_rms"]].rename(
            columns={"scanner_centroid_rms": "reinhard_radius"}
        )
        radius = reinhard_radius.merge(
            hybrid_radius, on="slide_id", validate="one_to_one"
        ).sort_values("slide_id")
        if len(radius) != 109:
            raise ValueError(f"{model_id}: incomplete RF1/Reinhard radius pairing")
        radius_difference = radius["reinhard_radius"].to_numpy() - radius["rf1_radius"].to_numpy()
        radius_point, radius_lower, radius_upper = bootstrap_mean_ci(radius_difference, indices)

        hybrid_content = (
            content[
                (content["encoder_id"] == model_id)
                & (content["condition"] == RF1_CONDITION)
            ]
            .groupby("slide_id", as_index=False)["content_margin"]
            .mean()
            .rename(columns={"content_margin": "rf1_content"})
        )
        reinhard_content = (
            locked_content[locked_content["encoder_id"] == model_id]
            .groupby("slide_id", as_index=False)["content_margin"]
            .mean()
            .rename(columns={"content_margin": "reinhard_content"})
        )
        content_pair = reinhard_content.merge(
            hybrid_content, on="slide_id", validate="one_to_one"
        ).sort_values("slide_id")
        if len(content_pair) != 109:
            raise ValueError(f"{model_id}: incomplete RF1/Reinhard content pairing")
        content_difference = (
            content_pair["rf1_content"].to_numpy()
            - content_pair["reinhard_content"].to_numpy()
        )
        content_point, content_lower, content_upper = bootstrap_mean_ci(
            content_difference, indices
        )
        rows.append(
            {
                "encoder_id": model_id,
                "rf1_condition": RF1_CONDITION,
                "reference_condition": "reinhard_lab",
                "reinhard_minus_rf1_radius": radius_point,
                "radius_difference_ci_lower": radius_lower,
                "radius_difference_ci_upper": radius_upper,
                "rf1_radius_better_than_reinhard": bool(radius_lower > 0),
                "rf1_minus_reinhard_content_margin": content_point,
                "content_difference_ci_lower": content_lower,
                "content_difference_ci_upper": content_upper,
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    audit_path = Path(args.audit)
    audit = json.loads(audit_path.read_text())
    if not (
        audit.get("analysis") == "e5_rf1_feature_population_audit"
        and audit.get("audit_pass") is True
        and audit.get("features_observed") == 261_600
    ):
        raise RuntimeError("E5-RF1 feature population audit has not passed")
    contract_path = Path(args.execution_contract)
    if "**Status:** PRE-OUTCOME FROZEN" not in contract_path.read_text():
        raise RuntimeError("E5-RF1 execution contract is not frozen")
    contract = json.loads(Path(args.contract).read_text())
    all_rows = [[], [], [], []]
    for model in contract["models"]:
        local = slide_rows(model, Path(args.raw), Path(args.features))
        for target, values in zip(all_rows, local):
            target.extend(values)
    invariance = pd.DataFrame(all_rows[0])
    content = pd.DataFrame(all_rows[1])
    collapse = pd.DataFrame(all_rows[2])
    image = pd.DataFrame(all_rows[3])
    endpoints, scanner_summary, collapse_summary = summarize(invariance, content, collapse)
    image_summary = summarize_image(image)
    incremental = incremental_vs_reinhard(
        invariance, content, Path(args.locked_e5_root)
    )

    locked = pd.read_csv(args.locked_e5)
    locked = locked[
        locked["condition"].isin(["reinhard_lab", "frequency_calibration"])
    ].copy()
    comparison_columns = [
        "encoder_id",
        "condition",
        "scanner_centroid_rms",
        "relative_radius_reduction",
        "invariance_improved",
        "content_margin",
        "delta_content_margin",
        "delta_content_ci_lower",
        "content_noninferiority_pass",
        "collapse_every_scanner_pass",
        "safe_for_pfm",
        "safe_and_invariance_improved",
    ]
    comparison = pd.concat(
        [
            locked[comparison_columns].assign(result_scope="locked_primary_e5"),
            endpoints[comparison_columns].assign(result_scope="post_core_e5_rf1"),
        ],
        ignore_index=True,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "slide_invariance.csv": invariance,
        "slide_content_margin.csv": content,
        "slide_collapse.csv": collapse,
        "slide_image_gamut.csv": image,
        "endpoint_summary.csv": endpoints,
        "content_scanner_summary.csv": scanner_summary,
        "collapse_summary.csv": collapse_summary,
        "image_gamut_summary.csv": image_summary,
        "comparison_to_locked_e5.csv": comparison,
        "incremental_vs_reinhard.csv": incremental,
    }
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False)
    stability_path = Path(args.stability)
    stability = json.loads(stability_path.read_text())
    summary = {
        "analysis": "e5_rf1_frontier",
        "rf1_version": RF1_VERSION,
        "condition": RF1_CONDITION,
        "pfms": 4,
        "slides": 109,
        "selected_gain_cap": stability["selected_gain_cap"],
        "bootstrap_unit": "physical_slide",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "content_noninferiority_margin": CONTENT_NONINFERIORITY_MARGIN,
        "collapse_point_threshold": COLLAPSE_POINT_THRESHOLD,
        "collapse_lower_ci_threshold": COLLAPSE_LOWER_THRESHOLD,
        "endpoint_rows": len(endpoints),
        "safe_pfms": int(endpoints["safe_for_pfm"].sum()),
        "safe_and_improved_pfms": int(endpoints["safe_and_invariance_improved"].sum()),
        "common_safe": bool(endpoints["safe_for_pfm"].all()),
        "common_safe_and_improved": bool(endpoints["safe_and_invariance_improved"].all()),
        "population_audit_sha256": sha256(audit_path),
        "input_stability_sha256": sha256(stability_path),
        "locked_e5_endpoint_sha256": sha256(Path(args.locked_e5)),
        "locked_e5_slide_invariance_sha256": sha256(
            Path(args.locked_e5_root) / "slide_invariance.csv"
        ),
        "locked_e5_slide_content_sha256": sha256(
            Path(args.locked_e5_root) / "slide_content_margin.csv"
        ),
        "execution_contract_sha256": sha256(contract_path),
        "analysis_gate_pass": bool(
            len(endpoints) == 4
            and len(invariance) == 872
            and len(content) == 4_360
            and len(collapse) == 13_080
            and len(incremental) == 4
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(endpoints.to_string(index=False))
    if not summary["analysis_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
