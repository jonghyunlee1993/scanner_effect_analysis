"""Score the RF1U multi-target conditions with the frozen E4--E7 fidelity gates.

Metric definitions, thresholds, the bootstrap seed and the replicate count are
reused unchanged.  The single generalization the contract allows is the content
anchor: a condition mapping toward target `T` is judged on agreement with raw
`T`, which reduces to the locked definition when `T` is AT2.

The primary comparison is the paired per-slide scanner-radius difference between
RF1U and the Reinhard comparator of the same target.
"""

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
from extract_rf1u_features import ANALYSIS as SHARD_ANALYSIS
from fetch_e0_pfm_checkpoints import sha256
from rf1u_unpaired import RF1U_TARGETS, RF1U_VERSION, source_indices, target_index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--features", default="outputs/rf1u_multitarget/features")
    parser.add_argument("--output", default="outputs/rf1u_multitarget/frontier")
    return parser.parse_args()


def slide_rows(model: dict, target: str, raw_root: Path, feature_root: Path):
    """Per-slide invariance, content and collapse rows for one target and model."""
    model_id = model["encoder_id"]
    reference = target_index(target)
    sources = source_indices(target)
    paths = sorted((feature_root / target / model_id / "shards").glob("*.h5"))
    if len(paths) != 109:
        raise ValueError(f"{target}/{model_id}: expected 109 shards, got {len(paths)}")
    invariance_rows, content_rows, collapse_rows, image_rows = [], [], [], []
    for path in paths:
        slide_id = path.stem
        summary = json.loads(path.with_suffix(".summary.json").read_text())
        if not (
            summary.get("analysis") == SHARD_ANALYSIS
            and summary.get("rf1u_version") == RF1U_VERSION
            and summary.get("target") == target
            and summary.get("encoder_id") == model_id
            and summary.get("output_sha256") == sha256(path)
            and summary.get("shard_gate_pass") is True
        ):
            raise RuntimeError(f"invalid RF1U shard: {path}")
        with h5py.File(raw_root / model_id / "shards" / path.name, "r") as source:
            raw = np.asarray(source["features"][:], dtype=np.float32)
        with h5py.File(path, "r") as source:
            rendered = np.asarray(source["features"][:], dtype=np.float32)
            conditions = [value.decode() for value in source["condition"][:]]
            alpha = np.asarray(source["band_alpha"][:], dtype=np.float64)
            gain = np.asarray(source["band_gain"][:], dtype=np.float64)
        if raw.shape != (6, 100, int(model["feature_dim"])) or rendered.shape[1:] != raw.shape:
            raise ValueError(f"{target}/{model_id}/{slide_id}: feature shape mismatch")

        raw_reference = raw[reference]
        q95 = unmatched_q95(raw_reference)
        reference_unit = l2_normalize(raw_reference)
        raw_collapse = {
            index: collapse_metric_values(raw[index]) for index in sources
        }
        for condition, values in [("raw", raw)] + list(zip(conditions, rendered)):
            unit = l2_normalize(values)
            invariance_rows.append(
                {
                    "target": target,
                    "encoder_id": model_id,
                    "slide_id": slide_id,
                    "condition": condition,
                    "scanner_centroid_rms": scanner_centroid_rms(values),
                }
            )
            for index in sources:
                margin = float(
                    (np.sum(unit[index] * reference_unit, axis=-1) - q95).mean()
                )
                content_rows.append(
                    {
                        "target": target,
                        "encoder_id": model_id,
                        "slide_id": slide_id,
                        "condition": condition,
                        "scanner": SCANNERS[index],
                        "content_margin": margin,
                    }
                )
                measured = (
                    raw_collapse[index]
                    if condition == "raw"
                    else collapse_metric_values(values[index])
                )
                for metric in COLLAPSE_METRICS:
                    collapse_rows.append(
                        {
                            "target": target,
                            "encoder_id": model_id,
                            "slide_id": slide_id,
                            "condition": condition,
                            "scanner": SCANNERS[index],
                            "metric": metric,
                            "ratio": measured[metric] / raw_collapse[index][metric],
                        }
                    )
        for position, index in enumerate(sources):
            image_rows.append(
                {
                    "target": target,
                    "encoder_id": model_id,
                    "slide_id": slide_id,
                    "scanner": SCANNERS[index],
                    **{
                        f"alpha_sigma{band + 1}": float(alpha[position, band])
                        for band in range(alpha.shape[1])
                    },
                    **{
                        f"gain_sigma{band + 1}": float(gain[position, band])
                        for band in range(gain.shape[1])
                    },
                }
            )
    return invariance_rows, content_rows, collapse_rows, image_rows


def score(invariance, content, collapse, target: str, model_id: str, condition: str, indices):
    """Frozen invariance, content and collapse verdict for one condition."""
    sources = [SCANNERS[index] for index in source_indices(target)]
    raw_radius = (
        invariance[invariance["condition"] == "raw"].sort_values("slide_id")
    )["scanner_centroid_rms"].to_numpy()
    radius = (
        invariance[invariance["condition"] == condition].sort_values("slide_id")
    )["scanner_centroid_rms"].to_numpy()
    radius_point, radius_lower, radius_upper = bootstrap_mean_ci(radius, indices)
    difference = bootstrap_mean_ci(raw_radius - radius, indices)
    raw_point = float(raw_radius.mean())
    rr = 1.0 - radius_point / raw_point if raw_point >= RAW_RADIUS_EPSILON else np.nan

    raw_content = (
        content[content["condition"] == "raw"]
        .groupby("slide_id", sort=True)["content_margin"]
        .mean()
        .to_numpy()
    )
    selected = content[content["condition"] == condition]
    pooled = selected.groupby("slide_id", sort=True)["content_margin"].mean().to_numpy()
    content_point, content_lower, content_upper = bootstrap_mean_ci(pooled, indices)
    delta = bootstrap_mean_ci(pooled - raw_content, indices)
    content_pass = bool(delta[1] >= CONTENT_NONINFERIORITY_MARGIN)

    collapse_rows = []
    gates = []
    for scanner in sources:
        for metric in COLLAPSE_METRICS:
            ratios = collapse[
                (collapse["condition"] == condition)
                & (collapse["scanner"] == scanner)
                & (collapse["metric"] == metric)
            ].sort_values("slide_id")["ratio"].to_numpy()
            values = bootstrap_mean_ci(ratios, indices)
            gate = bool(
                values[0] >= COLLAPSE_POINT_THRESHOLD
                and values[1] >= COLLAPSE_LOWER_THRESHOLD
            )
            gates.append(gate)
            collapse_rows.append(
                {
                    "target": target,
                    "encoder_id": model_id,
                    "condition": condition,
                    "scanner": scanner,
                    "metric": metric,
                    "ratio": values[0],
                    "ratio_ci_lower": values[1],
                    "ratio_ci_upper": values[2],
                    "gate_pass": gate,
                }
            )
    collapse_pass = bool(all(gates))
    improved = bool(difference[1] > 0)
    safe = bool(content_pass and collapse_pass)
    endpoint = {
        "target": target,
        "encoder_id": model_id,
        "condition": condition,
        "scanner_centroid_rms": radius_point,
        "radius_ci_lower": radius_lower,
        "radius_ci_upper": radius_upper,
        "raw_scanner_centroid_rms": raw_point,
        "raw_minus_condition": difference[0],
        "difference_ci_lower": difference[1],
        "difference_ci_upper": difference[2],
        "relative_radius_reduction": rr,
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
    }
    return endpoint, collapse_rows, radius


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads(Path(args.contract).read_text())
    models = contract["models"] if isinstance(contract, dict) else contract
    indices = bootstrap_indices(109)

    endpoint_rows, collapse_rows, incremental_rows, image_rows = [], [], [], []
    invariance_all, content_all = [], []
    for target in RF1U_TARGETS:
        for model in models:
            model_id = model["encoder_id"]
            invariance, content, collapse, image = slide_rows(
                model, target, Path(args.raw), Path(args.features)
            )
            invariance_all.extend(invariance)
            content_all.extend(content)
            image_rows.extend(image)
            invariance = pd.DataFrame(invariance)
            content = pd.DataFrame(content)
            collapse = pd.DataFrame(collapse)
            radii = {}
            for condition in sorted(set(invariance["condition"]) - {"raw"}):
                endpoint, rows, radius = score(
                    invariance, content, collapse, target, model_id, condition, indices
                )
                endpoint_rows.append(endpoint)
                collapse_rows.extend(rows)
                radii[condition] = radius
            comparator = f"reinhard_{target}"
            method = next(name for name in radii if name != comparator)
            paired = bootstrap_mean_ci(radii[comparator] - radii[method], indices)
            incremental_rows.append(
                {
                    "target": target,
                    "encoder_id": model_id,
                    "comparator": comparator,
                    "method": method,
                    "reinhard_minus_rf1u_radius": paired[0],
                    "paired_ci_lower": paired[1],
                    "paired_ci_upper": paired[2],
                    "rf1u_better_than_reinhard": bool(paired[1] > 0),
                }
            )
            print(f"scored {target} / {model_id}", flush=True)

    endpoints = pd.DataFrame(endpoint_rows)
    incremental = pd.DataFrame(incremental_rows)
    tables = {
        "endpoint_summary.csv": endpoints,
        "incremental_vs_reinhard.csv": incremental,
        "collapse_gates.csv": pd.DataFrame(collapse_rows),
        "fitted_band_gains.csv": pd.DataFrame(image_rows),
        "slide_invariance.csv": pd.DataFrame(invariance_all),
    }
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)

    merged = endpoints.merge(
        incremental[["target", "encoder_id", "method", "rf1u_better_than_reinhard"]],
        left_on=["target", "encoder_id", "condition"],
        right_on=["target", "encoder_id", "method"],
        how="left",
    )
    merged["safe_and_better_than_reinhard"] = (
        merged["safe_for_pfm"] & merged["rf1u_better_than_reinhard"].fillna(False)
    )
    summary = {
        "analysis": "rf1u_multitarget_frontier",
        "rf1u_version": RF1U_VERSION,
        "status": "POST_CORE_EXTENSION_RESULT",
        "contract": "docs/e5_rf1u_multitarget_contract.md",
        "content_anchor": "raw acquisition of each condition's own target",
        "targets": list(RF1U_TARGETS),
        "encoders": [model["encoder_id"] for model in models],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "conditions_scored": int(len(endpoints)),
        "rf1u_cells": int(len(incremental)),
        "rf1u_better_than_reinhard_cells": int(
            incremental["rf1u_better_than_reinhard"].sum()
        ),
        "rf1u_safe_and_better_cells": int(
            merged.loc[merged["condition"].str.startswith("reinhard_unpaired"),
                       "safe_and_better_than_reinhard"].sum()
        ),
        "artifacts": {name: sha256(output / name) for name in tables},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
