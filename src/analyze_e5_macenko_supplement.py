"""Score Supplement-only LOSO Macenko with the frozen E5 endpoint rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from analyze_e4_control_frontier import (
    COLLAPSE_LOWER_THRESHOLD,
    COLLAPSE_POINT_THRESHOLD,
    CONTENT_NONINFERIORITY_MARGIN,
    RAW_RADIUS_EPSILON,
    bootstrap_indices,
)
from e4_primary_metrics import COLLAPSE_METRICS, bootstrap_mean_ci, collapse_metric_values, l2_normalize, scanner_centroid_rms, unmatched_q95
from e5_comparator_population import SCANNERS
from e5_macenko_supplement import MACENKO_VERSION
from fetch_e0_pfm_checkpoints import sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--features", default="outputs/e5_macenko_features")
    parser.add_argument("--audit", default="outputs/e5_macenko_features/audit/summary.json")
    parser.add_argument("--supplement-contract", default="docs/e5_macenko_supplement_contract.md")
    parser.add_argument("--output", default="outputs/e5_macenko_supplement")
    return parser.parse_args()


def main():
    args = parse_args()
    audit_path = Path(args.audit)
    audit = json.loads(audit_path.read_text())
    if audit.get("audit_pass") is not True or audit.get("features_observed") != 261_600:
        raise RuntimeError("Macenko feature population audit has not passed")
    contract_path = Path(args.supplement_contract)
    if audit.get("supplement_contract_sha256") != sha256(contract_path):
        raise RuntimeError("Macenko supplement contract hash drift")
    contract = json.loads(Path(args.contract).read_text())
    invariance_rows = []
    content_rows = []
    collapse_rows = []
    clipping_rows = []
    for model in contract["models"]:
        model_id = model["encoder_id"]
        paths = sorted((Path(args.features) / model_id / "shards").glob("*.h5"))
        if len(paths) != 109:
            raise ValueError(f"{model_id}: expected 109 Macenko shards")
        for index, path in enumerate(paths):
            slide_id = path.stem
            raw_path = Path(args.raw) / model_id / "shards" / path.name
            with h5py.File(raw_path, "r") as source:
                raw = np.asarray(source["features"][:], dtype=np.float32)
            with h5py.File(path, "r") as source:
                macenko = np.asarray(source["features"][0], dtype=np.float32)
                clip = np.asarray(source["preclip_range_fraction"][0], dtype=np.float32)
                clip_mae = np.asarray(source["clip_pixel_mae"][0], dtype=np.float32)
                fallback = np.asarray(source["fallback"][0], dtype=np.uint8)
            raw_unit = l2_normalize(raw)
            macenko_unit = l2_normalize(macenko)
            raw_at2 = raw_unit[0]
            q95 = unmatched_q95(raw_at2)
            raw_collapse = {scanner_index: collapse_metric_values(raw[scanner_index]) for scanner_index in range(1, 6)}
            mac_collapse = {scanner_index: collapse_metric_values(macenko[scanner_index]) for scanner_index in range(1, 6)}
            for condition, values in (("raw", raw), ("macenko_supplement", macenko)):
                invariance_rows.append({
                    "encoder_id": model_id,
                    "slide_id": slide_id,
                    "condition": condition,
                    "scanner_centroid_rms": scanner_centroid_rms(values),
                })
                unit = raw_unit if condition == "raw" else macenko_unit
                margins = np.sum(unit[1:] * raw_at2[None], axis=-1) - q95[None]
                for scanner_index, scanner in enumerate(SCANNERS[1:], start=1):
                    content_rows.append({
                        "encoder_id": model_id,
                        "slide_id": slide_id,
                        "condition": condition,
                        "scanner": scanner,
                        "content_margin": float(margins[scanner_index - 1].mean()),
                    })
                    current = raw_collapse[scanner_index] if condition == "raw" else mac_collapse[scanner_index]
                    for metric in COLLAPSE_METRICS:
                        collapse_rows.append({
                            "encoder_id": model_id,
                            "slide_id": slide_id,
                            "condition": condition,
                            "scanner": scanner,
                            "metric": metric,
                            "ratio": current[metric] / raw_collapse[scanner_index][metric],
                        })
            for scanner_index, scanner in enumerate(SCANNERS[1:], start=1):
                clipping_rows.append({
                    "encoder_id": model_id,
                    "slide_id": slide_id,
                    "scanner": scanner,
                    "preclip_range_fraction": float(clip[scanner_index].mean()),
                    "clip_pixel_mae": float(clip_mae[scanner_index].mean()),
                    "fallback_fraction": float(fallback[scanner_index].mean()),
                })
            print(f"[{index + 1}/109] {model_id} {slide_id}", flush=True)
    invariance = pd.DataFrame(invariance_rows)
    content = pd.DataFrame(content_rows)
    collapse = pd.DataFrame(collapse_rows)
    clipping = pd.DataFrame(clipping_rows)
    indices = bootstrap_indices(109)
    endpoint_rows = []
    scanner_content_rows = []
    collapse_summary_rows = []
    for model_id in [model["encoder_id"] for model in contract["models"]]:
        raw_radius = invariance[(invariance.encoder_id == model_id) & (invariance.condition == "raw")].sort_values("slide_id").scanner_centroid_rms.to_numpy()
        corrected_radius = invariance[(invariance.encoder_id == model_id) & (invariance.condition == "macenko_supplement")].sort_values("slide_id").scanner_centroid_rms.to_numpy()
        radius = bootstrap_mean_ci(corrected_radius, indices)
        difference = bootstrap_mean_ci(raw_radius - corrected_radius, indices)
        raw_radius_point = float(raw_radius.mean())
        rr = 1.0 - radius[0] / raw_radius_point if raw_radius_point >= RAW_RADIUS_EPSILON else np.nan
        model_content = content[content.encoder_id == model_id]
        raw_pooled = model_content[model_content.condition == "raw"].groupby("slide_id", sort=True).content_margin.mean().to_numpy()
        corrected_frame = model_content[model_content.condition == "macenko_supplement"]
        corrected_pooled = corrected_frame.groupby("slide_id", sort=True).content_margin.mean().to_numpy()
        corrected_content = bootstrap_mean_ci(corrected_pooled, indices)
        delta = bootstrap_mean_ci(corrected_pooled - raw_pooled, indices)
        for scanner in SCANNERS[1:]:
            values = corrected_frame[corrected_frame.scanner == scanner].sort_values("slide_id").content_margin.to_numpy()
            raw_values = model_content[(model_content.condition == "raw") & (model_content.scanner == scanner)].sort_values("slide_id").content_margin.to_numpy()
            local = bootstrap_mean_ci(values - raw_values, indices)
            scanner_content_rows.append({
                "encoder_id": model_id,
                "scanner": scanner,
                "delta_content_margin": local[0],
                "delta_ci_lower": local[1],
                "delta_ci_upper": local[2],
            })
        collapse_passes = []
        model_collapse = collapse[(collapse.encoder_id == model_id) & (collapse.condition == "macenko_supplement")]
        for scanner in SCANNERS[1:]:
            for metric in COLLAPSE_METRICS:
                ratios = model_collapse[(model_collapse.scanner == scanner) & (model_collapse.metric == metric)].sort_values("slide_id").ratio.to_numpy()
                estimate = bootstrap_mean_ci(ratios, indices)
                passed = bool(estimate[0] >= COLLAPSE_POINT_THRESHOLD and estimate[1] >= COLLAPSE_LOWER_THRESHOLD)
                collapse_passes.append(passed)
                collapse_summary_rows.append({
                    "encoder_id": model_id,
                    "scanner": scanner,
                    "metric": metric,
                    "ratio": estimate[0],
                    "ratio_ci_lower": estimate[1],
                    "ratio_ci_upper": estimate[2],
                    "gate_pass": passed,
                })
        content_pass = bool(delta[1] >= CONTENT_NONINFERIORITY_MARGIN)
        collapse_pass = bool(all(collapse_passes))
        improved = bool(difference[1] > 0)
        endpoint_rows.append({
            "encoder_id": model_id,
            "condition": "macenko_supplement",
            "scanner_centroid_rms": radius[0],
            "radius_ci_lower": radius[1],
            "radius_ci_upper": radius[2],
            "raw_minus_condition": difference[0],
            "difference_ci_lower": difference[1],
            "difference_ci_upper": difference[2],
            "relative_radius_reduction": rr,
            "invariance_improved": improved,
            "content_margin": corrected_content[0],
            "delta_content_margin": delta[0],
            "delta_content_ci_lower": delta[1],
            "delta_content_ci_upper": delta[2],
            "content_noninferiority_pass": content_pass,
            "collapse_every_scanner_pass": collapse_pass,
            "safe_for_pfm": content_pass and collapse_pass,
            "safe_and_invariance_improved": content_pass and collapse_pass and improved,
        })
    endpoints = pd.DataFrame(endpoint_rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for filename, frame in {
        "slide_invariance.csv": invariance,
        "slide_content_margin.csv": content,
        "slide_collapse.csv": collapse,
        "slide_clipping_fallback.csv": clipping,
        "endpoint_summary.csv": endpoints,
        "content_scanner_summary.csv": pd.DataFrame(scanner_content_rows),
        "collapse_summary.csv": pd.DataFrame(collapse_summary_rows),
    }.items():
        frame.to_csv(output / filename, index=False)
    summary = {
        "analysis": "e5_macenko_supplement_frontier",
        "macenko_version": MACENKO_VERSION,
        "pfms": 4,
        "slides": 109,
        "bootstrap_replicates": 5000,
        "bootstrap_seed": 20260803,
        "endpoint_rows": len(endpoints),
        "safe_pfms": int(endpoints.safe_for_pfm.sum()),
        "safe_and_improved_pfms": int(endpoints.safe_and_invariance_improved.sum()),
        "common_safe_across_four_pfms": bool(endpoints.safe_for_pfm.all()),
        "safe_and_improved_all_four_pfms": bool(endpoints.safe_and_invariance_improved.all()),
        "fallback_patches": audit["fallback_patches"],
        "population_audit_sha256": sha256(audit_path),
        "supplement_contract_sha256": sha256(contract_path),
        "analysis_gate_pass": bool(len(endpoints) == 4 and len(invariance) == 872),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["analysis_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

