"""Aggregate input-only E5-RF1 audits and freeze the panel-wide gain cap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from e5_comparator_population import FOVS, SCANNERS
from e5_reinhard_residual_frequency import (
    RF1_FOLDS,
    RF1_GAIN_CAP_CANDIDATES,
    RF1_VERSION,
    log_spectrum_rmse,
)
from fetch_e0_pfm_checkpoints import sha256


MATERIAL_MEAN_LIMIT = 0.005
MATERIAL_Q99_LIMIT = 0.05
PROJECTION_RGB_MAE_LIMIT = 0.002
FINAL_CLAMP_MAE_LIMIT = 1e-6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-audit", default="outputs/e5_rf1_input_fold_audit")
    parser.add_argument("--statistics", default="outputs/e5_rf1_fold_statistics")
    parser.add_argument(
        "--execution-contract", default="docs/e5_reinhard_residual_frequency_contract.md"
    )
    parser.add_argument("--output", default="outputs/e5_rf1_input_stability")
    return parser.parse_args()


def main():
    args = parse_args()
    contract_path = Path(args.execution_contract)
    contract_sha = sha256(contract_path)
    rows = []
    fold_hashes = {}
    statistics_hashes = {}
    all_pass = True
    caps = np.asarray(RF1_GAIN_CAP_CANDIDATES, dtype=np.float64)
    for fov in FOVS:
        fold_values = []
        for fold in range(RF1_FOLDS):
            path = Path(args.fold_audit) / f"fov_{fov}_fold_{fold}.npz"
            summary_path = path.with_suffix(".summary.json")
            summary = json.loads(summary_path.read_text())
            statistics_path = Path(args.statistics) / f"fov_{fov}_fold_{fold}.npz"
            statistics_summary = json.loads(statistics_path.with_suffix(".summary.json").read_text())
            valid = bool(
                summary.get("analysis") == "e5_rf1_input_fold_audit"
                and summary.get("rf1_version") == RF1_VERSION
                and summary.get("outcome_access") is False
                and summary.get("fov") == fov
                and summary.get("fold") == fold
                and summary.get("execution_contract_sha256") == contract_sha
                and summary.get("statistics_sha256") == sha256(statistics_path)
                and summary.get("output_sha256") == sha256(path)
                and summary.get("input_fold_gate_pass") is True
                and statistics_summary.get("statistics_gate_pass") is True
            )
            all_pass &= valid
            if not valid:
                raise RuntimeError(f"invalid E5-RF1 input fold audit: {path}")
            with np.load(path) as source:
                fold_values.append({name: source[name] for name in source.files})
            fold_hashes[f"fov_{fov}_fold_{fold}"] = sha256(path)
            statistics_hashes[f"fov_{fov}_fold_{fold}"] = sha256(statistics_path)

        if sum(len(value["heldout_slide_ids"]) for value in fold_values) != 109:
            raise RuntimeError(f"FOV {fov}: held-out folds do not cover 109 slides")
        target_power = sum(value["validation_target_power"] for value in fold_values)
        base_power = sum(value["validation_base_power"] for value in fold_values)
        output_power = sum(value["validation_output_power"] for value in fold_values)
        frequency_path = Path(args.statistics) / f"fov_{fov}_fold_0.npz"
        with np.load(frequency_path) as frequency_source:
            frequency = frequency_source["radial_frequency"]
        for cap_index, cap in enumerate(caps):
            for scanner_index, scanner in enumerate(SCANNERS[1:]):
                material = np.concatenate(
                    [value["material_range_fraction"][cap_index, :, scanner_index].reshape(-1) for value in fold_values]
                )
                zero_range = np.concatenate(
                    [value["preproject_range_fraction"][cap_index, :, scanner_index].reshape(-1) for value in fold_values]
                )
                projection = np.concatenate(
                    [value["projection_fraction"][cap_index, :, scanner_index].reshape(-1) for value in fold_values]
                )
                projection_mae = np.concatenate(
                    [value["projection_rgb_mae"][cap_index, :, scanner_index].reshape(-1) for value in fold_values]
                )
                final_mae = np.concatenate(
                    [value["final_clamp_mae"][cap_index, :, scanner_index].reshape(-1) for value in fold_values]
                )
                base_rmse = log_spectrum_rmse(base_power[scanner_index], target_power, frequency)
                output_rmse = log_spectrum_rmse(
                    output_power[cap_index, scanner_index], target_power, frequency
                )
                gate = bool(
                    np.isfinite(material).all()
                    and float(material.mean()) <= MATERIAL_MEAN_LIMIT
                    and float(np.quantile(material, 0.99)) <= MATERIAL_Q99_LIMIT
                    and float(projection_mae.mean()) <= PROJECTION_RGB_MAE_LIMIT
                    and float(final_mae.max()) <= FINAL_CLAMP_MAE_LIMIT
                )
                rows.append(
                    {
                        "fov": fov,
                        "scanner": scanner,
                        "gain_cap": float(cap),
                        "patches": len(material),
                        "material_range_fraction_mean": float(material.mean()),
                        "material_range_fraction_q99": float(np.quantile(material, 0.99)),
                        "zero_range_fraction_mean": float(zero_range.mean()),
                        "projection_fraction_mean": float(projection.mean()),
                        "projection_rgb_mae_mean": float(projection_mae.mean()),
                        "final_clamp_mae_max": float(final_mae.max()),
                        "base_log_spectrum_rmse": base_rmse,
                        "output_log_spectrum_rmse": output_rmse,
                        "log_spectrum_rmse_change": output_rmse - base_rmse,
                        "cell_gate_pass": gate,
                    }
                )

    frame = pd.DataFrame(rows)
    cap_pass = frame.groupby("gain_cap", sort=True)["cell_gate_pass"].all()
    passing = [float(cap) for cap, value in cap_pass.items() if bool(value)]
    selected = max(passing) if passing else None
    gate = bool(all_pass and selected is not None and len(frame) == len(caps) * 3 * 5)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "gain_cap_candidates.csv", index=False)
    summary = {
        "analysis": "e5_rf1_input_stability",
        "rf1_version": RF1_VERSION,
        "outcome_access": False,
        "fovs": list(FOVS),
        "folds": RF1_FOLDS,
        "slides": 109,
        "candidate_caps": caps.tolist(),
        "selected_gain_cap": selected,
        "material_mean_limit": MATERIAL_MEAN_LIMIT,
        "material_q99_limit": MATERIAL_Q99_LIMIT,
        "projection_rgb_mae_limit": PROJECTION_RGB_MAE_LIMIT,
        "final_clamp_mae_limit": FINAL_CLAMP_MAE_LIMIT,
        "candidate_rows": len(frame),
        "fold_audit_sha256": fold_hashes,
        "fold_statistics_sha256": statistics_hashes,
        "execution_contract": str(contract_path.resolve()),
        "execution_contract_sha256": contract_sha,
        "stability_gate_pass": gate,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
