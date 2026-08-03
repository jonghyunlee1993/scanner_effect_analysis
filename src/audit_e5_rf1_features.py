"""Audit the complete four-PFM E5-RF1 feature population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from e5_comparator_population import SCANNERS
from e5_reinhard_residual_frequency import RF1_CONDITION, RF1_VERSION
from extract_e5_rf1_features import METRICS
from fetch_e0_pfm_checkpoints import sha256


EXPECTED_SHARDS = 436
EXPECTED_FEATURES = 261_600


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--features", default="outputs/e5_rf1_features")
    parser.add_argument("--stability", default="outputs/e5_rf1_input_stability/summary.json")
    parser.add_argument("--output", default="outputs/e5_rf1_features/audit")
    parser.add_argument("--skip-sha256", action="store_true")
    return parser.parse_args()


def strings(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def main():
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text())
    stability_path = Path(args.stability)
    stability = json.loads(stability_path.read_text())
    if stability.get("stability_gate_pass") is not True:
        raise RuntimeError("E5-RF1 stability gate has not passed")
    stability_sha = sha256(stability_path)
    rows = []
    for model in contract["models"]:
        model_id = model["encoder_id"]
        raw_paths = sorted((Path(args.raw) / model_id / "shards").glob("*.h5"))
        observed = {path.name for path in (Path(args.features) / model_id / "shards").glob("*.h5")}
        expected = {path.name for path in raw_paths}
        for filename in sorted(expected | observed):
            try:
                if filename not in expected:
                    raise ValueError("unexpected RF1 shard")
                raw_path = Path(args.raw) / model_id / "shards" / filename
                path = Path(args.features) / model_id / "shards" / filename
                summary = json.loads(path.with_suffix(".summary.json").read_text())
                checks = {
                    "analysis": "e5_rf1_feature_shard",
                    "rf1_version": RF1_VERSION,
                    "condition": RF1_CONDITION,
                    "slide_id": Path(filename).stem,
                    "encoder_id": model_id,
                    "feature_dim": int(model["feature_dim"]),
                    "features": 600,
                    "checkpoint_sha256": model["checkpoint_sha256"],
                    "stability_manifest_sha256": stability_sha,
                    "shard_gate_pass": True,
                }
                for key, expected_value in checks.items():
                    if summary.get(key) != expected_value:
                        raise ValueError(f"{key} mismatch")
                if not args.skip_sha256 and sha256(path) != summary.get("output_sha256"):
                    raise ValueError("output SHA256 mismatch")
                raw_summary = json.loads(raw_path.with_suffix(".summary.json").read_text())
                if summary.get("raw_feature_sha256") != raw_summary.get("output_sha256"):
                    raise ValueError("raw feature hash mismatch")
                fov = int(model["native_fov_px"])
                fold = int(summary["fold"])
                expected_stats = stability["fold_statistics_sha256"][f"fov_{fov}_fold_{fold}"]
                if summary.get("fold_statistics_sha256") != expected_stats:
                    raise ValueError("fold statistics hash mismatch")
                with h5py.File(raw_path, "r") as raw_source, h5py.File(path, "r") as source:
                    raw = np.asarray(raw_source["features"][:], dtype=np.float32)
                    values = np.asarray(source["features"][:], dtype=np.float32)
                    if values.shape != (1, 6, 100, int(model["feature_dim"])):
                        raise ValueError("feature schema mismatch")
                    if strings(source["condition"][:]) != [RF1_CONDITION]:
                        raise ValueError("condition mismatch")
                    if strings(source["scanner"][:]) != list(SCANNERS):
                        raise ValueError("scanner mismatch")
                    for name in (
                        "scanner",
                        "location_id",
                        "replicate_id",
                        "canonical_center_x",
                        "canonical_center_y",
                    ):
                        observed_value = source[name][:]
                        expected_value = raw_source[name][:]
                        equal = (
                            strings(observed_value) == strings(expected_value)
                            if observed_value.dtype.kind in {"S", "O", "U"}
                            else np.array_equal(observed_value, expected_value)
                        )
                        if not equal:
                            raise ValueError(f"{name} identity mismatch")
                    if not np.array_equal(values[0, 0], raw[0]):
                        raise ValueError("raw AT2 equality failed")
                    norms = np.linalg.norm(values.astype(np.float64), axis=-1)
                    if not np.isfinite(values).all() or np.any(norms <= 0):
                        raise ValueError("invalid feature values")
                    for name in METRICS:
                        metric = np.asarray(source[name][:], dtype=np.float32)
                        if metric.shape != (1, 6, 100) or not np.isfinite(metric).all():
                            raise ValueError(f"{name} schema/nonfinite")
                        if np.any(metric[:, 0] != 0):
                            raise ValueError(f"{name} AT2 values are nonzero")
                rows.append(
                    {
                        "encoder_id": model_id,
                        "slide_id": Path(filename).stem,
                        "status": "pass",
                        "features": 600,
                        "bytes": int(path.stat().st_size),
                        "minimum_norm": float(norms.min()),
                        "maximum_norm": float(norms.max()),
                        "sha256_verified": not args.skip_sha256,
                        "error": "",
                    }
                )
            except Exception as error:
                rows.append(
                    {
                        "encoder_id": model_id,
                        "slide_id": Path(filename).stem,
                        "status": "fail",
                        "features": 0,
                        "bytes": 0,
                        "minimum_norm": np.nan,
                        "maximum_norm": np.nan,
                        "sha256_verified": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    frame = pd.DataFrame(rows)
    gate = bool(
        len(frame) == EXPECTED_SHARDS
        and frame["status"].eq("pass").all()
        and int(frame["features"].sum()) == EXPECTED_FEATURES
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "shard_audit.csv", index=False)
    summary = {
        "analysis": "e5_rf1_feature_population_audit",
        "rf1_version": RF1_VERSION,
        "models": [model["encoder_id"] for model in contract["models"]],
        "slides_per_model": 109,
        "shards_expected": EXPECTED_SHARDS,
        "shards_passing": int(frame["status"].eq("pass").sum()),
        "features_expected": EXPECTED_FEATURES,
        "features_observed": int(frame["features"].sum()),
        "feature_bytes": int(frame["bytes"].sum()),
        "stability_manifest_sha256": stability_sha,
        "audit_pass": gate,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not gate:
        print(frame[frame["status"] == "fail"].to_string(index=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()

