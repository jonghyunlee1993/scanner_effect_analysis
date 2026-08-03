"""Audit the complete four-PFM Supplement-only Macenko feature population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from e5_comparator_population import SCANNERS
from e5_macenko_supplement import MACENKO_VERSION
from extract_e5_macenko_features import CONDITION
from fetch_e0_pfm_checkpoints import sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--features", default="outputs/e5_macenko_features")
    parser.add_argument("--supplement-contract", default="docs/e5_macenko_supplement_contract.md")
    parser.add_argument("--output", default="outputs/e5_macenko_features/audit")
    parser.add_argument("--skip-sha256", action="store_true")
    return parser.parse_args()


def strings(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def main():
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text())
    contract_sha = sha256(Path(args.supplement_contract))
    rows = []
    for model in contract["models"]:
        model_id = model["encoder_id"]
        raw_paths = sorted((Path(args.raw) / model_id / "shards").glob("*.h5"))
        expected = {path.name for path in raw_paths}
        observed = {path.name for path in (Path(args.features) / model_id / "shards").glob("*.h5")}
        for filename in sorted(observed - expected):
            rows.append({
                "encoder_id": model_id,
                "slide_id": Path(filename).stem,
                "status": "fail",
                "features": 0,
                "fallback_patches": 0,
                "bytes": 0,
                "minimum_norm": np.nan,
                "error": "unexpected Macenko shard",
            })
        for raw_path in raw_paths:
            slide_id = raw_path.stem
            path = Path(args.features) / model_id / "shards" / raw_path.name
            try:
                if path.name not in observed or not path.with_suffix(".summary.json").exists():
                    raise FileNotFoundError(path)
                summary = json.loads(path.with_suffix(".summary.json").read_text())
                raw_summary = json.loads(raw_path.with_suffix(".summary.json").read_text())
                checks = {
                    "analysis": "e5_macenko_feature_shard",
                    "macenko_version": MACENKO_VERSION,
                    "slide_id": slide_id,
                    "encoder_id": model_id,
                    "feature_dim": int(model["feature_dim"]),
                    "condition": CONDITION,
                    "features": 600,
                    "loso_target_train_slides": 108,
                    "raw_feature_sha256": raw_summary["output_sha256"],
                    "supplement_contract_sha256": contract_sha,
                    "shard_gate_pass": True,
                }
                for key, expected in checks.items():
                    if summary.get(key) != expected:
                        raise ValueError(f"{key} mismatch")
                if not args.skip_sha256 and summary.get("output_sha256") != sha256(path):
                    raise ValueError("SHA256 mismatch")
                with h5py.File(raw_path, "r") as raw_source, h5py.File(path, "r") as source:
                    raw = np.asarray(raw_source["features"][:], dtype=np.float32)
                    values = np.asarray(source["features"][:], dtype=np.float32)
                    fallback = np.asarray(source["fallback"][:], dtype=np.uint8)
                    if values.shape != (1, 6, 100, int(model["feature_dim"])) or fallback.shape != (1, 6, 100):
                        raise ValueError("feature/fallback shape mismatch")
                    if strings(source["condition"][:]) != [CONDITION] or strings(source["scanner"][:]) != list(SCANNERS):
                        raise ValueError("condition/scanner order mismatch")
                    if not np.array_equal(values[0, 0], raw[0]) or np.any(fallback[:, 0] != 0):
                        raise ValueError("raw AT2 equality failed")
                    for name in ("scanner", "location_id", "replicate_id", "canonical_center_x", "canonical_center_y"):
                        observed_identity = source[name][:]
                        expected_identity = raw_source[name][:]
                        equal = strings(observed_identity) == strings(expected_identity) if observed_identity.dtype.kind in {"S", "O", "U"} else np.array_equal(observed_identity, expected_identity)
                        if not equal:
                            raise ValueError(f"{name} identity mismatch")
                    norms = np.linalg.norm(values.astype(np.float64), axis=-1)
                    if not np.isfinite(values).all() or np.any(norms <= 0):
                        raise ValueError("invalid features")
                rows.append({
                    "encoder_id": model_id,
                    "slide_id": slide_id,
                    "status": "pass",
                    "features": 600,
                    "fallback_patches": int(fallback.sum()),
                    "bytes": int(path.stat().st_size),
                    "minimum_norm": float(norms.min()),
                    "error": "",
                })
            except Exception as error:
                rows.append({
                    "encoder_id": model_id,
                    "slide_id": slide_id,
                    "status": "fail",
                    "features": 0,
                    "fallback_patches": 0,
                    "bytes": 0,
                    "minimum_norm": np.nan,
                    "error": f"{type(error).__name__}: {error}",
                })
    frame = pd.DataFrame(rows)
    gate = bool(len(frame) == 436 and frame["status"].eq("pass").all() and frame["features"].sum() == 261_600)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "shard_audit.csv", index=False)
    summary = {
        "analysis": "e5_macenko_feature_population_audit",
        "macenko_version": MACENKO_VERSION,
        "shards_expected": 436,
        "shards_passing": int(frame["status"].eq("pass").sum()),
        "features_expected": 261_600,
        "features_observed": int(frame["features"].sum()),
        "fallback_patches": int(frame["fallback_patches"].sum()),
        "feature_bytes": int(frame["bytes"].sum()),
        "supplement_contract_sha256": contract_sha,
        "audit_pass": gate,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not gate:
        print(frame[frame["status"] == "fail"][["encoder_id", "slide_id", "error"]].to_string(index=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
