"""Audit all four frozen PFM feature populations and materialize sample identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from extract_e0_pfm_features import FEATURE_VERSION, SCANNERS
from fetch_e0_pfm_checkpoints import sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json"
    )
    parser.add_argument(
        "--geometry",
        default="outputs/e0_native_geometry_final/native_geometry_manifest.csv",
    )
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--features", default="outputs/e0_pfm_features")
    parser.add_argument("--output", default="outputs/e0_pfm_features/audit")
    parser.add_argument("--skip-sha256", action="store_true")
    return parser.parse_args()


def strings(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def audit_feature_shard(
    model: dict,
    grid_path: Path,
    feature_path: Path,
    verify_sha256: bool = True,
):
    slide_id = grid_path.stem
    source_summary = json.loads(grid_path.with_suffix(".summary.json").read_text())
    summary_path = feature_path.with_suffix(".summary.json")
    if not feature_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"{model['encoder_id']}/{slide_id}: missing feature shard")
    summary = json.loads(summary_path.read_text())
    checks = {
        "analysis": "e0_pfm_feature_shard",
        "feature_version": FEATURE_VERSION,
        "slide_id": slide_id,
        "encoder_id": model["encoder_id"],
        "feature_dim": int(model["feature_dim"]),
        "native_fov_px": int(model["native_fov_px"]),
        "checkpoint_sha256": model["checkpoint_sha256"],
        "source_grid_sha256": source_summary["output_sha256"],
        "shard_gate_pass": True,
    }
    for key, expected in checks.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"{model['encoder_id']}/{slide_id}: summary {key} differs: "
                f"{summary.get(key)!r} != {expected!r}"
            )
    if verify_sha256 and sha256(feature_path) != summary.get("output_sha256"):
        raise ValueError(f"{model['encoder_id']}/{slide_id}: feature SHA256 mismatch")

    with h5py.File(grid_path, "r") as grid, h5py.File(feature_path, "r") as feature:
        for key, expected in checks.items():
            if key == "shard_gate_pass":
                continue
            if key in feature.attrs and feature.attrs[key] != expected:
                raise ValueError(f"{model['encoder_id']}/{slide_id}: HDF5 {key} differs")
        values = feature["features"][:]
        expected_shape = (6, 100, int(model["feature_dim"]))
        if values.shape != expected_shape or values.dtype != np.float32:
            raise ValueError(
                f"{model['encoder_id']}/{slide_id}: feature array {values.shape}/{values.dtype} "
                f"!= {expected_shape}/float32"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{model['encoder_id']}/{slide_id}: non-finite features")
        norms = np.linalg.norm(values.astype(np.float64), axis=2)
        if np.any(norms <= 1e-12):
            raise ValueError(f"{model['encoder_id']}/{slide_id}: zero-norm feature")
        total_variance = float(np.var(values.astype(np.float64), axis=(0, 1)).sum())
        if not np.isfinite(total_variance) or total_variance <= 1e-12:
            raise ValueError(f"{model['encoder_id']}/{slide_id}: collapsed feature shard")
        for name in (
            "scanner",
            "location_id",
            "replicate_id",
            "canonical_center_x",
            "canonical_center_y",
        ):
            observed = feature[name][:]
            expected = grid[name][:]
            if observed.dtype.kind in {"S", "O", "U"}:
                equal = strings(observed) == strings(expected)
            else:
                equal = np.array_equal(observed, expected)
            if not equal:
                raise ValueError(f"{model['encoder_id']}/{slide_id}: identity {name} differs")
        if strings(feature["scanner"][:]) != list(SCANNERS):
            raise ValueError(f"{model['encoder_id']}/{slide_id}: scanner order differs")
    return {
        "encoder_id": model["encoder_id"],
        "slide_id": slide_id,
        "status": "pass",
        "features": 600,
        "feature_dim": int(model["feature_dim"]),
        "feature_bytes": int(feature_path.stat().st_size),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "total_variance": total_variance,
        "sha256_verified": bool(verify_sha256),
        "error": "",
    }


def build_feature_index(geometry: pd.DataFrame, models: list[dict], feature_root: Path):
    columns = [
        "slide_id",
        "tissue_type",
        "scanner",
        "location_id",
        "replicate_id",
        "canonical_center_x",
        "canonical_center_y",
        "pixel_source",
    ]
    index = geometry[columns].copy()
    index["scanner"] = pd.Categorical(index["scanner"], categories=SCANNERS, ordered=True)
    index = index.sort_values(["slide_id", "scanner", "location_id"]).reset_index(drop=True)
    index["scanner_index"] = index["scanner"].cat.codes.astype(np.int8)
    index["location_index"] = index["location_id"].astype(np.int16)
    index["scanner"] = index["scanner"].astype("object")
    for model in models:
        encoder_id = model["encoder_id"]
        index[f"{encoder_id}_feature_path"] = index["slide_id"].map(
            lambda slide_id: str(feature_root / encoder_id / "shards" / f"{slide_id}.h5")
        )
        index[f"{encoder_id}_feature_dim"] = int(model["feature_dim"])
        index[f"{encoder_id}_native_fov_px"] = int(model["native_fov_px"])
    if len(index) != 65_400 or index.duplicated(["slide_id", "scanner", "location_id"]).any():
        raise ValueError("feature index identity is incomplete or duplicated")
    return index


def audit_population(contract, geometry, grid_root: Path, feature_root: Path, verify_sha256=True):
    slides = sorted(geometry["slide_id"].astype(str).unique())
    rows = []
    for model in contract["models"]:
        encoder_id = model["encoder_id"]
        expected = {f"{slide_id}.h5" for slide_id in slides}
        shard_root = feature_root / encoder_id / "shards"
        observed = {path.name for path in shard_root.glob("*.h5")}
        for filename in sorted(expected | observed):
            slide_id = Path(filename).stem
            if filename not in expected:
                rows.append(
                    {
                        "encoder_id": encoder_id,
                        "slide_id": slide_id,
                        "status": "fail",
                        "features": 0,
                        "feature_dim": int(model["feature_dim"]),
                        "feature_bytes": 0,
                        "norm_min": np.nan,
                        "norm_max": np.nan,
                        "total_variance": np.nan,
                        "sha256_verified": False,
                        "error": "unexpected feature shard",
                    }
                )
                continue
            try:
                rows.append(
                    audit_feature_shard(
                        model,
                        grid_root / filename,
                        shard_root / filename,
                        verify_sha256,
                    )
                )
            except Exception as error:
                rows.append(
                    {
                        "encoder_id": encoder_id,
                        "slide_id": slide_id,
                        "status": "fail",
                        "features": 0,
                        "feature_dim": int(model["feature_dim"]),
                        "feature_bytes": 0,
                        "norm_min": np.nan,
                        "norm_max": np.nan,
                        "total_variance": np.nan,
                        "sha256_verified": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    audit = pd.DataFrame(rows)
    expected_rows = len(contract["models"]) * len(slides)
    population_pass = bool(
        len(audit) == expected_rows
        and audit["status"].eq("pass").all()
        and audit["features"].sum() == expected_rows * 600
    )
    summary = {
        "analysis": "e0_pfm_feature_population_audit",
        "feature_version": FEATURE_VERSION,
        "models_expected": len(contract["models"]),
        "models": [model["encoder_id"] for model in contract["models"]],
        "slides_per_model_expected": len(slides),
        "shards_expected": expected_rows,
        "shards_passing": int(audit["status"].eq("pass").sum()),
        "shards_failing": int(audit["status"].eq("fail").sum()),
        "features_expected": expected_rows * 600,
        "features_observed": int(audit["features"].sum()),
        "feature_bytes": int(audit["feature_bytes"].sum()),
        "population_gate_pass": population_pass,
    }
    return audit, summary


def main():
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text())
    geometry = pd.read_csv(args.geometry, dtype={"slide_id": str})
    audit, summary = audit_population(
        contract,
        geometry,
        Path(args.grid),
        Path(args.features),
        not args.skip_sha256,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "shard_audit.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["population_gate_pass"]:
        index = build_feature_index(geometry, contract["models"], Path(args.features))
        index.to_csv(output / "feature_index.csv", index=False)
    print(json.dumps(summary, indent=2))
    if not summary["population_gate_pass"]:
        print(audit[audit["status"].eq("fail")][["encoder_id", "slide_id", "error"]].to_string(index=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
