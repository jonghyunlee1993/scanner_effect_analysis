"""Audit the complete frozen E4 four-PFM control feature population."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from e4_control_population import CONTROL_SPECS, CONTROL_VERSION, SCANNERS, condition_names
from fetch_e0_pfm_checkpoints import sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json"
    )
    parser.add_argument("--features", default="outputs/e4_control_features")
    parser.add_argument("--reference", default="outputs/e4_control_references")
    parser.add_argument("--output", default="outputs/e4_control_features/audit")
    return parser.parse_args()


def decoded(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def audit_shard(path: Path, model: dict, reference_sha256: str):
    summary_path = path.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    failures = []
    if summary.get("output_sha256") != sha256(path):
        failures.append("output_sha256")
    expected_shape = (len(CONTROL_SPECS), 6, 100, int(model["feature_dim"]))
    with h5py.File(path, "r") as source:
        if source["features"].shape != expected_shape:
            failures.append("feature_shape")
        if decoded(source["condition"][:]) != list(condition_names()):
            failures.append("conditions")
        if decoded(source["scanner"][:]) != list(SCANNERS):
            failures.append("scanners")
        if str(source.attrs.get("encoder_id")) != model["encoder_id"]:
            failures.append("encoder_id")
        if str(source.attrs.get("checkpoint_sha256")) != model["checkpoint_sha256"]:
            failures.append("checkpoint_sha256")
        if str(source.attrs.get("global_reference_sha256")) != reference_sha256:
            failures.append("reference_sha256")
        features = source["features"][:]
        if not np.isfinite(features).all():
            failures.append("feature_finite")
        norms = np.linalg.norm(features, axis=-1)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            failures.append("feature_norm")
        for name in (
            "preclip_range_fraction",
            "clip_pixel_mae",
            "requested_hf_rms",
            "operational_hf_rms",
        ):
            values = source[name][:]
            if values.shape != (len(CONTROL_SPECS), 6, 100):
                failures.append(f"{name}_shape")
            if not np.isfinite(values).all() or np.any(values < 0):
                failures.append(f"{name}_values")
        clipping = source["preclip_range_fraction"][:]
        if np.any(clipping > 1):
            failures.append("preclip_fraction_range")
        slide_id = str(source.attrs["slide_id"])
    return {
        "encoder_id": model["encoder_id"],
        "slide_id": slide_id,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "features": int(np.prod(expected_shape[:-1])),
        "minimum_norm": float(norms.min()),
        "maximum_preclip_range_fraction": float(clipping.max()),
        "failures": ";".join(sorted(set(failures))),
        "pass": not failures,
    }


def audit_reference(path: Path, summary: dict, fov: int):
    file_sha = sha256(path)
    failures = []
    if not (
        summary.get("reference_gate_pass") is True
        and summary.get("control_version") == CONTROL_VERSION
        and summary.get("output_sha256") == file_sha
    ):
        failures.append("summary_contract")
    expected_sizes = [fov // (2**index) for index in range(4)]
    with np.load(path) as source:
        for index, size in enumerate(expected_sizes):
            values = source[f"band_sum_{index}"]
            if values.shape != (3, size, size) or not np.isfinite(values).all():
                failures.append(f"band_sum_{index}")
        if int(source["image_count"]) != 65400:
            failures.append("image_count")
        slide_ids = [str(value) for value in source["slide_ids"]]
        if len(slide_ids) != 109 or len(set(slide_ids)) != 109:
            failures.append("slide_ids")
    return {
        "fov": fov,
        "path": str(path.resolve()),
        "sha256": file_sha,
        "summary_sha256": summary.get("output_sha256"),
        "failures": ";".join(failures),
        "pass": not failures,
    }


def main():
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text())
    feature_root = Path(args.features)
    reference_root = Path(args.reference)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    unexpected = []
    temporary = []
    reference_rows = []

    expected_model_ids = {model["encoder_id"] for model in contract["models"]}
    actual_model_ids = {
        path.name for path in feature_root.iterdir() if path.is_dir() and path.name != "audit"
    }
    unexpected.extend(sorted(actual_model_ids - expected_model_ids))
    for model in contract["models"]:
        model_id = model["encoder_id"]
        fov = int(model["native_fov_px"])
        reference_path = reference_root / f"fov_{fov}.npz"
        reference_summary_path = reference_root / f"fov_{fov}.summary.json"
        reference_summary = json.loads(reference_summary_path.read_text())
        reference_sha = sha256(reference_path)
        reference_rows.append(audit_reference(reference_path, reference_summary, fov))
        shard_root = feature_root / model_id / "shards"
        paths = sorted(shard_root.glob("*.h5"))
        summary_paths = sorted(shard_root.glob("*.summary.json"))
        temporary.extend(str(path) for path in shard_root.glob(".*.tmp"))
        if len(paths) != 109:
            missing.append(f"{model_id}: expected 109 shards, got {len(paths)}")
        if len(summary_paths) != 109:
            missing.append(f"{model_id}: expected 109 summaries, got {len(summary_paths)}")
        for path in paths:
            if not path.with_suffix(".summary.json").exists():
                missing.append(str(path.with_suffix(".summary.json")))
                continue
            rows.append(audit_shard(path, model, reference_sha))

    fields = [
        "encoder_id",
        "slide_id",
        "path",
        "bytes",
        "sha256",
        "features",
        "minimum_norm",
        "maximum_preclip_range_fraction",
        "failures",
        "pass",
    ]
    with (output_root / "shard_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_root / "reference_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=reference_rows[0].keys())
        writer.writeheader()
        writer.writerows(reference_rows)

    expected_features = 4 * 109 * len(CONTROL_SPECS) * 600
    passed = sum(int(row["pass"]) for row in rows)
    observed_features = sum(row["features"] for row in rows)
    audit_pass = bool(
        len(rows) == 436
        and passed == 436
        and observed_features == expected_features
        and all(row["pass"] for row in reference_rows)
        and not missing
        and not unexpected
        and not temporary
    )
    summary = {
        "analysis": "e4_control_feature_population_audit",
        "control_version": CONTROL_VERSION,
        "models": len(expected_model_ids),
        "conditions": list(condition_names()),
        "shards_expected": 436,
        "shards_observed": len(rows),
        "shards_passing": passed,
        "features_expected": expected_features,
        "features_observed": observed_features,
        "bytes": sum(row["bytes"] for row in rows),
        "missing": missing,
        "unexpected": unexpected,
        "temporary": temporary,
        "reference_gate_pass": all(row["pass"] for row in reference_rows),
        "audit_pass": audit_pass,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not audit_pass:
        raise RuntimeError("E4 control feature population audit failed")


if __name__ == "__main__":
    main()
