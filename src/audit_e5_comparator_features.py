"""Audit the complete E5 image- and feature-space comparator populations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from e5_comparator_population import E5_VERSION, FEATURE_CONDITIONS, IMAGE_CONDITIONS, SCANNERS
from fetch_e0_pfm_checkpoints import sha256


EXPECTED_IMAGE_FEATURES = 784_800
EXPECTED_FEATURE_FEATURES = 523_200


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--image", default="outputs/e5_image_features")
    parser.add_argument("--feature", default="outputs/e5_feature_harmonization")
    parser.add_argument("--stability", default="outputs/e5_input_stability/summary.json")
    parser.add_argument("--output", default="outputs/e5_comparator_features/audit")
    parser.add_argument("--skip-sha256", action="store_true")
    return parser.parse_args()


def strings(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def audit_shard(
    family: str,
    model: dict,
    raw_path: Path,
    path: Path,
    stability_sha: str,
    verify_sha: bool,
):
    slide_id = raw_path.stem
    conditions = IMAGE_CONDITIONS if family == "image" else FEATURE_CONDITIONS
    analysis = "e5_image_feature_shard" if family == "image" else "e5_feature_harmonization_shard"
    expected_shape = (len(conditions), 6, 100, int(model["feature_dim"]))
    summary_path = path.with_suffix(".summary.json")
    if not path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"{family}/{model['encoder_id']}/{slide_id}: missing shard")
    summary = json.loads(summary_path.read_text())
    checks = {
        "analysis": analysis,
        "e5_version": E5_VERSION,
        "slide_id": slide_id,
        "encoder_id": model["encoder_id"],
        "feature_dim": int(model["feature_dim"]),
        "conditions": list(conditions),
        "features": len(conditions) * 600,
        "checkpoint_sha256": model["checkpoint_sha256"],
        "stability_manifest_sha256": stability_sha,
        "shard_gate_pass": True,
    }
    for key, expected in checks.items():
        if summary.get(key) != expected:
            raise ValueError(f"{family}/{model['encoder_id']}/{slide_id}: {key} mismatch")
    if verify_sha and sha256(path) != summary.get("output_sha256"):
        raise ValueError(f"{family}/{model['encoder_id']}/{slide_id}: SHA256 mismatch")
    raw_summary = json.loads(raw_path.with_suffix(".summary.json").read_text())
    if summary.get("raw_feature_sha256") != raw_summary.get("output_sha256"):
        raise ValueError(f"{family}/{model['encoder_id']}/{slide_id}: raw hash mismatch")
    with h5py.File(raw_path, "r") as raw_source, h5py.File(path, "r") as source:
        raw = np.asarray(raw_source["features"][:], dtype=np.float32)
        values = np.asarray(source["features"][:], dtype=np.float32)
        if values.shape != expected_shape or source["features"].dtype != np.float32:
            raise ValueError(f"{family}/{model['encoder_id']}/{slide_id}: feature schema mismatch")
        if strings(source["condition"][:]) != list(conditions):
            raise ValueError(f"{family}/{model['encoder_id']}/{slide_id}: condition order mismatch")
        if strings(source["scanner"][:]) != list(SCANNERS):
            raise ValueError(f"{family}/{model['encoder_id']}/{slide_id}: scanner order mismatch")
        for name in ("scanner", "location_id", "replicate_id", "canonical_center_x", "canonical_center_y"):
            observed = source[name][:]
            expected = raw_source[name][:]
            equal = strings(observed) == strings(expected) if observed.dtype.kind in {"S", "O", "U"} else np.array_equal(observed, expected)
            if not equal:
                raise ValueError(f"{family}/{model['encoder_id']}/{slide_id}: {name} identity mismatch")
        if not np.array_equal(values[:, 0], np.broadcast_to(raw[0], values[:, 0].shape)):
            raise ValueError(f"{family}/{model['encoder_id']}/{slide_id}: raw AT2 equality failed")
        norms = np.linalg.norm(values.astype(np.float64), axis=-1)
        if not np.isfinite(values).all() or np.any(norms <= 0):
            raise ValueError(f"{family}/{model['encoder_id']}/{slide_id}: invalid feature values")
        if family == "image":
            clipping = np.asarray(source["preclip_range_fraction"][:], dtype=np.float32)
            clipping_mae = np.asarray(source["clip_pixel_mae"][:], dtype=np.float32)
            if clipping.shape != (3, 6, 100) or clipping_mae.shape != (3, 6, 100):
                raise ValueError(f"{family}/{model['encoder_id']}/{slide_id}: clipping schema mismatch")
            if not (np.isfinite(clipping).all() and np.isfinite(clipping_mae).all()):
                raise ValueError(f"{family}/{model['encoder_id']}/{slide_id}: clipping non-finite")
            if np.any(clipping[:, 0] != 0) or np.any(clipping_mae[:, 0] != 0):
                raise ValueError(f"{family}/{model['encoder_id']}/{slide_id}: AT2 clipping nonzero")
    return {
        "family": family,
        "encoder_id": model["encoder_id"],
        "slide_id": slide_id,
        "status": "pass",
        "features": len(conditions) * 600,
        "bytes": int(path.stat().st_size),
        "minimum_norm": float(norms.min()),
        "maximum_norm": float(norms.max()),
        "sha256_verified": bool(verify_sha),
        "error": "",
    }


def main():
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text())
    stability_path = Path(args.stability)
    stability = json.loads(stability_path.read_text())
    if stability.get("stability_gate_pass") is not True:
        raise RuntimeError("E5 input stability audit has not passed")
    stability_sha = sha256(stability_path)
    rows = []
    for family, root in (("image", Path(args.image)), ("feature", Path(args.feature))):
        for model in contract["models"]:
            raw_paths = sorted((Path(args.raw) / model["encoder_id"] / "shards").glob("*.h5"))
            expected_names = {path.name for path in raw_paths}
            observed_names = {path.name for path in (root / model["encoder_id"] / "shards").glob("*.h5")}
            for filename in sorted(expected_names | observed_names):
                try:
                    if filename not in expected_names:
                        raise ValueError("unexpected comparator shard")
                    rows.append(
                        audit_shard(
                            family,
                            model,
                            Path(args.raw) / model["encoder_id"] / "shards" / filename,
                            root / model["encoder_id"] / "shards" / filename,
                            stability_sha,
                            not args.skip_sha256,
                        )
                    )
                except Exception as error:
                    rows.append(
                        {
                            "family": family,
                            "encoder_id": model["encoder_id"],
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
    image = frame[frame["family"] == "image"]
    feature = frame[frame["family"] == "feature"]
    gate = bool(
        len(image) == 436
        and len(feature) == 436
        and frame["status"].eq("pass").all()
        and int(image["features"].sum()) == EXPECTED_IMAGE_FEATURES
        and int(feature["features"].sum()) == EXPECTED_FEATURE_FEATURES
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "shard_audit.csv", index=False)
    summary = {
        "analysis": "e5_comparator_feature_population_audit",
        "e5_version": E5_VERSION,
        "models": [model["encoder_id"] for model in contract["models"]],
        "slides_per_model": 109,
        "image_shards_expected": 436,
        "image_shards_passing": int((image["status"] == "pass").sum()),
        "image_features_expected": EXPECTED_IMAGE_FEATURES,
        "image_features_observed": int(image["features"].sum()),
        "feature_shards_expected": 436,
        "feature_shards_passing": int((feature["status"] == "pass").sum()),
        "feature_features_expected": EXPECTED_FEATURE_FEATURES,
        "feature_features_observed": int(feature["features"].sum()),
        "total_features_expected": EXPECTED_IMAGE_FEATURES + EXPECTED_FEATURE_FEATURES,
        "total_features_observed": int(frame["features"].sum()),
        "feature_bytes": int(frame["bytes"].sum()),
        "stability_manifest": str(stability_path.resolve()),
        "stability_manifest_sha256": stability_sha,
        "audit_pass": gate,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not gate:
        print(frame[frame["status"] == "fail"][["family", "encoder_id", "slide_id", "error"]].to_string(index=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()

