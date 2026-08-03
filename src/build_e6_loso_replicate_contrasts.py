"""Materialize frozen E6 replicate-level LOSO radius/content contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from e4_primary_metrics import l2_normalize, unmatched_q95
from e5_comparator_population import FEATURE_CONDITIONS, IMAGE_CONDITIONS, SCANNERS
from fetch_e0_pfm_checkpoints import sha256


METHODS = (*IMAGE_CONDITIONS, *FEATURE_CONDITIONS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--geometry", default="outputs/e0_native_geometry_final/native_geometry_manifest.csv")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--image", default="outputs/e5_image_features")
    parser.add_argument("--feature", default="outputs/e5_feature_harmonization")
    parser.add_argument("--e5-lock", default="outputs/e5_comparator_results_lock/summary.json")
    parser.add_argument("--e6-contract", default="docs/e6_heterogeneity_execution_contract.md")
    parser.add_argument("--output", default="outputs/e6_loso_replicate_contrasts")
    return parser.parse_args()


def location_radius(unit: np.ndarray):
    centroid = unit.mean(axis=0)
    return np.sqrt(np.mean(np.sum((unit - centroid[None]) ** 2, axis=-1), axis=0))


def main():
    args = parse_args()
    lock_path = Path(args.e5_lock)
    lock = json.loads(lock_path.read_text())
    if lock.get("result_lock_pass") is not True or lock.get("population_features") != 1_308_000:
        raise RuntimeError("E5 primary result lock has not passed")
    e6_contract = Path(args.e6_contract)
    if "**Status:** FROZEN" not in e6_contract.read_text():
        raise RuntimeError("E6 execution contract is not frozen")
    geometry = pd.read_csv(args.geometry, dtype={"slide_id": str})
    annotation = geometry[["slide_id", "tissue_type"]].drop_duplicates()
    if len(annotation) != 109 or annotation.tissue_type.nunique() != 37 or annotation.slide_id.duplicated().any():
        raise ValueError("invalid E6 tissue annotation")
    tissue_by_slide = annotation.set_index("slide_id").tissue_type.to_dict()
    tissue_sizes = annotation.groupby("tissue_type").size().to_dict()
    contract = json.loads(Path(args.contract).read_text())
    radius_rows = []
    content_rows = []
    for model in contract["models"]:
        model_id = model["encoder_id"]
        image_paths = sorted((Path(args.image) / model_id / "shards").glob("*.h5"))
        feature_paths = sorted((Path(args.feature) / model_id / "shards").glob("*.h5"))
        if len(image_paths) != 109 or [p.name for p in image_paths] != [p.name for p in feature_paths]:
            raise ValueError(f"{model_id}: incomplete E5 input population")
        for slide_index, (image_path, feature_path) in enumerate(zip(image_paths, feature_paths)):
            slide_id = image_path.stem
            raw_path = Path(args.raw) / model_id / "shards" / image_path.name
            with h5py.File(raw_path, "r") as source:
                raw = np.asarray(source["features"][:], dtype=np.float32)
                replicate_id = np.asarray(source["replicate_id"][:], dtype=np.int8)
                location_id = np.asarray(source["location_id"][:])
            with h5py.File(image_path, "r") as source:
                image = np.asarray(source["features"][:], dtype=np.float32)
                image_location = source["location_id"][:]
            with h5py.File(feature_path, "r") as source:
                feature = np.asarray(source["features"][:], dtype=np.float32)
                feature_location = source["location_id"][:]
            dimension = int(model["feature_dim"])
            if raw.shape != (6, 100, dimension) or image.shape != (3, 6, 100, dimension) or feature.shape != (2, 6, 100, dimension):
                raise ValueError(f"{model_id}/{slide_id}: feature shape mismatch")
            if not np.array_equal(location_id, image_location) or not np.array_equal(location_id, feature_location):
                raise ValueError(f"{model_id}/{slide_id}: location identity mismatch")
            replicate_counts = np.bincount(replicate_id, minlength=5)
            if not np.array_equal(replicate_counts, np.full(5, 20)):
                raise ValueError(f"{model_id}/{slide_id}: replicate partition mismatch")
            method_features = {
                **{condition: image[index] for index, condition in enumerate(IMAGE_CONDITIONS)},
                **{condition: feature[index] for index, condition in enumerate(FEATURE_CONDITIONS)},
            }
            raw_unit = l2_normalize(raw)
            raw_radius = location_radius(raw_unit)
            raw_at2 = raw_unit[0]
            q95 = unmatched_q95(raw_at2)
            raw_content = np.sum(raw_unit[1:] * raw_at2[None], axis=-1) - q95[None]
            tissue = tissue_by_slide[slide_id]
            for method in METHODS:
                corrected_unit = l2_normalize(method_features[method])
                radius_delta = raw_radius - location_radius(corrected_unit)
                corrected_content = np.sum(corrected_unit[1:] * raw_at2[None], axis=-1) - q95[None]
                content_delta = corrected_content - raw_content
                for replicate in range(5):
                    selected = replicate_id == replicate
                    radius_rows.append({
                        "encoder_id": model_id,
                        "method": method,
                        "tissue_type": tissue,
                        "slides_in_tissue": int(tissue_sizes[tissue]),
                        "slide_id": slide_id,
                        "replicate": replicate,
                        "locations": int(selected.sum()),
                        "delta_radius": float(radius_delta[selected].mean()),
                    })
                    for scanner_index, scanner in enumerate(SCANNERS[1:]):
                        content_rows.append({
                            "encoder_id": model_id,
                            "method": method,
                            "scanner": scanner,
                            "tissue_type": tissue,
                            "slides_in_tissue": int(tissue_sizes[tissue]),
                            "slide_id": slide_id,
                            "replicate": replicate,
                            "locations": int(selected.sum()),
                            "delta_content_margin": float(content_delta[scanner_index, selected].mean()),
                        })
            print(f"[{slide_index + 1}/109] {model_id} {slide_id}", flush=True)
    radius = pd.DataFrame(radius_rows)
    content = pd.DataFrame(content_rows)
    gate = bool(
        len(radius) == 10_900
        and len(content) == 54_500
        and radius.slide_id.nunique() == content.slide_id.nunique() == 109
        and radius.tissue_type.nunique() == content.tissue_type.nunique() == 37
        and set(radius.replicate) == set(content.replicate) == set(range(5))
        and np.isfinite(radius.delta_radius).all()
        and np.isfinite(content.delta_content_margin).all()
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    radius_path = output / "radius_replicate_contrasts.csv"
    content_path = output / "content_replicate_contrasts.csv"
    radius.to_csv(radius_path, index=False)
    content.to_csv(content_path, index=False)
    summary = {
        "analysis": "e6_loso_replicate_contrasts",
        "models": 4,
        "methods": list(METHODS),
        "source_scanners": list(SCANNERS[1:]),
        "slides": 109,
        "tissue_types": 37,
        "replicate_groups_per_slide": 5,
        "locations_per_replicate": 20,
        "radius_rows": len(radius),
        "content_rows": len(content),
        "radius_sha256": sha256(radius_path),
        "content_sha256": sha256(content_path),
        "e5_result_lock_sha256": sha256(lock_path),
        "e6_contract_sha256": sha256(e6_contract),
        "contrast_gate_pass": gate,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

