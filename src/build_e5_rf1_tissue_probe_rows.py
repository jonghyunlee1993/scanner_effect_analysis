"""Materialize grouped tissue-probe rows for the post-core E5-RF1 condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from build_e7_tissue_probe_rows import (
    heldout_centroids,
    load_raw_at2,
    summarize_at2,
    summarize_queries,
    tissue_reference_sums,
)
from e5_comparator_population import SCANNERS
from e5_reinhard_residual_frequency import RF1_CONDITION, RF1_VERSION
from e6_loto_population import load_tissue_annotation
from extract_e0_pfm_features import select_model
from fetch_e0_pfm_checkpoints import sha256


CONDITIONS = ("raw", RF1_CONDITION)


def parse_args():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--encoder-id")
    selection.add_argument("--encoder-index", type=int)
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--features", default="outputs/e5_rf1_features")
    parser.add_argument("--feature-audit", default="outputs/e5_rf1_features/audit/summary.json")
    parser.add_argument("--geometry", default="outputs/e0_native_geometry_final/native_geometry_manifest.csv")
    parser.add_argument("--rf1-contract", default="docs/e5_reinhard_residual_frequency_contract.md")
    parser.add_argument("--e7-contract", default="docs/e7_tissue_probe_execution_contract.md")
    parser.add_argument("--output", default="outputs/e5_rf1_tissue_probe_rows")
    return parser.parse_args()


def load_queries(raw_path: Path, feature_path: Path, feature_dim: int):
    with h5py.File(raw_path, "r") as source:
        raw = np.asarray(source["features"][:], dtype=np.float32)
        raw_locations = source["location_id"][:]
    with h5py.File(feature_path, "r") as source:
        hybrid = np.asarray(source["features"][:], dtype=np.float32)
        feature_locations = source["location_id"][:]
        conditions = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in source["condition"][:]
        ]
    if (
        raw.shape != (6, 100, feature_dim)
        or hybrid.shape != (1, 6, 100, feature_dim)
        or conditions != [RF1_CONDITION]
        or not np.array_equal(raw_locations, feature_locations)
        or not np.array_equal(raw[0], hybrid[0, 0])
    ):
        raise ValueError(f"{raw_path}: RF1 tissue-query identity/schema mismatch")
    return np.concatenate((raw[None], hybrid), axis=0)


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E5-RF1 tissue-probe row construction requires CUDA")
    feature_audit_path = Path(args.feature_audit)
    feature_audit = json.loads(feature_audit_path.read_text())
    if not (
        feature_audit.get("analysis") == "e5_rf1_feature_population_audit"
        and feature_audit.get("features_observed") == 261_600
        and feature_audit.get("audit_pass") is True
    ):
        raise RuntimeError("E5-RF1 feature population has not passed")
    rf1_contract_path = Path(args.rf1_contract)
    e7_contract_path = Path(args.e7_contract)
    if "**Status:** PRE-OUTCOME FROZEN" not in rf1_contract_path.read_text():
        raise RuntimeError("E5-RF1 contract is not frozen")
    if "**Status:** FROZEN" not in e7_contract_path.read_text():
        raise RuntimeError("E7 tissue-probe contract is not frozen")

    contract = json.loads(Path(args.contract).read_text())
    model = select_model(contract, args.encoder_id, args.encoder_index)
    model_id = model["encoder_id"]
    feature_dim = int(model["feature_dim"])
    tissue_by_slide, tissues = load_tissue_annotation(Path(args.geometry))
    tissue_sizes = {
        tissue: sum(value == tissue for value in tissue_by_slide.values())
        for tissue in tissues
    }
    eligible_tissues = sorted(tissue for tissue, size in tissue_sizes.items() if size >= 2)
    singleton_tissues = sorted(tissue for tissue, size in tissue_sizes.items() if size == 1)
    if len(eligible_tissues) != 36 or len(singleton_tissues) != 1:
        raise ValueError("RF1 evaluable tissue population mismatch")

    raw_paths = sorted((Path(args.raw) / model_id / "shards").glob("*.h5"))
    if len(raw_paths) != 109 or {path.stem for path in raw_paths} != set(tissue_by_slide):
        raise ValueError(f"{model_id}: raw slide population mismatch")
    raw_at2 = load_raw_at2(raw_paths, feature_dim)
    reference_sums = tissue_reference_sums(raw_at2, tissue_by_slide, eligible_tissues)
    evaluable_ids = [
        path.stem for path in raw_paths if tissue_by_slide[path.stem] in eligible_tissues
    ]
    source_rows = []
    at2_rows = []
    for slide_index, slide_id in enumerate(evaluable_ids):
        tissue = tissue_by_slide[slide_id]
        centroids, target_index = heldout_centroids(
            slide_id,
            raw_at2,
            tissue_by_slide,
            eligible_tissues,
            reference_sums,
        )
        at2_rows.append(
            {
                "encoder_id": model_id,
                "slide_id": slide_id,
                "tissue_type": tissue,
                "slides_in_tissue": tissue_sizes[tissue],
                "centroid_classes": len(eligible_tissues),
                **summarize_at2(raw_at2[slide_id], centroids, target_index),
            }
        )
        raw_path = Path(args.raw) / model_id / "shards" / f"{slide_id}.h5"
        feature_path = Path(args.features) / model_id / "shards" / f"{slide_id}.h5"
        metrics = summarize_queries(
            load_queries(raw_path, feature_path, feature_dim),
            centroids,
            target_index,
            raw_at2[slide_id],
        )
        for condition_index, condition in enumerate(CONDITIONS):
            for scanner_index, scanner in enumerate(SCANNERS[1:]):
                source_rows.append(
                    {
                        "encoder_id": model_id,
                        "condition": condition,
                        "scanner": scanner,
                        "slide_id": slide_id,
                        "tissue_type": tissue,
                        "slides_in_tissue": tissue_sizes[tissue],
                        "centroid_classes": len(eligible_tissues),
                        **{
                            name: float(values[condition_index, scanner_index])
                            for name, values in metrics.items()
                        },
                    }
                )
        print(f"[{slide_index + 1}/{len(evaluable_ids)}] {model_id} {slide_id}", flush=True)

    source = pd.DataFrame(source_rows)
    at2 = pd.DataFrame(at2_rows)
    if (
        len(source) != 1_080
        or len(at2) != 108
        or source["tissue_type"].nunique() != 36
        or at2["tissue_type"].nunique() != 36
        or not np.isfinite(source.select_dtypes(include=[np.number])).all().all()
        or not np.isfinite(at2.select_dtypes(include=[np.number])).all().all()
    ):
        raise ValueError(f"{model_id}: RF1 tissue row population gate failed")

    output = Path(args.output) / model_id
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / "slide_source_probe.csv"
    at2_path = output / "slide_at2_probe.csv"
    source.to_csv(source_path, index=False)
    at2.to_csv(at2_path, index=False)
    summary = {
        "analysis": "e5_rf1_grouped_tissue_probe_rows",
        "rf1_version": RF1_VERSION,
        "encoder_id": model_id,
        "checkpoint_sha256": model["checkpoint_sha256"],
        "slides": 108,
        "evaluable_tissues": 36,
        "excluded_singleton_tissue": singleton_tissues[0],
        "centroid_reference": "raw AT2 training slides only",
        "conditions": list(CONDITIONS),
        "source_scanners": list(SCANNERS[1:]),
        "source_rows": len(source),
        "at2_rows": len(at2),
        "source_sha256": sha256(source_path),
        "at2_sha256": sha256(at2_path),
        "feature_population_audit_sha256": sha256(feature_audit_path),
        "rf1_contract_sha256": sha256(rf1_contract_path),
        "e7_contract_sha256": sha256(e7_contract_path),
        "device": torch.cuda.get_device_name(0),
        "row_gate_pass": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
