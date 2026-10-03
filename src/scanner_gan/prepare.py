#!/usr/bin/env python3
"""Freeze the Pix2Pix scanner-to-AT2 data and split contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import (
    PRIMARY_SOURCE_SCANNERS,
    PRIMARY_TARGET_SCANNER,
    assign_split_roles,
    build_locked_evaluation_index,
    build_sample_index,
)


CONTRACT_VERSION = "pix2pix_scanner_to_at2_contract_v1"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare(
    cohort_path: Path,
    folds_path: Path,
    metrics_root: Path,
    output_root: Path,
) -> dict:
    contract_root = output_root / "00_contract"
    contract_root.mkdir(parents=True, exist_ok=True)
    sample_index_path = contract_root / "sample_index.csv.gz"
    sample_index = build_sample_index(cohort_path, folds_path)
    temporary = sample_index_path.with_suffix(sample_index_path.suffix + ".tmp")
    sample_index.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(sample_index_path)

    locked = build_locked_evaluation_index(sample_index, metrics_root)
    locked_path = contract_root / "locked_image_evaluation_index.csv.gz"
    temporary = locked_path.with_suffix(locked_path.suffix + ".tmp")
    locked.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(locked_path)

    split_rows = []
    for test_fold in range(5):
        split = assign_split_roles(sample_index, test_fold)
        by_role = (
            split.groupby("split_role")
            .agg(slides=("slide_id", "nunique"), locations=("location_index", "size"))
            .reset_index()
        )
        for row in by_role.itertuples(index=False):
            split_rows.append(
                {
                    "test_fold": test_fold,
                    "validation_fold": (test_fold + 1) % 5,
                    "split_role": row.split_role,
                    "slides": int(row.slides),
                    "locations": int(row.locations),
                }
            )
    split_summary = pd.DataFrame(split_rows)
    split_summary_path = contract_root / "split_summary.csv"
    split_summary.to_csv(split_summary_path, index=False)

    payload = {
        "contract_version": CONTRACT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "direction": [f"{scanner}->at2" for scanner in PRIMARY_SOURCE_SCANNERS],
        "source_scanners": list(PRIMARY_SOURCE_SCANNERS),
        "target_scanner": PRIMARY_TARGET_SCANNER,
        "slides": int(sample_index["slide_id"].nunique()),
        "locations": int(len(sample_index)),
        "tissues": int(sample_index["tissue_type"].nunique()),
        "locked_image_locations": int(len(locked)),
        "locked_image_locations_per_slide": 40,
        "outer_test_folds": 5,
        "inner_validation_rule": "(outer_test_fold + 1) mod 5",
        "training_folds_per_model": 3,
        "generator_input": "raw source-scanner image only",
        "paired_target": "same-location AT2",
        "alignment": (
            "loss/D only: target_crop_shift_xy=q_AT2-q_source="
            "applied_source-gradient_source; shift is not a generator input"
        ),
        "selection_boundary": (
            "checkpoint selection uses inner-validation image evidence only; "
            "frozen UNI is opened only after image predictions/checkpoints are locked"
        ),
        "primary_learned_endpoint": (
            "paired frozen-UNI gain=(1-cos(source,AT2))-(1-cos(method,AT2))"
        ),
        "estimand_boundary": (
            "AT2-registered paired patch translation; not standalone unregistered WSI deployment"
        ),
        "inputs": {
            "cohort": {"path": str(cohort_path.resolve()), "sha256": sha256(cohort_path)},
            "folds": {"path": str(folds_path.resolve()), "sha256": sha256(folds_path)},
        },
        "outputs": {
            "sample_index": {
                "path": str(sample_index_path.resolve()),
                "sha256": sha256(sample_index_path),
            },
            "locked_image_index": {
                "path": str(locked_path.resolve()),
                "sha256": sha256(locked_path),
            },
            "split_summary": {
                "path": str(split_summary_path.resolve()),
                "sha256": sha256(split_summary_path),
            },
        },
    }
    write_json(contract_root / "analysis_contract.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare(
        args.cohort.resolve(),
        args.folds.resolve(),
        args.metrics_root.resolve(),
        args.output_root.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
