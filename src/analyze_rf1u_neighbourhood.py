"""Scanner composition of each patch's nearest neighbours, before and after correction.

Scanner radius says how far apart the six per-scanner centroids sit. It does not
say what a retrieval actually returns. This asks the question directly: take one
patch, look at its k nearest neighbours inside the same slide, and count how many
come from its own scanner.

Under a perfectly scanner-invariant representation the neighbours are drawn from
all six acquisitions in proportion, so the same-scanner share falls to the chance
level of 99/599. Under a scanner-dominated representation it stays near one. The
same-location share is the complementary content check: the five other scanners
imaged the identical physical spot, so a representation that encodes content
rather than instrument should surface exactly those.

No threshold is applied here and no decision is taken; this is a descriptive
companion to the frozen frontier endpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from analyze_e4_control_frontier import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    bootstrap_indices,
)
from e4_primary_metrics import bootstrap_mean_ci, l2_normalize
from e5_comparator_population import SCANNERS
from extract_rf1u_features import ANALYSIS as SHARD_ANALYSIS
from fetch_e0_pfm_checkpoints import sha256
from rf1u_unpaired import RF1U_TARGETS, RF1U_VERSION


NEIGHBOURS = 10
LOCATIONS = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--features", default="outputs/rf1u_multitarget/features")
    parser.add_argument("--output", default="outputs/rf1u_multitarget/neighbourhood")
    parser.add_argument("--neighbours", type=int, default=NEIGHBOURS)
    return parser.parse_args()


def neighbourhood_composition(features: np.ndarray, k: int) -> dict[str, float]:
    """Same-scanner and same-location share of each patch's k nearest neighbours.

    `features` is scanners x locations x dim. Similarity is cosine on unit
    vectors, the same geometry the frozen endpoints use. A patch never counts
    itself as its own neighbour.
    """
    unit = l2_normalize(features)
    n_scanners, n_locations = unit.shape[0], unit.shape[1]
    total = n_scanners * n_locations
    if k < 1 or k >= total:
        raise ValueError("neighbour count must lie inside the pool")
    flat = unit.reshape(total, -1)
    scanner_of = np.repeat(np.arange(n_scanners), n_locations)
    location_of = np.tile(np.arange(n_locations), n_scanners)

    similarity = flat @ flat.T
    np.fill_diagonal(similarity, -np.inf)
    top = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]

    same_scanner = scanner_of[top] == scanner_of[:, None]
    same_location = location_of[top] == location_of[:, None]
    distinct = np.asarray(
        [len(np.unique(scanner_of[row])) for row in top], dtype=np.float64
    )
    reachable = min(k, n_scanners - 1)
    return {
        "same_scanner_share": float(same_scanner.mean()),
        "same_location_share": float(same_location.sum(axis=1).mean() / reachable),
        "distinct_scanners": float(distinct.mean()),
        "chance_same_scanner_share": float((n_locations - 1) / (total - 1)),
    }


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads(Path(args.contract).read_text())
    models = contract["models"]
    indices = bootstrap_indices(109)

    slide_rows = []
    for target in RF1U_TARGETS:
        for model in models:
            model_id = model["encoder_id"]
            paths = sorted((Path(args.features) / target / model_id / "shards").glob("*.h5"))
            if len(paths) != 109:
                raise ValueError(f"{target}/{model_id}: expected 109 shards")
            for path in paths:
                summary = json.loads(path.with_suffix(".summary.json").read_text())
                if not (
                    summary.get("analysis") == SHARD_ANALYSIS
                    and summary.get("target") == target
                    and summary.get("shard_gate_pass") is True
                ):
                    raise RuntimeError(f"invalid RF1U shard: {path}")
                with h5py.File(path, "r") as source:
                    rendered = np.asarray(source["features"][:], dtype=np.float32)
                    conditions = [value.decode() for value in source["condition"][:]]
                pairs = list(zip(conditions, rendered))
                if target == RF1U_TARGETS[0]:
                    with h5py.File(
                        Path(args.raw) / model_id / "shards" / path.name, "r"
                    ) as source:
                        pairs.insert(
                            0, ("raw", np.asarray(source["features"][:], dtype=np.float32))
                        )
                for condition, values in pairs:
                    slide_rows.append(
                        {
                            "target": "shared" if condition == "raw" else target,
                            "encoder_id": model_id,
                            "slide_id": path.stem,
                            "condition": condition,
                            **neighbourhood_composition(values, args.neighbours),
                        }
                    )
            print(f"neighbourhood: {target} / {model_id}", flush=True)

    slides = sorted({row["slide_id"] for row in slide_rows})
    rows = []
    keys = ("same_scanner_share", "same_location_share", "distinct_scanners")
    for target in sorted({row["target"] for row in slide_rows}):
        for model in models:
            model_id = model["encoder_id"]
            for condition in sorted(
                {
                    row["condition"]
                    for row in slide_rows
                    if row["target"] == target and row["encoder_id"] == model_id
                }
            ):
                selected = sorted(
                    (
                        row
                        for row in slide_rows
                        if row["target"] == target
                        and row["encoder_id"] == model_id
                        and row["condition"] == condition
                    ),
                    key=lambda row: row["slide_id"],
                )
                if len(selected) != len(slides):
                    raise RuntimeError("incomplete slide population")
                entry = {
                    "target": target,
                    "encoder_id": model_id,
                    "condition": condition,
                    "slides": len(selected),
                    "neighbours": args.neighbours,
                    "chance_same_scanner_share": selected[0]["chance_same_scanner_share"],
                }
                for key in keys:
                    values = np.asarray([row[key] for row in selected])
                    point, lower, upper = bootstrap_mean_ci(values, indices)
                    entry[key] = point
                    entry[f"{key}_ci_lower"] = lower
                    entry[f"{key}_ci_upper"] = upper
                rows.append(entry)

    slide_path = output / "slide_neighbourhood.csv"
    summary_path = output / "neighbourhood_summary.csv"
    for path, data in ((slide_path, slide_rows), (summary_path, rows)):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    result = {
        "analysis": "rf1u_neighbourhood_composition",
        "rf1u_version": RF1U_VERSION,
        "status": "DESCRIPTIVE_COMPANION_NO_THRESHOLD",
        "question": (
            "Of a patch's k nearest neighbours inside its own slide, how many come "
            "from the same scanner, and how many are the identical physical location "
            "seen by a different scanner?"
        ),
        "neighbours": args.neighbours,
        "pool_per_slide": len(SCANNERS) * LOCATIONS,
        "chance_same_scanner_share": rows[0]["chance_same_scanner_share"],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "cells": len(rows),
        "artifacts": {
            path.name: sha256(path) for path in (slide_path, summary_path)
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
