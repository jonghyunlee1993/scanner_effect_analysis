"""Classify completed from-scratch native-rigid cells for promotion or candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from select_e0_native_fallback_actions import bounds_only, truth


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cells", default="outputs/e0_native_fallback_actions/from_scratch_cells.csv"
    )
    parser.add_argument(
        "--fromscratch-root", default="outputs/e0_valis_from_scratch_native_full"
    )
    parser.add_argument(
        "--primary-root", default="outputs/e0_native_geometry_cohort/merged"
    )
    parser.add_argument(
        "--output", default="outputs/e0_native_fromscratch_outcomes"
    )
    return parser.parse_args()


def primary_candidate_if_not_worse(
    slide_id: str,
    scanner: str,
    fromscratch_failures: int,
    primary_locations: pd.DataFrame | None,
    primary_cells: pd.DataFrame | None,
):
    if primary_locations is None or primary_cells is None:
        return None
    locations = primary_locations[
        primary_locations["slide_id"].eq(slide_id)
        & primary_locations["scanner"].eq(scanner)
    ].copy()
    cells = primary_cells[
        primary_cells["slide_id"].eq(slide_id)
        & primary_cells["scanner"].eq(scanner)
    ].copy()
    if len(locations) != 100 or locations["location_id"].nunique() != 100 or len(cells) != 1:
        return None
    if "global_transform_pass" not in cells or not truth(
        cells.iloc[0]["global_transform_pass"]
    ):
        return None
    failed = locations[~locations["geometry_pass"].map(truth)].copy()
    if failed.empty or len(failed) > fromscratch_failures:
        return None
    return {
        "action": "candidate_primary_route",
        "reason": "primary_valid_global_has_no_more_failures_than_fromscratch",
        "failed_location_ids": ";".join(
            str(value) for value in sorted(failed["location_id"].astype(int))
        ),
    }


def classify_fromscratch(
    slide_id: str,
    scanner: str,
    root: Path,
    primary_locations: pd.DataFrame | None = None,
    primary_cells: pd.DataFrame | None = None,
):
    shard = root / scanner / "shards" / slide_id
    location_path = shard / "native_geometry_locations.csv"
    cell_path = shard / "native_geometry_cells.csv"
    if not location_path.exists() or not cell_path.exists():
        return {
            "action": "unresolved_fromscratch_missing",
            "reason": "fromscratch_native_audit_missing",
            "failed_location_ids": "",
        }
    locations = pd.read_csv(location_path, dtype={"slide_id": str})
    cells = pd.read_csv(cell_path, dtype={"slide_id": str})
    locations = locations[
        locations["slide_id"].eq(slide_id) & locations["scanner"].eq(scanner)
    ].copy()
    cells = cells[cells["slide_id"].eq(slide_id) & cells["scanner"].eq(scanner)].copy()
    valid = bool(
        len(locations) == 100
        and locations["location_id"].nunique() == 100
        and len(cells) == 1
    )
    if not valid:
        return {
            "action": "unresolved_fromscratch_invalid",
            "reason": "fromscratch_native_audit_keys_invalid",
            "failed_location_ids": "",
        }
    failed = locations[~locations["geometry_pass"].map(truth)].copy()
    if truth(cells.iloc[0]["cell_pass"]) and failed.empty:
        return {
            "action": "promote_fromscratch_rigid",
            "reason": "fromscratch_native_gate_pass",
            "failed_location_ids": "",
        }
    failed_ids = ";".join(
        str(value) for value in sorted(failed["location_id"].astype(int))
    )
    primary = primary_candidate_if_not_worse(
        slide_id,
        scanner,
        len(failed),
        primary_locations,
        primary_cells,
    )
    if primary is not None:
        return primary
    if len(failed) and bounds_only(failed["failure_reason"]):
        return {
            "action": "candidate_fromscratch_route",
            "reason": "fromscratch_failure_is_strict_bounds_only",
            "failed_location_ids": failed_ids,
        }
    global_transform_pass = (
        truth(cells.iloc[0]["global_transform_pass"])
        if "global_transform_pass" in cells.columns
        else False
    )
    if len(failed) and global_transform_pass:
        return {
            "action": "candidate_fromscratch_route",
            "reason": "fromscratch_location_geometry_failure_with_valid_global_transform",
            "failed_location_ids": failed_ids,
        }
    return {
        "action": "unresolved_fromscratch_geometry",
        "reason": "fromscratch_global_or_unclassified_geometry_failure",
        "failed_location_ids": failed_ids,
    }


def main():
    args = parse_args()
    cells = pd.read_csv(args.cells, dtype={"slide_id": str})
    primary_root = Path(args.primary_root)
    primary_locations = pd.read_csv(
        primary_root / "native_geometry_manifest.csv", dtype={"slide_id": str}
    )
    primary_cells = pd.read_csv(
        primary_root / "cell_qc.csv", dtype={"slide_id": str}
    )
    keys = sorted(set(zip(cells["slide_id"], cells["scanner"])))
    rows = []
    for slide_id, scanner in keys:
        rows.append(
            {
                "slide_id": str(slide_id),
                "scanner": str(scanner),
                **classify_fromscratch(
                    str(slide_id),
                    str(scanner),
                    Path(args.fromscratch_root),
                    primary_locations,
                    primary_cells,
                ),
            }
        )
    outcomes = pd.DataFrame(rows).sort_values(["slide_id", "scanner"]).reset_index(drop=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(output / "outcomes.csv", index=False)
    for action in sorted(outcomes["action"].unique()):
        outcomes[outcomes["action"].eq(action)].to_csv(
            output / f"{action}_cells.csv", index=False
        )
    summary = {
        "analysis": "e0_fromscratch_native_outcome_selection",
        "cells": int(len(outcomes)),
        "action_counts": {
            key: int(value)
            for key, value in outcomes["action"].value_counts().sort_index().items()
        },
        "selection_uses": "registration geometry only; no PFM, correction, or tissue outcome",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(outcomes.to_string(index=False))
    print(json.dumps(summary, indent=2))
    if outcomes["action"].str.startswith("unresolved").any():
        raise SystemExit(2)


if __name__ == "__main__":
    main()
