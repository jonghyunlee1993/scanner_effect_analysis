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
        "--output", default="outputs/e0_native_fromscratch_outcomes"
    )
    return parser.parse_args()


def classify_fromscratch(slide_id: str, scanner: str, root: Path):
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
    if len(failed) and bounds_only(failed["failure_reason"]):
        return {
            "action": "candidate_fromscratch_route",
            "reason": "fromscratch_failure_is_strict_bounds_only",
            "failed_location_ids": failed_ids,
        }
    return {
        "action": "unresolved_fromscratch_geometry",
        "reason": "fromscratch_non_bounds_geometry_failure",
        "failed_location_ids": failed_ids,
    }


def main():
    args = parse_args()
    cells = pd.read_csv(args.cells, dtype={"slide_id": str})
    keys = sorted(set(zip(cells["slide_id"], cells["scanner"])))
    rows = []
    for slide_id, scanner in keys:
        rows.append(
            {
                "slide_id": str(slide_id),
                "scanner": str(scanner),
                **classify_fromscratch(
                    str(slide_id), str(scanner), Path(args.fromscratch_root)
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
