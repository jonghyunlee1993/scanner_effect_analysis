"""Classify failed E0 cells under the frozen fallback hierarchy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_e0_rigid_native_fallback import rigid_branch


STRICT_BOUNDS_REASONS = {"target_fov_bounds", "native_fov_bounds"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary-failure-cells",
        default="outputs/e0_native_geometry_cohort/merged/failure_cells.csv",
    )
    parser.add_argument(
        "--primary-failure-locations",
        default="outputs/e0_native_geometry_cohort/merged/failure_locations.csv",
    )
    parser.add_argument("--fallback-root", default="outputs/e0_rigid_native_fallback")
    parser.add_argument(
        "--rigid-root",
        default=(
            "/mnt/isilon/oldridge_lab/batch_effects/pan_normal/"
            "registered_ref_at2_all/registered_images"
        ),
    )
    parser.add_argument("--output", default="outputs/e0_native_fallback_actions")
    return parser.parse_args()


def truth(value) -> bool:
    return str(value).strip().lower() == "true"


def reason_tokens(values) -> set[str]:
    return {
        token
        for value in values
        for token in str(value).split(";")
        if token and token != "pass"
    }


def bounds_only(values) -> bool:
    tokens = reason_tokens(values)
    return bool(tokens and tokens.issubset(STRICT_BOUNDS_REASONS))


def classify_cell(
    slide_id: str,
    scanner: str,
    primary_locations: pd.DataFrame,
    fallback_root: Path,
    rigid_root: Path | None = None,
):
    primary_failed = primary_locations[
        primary_locations["slide_id"].eq(slide_id)
        & primary_locations["scanner"].eq(scanner)
    ]
    if len(primary_failed) and bounds_only(primary_failed["failure_reason"]):
        return {
            "action": "candidate_primary_route",
            "reason": "primary_failure_is_strict_bounds_only",
            "route_after_action": "primary",
            "failed_location_ids": ";".join(
                str(value) for value in sorted(primary_failed["location_id"].astype(int))
            ),
        }

    shard = fallback_root / scanner / "shards" / slide_id
    location_path = shard / "native_geometry_locations.csv"
    cell_path = shard / "native_geometry_cells.csv"
    if not location_path.exists() or not cell_path.exists():
        preserved_path = (
            rigid_root
            / rigid_branch(scanner)
            / scanner
            / f"{slide_id}.ome.tiff"
            if rigid_root is not None
            else None
        )
        if preserved_path is not None and preserved_path.exists():
            return {
                "action": "audit_preserved_rigid",
                "reason": "preserved_rigid_exists_but_native_audit_missing",
                "route_after_action": "preserved_rigid_pending",
                "failed_location_ids": "",
            }
        return {
            "action": "from_scratch_valis",
            "reason": "preserved_rigid_output_missing",
            "route_after_action": "from_scratch_pending",
            "failed_location_ids": "",
        }
    fallback_locations = pd.read_csv(location_path, dtype={"slide_id": str})
    fallback_cells = pd.read_csv(cell_path, dtype={"slide_id": str})
    fallback_locations = fallback_locations[
        fallback_locations["slide_id"].eq(slide_id)
        & fallback_locations["scanner"].eq(scanner)
    ].copy()
    fallback_cells = fallback_cells[
        fallback_cells["slide_id"].eq(slide_id)
        & fallback_cells["scanner"].eq(scanner)
    ].copy()
    valid = bool(
        len(fallback_locations) == 100
        and fallback_locations["location_id"].nunique() == 100
        and len(fallback_cells) == 1
    )
    if not valid:
        return {
            "action": "from_scratch_valis",
            "reason": "preserved_rigid_audit_keys_invalid",
            "route_after_action": "from_scratch_pending",
            "failed_location_ids": "",
        }
    failed = fallback_locations[
        ~fallback_locations["geometry_pass"].map(truth)
    ].copy()
    cell_pass = truth(fallback_cells.iloc[0]["cell_pass"])
    if cell_pass and failed.empty:
        return {
            "action": "promote_preserved_rigid",
            "reason": "preserved_rigid_gate_pass",
            "route_after_action": "preserved_rigid",
            "failed_location_ids": "",
        }
    if len(failed) and bounds_only(failed["failure_reason"]):
        return {
            "action": "candidate_preserved_rigid_route",
            "reason": "preserved_rigid_failure_is_strict_bounds_only",
            "route_after_action": "preserved_rigid",
            "failed_location_ids": ";".join(
                str(value) for value in sorted(failed["location_id"].astype(int))
            ),
        }
    return {
        "action": "from_scratch_valis",
        "reason": "preserved_rigid_geometry_gate_failure",
        "route_after_action": "from_scratch_pending",
        "failed_location_ids": ";".join(
            str(value) for value in sorted(failed["location_id"].astype(int))
        ),
    }


def main():
    args = parse_args()
    failure_cells = pd.read_csv(args.primary_failure_cells, dtype={"slide_id": str})
    failure_locations = pd.read_csv(
        args.primary_failure_locations, dtype={"slide_id": str}
    )
    cells = sorted(set(zip(failure_cells["slide_id"], failure_cells["scanner"])))
    rows = []
    for slide_id, scanner in cells:
        action = classify_cell(
            str(slide_id),
            str(scanner),
            failure_locations,
            Path(args.fallback_root),
            Path(args.rigid_root),
        )
        rows.append({"slide_id": str(slide_id), "scanner": str(scanner), **action})
    actions = pd.DataFrame(rows).sort_values(["slide_id", "scanner"]).reset_index(drop=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    actions.to_csv(output / "actions.csv", index=False)
    for action, filename in (
        ("from_scratch_valis", "from_scratch_cells.csv"),
        ("promote_preserved_rigid", "promoted_cells.csv"),
        ("candidate_primary_route", "candidate_primary_cells.csv"),
        ("candidate_preserved_rigid_route", "candidate_preserved_cells.csv"),
        ("audit_preserved_rigid", "audit_preserved_cells.csv"),
    ):
        actions[actions["action"].eq(action)].to_csv(output / filename, index=False)
    counts = actions["action"].value_counts().sort_index().to_dict()
    summary = {
        "analysis": "e0_native_fallback_action_selection",
        "failed_primary_cells": int(len(actions)),
        "action_counts": {key: int(value) for key, value in counts.items()},
        "strict_bounds_reasons": sorted(STRICT_BOUNDS_REASONS),
        "selection_uses": "registration geometry only; no PFM, correction, or tissue outcome",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(actions.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
