"""Merge E0 native-geometry shards and evaluate the frozen cohort gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SCANNERS = ("at2", "gt450", "versa", "akoya", "s60", "s360")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="outputs/e0_feature_manifest_109/feature_manifest.csv"
    )
    parser.add_argument("--input", default="outputs/e0_native_geometry_cohort")
    parser.add_argument("--output", default="outputs/e0_native_geometry_cohort/merged")
    return parser.parse_args()


def expected_keys(manifest: pd.DataFrame):
    locations = manifest[["slide_id", "location_id"]].drop_duplicates()
    scanners = pd.DataFrame({"scanner": SCANNERS})
    locations["_join"] = 1
    scanners["_join"] = 1
    return locations.merge(scanners, on="_join").drop(columns="_join")


def merge_shards(manifest: pd.DataFrame, root: Path):
    slides = sorted(manifest["slide_id"].astype(str).unique())
    locations = []
    cells = []
    missing = []
    invalid_summaries = []
    for slide_id in slides:
        shard = root / "shards" / slide_id
        location_path = shard / "native_geometry_locations.csv"
        cell_path = shard / "native_geometry_cells.csv"
        summary_path = shard / "summary.json"
        absent = [
            path.name
            for path in (location_path, cell_path, summary_path)
            if not path.exists()
        ]
        if absent:
            missing.append({"slide_id": slide_id, "missing": ";".join(absent)})
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("locations_written") != 600 or summary.get("cells_written") != 6:
            invalid_summaries.append(
                {
                    "slide_id": slide_id,
                    "locations_written": summary.get("locations_written"),
                    "cells_written": summary.get("cells_written"),
                }
            )
        locations.append(pd.read_csv(location_path, dtype={"slide_id": str}))
        cells.append(pd.read_csv(cell_path, dtype={"slide_id": str}))
    return (
        pd.concat(locations, ignore_index=True) if locations else pd.DataFrame(),
        pd.concat(cells, ignore_index=True) if cells else pd.DataFrame(),
        pd.DataFrame(missing),
        pd.DataFrame(invalid_summaries),
    )


def audit_merged(manifest, locations, cells, missing, invalid_summaries):
    expected = expected_keys(manifest)
    key_columns = ["slide_id", "location_id", "scanner"]
    duplicate_location_keys = int(locations.duplicated(key_columns).sum()) if len(locations) else 0
    observed_keys = locations[key_columns].drop_duplicates() if len(locations) else locations
    key_audit = expected.merge(observed_keys, on=key_columns, how="outer", indicator=True)
    missing_location_keys = int(key_audit["_merge"].eq("left_only").sum())
    unexpected_location_keys = int(key_audit["_merge"].eq("right_only").sum())

    cell_key_columns = ["slide_id", "scanner"]
    duplicate_cell_keys = int(cells.duplicated(cell_key_columns).sum()) if len(cells) else 0
    expected_cells = expected[cell_key_columns].drop_duplicates()
    observed_cells = cells[cell_key_columns].drop_duplicates() if len(cells) else cells
    cell_audit = expected_cells.merge(
        observed_cells, on=cell_key_columns, how="outer", indicator=True
    )
    missing_cell_keys = int(cell_audit["_merge"].eq("left_only").sum())
    unexpected_cell_keys = int(cell_audit["_merge"].eq("right_only").sum())

    if len(locations):
        geometry = locations["geometry_pass"].astype(str).str.lower().eq("true")
        locations = locations.assign(_geometry_pass=geometry)
        tuple_qc = (
            locations.groupby(["slide_id", "location_id"], as_index=False)
            .agg(scanners_present=("scanner", "nunique"), scanners_passing=("_geometry_pass", "sum"))
        )
        tuple_qc["tuple_pass"] = tuple_qc["scanners_present"].eq(6) & tuple_qc[
            "scanners_passing"
        ].eq(6)
        passing_locations = int(geometry.sum())
        failing_locations = int((~geometry).sum())
        passing_tuples = int(tuple_qc["tuple_pass"].sum())
    else:
        tuple_qc = pd.DataFrame(
            columns=["slide_id", "location_id", "scanners_present", "scanners_passing", "tuple_pass"]
        )
        passing_locations = failing_locations = passing_tuples = 0

    if len(cells):
        cell_pass = cells["cell_pass"].astype(str).str.lower().eq("true")
        passing_cells = int(cell_pass.sum())
        failing_cells = int((~cell_pass).sum())
    else:
        passing_cells = failing_cells = 0

    expected_location_rows = int(len(expected))
    expected_cells_count = int(len(expected_cells))
    expected_tuples = int(manifest[["slide_id", "location_id"]].drop_duplicates().shape[0])
    cohort_gate = bool(
        missing.empty
        and invalid_summaries.empty
        and len(locations) == expected_location_rows
        and len(cells) == expected_cells_count
        and duplicate_location_keys == 0
        and duplicate_cell_keys == 0
        and missing_location_keys == 0
        and unexpected_location_keys == 0
        and missing_cell_keys == 0
        and unexpected_cell_keys == 0
        and failing_locations == 0
        and failing_cells == 0
        and passing_tuples == expected_tuples
    )
    summary = {
        "analysis": "e0_native_geometry_cohort_merge",
        "manifest_version": "e0_native_geometry_v1",
        "slides_expected": int(manifest["slide_id"].nunique()),
        "slides_missing": int(len(missing)),
        "invalid_shard_summaries": int(len(invalid_summaries)),
        "location_rows_expected": expected_location_rows,
        "location_rows_observed": int(len(locations)),
        "locations_passing": passing_locations,
        "locations_failing": failing_locations,
        "duplicate_location_keys": duplicate_location_keys,
        "missing_location_keys": missing_location_keys,
        "unexpected_location_keys": unexpected_location_keys,
        "cells_expected": expected_cells_count,
        "cells_observed": int(len(cells)),
        "cells_passing": passing_cells,
        "cells_failing": failing_cells,
        "duplicate_cell_keys": duplicate_cell_keys,
        "missing_cell_keys": missing_cell_keys,
        "unexpected_cell_keys": unexpected_cell_keys,
        "six_scanner_tuples_expected": expected_tuples,
        "six_scanner_tuples_passing": passing_tuples,
        "cohort_gate_pass": cohort_gate,
    }
    return locations.drop(columns="_geometry_pass", errors="ignore"), tuple_qc, summary


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"slide_id": str})
    root = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    locations, cells, missing, invalid = merge_shards(manifest, root)
    locations, tuples, summary = audit_merged(
        manifest, locations, cells, missing, invalid
    )
    locations.to_csv(output / "native_geometry_manifest.csv", index=False)
    cells.to_csv(output / "cell_qc.csv", index=False)
    tuples.to_csv(output / "tuple_qc.csv", index=False)
    missing.to_csv(output / "missing_shards.csv", index=False)
    invalid.to_csv(output / "invalid_shards.csv", index=False)
    if len(locations):
        locations[
            ~locations["geometry_pass"].astype(str).str.lower().eq("true")
        ].to_csv(output / "failure_locations.csv", index=False)
    if len(cells):
        cells[~cells["cell_pass"].astype(str).str.lower().eq("true")].to_csv(
            output / "failure_cells.csv", index=False
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
