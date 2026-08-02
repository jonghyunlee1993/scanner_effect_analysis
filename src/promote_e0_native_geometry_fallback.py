"""Promote passing rigid native fallbacks into the E0 cohort manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from merge_e0_native_geometry_cohort import audit_merged


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="outputs/e0_feature_manifest_109/feature_manifest.csv"
    )
    parser.add_argument(
        "--primary", default="outputs/e0_native_geometry_cohort/merged"
    )
    parser.add_argument("--fallback", default="outputs/e0_rigid_native_fallback")
    parser.add_argument("--output", default="outputs/e0_native_geometry_final")
    return parser.parse_args()


def as_bool(series):
    return series.astype(str).str.strip().str.lower().eq("true")


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"slide_id": str})
    primary_root = Path(args.primary)
    fallback_root = Path(args.fallback)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    locations = pd.read_csv(
        primary_root / "native_geometry_manifest.csv", dtype={"slide_id": str}
    )
    cells = pd.read_csv(primary_root / "cell_qc.csv", dtype={"slide_id": str})
    locations["fallback_used"] = False
    cells["fallback_used"] = False

    failed_cells = cells[~as_bool(cells["cell_pass"])][["slide_id", "scanner"]]
    audit_rows = []
    for failed in failed_cells.itertuples(index=False):
        slide_id, scanner = str(failed.slide_id), str(failed.scanner)
        shard = fallback_root / scanner / "shards" / slide_id
        location_path = shard / "native_geometry_locations.csv"
        cell_path = shard / "native_geometry_cells.csv"
        if not location_path.exists() or not cell_path.exists():
            audit_rows.append(
                {
                    "slide_id": slide_id,
                    "scanner": scanner,
                    "fallback_available": False,
                    "fallback_pass": False,
                    "promoted": False,
                    "reason": "missing_fallback_shard",
                }
            )
            continue
        fallback_locations = pd.read_csv(location_path, dtype={"slide_id": str})
        fallback_cell = pd.read_csv(cell_path, dtype={"slide_id": str})
        valid_keys = bool(
            len(fallback_locations) == 100
            and fallback_locations["location_id"].nunique() == 100
            and len(fallback_cell) == 1
            and str(fallback_cell.iloc[0]["slide_id"]) == slide_id
            and str(fallback_cell.iloc[0]["scanner"]) == scanner
        )
        fallback_pass = bool(
            valid_keys
            and as_bool(fallback_cell["cell_pass"]).iloc[0]
            and as_bool(fallback_locations["geometry_pass"]).all()
        )
        if fallback_pass:
            location_mask = locations["slide_id"].eq(slide_id) & locations[
                "scanner"
            ].eq(scanner)
            cell_mask = cells["slide_id"].eq(slide_id) & cells["scanner"].eq(scanner)
            if int(location_mask.sum()) != 100 or int(cell_mask.sum()) != 1:
                raise ValueError(f"primary key mismatch: {slide_id}/{scanner}")
            fallback_locations = fallback_locations.copy()
            fallback_locations["fallback_used"] = True
            fallback_cell = fallback_cell.copy()
            fallback_cell["fallback_used"] = True
            locations = pd.concat(
                [locations.loc[~location_mask], fallback_locations], ignore_index=True
            )
            cells = pd.concat([cells.loc[~cell_mask], fallback_cell], ignore_index=True)
        audit_rows.append(
            {
                "slide_id": slide_id,
                "scanner": scanner,
                "fallback_available": True,
                "fallback_pass": fallback_pass,
                "promoted": fallback_pass,
                "reason": "promoted"
                if fallback_pass
                else "fallback_gate_failure_or_invalid_keys",
            }
        )

    locations = locations.sort_values(
        ["slide_id", "location_id", "scanner"]
    ).reset_index(drop=True)
    cells = cells.sort_values(["slide_id", "scanner"]).reset_index(drop=True)
    locations, tuples, summary = audit_merged(
        manifest, locations, cells, pd.DataFrame(), pd.DataFrame()
    )
    audit = pd.DataFrame(audit_rows)
    summary.update(
        {
            "analysis": "e0_native_geometry_fallback_promotion",
            "primary_failed_cells": int(len(failed_cells)),
            "fallback_cells_available": int(audit["fallback_available"].sum())
            if len(audit)
            else 0,
            "fallback_cells_promoted": int(audit["promoted"].sum())
            if len(audit)
            else 0,
            "fallback_cells_unresolved": int((~audit["promoted"]).sum())
            if len(audit)
            else 0,
        }
    )
    locations.to_csv(output / "native_geometry_manifest.csv", index=False)
    cells.to_csv(output / "cell_qc.csv", index=False)
    tuples.to_csv(output / "tuple_qc.csv", index=False)
    audit.to_csv(output / "fallback_audit.csv", index=False)
    if len(audit):
        audit[~audit["promoted"]].to_csv(output / "unresolved_cells.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
