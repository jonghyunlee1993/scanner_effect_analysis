"""Compose the final frozen location and native-geometry manifests from audited routes."""

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
    parser.add_argument(
        "--manifest-plan",
        help="CSV columns: slide_id, manifest_path, manifest_route_label.",
    )
    parser.add_argument(
        "--cell-plan",
        help=(
            "CSV columns: slide_id, scanner, location_path, cell_path, "
            "geometry_route_label."
        ),
    )
    parser.add_argument("--output", default="outputs/e0_native_geometry_final")
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def replace_manifest_slides(manifest: pd.DataFrame, plan: pd.DataFrame):
    result = manifest.copy()
    audit = []
    if plan.empty:
        return result, pd.DataFrame(audit)
    if plan["slide_id"].astype(str).duplicated().any():
        raise ValueError("manifest plan contains duplicate slide IDs")
    for row in plan.itertuples(index=False):
        slide_id = str(row.slide_id)
        original = result[result["slide_id"].astype(str).eq(slide_id)].copy()
        replacement = pd.read_csv(row.manifest_path, dtype={"slide_id": str})
        replacement = replacement[replacement["slide_id"].eq(slide_id)].copy()
        if (
            len(original) != 100
            or original["location_id"].nunique() != 100
            or len(replacement) != 100
            or replacement["location_id"].nunique() != 100
        ):
            raise ValueError(f"{slide_id}: manifest override must be 100-for-100")
        original_slots = original[["location_id", "replicate_id"]].sort_values(
            "location_id"
        )
        replacement_slots = replacement[
            ["location_id", "replicate_id"]
        ].sort_values("location_id")
        if not original_slots.reset_index(drop=True).astype(int).equals(
            replacement_slots.reset_index(drop=True).astype(int)
        ):
            raise ValueError(f"{slide_id}: manifest override changed slot identities")
        if set(original["tissue_type"].astype(str)) != set(
            replacement["tissue_type"].astype(str)
        ):
            raise ValueError(f"{slide_id}: manifest override changed tissue type")
        replacement["manifest_route_label"] = str(row.manifest_route_label)
        result = pd.concat(
            [result[~result["slide_id"].astype(str).eq(slide_id)], replacement],
            ignore_index=True,
        )
        audit.append(
            {
                "slide_id": slide_id,
                "manifest_path": str(Path(row.manifest_path).resolve()),
                "manifest_route_label": str(row.manifest_route_label),
                "locations_replaced": 100,
            }
        )
    return result.sort_values(["slide_id", "location_id"]).reset_index(drop=True), pd.DataFrame(audit)


def route_matches_manifest(locations: pd.DataFrame, manifest: pd.DataFrame) -> bool:
    expected = manifest[
        ["slide_id", "location_id", "replicate_id", "center_x", "center_y"]
    ].rename(
        columns={
            "replicate_id": "expected_replicate_id",
            "center_x": "expected_center_x",
            "center_y": "expected_center_y",
        }
    )
    observed = locations[
        [
            "slide_id",
            "location_id",
            "replicate_id",
            "canonical_center_x",
            "canonical_center_y",
        ]
    ]
    joined = expected.merge(
        observed, on=["slide_id", "location_id"], how="outer", validate="one_to_one"
    )
    return bool(
        len(joined) == len(expected) == len(observed)
        and joined["expected_replicate_id"].astype(int).eq(joined["replicate_id"].astype(int)).all()
        and joined["expected_center_x"].astype(int).eq(joined["canonical_center_x"].astype(int)).all()
        and joined["expected_center_y"].astype(int).eq(joined["canonical_center_y"].astype(int)).all()
    )


def replace_geometry_cells(locations, cells, manifest, plan):
    result_locations = locations.copy()
    result_cells = cells.copy()
    audit = []
    if plan.empty:
        return result_locations, result_cells, pd.DataFrame(audit)
    keys = ["slide_id", "scanner"]
    if plan.assign(slide_id=plan["slide_id"].astype(str)).duplicated(keys).any():
        raise ValueError("cell plan contains duplicate slide/scanner keys")
    for row in plan.itertuples(index=False):
        slide_id, scanner = str(row.slide_id), str(row.scanner)
        route_locations = pd.read_csv(row.location_path, dtype={"slide_id": str})
        route_cells = pd.read_csv(row.cell_path, dtype={"slide_id": str})
        route_locations = route_locations[
            route_locations["slide_id"].eq(slide_id)
            & route_locations["scanner"].eq(scanner)
        ].copy()
        route_cells = route_cells[
            route_cells["slide_id"].eq(slide_id)
            & route_cells["scanner"].eq(scanner)
        ].copy()
        route_manifest = manifest[manifest["slide_id"].astype(str).eq(slide_id)]
        valid = bool(
            len(route_locations) == 100
            and route_locations["location_id"].nunique() == 100
            and len(route_cells) == 1
            and as_bool(route_locations["geometry_pass"]).all()
            and as_bool(route_cells["cell_pass"]).iloc[0]
            and set(route_locations["pixel_source"].astype(str)) == {"native_wsi"}
            and route_matches_manifest(route_locations, route_manifest)
        )
        if "historical_pixels_primary" in route_locations:
            valid = valid and not as_bool(
                route_locations["historical_pixels_primary"]
            ).any()
        if not valid:
            raise ValueError(f"invalid final route override: {slide_id}/{scanner}")
        location_mask = result_locations["slide_id"].astype(str).eq(slide_id) & result_locations[
            "scanner"
        ].eq(scanner)
        cell_mask = result_cells["slide_id"].astype(str).eq(slide_id) & result_cells[
            "scanner"
        ].eq(scanner)
        if int(location_mask.sum()) != 100 or int(cell_mask.sum()) != 1:
            raise ValueError(f"primary geometry key mismatch: {slide_id}/{scanner}")
        label = str(row.geometry_route_label)
        route_locations["geometry_route_label"] = label
        route_locations["geometry_route_location_path"] = str(
            Path(row.location_path).resolve()
        )
        route_cells["geometry_route_label"] = label
        route_cells["geometry_route_cell_path"] = str(Path(row.cell_path).resolve())
        result_locations = pd.concat(
            [result_locations.loc[~location_mask], route_locations], ignore_index=True
        )
        result_cells = pd.concat(
            [result_cells.loc[~cell_mask], route_cells], ignore_index=True
        )
        audit.append(
            {
                "slide_id": slide_id,
                "scanner": scanner,
                "geometry_route_label": label,
                "location_path": str(Path(row.location_path).resolve()),
                "cell_path": str(Path(row.cell_path).resolve()),
                "locations_replaced": 100,
                "route_pass": True,
            }
        )
    return (
        result_locations.sort_values(["slide_id", "location_id", "scanner"]).reset_index(drop=True),
        result_cells.sort_values(["slide_id", "scanner"]).reset_index(drop=True),
        pd.DataFrame(audit),
    )


def empty_plan(columns):
    return pd.DataFrame(columns=columns)


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"slide_id": str})
    primary = Path(args.primary)
    locations = pd.read_csv(primary / "native_geometry_manifest.csv", dtype={"slide_id": str})
    cells = pd.read_csv(primary / "cell_qc.csv", dtype={"slide_id": str})
    manifest_plan = (
        pd.read_csv(args.manifest_plan, dtype={"slide_id": str})
        if args.manifest_plan
        else empty_plan(["slide_id", "manifest_path", "manifest_route_label"])
    )
    cell_plan = (
        pd.read_csv(args.cell_plan, dtype={"slide_id": str})
        if args.cell_plan
        else empty_plan(
            [
                "slide_id",
                "scanner",
                "location_path",
                "cell_path",
                "geometry_route_label",
            ]
        )
    )
    manifest, manifest_audit = replace_manifest_slides(manifest, manifest_plan)
    locations, cells, route_audit = replace_geometry_cells(
        locations, cells, manifest, cell_plan
    )
    locations, tuples, summary = audit_merged(
        manifest, locations, cells, pd.DataFrame(), pd.DataFrame()
    )
    summary.update(
        {
            "analysis": "e0_native_geometry_finalization",
            "manifest_slide_overrides": int(len(manifest_audit)),
            "geometry_cell_overrides": int(len(route_audit)),
            "pixel_source": "native_wsi_only",
        }
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / "feature_manifest.csv", index=False)
    locations.to_csv(output / "native_geometry_manifest.csv", index=False)
    cells.to_csv(output / "cell_qc.csv", index=False)
    tuples.to_csv(output / "tuple_qc.csv", index=False)
    manifest_audit.to_csv(output / "manifest_override_audit.csv", index=False)
    route_audit.to_csv(output / "geometry_route_audit.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["cohort_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
