"""Audit a candidate trial after composing its prespecified scanner routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from merge_e0_native_geometry_cohort import SCANNERS, audit_merged


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide-id", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        metavar="SCANNER=ROOT",
        help="Override a primary scanner with ROOT/shards/<slide>.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def parse_routes(primary: str, overrides: list[str]) -> dict[str, Path]:
    routes = {scanner: Path(primary) for scanner in SCANNERS}
    seen = set()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"route must be SCANNER=ROOT: {item}")
        scanner, root = item.split("=", 1)
        scanner = scanner.strip()
        if scanner not in SCANNERS:
            raise ValueError(f"unsupported scanner route: {scanner}")
        if scanner in seen:
            raise ValueError(f"duplicate scanner route: {scanner}")
        if not root.strip():
            raise ValueError(f"empty route root: {item}")
        seen.add(scanner)
        routes[scanner] = Path(root)
    return routes


def read_route(root: Path, slide_id: str, scanner: str):
    shard = root / "shards" / slide_id
    location_path = shard / "native_geometry_locations.csv"
    cell_path = shard / "native_geometry_cells.csv"
    if not location_path.exists() or not cell_path.exists():
        raise FileNotFoundError(f"missing route shard for {slide_id}/{scanner}: {shard}")
    locations = pd.read_csv(location_path, dtype={"slide_id": str})
    cells = pd.read_csv(cell_path, dtype={"slide_id": str})
    locations = locations[
        locations["slide_id"].eq(slide_id) & locations["scanner"].eq(scanner)
    ].copy()
    cells = cells[cells["slide_id"].eq(slide_id) & cells["scanner"].eq(scanner)].copy()
    return locations, cells


def validate_route(locations, cells, manifest, slide_id, scanner, root):
    expected_ids = set(manifest["location_id"].astype(int))
    observed_ids = set(locations["location_id"].astype(int))
    errors = []
    if len(locations) != 100 or locations["location_id"].nunique() != 100:
        errors.append("invalid_location_keys")
    if observed_ids != expected_ids:
        errors.append("location_ids_do_not_match_trial")
    if observed_ids == expected_ids and len(locations) == 100:
        expected = manifest[
            ["location_id", "replicate_id", "center_x", "center_y"]
        ].rename(
            columns={
                "replicate_id": "expected_replicate_id",
                "center_x": "expected_center_x",
                "center_y": "expected_center_y",
            }
        )
        observed = locations[
            ["location_id", "replicate_id", "canonical_center_x", "canonical_center_y"]
        ]
        coordinate_audit = expected.merge(
            observed, on="location_id", how="outer", validate="one_to_one"
        )
        coordinates_match = bool(
            coordinate_audit["expected_replicate_id"]
            .astype(int)
            .eq(coordinate_audit["replicate_id"].astype(int))
            .all()
            and coordinate_audit["expected_center_x"]
            .astype(int)
            .eq(coordinate_audit["canonical_center_x"].astype(int))
            .all()
            and coordinate_audit["expected_center_y"]
            .astype(int)
            .eq(coordinate_audit["canonical_center_y"].astype(int))
            .all()
        )
        if not coordinates_match:
            errors.append("canonical_centers_do_not_match_trial")
    if len(cells) != 1:
        errors.append("invalid_cell_key")
    if len(locations) and not as_bool(locations["geometry_pass"]).all():
        errors.append("location_geometry_failure")
    if len(cells) == 1 and not as_bool(cells["cell_pass"]).iloc[0]:
        errors.append("cell_gate_failure")
    if len(locations) and set(locations["pixel_source"].astype(str)) != {"native_wsi"}:
        errors.append("non_native_pixel_source")
    if len(locations) and "historical_pixels_primary" in locations:
        if as_bool(locations["historical_pixels_primary"]).any():
            errors.append("historical_pixels_marked_primary")
    return {
        "slide_id": slide_id,
        "scanner": scanner,
        "route_root": str(root.resolve()),
        "locations_observed": int(len(locations)),
        "locations_passing": int(as_bool(locations["geometry_pass"]).sum())
        if len(locations)
        else 0,
        "cell_pass": bool(len(cells) == 1 and as_bool(cells["cell_pass"]).iloc[0]),
        "route_pass": not errors,
        "failure_reason": ";".join(errors) if errors else "pass",
    }


def compose_routes(manifest, slide_id: str, routes: dict[str, Path]):
    location_frames = []
    cell_frames = []
    audits = []
    for scanner in SCANNERS:
        root = routes[scanner]
        locations, cells = read_route(root, slide_id, scanner)
        audit = validate_route(locations, cells, manifest, slide_id, scanner, root)
        locations["geometry_route_root"] = str(root.resolve())
        cells["geometry_route_root"] = str(root.resolve())
        location_frames.append(locations)
        cell_frames.append(cells)
        audits.append(audit)
    return (
        pd.concat(location_frames, ignore_index=True),
        pd.concat(cell_frames, ignore_index=True),
        pd.DataFrame(audits),
    )


def main():
    args = parse_args()
    slide_id = str(args.slide_id)
    manifest = pd.read_csv(args.manifest, dtype={"slide_id": str})
    manifest = manifest[manifest["slide_id"].eq(slide_id)].copy()
    if len(manifest) != 100 or manifest["location_id"].nunique() != 100:
        raise ValueError(f"{slide_id}: trial manifest must contain 100 unique locations")
    routes = parse_routes(args.primary, args.route)
    locations, cells, route_audit = compose_routes(manifest, slide_id, routes)
    locations = locations.sort_values(["location_id", "scanner"]).reset_index(drop=True)
    cells = cells.sort_values("scanner").reset_index(drop=True)
    locations, tuples, cohort_summary = audit_merged(
        manifest, locations, cells, pd.DataFrame(), pd.DataFrame()
    )
    route_gate = bool(route_audit["route_pass"].all())
    trial_gate = bool(route_gate and cohort_summary["cohort_gate_pass"])
    summary = {
        "analysis": "e0_native_candidate_trial_route_audit",
        "slide_id": slide_id,
        "locations": 100,
        "scanner_routes": 6,
        "route_gate_pass": route_gate,
        "six_scanner_tuples_passing": int(cohort_summary["six_scanner_tuples_passing"]),
        "trial_gate_pass": trial_gate,
        "pixel_source": "native_wsi_only",
        "selection_uses": "registration geometry only; no PFM, correction, or tissue outcome",
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    locations.to_csv(output / "native_geometry_manifest.csv", index=False)
    cells.to_csv(output / "cell_qc.csv", index=False)
    tuples.to_csv(output / "tuple_qc.csv", index=False)
    route_audit.to_csv(output / "route_audit.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(route_audit.to_string(index=False))
    print(json.dumps(summary, indent=2))
    if not trial_gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
