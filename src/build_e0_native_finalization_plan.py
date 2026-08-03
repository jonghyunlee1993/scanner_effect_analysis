"""Build explicit manifest and scanner-cell plans for final E0 composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from merge_e0_native_geometry_cohort import SCANNERS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route-plan", default="outputs/e0_candidate_execution_plan/route_plan.csv"
    )
    parser.add_argument(
        "--candidate-slides",
        default="outputs/e0_candidate_execution_plan/candidate_slides.csv",
    )
    parser.add_argument(
        "--preserved-root", default="outputs/e0_rigid_native_fallback"
    )
    parser.add_argument(
        "--fromscratch-root", default="outputs/e0_valis_from_scratch_native_full"
    )
    parser.add_argument("--output", default="outputs/e0_native_finalization_plan")
    return parser.parse_args()


def truth(value) -> bool:
    return str(value).strip().lower() == "true"


def build_plans(
    routes: pd.DataFrame,
    candidates: pd.DataFrame,
    preserved_root: Path,
    fromscratch_root: Path,
):
    candidate_ids = set(candidates["slide_id"].astype(str))
    manifest_rows = []
    cell_rows = []
    for candidate in candidates.itertuples(index=False):
        slide_id = str(candidate.slide_id)
        if not truth(candidate.accepted_existing):
            raise ValueError(f"{slide_id}: candidate trial is not accepted")
        trial_root = Path(candidate.trial_root)
        accepted = trial_root / "accepted"
        summary_path = accepted / "summary.json"
        if not summary_path.exists() or not truth(
            json.loads(summary_path.read_text()).get("trial_gate_pass")
        ):
            raise ValueError(f"{slide_id}: accepted candidate summary is missing or failed")
        manifest_rows.append(
            {
                "slide_id": slide_id,
                "manifest_path": str(Path(candidate.manifest_path)),
                "manifest_route_label": trial_root.name,
            }
        )
        slide_routes = {
            str(row.scanner): str(row.route_kind)
            for row in routes[routes["slide_id"].astype(str).eq(slide_id)].itertuples(
                index=False
            )
        }
        for scanner in SCANNERS:
            cell_rows.append(
                {
                    "slide_id": slide_id,
                    "scanner": scanner,
                    "location_path": str(accepted / "native_geometry_manifest.csv"),
                    "cell_path": str(accepted / "cell_qc.csv"),
                    "geometry_route_label": (
                        f"candidate:{trial_root.name}:"
                        f"{slide_routes.get(scanner, 'primary')}"
                    ),
                }
            )

    for route in routes.itertuples(index=False):
        slide_id, scanner = str(route.slide_id), str(route.scanner)
        if slide_id in candidate_ids:
            continue
        route_kind = str(route.route_kind)
        if route_kind == "preserved":
            root = preserved_root
        elif route_kind == "fromscratch":
            root = fromscratch_root
        else:
            raise ValueError(
                f"{slide_id}/{scanner}: noncandidate final route is {route_kind}"
            )
        shard = root / scanner / "shards" / slide_id
        cell_rows.append(
            {
                "slide_id": slide_id,
                "scanner": scanner,
                "location_path": str(shard / "native_geometry_locations.csv"),
                "cell_path": str(shard / "native_geometry_cells.csv"),
                "geometry_route_label": route_kind,
            }
        )
    manifest_plan = pd.DataFrame(manifest_rows)
    cell_plan = pd.DataFrame(cell_rows)
    if not cell_plan.empty and cell_plan.duplicated(["slide_id", "scanner"]).any():
        raise ValueError("finalization cell plan contains duplicate slide/scanner keys")
    return manifest_plan, cell_plan


def validate_paths(manifest_plan: pd.DataFrame, cell_plan: pd.DataFrame):
    missing = []
    for column, frame in (
        ("manifest_path", manifest_plan),
        ("location_path", cell_plan),
        ("cell_path", cell_plan),
    ):
        if column not in frame:
            continue
        missing.extend(str(path) for path in frame[column] if not Path(path).exists())
    if missing:
        raise FileNotFoundError("missing finalization artifacts: " + ";".join(missing))


def main():
    args = parse_args()
    routes = pd.read_csv(args.route_plan, dtype={"slide_id": str})
    candidates = pd.read_csv(args.candidate_slides, dtype={"slide_id": str})
    manifest, cells = build_plans(
        routes,
        candidates,
        Path(args.preserved_root),
        Path(args.fromscratch_root),
    )
    validate_paths(manifest, cells)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / "manifest_plan.csv", index=False)
    cells.to_csv(output / "cell_plan.csv", index=False)
    summary = {
        "analysis": "e0_native_finalization_plan",
        "candidate_manifest_slides": int(len(manifest)),
        "geometry_cell_overrides": int(len(cells)),
        "selection_uses": (
            "registration geometry only; no PFM, correction, or tissue outcome"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
