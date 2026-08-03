"""Audit every materialized candidate trial under its explicit scanner route plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from audit_e0_native_candidate_trial import compose_routes
from merge_e0_native_geometry_cohort import SCANNERS, audit_merged


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-slides",
        default="outputs/e0_candidate_execution_plan/candidate_slides.csv",
    )
    parser.add_argument(
        "--route-plan", default="outputs/e0_candidate_execution_plan/route_plan.csv"
    )
    parser.add_argument(
        "--output", default="outputs/e0_candidate_execution_plan/audit"
    )
    return parser.parse_args()


def route_root(trial_root: Path, route_kind: str, scanner: str) -> Path:
    if route_kind == "primary":
        return trial_root / "primary"
    if route_kind in {"preserved", "fromscratch"}:
        return trial_root / "fallback" / route_kind / scanner
    raise ValueError(f"route is not auditable: {route_kind}")


def audit_trial(candidate, routes: pd.DataFrame):
    slide_id = str(candidate.slide_id)
    trial_root = Path(candidate.trial_root)
    manifest = pd.read_csv(candidate.manifest_path, dtype={"slide_id": str})
    manifest = manifest[manifest["slide_id"].eq(slide_id)].copy()
    scanner_routes = {scanner: trial_root / "primary" for scanner in SCANNERS}
    scanner_route_kinds = {scanner: "primary" for scanner in SCANNERS}
    route_rows = routes[routes["slide_id"].astype(str).eq(slide_id)]
    for row in route_rows.itertuples(index=False):
        scanner_routes[str(row.scanner)] = route_root(
            trial_root, str(row.route_kind), str(row.scanner)
        )
        scanner_route_kinds[str(row.scanner)] = str(row.route_kind)
    locations, cells, route_audit = compose_routes(
        manifest, slide_id, scanner_routes
    )
    locations = locations.sort_values(["location_id", "scanner"]).reset_index(drop=True)
    cells = cells.sort_values("scanner").reset_index(drop=True)
    locations, tuples, cohort = audit_merged(
        manifest, locations, cells, pd.DataFrame(), pd.DataFrame()
    )
    trial_pass = bool(route_audit["route_pass"].all() and cohort["cohort_gate_pass"])
    failed = locations[
        ~locations["geometry_pass"].astype(str).str.lower().eq("true")
    ]
    failed_ids = sorted(
        int(value) for value in failed["location_id"].astype(int).unique()
    )
    accepted = trial_root / "accepted"
    accepted.mkdir(parents=True, exist_ok=True)
    locations.to_csv(accepted / "native_geometry_manifest.csv", index=False)
    cells.to_csv(accepted / "cell_qc.csv", index=False)
    tuples.to_csv(accepted / "tuple_qc.csv", index=False)
    route_audit.to_csv(accepted / "route_audit.csv", index=False)
    summary = {
        "analysis": "e0_native_candidate_trial_route_audit",
        "slide_id": slide_id,
        "locations": 100,
        "scanner_routes": 6,
        "route_kinds": scanner_route_kinds,
        "route_gate_pass": bool(route_audit["route_pass"].all()),
        "six_scanner_tuples_passing": int(cohort["six_scanner_tuples_passing"]),
        "failed_location_ids": failed_ids,
        "trial_gate_pass": trial_pass,
        "pixel_source": "native_wsi_only",
        "selection_uses": "registration geometry only; no PFM, correction, or tissue outcome",
    }
    (accepted / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    args = parse_args()
    candidates = pd.read_csv(args.candidate_slides, dtype={"slide_id": str})
    routes = pd.read_csv(args.route_plan, dtype={"slide_id": str})
    rows = []
    for candidate in candidates.itertuples(index=False):
        slide_id = str(candidate.slide_id)
        if bool(candidate.accepted_existing):
            summary = json.loads(
                (Path(candidate.trial_root) / "accepted" / "summary.json").read_text()
            )
            rows.append(
                {
                    "slide_id": slide_id,
                    "status": "accepted_existing",
                    "trial_gate_pass": bool(summary["trial_gate_pass"]),
                    "failed_location_ids": "",
                    "error": "",
                }
            )
            continue
        try:
            summary = audit_trial(candidate, routes)
            rows.append(
                {
                    "slide_id": slide_id,
                    "status": "audited",
                    "trial_gate_pass": bool(summary["trial_gate_pass"]),
                    "failed_location_ids": ";".join(
                        str(value) for value in summary["failed_location_ids"]
                    ),
                    "error": "",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "slide_id": slide_id,
                    "status": "incomplete_or_error",
                    "trial_gate_pass": False,
                    "failed_location_ids": "",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    audit = pd.DataFrame(rows).sort_values("slide_id").reset_index(drop=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "trial_audit.csv", index=False)
    audit[~audit["trial_gate_pass"]].to_csv(output / "failed_trials.csv", index=False)
    summary = {
        "analysis": "e0_candidate_execution_plan_audit",
        "candidate_slides": int(len(audit)),
        "candidate_slides_passing": int(audit["trial_gate_pass"].sum()),
        "candidate_slides_failing_or_incomplete": int((~audit["trial_gate_pass"]).sum()),
        "all_candidate_slides_pass": bool(audit["trial_gate_pass"].all()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(audit.to_string(index=False))
    print(json.dumps(summary, indent=2))
    if not summary["all_candidate_slides_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
