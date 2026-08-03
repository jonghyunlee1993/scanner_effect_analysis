"""Advance only failed E0 candidate slots to unused reserve ranks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_e0_candidate_execution_plan import (
    fallback_execution_rows,
    parse_location_ids,
)
from build_e0_native_candidate_trial import TRIAL_VERSION, build_trial


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-slides", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--route-plan", required=True)
    parser.add_argument(
        "--only-slides",
        nargs="+",
        help="Advance only this audited subset while retaining the full route plan.",
    )
    parser.add_argument(
        "--manifest", default="outputs/e0_feature_manifest_109/feature_manifest.csv"
    )
    parser.add_argument(
        "--reserve-root", default="outputs/e0_candidate_reserve/shards"
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def as_bool(value) -> bool:
    return str(value).strip().lower() == "true"


def trial_number(trial_root: Path) -> int:
    prefix = "trial_"
    if not trial_root.name.startswith(prefix):
        raise ValueError(f"candidate trial directory is not versioned: {trial_root}")
    return int(trial_root.name[len(prefix) :])


def advance_mapping(mapping: dict[int, int], failed_ids: list[int]):
    """Keep passing slots fixed and give failed slots never-used later ranks."""

    if not failed_ids:
        raise ValueError("cannot advance a trial without failed location IDs")
    if len(failed_ids) != len(set(failed_ids)):
        raise ValueError("failed location IDs must be unique")
    advanced = {int(location): int(rank) for location, rank in mapping.items()}
    next_rank = max(advanced.values(), default=-1) + 1
    rejected = {}
    for location_id in sorted(int(value) for value in failed_ids):
        rejected[location_id] = advanced.get(location_id)
        advanced[location_id] = next_rank
        next_rank += 1
    return advanced, rejected


def materialize_advanced_trials(
    candidates: pd.DataFrame,
    audit: pd.DataFrame,
    base_manifest: pd.DataFrame,
    reserve_root: Path,
):
    audit_lookup = {
        str(row.slide_id): row._asdict() for row in audit.itertuples(index=False)
    }
    rows = []
    for candidate in candidates.itertuples(index=False):
        slide_id = str(candidate.slide_id)
        if slide_id not in audit_lookup:
            raise KeyError(f"candidate audit missing slide: {slide_id}")
        result = audit_lookup[slide_id]
        if as_bool(result["trial_gate_pass"]):
            continue
        if str(result["status"]) != "audited":
            raise ValueError(
                f"{slide_id}: cannot advance non-audited trial status "
                f"{result['status']}"
            )
        failed_ids = parse_location_ids(result["failed_location_ids"])
        previous_root = Path(candidate.trial_root)
        previous_summary = json.loads((previous_root / "summary.json").read_text())
        previous_mapping = {
            int(location): int(rank)
            for location, rank in previous_summary["replacement_mapping"].items()
        }
        mapping, rejected = advance_mapping(previous_mapping, failed_ids)
        next_number = trial_number(previous_root) + 1
        trial_root = previous_root.parent / f"trial_{next_number:03d}"
        manifest_path = trial_root / "manifest.csv"
        base = base_manifest[base_manifest["slide_id"].eq(slide_id)].copy()
        reserve_path = reserve_root / slide_id / "reserve_candidates.csv"
        if not reserve_path.exists():
            raise FileNotFoundError(reserve_path)
        reserve = pd.read_csv(reserve_path, dtype={"slide_id": str})
        reserve = reserve[reserve["slide_id"].eq(slide_id)].copy()
        missing_ranks = sorted(
            set(mapping.values()) - set(reserve["reserve_rank"].astype(int))
        )
        if missing_ranks:
            raise ValueError(
                f"{slide_id}: reserve ranks unavailable through {max(missing_ranks)}"
            )
        trial, replacement_audit = build_trial(base, reserve, mapping)
        trial_root.mkdir(parents=True, exist_ok=True)
        trial.to_csv(manifest_path, index=False)
        replacement_audit.to_csv(trial_root / "replacement_audit.csv", index=False)
        summary = {
            "analysis": "e0_native_candidate_trial",
            "trial_version": TRIAL_VERSION,
            "slide_id": slide_id,
            "trial_number": next_number,
            "locations": 100,
            "previous_trial_root": str(previous_root),
            "advanced_failed_location_ids": failed_ids,
            "rejected_reserve_mapping": {
                str(location): rank for location, rank in sorted(rejected.items())
            },
            "replacement_mapping": {
                str(location): rank for location, rank in sorted(mapping.items())
            },
            "reserve_ranks_consumed_through": max(mapping.values()),
            "selection_uses": (
                "registration geometry only; no PFM, correction, or tissue outcome"
            ),
        }
        (trial_root / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        rows.append(
            {
                "slide_id": slide_id,
                "location_ids": ";".join(str(value) for value in sorted(mapping)),
                "replacement_locations": len(mapping),
                "trial_root": str(trial_root),
                "manifest_path": str(manifest_path),
                "accepted_existing": False,
                "needs_execution": True,
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    candidates = pd.read_csv(args.candidate_slides, dtype={"slide_id": str})
    audit = pd.read_csv(args.audit, dtype={"slide_id": str})
    if args.only_slides:
        requested = {str(value) for value in args.only_slides}
        available = set(candidates["slide_id"].astype(str))
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"requested candidate slides are absent: {missing}")
        candidates = candidates[
            candidates["slide_id"].astype(str).isin(requested)
        ].copy()
    routes = pd.read_csv(args.route_plan, dtype={"slide_id": str})
    base = pd.read_csv(args.manifest, dtype={"slide_id": str})
    trials = materialize_advanced_trials(
        candidates, audit, base, Path(args.reserve_root)
    )
    if trials.empty:
        raise ValueError("no failed audited candidate trials to advance")
    fallback = fallback_execution_rows(routes, trials)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    routes.to_csv(output / "route_plan.csv", index=False)
    trials.to_csv(output / "candidate_slides.csv", index=False)
    trials.to_csv(output / "candidate_primary_slides.csv", index=False)
    fallback.to_csv(output / "candidate_fallback_cells.csv", index=False)
    summary = {
        "analysis": "e0_candidate_execution_plan_advance",
        "candidate_slides_advanced": int(len(trials)),
        "replacement_locations": int(trials["replacement_locations"].sum()),
        "fallback_cells_needing_trial_audit": int(len(fallback)),
        "selection_uses": (
            "registration geometry only; no PFM, correction, or tissue outcome"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
