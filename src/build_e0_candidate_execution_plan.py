"""Build deterministic candidate manifests and route execution lists from E0 actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_e0_native_candidate_trial import TRIAL_VERSION, build_trial
from merge_e0_native_geometry_cohort import SCANNERS


EXISTING_RIGID_ROOT = (
    "/mnt/isilon/oldridge_lab/batch_effects/pan_normal/"
    "registered_ref_at2_all/registered_images"
)
FROMSCRATCH_RIGID_ROOT = "outputs/e0_valis_from_scratch_v2/registered_images"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actions", default="outputs/e0_native_fallback_actions/actions.csv"
    )
    parser.add_argument(
        "--fromscratch-outcomes",
        default="outputs/e0_native_fromscratch_outcomes_partial/outcomes.csv",
    )
    parser.add_argument(
        "--manifest", default="outputs/e0_feature_manifest_109/feature_manifest.csv"
    )
    parser.add_argument(
        "--reserve-root", default="outputs/e0_candidate_reserve/shards"
    )
    parser.add_argument(
        "--trials-root", default="outputs/e0_native_candidate_trials"
    )
    parser.add_argument("--output", default="outputs/e0_candidate_execution_plan")
    parser.add_argument("--build-trials", action="store_true")
    parser.add_argument(
        "--only-slides",
        nargs="+",
        help="Materialize only this execution subset while retaining the full route plan.",
    )
    return parser.parse_args()


def parse_location_ids(value) -> list[int]:
    if pd.isna(value) or not str(value).strip():
        return []
    parsed = set()
    for token in str(value).split(";"):
        if token == "":
            continue
        numeric = float(token)
        if not numeric.is_integer():
            raise ValueError(f"location ID is not an integer: {token}")
        parsed.add(int(numeric))
    return sorted(parsed)


def route_and_candidates(action_row, outcome_lookup):
    action = str(action_row.action)
    if action == "candidate_primary_route":
        return "primary", parse_location_ids(action_row.failed_location_ids)
    if action in {"promote_preserved_rigid", "candidate_preserved_rigid_route"}:
        candidates = (
            parse_location_ids(action_row.failed_location_ids)
            if action == "candidate_preserved_rigid_route"
            else []
        )
        return "preserved", candidates
    if action == "from_scratch_valis":
        key = (str(action_row.slide_id), str(action_row.scanner))
        if key not in outcome_lookup:
            return "fromscratch_pending", []
        outcome = outcome_lookup[key]
        if outcome["action"] == "promote_fromscratch_rigid":
            return "fromscratch", []
        if outcome["action"] == "candidate_fromscratch_route":
            return "fromscratch", parse_location_ids(outcome["failed_location_ids"])
        if outcome["action"] == "candidate_primary_route":
            return "primary", parse_location_ids(outcome["failed_location_ids"])
        return "fromscratch_unresolved", parse_location_ids(
            outcome["failed_location_ids"]
        )
    if action == "audit_preserved_rigid":
        return "preserved_pending", []
    raise ValueError(f"unknown fallback action: {action}")


def build_route_plan(actions: pd.DataFrame, outcomes: pd.DataFrame):
    outcome_lookup = {
        (str(row.slide_id), str(row.scanner)): row._asdict()
        for row in outcomes.itertuples(index=False)
    }
    routes = []
    candidate_by_slide: dict[str, set[int]] = {}
    for row in actions.itertuples(index=False):
        route_kind, candidates = route_and_candidates(row, outcome_lookup)
        slide_id, scanner = str(row.slide_id), str(row.scanner)
        routes.append(
            {
                "slide_id": slide_id,
                "scanner": scanner,
                "route_kind": route_kind,
                "candidate_location_ids": ";".join(str(value) for value in candidates),
                "candidate_locations": len(candidates),
            }
        )
        candidate_by_slide.setdefault(slide_id, set()).update(candidates)
    candidates = [
        {
            "slide_id": slide_id,
            "location_ids": ";".join(str(value) for value in sorted(location_ids)),
            "replacement_locations": len(location_ids),
        }
        for slide_id, location_ids in sorted(candidate_by_slide.items())
        if location_ids
    ]
    return pd.DataFrame(routes), pd.DataFrame(candidates)


def route_kind_from_root(root: str) -> str:
    parts = Path(root).parts
    if "fallback" not in parts:
        return "primary"
    index = parts.index("fallback")
    if index + 1 >= len(parts) or parts[index + 1] not in {"preserved", "fromscratch"}:
        raise ValueError(f"cannot infer candidate route kind from {root}")
    return parts[index + 1]


def audited_route_kinds(path: Path):
    audit_path = path / "accepted" / "route_audit.csv"
    if not audit_path.exists():
        return None
    audit = pd.read_csv(audit_path)
    if len(audit) != len(SCANNERS) or set(audit["scanner"].astype(str)) != set(SCANNERS):
        return None
    return {
        str(row.scanner): route_kind_from_root(str(row.route_root))
        for row in audit.itertuples(index=False)
    }


def expected_route_kinds(routes: pd.DataFrame, slide_id: str):
    expected = {scanner: "primary" for scanner in SCANNERS}
    selected = routes[routes["slide_id"].astype(str).eq(slide_id)]
    kinds = set(selected["route_kind"].astype(str))
    if not kinds.issubset({"primary", "preserved", "fromscratch"}):
        return None
    for row in selected.itertuples(index=False):
        expected[str(row.scanner)] = str(row.route_kind)
    return expected


def accepted_trial(path: Path, expected_routes: dict[str, str] | None = None) -> bool:
    summary_path = path / "accepted" / "summary.json"
    if not summary_path.exists():
        return False
    passed = bool(json.loads(summary_path.read_text()).get("trial_gate_pass"))
    if not passed:
        return False
    return expected_routes is None or audited_route_kinds(path) == expected_routes


def latest_accepted_trial(slide_root: Path, expected_routes=None):
    trials = sorted(slide_root.glob("trial_[0-9][0-9][0-9]"), reverse=True)
    return next(
        (path for path in trials if accepted_trial(path, expected_routes)), None
    )


def materialize_trials(
    candidates,
    base_manifest,
    reserve_root: Path,
    trials_root: Path,
    routes: pd.DataFrame,
):
    rows = []
    for candidate in candidates.itertuples(index=False):
        slide_id = str(candidate.slide_id)
        slide_root = trials_root / slide_id
        route_kinds = expected_route_kinds(routes, slide_id)
        accepted_root = latest_accepted_trial(slide_root, route_kinds)
        trial_root = accepted_root or slide_root / "trial_000"
        location_ids = parse_location_ids(candidate.location_ids)
        mapping = {location_id: rank for rank, location_id in enumerate(location_ids)}
        accepted = accepted_root is not None
        manifest_path = trial_root / "manifest.csv"
        if not accepted:
            base = base_manifest[base_manifest["slide_id"].eq(slide_id)].copy()
            reserve_path = reserve_root / slide_id / "reserve_candidates.csv"
            if not reserve_path.exists():
                raise FileNotFoundError(reserve_path)
            reserve = pd.read_csv(reserve_path, dtype={"slide_id": str})
            reserve = reserve[reserve["slide_id"].eq(slide_id)].copy()
            missing_ranks = sorted(set(mapping.values()) - set(reserve["reserve_rank"].astype(int)))
            if missing_ranks:
                raise ValueError(
                    f"{slide_id}: reserve ranks unavailable through {max(missing_ranks)}"
                )
            trial, audit = build_trial(base, reserve, mapping)
            trial_root.mkdir(parents=True, exist_ok=True)
            trial.to_csv(manifest_path, index=False)
            audit.to_csv(trial_root / "replacement_audit.csv", index=False)
            summary = {
                "analysis": "e0_native_candidate_trial",
                "trial_version": TRIAL_VERSION,
                "slide_id": slide_id,
                "locations": 100,
                "replacement_mapping": {
                    str(location_id): int(rank)
                    for location_id, rank in sorted(mapping.items())
                },
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
                "location_ids": candidate.location_ids,
                "replacement_locations": int(candidate.replacement_locations),
                "trial_root": str(trial_root),
                "manifest_path": str(manifest_path),
                "accepted_existing": accepted,
                "needs_execution": not accepted,
            }
        )
    return pd.DataFrame(rows)


def fallback_execution_rows(routes, trials):
    trial_lookup = {
        str(row.slide_id): row._asdict() for row in trials.itertuples(index=False)
    }
    rows = []
    for route in routes.itertuples(index=False):
        slide_id = str(route.slide_id)
        if slide_id not in trial_lookup or not trial_lookup[slide_id]["needs_execution"]:
            continue
        route_kind = str(route.route_kind)
        if route_kind not in {"preserved", "fromscratch"}:
            continue
        trial_root = Path(trial_lookup[slide_id]["trial_root"])
        scanner = str(route.scanner)
        if route_kind == "preserved":
            rigid_root = EXISTING_RIGID_ROOT
            transform_cache_root = "outputs/e0_rigid_native_fallback"
        else:
            rigid_root = FROMSCRATCH_RIGID_ROOT
            transform_cache_root = "outputs/e0_valis_from_scratch_native_full"
        rows.append(
            {
                "slide_id": slide_id,
                "scanner": scanner,
                "route_kind": route_kind,
                "manifest_path": trial_lookup[slide_id]["manifest_path"],
                "rigid_root": rigid_root,
                "alignment_root": str(
                    trial_root / "align" / route_kind / scanner / "shards"
                ),
                "fallback_output_root": str(
                    trial_root / "fallback" / route_kind
                ),
                "transform_cache_root": transform_cache_root,
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    actions = pd.read_csv(args.actions, dtype={"slide_id": str})
    outcomes = pd.read_csv(args.fromscratch_outcomes, dtype={"slide_id": str})
    routes, candidates = build_route_plan(actions, outcomes)
    if args.only_slides:
        requested = {str(value) for value in args.only_slides}
        available = set(candidates["slide_id"].astype(str))
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"requested candidate slides are absent: {missing}")
        candidates = candidates[
            candidates["slide_id"].astype(str).isin(requested)
        ].copy()
    unresolved = routes[routes["route_kind"].str.endswith(("pending", "unresolved"))]
    base_manifest = pd.read_csv(args.manifest, dtype={"slide_id": str})
    trials = (
        materialize_trials(
            candidates,
            base_manifest,
            Path(args.reserve_root),
            Path(args.trials_root),
            routes,
        )
        if args.build_trials
        else candidates.assign(
            trial_root="",
            manifest_path="",
            accepted_existing=False,
            needs_execution=True,
        )
    )
    fallback_rows = fallback_execution_rows(routes, trials)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    routes.to_csv(output / "route_plan.csv", index=False)
    trials.to_csv(output / "candidate_slides.csv", index=False)
    trials[trials["needs_execution"]].to_csv(
        output / "candidate_primary_slides.csv", index=False
    )
    fallback_rows.to_csv(output / "candidate_fallback_cells.csv", index=False)
    unresolved.to_csv(output / "unresolved_routes.csv", index=False)
    summary = {
        "analysis": "e0_candidate_execution_plan",
        "routes": int(len(routes)),
        "candidate_slides": int(len(trials)),
        "candidate_locations": int(trials["replacement_locations"].sum())
        if len(trials)
        else 0,
        "accepted_existing": int(trials["accepted_existing"].sum())
        if len(trials)
        else 0,
        "candidate_slides_needing_execution": int(trials["needs_execution"].sum())
        if len(trials)
        else 0,
        "fallback_cells_needing_trial_audit": int(len(fallback_rows)),
        "unresolved_or_pending_routes": int(len(unresolved)),
        "selection_uses": "registration geometry only; no PFM, correction, or tissue outcome",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
