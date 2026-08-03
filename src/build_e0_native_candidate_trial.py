"""Build a deterministic geometry-only candidate-replacement trial manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TRIAL_VERSION = "e0_native_candidate_trial_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide-id", required=True)
    parser.add_argument("--location-ids", type=int, nargs="+", required=True)
    parser.add_argument("--reserve-ranks", type=int, nargs="+")
    parser.add_argument(
        "--manifest", default="outputs/e0_feature_manifest_109/feature_manifest.csv"
    )
    parser.add_argument(
        "--reserve-root", default="outputs/e0_candidate_reserve/shards"
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def build_trial(base: pd.DataFrame, reserve: pd.DataFrame, mapping: dict[int, int]):
    if len(base) != 100 or base["location_id"].nunique() != 100:
        raise ValueError("base manifest must contain 100 unique locations")
    if len(mapping) != len(set(mapping)) or len(mapping.values()) != len(
        set(mapping.values())
    ):
        raise ValueError("location IDs and reserve ranks must be unique")

    trial = base.copy()
    audit_rows = []
    for location_id, reserve_rank in sorted(mapping.items()):
        base_match = trial["location_id"].eq(location_id)
        reserve_match = reserve["reserve_rank"].eq(reserve_rank)
        if int(base_match.sum()) != 1:
            raise KeyError(f"base location missing or duplicated: {location_id}")
        if int(reserve_match.sum()) != 1:
            raise KeyError(f"reserve rank missing or duplicated: {reserve_rank}")
        previous = trial.loc[base_match].iloc[0]
        candidate = reserve.loc[reserve_match].iloc[0].copy()
        candidate["manifest_version"] = TRIAL_VERSION
        candidate["tissue_type"] = previous["tissue_type"]
        candidate["slide_id"] = str(previous["slide_id"])
        candidate["location_id"] = int(location_id)
        candidate["replicate_id"] = int(previous["replicate_id"])
        candidate["native_candidate_replacement"] = True
        candidate["native_candidate_reserve_rank"] = int(reserve_rank)
        candidate["alignment_version"] = (
            f"{candidate['alignment_version']}+{TRIAL_VERSION}"
        )
        trial = pd.concat(
            [trial.loc[~base_match], candidate.to_frame().T], ignore_index=True
        )
        audit_rows.append(
            {
                "slide_id": str(previous["slide_id"]),
                "location_id": int(location_id),
                "replicate_id": int(previous["replicate_id"]),
                "previous_x": int(previous["x"]),
                "previous_y": int(previous["y"]),
                "previous_candidate_rank": int(previous["candidate_rank"]),
                "reserve_rank": int(reserve_rank),
                "replacement_candidate_rank": int(candidate["candidate_rank"]),
                "replacement_x": int(candidate["x"]),
                "replacement_y": int(candidate["y"]),
                "selection_uses": "geometry only",
            }
        )

    trial["native_candidate_replacement"] = trial[
        "native_candidate_replacement"
    ].eq(True)
    trial = trial.sort_values("location_id").reset_index(drop=True)
    if len(trial) != 100 or trial["location_id"].nunique() != 100:
        raise ValueError("trial replacement changed the 100-location key contract")
    if trial[["x", "y"]].duplicated().any():
        raise ValueError("trial contains duplicate canonical coordinates")
    return trial, pd.DataFrame(audit_rows)


def main():
    args = parse_args()
    slide_id = str(args.slide_id)
    location_ids = sorted(args.location_ids)
    if len(location_ids) != len(set(location_ids)):
        raise ValueError("--location-ids must be unique")
    reserve_ranks = (
        list(range(len(location_ids)))
        if args.reserve_ranks is None
        else list(args.reserve_ranks)
    )
    if len(reserve_ranks) != len(location_ids):
        raise ValueError("--reserve-ranks must match --location-ids length")
    if any(rank < 0 for rank in reserve_ranks):
        raise ValueError("reserve ranks must be non-negative")
    mapping = dict(zip(location_ids, reserve_ranks))

    base = pd.read_csv(args.manifest, dtype={"slide_id": str})
    base = base[base["slide_id"].eq(slide_id)].copy()
    reserve_path = Path(args.reserve_root) / slide_id / "reserve_candidates.csv"
    reserve = pd.read_csv(reserve_path, dtype={"slide_id": str})
    reserve = reserve[reserve["slide_id"].eq(slide_id)].copy()
    trial, audit = build_trial(base, reserve, mapping)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trial.to_csv(output / "manifest.csv", index=False)
    audit.to_csv(output / "replacement_audit.csv", index=False)
    summary = {
        "analysis": "e0_native_candidate_trial",
        "trial_version": TRIAL_VERSION,
        "slide_id": slide_id,
        "locations": 100,
        "replacement_mapping": {
            str(location_id): int(reserve_rank)
            for location_id, reserve_rank in sorted(mapping.items())
        },
        "selection_uses": "registration geometry only; no PFM, correction, or tissue outcome",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
