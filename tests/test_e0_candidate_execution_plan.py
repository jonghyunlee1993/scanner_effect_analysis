from types import SimpleNamespace

import pandas as pd

from build_e0_candidate_execution_plan import (
    accepted_trial,
    build_route_plan,
    latest_accepted_trial,
    parse_location_ids,
    route_kind_from_root,
    route_and_candidates,
)


def test_location_id_parser_is_sorted_and_unique():
    assert parse_location_ids("9;2;9;4") == [2, 4, 9]
    assert parse_location_ids("") == []
    assert parse_location_ids(78.0) == [78]


def test_route_resolution_uses_fromscratch_outcome_locations():
    action = SimpleNamespace(
        action="from_scratch_valis", slide_id="s1", scanner="akoya"
    )
    route, locations = route_and_candidates(
        action,
        {
            ("s1", "akoya"): {
                "action": "candidate_fromscratch_route",
                "failed_location_ids": "7;3",
            }
        },
    )
    assert route == "fromscratch"
    assert locations == [3, 7]


def test_fromscratch_outcome_can_retain_better_primary_route():
    action = SimpleNamespace(
        action="from_scratch_valis", slide_id="s1", scanner="akoya"
    )
    route, locations = route_and_candidates(
        action,
        {
            ("s1", "akoya"): {
                "action": "candidate_primary_route",
                "failed_location_ids": "8;2",
            }
        },
    )
    assert route == "primary"
    assert locations == [2, 8]


def test_candidate_plan_unions_failed_locations_across_scanners():
    actions = pd.DataFrame(
        [
            {
                "slide_id": "s1",
                "scanner": "akoya",
                "action": "candidate_preserved_rigid_route",
                "failed_location_ids": "2;7",
            },
            {
                "slide_id": "s1",
                "scanner": "versa",
                "action": "from_scratch_valis",
                "failed_location_ids": "",
            },
        ]
    )
    outcomes = pd.DataFrame(
        [
            {
                "slide_id": "s1",
                "scanner": "versa",
                "action": "candidate_fromscratch_route",
                "failed_location_ids": "7;9",
            }
        ]
    )
    routes, candidates = build_route_plan(actions, outcomes)
    assert set(routes["route_kind"]) == {"preserved", "fromscratch"}
    assert candidates.iloc[0]["location_ids"] == "2;7;9"
    assert candidates.iloc[0]["replacement_locations"] == 3


def test_latest_accepted_trial_must_match_current_route_map(tmp_path):
    slide = tmp_path / "s1"
    expected = {
        "at2": "primary",
        "gt450": "primary",
        "versa": "primary",
        "akoya": "preserved",
        "s60": "primary",
        "s360": "primary",
    }
    for number, akoya_kind in ((0, "preserved"), (1, "fromscratch")):
        accepted = slide / f"trial_{number:03d}" / "accepted"
        accepted.mkdir(parents=True)
        (accepted / "summary.json").write_text('{"trial_gate_pass": true}\n')
        pd.DataFrame(
            {
                "scanner": list(expected),
                "route_root": [
                    f"trial/fallback/{akoya_kind}/akoya"
                    if scanner == "akoya"
                    else "trial/primary"
                    for scanner in expected
                ],
            }
        ).to_csv(accepted / "route_audit.csv", index=False)
    assert accepted_trial(slide / "trial_000", expected)
    assert not accepted_trial(slide / "trial_001", expected)
    assert latest_accepted_trial(slide, expected) == slide / "trial_000"


def test_candidate_route_kind_is_inferred_from_materialized_root():
    assert route_kind_from_root("x/trial_001/primary") == "primary"
    assert route_kind_from_root("x/fallback/preserved/akoya") == "preserved"
