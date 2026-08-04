import pandas as pd

from select_e0_fromscratch_outcomes import (
    classify_fromscratch,
    primary_candidate_if_not_worse,
)


def test_valid_primary_with_fewer_failures_is_preferred_for_candidates():
    locations = pd.DataFrame(
        {
            "slide_id": ["s1"] * 100,
            "scanner": ["akoya"] * 100,
            "location_id": range(100),
            "geometry_pass": [False, False] + [True] * 98,
        }
    )
    cells = pd.DataFrame(
        {"slide_id": ["s1"], "scanner": ["akoya"], "global_transform_pass": [True]}
    )
    result = primary_candidate_if_not_worse(
        "s1", "akoya", 80, locations, cells
    )
    assert result["action"] == "candidate_primary_route"
    assert result["failed_location_ids"] == "0;1"


def write_route(root, failure_reason="pass", cell_pass=True, global_transform_pass=True):
    shard = root / "akoya" / "shards" / "s1"
    shard.mkdir(parents=True)
    locations = pd.DataFrame(
        {
            "slide_id": ["s1"] * 100,
            "scanner": ["akoya"] * 100,
            "location_id": range(100),
            "geometry_pass": [failure_reason == "pass"] + [True] * 99,
            "failure_reason": [failure_reason] + ["pass"] * 99,
        }
    )
    locations.to_csv(shard / "native_geometry_locations.csv", index=False)
    pd.DataFrame(
        {
            "slide_id": ["s1"],
            "scanner": ["akoya"],
            "cell_pass": [cell_pass],
            "global_transform_pass": [global_transform_pass],
        }
    ).to_csv(shard / "native_geometry_cells.csv", index=False)


def test_fromscratch_pass_promotes_and_bounds_failure_uses_candidate(tmp_path):
    passing = tmp_path / "passing"
    write_route(passing)
    assert classify_fromscratch("s1", "akoya", passing)["action"] == "promote_fromscratch_rigid"
    bounds = tmp_path / "bounds"
    write_route(bounds, "native_fov_bounds", cell_pass=False)
    action = classify_fromscratch("s1", "akoya", bounds)
    assert action["action"] == "candidate_fromscratch_route"
    assert action["failed_location_ids"] == "0"


def test_fromscratch_local_non_bounds_failure_uses_candidate(tmp_path):
    root = tmp_path / "bad"
    write_route(root, "same_scanner_low_ncc", cell_pass=False)
    assert (
        classify_fromscratch("s1", "akoya", root)["action"]
        == "candidate_fromscratch_route"
    )


def test_fromscratch_failed_global_transform_stays_unresolved(tmp_path):
    root = tmp_path / "global"
    write_route(
        root,
        "global_transform_gate;same_scanner_low_ncc",
        cell_pass=False,
        global_transform_pass=False,
    )
    assert (
        classify_fromscratch("s1", "akoya", root)["action"]
        == "unresolved_fromscratch_geometry"
    )
