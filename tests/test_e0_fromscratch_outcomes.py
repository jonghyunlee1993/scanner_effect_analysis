import pandas as pd

from select_e0_fromscratch_outcomes import classify_fromscratch


def write_route(root, failure_reason="pass", cell_pass=True):
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
        {"slide_id": ["s1"], "scanner": ["akoya"], "cell_pass": [cell_pass]}
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


def test_fromscratch_non_bounds_failure_stays_unresolved(tmp_path):
    root = tmp_path / "bad"
    write_route(root, "same_scanner_low_ncc", cell_pass=False)
    assert (
        classify_fromscratch("s1", "akoya", root)["action"]
        == "unresolved_fromscratch_geometry"
    )
