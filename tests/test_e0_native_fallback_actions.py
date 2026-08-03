import pandas as pd

from select_e0_native_fallback_actions import bounds_only, classify_cell


def test_bounds_only_is_strict_and_nonempty():
    assert bounds_only(["native_fov_bounds"])
    assert bounds_only(["target_fov_bounds;native_fov_bounds"])
    assert not bounds_only(["pass"])
    assert not bounds_only(["native_fov_bounds;low_ncc"])


def test_primary_strict_bounds_skips_rigid_rerun(tmp_path):
    primary = pd.DataFrame(
        {
            "slide_id": ["s1"],
            "scanner": ["akoya"],
            "location_id": [7],
            "failure_reason": ["native_fov_bounds"],
        }
    )
    action = classify_cell("s1", "akoya", primary, tmp_path)
    assert action["action"] == "candidate_primary_route"
    assert action["failed_location_ids"] == "7"


def test_missing_preserved_route_requests_from_scratch(tmp_path):
    primary = pd.DataFrame(
        {
            "slide_id": ["s1"],
            "scanner": ["akoya"],
            "location_id": [7],
            "failure_reason": ["low_ncc"],
        }
    )
    action = classify_cell("s1", "akoya", primary, tmp_path)
    assert action["action"] == "from_scratch_valis"
