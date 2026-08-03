import pandas as pd
import pytest

from audit_e0_native_candidate_trial import parse_routes, validate_route


def test_route_overrides_are_scanner_specific():
    routes = parse_routes("primary", ["akoya=fallback/a", "versa=fallback/v"])
    assert str(routes["at2"]) == "primary"
    assert str(routes["akoya"]) == "fallback/a"
    assert str(routes["versa"]) == "fallback/v"


def test_route_overrides_reject_duplicates_and_unknown_scanners():
    with pytest.raises(ValueError, match="duplicate"):
        parse_routes("primary", ["akoya=a", "akoya=b"])
    with pytest.raises(ValueError, match="unsupported"):
        parse_routes("primary", ["foo=a"])


def test_route_validation_rejects_stale_coordinates_for_same_location_ids(tmp_path):
    manifest = pd.DataFrame(
        {
            "location_id": range(100),
            "replicate_id": [value % 5 for value in range(100)],
            "center_x": [1000 + value for value in range(100)],
            "center_y": [2000 + value for value in range(100)],
        }
    )
    locations = pd.DataFrame(
        {
            "location_id": range(100),
            "replicate_id": [value % 5 for value in range(100)],
            "canonical_center_x": manifest["center_x"],
            "canonical_center_y": manifest["center_y"],
            "geometry_pass": True,
            "pixel_source": "native_wsi",
            "historical_pixels_primary": False,
        }
    )
    cells = pd.DataFrame({"cell_pass": [True]})
    passing = validate_route(
        locations, cells, manifest, "s1", "at2", tmp_path
    )
    assert passing["route_pass"]
    locations.loc[10, "canonical_center_x"] += 1
    stale = validate_route(locations, cells, manifest, "s1", "at2", tmp_path)
    assert not stale["route_pass"]
    assert stale["failure_reason"] == "canonical_centers_do_not_match_trial"
