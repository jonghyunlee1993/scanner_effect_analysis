import pytest

from audit_e0_native_candidate_trial import parse_routes


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
