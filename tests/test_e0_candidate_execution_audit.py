from pathlib import Path

import pytest

import pandas as pd

import audit_e0_candidate_execution_plan as candidate_audit
from audit_e0_candidate_execution_plan import audit_trial, route_root


def test_candidate_route_roots_match_launcher_layout():
    trial = Path("trials/s1/trial_000")
    assert route_root(trial, "primary", "akoya") == trial / "primary"
    assert route_root(trial, "preserved", "akoya") == (
        trial / "fallback" / "preserved" / "akoya"
    )
    assert route_root(trial, "fromscratch", "versa") == (
        trial / "fallback" / "fromscratch" / "versa"
    )
    with pytest.raises(ValueError, match="not auditable"):
        route_root(trial, "preserved_pending", "akoya")


def test_failed_location_ids_are_json_serializable_python_ints(tmp_path, monkeypatch):
    trial = tmp_path / "trials" / "s1" / "trial_000"
    trial.mkdir(parents=True)
    manifest_path = trial / "manifest.csv"
    pd.DataFrame(
        {
            "slide_id": ["s1"],
            "location_id": [7],
            "replicate_id": [0],
            "center_x": [100],
            "center_y": [100],
        }
    ).to_csv(manifest_path, index=False)
    locations = pd.DataFrame(
        {
            "slide_id": ["s1"],
            "location_id": pd.Series([7], dtype="int64"),
            "scanner": ["at2"],
            "geometry_pass": [False],
        }
    )
    cells = pd.DataFrame({"slide_id": ["s1"], "scanner": ["at2"]})
    routes = pd.DataFrame({"route_pass": [False]})
    tuples = pd.DataFrame()

    monkeypatch.setattr(
        candidate_audit,
        "compose_routes",
        lambda manifest, slide_id, scanner_routes: (locations, cells, routes),
    )
    monkeypatch.setattr(
        candidate_audit,
        "audit_merged",
        lambda manifest, locations, cells, missing, invalid: (
            locations,
            tuples,
            {"cohort_gate_pass": False, "six_scanner_tuples_passing": 0},
        ),
    )
    candidate = type(
        "Candidate",
        (),
        {
            "slide_id": "s1",
            "trial_root": str(trial),
            "manifest_path": str(manifest_path),
        },
    )()
    summary = audit_trial(candidate, pd.DataFrame(columns=["slide_id"]))
    assert summary["failed_location_ids"] == [7]
    assert type(summary["failed_location_ids"][0]) is int
