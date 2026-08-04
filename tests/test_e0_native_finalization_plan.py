import json
from pathlib import Path

import pandas as pd
import pytest

from build_e0_native_finalization_plan import build_plans


def candidate_frame(tmp_path, accepted=True):
    trial = tmp_path / "s1" / "trial_002"
    artifact = trial / "accepted"
    artifact.mkdir(parents=True)
    (artifact / "summary.json").write_text(
        json.dumps({"trial_gate_pass": accepted}) + "\n"
    )
    return pd.DataFrame(
        {
            "slide_id": ["s1"],
            "trial_root": [str(trial)],
            "manifest_path": [str(trial / "manifest.csv")],
            "accepted_existing": [accepted],
        }
    )


def test_finalization_replaces_all_six_cells_for_candidate_slide(tmp_path):
    candidates = candidate_frame(tmp_path)
    routes = pd.DataFrame(
        {
            "slide_id": ["s1", "s2"],
            "scanner": ["akoya", "versa"],
            "route_kind": ["fromscratch", "preserved"],
        }
    )
    manifests, cells = build_plans(
        routes, candidates, Path("preserved"), Path("fromscratch")
    )
    candidate_cells = cells[cells["slide_id"].eq("s1")]
    assert len(manifests) == 1
    assert len(candidate_cells) == 6
    assert set(candidate_cells["geometry_route_label"]) == {
        "candidate:trial_002:primary",
        "candidate:trial_002:fromscratch",
    }
    noncandidate = cells[cells["slide_id"].eq("s2")].iloc[0]
    assert noncandidate["location_path"] == (
        "preserved/versa/shards/s2/native_geometry_locations.csv"
    )


def test_finalization_rejects_unaccepted_candidate(tmp_path):
    with pytest.raises(ValueError, match="not accepted"):
        build_plans(
            pd.DataFrame(columns=["slide_id", "scanner", "route_kind"]),
            candidate_frame(tmp_path, accepted=False),
            Path("preserved"),
            Path("fromscratch"),
        )
