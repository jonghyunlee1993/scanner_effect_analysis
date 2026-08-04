import pandas as pd
import pytest

from finalize_e0_native_geometry import replace_manifest_slides, route_matches_manifest


def manifest_frame(slide_id="s1"):
    return pd.DataFrame(
        {
            "slide_id": [slide_id] * 100,
            "location_id": range(100),
            "replicate_id": [value % 5 for value in range(100)],
            "tissue_type": ["aorta"] * 100,
            "center_x": [1000 + value for value in range(100)],
            "center_y": [2000 + value for value in range(100)],
        }
    )


def test_manifest_override_preserves_slot_identity(tmp_path):
    base = manifest_frame()
    replacement = base.copy()
    replacement["center_x"] += 10
    path = tmp_path / "trial.csv"
    replacement.to_csv(path, index=False)
    plan = pd.DataFrame(
        {
            "slide_id": ["s1"],
            "manifest_path": [path],
            "manifest_route_label": ["trial_000"],
        }
    )
    result, audit = replace_manifest_slides(base, plan)
    assert result["center_x"].min() == 1010
    assert result["manifest_route_label"].eq("trial_000").all()
    assert len(audit) == 1
    replacement.loc[0, "replicate_id"] = 4
    replacement.to_csv(path, index=False)
    with pytest.raises(ValueError, match="slot identities"):
        replace_manifest_slides(base, plan)


def test_geometry_route_must_match_final_candidate_centers():
    manifest = manifest_frame()
    geometry = manifest[
        ["slide_id", "location_id", "replicate_id", "center_x", "center_y"]
    ].rename(
        columns={
            "center_x": "canonical_center_x",
            "center_y": "canonical_center_y",
        }
    )
    assert route_matches_manifest(geometry, manifest)
    geometry.loc[20, "canonical_center_y"] += 1
    assert not route_matches_manifest(geometry, manifest)
