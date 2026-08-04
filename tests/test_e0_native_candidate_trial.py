import pandas as pd

from build_e0_native_candidate_trial import TRIAL_VERSION, build_trial


def test_candidate_trial_preserves_slot_and_changes_only_selected_coordinate():
    base = pd.DataFrame(
        {
            "slide_id": ["s"] * 100,
            "location_id": range(100),
            "replicate_id": [index % 5 for index in range(100)],
            "x": [index * 10 for index in range(100)],
            "y": [index * 10 + 1 for index in range(100)],
            "candidate_rank": [-1] * 100,
            "tissue_type": ["normal"] * 100,
            "alignment_version": ["base"] * 100,
        }
    )
    reserve = pd.DataFrame(
        {
            "slide_id": ["s", "s"],
            "reserve_rank": [0, 1],
            "x": [5000, 6000],
            "y": [5001, 6001],
            "candidate_rank": [100, 101],
            "alignment_version": ["base", "base"],
        }
    )
    trial, audit = build_trial(base, reserve, {7: 0})
    replacement = trial[trial["location_id"].eq(7)].iloc[0]
    assert replacement["x"] == 5000
    assert replacement["replicate_id"] == 2
    assert replacement["manifest_version"] == TRIAL_VERSION
    assert trial[trial["location_id"].eq(8)].iloc[0]["x"] == 80
    assert audit.iloc[0]["previous_x"] == 70
