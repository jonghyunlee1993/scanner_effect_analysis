import numpy as np
import pandas as pd

from analyze_e6_loto_frontier import compare_loso_loto
from e6_loto_population import heldout_tissue_indices


def test_heldout_tissue_indices_exclude_all_slides_in_fold():
    slide_ids = [f"s{index:03d}" for index in range(109)]
    mapping = {
        slide_id: ("target" if index in (3, 9, 21) else f"other_{index}")
        for index, slide_id in enumerate(slide_ids)
    }
    tissue, indices = heldout_tissue_indices(slide_ids, "s009", mapping)
    assert tissue == "target"
    np.testing.assert_array_equal(indices, [3, 9, 21])


def test_loso_loto_comparison_flags_decision_change():
    base = {
        "encoder_id": "resnet50",
        "condition": "reinhard_lab",
        "relative_radius_reduction": 0.2,
        "difference_ci_lower": 0.01,
        "delta_content_margin": 0.0,
        "delta_content_ci_lower": -0.01,
        "content_noninferiority_pass": True,
        "collapse_every_scanner_pass": True,
        "safe_for_pfm": True,
        "safe_and_invariance_improved": True,
        "invariance_improved_but_unsafe": False,
        "safe_improved_rr_rank": 1.0,
    }
    loso = pd.DataFrame([base])
    loto_row = {**base, "relative_radius_reduction": 0.15, "safe_for_pfm": False,
                "safe_and_invariance_improved": False}
    comparison = compare_loso_loto(loso, pd.DataFrame([loto_row]))
    assert comparison.loc[0, "safe_decision_changed"]
    assert comparison.loc[0, "safe_improved_decision_changed"]
    assert np.isclose(
        comparison.loc[0, "relative_radius_reduction_delta_loto_minus_loso"],
        -0.05,
    )
