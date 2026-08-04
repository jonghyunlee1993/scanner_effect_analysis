import numpy as np
import pandas as pd

from analyze_e6_loso_heterogeneity import build_winner_diagnostic
from build_e6_loso_replicate_contrasts import location_radius


def test_location_radius_is_zero_for_scanner_invariant_locations():
    rng = np.random.default_rng(7)
    locations = rng.normal(size=(10, 8))
    locations /= np.linalg.norm(locations, axis=1, keepdims=True)
    features = np.repeat(locations[None], 6, axis=0)
    np.testing.assert_allclose(location_radius(features), 0.0, atol=1e-12)


def test_winner_diagnostic_ranks_only_globally_safe_methods():
    endpoint_rows = []
    tissue_rows = []
    for encoder_id in ("resnet50", "uni_v1", "conch_v1", "virchow2"):
        endpoint_rows.extend([
            {
                "encoder_id": encoder_id,
                "condition": "reinhard_lab",
                "safe_for_pfm": True,
                "safe_and_invariance_improved": True,
                "safe_improved_rr_rank": 2,
            },
            {
                "encoder_id": encoder_id,
                "condition": "orthogonal_procrustes",
                "safe_for_pfm": True,
                "safe_and_invariance_improved": True,
                "safe_improved_rr_rank": 1,
            },
            {
                "encoder_id": encoder_id,
                "condition": "paired_od_affine",
                "safe_for_pfm": False,
                "safe_and_invariance_improved": False,
                "safe_improved_rr_rank": np.nan,
            },
        ])
        for analysis_set in ("full_37_tissues", "min3_sensitivity"):
            for method, benefit in (
                ("reinhard_lab", 0.1),
                ("orthogonal_procrustes", 0.2),
                ("paired_od_affine", 9.0),
            ):
                tissue_rows.append({
                    "endpoint": "radius_benefit",
                    "analysis_set": analysis_set,
                    "encoder_id": encoder_id,
                    "method": method,
                    "tissue_type": "Liver",
                    "slides_in_tissue": 4,
                    "predicted_tissue_contrast": benefit,
                })
    ranks, summary = build_winner_diagnostic(
        pd.DataFrame(tissue_rows), pd.DataFrame(endpoint_rows)
    )
    assert set(ranks["method"]) == {"reinhard_lab", "orthogonal_procrustes"}
    assert ranks["global_winner_remains_first"].all()
    assert (summary["global_winner_first_fraction"] == 1.0).all()
