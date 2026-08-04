import numpy as np
import pandas as pd

from analyze_e0_native_aa_ert_sensitivity import (
    anchor_normalized_transfer,
    compare_routes,
)


def test_anchor_normalization_has_unit_geometric_mean_in_anchor():
    frequency = np.linspace(0.0, 0.2, 21)
    reference = np.ones_like(frequency)
    power = np.full_like(frequency, 4.0)
    transfer = anchor_normalized_transfer(power, reference, frequency, (0.03, 0.10))
    selected = (frequency >= 0.03) & (frequency <= 0.10)
    assert np.isclose(np.exp(np.log(transfer[selected]).mean()), 1.0)


def test_route_comparison_preserves_paired_identity():
    rows = []
    for slide in range(109):
        for scanner in ("at2", "gt450", "versa", "akoya", "s60", "s360"):
            for band in ("low_mid", "mid", "high"):
                rows.append(
                    {
                        "slide_id": str(slide),
                        "scanner": scanner,
                        "band": band,
                        "log2_relative_transfer": float(slide) / 100.0,
                    }
                )
    historical = pd.DataFrame(rows)
    native = historical.copy()
    native["log2_relative_transfer"] += 0.25
    paired, summary = compare_routes(historical, native)
    assert len(paired) == 109 * 6 * 3
    assert np.allclose(paired["delta_native_aa_minus_historical_log2"], 0.25)
    assert np.allclose(summary["signed_delta_median_log2"], 0.25)
