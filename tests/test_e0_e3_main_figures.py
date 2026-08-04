import pandas as pd
import pytest

from build_e0_e3_main_figures import (
    select_background_high,
    summarize_alias_profiles,
    summarize_geometry_recovery,
)


def test_alias_summary_uses_worst_sinusoid_or_broadband_ratio():
    rows = []
    for scanner in ("gt450", "versa"):
        for profile in ("q05", "q50", "q95"):
            for pipeline in ("original_bicubic", "explicit_aa_lanczos3"):
                explicit = pipeline == "explicit_aa_lanczos3"
                rows.append(
                    {
                        "scanner": scanner,
                        "transform_profile": profile,
                        "pipeline": pipeline,
                        "sinusoid_alias_to_inband_ratio": 0.01 if explicit else 1.1,
                        "white_noise_alias_to_inband_ratio_max": 0.02 if explicit else 1.2,
                        "alias_power_ratio_limit": 0.05,
                        "alias_gate_pass": explicit,
                    }
                )
    result = summarize_alias_profiles(pd.DataFrame(rows))
    explicit = result[result["pipeline"].eq("explicit_aa_lanczos3")]
    assert explicit["worst_alias_ratio"].eq(0.02).all()


def test_alias_summary_rejects_explicit_aa_above_gate():
    rows = []
    for scanner in ("gt450", "versa"):
        for profile in ("q05", "q50", "q95"):
            for pipeline in ("original_bicubic", "explicit_aa_lanczos3"):
                rows.append(
                    {
                        "scanner": scanner,
                        "transform_profile": profile,
                        "pipeline": pipeline,
                        "sinusoid_alias_to_inband_ratio": 0.06,
                        "white_noise_alias_to_inband_ratio_max": 0.02,
                        "alias_power_ratio_limit": 0.05,
                        "alias_gate_pass": pipeline == "explicit_aa_lanczos3",
                    }
                )
    with pytest.raises(ValueError, match="exceeds"):
        summarize_alias_profiles(pd.DataFrame(rows))


def test_background_panel_requires_all_scanners_and_zero_negative_fraction():
    scanners = ("at2", "gt450", "versa", "akoya", "s60", "s360")
    frame = pd.DataFrame(
        {
            "scanner": list(scanners) * 2,
            "band": ["high"] * 6 + ["mid"] * 6,
            "negative_after_subtraction_fraction": [0.0] * 12,
            "background_fraction_median": [0.001] * 12,
            "background_fraction_q95": [0.002] * 12,
        }
    )
    selected = select_background_high(frame)
    assert selected["scanner"].tolist() == list(scanners)


def test_geometry_recovery_counts_primary_failures_and_final_pass():
    actions = pd.DataFrame(
        {
            "scanner": ["akoya"] * 34
            + ["gt450"] * 10
            + ["versa"] * 9
            + ["s60"] * 6
            + ["s360"] * 2,
            "action": ["fallback"] * 61,
        }
    )
    result = summarize_geometry_recovery(actions)
    assert result.loc[result["scanner"].eq("akoya"), "primary_pass_fraction"].item() == 75 / 109
    assert result["final_pass_fraction"].eq(1.0).all()
