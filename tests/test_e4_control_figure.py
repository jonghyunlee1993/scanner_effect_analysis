import pytest

from build_e4_control_figure import condition_style


def test_all_e4_conditions_have_stable_figure_styles():
    conditions = (
        "raw",
        "hf_retention_0p75",
        "hf_retention_0p50",
        "hf_retention_0p25",
        "hf_retention_0p00",
        "hf_boost_1p25",
        "hf_boost_1p50",
        "hf_boost_2p00",
        "registered_loo_hf_mean_0p25",
        "global_train_hf_mean_1p00",
    )
    styles = [condition_style(condition) for condition in conditions]
    assert len({label for label, _ in styles}) == 10
    assert all(color.startswith("#") for _, color in styles)
    with pytest.raises(ValueError):
        condition_style("posthoc")

