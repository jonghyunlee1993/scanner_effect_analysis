import pandas as pd

from audit_e4_control_results import validate_endpoint_logic


def test_endpoint_logic_detects_posthoc_safe_flag():
    rows = []
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
    for model in ("resnet50", "uni_v1", "conch_v1", "virchow2"):
        for condition in conditions:
            rows.append(
                {
                    "encoder_id": model,
                    "condition": condition,
                    "content_noninferiority_pass": True,
                    "collapse_every_scanner_pass": True,
                    "invariance_improved": condition != "raw",
                    "safe_for_pfm": True,
                    "safe_and_invariance_improved": condition != "raw",
                }
            )
    frame = pd.DataFrame(rows)
    assert validate_endpoint_logic(frame) == []
    frame.loc[0, "safe_for_pfm"] = False
    assert "safe_logic" in validate_endpoint_logic(frame)

