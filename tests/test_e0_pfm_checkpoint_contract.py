from fetch_e0_pfm_checkpoints import MODELS, TRIDENT_COMMIT, validate_contract


def test_frozen_four_encoder_contract_is_exact():
    validate_contract()
    assert TRIDENT_COMMIT == "a6305acfef68d4c6da65e837dde1e0d1870d60e1"
    assert [model["encoder_id"] for model in MODELS] == [
        "resnet50",
        "uni_v1",
        "conch_v1",
        "virchow2",
    ]
    assert [model["feature_dim"] for model in MODELS] == [1024, 1024, 512, 2560]
    assert [model["native_fov_px"] for model in MODELS] == [256, 256, 512, 224]
    assert all(len(model["revision"]) == 40 for model in MODELS)
