from fetch_e0_pfm_checkpoints import (
    CONCH_SOURCE_COMMIT,
    MODELS,
    RUNTIME_DISTRIBUTIONS,
    TRIDENT_COMMIT,
    validate_contract,
)


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
    assert RUNTIME_DISTRIBUTIONS == {
        "torch": "2.5.1",
        "torchvision": "0.20.1",
        "timm": "0.9.8",
        "huggingface_hub": "0.29.1",
        "conch": "0.1.0",
    }
    assert CONCH_SOURCE_COMMIT == "02d6ac59cc20874bff0f581de258c2b257f69a84"
