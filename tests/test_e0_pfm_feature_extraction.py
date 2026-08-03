import numpy as np
import pytest

from extract_e0_pfm_features import centered_crop, select_model


def test_centered_crop_uses_the_same_biological_center():
    image = np.arange(512 * 512 * 3, dtype=np.int64).reshape(1, 512, 512, 3)
    for fov in (224, 256, 512):
        observed = centered_crop(image, fov)
        start = (512 - fov) // 2
        assert observed.shape == (1, fov, fov, 3)
        assert np.array_equal(observed[:, 0, 0], image[:, start, start])
        assert np.array_equal(observed[:, -1, -1], image[:, start + fov - 1, start + fov - 1])


def test_centered_crop_rejects_noncontract_inputs():
    with pytest.raises(ValueError):
        centered_crop(np.zeros((1, 256, 256, 3)), 256)
    with pytest.raises(ValueError):
        centered_crop(np.zeros((1, 512, 512, 3)), 225)


def test_model_selection_is_exact_and_ordered():
    contract = {
        "models": [
            {"encoder_id": "resnet50"},
            {"encoder_id": "uni_v1"},
            {"encoder_id": "conch_v1"},
            {"encoder_id": "virchow2"},
        ]
    }
    assert select_model(contract, None, 2)["encoder_id"] == "conch_v1"
    assert select_model(contract, "virchow2", None)["encoder_id"] == "virchow2"
    with pytest.raises(IndexError):
        select_model(contract, None, 4)
