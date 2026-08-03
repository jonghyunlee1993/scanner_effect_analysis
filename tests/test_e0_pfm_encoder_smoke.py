import numpy as np

from smoke_e0_pfm_encoders import encoder_kwargs, synthetic_rgb


def test_synthetic_smoke_image_is_deterministic_and_model_sized():
    first = np.asarray(synthetic_rgb(224))
    second = np.asarray(synthetic_rgb(224))
    assert first.shape == (224, 224, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)


def test_encoder_kwargs_freeze_conch_projection_and_virchow_pooling():
    assert encoder_kwargs("resnet50", "r.bin") == {"weights_path": "r.bin"}
    assert encoder_kwargs("conch_v1", "c.bin") == {
        "weights_path": "c.bin",
        "with_proj": False,
        "normalize": False,
    }
    assert encoder_kwargs("virchow2", "v.bin") == {
        "weights_path": "v.bin",
        "return_cls": False,
    }
