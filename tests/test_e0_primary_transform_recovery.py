import numpy as np

from run_e0_primary_transform_recovery_pilot import scale_thumbnail_affine_to_full


def test_scale_thumbnail_affine_to_full_recovers_full_coordinate_map():
    full = np.array(
        [[0.50, -0.01, 120.0], [0.01, 0.50, -40.0], [0.0, 0.0, 1.0]]
    )
    native_full = np.array([100_000.0, 80_000.0])
    native_thumb = np.array([4_000.0, 3_200.0])
    registered_full = np.array([50_000.0, 40_000.0])
    registered_thumb = np.array([4_000.0, 3_200.0])
    source_to_thumb = np.diag([*(native_thumb / native_full), 1.0])
    registered_to_thumb = np.diag([*(registered_thumb / registered_full), 1.0])
    thumbnail = registered_to_thumb @ full @ np.linalg.inv(source_to_thumb)
    recovered = scale_thumbnail_affine_to_full(
        thumbnail,
        native_full,
        native_thumb,
        registered_full,
        registered_thumb,
    )
    assert np.allclose(recovered, full)
