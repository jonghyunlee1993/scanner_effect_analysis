import numpy as np

from extract_e0_resampling_provenance import matrix_metrics


def test_matrix_metrics_recovers_target_mpp_from_isotropic_downsampling():
    native_mpp = 0.25
    target_mpp = 0.5
    matrix = np.eye(2) * (native_mpp / target_mpp)
    result = matrix_metrics(matrix, native_mpp, target_mpp)
    assert np.isclose(result["native_px_per_output_px"], 2.0)
    assert np.isclose(result["effective_target_mpp"], target_mpp)
    assert np.isclose(result["target_mpp_error_percent"], 0.0)


def test_matrix_metrics_separates_rotation_from_scale():
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    result = matrix_metrics(0.5 * rotation, 0.25, 0.5)
    assert np.isclose(result["affine_rotation_deg"], 37.0)
    assert np.isclose(result["affine_anisotropy_ratio"], 1.0)
    assert np.isclose(result["effective_target_mpp"], 0.5)
