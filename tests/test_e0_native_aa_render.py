from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from render_e0_native_aa_shard import (
    MODEL_FOV,
    affine_from_row,
    explicit_aa_pre_scale,
    residual_after_prescale,
    target_top_left,
    validate_slide_geometry,
)


def test_prescale_changes_only_native_coordinate_columns():
    matrix = np.array(
        [[0.5, -0.01, 123.0], [0.01, 0.5, -45.0], [0.0, 0.0, 1.0]]
    )
    scale = explicit_aa_pre_scale(matrix)
    residual = residual_after_prescale(matrix, scale)
    assert scale < 1.0
    assert np.allclose(residual[:2, :2] * scale, matrix[:2, :2])
    assert np.allclose(residual[:2, 2], matrix[:2, 2])


def test_upsampling_does_not_add_a_pre_enlarge():
    matrix = np.array([[1.2, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.0]])
    assert explicit_aa_pre_scale(matrix) == 1.0
    assert np.array_equal(residual_after_prescale(matrix, 1.0), matrix)


def test_model_fovs_are_centered_inside_the_shared_512_grid():
    assert MODEL_FOV == {
        "resnet50": 256,
        "uni_v1": 256,
        "conch_v1": 512,
        "virchow2": 224,
    }
    row = SimpleNamespace(
        canonical_center_x=1000,
        canonical_center_y=2000,
        total_target_dx=-7,
        total_target_dy=11,
    )
    assert target_top_left(row, 512) == (737, 1755)
    assert target_top_left(row, 256) == (865, 1883)
    assert target_top_left(row, 224) == (881, 1899)


def test_affine_from_row_rejects_singular_geometry():
    row = SimpleNamespace(
        native_to_target_m00=1,
        native_to_target_m01=0,
        native_to_target_m02=3,
        native_to_target_m10=0,
        native_to_target_m11=0,
        native_to_target_m12=4,
    )
    with pytest.raises(ValueError, match="singular"):
        affine_from_row(row)


def test_render_gate_requires_all_600_passing_rows():
    rows = []
    for scanner in ("at2", "gt450", "versa", "akoya", "s60", "s360"):
        for location_id in range(100):
            rows.append(
                {
                    "slide_id": "s1",
                    "scanner": scanner,
                    "location_id": location_id,
                    "geometry_pass": True,
                }
            )
    frame = pd.DataFrame(rows)
    assert len(validate_slide_geometry(frame, "s1")) == 600
    frame.loc[0, "geometry_pass"] = False
    with pytest.raises(ValueError, match="geometry failure"):
        validate_slide_geometry(frame, "s1")
