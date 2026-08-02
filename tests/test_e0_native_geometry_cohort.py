from types import SimpleNamespace

import numpy as np
import pandas as pd

from merge_e0_native_geometry_cohort import audit_merged, expected_keys
from run_e0_native_geometry_cohort import (
    corners_within_native,
    resolve_slide_id,
    target_square_native_corners,
    transform_gate,
)


def test_target_square_native_corners_inverts_native_to_target():
    native_to_target = np.array(
        [[0.5, 0.0, 20.0], [0.0, 0.5, 10.0], [0.0, 0.0, 1.0]]
    )
    corners = target_square_native_corners(native_to_target, 120, 60, 40)
    assert np.allclose(
        corners,
        np.array([[200, 100], [280, 100], [280, 180], [200, 180]]),
    )
    assert corners_within_native(corners, width=300, height=200)
    assert not corners_within_native(corners, width=275, height=200)


def test_transform_gate_uses_scale_anisotropy_and_reprojection_contract():
    args = SimpleNamespace(
        minimum_sift_inliers=80,
        maximum_thumbnail_q95_px=4.0,
        maximum_affine_anisotropy=1.02,
        maximum_scale_relative_error=0.05,
    )
    expected = 0.5052 / 0.262407
    native_to_target = np.diag([1.0 / expected, 1.0 / expected, 1.0])
    metrics = {"sift_inliers": 100, "thumbnail_reprojection_q95_px": 2.0}
    passing = transform_gate(metrics, native_to_target, "gt450", args)
    assert passing["global_transform_pass"]
    assert np.isclose(passing["native_px_per_target_px"], expected)

    bad = native_to_target.copy()
    bad[1, 1] *= 0.90
    failing = transform_gate(metrics, bad, "gt450", args)
    assert not failing["global_transform_pass"]
    assert not failing["global_transform_checks"]["maximum_affine_anisotropy"]


def test_slide_resolution_requires_exactly_one_selector():
    manifest = pd.DataFrame({"slide_id": ["b", "a"], "location_id": [0, 0]})
    slide, order = resolve_slide_id(manifest, None, 0)
    assert slide == "a"
    assert order == ["a", "b"]


def test_merge_gate_requires_every_six_scanner_tuple_and_cell():
    manifest = pd.DataFrame(
        {
            "slide_id": ["s1", "s1"],
            "location_id": [0, 1],
        }
    )
    locations = expected_keys(manifest)
    locations["geometry_pass"] = True
    cells = locations[["slide_id", "scanner"]].drop_duplicates()
    cells["cell_pass"] = True
    empty = pd.DataFrame()
    _, tuple_qc, summary = audit_merged(
        manifest, locations, cells, empty, empty
    )
    assert summary["cohort_gate_pass"]
    assert tuple_qc["tuple_pass"].all()

    locations.loc[
        locations["scanner"].eq("gt450") & locations["location_id"].eq(1),
        "geometry_pass",
    ] = False
    _, _, failed = audit_merged(manifest, locations, cells, empty, empty)
    assert not failed["cohort_gate_pass"]
    assert failed["six_scanner_tuples_passing"] == 1
