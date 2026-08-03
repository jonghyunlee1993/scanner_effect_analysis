from types import SimpleNamespace

import numpy as np
import pandas as pd

from build_e0_rigid_native_fallback import (
    fallback_location_checks,
    rigid_branch,
    truth,
)
from merge_e0_native_geometry_cohort import audit_merged, expected_keys
from run_e0_native_geometry_cohort import (
    corners_within_native,
    matrix_manifest_fields,
    resolve_slide_id,
    target_square_native_corners,
    transform_output_for_slide,
    transform_gate,
)
from run_e0_valis_rigid_from_scratch import stage_link


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


def test_matrix_manifest_fields_preserve_nonidentity_transform(tmp_path):
    affine = np.array(
        [[0.5, -0.01, 123.0], [0.01, 0.5, -45.0], [0.0, 0.0, 1.0]]
    )
    fields = matrix_manifest_fields(affine, tmp_path / "matrix.npz")
    assert fields["native_to_target_m00"] == 0.5
    assert fields["native_to_target_m02"] == 123.0
    assert fields["native_to_target_m12"] == -45.0
    assert fields["explicit_aa_pre_scale"] < 1.0


def test_slide_resolution_requires_exactly_one_selector():
    manifest = pd.DataFrame({"slide_id": ["b", "a"], "location_id": [0, 0]})
    slide, order = resolve_slide_id(manifest, None, 0)
    assert slide == "a"
    assert order == ["a", "b"]


def test_candidate_trial_can_reuse_frozen_transform_cache(tmp_path):
    output = tmp_path / "trial" / "shards" / "s1"
    default = transform_output_for_slide(
        SimpleNamespace(transform_cache_root=None), output, "s1"
    )
    cached = transform_output_for_slide(
        SimpleNamespace(transform_cache_root=str(tmp_path / "frozen")), output, "s1"
    )
    assert default == output
    assert cached == tmp_path / "frozen" / "shards" / "s1"


def test_rigid_branch_is_stable_for_candidate_transform_cache_layout():
    assert rigid_branch("akoya") == "rigid_akoya"
    assert rigid_branch("gt450") == "rigid_all"


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


def test_merge_gate_reports_fully_missing_outputs_without_crashing():
    manifest = pd.DataFrame({"slide_id": ["s1"], "location_id": [0]})
    missing = pd.DataFrame({"slide_id": ["s1"], "missing": ["all"]})
    _, tuples, summary = audit_merged(
        manifest, pd.DataFrame(), pd.DataFrame(), missing, pd.DataFrame()
    )
    assert tuples.empty
    assert summary["slides_missing"] == 1
    assert summary["missing_location_keys"] == 6
    assert not summary["cohort_gate_pass"]


def test_rigid_native_fallback_uses_scanner_specific_preserved_branch():
    assert rigid_branch("akoya") == "rigid_akoya"
    for scanner in ("gt450", "versa", "s60", "s360"):
        assert rigid_branch(scanner) == "rigid_all"


def test_rigid_native_fallback_parses_serialized_geometry_gate():
    assert truth(True)
    assert truth(" TRUE ")
    assert not truth(False)
    assert not truth("false")


def test_rigid_native_fallback_requires_same_scanner_local_gate():
    passing = dict(
        global_transform_pass=True,
        cross_scanner_geometry_pass=True,
        search_in_bounds=True,
        same_scanner_ncc=0.80,
        same_scanner_boundary=False,
        target_bounds=True,
        native_bounds=True,
        minimum_location_ncc=0.75,
    )
    assert all(fallback_location_checks(**passing).values())

    low_ncc = fallback_location_checks(**{**passing, "same_scanner_ncc": 0.74})
    assert not low_ncc["same_scanner_low_ncc"]
    missing = fallback_location_checks(**{**passing, "same_scanner_ncc": np.nan})
    assert not missing["same_scanner_low_ncc"]
    boundary = fallback_location_checks(**{**passing, "same_scanner_boundary": True})
    assert not boundary["same_scanner_search_boundary"]


def test_valis_from_scratch_staging_link_is_idempotent(tmp_path):
    source = tmp_path / "source.svs"
    source.write_bytes(b"slide")
    destination = tmp_path / "stage" / "slide_at2.svs"
    destination.parent.mkdir()
    stage_link(source, destination)
    stage_link(source, destination)
    assert destination.is_symlink()
    assert destination.resolve() == source.resolve()
