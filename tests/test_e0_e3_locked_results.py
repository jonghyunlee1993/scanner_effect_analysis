import pytest

from audit_e0_e3_locked_results import (
    build_locked_rows,
    validate_contract,
    validate_scanner_metadata,
)


SCANNERS = ("gt450", "versa", "akoya", "s60", "s360")


def contract_fixture():
    geometry = {
        "cohort_gate_pass": True,
        "location_rows_observed": 65_400,
        "locations_passing": 65_400,
        "cells_passing": 654,
        "six_scanner_tuples_passing": 10_900,
        "pixel_source": "native_wsi_only",
    }
    grid = {"grid_gate_pass": True, "slides_passing": 109, "patches_observed": 65_400}
    alias = {"explicit_aa_pipeline_all_profiles_pass": True, "original_pipeline_all_profiles_pass": False}
    background = {
        "noise_floor_sensitivity_complete": True,
        "background_patches": 65_400,
        "absolute_mtf_claim": False,
    }
    e1 = {
        "slides": 109,
        "high_band_results": [
            {
                "scanner": scanner,
                "fixed_fold_transfer": 1.0,
                "fixed_mean_log2_transfer": 0.0,
                "fixed_ci95_low": -0.1,
                "fixed_ci95_high": 0.1,
            }
            for scanner in SCANNERS
        ],
    }
    tissue = {
        "slides": 109,
        "tissues": 37,
        "high_band_results": [
            {
                "scanner": scanner,
                "optimizer_converged": True,
                "tissue_fraction_between_slide_variance": 0.2,
                "tissue_variance_q_bh": 0.05,
            }
            for scanner in SCANNERS
        ],
    }
    e3 = {"slides": 109, "bootstrap_replicates": 5000, "all_scanners_reject_scalar_family": True}
    pfm = {
        "population_gate_pass": True,
        "models_expected": 4,
        "models": ["resnet50", "uni_v1", "conch_v1", "virchow2"],
        "shards_passing": 436,
        "features_observed": 261_600,
    }
    return geometry, grid, alias, background, e1, tissue, e3, pfm


def test_locked_contract_accepts_completed_e0_e3_and_four_pfm_population():
    validate_contract(*contract_fixture())


def test_locked_contract_rejects_incomplete_geometry():
    fixture = list(contract_fixture())
    fixture[0] = dict(fixture[0], cohort_gate_pass=False)
    with pytest.raises(ValueError, match="geometry gate"):
        validate_contract(*fixture)


def test_scanner_metadata_gate_preserves_operator_required_fields():
    validate_scanner_metadata(
        {
            "header_population_gate_pass": True,
            "scanner_slide_headers_observed": 654,
            "scanner_slide_headers_passing": 654,
            "manuscript_acquisition_metadata_complete": False,
            "remaining_user_fields": [
                "exact_manufacturer_and_model",
                "firmware_version",
                "jpeg_quality_setting",
                "objective_numerical_aperture",
            ],
        }
    )


def test_locked_rows_keep_e4_e7_outcomes_out_of_scope():
    geometry, grid, _, _, e1, tissue, _, pfm = contract_fixture()
    alias_profiles = [
        {
            "pipeline": "explicit_aa_lanczos3",
            "sinusoid_alias_to_inband_ratio": "0.01",
            "white_noise_alias_to_inband_ratio_max": "0.02",
        }
        for _ in range(6)
    ]
    background_rows = [
        {
            "scanner": scanner,
            "band": "high",
            "minimum_snr_threshold": "1.0",
            "median_delta_corrected_minus_raw_log2": "-0.001",
            "median_delta_bootstrap_ci95_low": "-0.002",
            "median_delta_bootstrap_ci95_high": "-0.0005",
        }
        for scanner in SCANNERS
    ]
    e3_rows = [
        {
            "scanner": scanner,
            "scalar_family_rejected": "True",
            "excess_off_axis_rms_median": "0.1",
            "excess_off_axis_rms_median_ci95_low": "0.08",
            "excess_off_axis_rms_median_ci95_high": "0.12",
        }
        for scanner in SCANNERS
    ]
    rows = build_locked_rows(
        geometry, grid, alias_profiles, background_rows, e1, tissue, e3_rows, pfm
    )
    assert len(rows) == 26
    assert {row["phase"] for row in rows} == {"E0", "E0d", "E1", "E2", "E3"}
