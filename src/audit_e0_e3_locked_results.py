"""Lock audited E0--E3 results without opening E4--E7 PFM outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SCANNERS = ("gt450", "versa", "akoya", "s60", "s360")

SOURCE_ARTIFACTS = (
    ("e0_geometry_summary", "outputs/e0_native_geometry_final/summary.json", "E0 geometry gate"),
    ("e0_geometry_cells", "outputs/e0_native_geometry_final/cell_qc.csv", "E0 cell QC"),
    ("e0_geometry_tuples", "outputs/e0_native_geometry_final/tuple_qc.csv", "E0 tuple QC"),
    (
        "e0_geometry_manifest",
        "outputs/e0_native_geometry_final/native_geometry_manifest.csv",
        "frozen native geometry",
    ),
    ("e0_grid_summary", "outputs/e0_native_aa_grid/audit/summary.json", "native-AA grid gate"),
    ("e0_grid_shards", "outputs/e0_native_aa_grid/audit/shard_audit.csv", "native-AA shard audit"),
    ("e0_alias_summary", "outputs/e0_alias_audit_final_geometry/summary.json", "alias gate"),
    (
        "e0_alias_profiles",
        "outputs/e0_alias_audit_final_geometry/scanner_profile_summary.csv",
        "alias profile results",
    ),
    (
        "e0_transform_provenance",
        "outputs/e0_alias_audit_final_geometry/transform_provenance.csv",
        "final transform provenance",
    ),
    ("e0d_summary", "outputs/e0d_same_chain_background/summary.json", "same-chain background gate"),
    (
        "e0d_sensitivity",
        "outputs/e0d_same_chain_background/noise_floor_band_sensitivity_summary.csv",
        "noise-floor sensitivity",
    ),
    ("e1_summary", "outputs/e1_native_aa_spectral_109/summary.json", "E1 fixed spectrum"),
    (
        "e1_tissue_summary",
        "outputs/e1_native_aa_spectral_109/tissue_random_slope_summary.json",
        "E2 nested tissue/slide model",
    ),
    (
        "e1_tissue_table",
        "outputs/e1_native_aa_spectral_109/tissue_random_slope_variance_components.csv",
        "E2 variance components",
    ),
    ("e3_summary", "outputs/e3_native_aa_scalar_reducibility/summary.json", "E3 LOSO gate"),
    (
        "e3_scanner_table",
        "outputs/e3_native_aa_scalar_reducibility/scanner_summary.csv",
        "E3 scanner estimates",
    ),
    (
        "pfm_contract",
        "outputs/e0_pfm_contract/checkpoint_manifest.json",
        "frozen four-PFM contract",
    ),
    (
        "pfm_population_summary",
        "outputs/e0_pfm_features/audit/summary.json",
        "raw feature population gate",
    ),
    (
        "e0_e3_main_figure_summary",
        "outputs/e0_e3_main_figures/summary.json",
        "Main Figure 1--3 source/output provenance",
    ),
    (
        "e0_scanner_metadata_summary",
        "outputs/e0_scanner_acquisition_metadata/summary.json",
        "native WSI acquisition-metadata gate",
    ),
    (
        "e0_scanner_metadata_table",
        "outputs/e0_scanner_acquisition_metadata/scanner_summary.csv",
        "scanner-level acquisition metadata",
    ),
    (
        "e0_scanner_metadata_missing",
        "outputs/e0_scanner_acquisition_metadata/missing_metadata.csv",
        "operator-supplied acquisition fields",
    ),
)

FIGURE_COMPONENTS = (
    (
        "Figure 1",
        "registration cohort",
        "outputs/e0_registration_cohort_109/figure_e0_registration_cohort.png",
    ),
    (
        "Figure 1",
        "explicit-AA alias gate",
        "outputs/e0_alias_audit_final_geometry/figure_e0_alias_audit.png",
    ),
    (
        "Figure 2",
        "native-AA cohort spectrum",
        "outputs/e1_native_aa_spectral_109/figure1_cohort_spectral_overview.png",
    ),
    (
        "Figure 2",
        "scalar projection bridge",
        "outputs/e1_native_aa_spectral_109/figure3_blur_sharpen_bridge.png",
    ),
    (
        "Figure 3",
        "nested tissue random slopes",
        "outputs/e1_native_aa_spectral_109/figure4_tissue_random_slopes.png",
    ),
    (
        "Figure 3",
        "slide-level mixed effects",
        "outputs/e1_native_aa_spectral_109/figure2_mixed_effects.png",
    ),
)

MAIN_FIGURES = (
    (
        "Figure 1",
        "outputs/e0_e3_main_figures/figure1_study_estimator_validity.png",
        "Study design and estimator validity",
    ),
    (
        "Figure 2",
        "outputs/e0_e3_main_figures/figure2_audited_scanner_spectrum.png",
        "Audited scanner spectrum and scalar lack of fit",
    ),
    (
        "Figure 3",
        "outputs/e0_e3_main_figures/figure3_scanner_content_structure.png",
        "Scanner by content structure",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="outputs/e0_e3_locked_results")
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_csv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_contract(geometry, grid, alias, background, e1, tissue, e3, pfm):
    require(geometry.get("cohort_gate_pass") is True, "E0 geometry gate did not pass")
    require(geometry.get("location_rows_observed") == 65_400, "E0 geometry row count differs")
    require(geometry.get("cells_passing") == 654, "E0 scanner-slide cell count differs")
    require(geometry.get("six_scanner_tuples_passing") == 10_900, "E0 tuple count differs")
    require(geometry.get("pixel_source") == "native_wsi_only", "E0 pixel source is not native-only")

    require(grid.get("grid_gate_pass") is True, "native-AA grid gate did not pass")
    require(grid.get("slides_passing") == 109, "native-AA slide count differs")
    require(grid.get("patches_observed") == 65_400, "native-AA patch count differs")

    require(alias.get("explicit_aa_pipeline_all_profiles_pass") is True, "explicit-AA alias gate failed")
    require(alias.get("original_pipeline_all_profiles_pass") is False, "historical alias control changed")

    require(background.get("noise_floor_sensitivity_complete") is True, "E0d is incomplete")
    require(background.get("background_patches") == 65_400, "E0d patch count differs")
    require(background.get("absolute_mtf_claim") is False, "E0d claim scope changed")

    e1_rows = e1.get("high_band_results", [])
    require(e1.get("slides") == 109, "E1 slide count differs")
    require({row["scanner"] for row in e1_rows} == set(SCANNERS), "E1 scanner set differs")
    for row in e1_rows:
        require(row["fixed_ci95_low"] < row["fixed_ci95_high"], f"E1 CI invalid: {row['scanner']}")

    tissue_rows = tissue.get("high_band_results", [])
    require(tissue.get("slides") == 109 and tissue.get("tissues") == 37, "E2 cohort differs")
    require({row["scanner"] for row in tissue_rows} == set(SCANNERS), "E2 scanner set differs")
    require(all(row.get("optimizer_converged") is True for row in tissue_rows), "E2 fit did not converge")

    require(e3.get("slides") == 109, "E3 slide count differs")
    require(e3.get("bootstrap_replicates") == 5000, "E3 bootstrap count differs")
    require(e3.get("all_scanners_reject_scalar_family") is True, "E3 family rejection gate failed")

    require(pfm.get("population_gate_pass") is True, "four-PFM population gate failed")
    require(pfm.get("models_expected") == 4, "PFM count differs")
    require(
        set(pfm.get("models", [])) == {"resnet50", "uni_v1", "conch_v1", "virchow2"},
        "PFM panel differs",
    )
    require(pfm.get("shards_passing") == 436, "PFM shard count differs")
    require(pfm.get("features_observed") == 261_600, "PFM feature count differs")


def validate_scanner_metadata(metadata):
    require(metadata.get("header_population_gate_pass") is True, "scanner metadata header gate failed")
    require(metadata.get("scanner_slide_headers_observed") == 654, "scanner metadata header count differs")
    require(metadata.get("scanner_slide_headers_passing") == 654, "scanner metadata failures remain")
    require(
        metadata.get("manuscript_acquisition_metadata_complete") is False,
        "scanner metadata completion status changed without operator fields",
    )
    require(
        set(metadata.get("remaining_user_fields", []))
        == {
            "exact_manufacturer_and_model",
            "firmware_version",
            "jpeg_quality_setting",
            "objective_numerical_aperture",
        },
        "remaining scanner metadata field set differs",
    )


def result_row(phase, endpoint, scanner, band, estimate, ci_low, ci_high, unit, decision, source):
    return {
        "phase": phase,
        "endpoint": endpoint,
        "scanner": scanner,
        "band": band,
        "estimate": estimate,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "unit": unit,
        "decision": decision,
        "source_path": source,
    }


def build_locked_rows(geometry, grid, alias_profiles, background_rows, e1, tissue, e3_rows, pfm):
    rows = [
        result_row(
            "E0", "geometry locations passing", "", "", geometry["locations_passing"], "", "",
            "scanner-location rows", "pass", "outputs/e0_native_geometry_final/summary.json",
        ),
        result_row(
            "E0", "six-scanner tuples passing", "", "", geometry["six_scanner_tuples_passing"], "", "",
            "tuples", "pass", "outputs/e0_native_geometry_final/summary.json",
        ),
        result_row(
            "E0", "native-AA patches passing", "", "", grid["patches_observed"], "", "",
            "patches", "pass", "outputs/e0_native_aa_grid/audit/summary.json",
        ),
        result_row(
            "E0", "four-PFM raw features passing", "", "", pfm["features_observed"], "", "",
            "embeddings", "pass", "outputs/e0_pfm_features/audit/summary.json",
        ),
    ]

    explicit = [row for row in alias_profiles if row["pipeline"] == "explicit_aa_lanczos3"]
    require(len(explicit) == 6, "expected six explicit-AA transform profiles")
    max_sinusoid = max(float(row["sinusoid_alias_to_inband_ratio"]) for row in explicit)
    max_noise = max(float(row["white_noise_alias_to_inband_ratio_max"]) for row in explicit)
    require(max_sinusoid <= 0.05 and max_noise <= 0.05, "computed alias ratio exceeds gate")
    rows.extend(
        [
            result_row(
                "E0", "maximum explicit-AA sinusoid alias ratio", "", "high", max_sinusoid, "", "",
                "ratio", "pass", "outputs/e0_alias_audit_final_geometry/scanner_profile_summary.csv",
            ),
            result_row(
                "E0", "maximum explicit-AA broadband-noise alias ratio", "", "high", max_noise, "", "",
                "ratio", "pass", "outputs/e0_alias_audit_final_geometry/scanner_profile_summary.csv",
            ),
        ]
    )

    background_high = [
        row
        for row in background_rows
        if row["band"] == "high" and float(row["minimum_snr_threshold"]) == 1.0 and row["scanner"] != "at2"
    ]
    require({row["scanner"] for row in background_high} == set(SCANNERS), "E0d scanner set differs")
    for row in background_high:
        rows.append(
            result_row(
                "E0d",
                "noise-corrected minus raw ERT",
                row["scanner"],
                "high",
                row["median_delta_corrected_minus_raw_log2"],
                row["median_delta_bootstrap_ci95_low"],
                row["median_delta_bootstrap_ci95_high"],
                "log2 transfer",
                "sensitivity only",
                "outputs/e0d_same_chain_background/noise_floor_band_sensitivity_summary.csv",
            )
        )

    for row in sorted(e1["high_band_results"], key=lambda value: SCANNERS.index(value["scanner"])):
        rows.append(
            result_row(
                "E1", "AT2-relative fixed log2 transfer", row["scanner"], "high",
                row["fixed_mean_log2_transfer"], row["fixed_ci95_low"], row["fixed_ci95_high"],
                "log2 transfer", f"locked; fold={row['fixed_fold_transfer']}",
                "outputs/e1_native_aa_spectral_109/summary.json",
            )
        )

    for row in sorted(tissue["high_band_results"], key=lambda value: SCANNERS.index(value["scanner"])):
        rows.append(
            result_row(
                "E2", "tissue fraction of between-slide variance", row["scanner"], "high",
                row["tissue_fraction_between_slide_variance"], "", "", "fraction",
                f"BH q={row['tissue_variance_q_bh']}",
                "outputs/e1_native_aa_spectral_109/tissue_random_slope_summary.json",
            )
        )

    require({row["scanner"] for row in e3_rows} == set(SCANNERS), "E3 scanner table differs")
    for row in sorted(e3_rows, key=lambda value: SCANNERS.index(value["scanner"])):
        require(row["scalar_family_rejected"] == "True", f"E3 decision changed: {row['scanner']}")
        rows.append(
            result_row(
                "E3", "median excess off-axis RMS", row["scanner"], "0.10--0.90",
                row["excess_off_axis_rms_median"], row["excess_off_axis_rms_median_ci95_low"],
                row["excess_off_axis_rms_median_ci95_high"], "log2 RMS", "scalar family rejected",
                "outputs/e3_native_aa_scalar_reducibility/scanner_summary.csv",
            )
        )
    return rows


def write_csv(path: Path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(repo_root: Path, output: Path):
    paths = {artifact_id: repo_root / relative for artifact_id, relative, _ in SOURCE_ARTIFACTS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    require(not missing, f"missing source artifacts: {missing}")

    geometry = read_json(paths["e0_geometry_summary"])
    grid = read_json(paths["e0_grid_summary"])
    alias = read_json(paths["e0_alias_summary"])
    background = read_json(paths["e0d_summary"])
    e1 = read_json(paths["e1_summary"])
    tissue = read_json(paths["e1_tissue_summary"])
    e3 = read_json(paths["e3_summary"])
    pfm = read_json(paths["pfm_population_summary"])
    validate_contract(geometry, grid, alias, background, e1, tissue, e3, pfm)
    scanner_metadata = read_json(paths["e0_scanner_metadata_summary"])
    validate_scanner_metadata(scanner_metadata)

    artifact_rows = []
    source_hashes = {}
    for artifact_id, relative, role in SOURCE_ARTIFACTS:
        path = repo_root / relative
        digest = sha256_file(path)
        source_hashes[relative] = digest
        artifact_rows.append(
            {
                "artifact_id": artifact_id,
                "role": role,
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "status": "present",
            }
        )

    locked_rows = build_locked_rows(
        geometry,
        grid,
        read_csv(paths["e0_alias_profiles"]),
        read_csv(paths["e0d_sensitivity"]),
        e1,
        tissue,
        read_csv(paths["e3_scanner_table"]),
        pfm,
    )
    for row in locked_rows:
        row["source_sha256"] = source_hashes[row["source_path"]]

    figure_rows = []
    for figure, component, relative in FIGURE_COMPONENTS:
        path = repo_root / relative
        require(path.is_file(), f"missing figure component: {relative}")
        figure_rows.append(
            {
                "figure": figure,
                "component": component,
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "component_status": "complete",
                "composite_status": "complete",
            }
        )

    main_figure_rows = []
    for figure, relative, title in MAIN_FIGURES:
        path = repo_root / relative
        require(path.is_file(), f"missing main figure: {relative}")
        main_figure_rows.append(
            {
                "figure": figure,
                "title": title,
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "status": "complete",
                "e4_e7_outcomes_used": False,
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    write_csv(
        output / "artifact_manifest.csv",
        artifact_rows,
        ["artifact_id", "role", "path", "bytes", "sha256", "status"],
    )
    write_csv(
        output / "locked_results.csv",
        locked_rows,
        [
            "phase", "endpoint", "scanner", "band", "estimate", "ci95_low", "ci95_high",
            "unit", "decision", "source_path", "source_sha256",
        ],
    )
    write_csv(
        output / "figure_component_manifest.csv",
        figure_rows,
        [
            "figure", "component", "path", "bytes", "sha256", "component_status",
            "composite_status",
        ],
    )
    write_csv(
        output / "main_figure_manifest.csv",
        main_figure_rows,
        ["figure", "title", "path", "bytes", "sha256", "status", "e4_e7_outcomes_used"],
    )
    summary = {
        "analysis": "e0_e3_locked_results_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": len(artifact_rows),
        "locked_result_rows": len(locked_rows),
        "figure_components": len(figure_rows),
        "main_figures": len(main_figure_rows),
        "e0_e3_result_lock_complete": True,
        "scanner_acquisition_header_audit_complete": True,
        "scanner_acquisition_operator_fields_complete": False,
        "main_figure_1_3_components_complete": True,
        "main_figure_1_3_composites_complete": True,
        "e4_e7_endpoint_decision_required": True,
        "e4_e7_outcomes_opened_by_this_audit": False,
        "audit_pass": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    summary = run(repo_root, output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
