"""Aggregate E0 registration sentinel shards and apply the frozen route gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROUTE_CELL_PASS_FRACTION = 0.90
ROUTE_SCANNER_PASS_FRACTION = 0.80
ERT_MEDIAN_ABS_DELTA_LOG2 = 0.10
ERT_Q95_ABS_DELTA_LOG2 = 0.25


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="outputs/e0_registration_sentinel/sentinel_manifest.csv",
    )
    parser.add_argument(
        "--shards",
        default="outputs/e0_registration_sentinel/shards",
    )
    parser.add_argument("--output", default="outputs/e0_registration_sentinel")
    return parser.parse_args()


def _read_shards(manifest, root: Path, filename: str):
    frames = []
    missing = []
    for slide_id in manifest["slide_id"].astype(str):
        path = root / slide_id / filename
        if not path.exists():
            missing.append(str(path))
        else:
            frames.append(pd.read_csv(path, dtype={"slide_id": str}))
    if missing:
        raise FileNotFoundError("missing sentinel shards:\n" + "\n".join(missing))
    return pd.concat(frames, ignore_index=True)


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"slide_id": str})
    if len(manifest) != 15:
        raise ValueError(f"expected 15 frozen sentinels, got {len(manifest)}")
    shards = Path(args.shards)
    alignment = _read_shards(manifest, shards, "alignment_patch_metrics.csv")
    cell = _read_shards(manifest, shards, "alignment_summary.csv")
    fits = _read_shards(manifest, shards, "prior_fits.csv")
    spectral = _read_shards(manifest, shards, "spectral_bands.csv")
    annotation = manifest[["slide_id", "stratum", "tissue_type"]]
    alignment = alignment.merge(annotation, on="slide_id", validate="many_to_one")
    cell = cell.merge(annotation, on="slide_id", validate="many_to_one")
    spectral = spectral.merge(annotation, on="slide_id", validate="many_to_one")

    route_scanner = (
        cell.groupby(["branch", "scanner"], as_index=False)
        .agg(
            n_slide_cells=("slide_id", "size"),
            cell_pass_fraction=("cell_pass", "mean"),
            median_geometry_pass_fraction=("geometry_pass_fraction", "median"),
            worst_geometry_pass_fraction=("geometry_pass_fraction", "min"),
            median_corrected_ncc=("corrected_ncc_median", "median"),
            median_residual_shift=("residual_shift_median", "median"),
            q95_residual_shift=("residual_shift_q95", "median"),
            median_boundary_fraction=("corrected_boundary_fraction", "median"),
        )
    )
    route = (
        cell.groupby("branch", as_index=False)
        .agg(
            n_slide_scanner_cells=("cell_pass", "size"),
            cell_pass_fraction=("cell_pass", "mean"),
            median_geometry_pass_fraction=("geometry_pass_fraction", "median"),
        )
    )
    per_scanner_min = route_scanner.groupby("branch")["cell_pass_fraction"].min()
    route["minimum_scanner_cell_pass_fraction"] = route["branch"].map(per_scanner_min)
    route["route_pass"] = (
        route["cell_pass_fraction"].ge(ROUTE_CELL_PASS_FRACTION)
        & route["minimum_scanner_cell_pass_fraction"].ge(
            ROUTE_SCANNER_PASS_FRACTION
        )
    )

    spectral_wide = spectral.pivot_table(
        index=["slide_id", "stratum", "tissue_type", "scanner", "band"],
        columns="condition",
        values=["log2_relative_transfer", "n_common"],
        aggfunc="first",
    )
    spectral_wide.columns = [f"{metric}__{condition}" for metric, condition in spectral_wide]
    spectral_wide = spectral_wide.reset_index()
    spectral_wide["delta_current_corrected_vs_old_log2"] = (
        spectral_wide["log2_relative_transfer__current_corrected"]
        - spectral_wide["log2_relative_transfer__current_old_local16"]
    )
    spectral_wide["delta_current_vs_valis_corrected_log2"] = (
        spectral_wide["log2_relative_transfer__current_corrected_route_common"]
        - spectral_wide["log2_relative_transfer__valis_corrected_route_common"]
    )
    spectral_wide["abs_delta_current_vs_valis_corrected_log2"] = spectral_wide[
        "delta_current_vs_valis_corrected_log2"
    ].abs()
    spectral_sensitivity = (
        spectral_wide.groupby(["scanner", "band"], as_index=False)
        .agg(
            n_slides=("slide_id", "size"),
            median_current_correction_delta_log2=(
                "delta_current_corrected_vs_old_log2",
                "median",
            ),
            q05_current_correction_delta_log2=(
                "delta_current_corrected_vs_old_log2",
                lambda value: value.quantile(0.05),
            ),
            q95_current_correction_delta_log2=(
                "delta_current_corrected_vs_old_log2",
                lambda value: value.quantile(0.95),
            ),
            median_abs_current_vs_valis_log2=(
                "abs_delta_current_vs_valis_corrected_log2",
                "median",
            ),
            q95_abs_current_vs_valis_log2=(
                "abs_delta_current_vs_valis_corrected_log2",
                lambda value: value.quantile(0.95),
            ),
            minimum_current_locations=("n_common__current_corrected", "min"),
            minimum_route_common_locations=(
                "n_common__current_corrected_route_common",
                "min",
            ),
        )
    )
    ert_equivalent = bool(
        spectral_wide["abs_delta_current_vs_valis_corrected_log2"].median()
        <= ERT_MEDIAN_ABS_DELTA_LOG2
        and spectral_wide["abs_delta_current_vs_valis_corrected_log2"].quantile(0.95)
        <= ERT_Q95_ABS_DELTA_LOG2
    )
    pass_by_branch = route.set_index("branch")["route_pass"].to_dict()
    if pass_by_branch.get("current", False) and ert_equivalent:
        recommendation = "retain_current_registered_wsi_with_integer_refinement"
    elif pass_by_branch.get("valis", False):
        recommendation = "promote_existing_valis_rigid_with_integer_refinement"
    elif not pass_by_branch.get("current", False) and not pass_by_branch.get("valis", False):
        recommendation = "rerun_rigid_registration_from_native_wsi"
    else:
        recommendation = "review_route_specific_disagreement_before_population_rebuild"

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    alignment.to_csv(output / "alignment_patch_metrics.csv", index=False)
    cell.to_csv(output / "alignment_slide_scanner_cells.csv", index=False)
    fits.to_csv(output / "prior_fits.csv", index=False)
    spectral.to_csv(output / "spectral_bands.csv", index=False)
    route.to_csv(output / "route_gate_summary.csv", index=False)
    route_scanner.to_csv(output / "route_scanner_summary.csv", index=False)
    spectral_wide.to_csv(output / "spectral_route_comparison.csv", index=False)
    spectral_sensitivity.to_csv(output / "spectral_sensitivity_summary.csv", index=False)

    scanners = sorted(route_scanner["scanner"].unique())
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(scanners))
    width = 0.36
    for offset, branch in zip((-width / 2, width / 2), ("current", "valis")):
        values = (
            route_scanner[route_scanner["branch"].eq(branch)]
            .set_index("scanner")
            .reindex(scanners)["cell_pass_fraction"]
        )
        axes[0].bar(x + offset, values, width, label=branch)
    axes[0].axhline(ROUTE_SCANNER_PASS_FRACTION, color="#B91C1C", ls="--", lw=1)
    axes[0].set_xticks(x, [value.upper() for value in scanners])
    axes[0].set_ylim(0, 1.03)
    axes[0].set_ylabel("Passing slide–scanner cells")
    axes[0].set_title("A  Registration route gate")
    axes[0].legend(frameon=False)

    high = spectral_wide[spectral_wide["band"].eq("high")]
    axes[1].axhline(0, color="#6B7280", lw=1)
    for index_scanner, scanner in enumerate(scanners):
        values = high.loc[
            high["scanner"].eq(scanner), "delta_current_vs_valis_corrected_log2"
        ].to_numpy()
        axes[1].scatter(
            np.full(len(values), index_scanner), values, s=22, alpha=0.65
        )
    axes[1].axhline(ERT_MEDIAN_ABS_DELTA_LOG2, color="#B91C1C", ls="--", lw=1)
    axes[1].axhline(-ERT_MEDIAN_ABS_DELTA_LOG2, color="#B91C1C", ls="--", lw=1)
    axes[1].set_xticks(x, [value.upper() for value in scanners])
    axes[1].set_ylabel("Current − VALIS corrected high-band ERT (log₂)")
    axes[1].set_title("B  Route sensitivity")
    figure.tight_layout()
    figure.savefig(output / "figure_e0_registration_route_audit.png", dpi=200)
    figure.savefig(output / "figure_e0_registration_route_audit.pdf")
    plt.close(figure)

    summary = {
        "analysis": "e0_registration_sentinel_aggregate",
        "sentinel_slides": int(len(manifest)),
        "tissue_types": int(manifest["tissue_type"].nunique()),
        "slide_scanner_cells_per_route": int(5 * len(manifest)),
        "frozen_thresholds": {
            "route_cell_pass_fraction": ROUTE_CELL_PASS_FRACTION,
            "route_scanner_pass_fraction": ROUTE_SCANNER_PASS_FRACTION,
            "ert_median_abs_delta_log2": ERT_MEDIAN_ABS_DELTA_LOG2,
            "ert_q95_abs_delta_log2": ERT_Q95_ABS_DELTA_LOG2,
        },
        "route_pass": {key: bool(value) for key, value in pass_by_branch.items()},
        "ert_route_equivalent": ert_equivalent,
        "median_abs_ert_delta_log2": float(
            spectral_wide["abs_delta_current_vs_valis_corrected_log2"].median()
        ),
        "q95_abs_ert_delta_log2": float(
            spectral_wide["abs_delta_current_vs_valis_corrected_log2"].quantile(0.95)
        ),
        "recommendation": recommendation,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(route.to_string(index=False))
    print(spectral_sensitivity.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
