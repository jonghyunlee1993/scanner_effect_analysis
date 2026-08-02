"""Aggregate the 109-slide current-route E0 registration audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="outputs/e0_registration_cohort_109/cohort_manifest.csv",
    )
    parser.add_argument(
        "--shards",
        default="outputs/e0_registration_cohort_109/shards",
    )
    parser.add_argument("--output", default="outputs/e0_registration_cohort_109")
    return parser.parse_args()


def read_shards(manifest, root: Path, filename: str):
    frames = []
    missing = []
    for slide_id in manifest["slide_id"].astype(str):
        path = root / slide_id / filename
        if not path.exists():
            missing.append(str(path))
        else:
            frames.append(pd.read_csv(path, dtype={"slide_id": str}))
    if missing:
        raise FileNotFoundError("missing cohort shards:\n" + "\n".join(missing))
    return pd.concat(frames, ignore_index=True)


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"slide_id": str})
    if len(manifest) != 109:
        raise ValueError(f"expected 109 frozen slides, got {len(manifest)}")
    root = Path(args.shards)
    alignment = read_shards(manifest, root, "alignment_patch_metrics.csv")
    cell = read_shards(manifest, root, "alignment_summary.csv")
    fits = read_shards(manifest, root, "prior_fits.csv")
    spectral = read_shards(manifest, root, "spectral_bands.csv")
    annotation = manifest[["slide_id", "tissue_type", "common_coordinate_pool"]]
    alignment = alignment.merge(annotation, on="slide_id", validate="many_to_one")
    cell = cell.merge(annotation, on="slide_id", validate="many_to_one")
    spectral = spectral.merge(annotation, on="slide_id", validate="many_to_one")

    scanner_summary = (
        cell.groupby("scanner", as_index=False)
        .agg(
            n_slide_cells=("slide_id", "size"),
            cell_pass_fraction=("cell_pass", "mean"),
            median_geometry_pass_fraction=("geometry_pass_fraction", "median"),
            q05_geometry_pass_fraction=(
                "geometry_pass_fraction",
                lambda value: value.quantile(0.05),
            ),
            minimum_geometry_pass_fraction=("geometry_pass_fraction", "min"),
            median_corrected_ncc=("corrected_ncc_median", "median"),
            median_residual_shift=("residual_shift_median", "median"),
            median_boundary_fraction=("corrected_boundary_fraction", "median"),
        )
    )

    tuple_matrix = alignment.pivot_table(
        index=["slide_id", "patch_index"],
        columns="scanner",
        values="geometry_pass",
        aggfunc="first",
    ).astype(bool)
    expected_scanners = {"gt450", "versa", "akoya", "s60", "s360"}
    if set(tuple_matrix.columns) != expected_scanners:
        raise ValueError(f"unexpected scanners in tuple matrix: {tuple_matrix.columns.tolist()}")
    tuple_matrix["all_scanners"] = tuple_matrix.all(axis=1)
    tuple_summary = (
        tuple_matrix.groupby(level="slide_id")
        .agg(
            selected_locations=("all_scanners", "size"),
            valid_six_scanner_locations=("all_scanners", "sum"),
        )
        .reset_index()
        .merge(annotation, on="slide_id", validate="one_to_one")
    )
    tuple_summary["valid_fraction"] = (
        tuple_summary["valid_six_scanner_locations"]
        / tuple_summary["selected_locations"]
    )
    tuple_summary["replacements_needed_for_100"] = (
        100 - tuple_summary["valid_six_scanner_locations"]
    )

    spectral_wide = spectral.pivot_table(
        index=["slide_id", "tissue_type", "scanner", "band"],
        columns="condition",
        values=["log2_relative_transfer", "n_common"],
        aggfunc="first",
    )
    spectral_wide.columns = [f"{metric}__{condition}" for metric, condition in spectral_wide]
    spectral_wide = spectral_wide.reset_index()
    spectral_wide["delta_corrected_vs_old_log2"] = (
        spectral_wide["log2_relative_transfer__current_corrected"]
        - spectral_wide["log2_relative_transfer__current_old_local16"]
    )
    spectral_wide["abs_delta_corrected_vs_old_log2"] = spectral_wide[
        "delta_corrected_vs_old_log2"
    ].abs()
    spectral_summary = (
        spectral_wide.groupby(["scanner", "band"], as_index=False)
        .agg(
            n_slides=("slide_id", "size"),
            median_delta_log2=("delta_corrected_vs_old_log2", "median"),
            median_abs_delta_log2=("abs_delta_corrected_vs_old_log2", "median"),
            q95_abs_delta_log2=(
                "abs_delta_corrected_vs_old_log2",
                lambda value: value.quantile(0.95),
            ),
            maximum_abs_delta_log2=("abs_delta_corrected_vs_old_log2", "max"),
            minimum_valid_locations=("n_common__current_corrected", "min"),
        )
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    alignment.to_csv(output / "alignment_patch_metrics.csv", index=False)
    cell.to_csv(output / "alignment_slide_scanner_cells.csv", index=False)
    fits.to_csv(output / "prior_fits.csv", index=False)
    spectral.to_csv(output / "spectral_bands.csv", index=False)
    scanner_summary.to_csv(output / "scanner_summary.csv", index=False)
    tuple_summary.to_csv(output / "slide_tuple_summary.csv", index=False)
    spectral_wide.to_csv(output / "spectral_comparison.csv", index=False)
    spectral_summary.to_csv(output / "spectral_sensitivity_summary.csv", index=False)

    scanner_order = ["gt450", "versa", "akoya", "s60", "s360"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ordered = scanner_summary.set_index("scanner").reindex(scanner_order)
    axes[0].bar(np.arange(5), ordered["cell_pass_fraction"], color="#2563EB")
    axes[0].axhline(0.8, color="#B91C1C", ls="--", lw=1)
    axes[0].set_xticks(np.arange(5), [value.upper() for value in scanner_order])
    axes[0].set_ylim(0, 1.03)
    axes[0].set_ylabel("Passing slide–scanner cells")
    axes[0].set_title("A  Population geometry gate")

    axes[1].hist(
        tuple_summary["valid_six_scanner_locations"],
        bins=np.arange(0, 102, 5),
        color="#0F766E",
        edgecolor="white",
    )
    axes[1].axvline(80, color="#B91C1C", ls="--", lw=1)
    axes[1].set_xlabel("Valid locations among original 100")
    axes[1].set_ylabel("Slides")
    axes[1].set_title("B  Six-scanner tuple retention")

    high = spectral_wide[spectral_wide["band"].eq("high")]
    for index_scanner, scanner in enumerate(scanner_order):
        values = high.loc[
            high["scanner"].eq(scanner), "delta_corrected_vs_old_log2"
        ].to_numpy()
        axes[2].scatter(
            np.full(len(values), index_scanner), values, s=12, alpha=0.45
        )
    axes[2].axhline(0, color="#6B7280", lw=1)
    axes[2].set_xticks(np.arange(5), [value.upper() for value in scanner_order])
    axes[2].set_ylabel("Corrected − old high-band ERT (log2)")
    axes[2].set_title("C  ERT registration sensitivity")
    figure.tight_layout()
    figure.savefig(output / "figure_e0_registration_cohort.png", dpi=200)
    figure.savefig(output / "figure_e0_registration_cohort.pdf")
    plt.close(figure)

    summary = {
        "analysis": "e0_registration_current_route_cohort",
        "slides": int(len(manifest)),
        "tissue_types": int(manifest["tissue_type"].nunique()),
        "slide_scanner_cells": int(len(cell)),
        "cell_pass_fraction": float(cell["cell_pass"].mean()),
        "minimum_scanner_cell_pass_fraction": float(
            scanner_summary["cell_pass_fraction"].min()
        ),
        "six_scanner_locations": {
            "median": float(tuple_summary["valid_six_scanner_locations"].median()),
            "minimum": int(tuple_summary["valid_six_scanner_locations"].min()),
            "slides_at_least_80": int(
                tuple_summary["valid_six_scanner_locations"].ge(80).sum()
            ),
            "slides_at_least_90": int(
                tuple_summary["valid_six_scanner_locations"].ge(90).sum()
            ),
            "slides_all_100": int(
                tuple_summary["valid_six_scanner_locations"].eq(100).sum()
            ),
        },
        "next_gate": "deterministic_replacement_from_common_coordinate_pool",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(scanner_summary.to_string(index=False))
    print(spectral_summary.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
