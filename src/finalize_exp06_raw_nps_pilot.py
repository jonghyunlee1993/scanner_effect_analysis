"""Merge an Exp-06 supplement and rebuild complete pilot figures.

The incomplete source directory is preserved.  A new output directory is
created, the selected scanner/slide cell is replaced, and all visualizations
that depend on the patch store are rebuilt from the completed HDF5 file.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from run_exp06_raw_nps_pilot import (
    estimate_nps,
    render_montage,
    render_nps,
    render_nps_2d,
    render_spatial,
)


CELL_TABLES = (
    "accepted_glass_patches.csv",
    "candidate_glass_qc.csv",
    "raw_glass_nps_spectra.csv",
    "raw_glass_nps_replicates.csv",
    "glass_rejection_summary.csv",
    "raw_wsi_metadata.csv",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", required=True)
    parser.add_argument("--supplement", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--slide", required=True)
    parser.add_argument("--scanner", required=True)
    return parser.parse_args()


def replace_table_cell(main, supplement, output, name, slide_id, scanner):
    original = pd.read_csv(main / name)
    replacement = pd.read_csv(supplement / name)
    keep = ~(
        (original["slide_id"].astype(str) == slide_id)
        & (original["scanner"].astype(str) == scanner)
    )
    merged = pd.concat([original.loc[keep], replacement], ignore_index=True)
    merged.to_csv(output / name, index=False)
    return merged


def band_summary(spectra, rejection):
    plane = spectra[spectra["detrend"] == "plane"]
    rows = []
    bands = (
        (0.05, 0.20, "low_band_nps"),
        (0.20, 0.60, "mid_band_nps"),
        (0.60, 0.90, "high_band_nps"),
    )
    for (slide_id, scanner), frame in plane.groupby(["slide_id", "scanner"]):
        row = {"slide_id": slide_id, "scanner": scanner}
        for low, high, name in bands:
            values = frame.loc[
                (frame["frequency_cyc_per_um"] >= low)
                & (frame["frequency_cyc_per_um"] < high),
                "nps_od2_um2",
            ].to_numpy()
            row[name] = float(
                np.exp(np.mean(np.log(np.maximum(values, 1e-30))))
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    counts = rejection[["slide_id", "scanner", "accepted", "attempted"]].copy()
    counts["acceptance_rate"] = counts["accepted"] / counts["attempted"]
    return result.merge(counts, on=["slide_id", "scanner"], how="left")


def main():
    args = parse_args()
    main_dir = Path(args.main)
    supplement_dir = Path(args.supplement)
    output_dir = Path(args.output)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    shutil.copytree(main_dir, output_dir)

    tables = {}
    for name in CELL_TABLES:
        tables[name] = replace_table_cell(
            main_dir,
            supplement_dir,
            output_dir,
            name,
            args.slide,
            args.scanner,
        )

    output_h5 = output_dir / "glass_patches_at2_grid.h5"
    supplement_h5 = supplement_dir / "glass_patches_at2_grid.h5"
    with h5py.File(output_h5, "r+") as destination, h5py.File(
        supplement_h5, "r"
    ) as source:
        target = f"{args.slide}/{args.scanner}"
        if target in destination:
            del destination[target]
        source.copy(source[target], destination[args.slide], name=args.scanner)

    accepted = tables["accepted_glass_patches.csv"]
    spectra = tables["raw_glass_nps_spectra.csv"]
    rejection = tables["glass_rejection_summary.csv"]
    metadata = tables["raw_wsi_metadata.csv"]
    summary_path = main_dir / "summary.json"
    with summary_path.open() as handle:
        summary = json.load(handle)
    scanners = summary["scanners"]
    slides = summary["slides"]
    target_mpp = float(summary["target_grid"]["mpp"])
    bins = int(
        spectra.groupby(["slide_id", "scanner", "detrend"]).size().iloc[0]
    )

    nps_2d = {scanner: [] for scanner in scanners}
    spectra_rows = []
    replicate_rows = []
    montage_slide = slides[min(2, len(slides) - 1)]
    montage = {}
    with h5py.File(output_h5, "r") as store:
        for scanner in scanners:
            montage[scanner] = store[f"{montage_slide}/{scanner}/rgb"][:8]
            for slide_id in slides:
                images = store[f"{slide_id}/{scanner}/rgb"][:]
                estimates = estimate_nps(images, target_mpp, bins)
                nps_2d[scanner].append(estimates["plane"]["mean_2d"])
                for method, estimate in estimates.items():
                    for frequency, value in zip(
                        estimate["frequency"], estimate["mean_radial"]
                    ):
                        spectra_rows.append(
                            {
                                "slide_id": slide_id,
                                "scanner": scanner,
                                "detrend": method,
                                "frequency_cyc_per_um": float(frequency),
                                "nps_od2_um2": float(value),
                                "patches": len(images),
                            }
                        )
                    for replicate_id, curve in enumerate(
                        estimate["replicate_radial"]
                    ):
                        for frequency, value in zip(
                            estimate["frequency"], curve
                        ):
                            replicate_rows.append(
                                {
                                    "slide_id": slide_id,
                                    "scanner": scanner,
                                    "detrend": method,
                                    "replicate": replicate_id,
                                    "frequency_cyc_per_um": float(frequency),
                                    "nps_od2_um2": float(value),
                                }
                            )
    nps_2d = {
        scanner: np.mean(values, axis=0)
        for scanner, values in nps_2d.items()
    }
    spectra = pd.DataFrame(spectra_rows)
    replicates = pd.DataFrame(replicate_rows)
    spectra.to_csv(output_dir / "raw_glass_nps_spectra.csv", index=False)
    replicates.to_csv(
        output_dir / "raw_glass_nps_replicates.csv", index=False
    )

    figures = []
    figures.extend(
        render_montage(montage, scanners, output_dir, montage_slide)
    )
    figures.extend(render_spatial(accepted, scanners, slides, output_dir))
    figures.extend(render_nps(spectra, accepted, scanners, slides, output_dir))
    figures.extend(render_nps_2d(nps_2d, scanners, output_dir, target_mpp))

    bands = band_summary(spectra, rejection)
    bands.to_csv(output_dir / "operational_nps_band_summary.csv", index=False)

    summary["accepted_by_scanner"] = (
        accepted.groupby("scanner").size().to_dict()
    )
    summary["accepted_by_slide_scanner"] = (
        accepted.groupby(["slide_id", "scanner"])
        .size()
        .rename("count")
        .reset_index()
        .to_dict("records")
    )
    summary["supplemented_cells"] = [
        {
            "slide_id": args.slide,
            "scanner": args.scanner,
            "source": str(supplement_dir),
        }
    ]
    summary["optical_density_definition"] = "-log10((I + 1) / 256)"
    summary["nps_units"] = "OD^2 * micrometre^2"
    summary["tables"]["band_summary"] = str(
        output_dir / "operational_nps_band_summary.csv"
    )
    summary["figures"] = figures
    summary["complete_requested_patch_grid"] = bool(
        (metadata["accepted_patches"] == summary["requested_patches_per_slide_scanner"]).all()
    )
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(
        accepted.groupby(["slide_id", "scanner"]).size().unstack().to_string(),
        flush=True,
    )
    print(f"[exp06-finalize] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
