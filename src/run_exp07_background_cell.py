"""Extract one scanner-slide cell of strict raw-glass background patches."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd

from run_exp06_raw_nps_pilot import (
    QC_THRESHOLDS,
    estimate_nps,
    sample_glass_patches,
    scanner_mpp,
    stable_rng,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--patches", type=int, default=100)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--target-mpp", type=float, default=0.5052)
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--attempts-per-cell", type=int, default=1200)
    parser.add_argument("--fallback-attempts", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def background_moments(images):
    sampled = images[:, ::4, ::4, :].reshape(-1, 3).astype(np.float32)
    od = -np.log10((sampled + 1.0) / 256.0)

    def describe(values):
        return {
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "p01": np.quantile(values, 0.01, axis=0).tolist(),
            "p05": np.quantile(values, 0.05, axis=0).tolist(),
            "p50": np.quantile(values, 0.50, axis=0).tolist(),
            "p95": np.quantile(values, 0.95, axis=0).tolist(),
            "p99": np.quantile(values, 0.99, axis=0).tolist(),
            "covariance": np.cov(values, rowvar=False).tolist(),
        }

    return {
        "subsampling": "every fourth pixel in x and y",
        "sampled_pixels": len(sampled),
        "rgb": describe(sampled),
        "od10": describe(od),
    }


def write_nps(images, output_dir, target_mpp, bins):
    estimates = estimate_nps(images, target_mpp, bins)
    spectra_rows = []
    replicate_rows = []
    with h5py.File(output_dir / "nps_2d.h5", "w") as store:
        for method, estimate in estimates.items():
            store.create_dataset(
                method,
                data=estimate["mean_2d"].astype(np.float32),
                compression="gzip",
                compression_opts=4,
            )
            for frequency, value in zip(
                estimate["frequency"], estimate["mean_radial"]
            ):
                spectra_rows.append(
                    {
                        "detrend": method,
                        "frequency_cyc_per_um": float(frequency),
                        "nps_od2_um2": float(value),
                        "patches": len(images),
                    }
                )
            for replicate, curve in enumerate(estimate["replicate_radial"]):
                for frequency, value in zip(estimate["frequency"], curve):
                    replicate_rows.append(
                        {
                            "detrend": method,
                            "replicate": replicate,
                            "frequency_cyc_per_um": float(frequency),
                            "nps_od2_um2": float(value),
                        }
                    )
    pd.DataFrame(spectra_rows).to_csv(
        output_dir / "raw_glass_nps_spectra.csv", index=False
    )
    pd.DataFrame(replicate_rows).to_csv(
        output_dir / "raw_glass_nps_replicates.csv", index=False
    )


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    if args.task_index < 0 or args.task_index >= len(manifest):
        raise IndexError((args.task_index, len(manifest)))
    row = manifest.iloc[args.task_index]
    if int(row["task_id"]) != args.task_index:
        raise ValueError("manifest task_id is not aligned with row index")
    slide_id = str(row["slide_id"])
    scanner = str(row["scanner"])
    tissue_type = str(row["tissue_type"])
    raw_path = Path(row["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    output_root = Path(args.output_root)
    task_name = f"{args.task_index:04d}_{slide_id}_{scanner}"
    target = output_root / "cells" / task_name
    if (target / "_SUCCESS").exists():
        print(f"[exp07] skip completed {target}", flush=True)
        return
    if target.exists():
        raise FileExistsError(f"unmarked target exists: {target}")
    temporary = output_root / "_tmp" / f"{task_name}_{os.getpid()}"
    temporary.mkdir(parents=True, exist_ok=False)

    slide = openslide.OpenSlide(str(raw_path))
    try:
        raw_root = raw_path.parents[1]
        native_mpp, mpp_source = scanner_mpp(
            raw_root, scanner, raw_path, slide
        )
        width, height = slide.dimensions
        rng = stable_rng(args.seed, slide_id, scanner)
        (
            images,
            accepted,
            candidates,
            rejection_counts,
            native_size,
        ) = sample_glass_patches(
            slide,
            scanner,
            slide_id,
            native_mpp,
            args.target_mpp,
            args.target_size,
            args.patches,
            args.grid_size,
            args.attempts_per_cell,
            args.fallback_attempts,
            rng,
        )
    finally:
        slide.close()

    pd.DataFrame(accepted).to_csv(
        temporary / "accepted_glass_patches.csv", index=False
    )
    with h5py.File(temporary / "glass_patches_at2_grid.h5", "w") as store:
        store.create_dataset(
            "rgb",
            data=images,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        store.create_dataset(
            "coords_native",
            data=np.asarray(
                [[item["x_native"], item["y_native"]] for item in accepted],
                dtype=np.int64,
            ).reshape(-1, 2),
        )
        store.attrs["native_mpp"] = native_mpp
        store.attrs["target_mpp"] = args.target_mpp
        store.attrs["native_patch_size"] = native_size
        store.attrs["target_patch_size"] = args.target_size

    metadata = {
        "task_id": args.task_index,
        "slide_id": slide_id,
        "tissue_type": tissue_type,
        "scanner": scanner,
        "raw_path": str(raw_path),
        "raw_dimensions": [width, height],
        "native_mpp": native_mpp,
        "mpp_source": mpp_source,
        "native_patch_size": native_size,
        "target_mpp": args.target_mpp,
        "target_patch_size": args.target_size,
        "requested_patches": args.patches,
        "accepted_patches": len(images),
        "attempted_patches": len(candidates),
        "complete": len(images) == args.patches,
        "qc_thresholds": QC_THRESHOLDS,
        "rejection_counts": dict(rejection_counts),
        "coordinates_paired_across_scanners": False,
        "optical_density_definition": "-log10((I + 1) / 256)",
        "nps_claim": "operational background NPS, not controlled detector NPS",
    }
    with (temporary / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    if len(images):
        with (temporary / "background_color_summary.json").open("w") as handle:
            json.dump(background_moments(images), handle, indent=2)
    if len(images) >= 20:
        write_nps(images, temporary, args.target_mpp, args.bins)

    if len(images) != args.patches:
        pd.DataFrame(candidates).to_csv(
            temporary / "candidate_glass_qc.csv.gz",
            index=False,
            compression="gzip",
        )
        incomplete = output_root / "incomplete" / task_name
        if incomplete.exists():
            incomplete = output_root / "incomplete" / f"{task_name}_{os.getpid()}"
        os.replace(temporary, incomplete)
        raise RuntimeError(
            f"accepted {len(images)}/{args.patches} after {len(candidates)} attempts; "
            f"partial output: {incomplete}"
        )

    (temporary / "_SUCCESS").write_text("complete\n")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, target)
    print(
        f"[exp07] task={args.task_index} {slide_id}/{scanner} "
        f"accepted={len(images)}/{len(candidates)} output={target}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Preserve the original traceback in array logs.
        raise
