"""Audit and aggregate the 109-slide by 6-scanner background extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_exp05_spectral_pilot import SCANNER_COLORS, save_figure


CHANNELS = ("r", "g", "b")
SPACES = ("rgb", "od10")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", default="outputs/exp07_raw_background_109x6"
    )
    parser.add_argument("--reference", default="gt450")
    return parser.parse_args()


def flatten_moments(task, summary):
    row = {
        "task_id": task.task_id,
        "slide_id": task.slide_id,
        "tissue_type": task.tissue_type,
        "scanner": task.scanner,
        "sampled_pixels": summary["sampled_pixels"],
    }
    for space in SPACES:
        values = summary[space]
        for statistic in ("mean", "std", "p01", "p05", "p50", "p95", "p99"):
            for channel, value in zip(CHANNELS, values[statistic]):
                row[f"{space}_{statistic}_{channel}"] = value
        covariance = np.asarray(values["covariance"])
        for i, left in enumerate(CHANNELS):
            for j, right in enumerate(CHANNELS):
                row[f"{space}_cov_{left}{right}"] = covariance[i, j]
    return row


def pooled_moments(frame, space):
    counts = frame["sampled_pixels"].to_numpy(dtype=float)
    means = frame[[f"{space}_mean_{channel}" for channel in CHANNELS]].to_numpy()
    covariances = np.stack(
        [
            frame[
                [f"{space}_cov_{left}{right}" for left in CHANNELS]
            ].to_numpy()
            for right in CHANNELS
        ],
        axis=1,
    )
    total = counts.sum()
    mean = (counts[:, None] * means).sum(axis=0) / total
    scatter = np.zeros((3, 3), dtype=float)
    for count, cell_mean, covariance in zip(counts, means, covariances):
        delta = cell_mean - mean
        scatter += (count - 1) * covariance + count * np.outer(delta, delta)
    return mean, scatter / (total - 1), int(total)


def symmetric_root(matrix, inverse=False):
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 1e-12)
    if inverse:
        values = 1.0 / np.sqrt(values)
    else:
        values = np.sqrt(values)
    return (vectors * values[None, :]) @ vectors.T


def fit_background_transforms(moments, scanners):
    affine_rows = []
    for source in scanners:
        source_frame = moments[moments["scanner"] == source]
        for target in scanners:
            if source == target:
                continue
            target_frame = moments[moments["scanner"] == target]
            paired = source_frame.merge(
                target_frame,
                on="slide_id",
                suffixes=("_source", "_target"),
                validate="one_to_one",
            )
            for space in SPACES:
                for channel in CHANNELS:
                    x = paired[f"{space}_mean_{channel}_source"].to_numpy()
                    y = paired[f"{space}_mean_{channel}_target"].to_numpy()
                    slope, intercept = np.polyfit(x, y, 1)
                    predicted = intercept + slope * x
                    residual = y - predicted
                    denominator = np.square(y - y.mean()).sum()
                    r_squared = (
                        1.0 - np.square(residual).sum() / denominator
                        if denominator > 0
                        else np.nan
                    )
                    affine_rows.append(
                        {
                            "source": source,
                            "target": target,
                            "space": space,
                            "channel": channel,
                            "slope": slope,
                            "intercept": intercept,
                            "rmse": np.sqrt(np.mean(np.square(residual))),
                            "r_squared": r_squared,
                            "slides": len(paired),
                        }
                    )

    pooled = {}
    for scanner in scanners:
        frame = moments[moments["scanner"] == scanner]
        pooled[scanner] = {
            space: pooled_moments(frame, space) for space in SPACES
        }
    coral = []
    for source in scanners:
        for target in scanners:
            if source == target:
                continue
            for space in SPACES:
                source_mean, source_covariance, source_n = pooled[source][space]
                target_mean, target_covariance, target_n = pooled[target][space]
                ridge = 1e-6 * max(
                    np.trace(source_covariance) / 3.0,
                    np.trace(target_covariance) / 3.0,
                    1e-12,
                )
                source_regularized = source_covariance + ridge * np.eye(3)
                target_regularized = target_covariance + ridge * np.eye(3)
                # Row-vector convention: y = (x - mu_source) @ A + mu_target.
                transform = symmetric_root(
                    source_regularized, inverse=True
                ) @ symmetric_root(target_regularized)
                coral.append(
                    {
                        "source": source,
                        "target": target,
                        "space": space,
                        "row_vector_formula": "y = (x - source_mean) @ matrix + target_mean",
                        "source_mean": source_mean.tolist(),
                        "target_mean": target_mean.tolist(),
                        "source_covariance": source_covariance.tolist(),
                        "target_covariance": target_covariance.tolist(),
                        "matrix": transform.tolist(),
                        "ridge": ridge,
                        "source_sampled_pixels": source_n,
                        "target_sampled_pixels": target_n,
                    }
                )
    return pd.DataFrame(affine_rows), coral


def nps_ratios(spectra, scanners):
    plane = spectra[spectra["detrend"] == "plane"]
    wide = plane.pivot(
        index=["slide_id", "frequency_cyc_per_um"],
        columns="scanner",
        values="nps_od2_um2",
    )
    rows = []
    for source in scanners:
        for target in scanners:
            if source == target:
                continue
            log_power_ratio = np.log(
                np.maximum(wide[target].to_numpy(), 1e-30)
                / np.maximum(wide[source].to_numpy(), 1e-30)
            )
            frame = pd.DataFrame(
                {
                    "frequency_cyc_per_um": wide.index.get_level_values(
                        "frequency_cyc_per_um"
                    ),
                    "log_power_ratio": log_power_ratio,
                }
            )
            for frequency, values in frame.groupby("frequency_cyc_per_um"):
                amplitude = np.exp(0.5 * values["log_power_ratio"].to_numpy())
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "frequency_cyc_per_um": frequency,
                        "amplitude_ratio_median": np.median(amplitude),
                        "amplitude_ratio_q10": np.quantile(amplitude, 0.10),
                        "amplitude_ratio_q90": np.quantile(amplitude, 0.90),
                        "slides": len(amplitude),
                    }
                )
    return pd.DataFrame(rows)


def render_overview(metadata, moments, spectra, ratios, scanners, reference, output):
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axis = axes[0, 0]
    values = [
        metadata.loc[metadata.scanner == scanner, "acceptance_rate"].to_numpy()
        for scanner in scanners
    ]
    boxes = axis.boxplot(
        values,
        tick_labels=[s.upper() for s in scanners],
        patch_artist=True,
    )
    for box, scanner in zip(boxes["boxes"], scanners):
        box.set_facecolor(SCANNER_COLORS[scanner])
        box.set_alpha(0.65)
    axis.set_ylabel("Accepted / attempted")
    axis.set_title("A  Strict glass-QC acceptance rate")
    axis.grid(axis="y", alpha=0.2)

    axis = axes[0, 1]
    positions = np.arange(len(scanners))
    offsets = (-0.22, 0.0, 0.22)
    channel_colours = ("#ef4444", "#22c55e", "#3b82f6")
    for offset, channel, colour in zip(offsets, CHANNELS, channel_colours):
        grouped = [
            moments.loc[moments.scanner == scanner, f"rgb_p50_{channel}"].to_numpy()
            for scanner in scanners
        ]
        medians = [np.median(value) for value in grouped]
        q10 = [np.quantile(value, 0.10) for value in grouped]
        q90 = [np.quantile(value, 0.90) for value in grouped]
        axis.errorbar(
            positions + offset,
            medians,
            yerr=[np.asarray(medians) - q10, np.asarray(q90) - medians],
            fmt="o",
            color=colour,
            label=channel.upper(),
            capsize=2,
        )
    axis.set_xticks(positions, [s.upper() for s in scanners])
    axis.set_ylabel("Raw background digital value")
    axis.set_title("B  Background white point (median, slide q10–q90)")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)

    axis = axes[1, 0]
    plane = spectra[spectra["detrend"] == "plane"]
    for scanner in scanners:
        frame = plane[plane.scanner == scanner]
        curve = frame.groupby("frequency_cyc_per_um").nps_od2_um2.median()
        axis.plot(
            curve.index,
            curve.values,
            color=SCANNER_COLORS[scanner],
            label=scanner.upper(),
        )
    axis.set_yscale("log")
    axis.set_xlim(0.02, 0.98)
    axis.set_xlabel("Spatial frequency (cycles/µm)")
    axis.set_ylabel("Operational NPS (OD²·µm²)")
    axis.set_title("C  109-slide median background NPS")
    axis.legend(frameon=False, ncol=2, fontsize=8)
    axis.grid(alpha=0.15)

    axis = axes[1, 1]
    selected = ratios[ratios.target == reference]
    for scanner in scanners:
        if scanner == reference:
            continue
        frame = selected[selected.source == scanner]
        axis.plot(
            frame.frequency_cyc_per_um,
            frame.amplitude_ratio_median,
            color=SCANNER_COLORS[scanner],
            label=f"{scanner.upper()} → {reference.upper()}",
        )
        axis.fill_between(
            frame.frequency_cyc_per_um,
            frame.amplitude_ratio_q10,
            frame.amplitude_ratio_q90,
            color=SCANNER_COLORS[scanner],
            alpha=0.12,
        )
    axis.axhline(1.0, color="black", lw=1, ls="--")
    axis.set_yscale("log")
    axis.set_xlim(0.02, 0.98)
    axis.set_xlabel("Spatial frequency (cycles/µm)")
    axis.set_ylabel("Noise amplitude ratio")
    axis.set_title(f"D  Background NPS amplitude mapping to {reference.upper()}")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.15)

    figure.suptitle(
        "109-slide raw-background extraction and scanner calibration",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(figure, output, "figure_exp07_background_overview")


def main():
    args = parse_args()
    root = Path(args.input_root)
    output = root / "aggregate"
    output.mkdir(exist_ok=True)
    manifest = pd.read_csv(root / "manifest.csv")
    scanners = manifest.scanner.drop_duplicates().tolist()
    if args.reference not in scanners:
        raise ValueError(args.reference)

    metadata_rows = []
    moment_rows = []
    spectra_frames = []
    replicate_frames = []
    errors = []
    for task in manifest.itertuples(index=False):
        task_name = f"{task.task_id:04d}_{task.slide_id}_{task.scanner}"
        cell = root / "cells" / task_name
        try:
            if not (cell / "_SUCCESS").exists():
                raise FileNotFoundError(cell / "_SUCCESS")
            with (cell / "metadata.json").open() as handle:
                metadata = json.load(handle)
            with (cell / "background_color_summary.json").open() as handle:
                moments = json.load(handle)
            with h5py.File(cell / "glass_patches_at2_grid.h5", "r") as store:
                if store["rgb"].shape != (100, 256, 256, 3):
                    raise ValueError((task_name, store["rgb"].shape))
                if store["coords_native"].shape != (100, 2):
                    raise ValueError((task_name, store["coords_native"].shape))
            with h5py.File(cell / "nps_2d.h5", "r") as store:
                if set(store) != {"mean", "plane"}:
                    raise ValueError((task_name, list(store)))
                if any(store[key].shape != (256, 256) for key in store):
                    raise ValueError((task_name, "bad NPS shape"))
            accepted = pd.read_csv(cell / "accepted_glass_patches.csv")
            if len(accepted) != 100:
                raise ValueError((task_name, len(accepted)))
            metadata_rows.append(
                {
                    **{key: value for key, value in metadata.items() if not isinstance(value, (dict, list))},
                    "tissue_type": task.tissue_type,
                    "acceptance_rate": metadata["accepted_patches"]
                    / metadata["attempted_patches"],
                }
            )
            moment_rows.append(flatten_moments(task, moments))
            spectra = pd.read_csv(cell / "raw_glass_nps_spectra.csv")
            spectra.insert(0, "scanner", task.scanner)
            spectra.insert(0, "tissue_type", task.tissue_type)
            spectra.insert(0, "slide_id", task.slide_id)
            spectra.insert(0, "task_id", task.task_id)
            spectra_frames.append(spectra)
            replicate = pd.read_csv(cell / "raw_glass_nps_replicates.csv")
            replicate.insert(0, "scanner", task.scanner)
            replicate.insert(0, "tissue_type", task.tissue_type)
            replicate.insert(0, "slide_id", task.slide_id)
            replicate.insert(0, "task_id", task.task_id)
            replicate_frames.append(replicate)
        except Exception as error:
            errors.append({"task_id": task.task_id, "cell": str(cell), "error": repr(error)})

    validation = {
        "expected_tasks": len(manifest),
        "validated_tasks": len(metadata_rows),
        "errors": errors,
    }
    with (output / "validation.json").open("w") as handle:
        json.dump(validation, handle, indent=2)
    if errors:
        raise RuntimeError(f"validation failed for {len(errors)} cells")

    metadata = pd.DataFrame(metadata_rows)
    moments = pd.DataFrame(moment_rows)
    spectra = pd.concat(spectra_frames, ignore_index=True)
    replicates = pd.concat(replicate_frames, ignore_index=True)
    metadata.to_csv(output / "cell_metadata.csv", index=False)
    moments.to_csv(output / "background_moments.csv", index=False)
    spectra.to_csv(output / "raw_glass_nps_spectra.csv.gz", index=False, compression="gzip")
    replicates.to_csv(
        output / "raw_glass_nps_replicates.csv.gz", index=False, compression="gzip"
    )

    affine, coral = fit_background_transforms(moments, scanners)
    affine.to_csv(output / "background_mean_affine_transforms.csv", index=False)
    with (output / "background_coral_transforms.json").open("w") as handle:
        json.dump(coral, handle, indent=2)
    ratios = nps_ratios(spectra, scanners)
    ratios.to_csv(output / "background_nps_amplitude_ratios.csv", index=False)
    figures = render_overview(
        metadata,
        moments,
        spectra,
        ratios,
        scanners,
        args.reference,
        output,
    )
    with (output / "summary.json").open("w") as handle:
        json.dump(
            {
                "slides": int(manifest.slide_id.nunique()),
                "scanners": scanners,
                "cells": len(metadata),
                "patches": int(metadata.accepted_patches.sum()),
                "reference_scanner": args.reference,
                "background_transformation_scope": (
                    "near-white background calibration only; not a tissue colour, "
                    "stain, registration, PSF, or MTF transform"
                ),
                "figures": figures,
            },
            handle,
            indent=2,
        )
    print(metadata.groupby("scanner").acceptance_rate.describe().to_string(), flush=True)
    print(f"[exp07-aggregate] validated {len(metadata)}/{len(manifest)} cells", flush=True)


if __name__ == "__main__":
    main()
