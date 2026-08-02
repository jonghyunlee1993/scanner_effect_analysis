"""Synthetic 2D alias audit for the retained VALIS/libvips resampling chain.

The audit probes exact full-resolution linear maps recovered from historical
VALIS registrar objects.  It compares the original one-pass bicubic affine with
an explicit anti-aliased chain: Lanczos3 reduction at the smallest affine
singular value, followed by the residual bicubic affine.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCANNERS = ("gt450", "versa")
PIPELINES = ("original_bicubic", "explicit_aa_lanczos3")
PRIMARY_BAND = (0.60, 0.90)
ALIAS_POWER_RATIO_LIMIT = 0.05


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provenance",
        default="outputs/e0_resampling_provenance/transform_provenance.csv",
    )
    parser.add_argument("--output", default="outputs/e0_alias_audit")
    parser.add_argument("--source-size", type=int, default=640)
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--frequency-step", type=float, default=0.04)
    parser.add_argument("--output-bin-width", type=float, default=0.025)
    parser.add_argument("--angles", type=int, default=8)
    parser.add_argument("--phases", type=int, default=2)
    parser.add_argument("--noise-replicates", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def row_matrix(row) -> np.ndarray:
    return np.array(
        [[row.forward_xx, row.forward_xy], [row.forward_yx, row.forward_yy]],
        dtype=float,
    )


def select_transform_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Select observed transforms closest to q05/q50/q95 effective scale."""

    selected = []
    for scanner, group in frame.groupby("scanner", sort=False):
        group = group.sort_values("slide_id").copy()
        scale = group["output_px_per_native_px_geom"].astype(float)
        used = set()
        for label, quantile in (("q05", 0.05), ("q50", 0.50), ("q95", 0.95)):
            target = float(scale.quantile(quantile))
            order = (scale - target).abs().sort_values(kind="stable").index
            index = next((idx for idx in order if idx not in used), order[0])
            used.add(index)
            row = group.loc[index].copy()
            row["transform_profile"] = label
            row["profile_target_scale"] = target
            selected.append(row)
    return pd.DataFrame(selected).reset_index(drop=True)


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height < size or width < size:
        raise ValueError(f"warped image {image.shape} is smaller than {size} px crop")
    y = (height - size) // 2
    x = (width - size) // 2
    return np.asarray(image[y : y + size, x : x + size], dtype=np.float64)


def warp_synthetic(source: np.ndarray, matrix: np.ndarray, pipeline: str, output_size: int):
    import pyvips

    image = pyvips.Image.new_from_array(np.asarray(source, dtype=np.float32))
    interpolator = pyvips.Interpolate.new("bicubic")
    if pipeline == "original_bicubic":
        residual = matrix
    elif pipeline == "explicit_aa_lanczos3":
        pre_scale = float(np.linalg.svd(matrix, compute_uv=False).min())
        if pre_scale < 1.0:
            image = image.resize(pre_scale, kernel="lanczos3")
            residual = matrix / pre_scale
        else:
            residual = matrix
    else:
        raise ValueError(f"unknown pipeline: {pipeline}")
    warped = image.affine(
        residual.reshape(-1).tolist(),
        interpolate=interpolator,
        premultiplied=True,
    )
    return center_crop(np.asarray(warped), output_size)


def radial_periodogram(image: np.ndarray, mpp: float, edges: np.ndarray, input_variance=1.0):
    size = image.shape[0]
    if image.shape != (size, size):
        raise ValueError(f"expected square image, got {image.shape}")
    centered = np.asarray(image, dtype=np.float64) - float(np.mean(image))
    window_1d = np.hanning(size)
    window = np.outer(window_1d, window_1d)
    window_power = float(np.mean(window**2))
    spectrum = np.fft.fftshift(np.fft.fft2(centered * window))
    power = np.abs(spectrum) ** 2 / (size**4 * window_power * input_variance)
    frequency = np.fft.fftshift(np.fft.fftfreq(size, d=mpp))
    fy, fx = np.meshgrid(frequency, frequency, indexing="ij")
    radius = np.sqrt(fx**2 + fy**2)
    index = np.digitize(radius.ravel(), edges) - 1
    valid = (index >= 0) & (index < len(edges) - 1)
    radial = np.bincount(
        index[valid], weights=power.ravel()[valid], minlength=len(edges) - 1
    )
    return radial


def alias_ratio_from_matrix(
    mixing: np.ndarray,
    input_frequency: np.ndarray,
    output_centers: np.ndarray,
    target_nyquist: float,
    band=PRIMARY_BAND,
):
    output_mask = (output_centers >= band[0]) & (output_centers < band[1])
    inband_input = (input_frequency >= band[0]) & (input_frequency < band[1])
    alias_input = input_frequency > target_nyquist
    radial_weight = input_frequency
    true_power = float(
        np.sum(mixing[inband_input][:, output_mask] * radial_weight[inband_input, None])
    )
    alias_power = float(
        np.sum(mixing[alias_input][:, output_mask] * radial_weight[alias_input, None])
    )
    ratio = alias_power / true_power if true_power > 0 else math.inf
    fraction = alias_power / (alias_power + true_power) if alias_power + true_power > 0 else math.nan
    return {
        "sinusoid_true_inband_power": true_power,
        "sinusoid_alias_power": alias_power,
        "sinusoid_alias_to_inband_ratio": ratio,
        "sinusoid_alias_fraction": fraction,
    }


def bandlimited_noise(rng, size: int, mpp: float, lower: float, upper: float):
    white = rng.standard_normal((size, size))
    spectrum = np.fft.fft2(white)
    frequency = np.fft.fftfreq(size, d=mpp)
    fy, fx = np.meshgrid(frequency, frequency, indexing="ij")
    radius = np.sqrt(fx**2 + fy**2)
    mask = (radius >= lower) & (radius < upper)
    return np.fft.ifft2(spectrum * mask).real


def apply_sinusoid_sweep(
    profiles,
    source_size,
    output_size,
    frequency_step,
    output_bin_width,
    n_angles,
    n_phases,
):
    response_rows = []
    long_rows = []
    matrices = {}
    for scanner, scanner_profiles in profiles.groupby("scanner", sort=False):
        native_mpp = float(scanner_profiles["native_mpp"].median())
        target_mpp = float(scanner_profiles["target_mpp"].median())
        source_nyquist = 1.0 / (2.0 * native_mpp)
        target_nyquist = 1.0 / (2.0 * target_mpp)
        input_frequency = np.arange(
            frequency_step, source_nyquist - frequency_step / 2.0, frequency_step
        )
        output_edges = np.arange(
            0.0, target_nyquist + output_bin_width * 1.001, output_bin_width
        )
        output_centers = (output_edges[:-1] + output_edges[1:]) / 2.0
        angles = np.arange(n_angles) * np.pi / n_angles
        # A sign-flipped sinusoid has identical sampling phase for a power audit.
        # Span one half-cycle so the default probes cosine and quadrature phases.
        phases = np.arange(n_phases) * np.pi / n_phases
        axis = (np.arange(source_size) - (source_size - 1) / 2.0) * native_mpp
        yy, xx = np.meshgrid(axis, axis, indexing="ij")

        accumulators = {}
        counts = defaultdict(int)
        for profile in scanner_profiles.itertuples(index=False):
            for pipeline in PIPELINES:
                key = (str(profile.transform_profile), pipeline)
                accumulators[key] = np.zeros(
                    (len(input_frequency), len(output_centers)), dtype=np.float64
                )

        for input_index, frequency in enumerate(input_frequency):
            for angle_index, angle in enumerate(angles):
                coordinate = xx * np.cos(angle) + yy * np.sin(angle)
                for phase_index, phase in enumerate(phases):
                    source = np.cos(2.0 * np.pi * frequency * coordinate + phase)
                    input_variance = float(np.var(source))
                    for profile in scanner_profiles.itertuples(index=False):
                        matrix = row_matrix(profile)
                        for pipeline in PIPELINES:
                            warped = warp_synthetic(source, matrix, pipeline, output_size)
                            radial = radial_periodogram(
                                warped, target_mpp, output_edges, input_variance=input_variance
                            )
                            key = (str(profile.transform_profile), pipeline)
                            accumulators[key][input_index] += radial
                            counts[key] += 1
                            response_rows.append(
                                {
                                    "scanner": scanner,
                                    "transform_profile": str(profile.transform_profile),
                                    "slide_id": str(profile.slide_id),
                                    "pipeline": pipeline,
                                    "input_frequency_cyc_per_um": float(frequency),
                                    "input_angle_deg": float(np.degrees(angle)),
                                    "phase_index": int(phase_index),
                                    "output_variance_transfer": float(
                                        np.var(warped) / input_variance
                                    ),
                                    "output_band_power_transfer": float(
                                        radial[
                                            (output_centers >= PRIMARY_BAND[0])
                                            & (output_centers < PRIMARY_BAND[1])
                                        ].sum()
                                    ),
                                    "peak_output_frequency_cyc_per_um": float(
                                        output_centers[int(np.argmax(radial))]
                                    ),
                                }
                            )
        normalizer = float(n_angles * n_phases)
        for profile in scanner_profiles.itertuples(index=False):
            for pipeline in PIPELINES:
                key = (str(profile.transform_profile), pipeline)
                mixing = accumulators[key] / normalizer
                matrices[(scanner, *key)] = {
                    "mixing": mixing,
                    "input_frequency": input_frequency,
                    "output_edges": output_edges,
                    "output_centers": output_centers,
                    "source_nyquist": source_nyquist,
                    "target_nyquist": target_nyquist,
                    "slide_id": str(profile.slide_id),
                }
                for i, input_value in enumerate(input_frequency):
                    for j, output_value in enumerate(output_centers):
                        long_rows.append(
                            {
                                "scanner": scanner,
                                "transform_profile": str(profile.transform_profile),
                                "slide_id": str(profile.slide_id),
                                "pipeline": pipeline,
                                "input_frequency_cyc_per_um": float(input_value),
                                "output_frequency_cyc_per_um": float(output_value),
                                "output_power_per_input_variance": float(mixing[i, j]),
                            }
                        )
    return pd.DataFrame(response_rows), pd.DataFrame(long_rows), matrices


def apply_noise_audit(profiles, source_size, output_size, output_bin_width, replicates, seed):
    rows = []
    for scanner, scanner_profiles in profiles.groupby("scanner", sort=False):
        native_mpp = float(scanner_profiles["native_mpp"].median())
        target_mpp = float(scanner_profiles["target_mpp"].median())
        source_nyquist = 1.0 / (2.0 * native_mpp)
        target_nyquist = 1.0 / (2.0 * target_mpp)
        edges = np.arange(0.0, target_nyquist + output_bin_width * 1.001, output_bin_width)
        centers = (edges[:-1] + edges[1:]) / 2.0
        band_mask = (centers >= PRIMARY_BAND[0]) & (centers < PRIMARY_BAND[1])
        for replicate in range(replicates):
            rng = np.random.default_rng(seed + 1000 * SCANNERS.index(scanner) + replicate)
            components = {
                "true_inband": bandlimited_noise(
                    rng, source_size, native_mpp, PRIMARY_BAND[0], PRIMARY_BAND[1]
                ),
                "supra_target_nyquist": bandlimited_noise(
                    rng, source_size, native_mpp, target_nyquist, source_nyquist
                ),
            }
            for profile in scanner_profiles.itertuples(index=False):
                matrix = row_matrix(profile)
                for pipeline in PIPELINES:
                    for component, source in components.items():
                        warped = warp_synthetic(source, matrix, pipeline, output_size)
                        radial = radial_periodogram(warped, target_mpp, edges)
                        rows.append(
                            {
                                "scanner": scanner,
                                "transform_profile": str(profile.transform_profile),
                                "slide_id": str(profile.slide_id),
                                "pipeline": pipeline,
                                "replicate": replicate,
                                "input_component": component,
                                "source_variance": float(np.var(source)),
                                "output_band_power": float(radial[band_mask].sum()),
                            }
                        )
    return pd.DataFrame(rows)


def apply_edge_audit(profiles, source_size, output_size):
    rows = []
    axis = np.arange(source_size) - (source_size - 1) / 2.0
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    for profile in profiles.itertuples(index=False):
        matrix = row_matrix(profile)
        for edge_angle in np.arange(8) * np.pi / 8.0:
            source = (
                xx * np.cos(edge_angle) + yy * np.sin(edge_angle) >= 0
            ).astype(np.float64)
            for pipeline in PIPELINES:
                warped = warp_synthetic(source, matrix, pipeline, output_size)
                rows.append(
                    {
                        "scanner": str(profile.scanner),
                        "transform_profile": str(profile.transform_profile),
                        "slide_id": str(profile.slide_id),
                        "pipeline": pipeline,
                        "edge_angle_deg": float(np.degrees(edge_angle)),
                        "overshoot": float(max(0.0, np.max(warped) - 1.0)),
                        "undershoot": float(max(0.0, -np.min(warped))),
                        "range_expansion": float(max(0.0, np.max(warped) - np.min(warped) - 1.0)),
                    }
                )
    return pd.DataFrame(rows)


def summarize(profiles, response, noise, edge, matrices):
    rows = []
    for profile in profiles.itertuples(index=False):
        scanner = str(profile.scanner)
        transform_profile = str(profile.transform_profile)
        for pipeline in PIPELINES:
            key = (scanner, transform_profile, pipeline)
            matrix_data = matrices[key]
            alias = alias_ratio_from_matrix(
                matrix_data["mixing"],
                matrix_data["input_frequency"],
                matrix_data["output_centers"],
                matrix_data["target_nyquist"],
            )
            subset = response[
                response["scanner"].eq(scanner)
                & response["transform_profile"].eq(transform_profile)
                & response["pipeline"].eq(pipeline)
            ]
            passband = subset[
                subset["input_frequency_cyc_per_um"].ge(PRIMARY_BAND[0])
                & subset["input_frequency_cyc_per_um"].lt(PRIMARY_BAND[1])
            ]
            by_angle = passband.groupby(
                ["input_frequency_cyc_per_um", "input_angle_deg"], as_index=False
            )["output_variance_transfer"].mean()
            anisotropy = by_angle.groupby("input_frequency_cyc_per_um")[
                "output_variance_transfer"
            ].agg(["min", "max"])
            anisotropy_db = 10.0 * np.log10(anisotropy["max"] / anisotropy["min"])

            noise_subset = noise[
                noise["scanner"].eq(scanner)
                & noise["transform_profile"].eq(transform_profile)
                & noise["pipeline"].eq(pipeline)
            ]
            pivot = noise_subset.pivot(
                index="replicate", columns="input_component", values="output_band_power"
            )
            noise_ratio = pivot["supra_target_nyquist"] / pivot["true_inband"]
            edge_subset = edge[
                edge["scanner"].eq(scanner)
                & edge["transform_profile"].eq(transform_profile)
                & edge["pipeline"].eq(pipeline)
            ]
            sine_ratio = alias["sinusoid_alias_to_inband_ratio"]
            white_ratio = float(noise_ratio.mean())
            rows.append(
                {
                    "scanner": scanner,
                    "transform_profile": transform_profile,
                    "slide_id": str(profile.slide_id),
                    "pipeline": pipeline,
                    "native_mpp": float(profile.native_mpp),
                    "target_mpp": float(profile.target_mpp),
                    "native_px_per_output_px": float(profile.native_px_per_output_px),
                    "affine_rotation_deg": float(profile.affine_rotation_deg),
                    "affine_anisotropy_ratio": float(profile.affine_anisotropy_ratio),
                    **alias,
                    "white_noise_alias_to_inband_ratio_mean": white_ratio,
                    "white_noise_alias_to_inband_ratio_max": float(noise_ratio.max()),
                    "passband_amplitude_retention_median": float(
                        np.sqrt(passband["output_variance_transfer"].median())
                    ),
                    "passband_amplitude_retention_min": float(
                        np.sqrt(passband["output_variance_transfer"].min())
                    ),
                    "passband_angular_anisotropy_db_median": float(anisotropy_db.median()),
                    "passband_angular_anisotropy_db_max": float(anisotropy_db.max()),
                    "edge_overshoot_max": float(edge_subset["overshoot"].max()),
                    "edge_undershoot_max": float(edge_subset["undershoot"].max()),
                    "alias_power_ratio_limit": ALIAS_POWER_RATIO_LIMIT,
                    "alias_gate_pass": bool(
                        sine_ratio <= ALIAS_POWER_RATIO_LIMIT
                        and white_ratio <= ALIAS_POWER_RATIO_LIMIT
                    ),
                }
            )
    return pd.DataFrame(rows)


def render_figure(summary, matrices, output: Path):
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for row_index, scanner in enumerate(SCANNERS):
        for column_index, pipeline in enumerate(PIPELINES):
            data = matrices[(scanner, "q50", pipeline)]
            image = np.log10(np.maximum(data["mixing"], 1e-10))
            axis = axes[row_index, column_index]
            shown = axis.imshow(
                image,
                origin="lower",
                aspect="auto",
                extent=[
                    data["output_edges"][0],
                    data["output_edges"][-1],
                    data["input_frequency"][0],
                    data["input_frequency"][-1],
                ],
                vmin=-7,
                vmax=-1,
                cmap="magma",
            )
            axis.axvspan(*PRIMARY_BAND, color="cyan", alpha=0.10)
            axis.axhline(data["target_nyquist"], color="white", linestyle="--", linewidth=1)
            short_pipeline = "original bicubic" if pipeline == PIPELINES[0] else "explicit AA"
            axis.set_title(f"{scanner.upper()} · {short_pipeline}")
            axis.set_xlabel("Output frequency (cycles/µm)")
            axis.set_ylabel("Input frequency (cycles/µm)")
        subset = summary[summary["scanner"].eq(scanner)]
        pivot = subset.pivot(
            index="transform_profile",
            columns="pipeline",
            values="white_noise_alias_to_inband_ratio_mean",
        ).reindex(["q05", "q50", "q95"])
        axis = axes[row_index, 2]
        x = np.arange(len(pivot))
        width = 0.36
        for offset, pipeline in zip((-width / 2, width / 2), PIPELINES):
            axis.bar(x + offset, pivot[pipeline], width=width, label=pipeline)
        axis.axhline(ALIAS_POWER_RATIO_LIMIT, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(x, pivot.index)
        axis.set_yscale("log")
        axis.set_ylabel("Alias / true in-band power")
        axis.set_title(f"{scanner.upper()} broadband validation")
        axis.legend(fontsize=8)
    figure.colorbar(
        shown,
        ax=axes[:, :2],
        orientation="horizontal",
        pad=0.07,
        fraction=0.05,
        label="log10 output power / input variance",
    )
    figure.savefig(output / "figure_e0_alias_audit.png", dpi=220)
    figure.savefig(output / "figure_e0_alias_audit.pdf")
    plt.close(figure)


def main():
    args = parse_args()
    provenance = pd.read_csv(args.provenance, dtype={"slide_id": str})
    provenance = provenance[provenance["scanner"].isin(SCANNERS)].copy()
    if set(provenance["scanner"]) != set(SCANNERS):
        raise ValueError(f"missing scanner provenance: {set(SCANNERS) - set(provenance['scanner'])}")
    profiles = select_transform_profiles(provenance)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(output / "selected_transform_profiles.csv", index=False)

    response, mixing_long, matrices = apply_sinusoid_sweep(
        profiles,
        args.source_size,
        args.output_size,
        args.frequency_step,
        args.output_bin_width,
        args.angles,
        args.phases,
    )
    noise = apply_noise_audit(
        profiles,
        args.source_size,
        args.output_size,
        args.output_bin_width,
        args.noise_replicates,
        args.seed,
    )
    edge = apply_edge_audit(profiles, args.source_size, args.output_size)
    summary = summarize(profiles, response, noise, edge, matrices)

    response.to_csv(output / "sinusoid_response.csv", index=False)
    mixing_long.to_csv(output / "mixing_matrix_long.csv", index=False)
    noise.to_csv(output / "white_noise_response.csv", index=False)
    edge.to_csv(output / "edge_response.csv", index=False)
    summary.to_csv(output / "scanner_profile_summary.csv", index=False)
    for key, data in matrices.items():
        scanner, profile, pipeline = key
        np.savez_compressed(
            output / f"mixing_{scanner}_{profile}_{pipeline}.npz",
            mixing=data["mixing"],
            input_frequency=data["input_frequency"],
            output_edges=data["output_edges"],
            output_centers=data["output_centers"],
            source_nyquist=data["source_nyquist"],
            target_nyquist=data["target_nyquist"],
            slide_id=data["slide_id"],
        )
    render_figure(summary, matrices, output)

    route_gate = (
        summary.groupby("pipeline")["alias_gate_pass"].all().to_dict()
    )
    original_pass = bool(route_gate["original_bicubic"])
    aa_pass = bool(route_gate["explicit_aa_lanczos3"])
    if original_pass:
        decision = "retain_original_grid_and_primary_0.60_0.90_band"
    elif aa_pass:
        decision = "rebuild_common_grid_with_explicit_antialias_then_retain_0.60_0.90_band"
    else:
        decision = "do_not_lock_primary_high_band; redesign_filter_or_restrict_band"
    run_summary = {
        "analysis": "e0_2d_alias_audit",
        "scanners": list(SCANNERS),
        "transform_profiles_per_scanner": ["q05", "q50", "q95"],
        "sinusoid_angles": args.angles,
        "sinusoid_phases": args.phases,
        "source_size": args.source_size,
        "output_size": args.output_size,
        "primary_band_cycles_per_um": list(PRIMARY_BAND),
        "alias_power_ratio_limit": ALIAS_POWER_RATIO_LIMIT,
        "gate_requires": "both sinusoid mixing and broadband white-noise ratio <= limit for all profiles",
        "original_pipeline_all_profiles_pass": original_pass,
        "explicit_aa_pipeline_all_profiles_pass": aa_pass,
        "decision": decision,
        "original_chain": "single-pass libvips bicubic affine; no explicit prefilter",
        "explicit_aa_chain": (
            "libvips Lanczos3 reduction at min affine singular value, then residual bicubic affine"
        ),
        "translation_handling": "omitted because phase sweep covers translation for the linear shift-invariant audit",
    }
    (output / "summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
