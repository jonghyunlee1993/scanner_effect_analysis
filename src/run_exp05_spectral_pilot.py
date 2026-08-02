"""Paired-scanner spectral pilot with Exp-02 HF-gain controls.

This is an exploratory effective-transfer analysis, not an absolute MTF
measurement.  It reads registered WSIs on their common AT2 canvas, samples
coordinates present in every scanner's curated tissue grid, estimates radial
OD power ratios relative to AT2, and renders the curves beside the exact
Laplacian-detail gain family used by Exp-02's blur/sharpen trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import openslide
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prenorm.exp01.frequency import FixedLaplacianPyramid
from utils import align


SCANNERS = ("at2", "gt450", "versa", "akoya", "s60", "s360")
SCANNER_COLORS = {
    "at2": "#374151",
    "gt450": "#F59E0B",
    "versa": "#10B981",
    "akoya": "#8B5CF6",
    "s60": "#0EA5E9",
    "s360": "#EC4899",
}
SCANNER_MARKERS = {
    "at2": "o",
    "gt450": "s",
    "versa": "^",
    "akoya": "D",
    "s60": "P",
    "s360": "X",
}
HF_GAINS = (0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0)
BANDS = {
    "low_mid": (0.10, 0.30),
    "mid": (0.30, 0.60),
    "high": (0.60, 0.90),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=(
            "/mnt/isilon/oldridge_lab/batch_effects/pan_normal/"
            "registered_ref_at2_all"
        ),
    )
    parser.add_argument(
        "--output", default="outputs/exp05_spectral_pilot_5slides"
    )
    parser.add_argument("--slides", type=int, default=5)
    parser.add_argument(
        "--slide-id",
        default=None,
        help="Measure one explicit registry slide (used by the cohort job array).",
    )
    parser.add_argument("--patches-per-slide", type=int, default=100)
    parser.add_argument(
        "--replicate-groups",
        type=int,
        default=5,
        help="Split sampled patches into this many independent spectrum estimates.",
    )
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--align-margin", type=int, default=32)
    parser.add_argument("--align-radius", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Write measurement tables only; cohort aggregation renders figures.",
    )
    return parser.parse_args(argv)


def curated_path(root: Path, scanner: str, slide_id: str) -> Path:
    return (
        root
        / "registered_curated_features"
        / scanner
        / "20x_256px_0px_overlap"
        / "patches"
        / f"{slide_id}_patches.h5"
    )


def common_curated_coords(root: Path, slide_id: str) -> np.ndarray:
    coordinate_sets = []
    for scanner in SCANNERS:
        path = curated_path(root, scanner, slide_id)
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            coords = np.asarray(handle["coords"], dtype=np.int64)
        coordinate_sets.append({(int(x), int(y)) for x, y in coords})
    common = sorted(set.intersection(*coordinate_sets))
    if not common:
        raise ValueError(f"no six-scanner curated-coordinate intersection: {slide_id}")
    return np.asarray(common, dtype=np.int64)


def frequency_geometry(size: int, mpp: float, bins: int):
    one_d = np.fft.fftfreq(size) / mpp
    radius = np.sqrt(one_d[:, None] ** 2 + one_d[None, :] ** 2)
    nyquist = 1.0 / (2.0 * mpp)
    edges = np.linspace(0.0, nyquist, bins + 1)
    index = np.digitize(radius.ravel(), edges) - 1
    valid = (index >= 0) & (index < bins)
    counts = np.bincount(index[valid], minlength=bins).astype(np.float64)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return index, valid, counts, centres, nyquist


def radial_mean(value, index, valid, counts):
    return np.bincount(
        index[valid], weights=np.asarray(value).ravel()[valid], minlength=len(counts)
    ) / np.maximum(counts, 1.0)


def total_od_fft(rgb: np.ndarray, window: np.ndarray) -> np.ndarray:
    value = -np.log((rgb.astype(np.float32) + 1.0) / 256.0).mean(axis=2)
    value -= value.mean()
    return np.fft.fft2(value * window)


def batch_power(rgb: np.ndarray, window: np.ndarray) -> np.ndarray:
    value = -np.log((rgb.astype(np.float32) + 1.0) / 256.0).mean(axis=3)
    value -= value.mean(axis=(1, 2), keepdims=True)
    fourier = np.fft.fft2(value * window[None, :, :], axes=(-2, -1))
    return np.abs(fourier) ** 2


def normalized_transfer(power, reference, frequency):
    ratio = np.sqrt(
        np.maximum(np.asarray(power), 1e-20)
        / np.maximum(np.asarray(reference), 1e-20)
    )
    normalization = (frequency >= 0.03) & (frequency <= 0.10)
    scale = np.exp(np.mean(np.log(np.maximum(ratio[normalization], 1e-20))))
    return ratio / scale


def geometric_band(curve, frequency, bounds):
    selected = (frequency >= bounds[0]) & (frequency < bounds[1])
    return float(np.exp(np.mean(np.log(np.maximum(curve[selected], 1e-20)))))


@torch.no_grad()
def intervention_power(images, window, pyramid, gains):
    tensor = (
        torch.from_numpy(images.astype(np.float32))
        .permute(0, 3, 1, 2)
        .div(127.5)
        .sub(1.0)
    )
    low, bands = pyramid.decompose(tensor)
    result = {}
    for gain in gains:
        transformed = pyramid.reconstruct(
            low, [float(gain) * band for band in bands]
        ).clamp(-1.0, 1.0)
        rgb = (
            transformed.add(1.0)
            .mul(127.5)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        result[float(gain)] = batch_power(rgb, window).mean(axis=0)
    return result


def select_slides(registry: pd.DataFrame, count: int, rng) -> list[str]:
    eligible = []
    for row in registry.itertuples():
        if not bool(row.paired_all_scanners):
            continue
        if all(Path(getattr(row, f"{scanner}_path")).exists() for scanner in SCANNERS):
            eligible.append(str(row.slide_id))
    eligible = sorted(set(eligible))
    if len(eligible) < count:
        raise ValueError(f"requested {count} slides but only {len(eligible)} are eligible")
    return [str(value) for value in rng.choice(eligible, count, replace=False)]


def scanner_path_map(registry: pd.DataFrame, slide_id: str) -> dict[str, str]:
    selected = registry[registry["slide_id"].astype(str) == str(slide_id)]
    if len(selected) != 1:
        raise ValueError(f"expected one registry row for {slide_id}, got {len(selected)}")
    row = selected.iloc[0]
    return {scanner: str(row[f"{scanner}_path"]) for scanner in SCANNERS}


def measure_slide(
    root,
    registry,
    slide_id,
    coords,
    patch_size,
    bins,
    pyramid,
    align_margin,
    align_radius,
    replicate_groups,
):
    paths = scanner_path_map(registry, slide_id)
    slides = {scanner: openslide.OpenSlide(path) for scanner, path in paths.items()}
    try:
        dimensions = {scanner: slide.dimensions for scanner, slide in slides.items()}
        if len(set(dimensions.values())) != 1:
            raise ValueError(f"registered dimensions differ for {slide_id}: {dimensions}")
        mpp = float(slides["at2"].properties["openslide.mpp-x"])
        index, valid, counts, frequency, nyquist = frequency_geometry(
            patch_size, mpp, bins
        )
        window_1d = np.hanning(patch_size).astype(np.float32)
        window = window_1d[:, None] * window_1d[None, :]
        power_sum = {
            scanner: np.zeros((patch_size, patch_size), np.float64)
            for scanner in SCANNERS
        }
        replicate_power_sum = [
            {
                scanner: np.zeros((patch_size, patch_size), np.float64)
                for scanner in SCANNERS
            }
            for _ in range(replicate_groups)
        ]
        replicate_counts = np.zeros(replicate_groups, dtype=np.int64)
        cross_sum = {
            scanner: np.zeros((patch_size, patch_size), np.complex128)
            for scanner in SCANNERS
        }
        at2_images = np.empty(
            (len(coords), patch_size, patch_size, 3), dtype=np.uint8
        )
        alignment = []
        for patch_index, (x, y) in enumerate(coords):
            fourier = {}
            reference_rgb = np.asarray(
                slides["at2"]
                .read_region(
                    (int(x), int(y)), 0, (patch_size, patch_size)
                )
                .convert("RGB"),
                dtype=np.uint8,
            )
            at2_images[patch_index] = reference_rgb
            fourier["at2"] = total_od_fft(reference_rgb, window)
            power_sum["at2"] += np.abs(fourier["at2"]) ** 2
            alignment.append({
                "slide_id": slide_id,
                "patch_index": patch_index,
                "scanner": "at2",
                "dy": 0,
                "dx": 0,
                "ncc": 1.0,
            })
            for scanner in SCANNERS[1:]:
                extended_size = patch_size + 2 * align_margin
                extended = np.asarray(
                    slides[scanner]
                    .read_region(
                        (int(x) - align_margin, int(y) - align_margin),
                        0,
                        (extended_size, extended_size),
                    )
                    .convert("RGB"),
                    dtype=np.uint8,
                )
                ncc = align.ncc_map(extended, reference_rgb)
                dy, dx, score = align.peak(
                    ncc,
                    align_margin,
                    around=(0, 0),
                    radius=align_radius,
                )
                rgb = align.crop_at(
                    extended, dy, dx, patch_size, align_margin
                )
                fourier[scanner] = total_od_fft(rgb, window)
                power_sum[scanner] += np.abs(fourier[scanner]) ** 2
                alignment.append({
                    "slide_id": slide_id,
                    "patch_index": patch_index,
                    "scanner": scanner,
                    "dy": dy,
                    "dx": dx,
                    "ncc": score,
                })
            for scanner in SCANNERS:
                cross_sum[scanner] += fourier[scanner] * np.conj(fourier["at2"])
                replicate_power_sum[patch_index % replicate_groups][scanner] += (
                    np.abs(fourier[scanner]) ** 2
                )
            replicate_counts[patch_index % replicate_groups] += 1

        count = float(len(coords))
        mean_power_2d = {scanner: value / count for scanner, value in power_sum.items()}
        radial_power = {
            scanner: radial_mean(value, index, valid, counts)
            for scanner, value in mean_power_2d.items()
        }
        spectra = []
        bands = []
        reference = radial_power["at2"]
        for scanner in SCANNERS:
            transfer = normalized_transfer(radial_power[scanner], reference, frequency)
            coherence_2d = np.divide(
                np.abs(cross_sum[scanner] / count) ** 2,
                mean_power_2d[scanner] * mean_power_2d["at2"],
                out=np.zeros_like(mean_power_2d[scanner]),
                where=(mean_power_2d[scanner] * mean_power_2d["at2"]) > 1e-20,
            )
            coherence = np.clip(
                radial_mean(coherence_2d, index, valid, counts), 0.0, 1.0
            )
            for bin_index, value in enumerate(frequency):
                spectra.append({
                    "slide_id": slide_id,
                    "scanner": scanner,
                    "frequency_cyc_per_um": float(value),
                    "relative_transfer": float(transfer[bin_index]),
                    "log2_relative_transfer": float(np.log2(max(transfer[bin_index], 1e-20))),
                    "coherence_to_at2": float(coherence[bin_index]),
                    "radial_power": float(radial_power[scanner][bin_index]),
                })
            for name, bounds in BANDS.items():
                value = geometric_band(transfer, frequency, bounds)
                bands.append({
                    "slide_id": slide_id,
                    "scanner": scanner,
                    "band": name,
                    "lower_frequency": bounds[0],
                    "upper_frequency": bounds[1],
                    "relative_transfer": value,
                    "log2_relative_transfer": float(np.log2(value)),
                })

        replicate_bands = []
        for replicate_index, replicate_count in enumerate(replicate_counts):
            if replicate_count == 0:
                raise ValueError(
                    f"{slide_id}: replicate group {replicate_index} is empty"
                )
            replicate_radial = {
                scanner: radial_mean(
                    replicate_power_sum[replicate_index][scanner]
                    / float(replicate_count),
                    index,
                    valid,
                    counts,
                )
                for scanner in SCANNERS
            }
            replicate_reference = replicate_radial["at2"]
            for scanner in SCANNERS:
                transfer = normalized_transfer(
                    replicate_radial[scanner], replicate_reference, frequency
                )
                for name, bounds in BANDS.items():
                    value = geometric_band(transfer, frequency, bounds)
                    replicate_bands.append({
                        "slide_id": slide_id,
                        "scanner": scanner,
                        "replicate": int(replicate_index),
                        "patches_in_replicate": int(replicate_count),
                        "band": name,
                        "lower_frequency": bounds[0],
                        "upper_frequency": bounds[1],
                        "relative_transfer": value,
                        "log2_relative_transfer": float(np.log2(value)),
                    })

        intervention_2d = intervention_power(
            at2_images, window, pyramid, HF_GAINS
        )
        intervention_radial = {
            gain: radial_mean(value, index, valid, counts)
            for gain, value in intervention_2d.items()
        }
        intervention_reference = intervention_radial[1.0]
        interventions = []
        for gain in HF_GAINS:
            transfer = normalized_transfer(
                intervention_radial[gain], intervention_reference, frequency
            )
            family = "raw" if gain == 1.0 else ("blur" if gain < 1.0 else "sharpen")
            parameter = 0.0 if gain == 1.0 else (
                1.0 - gain if gain < 1.0 else gain - 1.0
            )
            for bin_index, value in enumerate(frequency):
                interventions.append({
                    "slide_id": slide_id,
                    "family": family,
                    "parameter": float(parameter),
                    "effective_hf_gain": float(gain),
                    "frequency_cyc_per_um": float(value),
                    "relative_transfer": float(transfer[bin_index]),
                    "log2_relative_transfer": float(np.log2(max(transfer[bin_index], 1e-20))),
                })
        return (
            spectra,
            bands,
            replicate_bands,
            interventions,
            alignment,
            mpp,
            nyquist,
        )
    finally:
        for slide in slides.values():
            slide.close()


def feature_coordinates(frame: pd.DataFrame):
    frequency = frame["frequency_cyc_per_um"].to_numpy(dtype=float)
    log_curve = frame["log2_relative_transfer"].to_numpy(dtype=float)
    mid = (frequency >= 0.10) & (frequency < 0.40)
    high = (frequency >= 0.55) & (frequency < 0.90)
    return float(log_curve[mid].mean()), float(log_curve[high].mean())


def project_to_hf_gain(spectra, interventions):
    rows = []
    dense_gain = np.linspace(min(HF_GAINS), max(HF_GAINS), 401)
    for slide_id, scanner_frame in spectra.groupby("slide_id", sort=False):
        local_intervention = interventions[interventions["slide_id"] == slide_id]
        gains = np.asarray(sorted(local_intervention["effective_hf_gain"].unique()))
        intervention_curves = []
        for gain in gains:
            selected = local_intervention[
                np.isclose(local_intervention["effective_hf_gain"], gain)
            ].sort_values("frequency_cyc_per_um")
            intervention_curves.append(
                selected["log2_relative_transfer"].to_numpy(dtype=float)
            )
        intervention_curves = np.asarray(intervention_curves)
        frequency = selected["frequency_cyc_per_um"].to_numpy(dtype=float)
        dense_curves = np.stack([
            np.interp(dense_gain, gains, intervention_curves[:, column])
            for column in range(intervention_curves.shape[1])
        ], axis=1)
        fit_band = (frequency >= 0.10) & (frequency < 0.90)
        for scanner, frame in scanner_frame.groupby("scanner", sort=False):
            frame = frame.sort_values("frequency_cyc_per_um")
            observed = frame["log2_relative_transfer"].to_numpy(dtype=float)
            error = np.sqrt(
                np.mean((dense_curves[:, fit_band] - observed[None, fit_band]) ** 2, axis=1)
            )
            best = int(np.argmin(error))
            mid, high = feature_coordinates(frame)
            rows.append({
                "slide_id": slide_id,
                "scanner": scanner,
                "equivalent_hf_gain": float(dense_gain[best]),
                "off_axis_rms_log2": float(error[best]),
                "mid_log2_transfer": mid,
                "high_log2_transfer": high,
            })
    return pd.DataFrame(rows)


def save_figure(figure, output_dir: Path, stem: str):
    paths = []
    for suffix, kwargs in (("png", {"dpi": 200}), ("pdf", {})):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(figure)
    return paths


def aggregate_curve(frame, value):
    return (
        frame.groupby("frequency_cyc_per_um", sort=True)[value]
        .agg(["mean", "std"])
        .reset_index()
    )


def render_overview(spectra, bands, slide_order, output_dir):
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axis = axes[0]
    for scanner in SCANNERS[1:]:
        aggregate = aggregate_curve(
            spectra[spectra["scanner"] == scanner], "log2_relative_transfer"
        )
        x = aggregate["frequency_cyc_per_um"].to_numpy()
        mean = aggregate["mean"].to_numpy()
        std = aggregate["std"].fillna(0).to_numpy()
        axis.plot(x, mean, color=SCANNER_COLORS[scanner], lw=2.2, label=scanner.upper())
        axis.fill_between(x, mean - std, mean + std, color=SCANNER_COLORS[scanner], alpha=0.15)
    axis.axhline(0.0, color="#6B7280", lw=1, ls="--")
    axis.set(xlim=(0.03, 0.95), xlabel="Spatial frequency (cycles/µm)", ylabel="Relative transfer to AT2 (log₂)")
    axis.set_title("A  Paired-tissue effective transfer")
    axis.legend(frameon=False, ncol=2, fontsize=9)

    axis = axes[1]
    for scanner in SCANNERS[1:]:
        aggregate = aggregate_curve(
            spectra[spectra["scanner"] == scanner], "coherence_to_at2"
        )
        x = aggregate["frequency_cyc_per_um"].to_numpy()
        mean = aggregate["mean"].to_numpy()
        std = aggregate["std"].fillna(0).to_numpy()
        axis.plot(x, mean, color=SCANNER_COLORS[scanner], lw=2.0, label=scanner.upper())
        axis.fill_between(x, np.maximum(0, mean - std), np.minimum(1, mean + std), color=SCANNER_COLORS[scanner], alpha=0.12)
    axis.set(xlim=(0.03, 0.95), ylim=(0, 1.02), xlabel="Spatial frequency (cycles/µm)", ylabel="Magnitude-squared coherence")
    axis.set_title("B  Linear shared-detail agreement")

    high = bands[bands["band"] == "high"].pivot(
        index="scanner", columns="slide_id", values="log2_relative_transfer"
    ).reindex(index=SCANNERS[1:], columns=slide_order)
    axis = axes[2]
    limit = max(1.0, float(np.nanmax(np.abs(high.to_numpy()))))
    image = axis.imshow(high, cmap="coolwarm", aspect="auto", vmin=-limit, vmax=limit)
    for row in range(high.shape[0]):
        for column in range(high.shape[1]):
            value = high.iloc[row, column]
            axis.text(column, row, f"{value:+.2f}", ha="center", va="center", fontsize=8, color="white" if abs(value) > 0.55 * limit else "black")
    axis.set_xticks(range(len(slide_order)), slide_order, rotation=45, ha="right")
    axis.set_yticks(range(len(SCANNERS) - 1), [s.upper() for s in SCANNERS[1:]])
    axis.set_title("C  High band (0.60–0.90 cycles/µm)")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("log₂ transfer to AT2")
    figure.suptitle("Five-slide spectral pilot · 100 common tissue patches per slide", fontsize=15, y=1.03)
    figure.tight_layout()
    return save_figure(figure, output_dir, "figure1_spectral_overview")


def render_slide_curves(spectra, slide_order, output_dir):
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for axis, slide_id in zip(axes.flat, slide_order):
        selected = spectra[spectra["slide_id"] == slide_id]
        for scanner in SCANNERS[1:]:
            frame = selected[selected["scanner"] == scanner].sort_values(
                "frequency_cyc_per_um"
            )
            axis.plot(
                frame["frequency_cyc_per_um"],
                frame["log2_relative_transfer"],
                color=SCANNER_COLORS[scanner],
                lw=1.8,
            )
        axis.axhline(0.0, color="#9CA3AF", lw=0.8, ls="--")
        axis.set_title(slide_id)
        axis.set_xlim(0.03, 0.95)
        axis.grid(alpha=0.15)
    axes.flat[-1].axis("off")
    figure.supxlabel("Spatial frequency (cycles/µm)")
    figure.supylabel("Relative transfer to AT2 (log₂)")
    handles = [
        Line2D([0], [0], color=SCANNER_COLORS[s], lw=2, label=s.upper())
        for s in SCANNERS[1:]
    ]
    axes.flat[-1].legend(
        handles=handles,
        loc="center",
        ncol=2,
        frameon=False,
        title="Scanner",
    )
    figure.suptitle("Slide-level transfer curves: stability versus tissue dependence", fontsize=15)
    figure.tight_layout(rect=(0, 0.02, 1, 0.96))
    return save_figure(figure, output_dir, "figure2_slide_transfer_curves")


def render_blur_sharpen_bridge(spectra, interventions, projection, output_dir):
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    axis = axes[0]
    displayed_gains = (0.25, 0.5, 1.0, 1.5, 2.0)
    for gain in displayed_gains:
        frame = interventions[np.isclose(interventions["effective_hf_gain"], gain)]
        aggregate = aggregate_curve(frame, "log2_relative_transfer")
        color = "#2563EB" if gain < 1 else ("#DC2626" if gain > 1 else "#111827")
        axis.plot(
            aggregate["frequency_cyc_per_um"], aggregate["mean"],
            color=color, lw=1.2, ls="--", alpha=0.55,
            label=f"Exp-02 gain {gain:g}",
        )
    for scanner in SCANNERS[1:]:
        aggregate = aggregate_curve(
            spectra[spectra["scanner"] == scanner], "log2_relative_transfer"
        )
        axis.plot(
            aggregate["frequency_cyc_per_um"], aggregate["mean"],
            color=SCANNER_COLORS[scanner], lw=2.2, label=scanner.upper(),
        )
    axis.axhline(0, color="#9CA3AF", lw=0.8)
    axis.set(
        xlim=(0.03, 0.95),
        ylim=(-3.0, 1.35),
        xlabel="Spatial frequency (cycles/µm)",
        ylabel="Relative transfer (log₂)",
    )
    axis.text(
        0.05,
        -2.75,
        "Exp-02 gain 0 (complete detail removal) falls below this panel",
        fontsize=7.5,
        color="#2563EB",
    )
    axis.set_title("A  Scanner curves versus Exp-02 controls")
    axis.legend(frameon=False, fontsize=7.5, ncol=2)

    axis = axes[1]
    trajectory_rows = []
    for (slide_id, gain), frame in interventions.groupby(
        ["slide_id", "effective_hf_gain"], sort=True
    ):
        mid, high = feature_coordinates(frame.sort_values("frequency_cyc_per_um"))
        trajectory_rows.append({"slide_id": slide_id, "gain": gain, "mid": mid, "high": high})
    trajectory = pd.DataFrame(trajectory_rows)
    trajectory_mean = (
        trajectory[trajectory["gain"] > 0]
        .groupby("gain")[["mid", "high"]]
        .mean()
        .reset_index()
    )
    axis.plot(trajectory_mean["mid"], trajectory_mean["high"], color="#6B7280", lw=2.0, zorder=1)
    for row in trajectory_mean.itertuples():
        color = "#2563EB" if row.gain < 1 else ("#DC2626" if row.gain > 1 else "#111827")
        axis.scatter(row.mid, row.high, s=48, color=color, zorder=3)
        axis.annotate(f"g={row.gain:g}", (row.mid, row.high), xytext=(4, 4), textcoords="offset points", fontsize=7, color=color)
    for scanner in SCANNERS[1:]:
        frame = projection[projection["scanner"] == scanner]
        axis.scatter(frame["mid_log2_transfer"], frame["high_log2_transfer"], marker=SCANNER_MARKERS[scanner], color=SCANNER_COLORS[scanner], s=35, alpha=0.28)
        mean = frame[["mid_log2_transfer", "high_log2_transfer"]].mean()
        axis.scatter(mean.iloc[0], mean.iloc[1], marker=SCANNER_MARKERS[scanner], color=SCANNER_COLORS[scanner], edgecolor="white", linewidth=0.8, s=110, label=scanner.upper(), zorder=4)
        axis.annotate(scanner.upper(), (mean.iloc[0], mean.iloc[1]), xytext=(6, -9), textcoords="offset points", fontsize=8, color=SCANNER_COLORS[scanner], weight="bold")
    axis.axhline(0, color="#D1D5DB", lw=0.8)
    axis.axvline(0, color="#D1D5DB", lw=0.8)
    axis.set(
        xlim=(-1.15, 0.35),
        ylim=(-2.75, 1.15),
        xlabel="0.10–0.40 cycles/µm (mean log₂ transfer)",
        ylabel="0.55–0.90 cycles/µm (mean log₂ transfer)",
    )
    axis.text(
        -1.1,
        -2.55,
        "gain 0 endpoint is outside the displayed range",
        fontsize=7.5,
        color="#2563EB",
    )
    axis.set_title("B  Is scanner variation one HF-gain axis?")

    axis = axes[2]
    for scanner in SCANNERS[1:]:
        frame = projection[projection["scanner"] == scanner]
        axis.scatter(frame["equivalent_hf_gain"], frame["off_axis_rms_log2"], marker=SCANNER_MARKERS[scanner], color=SCANNER_COLORS[scanner], s=40, alpha=0.30)
        mean = frame[["equivalent_hf_gain", "off_axis_rms_log2"]].mean()
        axis.scatter(mean.iloc[0], mean.iloc[1], marker=SCANNER_MARKERS[scanner], color=SCANNER_COLORS[scanner], edgecolor="white", linewidth=0.8, s=120, zorder=4)
        axis.annotate(scanner.upper(), (mean.iloc[0], mean.iloc[1]), xytext=(6, 2), textcoords="offset points", fontsize=8, color=SCANNER_COLORS[scanner], weight="bold")
    axis.axvline(1.0, color="#9CA3AF", lw=1, ls="--")
    axis.set(xlim=(-0.05, 2.05), xlabel="Nearest Exp-02 effective HF gain", ylabel="Off-axis RMS (log₂ transfer)")
    axis.set_title("C  Equivalent gain and unexplained shape")

    figure.suptitle("Bridge to the existing blur/sharpen experiment", fontsize=15, y=1.04)
    figure.text(
        0.5,
        -0.01,
        "Prior Exp-02 endpoints: gain 0 / Blur 1.0 → self-cosine 0.065, retrieval 0.67; "
        "gain 2 / Sharpen 1.0 → self-cosine 0.828, retrieval 1.00, scanner radius 0.095→0.120.",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return save_figure(figure, output_dir, "figure3_blur_sharpen_bridge")


def validate_outputs(
    spectra,
    replicate_bands,
    interventions,
    projection,
    selected_slides,
    patches,
    alignment,
    expected_slides,
    patches_per_slide,
    replicate_groups,
):
    if (
        len(selected_slides) != expected_slides
        or len(set(selected_slides)) != expected_slides
    ):
        raise AssertionError(
            f"pilot must contain {expected_slides} unique slides"
        )
    if not np.isfinite(spectra.select_dtypes(include=[np.number])).all().all():
        raise AssertionError("non-finite spectrum output")
    at2 = spectra[spectra["scanner"] == "at2"]["relative_transfer"].to_numpy()
    if not np.allclose(at2, 1.0, atol=1e-10):
        raise AssertionError("AT2 self-transfer is not one")
    raw = interventions[np.isclose(interventions["effective_hf_gain"], 1.0)][
        "relative_transfer"
    ].to_numpy()
    if not np.allclose(raw, 1.0, atol=2e-4):
        raise AssertionError("Exp-02 gain-one reconstruction is not identity")
    if (
        projection.groupby("scanner")["slide_id"].nunique().min()
        != expected_slides
    ):
        raise AssertionError("projection is missing scanner-slide cells")
    expected_patches = expected_slides * patches_per_slide
    if len(patches) != expected_patches:
        raise AssertionError(
            f"expected {expected_patches} selected patches, got {len(patches)}"
        )
    if len(alignment) != expected_patches * len(SCANNERS):
        raise AssertionError("alignment output is incomplete")
    expected_replicate_rows = (
        expected_slides * len(SCANNERS) * len(BANDS) * replicate_groups
    )
    if len(replicate_bands) != expected_replicate_rows:
        raise AssertionError(
            f"expected {expected_replicate_rows} replicate-band rows, "
            f"got {len(replicate_bands)}"
        )


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.data_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = (
        root / "registered_images" / "qc_registered_eval" / "registered_pair_summary.csv"
    )
    registry = pd.read_csv(registry_path)
    rng = np.random.default_rng(args.seed)
    if args.slide_id is None:
        selected_slides = select_slides(registry, args.slides, rng)
    else:
        eligible = set(select_slides(registry, int(registry.paired_all_scanners.sum()), rng))
        if str(args.slide_id) not in eligible:
            raise ValueError(f"slide is not six-scanner eligible: {args.slide_id}")
        selected_slides = [str(args.slide_id)]
    expected_slides = len(selected_slides)
    if args.replicate_groups < 1:
        raise ValueError("--replicate-groups must be positive")
    if args.patches_per_slide < args.replicate_groups:
        raise ValueError("patch count must be at least the number of replicate groups")
    print(f"[exp05] selected slides: {', '.join(selected_slides)}", flush=True)

    pyramid = FixedLaplacianPyramid(levels=4)
    spectra_rows = []
    band_rows = []
    replicate_band_rows = []
    intervention_rows = []
    alignment_rows = []
    patch_rows = []
    mpp_values = {}
    nyquist_values = {}
    for slide_index, slide_id in enumerate(selected_slides, start=1):
        common = common_curated_coords(root, slide_id)
        at2_path = scanner_path_map(registry, slide_id)["at2"]
        at2_slide = openslide.OpenSlide(at2_path)
        try:
            width, height = at2_slide.dimensions
        finally:
            at2_slide.close()
        common = common[
            (common[:, 0] >= args.align_margin)
            & (common[:, 1] >= args.align_margin)
            & (
                common[:, 0] + args.patch_size + args.align_margin
                <= width
            )
            & (
                common[:, 1] + args.patch_size + args.align_margin
                <= height
            )
        ]
        if len(common) < args.patches_per_slide:
            raise ValueError(
                f"{slide_id}: only {len(common)} common coordinates for "
                f"{args.patches_per_slide} requested patches"
            )
        digest = hashlib.sha256(
            f"{args.seed}:{slide_id}".encode("utf-8")
        ).digest()
        slide_rng = np.random.default_rng(
            int.from_bytes(digest[:8], "little", signed=False)
        )
        selected_index = slide_rng.choice(
            len(common), args.patches_per_slide, replace=False
        )
        coords = common[selected_index]
        for patch_index, (x, y) in enumerate(coords):
            patch_rows.append({
                "slide_id": slide_id,
                "patch_index": patch_index,
                "x": int(x),
                "y": int(y),
                "common_coordinate_pool": len(common),
            })
        print(
            f"[exp05] {slide_index}/{len(selected_slides)} {slide_id}: "
            f"{args.patches_per_slide} of {len(common)} six-scanner common coordinates",
            flush=True,
        )
        (
            spectra,
            bands,
            replicate_bands,
            interventions,
            alignment_rows_slide,
            mpp,
            nyquist,
        ) = measure_slide(
            root,
            registry,
            slide_id,
            coords,
            args.patch_size,
            args.bins,
            pyramid,
            args.align_margin,
            args.align_radius,
            args.replicate_groups,
        )
        spectra_rows.extend(spectra)
        band_rows.extend(bands)
        replicate_band_rows.extend(replicate_bands)
        intervention_rows.extend(interventions)
        alignment_rows.extend(alignment_rows_slide)
        mpp_values[slide_id] = mpp
        nyquist_values[slide_id] = nyquist

    spectra = pd.DataFrame(spectra_rows)
    bands = pd.DataFrame(band_rows)
    replicate_bands = pd.DataFrame(replicate_band_rows)
    interventions = pd.DataFrame(intervention_rows)
    patches = pd.DataFrame(patch_rows)
    alignment_frame = pd.DataFrame(alignment_rows)
    projection = project_to_hf_gain(spectra, interventions)
    validate_outputs(
        spectra,
        replicate_bands,
        interventions,
        projection,
        selected_slides,
        patches,
        alignment_frame,
        expected_slides,
        args.patches_per_slide,
        args.replicate_groups,
    )

    paths = {
        "spectra": output_dir / "slide_spectra.csv",
        "bands": output_dir / "slide_band_summary.csv",
        "replicate_bands": output_dir / "slide_band_replicates.csv",
        "interventions": output_dir / "exp02_hf_gain_spectra.csv",
        "projection": output_dir / "scanner_hf_gain_projection.csv",
        "patches": output_dir / "selected_patches.csv",
        "alignment": output_dir / "patch_alignment.csv",
    }
    spectra.to_csv(paths["spectra"], index=False)
    bands.to_csv(paths["bands"], index=False)
    replicate_bands.to_csv(paths["replicate_bands"], index=False)
    interventions.to_csv(paths["interventions"], index=False)
    projection.to_csv(paths["projection"], index=False)
    patches.to_csv(paths["patches"], index=False)
    alignment_frame.to_csv(paths["alignment"], index=False)

    figures = []
    if not args.skip_figures:
        figures.extend(render_overview(spectra, bands, selected_slides, output_dir))
        figures.extend(render_slide_curves(spectra, selected_slides, output_dir))
        figures.extend(render_blur_sharpen_bridge(spectra, interventions, projection, output_dir))

    scanner_summary = (
        projection.groupby("scanner")
        .agg(
            equivalent_hf_gain_mean=("equivalent_hf_gain", "mean"),
            equivalent_hf_gain_sd=("equivalent_hf_gain", "std"),
            off_axis_rms_log2_mean=("off_axis_rms_log2", "mean"),
            off_axis_rms_log2_sd=("off_axis_rms_log2", "std"),
            mid_log2_transfer_mean=("mid_log2_transfer", "mean"),
            high_log2_transfer_mean=("high_log2_transfer", "mean"),
        )
        .reset_index()
    )
    scanner_summary.to_csv(output_dir / "scanner_summary.csv", index=False)
    summary = {
        "analysis": "paired_registered_scene_process_effective_transfer",
        "absolute_mtf_claim": False,
        "seed": args.seed,
        "selected_slides": selected_slides,
        "patches_per_slide": args.patches_per_slide,
        "replicate_groups": args.replicate_groups,
        "scanners": list(SCANNERS),
        "common_coordinate_policy": "intersection of all six curated tissue grids",
        "channel": "mean total optical density with per-patch mean removal",
        "window": "2D Hann",
        "patch_alignment": {
            "method": "low-pass NCC integer crop; no interpolation",
            "margin_px": args.align_margin,
            "search_radius_px": args.align_radius,
            "ncc_by_scanner": (
                alignment_frame.groupby("scanner")["ncc"]
                .agg(["mean", "median", "min"])
                .reset_index()
                .to_dict("records")
            ),
        },
        "mpp_from_registered_at2": mpp_values,
        "nyquist_cyc_per_um": nyquist_values,
        "exp02_control": {
            "definition": "FixedLaplacianPyramid detail-band gain with low band fixed",
            "effective_hf_gains": list(HF_GAINS),
            "documented_endpoints": {
                "gain_0_blur_1": {
                    "self_cosine": 0.065,
                    "cross_scanner_retrieval": 0.67,
                    "consensus_gain": -0.781,
                },
                "gain_2_sharpen_1": {
                    "self_cosine": 0.828,
                    "cross_scanner_retrieval": 1.0,
                    "scanner_radius_raw": 0.095,
                    "scanner_radius_sharpen": 0.120,
                },
            },
        },
        "tables": {key: str(value) for key, value in paths.items()},
        "scanner_summary": scanner_summary.to_dict("records"),
        "figures": figures,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(scanner_summary.to_string(index=False), flush=True)
    print(f"[exp05] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
