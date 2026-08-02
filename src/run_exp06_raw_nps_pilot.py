"""Sample scanner-native glass patches and estimate operational background NPS.

Glass coordinates are selected independently for each raw WSI.  Every accepted
native patch covers the same physical field of view and is area-resampled to
the AT2 analysis grid (256 px at 0.5052 micrometres/px), making the resulting
NPS directly compatible with the registered-domain Exp-05 tissue periodogram.

This is an operational background NPS (glass + scanner pipeline + compression),
not an IEC detector NPS measurement under controlled uniform exposure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_exp05_spectral_pilot import (
    SCANNERS,
    SCANNER_COLORS,
    radial_mean,
    save_figure,
)
from utils.align import flat_frac


RAW_EXTENSIONS = {
    "at2": ".svs",
    "gt450": ".svs",
    "versa": ".svs",
    "akoya": ".qptiff",
    "s60": ".ndpi",
    "s360": ".ndpi",
}
DEFAULT_SLIDES = (
    "2-8_7",
    "8-12_9",
    "12.5_13",
    "8-12_30",
    "8-12_19",
)
QC_THRESHOLDS = {
    "gray_mean_min": 205.0,
    "gray_p05_min": 175.0,
    "gray_std_min": 0.35,
    "gray_std_max": 20.0,
    "saturation_p95_max": 50.0,
    "stained_fraction_max": 0.0002,
    "dark_fraction_max": 0.002,
    "dark_outlier_fraction_10_max": 0.003,
    "dark_outlier_fraction_20_max": 0.0005,
    "dark_component_fraction_15_max": 0.0003,
    "chromatic_outlier_fraction_15_max": 0.0002,
    "chromatic_component_fraction_15_max": 0.0001,
    "exact_white_fraction_max": 0.980,
    "flat_fraction_max": 0.995,
    "gradient_mean_max": 8.0,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default=(
            "/mnt/isilon/oldridge_lab/batch_effects/pan_normal/raw/raw_images"
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/exp06_raw_nps_pilot_5slides",
    )
    parser.add_argument(
        "--slides",
        nargs="+",
        default=list(DEFAULT_SLIDES),
    )
    parser.add_argument(
        "--scanners",
        nargs="+",
        default=list(SCANNERS),
        choices=list(SCANNERS),
    )
    parser.add_argument("--patches-per-slide", type=int, default=100)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--target-mpp", type=float, default=0.5052)
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--attempts-per-cell", type=int, default=100)
    parser.add_argument("--fallback-attempts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args(argv)


def raw_path(root: Path, scanner: str, slide_id: str):
    path = root / scanner / f"{slide_id}{RAW_EXTENSIONS[scanner]}"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def scanner_mpp(root: Path, scanner: str, path: Path, slide):
    value = slide.properties.get("openslide.mpp-x")
    if value is not None:
        numeric = float(value)
        if np.isfinite(numeric) and 0.1 < numeric < 2.0:
            return numeric, "openslide.mpp-x"
    resolution_path = root / scanner / f"{scanner}_resolution_data.csv"
    if resolution_path.exists():
        resolution = pd.read_csv(resolution_path)
        selected = resolution[resolution["wsi"].astype(str) == path.name]
        if len(selected) == 1:
            return float(selected.iloc[0]["mpp"]), "resolution_data.csv"
    raise ValueError(f"no valid native mpp: {scanner}/{path.name}")


def patch_statistics(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    stained = (saturation >= 31) & (gray <= 220)
    median_gray = float(np.median(gray))
    median_rgb = np.median(rgb.reshape(-1, 3), axis=0)
    rgb_delta = rgb.astype(np.float32) - median_rgb[None, None, :]
    luminance_delta = rgb_delta.mean(axis=2, keepdims=True)
    chromatic_residual = np.sqrt(
        np.square(rgb_delta - luminance_delta).sum(axis=2)
    )
    dark_10 = gray < median_gray - 10.0
    dark_15 = gray < median_gray - 15.0
    dark_20 = gray < median_gray - 20.0
    chromatic_15 = chromatic_residual > 15.0

    def largest_component_fraction(mask):
        count, _, component_stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if count <= 1:
            return 0.0
        return float(
            component_stats[1:, cv2.CC_STAT_AREA].max() / mask.size
        )

    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "gray_p05": float(np.quantile(gray, 0.05)),
        "gray_p95": float(np.quantile(gray, 0.95)),
        "saturation_mean": float(saturation.mean()),
        "saturation_p95": float(np.quantile(saturation, 0.95)),
        "stained_fraction": float(stained.mean()),
        "dark_fraction": float((gray < 160).mean()),
        "dark_outlier_fraction_10": float(dark_10.mean()),
        "dark_outlier_fraction_20": float(dark_20.mean()),
        "dark_component_fraction_15": largest_component_fraction(dark_15),
        "chromatic_outlier_fraction_15": float(chromatic_15.mean()),
        "chromatic_component_fraction_15": largest_component_fraction(
            chromatic_15
        ),
        "exact_white_fraction": float((rgb >= 254).all(axis=2).mean()),
        "flat_fraction": float(flat_frac(rgb, win=5, thresh=0.5)),
        "gradient_mean": float(gradient.mean()),
        "gradient_p95": float(np.quantile(gradient, 0.95)),
    }


def glass_rejection_reasons(stats):
    reasons = []
    comparisons = (
        (stats["gray_mean"] < QC_THRESHOLDS["gray_mean_min"], "dim_mean"),
        (stats["gray_p05"] < QC_THRESHOLDS["gray_p05_min"], "dim_tail"),
        (stats["gray_std"] < QC_THRESHOLDS["gray_std_min"], "too_flat_std"),
        (stats["gray_std"] > QC_THRESHOLDS["gray_std_max"], "high_std"),
        (
            stats["saturation_p95"]
            > QC_THRESHOLDS["saturation_p95_max"],
            "saturated_colour",
        ),
        (
            stats["stained_fraction"]
            > QC_THRESHOLDS["stained_fraction_max"],
            "stain",
        ),
        (
            stats["dark_fraction"] > QC_THRESHOLDS["dark_fraction_max"],
            "dark_pixels",
        ),
        (
            stats["dark_outlier_fraction_10"]
            > QC_THRESHOLDS["dark_outlier_fraction_10_max"],
            "local_dark_outliers",
        ),
        (
            stats["dark_outlier_fraction_20"]
            > QC_THRESHOLDS["dark_outlier_fraction_20_max"],
            "deep_dark_outliers",
        ),
        (
            stats["dark_component_fraction_15"]
            > QC_THRESHOLDS["dark_component_fraction_15_max"],
            "connected_dark_debris",
        ),
        (
            stats["chromatic_outlier_fraction_15"]
            > QC_THRESHOLDS["chromatic_outlier_fraction_15_max"],
            "chromatic_outliers",
        ),
        (
            stats["chromatic_component_fraction_15"]
            > QC_THRESHOLDS["chromatic_component_fraction_15_max"],
            "connected_chromatic_debris",
        ),
        (
            stats["exact_white_fraction"]
            > QC_THRESHOLDS["exact_white_fraction_max"],
            "clipped_white",
        ),
        (
            stats["flat_fraction"] > QC_THRESHOLDS["flat_fraction_max"],
            "synthetic_flat",
        ),
        (
            stats["gradient_mean"] > QC_THRESHOLDS["gradient_mean_max"],
            "edges_or_debris",
        ),
    )
    for failed, name in comparisons:
        if failed:
            reasons.append(name)
    return reasons


def stable_rng(seed, slide_id, scanner):
    digest = hashlib.sha256(
        f"{seed}:{slide_id}:{scanner}".encode("utf-8")
    ).digest()
    return np.random.default_rng(
        int.from_bytes(digest[:8], "little", signed=False)
    )


def read_target_patch(slide, x, y, native_size, target_size):
    image = np.asarray(
        slide.read_region(
            (int(x), int(y)), 0, (native_size, native_size)
        ).convert("RGB"),
        dtype=np.uint8,
    )
    if native_size != target_size:
        image = cv2.resize(
            image,
            (target_size, target_size),
            interpolation=cv2.INTER_AREA,
        )
    return image


def candidate_coordinates(rng, width, height, patch_size, grid_size, attempts_per_cell):
    coordinates = []
    cells = [(row, column) for row in range(grid_size) for column in range(grid_size)]
    for _ in range(attempts_per_cell):
        for cell_index in rng.permutation(len(cells)):
            row, column = cells[cell_index]
            x0 = int(column * width / grid_size)
            x1 = int((column + 1) * width / grid_size) - patch_size
            y0 = int(row * height / grid_size)
            y1 = int((row + 1) * height / grid_size) - patch_size
            if x1 <= x0:
                continue
            if y1 <= y0:
                continue
            coordinates.append((
                int(rng.integers(x0, x1 + 1)),
                int(rng.integers(y0, y1 + 1)),
                int(row),
                int(column),
            ))
    return coordinates


def sample_glass_patches(
    slide,
    scanner,
    slide_id,
    native_mpp,
    target_mpp,
    target_size,
    requested,
    grid_size,
    attempts_per_cell,
    fallback_attempts,
    rng,
):
    native_size = max(
        16, int(round(target_size * target_mpp / native_mpp))
    )
    width, height = slide.dimensions
    candidates = candidate_coordinates(
        rng,
        width,
        height,
        native_size,
        grid_size,
        attempts_per_cell,
    )
    for _ in range(fallback_attempts):
        candidates.append((
            int(rng.integers(0, width - native_size + 1)),
            int(rng.integers(0, height - native_size + 1)),
            -1,
            -1,
        ))

    accepted_images = []
    accepted_rows = []
    candidate_rows = []
    rejection_counts = Counter()
    used = set()
    for attempt, (x, y, grid_row, grid_column) in enumerate(candidates):
        key = (x, y)
        if key in used:
            continue
        used.add(key)
        rgb = read_target_patch(slide, x, y, native_size, target_size)
        stats = patch_statistics(rgb)
        reasons = glass_rejection_reasons(stats)
        accepted = not reasons
        row = {
            "slide_id": slide_id,
            "scanner": scanner,
            "attempt": attempt,
            "x_native": x,
            "y_native": y,
            "x_normalized": x / max(width - native_size, 1),
            "y_normalized": y / max(height - native_size, 1),
            "grid_row": grid_row,
            "grid_column": grid_column,
            "native_patch_size": native_size,
            "native_mpp": native_mpp,
            "target_patch_size": target_size,
            "target_mpp": target_mpp,
            "accepted": accepted,
            "rejection_reasons": ";".join(reasons),
            **stats,
        }
        candidate_rows.append(row)
        if accepted:
            row = dict(row)
            row["patch_index"] = len(accepted_images)
            accepted_rows.append(row)
            accepted_images.append(rgb)
            if len(accepted_images) >= requested:
                break
        else:
            rejection_counts.update(reasons)
    images = (
        np.stack(accepted_images)
        if accepted_images
        else np.empty((0, target_size, target_size, 3), dtype=np.uint8)
    )
    return (
        images,
        accepted_rows,
        candidate_rows,
        rejection_counts,
        native_size,
    )


def frequency_geometry(size, mpp, bins):
    one_d = np.fft.fftfreq(size, d=mpp)
    radius = np.sqrt(one_d[:, None] ** 2 + one_d[None, :] ** 2)
    nyquist = 1.0 / (2.0 * mpp)
    edges = np.linspace(0.0, nyquist, bins + 1)
    index = np.digitize(radius.ravel(), edges) - 1
    valid = (index >= 0) & (index < bins)
    counts = np.bincount(index[valid], minlength=bins).astype(float)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return index, valid, counts, centres


def od_values(images):
    return -np.log10((images.astype(np.float32) + 1.0) / 256.0).mean(axis=3)


def detrend(values, method):
    values = values.astype(np.float32, copy=False)
    mean = values.mean(axis=(1, 2), keepdims=True)
    result = values - mean
    if method == "mean":
        return result
    size = values.shape[-1]
    coordinate = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    x = coordinate[None, None, :]
    y = coordinate[None, :, None]
    x_norm = float(np.square(coordinate).sum() * size)
    y_norm = x_norm
    slope_x = (result * x).sum(axis=(1, 2), keepdims=True) / x_norm
    slope_y = (result * y).sum(axis=(1, 2), keepdims=True) / y_norm
    return result - slope_x * x - slope_y * y


def estimate_nps(images, mpp, bins, replicate_groups=5):
    size = images.shape[1]
    index, valid, counts, frequency = frequency_geometry(size, mpp, bins)
    window_1d = np.hanning(size).astype(np.float32)
    window = window_1d[:, None] * window_1d[None, :]
    window_power = float(np.square(window).mean())
    scale = (mpp * mpp) / (size * size * window_power)
    result = {}
    for method in ("mean", "plane"):
        total_2d = np.zeros((size, size), dtype=np.float64)
        per_patch_radial = []
        values = od_values(images)
        for start in range(0, len(images), 10):
            batch = detrend(values[start : start + 10], method)
            fourier = np.fft.fft2(
                batch * window[None, :, :], axes=(-2, -1)
            )
            periodogram = scale * np.abs(fourier) ** 2
            total_2d += periodogram.sum(axis=0)
            for item in periodogram:
                per_patch_radial.append(
                    radial_mean(item, index, valid, counts)
                )
        mean_2d = total_2d / len(images)
        per_patch_radial = np.asarray(per_patch_radial)
        replicate_rows = []
        for group in range(min(replicate_groups, len(images))):
            selected = np.arange(len(images)) % replicate_groups == group
            if selected.any():
                replicate_rows.append(per_patch_radial[selected].mean(axis=0))
        result[method] = {
            "frequency": frequency,
            "mean_2d": mean_2d,
            "per_patch_radial": per_patch_radial,
            "mean_radial": per_patch_radial.mean(axis=0),
            "replicate_radial": np.asarray(replicate_rows),
            "window_power": window_power,
        }
    return result


def render_montage(montage, scanners, output_dir, slide_id):
    columns = 8
    figure, axes = plt.subplots(
        len(scanners), columns, figsize=(2.0 * columns, 2.0 * len(scanners))
    )
    axes = np.atleast_2d(axes)
    for row, scanner in enumerate(scanners):
        images = montage.get(scanner, [])
        for column in range(columns):
            axis = axes[row, column]
            axis.axis("off")
            if column < len(images):
                axis.imshow(images[column])
            if column == 0:
                axis.set_title(scanner.upper(), loc="left", fontsize=10, weight="bold")
    figure.suptitle(
        f"Accepted raw-glass patches · {slide_id} · 129 µm field resampled to 256 px",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(figure, output_dir, "figure1_accepted_glass_montage")


def render_spatial(accepted, scanners, slide_order, output_dir):
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    slide_colors = plt.cm.tab10(np.linspace(0, 1, len(slide_order)))
    for axis, scanner in zip(axes.flat, scanners):
        frame = accepted[accepted["scanner"] == scanner]
        for color, slide_id in zip(slide_colors, slide_order):
            selected = frame[frame["slide_id"] == slide_id]
            axis.scatter(
                selected["x_normalized"],
                1.0 - selected["y_normalized"],
                s=8,
                alpha=0.35,
                color=color,
                label=slide_id,
            )
        axis.set_title(scanner.upper(), color=SCANNER_COLORS[scanner], weight="bold")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(slide_order),
        frameon=False,
    )
    figure.supxlabel("Normalized raw-WSI x", y=0.065)
    figure.supylabel("Normalized raw-WSI y")
    figure.suptitle("Scanner-independent glass coordinates and spatial coverage", fontsize=14)
    figure.tight_layout(rect=(0, 0.11, 1, 0.96))
    return save_figure(figure, output_dir, "figure2_glass_spatial_coverage")


def render_nps(spectra, accepted, scanners, slide_order, output_dir):
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    plane = spectra[spectra["detrend"] == "plane"]
    axis = axes[0]
    for scanner in scanners:
        frame = plane[plane["scanner"] == scanner]
        aggregate = frame.groupby("frequency_cyc_per_um")["nps_od2_um2"].agg(
            median="median", q10=lambda x: x.quantile(0.10), q90=lambda x: x.quantile(0.90)
        ).reset_index()
        x = aggregate["frequency_cyc_per_um"].to_numpy()
        axis.plot(x, aggregate["median"], color=SCANNER_COLORS[scanner], lw=2, label=scanner.upper())
        axis.fill_between(x, aggregate["q10"], aggregate["q90"], color=SCANNER_COLORS[scanner], alpha=0.12)
    axis.set_yscale("log")
    axis.set(xlim=(0.02, 0.98), xlabel="Spatial frequency (cycles/µm)", ylabel="Operational NPS (OD²·µm²)")
    axis.set_title("A  Plane-detrended glass NPS")
    axis.legend(frameon=False, fontsize=8, ncol=2)

    axis = axes[1]
    high = plane[(plane["frequency_cyc_per_um"] >= 0.60) & (plane["frequency_cyc_per_um"] < 0.90)]
    high = high.groupby(["slide_id", "scanner"])["nps_od2_um2"].apply(
        lambda value: float(np.exp(np.mean(np.log(np.maximum(value, 1e-30)))))
    ).reset_index(name="high_band_nps")
    for position, scanner in enumerate(scanners):
        value = high.loc[high["scanner"] == scanner, "high_band_nps"].to_numpy()
        axis.scatter(np.full(len(value), position), value, color=SCANNER_COLORS[scanner], s=32, alpha=0.65)
        if len(value):
            axis.plot([position - 0.18, position + 0.18], [np.median(value)] * 2, color=SCANNER_COLORS[scanner], lw=3)
    axis.set_yscale("log")
    axis.set_xticks(range(len(scanners)), [s.upper() for s in scanners], rotation=30)
    axis.set(ylabel="0.60–0.90 cycles/µm NPS", title="B  Slide-level high-band background")

    axis = axes[2]
    accepted_count = (
        accepted.groupby(["scanner", "slide_id"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=scanners, columns=slide_order, fill_value=0)
        .fillna(0)
        .astype(int)
    )
    image = axis.imshow(accepted_count, aspect="auto", cmap="YlGn", vmin=0)
    for row in range(accepted_count.shape[0]):
        for column in range(accepted_count.shape[1]):
            axis.text(column, row, str(accepted_count.iloc[row, column]), ha="center", va="center", fontsize=8)
    axis.set_xticks(range(len(slide_order)), slide_order, rotation=45, ha="right")
    axis.set_yticks(range(len(scanners)), [s.upper() for s in scanners])
    axis.set_title("C  Accepted glass patches")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Count")
    figure.suptitle("Raw-slide operational background NPS pilot", fontsize=15, y=1.03)
    figure.tight_layout()
    return save_figure(figure, output_dir, "figure3_raw_nps_summary")


def render_nps_2d(nps_2d, scanners, output_dir, mpp):
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    extent = [-1 / (2 * mpp), 1 / (2 * mpp), -1 / (2 * mpp), 1 / (2 * mpp)]
    for axis, scanner in zip(axes.flat, scanners):
        value = np.log10(np.maximum(np.fft.fftshift(nps_2d[scanner]), 1e-30))
        low, high = np.quantile(value, [0.05, 0.995])
        image = axis.imshow(value, cmap="magma", extent=extent, origin="lower", vmin=low, vmax=high)
        axis.set_title(scanner.upper(), color=SCANNER_COLORS[scanner], weight="bold")
        axis.set_xlabel("cycles/µm")
        axis.set_ylabel("cycles/µm")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Plane-detrended 2D operational NPS (independent scale per panel)", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(figure, output_dir, "figure4_raw_nps_2d")


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.raw_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.patches_per_slide < 1:
        raise ValueError("patch count must be positive")

    accepted_rows = []
    candidate_rows = []
    spectra_rows = []
    replicate_rows = []
    rejection_summary = []
    montage = {}
    nps_2d_by_scanner = {scanner: [] for scanner in args.scanners}
    slide_metadata = []
    h5_path = output_dir / "glass_patches_at2_grid.h5"
    with h5py.File(h5_path, "w") as store:
        for slide_index, slide_id in enumerate(args.slides, start=1):
            for scanner in args.scanners:
                path = raw_path(root, scanner, slide_id)
                slide = openslide.OpenSlide(str(path))
                try:
                    mpp, mpp_source = scanner_mpp(root, scanner, path, slide)
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
                        mpp,
                        args.target_mpp,
                        args.target_size,
                        args.patches_per_slide,
                        args.grid_size,
                        args.attempts_per_cell,
                        args.fallback_attempts,
                        rng,
                    )
                    width, height = slide.dimensions
                    slide_metadata.append({
                        "slide_id": slide_id,
                        "scanner": scanner,
                        "raw_path": str(path),
                        "raw_width": width,
                        "raw_height": height,
                        "native_mpp": mpp,
                        "mpp_source": mpp_source,
                        "native_patch_size": native_size,
                        "accepted_patches": len(images),
                        "attempted_patches": len(candidates),
                    })
                finally:
                    slide.close()

                print(
                    f"[exp06] {slide_index}/{len(args.slides)} {slide_id}/{scanner}: "
                    f"accepted {len(images)}/{len(candidates)} candidates, "
                    f"native {native_size}px @ {mpp:.4f} µm/px",
                    flush=True,
                )
                accepted_rows.extend(accepted)
                candidate_rows.extend(candidates)
                rejection_summary.append({
                    "slide_id": slide_id,
                    "scanner": scanner,
                    "accepted": len(images),
                    "attempted": len(candidates),
                    **{f"reject_{key}": value for key, value in rejection_counts.items()},
                })
                group = store.require_group(f"{slide_id}/{scanner}")
                group.create_dataset(
                    "rgb",
                    data=images,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                group.create_dataset(
                    "coords_native",
                    data=np.asarray(
                        [[row["x_native"], row["y_native"]] for row in accepted],
                        dtype=np.int64,
                    ).reshape(-1, 2),
                )
                group.attrs["native_mpp"] = mpp
                group.attrs["target_mpp"] = args.target_mpp
                group.attrs["native_patch_size"] = native_size
                group.attrs["target_patch_size"] = args.target_size

                if slide_id == args.slides[min(2, len(args.slides) - 1)]:
                    montage[scanner] = images[:8]
                if len(images) < 20:
                    print(
                        f"[exp06] WARNING {slide_id}/{scanner}: fewer than 20 valid glass patches; NPS skipped",
                        flush=True,
                    )
                    continue
                estimates = estimate_nps(images, args.target_mpp, args.bins)
                for method, estimate in estimates.items():
                    for frequency, value in zip(
                        estimate["frequency"], estimate["mean_radial"]
                    ):
                        spectra_rows.append({
                            "slide_id": slide_id,
                            "scanner": scanner,
                            "detrend": method,
                            "frequency_cyc_per_um": float(frequency),
                            "nps_od2_um2": float(value),
                            "patches": len(images),
                        })
                    for replicate, curve in enumerate(estimate["replicate_radial"]):
                        for frequency, value in zip(estimate["frequency"], curve):
                            replicate_rows.append({
                                "slide_id": slide_id,
                                "scanner": scanner,
                                "detrend": method,
                                "replicate": replicate,
                                "frequency_cyc_per_um": float(frequency),
                                "nps_od2_um2": float(value),
                            })
                nps_2d_by_scanner[scanner].append(estimates["plane"]["mean_2d"])

    accepted = pd.DataFrame(accepted_rows)
    candidates = pd.DataFrame(candidate_rows)
    spectra = pd.DataFrame(spectra_rows)
    replicates = pd.DataFrame(replicate_rows)
    rejection = pd.DataFrame(rejection_summary).fillna(0)
    metadata = pd.DataFrame(slide_metadata)
    accepted.to_csv(output_dir / "accepted_glass_patches.csv", index=False)
    candidates.to_csv(output_dir / "candidate_glass_qc.csv", index=False)
    spectra.to_csv(output_dir / "raw_glass_nps_spectra.csv", index=False)
    replicates.to_csv(output_dir / "raw_glass_nps_replicates.csv", index=False)
    rejection.to_csv(output_dir / "glass_rejection_summary.csv", index=False)
    metadata.to_csv(output_dir / "raw_wsi_metadata.csv", index=False)

    figures = []
    montage_slide = args.slides[min(2, len(args.slides) - 1)]
    figures.extend(render_montage(montage, args.scanners, output_dir, montage_slide))
    figures.extend(render_spatial(accepted, args.scanners, args.slides, output_dir))
    figures.extend(render_nps(spectra, accepted, args.scanners, args.slides, output_dir))
    complete_2d = {
        scanner: np.mean(values, axis=0)
        for scanner, values in nps_2d_by_scanner.items()
        if values
    }
    if len(complete_2d) == len(args.scanners):
        figures.extend(render_nps_2d(complete_2d, args.scanners, output_dir, args.target_mpp))

    summary = {
        "analysis": "scanner_native_glass_operational_nps_pilot",
        "detector_nps_claim": False,
        "slides": args.slides,
        "scanners": args.scanners,
        "coordinates_paired_across_scanners": False,
        "requested_patches_per_slide_scanner": args.patches_per_slide,
        "physical_field_um": args.target_size * args.target_mpp,
        "target_grid": {
            "size_px": args.target_size,
            "mpp": args.target_mpp,
            "resampling": "OpenCV INTER_AREA from scanner-native L0",
        },
        "qc_thresholds": QC_THRESHOLDS,
        "accepted_by_scanner": accepted.groupby("scanner").size().to_dict(),
        "accepted_by_slide_scanner": (
            accepted.groupby(["slide_id", "scanner"]).size().rename("count").reset_index().to_dict("records")
        ),
        "tables": {
            "accepted": str(output_dir / "accepted_glass_patches.csv"),
            "candidates": str(output_dir / "candidate_glass_qc.csv"),
            "nps": str(output_dir / "raw_glass_nps_spectra.csv"),
            "nps_replicates": str(output_dir / "raw_glass_nps_replicates.csv"),
            "rejections": str(output_dir / "glass_rejection_summary.csv"),
            "raw_metadata": str(output_dir / "raw_wsi_metadata.csv"),
            "patch_store": str(h5_path),
        },
        "figures": figures,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(metadata[["slide_id", "scanner", "accepted_patches", "attempted_patches"]].to_string(index=False), flush=True)
    print(f"[exp06] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
