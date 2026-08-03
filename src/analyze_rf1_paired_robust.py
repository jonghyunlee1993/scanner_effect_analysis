"""Image-only RF1 pilot using paired robust log-spectrum ratios.

This pilot is deliberately isolated from the locked RF1 implementation.  It
uses only paired source/AT2 images in the training fold and evaluates only
image-spectrum and gamut quantities on the held-out fold.  No PFM features or
endpoints are read.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from e5_comparator_population import (
    FOVS,
    FREQUENCY_ANCHOR,
    FREQUENCY_UNITY_BELOW,
    RADIAL_BINS,
    SCANNERS,
    batch_radial_power,
    centered_crop,
    radial_geometry,
    reinhard_lab,
    rgb01_to_od,
    rgb8_to_rgb01,
    smooth_log_curve,
)
from e5_reinhard_residual_frequency import (
    RF1_FOLDS,
    fitted_residual_gains,
    log_spectrum_rmse,
    shared_od_residual_frequency_many,
)
from fetch_e0_pfm_checkpoints import sha256


PILOT_VERSION = "rf1_paired_robust_v1"
METHODS = ("current_pooled", "paired_robust", "paired_reliable")
GAIN_CAP = 1.25
HUBER_C = 1.345
RELIABILITY_Z = 1.96
EPSILON = 1e-20


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--fov", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--statistics", default="outputs/e5_rf1_fold_statistics")
    parser.add_argument("--current-audit", default="outputs/e5_rf1_input_fold_audit")
    parser.add_argument(
        "--output", default="outputs/rf1_improvement_pilot/paired_robust"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def selected_cell(args) -> tuple[int, int]:
    if args.task_index is not None:
        if not 0 <= args.task_index < len(FOVS) * RF1_FOLDS:
            raise ValueError("paired robust task index is out of range")
        return FOVS[args.task_index // RF1_FOLDS], args.task_index % RF1_FOLDS
    if args.fov not in FOVS or args.fold is None or not 0 <= args.fold < RF1_FOLDS:
        raise ValueError("provide a valid --task-index or --fov/--fold pair")
    return int(args.fov), int(args.fold)


def batch_radial_power_per_image(
    rgb01: torch.Tensor,
    geometry,
) -> torch.Tensor:
    """Return Hann-windowed mean-OD radial power for every image separately."""
    if rgb01.ndim != 4 or rgb01.shape[-1] != 3 or rgb01.shape[1] != rgb01.shape[2]:
        raise ValueError("radial power requires NxHxWx3 square RGB")
    size = int(rgb01.shape[1])
    window = torch.hann_window(
        size, periodic=False, dtype=rgb01.dtype, device=rgb01.device
    )
    window = window[:, None] * window[None, :]
    value = rgb01_to_od(rgb01).mean(dim=-1)
    value = value - value.mean(dim=(1, 2), keepdim=True)
    power = torch.fft.fft2(value * window).abs().square().double().reshape(len(value), -1)
    valid = torch.as_tensor(geometry.valid, dtype=torch.bool, device=power.device)
    index = torch.as_tensor(
        geometry.index[geometry.valid], dtype=torch.long, device=power.device
    )
    output = torch.zeros(
        (len(value), len(geometry.counts)), dtype=torch.float64, device=power.device
    )
    output.scatter_add_(1, index[None].expand(len(value), -1), power[:, valid])
    return output


def paired_anchor_log_ratio(
    source_power: np.ndarray,
    target_power: np.ndarray,
    frequency: np.ndarray,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Compute anchor-centered paired amplitude log ratios per patch.

    For patch i and frequency f, the pre-centered value is
    0.5 * (log(P_target[i,f]) - log(P_source[i,f])).  Its mean over the
    frozen 0.03--0.10 cycles/um anchor is removed separately for each patch.
    """
    source = np.asarray(source_power, dtype=np.float64)
    target = np.asarray(target_power, dtype=np.float64)
    freq = np.asarray(frequency, dtype=np.float64)
    if source.shape != target.shape or source.ndim < 2 or source.shape[-1] != len(freq):
        raise ValueError("paired spectrum arrays differ in shape")
    if np.any(source < 0) or np.any(target < 0) or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("paired spectrum arrays must be finite and nonnegative")
    anchor = (freq >= FREQUENCY_ANCHOR[0]) & (freq <= FREQUENCY_ANCHOR[1])
    if not np.any(anchor):
        raise ValueError("frequency grid does not cover the frozen anchor")
    value = 0.5 * (
        np.log(np.maximum(target, float(epsilon)))
        - np.log(np.maximum(source, float(epsilon)))
    )
    return value - value[..., anchor].mean(axis=-1, keepdims=True)


def huber_location(
    values: np.ndarray,
    c: float = HUBER_C,
    max_iter: int = 50,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Coordinate-wise Huber M-location over axis zero using MAD scale."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim < 1 or data.shape[0] < 2 or not np.isfinite(data).all():
        raise ValueError("Huber input must have at least two finite observations")
    location = np.median(data, axis=0)
    mad = np.median(np.abs(data - location), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-12)
    for _ in range(int(max_iter)):
        standardized = np.abs(data - location) / (float(c) * scale)
        weight = np.ones_like(standardized)
        mask = standardized > 1.0
        weight[mask] = 1.0 / standardized[mask]
        updated = np.sum(weight * data, axis=0) / np.maximum(
            np.sum(weight, axis=0), 1e-12
        )
        if float(np.max(np.abs(updated - location))) <= float(tolerance):
            location = updated
            break
        location = updated
    return location


def fit_paired_robust_gains(
    patch_log_ratio: np.ndarray,
    frequency: np.ndarray,
    cap: float = GAIN_CAP,
    reliability_z: float = RELIABILITY_Z,
) -> dict[str, np.ndarray]:
    """Fit slide-balanced paired robust and reliability-shrunk gains.

    ``patch_log_ratio`` has shape slide x patch x scanner x frequency.  Patch
    effects are reduced to one median curve per slide, then a Huber location is
    fitted across slides.  Reliability is the fraction of the absolute effect
    remaining after subtracting a z*robust-SE uncertainty margin:

        r(f) = max(0, |theta(f)| - z*SE(f)) / (|theta(f)| + eps).
    """
    value = np.asarray(patch_log_ratio, dtype=np.float64)
    freq = np.asarray(frequency, dtype=np.float64)
    if (
        value.ndim != 4
        or value.shape[0] < 2
        or value.shape[1] < 1
        or value.shape[2] != len(SCANNERS) - 1
        or value.shape[3] != len(freq)
        or not np.isfinite(value).all()
    ):
        raise ValueError("expected finite slide x patch x 5 x frequency ratios")
    slide_location = np.median(value, axis=1)
    theta = huber_location(slide_location)
    mad = np.median(np.abs(slide_location - theta[None]), axis=0)
    robust_se = 1.4826 * mad / math.sqrt(value.shape[0])
    reliability = np.maximum(
        0.0,
        np.abs(theta) - float(reliability_z) * robust_se,
    ) / np.maximum(np.abs(theta), 1e-12)
    reliability = np.clip(reliability, 0.0, 1.0)

    def finalize(log_curve: np.ndarray) -> np.ndarray:
        result = np.empty_like(log_curve)
        bound = math.log(float(cap))
        for scanner_index in range(log_curve.shape[0]):
            smoothed = smooth_log_curve(log_curve[scanner_index])
            smoothed[freq < FREQUENCY_UNITY_BELOW] = 0.0
            result[scanner_index] = np.exp(np.clip(smoothed, -bound, bound))
        return result

    return {
        "slide_location": slide_location,
        "theta": theta,
        "robust_se": robust_se,
        "reliability": reliability,
        "paired_robust_gain": finalize(theta),
        "paired_reliable_gain": finalize(theta * reliability),
    }


def load_fold_statistics(root: Path, fov: int, fold: int) -> tuple[Path, dict[str, np.ndarray]]:
    path = root / f"fov_{fov}_fold_{fold}.npz"
    summary_path = path.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    if not (
        summary.get("analysis") == "e5_rf1_fold_statistics"
        and summary.get("fov") == fov
        and summary.get("fold") == fold
        and summary.get("statistics_gate_pass") is True
        and summary.get("output_sha256") == sha256(path)
    ):
        raise RuntimeError(f"invalid locked fold statistics: {path}")
    with np.load(path) as source:
        values = {name: source[name] for name in source.files}
    return path, values


def current_reference_error(
    current_audit_root: Path,
    fov: int,
    fold: int,
    observed_power: np.ndarray,
) -> float:
    path = current_audit_root / f"fov_{fov}_fold_{fold}.npz"
    if not path.exists():
        return float("nan")
    with np.load(path) as source:
        caps = source["caps"]
        matches = np.flatnonzero(np.isclose(caps, GAIN_CAP, atol=0.0, rtol=1e-12))
        if len(matches) != 1:
            raise RuntimeError(f"current audit does not contain cap {GAIN_CAP}: {path}")
        expected = source["validation_output_power"][int(matches[0])]
    denominator = np.maximum(np.abs(expected), EPSILON)
    return float(np.max(np.abs(observed_power - expected) / denominator))


def run_fold(args):
    if not torch.cuda.is_available():
        raise RuntimeError("paired robust fold analysis requires a visible CUDA device")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    fov, fold = selected_cell(args)
    statistics_path, statistics = load_fold_statistics(Path(args.statistics), fov, fold)
    train_ids = [str(value) for value in statistics["train_slide_ids"]]
    heldout_ids = [str(value) for value in statistics["heldout_slide_ids"]]
    frequency = statistics["radial_frequency"]
    geometry = radial_geometry(fov)
    if not np.allclose(frequency, geometry.frequency, atol=0.0, rtol=1e-12):
        raise RuntimeError("fold-statistics radial grid differs from renderer grid")

    output_root = Path(args.output)
    fold_root = output_root / "folds"
    fold_root.mkdir(parents=True, exist_ok=True)
    output_path = fold_root / f"fov_{fov}_fold_{fold}.npz"
    summary_path = output_path.with_suffix(".summary.json")

    lab_mean = torch.as_tensor(statistics["lab_mean"], dtype=torch.float32, device="cuda")
    lab_std = torch.as_tensor(statistics["lab_std"], dtype=torch.float32, device="cuda")
    patch_ratio = np.empty(
        (len(train_ids), 100, len(SCANNERS) - 1, RADIAL_BINS), dtype=np.float32
    )
    grid_root = Path(args.grid)
    with torch.inference_mode():
        for slide_index, slide_id in enumerate(train_ids):
            path = grid_root / f"{slide_id}.h5"
            with h5py.File(path, "r") as source:
                for start in range(0, 100, args.batch_size):
                    stop = min(start + args.batch_size, 100)
                    target = rgb8_to_rgb01(
                        centered_crop(source["rgb"][0, start:stop], fov), device="cuda"
                    )
                    target_power = batch_radial_power_per_image(target, geometry).cpu().numpy()
                    for scanner_index in range(1, len(SCANNERS)):
                        rgb = rgb8_to_rgb01(
                            centered_crop(source["rgb"][scanner_index, start:stop], fov),
                            device="cuda",
                        )
                        base = reinhard_lab(
                            rgb,
                            lab_mean[scanner_index].reshape(1, 1, 1, 3),
                            lab_std[scanner_index].reshape(1, 1, 1, 3),
                            lab_mean[0].reshape(1, 1, 1, 3),
                            lab_std[0].reshape(1, 1, 1, 3),
                        )["output"]
                        source_power = batch_radial_power_per_image(base, geometry).cpu().numpy()
                        patch_ratio[
                            slide_index, start:stop, scanner_index - 1
                        ] = paired_anchor_log_ratio(
                            source_power, target_power, frequency
                        ).astype(np.float32)
            print(
                f"[train {slide_index + 1}/{len(train_ids)}] FOV {fov} fold {fold} {slide_id}",
                flush=True,
            )

    fit = fit_paired_robust_gains(patch_ratio, frequency)
    current_gain = fitted_residual_gains(
        statistics["post_reinhard_source_power"],
        statistics["raw_at2_target_power"],
        frequency,
        GAIN_CAP,
    )
    gains = np.stack(
        [current_gain, fit["paired_robust_gain"], fit["paired_reliable_gain"]], axis=0
    )

    n_methods = len(METHODS)
    n_slides = len(heldout_ids)
    metric_names = (
        "preproject_range_fraction",
        "material_range_fraction",
        "projection_fraction",
        "projection_rgb_mae",
        "final_clamp_mae",
    )
    metric_shape = (n_methods, n_slides, len(SCANNERS) - 1, 100)
    metrics = {name: np.empty(metric_shape, dtype=np.float32) for name in metric_names}
    target_power = np.zeros(RADIAL_BINS, dtype=np.float64)
    base_power = np.zeros((len(SCANNERS) - 1, RADIAL_BINS), dtype=np.float64)
    output_power = np.zeros(
        (n_methods, len(SCANNERS) - 1, RADIAL_BINS), dtype=np.float64
    )
    with torch.inference_mode():
        for slide_index, slide_id in enumerate(heldout_ids):
            path = grid_root / f"{slide_id}.h5"
            with h5py.File(path, "r") as source:
                for start in range(0, 100, args.batch_size):
                    stop = min(start + args.batch_size, 100)
                    target = rgb8_to_rgb01(
                        centered_crop(source["rgb"][0, start:stop], fov), device="cuda"
                    )
                    target_power += batch_radial_power(target, geometry).cpu().numpy()
                    for scanner_index in range(1, len(SCANNERS)):
                        rgb = rgb8_to_rgb01(
                            centered_crop(source["rgb"][scanner_index, start:stop], fov),
                            device="cuda",
                        )
                        base = reinhard_lab(
                            rgb,
                            lab_mean[scanner_index].reshape(1, 1, 1, 3),
                            lab_std[scanner_index].reshape(1, 1, 1, 3),
                            lab_mean[0].reshape(1, 1, 1, 3),
                            lab_std[0].reshape(1, 1, 1, 3),
                        )["output"]
                        base_power[scanner_index - 1] += batch_radial_power(
                            base, geometry
                        ).cpu().numpy()
                        report = shared_od_residual_frequency_many(
                            base, gains[:, scanner_index - 1], frequency
                        )
                        for name in metric_names:
                            metrics[name][
                                :, slide_index, scanner_index - 1, start:stop
                            ] = report[name].cpu().numpy()
                        for method_index in range(n_methods):
                            output_power[method_index, scanner_index - 1] += (
                                batch_radial_power(
                                    report["output"][method_index], geometry
                                )
                                .cpu()
                                .numpy()
                            )
            print(
                f"[valid {slide_index + 1}/{n_slides}] FOV {fov} fold {fold} {slide_id}",
                flush=True,
            )

    reference_error = current_reference_error(
        Path(args.current_audit), fov, fold, output_power[0]
    )
    if not (
        np.isfinite(patch_ratio).all()
        and np.isfinite(gains).all()
        and np.isfinite(target_power).all()
        and np.isfinite(base_power).all()
        and np.isfinite(output_power).all()
        and all(np.isfinite(value).all() for value in metrics.values())
        and (not np.isfinite(reference_error) or reference_error <= 5e-5)
    ):
        raise RuntimeError("paired robust fold output failed finite/reference checks")

    temporary = fold_root / f".fov_{fov}_fold_{fold}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        temporary,
        fov=np.asarray(fov, dtype=np.int64),
        fold=np.asarray(fold, dtype=np.int64),
        train_slide_ids=np.asarray(train_ids),
        heldout_slide_ids=np.asarray(heldout_ids),
        methods=np.asarray(METHODS),
        scanners=np.asarray(SCANNERS[1:]),
        radial_frequency=frequency,
        gains=gains,
        slide_location=fit["slide_location"],
        robust_theta=fit["theta"],
        robust_se=fit["robust_se"],
        reliability=fit["reliability"],
        validation_target_power=target_power,
        validation_base_power=base_power,
        validation_output_power=output_power,
        current_reference_power_max_relative_error=np.asarray(reference_error),
        **metrics,
    )
    os.replace(temporary, output_path)
    summary = {
        "analysis": "rf1_paired_robust_fold",
        "pilot_version": PILOT_VERSION,
        "outcome_access": False,
        "pfm_access": False,
        "fov": fov,
        "fold": fold,
        "gain_cap": GAIN_CAP,
        "huber_c": HUBER_C,
        "reliability_z": RELIABILITY_Z,
        "train_slides": len(train_ids),
        "train_patches_per_scanner": len(train_ids) * 100,
        "heldout_slides": len(heldout_ids),
        "heldout_patches_per_scanner": len(heldout_ids) * 100,
        "statistics": str(statistics_path.resolve()),
        "statistics_sha256": sha256(statistics_path),
        "current_reference_power_max_relative_error": reference_error,
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "fold_gate_pass": True,
        "device": torch.cuda.get_device_name(0),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def aggregate(args):
    output_root = Path(args.output)
    rows = []
    diagnostic_rows = []
    fold_hashes = {}
    for fov in FOVS:
        folds = []
        for fold in range(RF1_FOLDS):
            path = output_root / "folds" / f"fov_{fov}_fold_{fold}.npz"
            summary = json.loads(path.with_suffix(".summary.json").read_text())
            if not (
                summary.get("analysis") == "rf1_paired_robust_fold"
                and summary.get("pilot_version") == PILOT_VERSION
                and summary.get("outcome_access") is False
                and summary.get("pfm_access") is False
                and summary.get("fov") == fov
                and summary.get("fold") == fold
                and summary.get("output_sha256") == sha256(path)
                and summary.get("fold_gate_pass") is True
            ):
                raise RuntimeError(f"invalid paired robust fold: {path}")
            with np.load(path) as source:
                folds.append({name: source[name] for name in source.files})
            fold_hashes[f"fov_{fov}_fold_{fold}"] = sha256(path)

        frequency = folds[0]["radial_frequency"]
        target = sum(value["validation_target_power"] for value in folds)
        base = sum(value["validation_base_power"] for value in folds)
        corrected = sum(value["validation_output_power"] for value in folds)
        for scanner_index, scanner in enumerate(SCANNERS[1:]):
            base_rmse = log_spectrum_rmse(base[scanner_index], target, frequency)
            for method_index, method in enumerate(METHODS):
                output_rmse = log_spectrum_rmse(
                    corrected[method_index, scanner_index], target, frequency
                )
                material = np.concatenate(
                    [
                        value["material_range_fraction"][
                            method_index, :, scanner_index
                        ].reshape(-1)
                        for value in folds
                    ]
                )
                projection_mae = np.concatenate(
                    [
                        value["projection_rgb_mae"][method_index, :, scanner_index].reshape(-1)
                        for value in folds
                    ]
                )
                final_mae = np.concatenate(
                    [
                        value["final_clamp_mae"][method_index, :, scanner_index].reshape(-1)
                        for value in folds
                    ]
                )
                rows.append(
                    {
                        "fov": fov,
                        "scanner": scanner,
                        "method": method,
                        "heldout_patches": len(material),
                        "base_log_spectrum_rmse": base_rmse,
                        "output_log_spectrum_rmse": output_rmse,
                        "rmse_change": output_rmse - base_rmse,
                        "rmse_relative_change": (output_rmse - base_rmse) / base_rmse,
                        "spectrum_improved": output_rmse < base_rmse,
                        "material_range_fraction_mean": float(material.mean()),
                        "material_range_fraction_q99": float(np.quantile(material, 0.99)),
                        "projection_rgb_mae_mean": float(projection_mae.mean()),
                        "final_clamp_mae_max": float(final_mae.max()),
                    }
                )
        for fold, value in enumerate(folds):
            for scanner_index, scanner in enumerate(SCANNERS[1:]):
                selected = frequency >= FREQUENCY_UNITY_BELOW
                diagnostic_rows.append(
                    {
                        "fov": fov,
                        "fold": fold,
                        "scanner": scanner,
                        "high_frequency_bins": int(selected.sum()),
                        "reliability_mean": float(
                            value["reliability"][scanner_index, selected].mean()
                        ),
                        "reliability_median": float(
                            np.median(value["reliability"][scanner_index, selected])
                        ),
                        "paired_robust_gain_abs_log_mean": float(
                            np.abs(
                                np.log(value["gains"][1, scanner_index, selected])
                            ).mean()
                        ),
                        "paired_reliable_gain_abs_log_mean": float(
                            np.abs(
                                np.log(value["gains"][2, scanner_index, selected])
                            ).mean()
                        ),
                    }
                )

    frame = pd.DataFrame(rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    frame.to_csv(output_root / "heldout_spectrum_gamut.csv", index=False)
    diagnostics.to_csv(output_root / "gain_diagnostics.csv", index=False)
    comparison = frame.pivot_table(
        index=["fov", "scanner"], columns="method", values="output_log_spectrum_rmse"
    ).reset_index()
    comparison["paired_robust_minus_current"] = (
        comparison["paired_robust"] - comparison["current_pooled"]
    )
    comparison["paired_reliable_minus_current"] = (
        comparison["paired_reliable"] - comparison["current_pooled"]
    )
    comparison.to_csv(output_root / "method_comparison.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "current_pooled": "#777777",
        "paired_robust": "#2673b8",
        "paired_reliable": "#e07a2d",
    }
    labels = {
        "current_pooled": "Current pooled",
        "paired_robust": "Paired robust",
        "paired_reliable": "Paired + reliability",
    }
    figure, axes = plt.subplots(1, len(FOVS), figsize=(15.5, 4.8), sharey=True)
    x = np.arange(len(SCANNERS) - 1)
    width = 0.24
    for axis, fov in zip(axes, FOVS):
        selected_frame = frame.loc[frame["fov"] == fov]
        for method_index, method in enumerate(METHODS):
            ordered = selected_frame.loc[
                selected_frame["method"] == method
            ].set_index("scanner").loc[list(SCANNERS[1:])]
            axis.bar(
                x + (method_index - 1) * width,
                100.0 * ordered["rmse_relative_change"].to_numpy(),
                width,
                color=colors[method],
                label=labels[method],
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"FOV {fov}")
        axis.set_xticks(x, [value.upper() for value in SCANNERS[1:]], rotation=35)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Held-out log-spectrum RMSE change vs Reinhard (%)")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle("RF1 paired robust pilot (negative is better; image-only)")
    figure.tight_layout()
    figure_path = output_root / "heldout_spectrum_comparison.png"
    figure_pdf = output_root / "heldout_spectrum_comparison.pdf"
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    figure.savefig(figure_pdf, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "analysis": "rf1_paired_robust_pilot",
        "pilot_version": PILOT_VERSION,
        "outcome_access": False,
        "pfm_access": False,
        "fovs": list(FOVS),
        "folds": RF1_FOLDS,
        "slides": 109,
        "patches_per_scanner_fov": 10_900,
        "methods": list(METHODS),
        "gain_cap": GAIN_CAP,
        "paired_formula": "ell_i(f)=0.5*[log(P_AT2,i(f))-log(P_source,i(f))], anchor-centered per patch",
        "aggregation": "median over 100 patches within slide; coordinate-wise Huber(c=1.345) over training slides",
        "reliability": "max(0,abs(theta)-1.96*robust_SE)/(abs(theta)+1e-12)",
        "cells": len(frame),
        "spectrum_improved_cells": {
            method: int(
                frame.loc[frame["method"] == method, "spectrum_improved"].sum()
            )
            for method in METHODS
        },
        "better_than_current_cells": {
            "paired_robust": int((comparison["paired_robust_minus_current"] < 0).sum()),
            "paired_reliable": int(
                (comparison["paired_reliable_minus_current"] < 0).sum()
            ),
        },
        "mean_rmse_relative_change": {
            method: float(
                frame.loc[frame["method"] == method, "rmse_relative_change"].mean()
            )
            for method in METHODS
        },
        "s360_mean_rmse_relative_change": {
            method: float(
                frame.loc[
                    (frame["method"] == method) & (frame["scanner"] == "s360"),
                    "rmse_relative_change",
                ].mean()
            )
            for method in METHODS
        },
        "maximum_final_clamp_mae": float(frame["final_clamp_mae_max"].max()),
        "maximum_material_range_fraction_mean": float(
            frame["material_range_fraction_mean"].max()
        ),
        "fold_sha256": fold_hashes,
        "heldout_spectrum_gamut": str(
            (output_root / "heldout_spectrum_gamut.csv").resolve()
        ),
        "heldout_spectrum_gamut_sha256": sha256(
            output_root / "heldout_spectrum_gamut.csv"
        ),
        "gain_diagnostics_sha256": sha256(output_root / "gain_diagnostics.csv"),
        "method_comparison_sha256": sha256(output_root / "method_comparison.csv"),
        "heldout_spectrum_figure": str(figure_path.resolve()),
        "heldout_spectrum_figure_sha256": sha256(figure_path),
        "heldout_spectrum_figure_pdf_sha256": sha256(figure_pdf),
        "aggregate_gate_pass": bool(
            len(frame) == len(FOVS) * (len(SCANNERS) - 1) * len(METHODS)
            and frame["heldout_patches"].eq(10_900).all()
            and np.isfinite(frame.select_dtypes(include=[np.number])).all().all()
            and float(frame["final_clamp_mae_max"].max()) <= 1e-6
        ),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["aggregate_gate_pass"]:
        raise SystemExit(2)


def main():
    args = parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        run_fold(args)


if __name__ == "__main__":
    main()
