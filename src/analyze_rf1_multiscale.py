"""Image-only pilot of a Laplacian-band alternative to radial E5-RF1.

The pilot intentionally does not read PFM features or outcomes.  It reuses the
cross-fitted Reinhard statistics and post-Reinhard radial sufficient statistics
from E5-RF1, collapses the latter into three broad Gaussian/Laplacian bands, and
evaluates the image-domain correction on the held-out folds at FOV 256.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from e5_comparator_population import (
    SCANNERS,
    TARGET_MPP,
    batch_radial_power,
    centered_crop,
    radial_geometry,
    reinhard_lab,
    rgb01_to_od,
    rgb8_to_rgb01,
)
from e5_reinhard_residual_frequency import (
    MATERIAL_EXCURSION,
    OD_RGB8_MAX,
    RF1_FOLDS,
    fitted_residual_gains,
    log_spectrum_rmse,
    shared_od_residual_frequency,
)
from e5_comparator_population import od_to_rgb01
from fetch_e0_pfm_checkpoints import sha256


PILOT_VERSION = "rf1_multiscale_laplacian_pilot_v1"
PILOT_FOV = 256
PILOT_FOVS = (224, 256, 512)
PILOT_GAIN_CAP = 1.25
PYRAMID_SIGMAS = (1.0, 2.0, 4.0)
CONDITIONS = ("reinhard", "radial_rf1", "multiscale_rf1")


def gaussian_transfer(frequency: np.ndarray, sigma_pixels: float) -> np.ndarray:
    """Continuous Gaussian amplitude response sampled at cycles/micrometre."""
    value = np.asarray(frequency, dtype=np.float64)
    if value.ndim != 1 or np.any(value < 0) or sigma_pixels <= 0:
        raise ValueError("invalid Gaussian transfer inputs")
    cycles_per_pixel = value * TARGET_MPP
    return np.exp(-2.0 * np.pi**2 * float(sigma_pixels) ** 2 * cycles_per_pixel**2)


def laplacian_transfer(
    frequency: np.ndarray,
    sigmas: tuple[float, ...] = PYRAMID_SIGMAS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return broad Laplacian-band and residual-lowpass transfer functions."""
    if not sigmas or any(a <= 0 for a in sigmas) or tuple(sorted(sigmas)) != sigmas:
        raise ValueError("pyramid sigmas must be positive and increasing")
    previous = np.ones_like(np.asarray(frequency, dtype=np.float64))
    bands = []
    for sigma in sigmas:
        current = gaussian_transfer(frequency, sigma)
        bands.append(previous - current)
        previous = current
    return np.stack(bands, axis=0), previous


def radial_power_to_band_energy(
    radial_power: np.ndarray,
    frequency: np.ndarray,
    sigmas: tuple[float, ...] = PYRAMID_SIGMAS,
) -> np.ndarray:
    """Approximate Laplacian-band energy from locked radial sufficient statistics."""
    power = np.asarray(radial_power, dtype=np.float64)
    freq = np.asarray(frequency, dtype=np.float64)
    if power.shape[-1] != len(freq) or np.any(power < 0) or not np.isfinite(power).all():
        raise ValueError("invalid radial power")
    transfer, _ = laplacian_transfer(freq, sigmas)
    return np.einsum("...f,bf->...b", power, np.square(transfer))


def fitted_multiscale_gains(
    source_power: np.ndarray,
    target_power: np.ndarray,
    frequency: np.ndarray,
    cap: float = PILOT_GAIN_CAP,
    sigmas: tuple[float, ...] = PYRAMID_SIGMAS,
) -> np.ndarray:
    """Fit one pooled energy-ratio gain per scanner and Laplacian band.

    This deliberately retains the pooled E5-RF1 estimator so that the pilot
    isolates only the representation change from 72 radial bins to three local
    spatial bands.  A paired-robust estimator is a separate RF1 improvement.
    """
    source = np.asarray(source_power, dtype=np.float64)
    target = np.asarray(target_power, dtype=np.float64)
    if source.ndim != 2 or target.shape != source.shape[1:]:
        raise ValueError("source/target power shapes differ")
    if not np.isfinite(cap) or cap < 1.0:
        raise ValueError("gain cap must be at least one")
    source_energy = radial_power_to_band_energy(source, frequency, sigmas)
    target_energy = radial_power_to_band_energy(target, frequency, sigmas)
    raw = np.sqrt(
        np.maximum(target_energy, 1e-20)[None, :] / np.maximum(source_energy, 1e-20)
    )
    return np.clip(raw, 1.0 / float(cap), float(cap))


def _gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(4.0 * float(sigma))))
    coordinate = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coordinate / float(sigma)).square())
    return kernel / kernel.sum()


def _gaussian_blur_increment(value: torch.Tensor, sigma: float) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError("Gaussian pyramid input must be BxHxW")
    kernel = _gaussian_kernel1d(sigma, value.device, value.dtype)
    radius = int((len(kernel) - 1) // 2)
    work = value[:, None]
    work = F.pad(work, (radius, radius, 0, 0), mode="reflect")
    work = F.conv2d(work, kernel.reshape(1, 1, 1, -1))
    work = F.pad(work, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(work, kernel.reshape(1, 1, -1, 1))[:, 0]


def laplacian_pyramid(
    value: torch.Tensor,
    sigmas: tuple[float, ...] = PYRAMID_SIGMAS,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Construct additive Gaussian/Laplacian bands with an exact residual base."""
    if value.ndim != 3 or value.shape[-1] != value.shape[-2]:
        raise ValueError("Laplacian pyramid requires BxHxW square input")
    if not sigmas or tuple(sorted(sigmas)) != sigmas:
        raise ValueError("pyramid sigmas must be increasing")
    previous = value
    previous_sigma = 0.0
    bands = []
    for sigma in sigmas:
        incremental = math.sqrt(float(sigma) ** 2 - previous_sigma**2)
        current = _gaussian_blur_increment(previous, incremental)
        bands.append(previous - current)
        previous = current
        previous_sigma = float(sigma)
    return tuple(bands), previous


def shared_od_multiscale(
    rgb01: torch.Tensor,
    gains: np.ndarray | torch.Tensor,
    sigmas: tuple[float, ...] = PYRAMID_SIGMAS,
) -> dict[str, torch.Tensor]:
    """Apply local Laplacian-band gains through a shared, gamut-safe OD residual."""
    if rgb01.ndim != 4 or rgb01.shape[-1] != 3 or rgb01.shape[1] != rgb01.shape[2]:
        raise ValueError("multiscale RF1 requires NxHxWx3 square RGB")
    if not torch.isfinite(rgb01).all():
        raise ValueError("multiscale RF1 input contains non-finite values")
    gain = torch.as_tensor(gains, dtype=rgb01.dtype, device=rgb01.device)
    if gain.shape != (len(sigmas),) or not torch.isfinite(gain).all() or torch.any(gain <= 0):
        raise ValueError("one positive finite gain is required per band")

    od = rgb01_to_od(rgb01)
    mean_od = od.mean(dim=-1)
    bands, base = laplacian_pyramid(mean_od, sigmas)
    corrected = base
    for band_index, band in enumerate(bands):
        corrected = corrected + gain[band_index] * band
    proposed_delta = corrected - mean_od

    lower = -od.amin(dim=-1)
    upper = OD_RGB8_MAX - od.amax(dim=-1)
    projected_delta = torch.minimum(torch.maximum(proposed_delta, lower), upper)
    proposed_od = od + proposed_delta[..., None]
    projected_od = od + projected_delta[..., None]
    proposed_rgb = od_to_rgb01(proposed_od)
    projected_rgb = od_to_rgb01(projected_od)
    output = projected_rgb.clamp(0.0, 1.0)

    out_of_range = (proposed_rgb < 0.0) | (proposed_rgb > 1.0)
    material = (proposed_rgb < -MATERIAL_EXCURSION) | (
        proposed_rgb > 1.0 + MATERIAL_EXCURSION
    )
    projection = (projected_delta - proposed_delta).abs()
    metric_dims = (1, 2, 3)
    pixel_dims = (1, 2)
    return {
        "output": output,
        "proposed_delta": proposed_delta,
        "projected_delta": projected_delta,
        "preproject_range_fraction": out_of_range.float().mean(dim=metric_dims),
        "material_range_fraction": material.float().mean(dim=metric_dims),
        "projection_fraction": (projection > 1e-7).float().mean(dim=pixel_dims),
        "projection_rgb_mae": (projected_rgb - proposed_rgb).abs().mean(dim=metric_dims),
        "final_clamp_mae": (output - projected_rgb).abs().mean(dim=metric_dims),
    }


def paired_patch_metrics(output: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    if output.shape != target.shape or output.ndim != 4 or output.shape[-1] != 3:
        raise ValueError("paired metrics require equal BxHxWx3 tensors")
    output_od = rgb01_to_od(output).mean(dim=-1)
    target_od = rgb01_to_od(target).mean(dim=-1)
    rgb_mae = (output - target).abs().mean(dim=(1, 2, 3))
    od_mae = (output_od - target_od).abs().mean(dim=(1, 2))
    dx = (output_od[:, :, 1:] - output_od[:, :, :-1]) - (
        target_od[:, :, 1:] - target_od[:, :, :-1]
    )
    dy = (output_od[:, 1:, :] - output_od[:, :-1, :]) - (
        target_od[:, 1:, :] - target_od[:, :-1, :]
    )
    gradient_mae = 0.5 * (dx.abs().mean(dim=(1, 2)) + dy.abs().mean(dim=(1, 2)))
    return {"rgb_mae": rgb_mae, "mean_od_mae": od_mae, "gradient_mae": gradient_mae}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("fold", "aggregate", "panel-aggregate", "visual"), required=True
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--fov", type=int, default=PILOT_FOV)
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--statistics", default="outputs/e5_rf1_fold_statistics")
    parser.add_argument(
        "--output", default="outputs/rf1_improvement_pilot/multiscale"
    )
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument(
        "--sample-manifest", default="outputs/e5_rf1_visual_audit/sample_manifest.csv"
    )
    return parser.parse_args()


def _load_statistics(
    root: Path, fold: int, fov: int = PILOT_FOV
) -> tuple[Path, dict[str, np.ndarray]]:
    path = root / f"fov_{fov}_fold_{fold}.npz"
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    if not (
        summary.get("analysis") == "e5_rf1_fold_statistics"
        and summary.get("fov") == fov
        and summary.get("fold") == fold
        and summary.get("output_sha256") == sha256(path)
        and summary.get("statistics_gate_pass") is True
    ):
        raise RuntimeError(f"invalid cross-fitted RF1 statistics: {path}")
    with np.load(path) as source:
        return path, {name: source[name] for name in source.files}


def run_fold(args):
    if args.fold is None or not 0 <= args.fold < RF1_FOLDS:
        raise ValueError("fold mode requires --fold in [0,4]")
    if not torch.cuda.is_available():
        raise RuntimeError("multiscale fold evaluation requires CUDA")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    if args.fov not in PILOT_FOVS:
        raise ValueError(f"unsupported pilot FOV: {args.fov}")
    fov = int(args.fov)
    statistics_path, statistics = _load_statistics(Path(args.statistics), args.fold, fov)
    heldout_ids = [str(value) for value in statistics["heldout_slide_ids"]]
    multiscale_gains = fitted_multiscale_gains(
        statistics["post_reinhard_source_power"],
        statistics["raw_at2_target_power"],
        statistics["radial_frequency"],
    )
    radial_gains = fitted_residual_gains(
        statistics["post_reinhard_source_power"],
        statistics["raw_at2_target_power"],
        statistics["radial_frequency"],
        PILOT_GAIN_CAP,
    )

    n_slides = len(heldout_ids)
    n_scanners = len(SCANNERS) - 1
    patch_shape = (len(CONDITIONS), n_slides, n_scanners, 100)
    paired = {
        name: np.empty(patch_shape, dtype=np.float32)
        for name in ("rgb_mae", "mean_od_mae", "gradient_mae")
    }
    gamut = {
        name: np.empty((n_slides, n_scanners, 100), dtype=np.float32)
        for name in (
            "preproject_range_fraction",
            "material_range_fraction",
            "projection_fraction",
            "projection_rgb_mae",
            "final_clamp_mae",
        )
    }
    base_clip_fraction = np.empty((n_slides, n_scanners, 100), dtype=np.float32)
    base_clip_mae = np.empty_like(base_clip_fraction)
    radial_power = np.zeros((len(CONDITIONS), n_scanners, 72), dtype=np.float64)
    target_power = np.zeros(72, dtype=np.float64)
    geometry = radial_geometry(fov)

    lab_mean = torch.as_tensor(statistics["lab_mean"], dtype=torch.float32, device="cuda")
    lab_std = torch.as_tensor(statistics["lab_std"], dtype=torch.float32, device="cuda")
    grid_root = Path(args.grid)
    with torch.inference_mode():
        for slide_index, slide_id in enumerate(heldout_ids):
            path = grid_root / f"{slide_id}.h5"
            with h5py.File(path, "r") as source:
                if str(source.attrs["slide_id"]) != slide_id:
                    raise RuntimeError(f"slide identity mismatch: {path}")
                for start in range(0, 100, args.batch_size):
                    stop = min(start + args.batch_size, 100)
                    target = rgb8_to_rgb01(
                        centered_crop(source["rgb"][0, start:stop], fov),
                        device="cuda",
                    )
                    target_power += batch_radial_power(target, geometry).cpu().numpy()
                    for scanner_index in range(1, len(SCANNERS)):
                        source_rgb = rgb8_to_rgb01(
                            centered_crop(
                                source["rgb"][scanner_index, start:stop], fov
                            ),
                            device="cuda",
                        )
                        base = reinhard_lab(
                            source_rgb,
                            lab_mean[scanner_index].reshape(1, 1, 1, 3),
                            lab_std[scanner_index].reshape(1, 1, 1, 3),
                            lab_mean[0].reshape(1, 1, 1, 3),
                            lab_std[0].reshape(1, 1, 1, 3),
                        )
                        base_clip_fraction[slide_index, scanner_index - 1, start:stop] = (
                            base["preclip_range_fraction"].cpu().numpy()
                        )
                        base_clip_mae[slide_index, scanner_index - 1, start:stop] = (
                            base["clip_pixel_mae"].cpu().numpy()
                        )
                        radial = shared_od_residual_frequency(
                            base["output"],
                            radial_gains[scanner_index - 1],
                            statistics["radial_frequency"],
                        )["output"]
                        multiscale = shared_od_multiscale(
                            base["output"], multiscale_gains[scanner_index - 1]
                        )
                        outputs = (base["output"], radial, multiscale["output"])
                        for condition_index, output in enumerate(outputs):
                            measured = paired_patch_metrics(output, target)
                            for name, value in measured.items():
                                paired[name][
                                    condition_index,
                                    slide_index,
                                    scanner_index - 1,
                                    start:stop,
                                ] = value.cpu().numpy()
                            radial_power[condition_index, scanner_index - 1] += (
                                batch_radial_power(output, geometry).cpu().numpy()
                            )
                        for name in gamut:
                            gamut[name][slide_index, scanner_index - 1, start:stop] = (
                                multiscale[name].cpu().numpy()
                            )
            print(
                f"[{slide_index + 1}/{n_slides}] multiscale FOV {fov} "
                f"fold {args.fold} {slide_id}",
                flush=True,
            )

    arrays = [*paired.values(), *gamut.values(), radial_power, target_power]
    if not all(np.isfinite(value).all() for value in arrays):
        raise RuntimeError("non-finite multiscale pilot result")
    output_root = Path(args.output)
    fold_root = output_root / "folds"
    fold_root.mkdir(parents=True, exist_ok=True)
    output_path = fold_root / f"fov_{fov}_fold_{args.fold}.npz"
    temporary = fold_root / f".{output_path.name}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        temporary,
        version=np.asarray(PILOT_VERSION),
        fov=np.asarray(fov),
        fold=np.asarray(args.fold),
        heldout_slide_ids=np.asarray(heldout_ids),
        scanners=np.asarray(SCANNERS[1:]),
        conditions=np.asarray(CONDITIONS),
        sigmas=np.asarray(PYRAMID_SIGMAS),
        gain_cap=np.asarray(PILOT_GAIN_CAP),
        multiscale_gains=multiscale_gains,
        radial_gains=radial_gains,
        radial_frequency=statistics["radial_frequency"],
        validation_target_power=target_power,
        validation_condition_power=radial_power,
        base_clip_fraction=base_clip_fraction,
        base_clip_mae=base_clip_mae,
        **paired,
        **gamut,
    )
    os.replace(temporary, output_path)
    summary = {
        "analysis": "rf1_multiscale_fold_pilot",
        "version": PILOT_VERSION,
        "outcome_access": False,
        "pfm_access": False,
        "fov": fov,
        "fold": args.fold,
        "heldout_slides": n_slides,
        "patches_per_source_scanner": n_slides * 100,
        "sigmas_pixels": list(PYRAMID_SIGMAS),
        "gain_cap": PILOT_GAIN_CAP,
        "statistics": str(statistics_path.resolve()),
        "statistics_sha256": sha256(statistics_path),
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "device": torch.cuda.get_device_name(0),
        "fold_gate_pass": True,
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_aggregate(args):
    output_root = Path(args.output)
    folds = []
    fold_hashes = {}
    heldout_ids = []
    for fold in range(RF1_FOLDS):
        path = output_root / "folds" / f"fov_{PILOT_FOV}_fold_{fold}.npz"
        summary = json.loads(path.with_suffix(".summary.json").read_text())
        if not (
            summary.get("analysis") == "rf1_multiscale_fold_pilot"
            and summary.get("version") == PILOT_VERSION
            and summary.get("pfm_access") is False
            and summary.get("fold") == fold
            and summary.get("output_sha256") == sha256(path)
            and summary.get("fold_gate_pass") is True
        ):
            raise RuntimeError(f"invalid multiscale fold output: {path}")
        with np.load(path) as source:
            value = {name: source[name] for name in source.files}
        folds.append(value)
        heldout_ids.extend(str(item) for item in value["heldout_slide_ids"])
        fold_hashes[str(fold)] = sha256(path)
    if len(heldout_ids) != 109 or len(set(heldout_ids)) != 109:
        raise RuntimeError("held-out folds do not cover 109 unique slides")

    target_power = sum(value["validation_target_power"] for value in folds)
    condition_power = sum(value["validation_condition_power"] for value in folds)
    frequency = folds[0]["radial_frequency"]
    spectrum_rows = []
    for scanner_index, scanner in enumerate(SCANNERS[1:]):
        base_rmse = log_spectrum_rmse(
            condition_power[0, scanner_index], target_power, frequency
        )
        for condition_index, condition in enumerate(CONDITIONS):
            rmse = log_spectrum_rmse(
                condition_power[condition_index, scanner_index], target_power, frequency
            )
            band_output = radial_power_to_band_energy(
                condition_power[condition_index, scanner_index], frequency
            )
            band_target = radial_power_to_band_energy(target_power, frequency)
            band_log_rmse = float(
                np.sqrt(
                    np.mean(
                        (
                            np.log(np.maximum(band_output, 1e-20))
                            - np.log(np.maximum(band_target, 1e-20))
                        )
                        ** 2
                    )
                )
            )
            spectrum_rows.append(
                {
                    "scanner": scanner,
                    "condition": condition,
                    "log_spectrum_rmse": rmse,
                    "relative_change_vs_reinhard": (rmse / base_rmse - 1.0),
                    "three_band_log_energy_rmse": band_log_rmse,
                }
            )

    paired_rows = []
    for scanner_index, scanner in enumerate(SCANNERS[1:]):
        for condition_index, condition in enumerate(CONDITIONS):
            row = {"scanner": scanner, "condition": condition, "patches": 10900}
            for name in ("rgb_mae", "mean_od_mae", "gradient_mae"):
                values = np.concatenate(
                    [fold[name][condition_index, :, scanner_index].reshape(-1) for fold in folds]
                )
                row[f"paired_{name}_mean"] = float(values.mean())
                row[f"paired_{name}_q95"] = float(np.quantile(values, 0.95))
            paired_rows.append(row)

    gamut_rows = []
    for scanner_index, scanner in enumerate(SCANNERS[1:]):
        material = np.concatenate(
            [fold["material_range_fraction"][:, scanner_index].reshape(-1) for fold in folds]
        )
        preproject = np.concatenate(
            [fold["preproject_range_fraction"][:, scanner_index].reshape(-1) for fold in folds]
        )
        projection = np.concatenate(
            [fold["projection_fraction"][:, scanner_index].reshape(-1) for fold in folds]
        )
        projection_mae = np.concatenate(
            [fold["projection_rgb_mae"][:, scanner_index].reshape(-1) for fold in folds]
        )
        clamp = np.concatenate(
            [fold["final_clamp_mae"][:, scanner_index].reshape(-1) for fold in folds]
        )
        gamut_rows.append(
            {
                "scanner": scanner,
                "patches": len(material),
                "preproject_range_fraction_mean": float(preproject.mean()),
                "material_range_fraction_mean": float(material.mean()),
                "material_range_fraction_q99": float(np.quantile(material, 0.99)),
                "projection_fraction_mean": float(projection.mean()),
                "projection_rgb_mae_mean": float(projection_mae.mean()),
                "final_clamp_mae_max": float(clamp.max()),
            }
        )

    gain_rows = []
    for fold_index, fold in enumerate(folds):
        for scanner_index, scanner in enumerate(SCANNERS[1:]):
            gain_rows.append(
                {
                    "fold": fold_index,
                    "scanner": scanner,
                    **{
                        f"sigma_{sigma:g}px_gain": float(
                            fold["multiscale_gains"][scanner_index, band_index]
                        )
                        for band_index, sigma in enumerate(PYRAMID_SIGMAS)
                    },
                }
            )

    _write_csv(output_root / "spectrum_comparison.csv", spectrum_rows)
    _write_csv(output_root / "paired_patch_comparison.csv", paired_rows)
    _write_csv(output_root / "gamut_diagnostics.csv", gamut_rows)
    _write_csv(output_root / "fitted_band_gains.csv", gain_rows)

    multiscale_spectrum = [
        row for row in spectrum_rows if row["condition"] == "multiscale_rf1"
    ]
    radial_spectrum = [row for row in spectrum_rows if row["condition"] == "radial_rf1"]
    multiscale_no_harm = sum(row["relative_change_vs_reinhard"] <= 0 for row in multiscale_spectrum)
    final_clamp_max = max(row["final_clamp_mae_max"] for row in gamut_rows)
    summary = {
        "analysis": "rf1_multiscale_aggregate_pilot",
        "version": PILOT_VERSION,
        "status": "IMAGE_ONLY_PILOT_NOT_RESULT_LOCKED",
        "outcome_access": False,
        "pfm_access": False,
        "fov": PILOT_FOV,
        "folds": RF1_FOLDS,
        "slides": 109,
        "paired_source_patches": 109 * 100 * 5,
        "pyramid_sigmas_pixels": list(PYRAMID_SIGMAS),
        "pyramid_sigmas_micrometres": [
            float(value * TARGET_MPP) for value in PYRAMID_SIGMAS
        ],
        "gain_cap": PILOT_GAIN_CAP,
        "multiscale_improved_scanners_vs_reinhard": multiscale_no_harm,
        "multiscale_mean_relative_spectrum_change": float(
            np.mean([row["relative_change_vs_reinhard"] for row in multiscale_spectrum])
        ),
        "radial_mean_relative_spectrum_change": float(
            np.mean([row["relative_change_vs_reinhard"] for row in radial_spectrum])
        ),
        "final_clamp_mae_max": final_clamp_max,
        "exact_gamut_gate_pass": bool(final_clamp_max <= 1e-6),
        "fold_output_sha256": fold_hashes,
        "artifacts": {
            name: sha256(output_root / name)
            for name in (
                "spectrum_comparison.csv",
                "paired_patch_comparison.csv",
                "gamut_diagnostics.csv",
                "fitted_band_gains.csv",
            )
        },
        "limitations": [
            "Pilot is restricted to the model-native FOV 256 population.",
            "Band gains retain the pooled RF1 energy-ratio estimator to isolate the multiscale representation change.",
            "Training band energy is approximated from locked 72-bin radial sufficient statistics; a future locked run should accumulate native pyramid energy directly.",
            "No PFM endpoint was read; representation benefit remains deliberately unopened.",
        ],
        "pilot_gate_pass": bool(final_clamp_max <= 1e-6 and len(spectrum_rows) == 15),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def _validated_panel_folds(output_root: Path, fov: int):
    folds = []
    hashes = {}
    slide_ids = []
    for fold in range(RF1_FOLDS):
        path = output_root / "folds" / f"fov_{fov}_fold_{fold}.npz"
        summary = json.loads(path.with_suffix(".summary.json").read_text())
        if not (
            summary.get("analysis") == "rf1_multiscale_fold_pilot"
            and summary.get("version") == PILOT_VERSION
            and summary.get("outcome_access") is False
            and summary.get("pfm_access") is False
            and summary.get("fov") == fov
            and summary.get("fold") == fold
            and summary.get("output_sha256") == sha256(path)
            and summary.get("fold_gate_pass") is True
        ):
            raise RuntimeError(f"invalid multiscale panel fold output: {path}")
        with np.load(path) as source:
            value = {name: source[name] for name in source.files}
        folds.append(value)
        slide_ids.extend(str(item) for item in value["heldout_slide_ids"])
        hashes[f"fov_{fov}_fold_{fold}"] = sha256(path)
    if len(slide_ids) != 109 or len(set(slide_ids)) != 109:
        raise RuntimeError(f"FOV {fov}: folds do not cover 109 unique slides")
    return folds, hashes


def run_panel_aggregate(args):
    """Aggregate all three FOVs without replacing the original FOV256 summary."""
    output_root = Path(args.output)
    spectrum_rows = []
    paired_rows = []
    gamut_rows = []
    gain_rows = []
    fold_hashes = {}
    for fov in PILOT_FOVS:
        folds, hashes = _validated_panel_folds(output_root, fov)
        fold_hashes.update(hashes)
        target_power = sum(value["validation_target_power"] for value in folds)
        condition_power = sum(value["validation_condition_power"] for value in folds)
        frequency = folds[0]["radial_frequency"]
        for scanner_index, scanner in enumerate(SCANNERS[1:]):
            base_rmse = log_spectrum_rmse(
                condition_power[0, scanner_index], target_power, frequency
            )
            for condition_index, condition in enumerate(CONDITIONS):
                rmse = log_spectrum_rmse(
                    condition_power[condition_index, scanner_index], target_power, frequency
                )
                band_output = radial_power_to_band_energy(
                    condition_power[condition_index, scanner_index], frequency
                )
                band_target = radial_power_to_band_energy(target_power, frequency)
                spectrum_rows.append(
                    {
                        "fov": fov,
                        "scanner": scanner,
                        "condition": condition,
                        "log_spectrum_rmse": rmse,
                        "relative_change_vs_reinhard": rmse / base_rmse - 1.0,
                        "three_band_log_energy_rmse": float(
                            np.sqrt(
                                np.mean(
                                    (
                                        np.log(np.maximum(band_output, 1e-20))
                                        - np.log(np.maximum(band_target, 1e-20))
                                    )
                                    ** 2
                                )
                            )
                        ),
                    }
                )
                row = {
                    "fov": fov,
                    "scanner": scanner,
                    "condition": condition,
                    "patches": 10900,
                }
                for name in ("rgb_mae", "mean_od_mae", "gradient_mae"):
                    values = np.concatenate(
                        [
                            fold[name][condition_index, :, scanner_index].reshape(-1)
                            for fold in folds
                        ]
                    )
                    row[f"paired_{name}_mean"] = float(values.mean())
                    row[f"paired_{name}_q95"] = float(np.quantile(values, 0.95))
                paired_rows.append(row)

            material = np.concatenate(
                [fold["material_range_fraction"][:, scanner_index].reshape(-1) for fold in folds]
            )
            preproject = np.concatenate(
                [
                    fold["preproject_range_fraction"][:, scanner_index].reshape(-1)
                    for fold in folds
                ]
            )
            projection = np.concatenate(
                [fold["projection_fraction"][:, scanner_index].reshape(-1) for fold in folds]
            )
            projection_mae = np.concatenate(
                [fold["projection_rgb_mae"][:, scanner_index].reshape(-1) for fold in folds]
            )
            clamp = np.concatenate(
                [fold["final_clamp_mae"][:, scanner_index].reshape(-1) for fold in folds]
            )
            gamut_rows.append(
                {
                    "fov": fov,
                    "scanner": scanner,
                    "patches": len(material),
                    "preproject_range_fraction_mean": float(preproject.mean()),
                    "material_range_fraction_mean": float(material.mean()),
                    "material_range_fraction_q99": float(np.quantile(material, 0.99)),
                    "projection_fraction_mean": float(projection.mean()),
                    "projection_rgb_mae_mean": float(projection_mae.mean()),
                    "final_clamp_mae_max": float(clamp.max()),
                }
            )
        for fold_index, fold in enumerate(folds):
            for scanner_index, scanner in enumerate(SCANNERS[1:]):
                gain_rows.append(
                    {
                        "fov": fov,
                        "fold": fold_index,
                        "scanner": scanner,
                        **{
                            f"sigma_{sigma:g}px_gain": float(
                                fold["multiscale_gains"][scanner_index, band_index]
                            )
                            for band_index, sigma in enumerate(PYRAMID_SIGMAS)
                        },
                    }
                )

    names = {
        "panel_spectrum_comparison.csv": spectrum_rows,
        "panel_paired_patch_comparison.csv": paired_rows,
        "panel_gamut_diagnostics.csv": gamut_rows,
        "panel_fitted_band_gains.csv": gain_rows,
    }
    for name, rows in names.items():
        _write_csv(output_root / name, rows)
    multiscale = [row for row in spectrum_rows if row["condition"] == "multiscale_rf1"]
    radial = [row for row in spectrum_rows if row["condition"] == "radial_rf1"]
    final_clamp_max = max(row["final_clamp_mae_max"] for row in gamut_rows)
    no_harm = sum(row["relative_change_vs_reinhard"] <= 0 for row in multiscale)
    by_fov = {}
    for fov in PILOT_FOVS:
        selected = [row for row in multiscale if row["fov"] == fov]
        by_fov[str(fov)] = {
            "improved_scanners_vs_reinhard": sum(
                row["relative_change_vs_reinhard"] <= 0 for row in selected
            ),
            "mean_relative_spectrum_change": float(
                np.mean([row["relative_change_vs_reinhard"] for row in selected])
            ),
        }
    summary = {
        "analysis": "rf1_multiscale_panel_aggregate_pilot",
        "version": PILOT_VERSION,
        "status": "IMAGE_ONLY_PILOT_NOT_RESULT_LOCKED",
        "outcome_access": False,
        "pfm_access": False,
        "fovs": list(PILOT_FOVS),
        "folds_per_fov": RF1_FOLDS,
        "slides_per_fov": 109,
        "paired_source_patches": 109 * 100 * 5 * len(PILOT_FOVS),
        "pyramid_sigmas_pixels": list(PYRAMID_SIGMAS),
        "gain_cap": PILOT_GAIN_CAP,
        "multiscale_improved_cells_vs_reinhard": no_harm,
        "multiscale_cells": len(multiscale),
        "multiscale_mean_relative_spectrum_change": float(
            np.mean([row["relative_change_vs_reinhard"] for row in multiscale])
        ),
        "radial_mean_relative_spectrum_change": float(
            np.mean([row["relative_change_vs_reinhard"] for row in radial])
        ),
        "by_fov": by_fov,
        "final_clamp_mae_max": final_clamp_max,
        "exact_gamut_gate_pass": bool(final_clamp_max <= 1e-6),
        "fold_output_sha256": fold_hashes,
        "artifacts": {name: sha256(output_root / name) for name in names},
        "limitations": [
            "Band gains retain the pooled RF1 energy-ratio estimator to isolate the multiscale representation change.",
            "Training band energy is approximated from locked 72-bin radial sufficient statistics; a future locked run should accumulate native pyramid energy directly.",
            "No PFM endpoint was read; representation benefit remains deliberately unopened.",
        ],
        "panel_gate_pass": bool(
            final_clamp_max <= 1e-6
            and len(spectrum_rows) == 45
            and len(paired_rows) == 45
            and len(gamut_rows) == 15
            and len(gain_rows) == 75
        ),
    }
    (output_root / "panel_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def run_visual(args):
    """Render the frozen five-example source-to-AT2 multiscale audit panel."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest_path = Path(args.sample_manifest)
    with manifest_path.open(newline="") as handle:
        samples = list(csv.DictReader(handle))
    if len(samples) != 5 or {row["scanner"] for row in samples} != set(SCANNERS[1:]):
        raise RuntimeError("visual audit requires the frozen five-scanner manifest")

    output_root = Path(args.output) / "visual"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    images = []
    grid_root = Path(args.grid)
    with torch.inference_mode():
        for row in samples:
            scanner = row["scanner"]
            scanner_index = SCANNERS.index(scanner)
            slide_id = row["slide_id"]
            location_index = int(row["location_index"])
            fold = int(row["fold"])
            _, statistics = _load_statistics(Path(args.statistics), fold, PILOT_FOV)
            lab_mean = torch.as_tensor(
                statistics["lab_mean"], dtype=torch.float32, device=device
            )
            lab_std = torch.as_tensor(
                statistics["lab_std"], dtype=torch.float32, device=device
            )
            with h5py.File(grid_root / f"{slide_id}.h5", "r") as source:
                target = rgb8_to_rgb01(
                    centered_crop(source["rgb"][0, location_index : location_index + 1], PILOT_FOV),
                    device=device,
                )
                raw = rgb8_to_rgb01(
                    centered_crop(
                        source["rgb"][scanner_index, location_index : location_index + 1],
                        PILOT_FOV,
                    ),
                    device=device,
                )
            base = reinhard_lab(
                raw,
                lab_mean[scanner_index].reshape(1, 1, 1, 3),
                lab_std[scanner_index].reshape(1, 1, 1, 3),
                lab_mean[0].reshape(1, 1, 1, 3),
                lab_std[0].reshape(1, 1, 1, 3),
            )["output"]
            radial_gains = fitted_residual_gains(
                statistics["post_reinhard_source_power"],
                statistics["raw_at2_target_power"],
                statistics["radial_frequency"],
                PILOT_GAIN_CAP,
            )
            radial = shared_od_residual_frequency(
                base, radial_gains[scanner_index - 1], statistics["radial_frequency"]
            )["output"]
            gains = fitted_multiscale_gains(
                statistics["post_reinhard_source_power"],
                statistics["raw_at2_target_power"],
                statistics["radial_frequency"],
            )
            multiscale_report = shared_od_multiscale(base, gains[scanner_index - 1])
            multiscale = multiscale_report["output"]
            residual = (0.5 + 10.0 * (multiscale - base)).clamp(0.0, 1.0)
            images.append(
                [value[0].cpu().numpy() for value in (target, raw, base, radial, multiscale, residual)]
            )
            base_mae = float((base - target).abs().mean().cpu())
            radial_mae = float((radial - target).abs().mean().cpu())
            multiscale_mae = float((multiscale - target).abs().mean().cpu())
            rows.append(
                {
                    "scanner": scanner,
                    "slide_id": slide_id,
                    "location_index": location_index,
                    "fold": fold,
                    "reinhard_paired_rgb_mae": base_mae,
                    "radial_rf1_paired_rgb_mae": radial_mae,
                    "multiscale_rf1_paired_rgb_mae": multiscale_mae,
                    "multiscale_vs_reinhard_rgb_mae": float(
                        (multiscale - base).abs().mean().cpu()
                    ),
                    "multiscale_projection_rgb_mae": float(
                        multiscale_report["projection_rgb_mae"].cpu()[0]
                    ),
                    "multiscale_final_clamp_mae": float(
                        multiscale_report["final_clamp_mae"].cpu()[0]
                    ),
                }
            )

    titles = (
        "Paired target AT2",
        "Raw source",
        "Reinhard",
        "Current radial RF1",
        "Multiscale RF1",
        "Signed residual x10",
    )
    figure, axes = plt.subplots(5, 6, figsize=(21, 18), constrained_layout=True)
    for row_index, (sample, row_images) in enumerate(zip(samples, images, strict=True)):
        for column_index, image in enumerate(row_images):
            axes[row_index, column_index].imshow(image)
            axes[row_index, column_index].set_xticks([])
            axes[row_index, column_index].set_yticks([])
            if row_index == 0:
                axes[row_index, column_index].set_title(titles[column_index], fontsize=13)
        axes[row_index, 0].set_ylabel(
            f"{sample['scanner'].upper()}\n{sample['slide_id']} / loc {sample['location_index']}",
            fontsize=11,
        )
    figure.suptitle(
        "RF1 multiscale image-only pilot | cross-fitted FOV 256 | cap 1.25",
        fontsize=17,
    )
    png = output_root / "paired_patch_multiscale_comparison.png"
    pdf = output_root / "paired_patch_multiscale_comparison.pdf"
    figure.savefig(png, dpi=180, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)
    metrics_path = output_root / "paired_patch_multiscale_metrics.csv"
    _write_csv(metrics_path, rows)
    summary = {
        "analysis": "rf1_multiscale_visual_pilot",
        "version": PILOT_VERSION,
        "outcome_access": False,
        "pfm_access": False,
        "examples": len(rows),
        "fov": PILOT_FOV,
        "gain_cap": PILOT_GAIN_CAP,
        "device": str(device),
        "sample_manifest": str(manifest_path.resolve()),
        "sample_manifest_sha256": sha256(manifest_path),
        "png": str(png.resolve()),
        "png_sha256": sha256(png),
        "pdf": str(pdf.resolve()),
        "pdf_sha256": sha256(pdf),
        "metrics": str(metrics_path.resolve()),
        "metrics_sha256": sha256(metrics_path),
        "final_clamp_mae_max": max(row["multiscale_final_clamp_mae"] for row in rows),
        "visual_gate_pass": True,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    if args.mode == "fold":
        run_fold(args)
    elif args.mode == "aggregate":
        run_aggregate(args)
    elif args.mode == "panel-aggregate":
        run_panel_aggregate(args)
    else:
        run_visual(args)


if __name__ == "__main__":
    main()
