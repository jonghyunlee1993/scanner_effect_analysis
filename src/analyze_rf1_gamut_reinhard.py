"""Evaluate source-ray gamut-safe Reinhard on outer-held-out paired patches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from e5_comparator_population import (
    FOVS,
    SCANNERS,
    batch_radial_power,
    centered_crop,
    radial_geometry,
    reinhard_lab,
    rgb01_to_od,
    rgb8_to_rgb01,
)
from e5_reinhard_residual_frequency import RF1_FOLDS, log_spectrum_rmse
from fetch_e0_pfm_checkpoints import sha256
from rf1_gamut_reinhard import GAMUT_REINHARD_VERSION, reinhard_lab_source_ray


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--statistics", default="outputs/e5_rf1_fold_statistics")
    parser.add_argument(
        "--pilot-contract", default="docs/e5_rf1_improvement_pilot_contract.md"
    )
    parser.add_argument(
        "--output", default="outputs/rf1_improvement_pilot/gamut_reinhard/shards"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def selected_cell(task_index: int) -> tuple[int, int]:
    if not 0 <= int(task_index) < len(FOVS) * RF1_FOLDS:
        raise ValueError("task index must select one of 15 FOV/fold cells")
    return FOVS[int(task_index) // RF1_FOLDS], int(task_index) % RF1_FOLDS


def mean_od_gradient_mae(output, target):
    source_mean = rgb01_to_od(output).mean(dim=-1)
    target_mean = rgb01_to_od(target).mean(dim=-1)
    source_dx = source_mean[:, :, 1:] - source_mean[:, :, :-1]
    target_dx = target_mean[:, :, 1:] - target_mean[:, :, :-1]
    source_dy = source_mean[:, 1:, :] - source_mean[:, :-1, :]
    target_dy = target_mean[:, 1:, :] - target_mean[:, :-1, :]
    x_error = (source_dx - target_dx).abs().mean(dim=(1, 2))
    y_error = (source_dy - target_dy).abs().mean(dim=(1, 2))
    return 0.5 * (x_error + y_error)


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("gamut-Reinhard population pilot requires a visible CUDA device")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    fov, fold = selected_cell(args.task_index)

    contract_path = Path(args.pilot_contract)
    if "FROZEN BEFORE NEW VARIANT PFM ACCESS" not in contract_path.read_text():
        raise RuntimeError("RF1 improvement pilot contract is not frozen")
    contract_sha = sha256(contract_path)
    statistics_path = Path(args.statistics) / f"fov_{fov}_fold_{fold}.npz"
    statistics_summary_path = statistics_path.with_suffix(".summary.json")
    statistics_summary = json.loads(statistics_summary_path.read_text())
    if not statistics_summary.get("statistics_gate_pass"):
        raise RuntimeError(f"invalid RF1 fold statistics: {statistics_path}")
    with np.load(statistics_path) as source:
        statistics = {name: source[name] for name in source.files}

    heldout_ids = [str(value) for value in statistics["heldout_slide_ids"]]
    if len(heldout_ids) not in (21, 22):
        raise RuntimeError("outer-held-out RF1 fold has an invalid slide count")
    lab_mean = torch.as_tensor(statistics["lab_mean"], dtype=torch.float32, device="cuda")
    lab_std = torch.as_tensor(statistics["lab_std"], dtype=torch.float32, device="cuda")
    geometry = radial_geometry(fov)
    target_power = np.zeros((len(SCANNERS) - 1, len(geometry.frequency)), dtype=np.float64)
    clipped_power = np.zeros_like(target_power)
    ray_power = np.zeros_like(target_power)
    rows: list[dict[str, object]] = []
    grid_root = Path(args.grid)

    with torch.inference_mode():
        for slide_number, slide_id in enumerate(heldout_ids, start=1):
            grid_path = grid_root / f"{slide_id}.h5"
            with h5py.File(grid_path, "r") as source:
                if str(source.attrs["slide_id"]) != slide_id:
                    raise RuntimeError(f"grid slide identity mismatch: {grid_path}")
                location_ids = np.asarray(source["location_id"][:]).astype(str)
                for scanner_index, scanner in enumerate(SCANNERS[1:], start=1):
                    for start in range(0, 100, args.batch_size):
                        stop = min(start + args.batch_size, 100)
                        target = rgb8_to_rgb01(
                            centered_crop(source["rgb"][0, start:stop], fov), device="cuda"
                        )
                        raw = rgb8_to_rgb01(
                            centered_crop(source["rgb"][scanner_index, start:stop], fov),
                            device="cuda",
                        )
                        clipped = reinhard_lab(
                            raw,
                            lab_mean[scanner_index],
                            lab_std[scanner_index],
                            lab_mean[0],
                            lab_std[0],
                        )
                        ray = reinhard_lab_source_ray(
                            raw,
                            lab_mean[scanner_index],
                            lab_std[scanner_index],
                            lab_mean[0],
                            lab_std[0],
                        )
                        target_power[scanner_index - 1] += batch_radial_power(
                            target, geometry
                        ).cpu().numpy()
                        clipped_power[scanner_index - 1] += batch_radial_power(
                            clipped["output"], geometry
                        ).cpu().numpy()
                        ray_power[scanner_index - 1] += batch_radial_power(
                            ray["output"], geometry
                        ).cpu().numpy()

                        raw_mae = (raw - target).abs().mean(dim=(1, 2, 3))
                        clipped_mae = (clipped["output"] - target).abs().mean(dim=(1, 2, 3))
                        ray_mae = (ray["output"] - target).abs().mean(dim=(1, 2, 3))
                        clipped_gradient = mean_od_gradient_mae(clipped["output"], target)
                        ray_gradient = mean_od_gradient_mae(ray["output"], target)
                        ray_vs_clipped = (ray["output"] - clipped["output"]).abs().mean(
                            dim=(1, 2, 3)
                        )
                        for offset in range(stop - start):
                            rows.append(
                                {
                                    "fov": fov,
                                    "fold": fold,
                                    "slide_id": slide_id,
                                    "location_index": start + offset,
                                    "location_id": location_ids[start + offset],
                                    "scanner": scanner,
                                    "raw_rgb_mae_to_at2": float(raw_mae[offset].cpu()),
                                    "clipped_rgb_mae_to_at2": float(clipped_mae[offset].cpu()),
                                    "ray_rgb_mae_to_at2": float(ray_mae[offset].cpu()),
                                    "ray_minus_clipped_rgb_mae": float(
                                        (ray_mae[offset] - clipped_mae[offset]).cpu()
                                    ),
                                    "clipped_mean_od_gradient_mae": float(
                                        clipped_gradient[offset].cpu()
                                    ),
                                    "ray_mean_od_gradient_mae": float(ray_gradient[offset].cpu()),
                                    "ray_minus_clipped_gradient_mae": float(
                                        (ray_gradient[offset] - clipped_gradient[offset]).cpu()
                                    ),
                                    "base_preclip_range_fraction": float(
                                        clipped["preclip_range_fraction"][offset].cpu()
                                    ),
                                    "base_clip_pixel_mae": float(
                                        clipped["clip_pixel_mae"][offset].cpu()
                                    ),
                                    "ray_limited_pixel_fraction": float(
                                        ray["limited_pixel_fraction"][offset].cpu()
                                    ),
                                    "ray_mean_scale_loss": float(
                                        ray["mean_scale_loss"][offset].cpu()
                                    ),
                                    "ray_projection_rgb_mae": float(
                                        ray["projection_rgb_mae"][offset].cpu()
                                    ),
                                    "ray_final_clamp_mae": float(
                                        ray["final_clamp_mae"][offset].cpu()
                                    ),
                                    "ray_vs_clipped_rgb_mae": float(
                                        ray_vs_clipped[offset].cpu()
                                    ),
                                }
                            )
            print(f"[{slide_number}/{len(heldout_ids)}] FOV {fov} fold {fold} {slide_id}", flush=True)

    frame = pd.DataFrame(rows)
    expected_rows = len(heldout_ids) * 100 * (len(SCANNERS) - 1)
    if len(frame) != expected_rows or not np.isfinite(frame.select_dtypes("number")).all().all():
        raise RuntimeError("gamut-Reinhard pilot population is incomplete or non-finite")
    spectrum_rows = []
    for scanner_index, scanner in enumerate(SCANNERS[1:]):
        clipped_rmse = log_spectrum_rmse(
            clipped_power[scanner_index], target_power[scanner_index], geometry.frequency
        )
        ray_rmse = log_spectrum_rmse(
            ray_power[scanner_index], target_power[scanner_index], geometry.frequency
        )
        spectrum_rows.append(
            {
                "fov": fov,
                "fold": fold,
                "scanner": scanner,
                "clipped_log_spectrum_rmse": clipped_rmse,
                "ray_log_spectrum_rmse": ray_rmse,
                "ray_minus_clipped_log_spectrum_rmse": ray_rmse - clipped_rmse,
            }
        )

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = f"fov_{fov}_fold_{fold}"
    patch_path = output_root / f"{stem}.patches.csv"
    spectrum_path = output_root / f"{stem}.spectrum.csv"
    array_path = output_root / f"{stem}.npz"
    frame.to_csv(patch_path, index=False)
    pd.DataFrame(spectrum_rows).to_csv(spectrum_path, index=False)
    temporary = output_root / f".{stem}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        temporary,
        radial_frequency=geometry.frequency,
        target_power=target_power,
        clipped_power=clipped_power,
        ray_power=ray_power,
    )
    os.replace(temporary, array_path)
    summary = {
        "analysis": "rf1_gamut_reinhard_outer_fold_pilot",
        "version": GAMUT_REINHARD_VERSION,
        "outcome_access": False,
        "fov": fov,
        "fold": fold,
        "heldout_slides": len(heldout_ids),
        "patch_rows": len(frame),
        "statistics_sha256": sha256(statistics_path),
        "pilot_contract_sha256": contract_sha,
        "patch_csv_sha256": sha256(patch_path),
        "spectrum_csv_sha256": sha256(spectrum_path),
        "array_sha256": sha256(array_path),
        "max_final_clamp_mae": float(frame["ray_final_clamp_mae"].max()),
        "pilot_gate_pass": bool(
            len(frame) == expected_rows and frame["ray_final_clamp_mae"].max() <= 1e-6
        ),
        "device": torch.cuda.get_device_name(0),
    }
    summary_path = output_root / f"{stem}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["pilot_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

