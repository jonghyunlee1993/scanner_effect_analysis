"""Audit all input-only E5-RF1 gain-cap candidates for one FOV/fold."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from e5_comparator_population import (
    FOVS,
    RADIAL_BINS,
    SCANNERS,
    batch_radial_power,
    centered_crop,
    radial_geometry,
    reinhard_lab,
    rgb8_to_rgb01,
)
from e5_reinhard_residual_frequency import (
    RF1_FOLDS,
    RF1_GAIN_CAP_CANDIDATES,
    RF1_VERSION,
    fitted_residual_gains,
    shared_od_residual_frequency_many,
)
from fetch_e0_pfm_checkpoints import sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--fov", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--statistics", default="outputs/e5_rf1_fold_statistics")
    parser.add_argument(
        "--execution-contract", default="docs/e5_reinhard_residual_frequency_contract.md"
    )
    parser.add_argument("--output", default="outputs/e5_rf1_input_fold_audit")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def selected_cell(args) -> tuple[int, int]:
    if args.task_index is not None:
        if not 0 <= args.task_index < len(FOVS) * RF1_FOLDS:
            raise ValueError("RF1 audit task index is out of range")
        return FOVS[args.task_index // RF1_FOLDS], args.task_index % RF1_FOLDS
    if args.fov not in FOVS or args.fold is None or not 0 <= args.fold < RF1_FOLDS:
        raise ValueError("provide a valid --task-index or --fov/--fold pair")
    return int(args.fov), int(args.fold)


def existing_output_passes(
    path: Path,
    summary_path: Path,
    statistics_sha: str,
    contract_sha: str,
):
    if not path.exists() or not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        return bool(
            summary.get("analysis") == "e5_rf1_input_fold_audit"
            and summary.get("rf1_version") == RF1_VERSION
            and summary.get("outcome_access") is False
            and summary.get("statistics_sha256") == statistics_sha
            and summary.get("execution_contract_sha256") == contract_sha
            and summary.get("output_sha256") == sha256(path)
            and summary.get("input_fold_gate_pass") is True
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E5-RF1 input audit requires a visible CUDA device")
    fov, fold = selected_cell(args)
    contract_path = Path(args.execution_contract)
    contract_sha = sha256(contract_path)
    statistics_path = Path(args.statistics) / f"fov_{fov}_fold_{fold}.npz"
    statistics_summary_path = statistics_path.with_suffix(".summary.json")
    statistics_summary = json.loads(statistics_summary_path.read_text())
    if not (
        statistics_summary.get("analysis") == "e5_rf1_fold_statistics"
        and statistics_summary.get("rf1_version") == RF1_VERSION
        and statistics_summary.get("fov") == fov
        and statistics_summary.get("fold") == fold
        and statistics_summary.get("execution_contract_sha256") == contract_sha
        and statistics_summary.get("output_sha256") == sha256(statistics_path)
        and statistics_summary.get("statistics_gate_pass") is True
    ):
        raise RuntimeError("invalid E5-RF1 fold statistics")
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"fov_{fov}_fold_{fold}.npz"
    summary_path = output_path.with_suffix(".summary.json")
    if existing_output_passes(
        output_path,
        summary_path,
        sha256(statistics_path),
        contract_sha,
    ):
        print(summary_path.read_text(), end="")
        return
    with np.load(statistics_path) as source:
        statistics = {name: source[name] for name in source.files}

    heldout_ids = [str(value) for value in statistics["heldout_slide_ids"]]
    caps = np.asarray(RF1_GAIN_CAP_CANDIDATES, dtype=np.float64)
    gains = np.stack(
        [
            fitted_residual_gains(
                statistics["post_reinhard_source_power"],
                statistics["raw_at2_target_power"],
                statistics["radial_frequency"],
                float(cap),
            )
            for cap in caps
        ],
        axis=0,
    )
    n_caps = len(caps)
    n_slides = len(heldout_ids)
    shape = (n_caps, n_slides, len(SCANNERS) - 1, 100)
    metric_names = (
        "preproject_range_fraction",
        "material_range_fraction",
        "projection_fraction",
        "projection_rgb_mae",
        "final_clamp_mae",
    )
    metrics = {name: np.empty(shape, dtype=np.float32) for name in metric_names}
    base_clip_fraction = np.empty((n_slides, len(SCANNERS) - 1, 100), dtype=np.float32)
    base_clip_mae = np.empty_like(base_clip_fraction)
    validation_target_power = np.zeros(RADIAL_BINS, dtype=np.float64)
    validation_base_power = np.zeros((len(SCANNERS) - 1, RADIAL_BINS), dtype=np.float64)
    validation_output_power = np.zeros(
        (n_caps, len(SCANNERS) - 1, RADIAL_BINS), dtype=np.float64
    )
    validation_target_images = 0
    validation_source_images = np.zeros(len(SCANNERS) - 1, dtype=np.int64)

    geometry = radial_geometry(fov)
    lab_mean = torch.as_tensor(statistics["lab_mean"], dtype=torch.float32, device="cuda")
    lab_std = torch.as_tensor(statistics["lab_std"], dtype=torch.float32, device="cuda")
    grid_root = Path(args.grid)
    with torch.inference_mode():
        for slide_index, slide_id in enumerate(heldout_ids):
            path = grid_root / f"{slide_id}.h5"
            with h5py.File(path, "r") as source:
                for start in range(0, 100, args.batch_size):
                    stop = min(start + args.batch_size, 100)
                    at2 = rgb8_to_rgb01(
                        centered_crop(source["rgb"][0, start:stop], fov), device="cuda"
                    )
                    validation_target_power += batch_radial_power(at2, geometry).cpu().numpy()
                    validation_target_images += stop - start
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
                        )
                        base_clip_fraction[slide_index, scanner_index - 1, start:stop] = (
                            base["preclip_range_fraction"].cpu().numpy()
                        )
                        base_clip_mae[slide_index, scanner_index - 1, start:stop] = (
                            base["clip_pixel_mae"].cpu().numpy()
                        )
                        validation_base_power[scanner_index - 1] += batch_radial_power(
                            base["output"], geometry
                        ).cpu().numpy()
                        report = shared_od_residual_frequency_many(
                            base["output"],
                            gains[:, scanner_index - 1],
                            statistics["radial_frequency"],
                        )
                        for name in metric_names:
                            metrics[name][:, slide_index, scanner_index - 1, start:stop] = (
                                report[name].cpu().numpy()
                            )
                        for cap_index in range(n_caps):
                            validation_output_power[cap_index, scanner_index - 1] += (
                                batch_radial_power(report["output"][cap_index], geometry)
                                .cpu()
                                .numpy()
                            )
                        validation_source_images[scanner_index - 1] += stop - start
            print(
                f"[{slide_index + 1}/{n_slides}] FOV {fov} fold {fold} {slide_id}",
                flush=True,
            )

    expected_images = n_slides * 100
    if not (
        validation_target_images == expected_images
        and np.all(validation_source_images == expected_images)
        and all(np.isfinite(value).all() for value in metrics.values())
        and np.isfinite(validation_output_power).all()
    ):
        raise RuntimeError("E5-RF1 validation input audit is incomplete")

    temporary = output_root / f".fov_{fov}_fold_{fold}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        temporary,
        fov=np.asarray(fov, dtype=np.int64),
        fold=np.asarray(fold, dtype=np.int64),
        caps=caps,
        heldout_slide_ids=np.asarray(heldout_ids),
        scanners=np.asarray(SCANNERS[1:]),
        base_clip_fraction=base_clip_fraction,
        base_clip_mae=base_clip_mae,
        validation_target_power=validation_target_power,
        validation_base_power=validation_base_power,
        validation_output_power=validation_output_power,
        validation_target_images=np.asarray(validation_target_images, dtype=np.int64),
        validation_source_images=validation_source_images,
        **metrics,
    )
    os.replace(temporary, output_path)
    summary = {
        "analysis": "e5_rf1_input_fold_audit",
        "rf1_version": RF1_VERSION,
        "outcome_access": False,
        "fov": fov,
        "fold": fold,
        "heldout_slides": n_slides,
        "candidate_caps": caps.tolist(),
        "statistics_sha256": sha256(statistics_path),
        "execution_contract_sha256": contract_sha,
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "input_fold_gate_pass": True,
        "device": torch.cuda.get_device_name(0),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
