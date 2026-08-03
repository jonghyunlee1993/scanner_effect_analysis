"""Build one FOV/fold post-Reinhard spectrum fit for E5-RF1."""

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
    mean_std,
    radial_geometry,
    reinhard_lab,
    rgb8_to_rgb01,
)
from e5_reinhard_residual_frequency import (
    RF1_FOLDS,
    RF1_VERSION,
    fold_assignments,
    fold_counts,
)
from fetch_e0_pfm_checkpoints import sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--fov", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--grid-audit", default="outputs/e0_native_aa_grid/audit/summary.json")
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--e5-statistics", default="outputs/e5_image_statistics")
    parser.add_argument(
        "--execution-contract", default="docs/e5_reinhard_residual_frequency_contract.md"
    )
    parser.add_argument("--output", default="outputs/e5_rf1_fold_statistics")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def selected_cell(args) -> tuple[int, int]:
    if args.task_index is not None:
        if not 0 <= args.task_index < len(FOVS) * RF1_FOLDS:
            raise ValueError("RF1 statistics task index is out of range")
        return FOVS[args.task_index // RF1_FOLDS], args.task_index % RF1_FOLDS
    if args.fov not in FOVS or args.fold is None or not 0 <= args.fold < RF1_FOLDS:
        raise ValueError("provide a valid --task-index or --fov/--fold pair")
    return int(args.fov), int(args.fold)


def load_e5_statistics(root: Path, fov: int, grid_audit_sha: str):
    path = root / f"fov_{fov}.npz"
    summary_path = root / f"fov_{fov}.summary.json"
    summary = json.loads(summary_path.read_text())
    if not (
        summary.get("analysis") == "e5_image_sufficient_statistics"
        and summary.get("fov") == fov
        and summary.get("grid_audit_sha256") == grid_audit_sha
        and summary.get("output_sha256") == sha256(path)
        and summary.get("statistics_gate_pass") is True
    ):
        raise RuntimeError(f"invalid locked E5 image statistics: {path}")
    with np.load(path) as source:
        values = {name: source[name] for name in source.files}
    return path, summary, values


def existing_output_passes(path: Path, summary_path: Path, contract_sha: str, source_sha: str):
    if not path.exists() or not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        return bool(
            summary.get("analysis") == "e5_rf1_fold_statistics"
            and summary.get("rf1_version") == RF1_VERSION
            and summary.get("execution_contract_sha256") == contract_sha
            and summary.get("e5_statistics_sha256") == source_sha
            and summary.get("output_sha256") == sha256(path)
            and summary.get("statistics_gate_pass") is True
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E5-RF1 fold statistics require a visible CUDA device")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    fov, fold = selected_cell(args)

    contract_path = Path(args.execution_contract)
    if "**Status:** PRE-OUTCOME FROZEN" not in contract_path.read_text():
        raise RuntimeError("E5-RF1 execution contract is not frozen")
    contract_sha = sha256(contract_path)
    audit_path = Path(args.grid_audit)
    audit = json.loads(audit_path.read_text())
    if audit.get("grid_gate_pass") is not True or audit.get("patches_observed") != 65_400:
        raise RuntimeError("native-AA grid audit has not passed")
    audit_sha = sha256(audit_path)
    source_path, source_summary, statistics = load_e5_statistics(
        Path(args.e5_statistics), fov, audit_sha
    )

    slide_ids = [str(value) for value in statistics["slide_ids"]]
    assignments = fold_assignments(slide_ids)
    counts = fold_counts(assignments)
    heldout_ids = [value for value in slide_ids if assignments[value] == fold]
    train_ids = [value for value in slide_ids if assignments[value] != fold]
    if counts != (22, 22, 22, 22, 21) or len(train_ids) + len(heldout_ids) != 109:
        raise RuntimeError(f"invalid RF1 fold population: {counts}")
    train_indices = np.asarray([slide_ids.index(value) for value in train_ids], dtype=int)

    lab_sum = statistics["lab_sum"][train_indices].sum(axis=0)
    lab_square = statistics["lab_square_sum"][train_indices].sum(axis=0)
    lab_count = statistics["lab_count"][train_indices].sum(axis=0)
    lab_mean = np.empty((len(SCANNERS), 3), dtype=np.float64)
    lab_std = np.empty_like(lab_mean)
    for scanner_index in range(len(SCANNERS)):
        lab_mean[scanner_index], lab_std[scanner_index] = mean_std(
            lab_sum[scanner_index], lab_square[scanner_index], int(lab_count[scanner_index])
        )

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"fov_{fov}_fold_{fold}.npz"
    summary_path = output_root / f"fov_{fov}_fold_{fold}.summary.json"
    if existing_output_passes(
        output_path, summary_path, contract_sha, source_summary["output_sha256"]
    ):
        print(summary_path.read_text(), end="")
        return

    geometry = radial_geometry(fov)
    source_power = np.zeros((len(SCANNERS) - 1, RADIAL_BINS), dtype=np.float64)
    source_images = np.zeros(len(SCANNERS) - 1, dtype=np.int64)
    base_clip_sum = np.zeros(len(SCANNERS) - 1, dtype=np.float64)
    base_clip_mae_sum = np.zeros(len(SCANNERS) - 1, dtype=np.float64)

    mean_tensor = torch.as_tensor(lab_mean, dtype=torch.float32, device="cuda")
    std_tensor = torch.as_tensor(lab_std, dtype=torch.float32, device="cuda")
    grid_root = Path(args.grid)
    with torch.inference_mode():
        for slide_number, slide_id in enumerate(train_ids, start=1):
            path = grid_root / f"{slide_id}.h5"
            with h5py.File(path, "r") as source:
                if str(source.attrs["slide_id"]) != slide_id:
                    raise ValueError(f"{path}: slide identity mismatch")
                for scanner_index in range(1, len(SCANNERS)):
                    for start in range(0, 100, args.batch_size):
                        stop = min(start + args.batch_size, 100)
                        rgb8 = centered_crop(source["rgb"][scanner_index, start:stop], fov)
                        rgb = rgb8_to_rgb01(rgb8, device="cuda")
                        report = reinhard_lab(
                            rgb,
                            mean_tensor[scanner_index],
                            std_tensor[scanner_index],
                            mean_tensor[0],
                            std_tensor[0],
                        )
                        source_power[scanner_index - 1] += batch_radial_power(
                            report["output"], geometry
                        ).cpu().numpy()
                        base_clip_sum[scanner_index - 1] += float(
                            report["preclip_range_fraction"].sum().cpu()
                        )
                        base_clip_mae_sum[scanner_index - 1] += float(
                            report["clip_pixel_mae"].sum().cpu()
                        )
                        source_images[scanner_index - 1] += stop - start
            print(
                f"[{slide_number}/{len(train_ids)}] FOV {fov} fold {fold} {slide_id}",
                flush=True,
            )

    target_power = statistics["radial_power"][train_indices, 0].sum(axis=0)
    target_images = int(statistics["radial_image_count"][train_indices, 0].sum())
    expected_images = len(train_ids) * 100
    if not (
        np.all(source_images == expected_images)
        and target_images == expected_images
        and np.isfinite(source_power).all()
        and np.isfinite(target_power).all()
        and np.all(source_power >= 0)
        and np.all(target_power >= 0)
    ):
        raise RuntimeError("E5-RF1 training-fold spectrum population is incomplete")

    temporary = output_root / f".fov_{fov}_fold_{fold}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        temporary,
        fov=np.asarray(fov, dtype=np.int64),
        fold=np.asarray(fold, dtype=np.int64),
        slide_ids=np.asarray(slide_ids),
        fold_assignment=np.asarray([assignments[value] for value in slide_ids], dtype=np.int8),
        train_slide_ids=np.asarray(train_ids),
        heldout_slide_ids=np.asarray(heldout_ids),
        scanners=np.asarray(SCANNERS),
        lab_mean=lab_mean,
        lab_std=lab_std,
        radial_frequency=statistics["radial_frequency"],
        post_reinhard_source_power=source_power,
        raw_at2_target_power=target_power,
        source_image_count=source_images,
        target_image_count=np.asarray(target_images, dtype=np.int64),
        base_clip_fraction_mean=base_clip_sum / source_images,
        base_clip_mae_mean=base_clip_mae_sum / source_images,
    )
    os.replace(temporary, output_path)
    summary = {
        "analysis": "e5_rf1_fold_statistics",
        "rf1_version": RF1_VERSION,
        "fov": fov,
        "fold": fold,
        "fold_counts": list(counts),
        "train_slides": len(train_ids),
        "heldout_slides": len(heldout_ids),
        "train_images_per_source_scanner": expected_images,
        "radial_bins": RADIAL_BINS,
        "grid_audit_sha256": audit_sha,
        "e5_statistics": str(source_path.resolve()),
        "e5_statistics_sha256": source_summary["output_sha256"],
        "execution_contract": str(contract_path.resolve()),
        "execution_contract_sha256": contract_sha,
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "statistics_gate_pass": True,
        "device": torch.cuda.get_device_name(0),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
