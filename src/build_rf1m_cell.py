"""Build one strict-nested RF1M image-only cell for a FOV and fold split.

Two stages share this builder because they differ only in which folds are
excluded from fitting:

* ``outer``: exclude fold ``h``, fit on the other four, render fold ``h``.  This
  is the honest held-out evaluation of whichever cap the inner stage selects.
* ``inner``: exclude the unordered pair ``{a,b}``, fit on the other three, render
  both ``a`` and ``b``.  One fit therefore serves both selection directions,
  turning 60 naive fits into 30 tasks.

Reinhard Lab sufficient statistics and raw AT2 radial power are reused
algebraically from the locked per-slide E5 statistics.  Laplacian band energy is
accumulated natively on training patches, as the RF1M contract requires, so raw
AT2 training patches are rendered here even though their radial power is not.

No PFM feature or endpoint is read.
"""

from __future__ import annotations

import argparse
import itertools
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
from e5_reinhard_residual_frequency import RF1_FOLDS, fold_assignments, fold_counts
from fetch_e0_pfm_checkpoints import sha256
from rf1m_combined import (
    RF1M_CAP_CANDIDATES,
    RF1M_SIGMAS,
    RF1M_VERSION,
    accumulate_band_energy,
    candidate_band_gains,
    shared_od_multiscale_many,
)


ANALYSIS = "rf1m_strict_nested_cell"
IDENTITY_TOLERANCE = 1e-6
METRIC_NAMES = (
    "preproject_range_fraction",
    "material_range_fraction",
    "projection_fraction",
    "projection_rgb_mae",
    "final_clamp_mae",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("outer", "inner"), required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument(
        "--grid-audit", default="outputs/e0_native_aa_grid/audit/summary.json"
    )
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--e5-statistics", default="outputs/e5_image_statistics")
    parser.add_argument("--output", default="outputs/rf1m_candidate/cells")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def unordered_fold_pairs() -> tuple[tuple[int, int], ...]:
    return tuple(itertools.combinations(range(RF1_FOLDS), 2))


def selected_split(stage: str, task_index: int) -> tuple[int, tuple[int, ...]]:
    """Map a task index to its FOV and the folds excluded from fitting."""
    if stage == "outer":
        groups: tuple[tuple[int, ...], ...] = tuple((fold,) for fold in range(RF1_FOLDS))
    elif stage == "inner":
        groups = unordered_fold_pairs()
    else:
        raise ValueError(f"unknown RF1M stage: {stage}")
    tasks = len(FOVS) * len(groups)
    if not 0 <= int(task_index) < tasks:
        raise ValueError(f"{stage} task index must lie in [0,{tasks})")
    fov_index, group_index = divmod(int(task_index), len(groups))
    return int(FOVS[fov_index]), groups[group_index]


def directional_pairs(stage: str, excluded: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Return the (outer fold, validation fold) pairs a cell must emit."""
    if stage == "outer":
        (fold,) = excluded
        return ((fold, fold),)
    first, second = excluded
    return ((first, second), (second, first))


def cell_name(fov: int, stage: str, outer_fold: int, validation_fold: int) -> str:
    if stage == "outer":
        return f"fov_{fov}_outer_{outer_fold}"
    return f"fov_{fov}_outer_{outer_fold}_inner_{validation_fold}"


def load_e5_statistics(root: Path, fov: int, grid_audit_sha: str):
    path = root / f"fov_{fov}.npz"
    summary = json.loads((root / f"fov_{fov}.summary.json").read_text())
    if not (
        summary.get("analysis") == "e5_image_sufficient_statistics"
        and summary.get("fov") == fov
        and summary.get("grid_audit_sha256") == grid_audit_sha
        and summary.get("output_sha256") == sha256(path)
        and summary.get("statistics_gate_pass") is True
    ):
        raise RuntimeError(f"invalid locked E5 image statistics: {path}")
    with np.load(path) as source:
        return path, summary, {name: source[name] for name in source.files}


def existing_output_passes(
    path: Path, fov: int, stage: str, outer_fold: int, validation_fold: int, source_sha: str
) -> bool:
    summary_path = path.with_suffix(".summary.json")
    if not path.exists() or not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    training_folds = summary.get("training_folds", [])
    return bool(
        summary.get("analysis") == ANALYSIS
        and summary.get("rf1m_version") == RF1M_VERSION
        and summary.get("outcome_access") is False
        and summary.get("pfm_feature_access") is False
        and summary.get("stage") == stage
        and summary.get("fov") == fov
        and summary.get("outer_fold") == outer_fold
        and summary.get("validation_fold") == validation_fold
        and outer_fold not in training_folds
        and validation_fold not in training_folds
        and summary.get("e5_statistics_sha256") == source_sha
        and summary.get("output_sha256") == sha256(path)
        and summary.get("cell_gate_pass") is True
    )


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("RF1M cell rendering requires a visible CUDA device")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    fov, excluded = selected_split(args.stage, args.task_index)
    directional = directional_pairs(args.stage, excluded)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    audit_path = Path(args.grid_audit)
    audit = json.loads(audit_path.read_text())
    if audit.get("grid_gate_pass") is not True or audit.get("patches_observed") != 65_400:
        raise RuntimeError("native-AA grid audit has not passed")
    audit_sha = sha256(audit_path)
    source_path, source_summary, statistics = load_e5_statistics(
        Path(args.e5_statistics), fov, audit_sha
    )
    source_sha = source_summary["output_sha256"]

    if all(
        existing_output_passes(
            output_root / f"{cell_name(fov, args.stage, outer, validation)}.npz",
            fov,
            args.stage,
            outer,
            validation,
            source_sha,
        )
        for outer, validation in directional
    ):
        print(
            json.dumps(
                {
                    "analysis": "rf1m_cell_cached",
                    "stage": args.stage,
                    "fov": fov,
                    "excluded_folds": list(excluded),
                    "outputs": len(directional),
                },
                indent=2,
            )
        )
        return

    slide_ids = [str(value) for value in statistics["slide_ids"]]
    assignments = fold_assignments(slide_ids)
    if fold_counts(assignments) != (22, 22, 22, 22, 21):
        raise RuntimeError("invalid RF1 fold population")
    training_folds = [fold for fold in range(RF1_FOLDS) if fold not in excluded]
    train_ids = [value for value in slide_ids if assignments[value] in training_folds]
    train_indices = np.asarray([slide_ids.index(value) for value in train_ids], dtype=int)
    if any(assignments[value] in excluded for value in train_ids):
        raise RuntimeError("RF1M training exclusion failed")

    lab_sum = statistics["lab_sum"][train_indices].sum(axis=0)
    lab_square = statistics["lab_square_sum"][train_indices].sum(axis=0)
    lab_count = statistics["lab_count"][train_indices].sum(axis=0)
    lab_mean = np.empty((len(SCANNERS), 3), dtype=np.float64)
    lab_std = np.empty_like(lab_mean)
    for scanner_index in range(len(SCANNERS)):
        lab_mean[scanner_index], lab_std[scanner_index] = mean_std(
            lab_sum[scanner_index],
            lab_square[scanner_index],
            int(lab_count[scanner_index]),
        )

    geometry = radial_geometry(fov)
    mean_tensor = torch.as_tensor(lab_mean, dtype=torch.float32, device="cuda")
    std_tensor = torch.as_tensor(lab_std, dtype=torch.float32, device="cuda")
    grid_root = Path(args.grid)

    n_sources = len(SCANNERS) - 1
    n_bands = len(RF1M_SIGMAS)
    source_energy = np.zeros((n_sources, n_bands), dtype=np.float64)
    target_energy = np.zeros(n_bands, dtype=np.float64)
    source_patches = np.zeros(n_sources, dtype=np.int64)
    target_patches = 0
    with torch.inference_mode():
        for slide_number, slide_id in enumerate(train_ids, start=1):
            with h5py.File(grid_root / f"{slide_id}.h5", "r") as source:
                if str(source.attrs["slide_id"]) != slide_id:
                    raise ValueError(f"{slide_id}: slide identity mismatch")
                for start in range(0, 100, args.batch_size):
                    stop = min(start + args.batch_size, 100)
                    target = rgb8_to_rgb01(
                        centered_crop(source["rgb"][0, start:stop], fov), device="cuda"
                    )
                    energy, count = accumulate_band_energy(target)
                    target_energy += energy
                    target_patches += count
                    for scanner_index in range(1, len(SCANNERS)):
                        rgb = rgb8_to_rgb01(
                            centered_crop(source["rgb"][scanner_index, start:stop], fov),
                            device="cuda",
                        )
                        base = reinhard_lab(
                            rgb,
                            mean_tensor[scanner_index],
                            std_tensor[scanner_index],
                            mean_tensor[0],
                            std_tensor[0],
                        )["output"]
                        energy, count = accumulate_band_energy(base)
                        source_energy[scanner_index - 1] += energy
                        source_patches[scanner_index - 1] += count
            print(
                f"[train {slide_number}/{len(train_ids)}] RF1M {args.stage} FOV {fov} "
                f"exclude {list(excluded)} {slide_id}",
                flush=True,
            )

    expected_training = len(train_ids) * 100
    if not (
        target_patches == expected_training
        and np.all(source_patches == expected_training)
        and np.isfinite(source_energy).all()
        and np.isfinite(target_energy).all()
        and np.all(source_energy > 0)
        and np.all(target_energy > 0)
    ):
        raise RuntimeError("RF1M training band-energy accumulation is incomplete")

    caps = np.asarray(RF1M_CAP_CANDIDATES, dtype=np.float64)
    gains = candidate_band_gains(source_energy, target_energy, caps)
    if not np.allclose(gains[0], 1.0):
        raise RuntimeError("the identity candidate must have unit band gains")

    emitted = []
    with torch.inference_mode():
        for outer_fold, validation_fold in directional:
            heldout_ids = [
                value for value in slide_ids if assignments[value] == validation_fold
            ]
            heldout_indices = np.asarray(
                [slide_ids.index(value) for value in heldout_ids], dtype=int
            )
            shape = (len(caps), len(heldout_ids), n_sources, 100)
            metrics = {name: np.empty(shape, dtype=np.float32) for name in METRIC_NAMES}
            base_clip_fraction = np.empty(
                (len(heldout_ids), n_sources, 100), dtype=np.float32
            )
            base_clip_mae = np.empty_like(base_clip_fraction)
            validation_base_power = np.zeros((n_sources, RADIAL_BINS), dtype=np.float64)
            validation_output_power = np.zeros(
                (len(caps), n_sources, RADIAL_BINS), dtype=np.float64
            )
            validation_patches = np.zeros(n_sources, dtype=np.int64)
            identity_deviation = 0.0

            for slide_index, slide_id in enumerate(heldout_ids):
                with h5py.File(grid_root / f"{slide_id}.h5", "r") as source:
                    if str(source.attrs["slide_id"]) != slide_id:
                        raise ValueError(f"{slide_id}: slide identity mismatch")
                    for scanner_index in range(1, len(SCANNERS)):
                        for start in range(0, 100, args.batch_size):
                            stop = min(start + args.batch_size, 100)
                            rgb = rgb8_to_rgb01(
                                centered_crop(
                                    source["rgb"][scanner_index, start:stop], fov
                                ),
                                device="cuda",
                            )
                            base = reinhard_lab(
                                rgb,
                                mean_tensor[scanner_index],
                                std_tensor[scanner_index],
                                mean_tensor[0],
                                std_tensor[0],
                            )
                            base_clip_fraction[
                                slide_index, scanner_index - 1, start:stop
                            ] = base["preclip_range_fraction"].cpu().numpy()
                            base_clip_mae[
                                slide_index, scanner_index - 1, start:stop
                            ] = base["clip_pixel_mae"].cpu().numpy()
                            validation_base_power[scanner_index - 1] += (
                                batch_radial_power(base["output"], geometry)
                                .cpu()
                                .numpy()
                            )
                            report = shared_od_multiscale_many(
                                base["output"], gains[:, scanner_index - 1]
                            )
                            identity_deviation = max(
                                identity_deviation,
                                float(
                                    (report["output"][0] - base["output"])
                                    .abs()
                                    .max()
                                    .cpu()
                                ),
                            )
                            for name in METRIC_NAMES:
                                metrics[name][
                                    :, slide_index, scanner_index - 1, start:stop
                                ] = report[name].cpu().numpy()
                            for candidate_index in range(len(caps)):
                                validation_output_power[
                                    candidate_index, scanner_index - 1
                                ] += (
                                    batch_radial_power(
                                        report["output"][candidate_index], geometry
                                    )
                                    .cpu()
                                    .numpy()
                                )
                            validation_patches[scanner_index - 1] += stop - start
                print(
                    f"[validate {slide_index + 1}/{len(heldout_ids)}] RF1M {args.stage} "
                    f"FOV {fov} outer {outer_fold} validation {validation_fold} {slide_id}",
                    flush=True,
                )

            validation_target_power = statistics["radial_power"][heldout_indices, 0].sum(
                axis=0
            )
            expected_validation = len(heldout_ids) * 100
            if not (
                np.all(validation_patches == expected_validation)
                and int(statistics["radial_image_count"][heldout_indices, 0].sum())
                == expected_validation
                and all(np.isfinite(value).all() for value in metrics.values())
                and np.isfinite(validation_base_power).all()
                and np.isfinite(validation_output_power).all()
            ):
                raise RuntimeError("RF1M validation render is incomplete")
            if identity_deviation > IDENTITY_TOLERANCE:
                raise RuntimeError(
                    f"identity candidate deviates from the Reinhard base by "
                    f"{identity_deviation:.3e}"
                )

            name = cell_name(fov, args.stage, outer_fold, validation_fold)
            output_path = output_root / f"{name}.npz"
            temporary = output_root / f".{name}.{os.getpid()}.tmp.npz"
            np.savez_compressed(
                temporary,
                rf1m_version=np.asarray(RF1M_VERSION),
                stage=np.asarray(args.stage),
                fov=np.asarray(fov, dtype=np.int64),
                outer_fold=np.asarray(outer_fold, dtype=np.int64),
                validation_fold=np.asarray(validation_fold, dtype=np.int64),
                training_folds=np.asarray(training_folds, dtype=np.int8),
                train_slide_ids=np.asarray(train_ids),
                heldout_slide_ids=np.asarray(heldout_ids),
                scanners=np.asarray(SCANNERS[1:]),
                sigmas=np.asarray(RF1M_SIGMAS, dtype=np.float64),
                caps=caps,
                gains=gains,
                training_source_band_energy=source_energy,
                training_target_band_energy=target_energy,
                training_patches_per_scanner=np.asarray(
                    expected_training, dtype=np.int64
                ),
                lab_mean=lab_mean,
                lab_std=lab_std,
                radial_frequency=geometry.frequency,
                validation_target_power=validation_target_power,
                validation_base_power=validation_base_power,
                validation_output_power=validation_output_power,
                base_clip_fraction=base_clip_fraction,
                base_clip_mae=base_clip_mae,
                **metrics,
            )
            os.replace(temporary, output_path)
            summary = {
                "analysis": ANALYSIS,
                "rf1m_version": RF1M_VERSION,
                "outcome_access": False,
                "pfm_feature_access": False,
                "stage": args.stage,
                "strict_nested_training_exclusion": True,
                "algebraic_reuse": [
                    "Lab sufficient statistics",
                    "raw AT2 validation radial power",
                ],
                "native_gpu_accumulation": [
                    "post-Reinhard training source Laplacian band energy",
                    "raw AT2 training Laplacian band energy",
                ],
                "nonlinear_gpu_rerender": [
                    "post-Reinhard validation base",
                    "validation candidate outputs",
                ],
                "fov": fov,
                "outer_fold": outer_fold,
                "validation_fold": validation_fold,
                "excluded_folds": sorted(excluded),
                "training_folds": training_folds,
                "training_slides": len(train_ids),
                "validation_slides": len(heldout_ids),
                "training_images_per_scanner": expected_training,
                "validation_images_per_scanner": expected_validation,
                "pyramid_sigmas_pixels": list(RF1M_SIGMAS),
                "candidate_caps": caps.tolist(),
                "radial_bins": RADIAL_BINS,
                "identity_max_abs_output_minus_base": identity_deviation,
                "grid_audit_sha256": audit_sha,
                "e5_statistics": str(source_path.resolve()),
                "e5_statistics_sha256": source_sha,
                "output": str(output_path.resolve()),
                "output_sha256": sha256(output_path),
                "device": torch.cuda.get_device_name(0),
                "cell_gate_pass": True,
            }
            output_path.with_suffix(".summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            emitted.append(summary["output"])

    print(
        json.dumps(
            {
                "analysis": "rf1m_cell_complete",
                "rf1m_version": RF1M_VERSION,
                "stage": args.stage,
                "fov": fov,
                "excluded_folds": list(excluded),
                "training_folds": training_folds,
                "training_slides": len(train_ids),
                "outputs": emitted,
                "outcome_access": False,
                "pfm_feature_access": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
