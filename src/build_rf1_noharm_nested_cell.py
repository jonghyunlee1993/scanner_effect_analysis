"""Build exact strict-nested RF1 image audits for one FOV/fold pair.

For unordered excluded folds {a,b}, the training population is the other three
folds.  The fitted Reinhard statistics and residual-frequency gains can then be
used for both directions: outer=a/inner=b and outer=b/inner=a.  One GPU task
therefore emits two directional inner-validation audits, reducing 60 naive
fits to 30 exact fits.

Raw AT2 power and Lab sufficient statistics are subtracted algebraically from
the locked per-slide E5 statistics.  Post-Reinhard source power must be
rerendered because the Lab transform, RGB clipping and spectrum are nonlinear.
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
from e5_reinhard_residual_frequency import (
    RF1_FOLDS,
    RF1_GAIN_CAP_CANDIDATES,
    RF1_VERSION,
    fitted_residual_gains,
    fold_assignments,
    fold_counts,
    shared_od_residual_frequency_many,
)
from fetch_e0_pfm_checkpoints import sha256


ANALYSIS = "rf1_noharm_strict_nested_inner_audit"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--grid-audit", default="outputs/e0_native_aa_grid/audit/summary.json")
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--e5-statistics", default="outputs/e5_image_statistics")
    parser.add_argument(
        "--output", default="outputs/rf1_improvement_pilot/noharm_nested/inner"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def unordered_fold_pairs() -> tuple[tuple[int, int], ...]:
    return tuple(itertools.combinations(range(RF1_FOLDS), 2))


def selected_nested_pair(task_index: int) -> tuple[int, tuple[int, int]]:
    pairs = unordered_fold_pairs()
    tasks = len(FOVS) * len(pairs)
    if not 0 <= int(task_index) < tasks:
        raise ValueError(f"nested task index must lie in [0,{tasks})")
    fov_index, pair_index = divmod(int(task_index), len(pairs))
    return int(FOVS[fov_index]), pairs[pair_index]


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


def existing_output_passes(
    path: Path,
    summary_path: Path,
    fov: int,
    outer_fold: int,
    inner_fold: int,
    source_sha: str,
):
    if not path.exists() or not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        return bool(
            summary.get("analysis") == ANALYSIS
            and summary.get("parent_rf1_version") == RF1_VERSION
            and summary.get("outcome_access") is False
            and summary.get("pfm_feature_access") is False
            and summary.get("strict_nested_training_exclusion") is True
            and summary.get("fov") == fov
            and summary.get("outer_fold") == outer_fold
            and summary.get("inner_validation_fold") == inner_fold
            and outer_fold not in summary.get("training_folds", [])
            and inner_fold not in summary.get("training_folds", [])
            and summary.get("e5_statistics_sha256") == source_sha
            and summary.get("output_sha256") == sha256(path)
            and summary.get("inner_audit_gate_pass") is True
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def save_direction(
    *,
    output_root: Path,
    fov: int,
    outer_fold: int,
    inner_fold: int,
    training_folds: list[int],
    train_ids: list[str],
    heldout_ids: list[str],
    caps: np.ndarray,
    lab_mean: np.ndarray,
    lab_std: np.ndarray,
    radial_frequency: np.ndarray,
    training_source_power: np.ndarray,
    training_target_power: np.ndarray,
    validation_target_power: np.ndarray,
    validation_base_power: np.ndarray,
    validation_output_power: np.ndarray,
    metrics: dict[str, np.ndarray],
    base_clip_fraction: np.ndarray,
    base_clip_mae: np.ndarray,
    source_path: Path,
    source_sha: str,
    grid_audit_sha: str,
    device: str,
):
    output_path = output_root / (
        f"fov_{fov}_outer_{outer_fold}_inner_{inner_fold}.npz"
    )
    summary_path = output_path.with_suffix(".summary.json")
    temporary = output_root / f".{output_path.stem}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        temporary,
        fov=np.asarray(fov, dtype=np.int64),
        outer_fold=np.asarray(outer_fold, dtype=np.int64),
        inner_validation_fold=np.asarray(inner_fold, dtype=np.int64),
        training_folds=np.asarray(training_folds, dtype=np.int8),
        train_slide_ids=np.asarray(train_ids),
        heldout_slide_ids=np.asarray(heldout_ids),
        scanners=np.asarray(SCANNERS[1:]),
        caps=caps,
        lab_mean=lab_mean,
        lab_std=lab_std,
        radial_frequency=radial_frequency,
        training_source_power=training_source_power,
        training_target_power=training_target_power,
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
        "parent_rf1_version": RF1_VERSION,
        "outcome_access": False,
        "pfm_feature_access": False,
        "strict_nested_training_exclusion": True,
        "algebraic_reuse": [
            "Lab sufficient statistics",
            "raw AT2 training power",
            "raw AT2 inner-validation power",
        ],
        "rerender_required": [
            "post-Reinhard training source power",
            "post-Reinhard inner-validation base and candidate outputs",
        ],
        "fov": fov,
        "outer_fold": outer_fold,
        "inner_validation_fold": inner_fold,
        "excluded_folds": sorted([outer_fold, inner_fold]),
        "training_folds": training_folds,
        "training_slides": len(train_ids),
        "inner_validation_slides": len(heldout_ids),
        "training_images_per_scanner": len(train_ids) * 100,
        "validation_images_per_scanner": len(heldout_ids) * 100,
        "candidate_caps": caps.tolist(),
        "radial_bins": RADIAL_BINS,
        "grid_audit_sha256": grid_audit_sha,
        "e5_statistics": str(source_path.resolve()),
        "e5_statistics_sha256": source_sha,
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "inner_audit_gate_pass": True,
        "device": device,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("strict nested RF1 audit requires a visible CUDA device")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    fov, excluded_pair = selected_nested_pair(args.task_index)
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

    directional = [(excluded_pair[0], excluded_pair[1]), (excluded_pair[1], excluded_pair[0])]
    if all(
        existing_output_passes(
            output_root / f"fov_{fov}_outer_{outer}_inner_{inner}.npz",
            output_root / f"fov_{fov}_outer_{outer}_inner_{inner}.summary.json",
            fov,
            outer,
            inner,
            source_sha,
        )
        for outer, inner in directional
    ):
        print(
            json.dumps(
                {
                    "analysis": "rf1_noharm_strict_nested_pair_cached",
                    "fov": fov,
                    "excluded_pair": list(excluded_pair),
                    "directional_outputs": len(directional),
                },
                indent=2,
            )
        )
        return

    slide_ids = [str(value) for value in statistics["slide_ids"]]
    assignments = fold_assignments(slide_ids)
    if fold_counts(assignments) != (22, 22, 22, 22, 21):
        raise RuntimeError("invalid RF1 fold population")
    training_folds = [fold for fold in range(RF1_FOLDS) if fold not in excluded_pair]
    train_ids = [value for value in slide_ids if assignments[value] in training_folds]
    train_indices = np.asarray([slide_ids.index(value) for value in train_ids], dtype=int)
    if any(assignments[value] in excluded_pair for value in train_ids):
        raise RuntimeError("strict nested training exclusion failed")

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
    training_source_power = np.zeros(
        (len(SCANNERS) - 1, RADIAL_BINS), dtype=np.float64
    )
    training_source_images = np.zeros(len(SCANNERS) - 1, dtype=np.int64)
    mean_tensor = torch.as_tensor(lab_mean, dtype=torch.float32, device="cuda")
    std_tensor = torch.as_tensor(lab_std, dtype=torch.float32, device="cuda")
    grid_root = Path(args.grid)
    with torch.inference_mode():
        for slide_number, slide_id in enumerate(train_ids, start=1):
            with h5py.File(grid_root / f"{slide_id}.h5", "r") as source:
                if str(source.attrs["slide_id"]) != slide_id:
                    raise ValueError(f"{slide_id}: slide identity mismatch")
                for scanner_index in range(1, len(SCANNERS)):
                    for start in range(0, 100, args.batch_size):
                        stop = min(start + args.batch_size, 100)
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
                        training_source_power[scanner_index - 1] += batch_radial_power(
                            base, geometry
                        ).cpu().numpy()
                        training_source_images[scanner_index - 1] += stop - start
            print(
                f"[train {slide_number}/{len(train_ids)}] FOV {fov} exclude {excluded_pair} {slide_id}",
                flush=True,
            )

    training_target_power = statistics["radial_power"][train_indices, 0].sum(axis=0)
    expected_training = len(train_ids) * 100
    if not (
        np.all(training_source_images == expected_training)
        and int(statistics["radial_image_count"][train_indices, 0].sum())
        == expected_training
        and np.isfinite(training_source_power).all()
        and np.isfinite(training_target_power).all()
    ):
        raise RuntimeError("strict nested training render is incomplete")

    caps = np.asarray(RF1_GAIN_CAP_CANDIDATES, dtype=np.float64)
    gains = np.stack(
        [
            fitted_residual_gains(
                training_source_power,
                training_target_power,
                geometry.frequency,
                float(cap),
            )
            for cap in caps
        ],
        axis=0,
    )
    metric_names = (
        "preproject_range_fraction",
        "material_range_fraction",
        "projection_fraction",
        "projection_rgb_mae",
        "final_clamp_mae",
    )
    emitted = []
    with torch.inference_mode():
        for outer_fold, inner_fold in directional:
            heldout_ids = [
                value for value in slide_ids if assignments[value] == inner_fold
            ]
            heldout_indices = np.asarray(
                [slide_ids.index(value) for value in heldout_ids], dtype=int
            )
            shape = (len(caps), len(heldout_ids), len(SCANNERS) - 1, 100)
            metrics = {
                name: np.empty(shape, dtype=np.float32) for name in metric_names
            }
            base_clip_fraction = np.empty(
                (len(heldout_ids), len(SCANNERS) - 1, 100), dtype=np.float32
            )
            base_clip_mae = np.empty_like(base_clip_fraction)
            validation_base_power = np.zeros(
                (len(SCANNERS) - 1, RADIAL_BINS), dtype=np.float64
            )
            validation_output_power = np.zeros(
                (len(caps), len(SCANNERS) - 1, RADIAL_BINS), dtype=np.float64
            )
            validation_source_images = np.zeros(len(SCANNERS) - 1, dtype=np.int64)

            for slide_index, slide_id in enumerate(heldout_ids):
                with h5py.File(grid_root / f"{slide_id}.h5", "r") as source:
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
                            validation_base_power[scanner_index - 1] += batch_radial_power(
                                base["output"], geometry
                            ).cpu().numpy()
                            report = shared_od_residual_frequency_many(
                                base["output"],
                                gains[:, scanner_index - 1],
                                geometry.frequency,
                            )
                            for name in metric_names:
                                metrics[name][
                                    :, slide_index, scanner_index - 1, start:stop
                                ] = report[name].cpu().numpy()
                            for cap_index in range(len(caps)):
                                validation_output_power[
                                    cap_index, scanner_index - 1
                                ] += batch_radial_power(
                                    report["output"][cap_index], geometry
                                ).cpu().numpy()
                            validation_source_images[scanner_index - 1] += stop - start
                print(
                    f"[validate {slide_index + 1}/{len(heldout_ids)}] FOV {fov} "
                    f"outer {outer_fold} inner {inner_fold} {slide_id}",
                    flush=True,
                )

            validation_target_power = statistics["radial_power"][
                heldout_indices, 0
            ].sum(axis=0)
            expected_validation = len(heldout_ids) * 100
            complete = bool(
                np.all(validation_source_images == expected_validation)
                and int(
                    statistics["radial_image_count"][heldout_indices, 0].sum()
                )
                == expected_validation
                and all(np.isfinite(value).all() for value in metrics.values())
                and np.isfinite(validation_base_power).all()
                and np.isfinite(validation_output_power).all()
            )
            if not complete:
                raise RuntimeError("strict nested validation render is incomplete")
            emitted.append(
                save_direction(
                    output_root=output_root,
                    fov=fov,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    training_folds=training_folds,
                    train_ids=train_ids,
                    heldout_ids=heldout_ids,
                    caps=caps,
                    lab_mean=lab_mean,
                    lab_std=lab_std,
                    radial_frequency=geometry.frequency,
                    training_source_power=training_source_power,
                    training_target_power=training_target_power,
                    validation_target_power=validation_target_power,
                    validation_base_power=validation_base_power,
                    validation_output_power=validation_output_power,
                    metrics=metrics,
                    base_clip_fraction=base_clip_fraction,
                    base_clip_mae=base_clip_mae,
                    source_path=source_path,
                    source_sha=source_sha,
                    grid_audit_sha=audit_sha,
                    device=torch.cuda.get_device_name(0),
                )
            )

    print(
        json.dumps(
            {
                "analysis": "rf1_noharm_strict_nested_pair_complete",
                "fov": fov,
                "excluded_pair": list(excluded_pair),
                "training_folds": training_folds,
                "training_slides": len(train_ids),
                "directional_outputs": [value["output"] for value in emitted],
                "strict_nested_training_exclusion": True,
                "outcome_access": False,
                "pfm_feature_access": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
