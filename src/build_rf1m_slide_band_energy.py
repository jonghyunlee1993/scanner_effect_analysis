"""Accumulate per-slide Laplacian band energy toward a chosen target scanner.

Originally written for the AT2 slide-adaptive test; `--target` generalizes it to
the RF1U multi-target design, where the Reinhard step maps each source scanner to
the chosen target and the raw target patches supply the target band energy.

The RF1M cells store band energy summed over all training slides, which is
exactly the quantity whose pooling the slide-adaptive proposal questions.  This
pass keeps one band-energy triple per slide instead, for the raw AT2 target and
for every source scanner under each fold's Reinhard transform.

It renders no candidates and reads no PFM feature.  Output is small: one array of
shape (slides, scanners, folds, bands) per FOV.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from build_rf1m_cell import load_e5_statistics
from e5_comparator_population import (
    FOVS,
    SCANNERS,
    centered_crop,
    mean_std,
    reinhard_lab,
    rgb8_to_rgb01,
)
from e5_reinhard_residual_frequency import RF1_FOLDS, fold_assignments, fold_counts
from fetch_e0_pfm_checkpoints import sha256
from rf1m_combined import RF1M_SIGMAS, RF1M_VERSION, accumulate_band_energy
from rf1u_unpaired import RF1U_TARGETS, source_indices, target_index


ANALYSIS = "rf1m_slide_band_energy"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument(
        "--grid-audit", default="outputs/e0_native_aa_grid/audit/summary.json"
    )
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--e5-statistics", default="outputs/e5_image_statistics")
    parser.add_argument("--target", choices=RF1U_TARGETS, default="at2")
    parser.add_argument("--output", default="outputs/rf1m_slide_adaptive/energy")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def fold_lab_statistics(statistics: dict, slide_ids: list[str], assignments: dict):
    """Reinhard Lab mean/SD for every fold, fitted on that fold's complement."""
    lab_mean = np.empty((RF1_FOLDS, len(SCANNERS), 3), dtype=np.float64)
    lab_std = np.empty_like(lab_mean)
    for fold in range(RF1_FOLDS):
        train = np.asarray(
            [index for index, value in enumerate(slide_ids) if assignments[value] != fold],
            dtype=int,
        )
        total = statistics["lab_sum"][train].sum(axis=0)
        square = statistics["lab_square_sum"][train].sum(axis=0)
        count = statistics["lab_count"][train].sum(axis=0)
        for scanner_index in range(len(SCANNERS)):
            lab_mean[fold, scanner_index], lab_std[fold, scanner_index] = mean_std(
                total[scanner_index], square[scanner_index], int(count[scanner_index])
            )
    return lab_mean, lab_std


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("slide band-energy accumulation requires a CUDA device")
    if not 0 <= args.task_index < len(FOVS):
        raise ValueError(f"task index must lie in [0,{len(FOVS)})")
    fov = int(FOVS[args.task_index])
    target = target_index(args.target)
    sources = source_indices(args.target)

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
    if fold_counts(assignments) != (22, 22, 22, 22, 21):
        raise RuntimeError("invalid RF1 fold population")
    lab_mean, lab_std = fold_lab_statistics(statistics, slide_ids, assignments)
    mean_tensor = torch.as_tensor(lab_mean, dtype=torch.float32, device="cuda")
    std_tensor = torch.as_tensor(lab_std, dtype=torch.float32, device="cuda")

    n_bands = len(RF1M_SIGMAS)
    n_sources = len(sources)
    target_energy = np.zeros((len(slide_ids), n_bands), dtype=np.float64)
    source_energy = np.zeros(
        (len(slide_ids), n_sources, RF1_FOLDS, n_bands), dtype=np.float64
    )
    patches = np.zeros(len(slide_ids), dtype=np.int64)

    grid_root = Path(args.grid)
    with torch.inference_mode():
        for slide_index, slide_id in enumerate(slide_ids):
            with h5py.File(grid_root / f"{slide_id}.h5", "r") as source:
                if str(source.attrs["slide_id"]) != slide_id:
                    raise ValueError(f"{slide_id}: slide identity mismatch")
                for start in range(0, 100, args.batch_size):
                    stop = min(start + args.batch_size, 100)
                    reference = rgb8_to_rgb01(
                        centered_crop(source["rgb"][target, start:stop], fov), device="cuda"
                    )
                    energy, count = accumulate_band_energy(reference)
                    target_energy[slide_index] += energy
                    patches[slide_index] += count
                    for position, scanner_index in enumerate(sources):
                        rgb = rgb8_to_rgb01(
                            centered_crop(source["rgb"][scanner_index, start:stop], fov),
                            device="cuda",
                        )
                        for fold in range(RF1_FOLDS):
                            base = reinhard_lab(
                                rgb,
                                mean_tensor[fold, scanner_index],
                                std_tensor[fold, scanner_index],
                                mean_tensor[fold, target],
                                std_tensor[fold, target],
                            )["output"]
                            energy, _ = accumulate_band_energy(base)
                            source_energy[slide_index, position, fold] += energy
            print(
                f"[{slide_index + 1}/{len(slide_ids)}] target {args.target} "
                f"FOV {fov} {slide_id}",
                flush=True,
            )

    if not (
        np.all(patches == 100)
        and np.all(target_energy > 0)
        and np.all(source_energy > 0)
        and np.isfinite(target_energy).all()
        and np.isfinite(source_energy).all()
    ):
        raise RuntimeError("slide band-energy accumulation is incomplete")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{args.target}_fov_{fov}.npz"
    temporary = output_root / f".{args.target}_fov_{fov}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        temporary,
        rf1m_version=np.asarray(RF1M_VERSION),
        fov=np.asarray(fov, dtype=np.int64),
        target=np.asarray(args.target),
        target_index=np.asarray(target, dtype=np.int64),
        source_indices=np.asarray(sources, dtype=np.int64),
        slide_ids=np.asarray(slide_ids),
        fold_of_slide=np.asarray([assignments[value] for value in slide_ids], dtype=np.int8),
        scanners=np.asarray([SCANNERS[index] for index in sources]),
        sigmas=np.asarray(RF1M_SIGMAS, dtype=np.float64),
        patches_per_slide=patches,
        target_band_energy=target_energy,
        source_band_energy=source_energy,
    )
    os.replace(temporary, output_path)
    summary = {
        "analysis": ANALYSIS,
        "rf1m_version": RF1M_VERSION,
        "outcome_access": False,
        "pfm_feature_access": False,
        "fov": fov,
        "target": args.target,
        "slides": len(slide_ids),
        "source_scanners": [SCANNERS[index] for index in sources],
        "folds": RF1_FOLDS,
        "pyramid_sigmas_pixels": list(RF1M_SIGMAS),
        "target_energy": "raw target Laplacian band energy per slide",
        "source": "post-Reinhard band energy per slide, scanner and fold",
        "patches_per_slide": 100,
        "grid_audit_sha256": audit_sha,
        "e5_statistics_sha256": source_summary["output_sha256"],
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "device": torch.cuda.get_device_name(0),
        "energy_gate_pass": True,
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
