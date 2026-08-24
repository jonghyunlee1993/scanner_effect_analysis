"""Build the E8 training cache: 256 px crops and residual paired alignments.

One array task per physical slide.  Each task writes only its own files, so
tasks never share a write target and no merge step is needed.

Two artifacts per slide:

- `crops/<slide_id>.npy`, `(6, 100, 256, 256, 3)` uint8 — the centred FOV-256
  crop of every scanner and location, which is exactly the image the frozen
  fov-256 RF1U statistics were fitted on.
- `align/<target>/<slide_id>.npy`, `(5, 100, 2)` int8 — the integer shift that
  best matches each source acquisition to that target's acquisition of the same
  location, in the fixed source order of `rf1u_unpaired.source_indices`.

The alignment is computed between raw source and raw target rather than between
the RF1U base and the target.  Reinhard is pointwise and the shared-OD band
rescaling uses zero-phase symmetric Gaussians, so neither can translate
structure; the shift is identical either way and this ordering keeps the cache
independent of the fold.

Image-only and outcome-blind: nothing here reads a PFM feature, an embedding or
a tissue label.  Frozen rules: `docs/e8_paired_residual_contract.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from e5_comparator_population import SCANNERS, centered_crop
from rf1u_unpaired import RF1U_TARGETS, source_indices, target_index


ANALYSIS = "e8_paired_residual_cache"
E8_CROP = 256
E8_MAX_SHIFT = 3
E8_VALID = E8_CROP - 2 * E8_MAX_SHIFT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument(
        "--statistics", default="outputs/e5_image_statistics/fov_256.npz"
    )
    parser.add_argument("--output", default="outputs/e8_residual/cache")
    return parser.parse_args()


def mean_optical_density(rgb8: np.ndarray) -> np.ndarray:
    """Channel-mean optical density of a uint8 batch, in float32.

    The formula matches `e5_comparator_population.rgb01_to_od` exactly so the
    alignment is measured in the same domain the losses are.
    """
    value = np.asarray(rgb8, dtype=np.float32)
    return (-np.log(np.maximum((value + 1.0) / 256.0, 1.0 / 256.0))).mean(axis=-1)


def best_integer_shift(
    source_od: np.ndarray,
    target_od: np.ndarray,
    max_shift: int = E8_MAX_SHIFT,
) -> np.ndarray:
    """Per-location integer shift minimising mean absolute mean-OD difference.

    `source_od` and `target_od` are `(N, C, C)` mean-OD stacks on the same grid.
    The returned shift `(ty, tx)` is defined so that

        source[max_shift + ty : max_shift + ty + valid, ...]

    aligns with `target[max_shift : max_shift + valid, ...]`, which is the exact
    convention the training loss applies.
    """
    if source_od.shape != target_od.shape or source_od.ndim != 3:
        raise ValueError("alignment needs matching NxCxC mean-OD stacks")
    size = source_od.shape[-1]
    valid = size - 2 * max_shift
    if valid < 1:
        raise ValueError("crop is too small for the requested shift range")
    anchor = target_od[:, max_shift : max_shift + valid, max_shift : max_shift + valid]
    best_cost = np.full(len(source_od), np.inf, dtype=np.float64)
    best = np.zeros((len(source_od), 2), dtype=np.int8)
    for ty in range(-max_shift, max_shift + 1):
        for tx in range(-max_shift, max_shift + 1):
            top = max_shift + ty
            left = max_shift + tx
            window = source_od[:, top : top + valid, left : left + valid]
            cost = np.abs(window - anchor).mean(axis=(1, 2)).astype(np.float64)
            better = cost < best_cost
            best_cost[better] = cost[better]
            best[better] = (ty, tx)
    return best, best_cost


def main():
    args = parse_args()
    with np.load(args.statistics) as statistics:
        slide_ids = [str(value) for value in statistics["slide_ids"]]
    if not 0 <= args.task_index < len(slide_ids):
        raise SystemExit(f"task index outside 0..{len(slide_ids) - 1}")
    slide_id = slide_ids[args.task_index]

    root = Path(args.output)
    crop_path = root / "crops" / f"{slide_id}.npy"
    crop_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(Path(args.grid) / f"{slide_id}.h5", "r") as grid:
        if grid["rgb"].shape != (6, 100, 512, 512, 3):
            raise RuntimeError(f"{slide_id}: unexpected grid shape")
        crops = centered_crop(np.asarray(grid["rgb"][:]), E8_CROP)
    if crops.shape != (6, 100, E8_CROP, E8_CROP, 3) or crops.dtype != np.uint8:
        raise RuntimeError(f"{slide_id}: invalid crop stack")
    temporary = crop_path.with_suffix(".tmp.npy")
    np.save(temporary, crops)
    temporary.replace(crop_path)

    density = np.stack([mean_optical_density(crops[index]) for index in range(6)])
    summary = {
        "analysis": ANALYSIS,
        "slide_id": slide_id,
        "task_index": args.task_index,
        "crop": E8_CROP,
        "max_shift": E8_MAX_SHIFT,
        "valid": E8_VALID,
        "scanners": list(SCANNERS),
        "pfm_feature_access": False,
        "outcome_access": False,
        "targets": {},
    }
    for target in RF1U_TARGETS:
        sources = source_indices(target)
        anchor = density[target_index(target)]
        shifts = np.zeros((len(sources), 100, 2), dtype=np.int8)
        costs = np.zeros((len(sources), 100), dtype=np.float64)
        for position, scanner_index in enumerate(sources):
            shifts[position], costs[position] = best_integer_shift(
                density[scanner_index], anchor
            )
        align_path = root / "align" / target / f"{slide_id}.npy"
        align_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = align_path.with_suffix(".tmp.npy")
        np.save(temporary, shifts)
        temporary.replace(align_path)
        boundary = np.abs(shifts).max(axis=-1) == E8_MAX_SHIFT
        summary["targets"][target] = {
            "sources": [SCANNERS[index] for index in sources],
            "shift_absolute_mean": float(np.abs(shifts).mean()),
            "shift_boundary_fraction": float(boundary.mean()),
            "residual_cost_median": float(np.median(costs)),
        }
    (root / "summary").mkdir(parents=True, exist_ok=True)
    (root / "summary" / f"{slide_id}.json").write_text(json.dumps(summary, indent=1))
    print(f"{slide_id}: cached {crops.shape} and {len(RF1U_TARGETS)} alignments")


if __name__ == "__main__":
    main()
