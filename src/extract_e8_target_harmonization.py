"""CORAL and orthogonal Procrustes aimed at a destination other than AT2.

Section 9 of the report shows that where an image correction aims decides
whether it helps or destroys, and that the governing quantity is the
destination's post-normalization detail power. The feature-space comparators
were never given that axis: they were fitted toward AT2 only. So the headline
contrast -- image correction aimed at GT450 against feature correction aimed at
AT2 -- crosses destinations, and the report's own central finding says that is
not a neutral difference.

This runs the same two transforms toward GT450 and S60 so the comparison closes.
It reuses the locked pieces wherever they are target-agnostic: per-scanner sums
and Grams from the E5 statistics, the exact leave-one-physical-slide-out
bookkeeping, and the frozen CORAL shrinkage. Only the cross-moment block is new,
and it comes from `build_e8_target_statistics.py`.

This runs on CPU. The locked AT2 harmonization required CUDA, but the GPU
partition is the scarce resource here and these are eigendecompositions of a few
thousand square, which CPU handles in seconds. The locked AT2 outputs are not
recomputed, so no locked number moves; the device is recorded in the summary.

Outputs follow the condition-registry layout of `docs/e8_condition_registry.md`,
`<root>/<encoder_id>/shards/<slide>.h5` with `condition` and `features`, so the
shared probe and frontier read them with no new code.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from build_e8_target_statistics import cross_to_target, source_order
from e5_comparator_population import (
    SCANNERS,
    centered_covariance,
    coral_transform,
    feature_sufficient_statistics,
    procrustes_transform,
    regularized_covariance,
    symmetric_matrix_power,
)
from fetch_e0_pfm_checkpoints import sha256


E8_VERSION = "e8_target_harmonization_v1"
CONDITIONS = ("coral", "orthogonal_procrustes")
TARGETS = ("gt450", "s60")
SLIDES = 109
LOSO_TRAIN_COUNT = 10_800


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task-index", type=int)
    selection.add_argument("--encoder-index", type=int)
    parser.add_argument("--target", choices=TARGETS)
    parser.add_argument("--slide-index", type=int)
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--statistics", default="outputs/e5_feature_statistics")
    parser.add_argument("--target-statistics", default="outputs/e8_target_statistics")
    parser.add_argument("--stability", default="outputs/e5_input_stability/summary.json")
    parser.add_argument("--output", default="outputs/e8_target_harmonization")
    return parser.parse_args()


def resolve_task(args):
    """Flatten (encoder, target, slide) so one array covers the whole grid."""
    if args.task_index is None:
        if args.target is None or args.slide_index is None:
            raise ValueError("--encoder-index needs --target and --slide-index")
        return args.encoder_index, args.target, args.slide_index
    total = 4 * len(TARGETS) * SLIDES
    if not 0 <= args.task_index < total:
        raise IndexError(f"task index {args.task_index} outside 0..{total - 1}")
    encoder_index, remainder = divmod(args.task_index, len(TARGETS) * SLIDES)
    target_index, slide_index = divmod(remainder, SLIDES)
    return encoder_index, TARGETS[target_index], slide_index


def load_statistics(path: Path, target_path: Path, model: dict, target: str):
    with h5py.File(path, "r") as source:
        if str(source.attrs["encoder_id"]) != model["encoder_id"]:
            raise ValueError(f"{path}: encoder mismatch")
        statistics = {
            "sum": np.asarray(source["sum"][:], dtype=np.float64),
            "gram": np.asarray(source["gram"][:], dtype=np.float64),
            "count": int(source.attrs["sample_count_per_scanner"]),
            "slide_ids": [value.decode() for value in source["slide_id"][:]],
        }
    with h5py.File(target_path, "r") as source:
        if str(source.attrs["encoder_id"]) != model["encoder_id"]:
            raise ValueError(f"{target_path}: encoder mismatch")
        key = f"cross_to_{target}"
        if key not in source:
            raise KeyError(f"{target_path}: missing {key}")
        statistics["cross"] = np.asarray(source[key][:], dtype=np.float64)
        statistics["source_order"] = [
            value.decode() for value in source[f"source_order_{target}"][:]
        ]
        if statistics["slide_ids"] != [value.decode() for value in source["slide_id"][:]]:
            raise ValueError("locked and target statistics disagree on slide order")
    return statistics


def transform_fold(raw: np.ndarray, statistics: dict, slide_id: str, target: str, shrinkage: float):
    target_index = SCANNERS.index(target)
    expected_sources = source_order(target_index)
    if statistics["source_order"] != expected_sources:
        raise ValueError(f"{target}: cross rows are {statistics['source_order']}")

    tensor = torch.from_numpy(raw)
    heldout_sum, heldout_gram, _, heldout_count = feature_sufficient_statistics(tensor)
    heldout_cross = cross_to_target(tensor, target_index)

    train_sum = torch.from_numpy(statistics["sum"]) - heldout_sum
    train_gram = torch.from_numpy(statistics["gram"]) - heldout_gram
    train_cross = torch.from_numpy(statistics["cross"]) - heldout_cross
    train_count = int(statistics["count"] - heldout_count)
    if train_count != LOSO_TRAIN_COUNT:
        raise ValueError(f"{slide_id}: LOSO train count is {train_count}")

    output = torch.empty(
        (len(CONDITIONS), len(SCANNERS), 100, raw.shape[-1]), dtype=torch.float32
    )
    # The destination scanner is its own reference and passes through untouched.
    output[:, target_index] = torch.from_numpy(raw[target_index]).float()

    for row, name in enumerate(expected_sources):
        scanner_index = SCANNERS.index(name)
        heldout_source = torch.from_numpy(raw[scanner_index]).double()
        coral = coral_transform(
            heldout_source,
            train_sum[scanner_index],
            train_gram[scanner_index],
            train_sum[target_index],
            train_gram[target_index],
            train_count,
            shrinkage,
        )
        procrustes = procrustes_transform(
            heldout_source,
            train_sum[scanner_index],
            train_sum[target_index],
            train_cross[row],
            train_count,
        )
        if not torch.isfinite(coral).all() or not torch.isfinite(procrustes).all():
            raise ValueError(f"{slide_id}/{name}->{target}: non-finite transform")
        output[0, scanner_index] = coral.float()
        output[1, scanner_index] = procrustes.float()
        print(f"{slide_id} {name}->{target} CORAL+Procrustes", flush=True)

    # Reported for the same reason the locked run reports it: a badly conditioned
    # target covariance is how CORAL turns into a numerically arbitrary map.
    eigenvalues = torch.linalg.eigvalsh(
        regularized_covariance(
            centered_covariance(
                train_sum[target_index], train_gram[target_index], train_count
            ),
            shrinkage,
        )
    )
    condition_number = float(eigenvalues.max() / eigenvalues.min().clamp_min(1e-30))
    return output.numpy(), train_count, condition_number


def main():
    args = parse_args()
    encoder_index, target, slide_index = resolve_task(args)
    contract = json.loads(Path(args.contract).read_text())
    model = contract["models"][encoder_index]
    model_id = model["encoder_id"]
    feature_dim = int(model["feature_dim"])

    stability = json.loads(Path(args.stability).read_text())
    if stability.get("stability_gate_pass") is not True:
        raise RuntimeError("E5 input-only stability manifest has not passed")
    # Reused, not reselected: it was chosen input-only on a calibration slide, so
    # carrying it over keeps the new destinations comparable to the locked one.
    shrinkage = float(stability["selected_coral_shrinkage"])

    paths = sorted((Path(args.raw) / model_id / "shards").glob("*.h5"))
    if len(paths) != SLIDES:
        raise ValueError(f"{model_id}: expected {SLIDES} raw shards, got {len(paths)}")
    path = paths[slide_index]
    slide_id = path.stem

    statistics = load_statistics(
        Path(args.statistics) / f"{model_id}.h5",
        Path(args.target_statistics) / f"{model_id}.h5",
        model,
        target,
    )

    with h5py.File(path, "r") as source:
        if str(source.attrs["slide_id"]) != slide_id:
            raise ValueError(f"{path}: slide identity mismatch")
        raw = np.asarray(source["features"][:], dtype=np.float32)
        location_id = np.asarray(source["location_id"][:])
        replicate_id = np.asarray(source["replicate_id"][:])
        center_x = np.asarray(source["canonical_center_x"][:])
        center_y = np.asarray(source["canonical_center_y"][:])
    if raw.shape != (len(SCANNERS), 100, feature_dim):
        raise ValueError(f"{model_id}/{slide_id}: unexpected raw feature shape")

    features, train_count, condition_number = transform_fold(
        raw, statistics, slide_id, target, shrinkage
    )

    output_root = Path(args.output) / target / model_id / "shards"
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{slide_id}.h5"
    temporary = output_root / f".{slide_id}.{os.getpid()}.tmp.h5"
    with h5py.File(temporary, "w") as sink:
        sink.create_dataset("features", data=features, compression="lzf", shuffle=True)
        sink.create_dataset(
            "condition", data=np.asarray(CONDITIONS, dtype=h5py.string_dtype("utf-8"))
        )
        sink.create_dataset(
            "scanner", data=np.asarray(SCANNERS, dtype=h5py.string_dtype("utf-8"))
        )
        sink.create_dataset("location_id", data=location_id)
        sink.create_dataset("replicate_id", data=replicate_id)
        sink.create_dataset("canonical_center_x", data=center_x)
        sink.create_dataset("canonical_center_y", data=center_y)
        sink.attrs["analysis"] = "e8_target_harmonization_shard"
        sink.attrs["e8_version"] = E8_VERSION
        sink.attrs["encoder_id"] = model_id
        sink.attrs["feature_dim"] = feature_dim
        sink.attrs["slide_id"] = slide_id
        sink.attrs["target"] = target
    temporary.replace(output_path)

    summary = {
        "analysis": "e8_target_harmonization_shard",
        "e8_version": E8_VERSION,
        "status": "POST_CORE_EXPLORATORY_EXTENSION",
        "note": (
            "Feature-space comparators aimed at a destination other than AT2, so "
            "the image-versus-feature contrast can be read within one "
            "destination. Does not enter the locked five-method ranking, and the "
            "locked AT2 harmonization is untouched."
        ),
        "encoder_id": model_id,
        "slide_id": slide_id,
        "target": target,
        "conditions": list(CONDITIONS),
        "coral_shrinkage": shrinkage,
        "coral_shrinkage_provenance": "reused from locked E5 input-only stability manifest",
        "target_covariance_condition": condition_number,
        "loso_train_count": train_count,
        "fit": "exact_leave_one_physical_slide_out",
        "device": "cpu",
        "output_sha256": sha256(output_path),
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
