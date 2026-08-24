"""How much of the feature-space correction is first moment, and how much is not.

The control probe settles that no frequency-domain image intervention removes
the scanner: the paired oracle, which substitutes the true registered
high-frequency content of another scanner, moves a linear probe by 0.03, and
deleting the high band outright -- which destroys the representation -- leaves
it at 0.81 or above. Feature-space correction moves the same probe by 0.87.

The explanation this study now leads with is that the signature is written
redundantly across image statistics, so an image correction can only match the
handful of moments it explicitly models while a feature-space affine matches the
whole second-moment structure. That sentence is only worth making if the ladder
is measured rather than asserted, so this fits three rungs toward one target and
scores them the same way:

    mean_shift  per-scanner mean only, first moment
    diag_scale  per-scanner mean and per-dimension standard deviation
    coral       full covariance whitening and recolouring

If mean_shift already recovers most of CORAL's gain, the scanner is close to a
translation in feature space and the claim should say so. If the rungs separate,
the claim that the full second-moment structure is what matters is earned.

The same machinery answers the composition question. Given RF1U-corrected
features as input instead of raw, it asks whether an image correction leaves
anything for the feature correction to still remove -- the direct test of
whether image-space work is subsumed or complementary.

Fits are exact leave-one-physical-slide-out, matching the locked comparators,
and outputs follow the condition-registry layout so the shared probe reads them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from e5_comparator_population import (
    SCANNERS,
    centered_covariance,
    regularized_covariance,
    symmetric_matrix_power,
)
from fetch_e0_pfm_checkpoints import sha256


E8_VERSION = "e8_moment_ladder_v1"
RUNGS = ("mean_shift", "diag_scale", "coral")
SLIDES = 109
LOSO_TRAIN_COUNT = 10_800


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--encoder-index", type=int, required=True)
    parser.add_argument(
        "--input",
        default="outputs/e0_pfm_features",
        help="feature root; raw features or an already-corrected condition",
    )
    parser.add_argument(
        "--input-condition",
        default="",
        help="condition to select when the input shards carry a condition axis",
    )
    parser.add_argument("--target", default="at2", choices=SCANNERS)
    parser.add_argument("--stability", default="outputs/e5_input_stability/summary.json")
    parser.add_argument("--label", required=True, help="name for this arm on disk")
    parser.add_argument("--output", default="outputs/e8_moment_ladder")
    return parser.parse_args()


def load_features(root: Path, model_id: str, condition: str, feature_dim: int):
    """Return (slide_ids, features) with features as slides x 6 x 100 x D."""
    paths = sorted((root / model_id / "shards").glob("*.h5"))
    if len(paths) != SLIDES:
        raise ValueError(f"{root}/{model_id}: expected {SLIDES} shards, got {len(paths)}")
    slide_ids, blocks = [], []
    for path in paths:
        with h5py.File(path, "r") as source:
            values = np.asarray(source["features"][:], dtype=np.float32)
            names = (
                [value.decode() for value in source["condition"][:]]
                if "condition" in source
                else []
            )
        if names:
            if not condition:
                raise ValueError(f"{path} has a condition axis; pass --input-condition")
            matches = [index for index, name in enumerate(names) if name == condition]
            if len(matches) != 1:
                raise ValueError(f"{path}: condition {condition!r} not unique in {names}")
            values = values[matches[0]]
        elif condition:
            raise ValueError(f"{path} has no condition axis but {condition!r} was requested")
        if values.shape != (len(SCANNERS), 100, feature_dim):
            raise ValueError(f"{path}: unexpected shape {values.shape}")
        slide_ids.append(path.stem)
        blocks.append(values)
    return slide_ids, np.stack(blocks)


def population_statistics(features: np.ndarray):
    """Per-scanner sums and Grams over all slides, kept in float64."""
    tensor = torch.from_numpy(features).double()
    flat = tensor.permute(1, 0, 2, 3).reshape(len(SCANNERS), -1, features.shape[-1])
    sums = flat.sum(dim=1)
    grams = torch.stack([block.T @ block for block in flat], dim=0)
    return sums, grams, int(flat.shape[1])


def slide_statistics(block: np.ndarray):
    tensor = torch.from_numpy(block).double()
    sums = tensor.sum(dim=1)
    grams = torch.stack([scanner.T @ scanner for scanner in tensor], dim=0)
    return sums, grams, int(tensor.shape[1])


def apply_rungs(
    heldout: torch.Tensor,
    source_sum: torch.Tensor,
    source_gram: torch.Tensor,
    target_sum: torch.Tensor,
    target_gram: torch.Tensor,
    train_count: int,
    shrinkage: float,
):
    """Three nested corrections sharing one set of training moments."""
    source_mean = source_sum / float(train_count)
    target_mean = target_sum / float(train_count)
    centered = heldout.double() - source_mean

    source_cov = centered_covariance(source_sum, source_gram, train_count)
    target_cov = centered_covariance(target_sum, target_gram, train_count)

    mean_shift = centered + target_mean

    source_scale = source_cov.diagonal().clamp_min(1e-12).sqrt()
    target_scale = target_cov.diagonal().clamp_min(1e-12).sqrt()
    diag_scale = centered * (target_scale / source_scale) + target_mean

    whitening = symmetric_matrix_power(
        regularized_covariance(source_cov, shrinkage), -0.5
    )
    recoloring = symmetric_matrix_power(
        regularized_covariance(target_cov, shrinkage), 0.5
    )
    coral = centered @ whitening @ recoloring + target_mean
    return {"mean_shift": mean_shift, "diag_scale": diag_scale, "coral": coral}


def main():
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text())
    model = contract["models"][args.encoder_index]
    model_id = model["encoder_id"]
    feature_dim = int(model["feature_dim"])
    target_index = SCANNERS.index(args.target)

    stability = json.loads(Path(args.stability).read_text())
    if stability.get("stability_gate_pass") is not True:
        raise RuntimeError("E5 input-only stability manifest has not passed")
    shrinkage = float(stability["selected_coral_shrinkage"])

    slide_ids, features = load_features(
        Path(args.input), model_id, args.input_condition, feature_dim
    )
    total_sum, total_gram, total_count = population_statistics(features)
    if total_count != 10_900:
        raise ValueError(f"{model_id}: population count {total_count}, expected 10900")

    output_root = Path(args.output) / args.label / model_id / "shards"
    output_root.mkdir(parents=True, exist_ok=True)
    written = []
    for index, slide_id in enumerate(slide_ids):
        block = features[index]
        heldout_sum, heldout_gram, heldout_count = slide_statistics(block)
        train_sum = total_sum - heldout_sum
        train_gram = total_gram - heldout_gram
        train_count = total_count - heldout_count
        if train_count != LOSO_TRAIN_COUNT:
            raise ValueError(f"{slide_id}: LOSO train count is {train_count}")

        corrected = torch.empty(
            (len(RUNGS), len(SCANNERS), 100, feature_dim), dtype=torch.float32
        )
        # The destination is its own reference on every rung.
        corrected[:, target_index] = torch.from_numpy(block[target_index]).float()
        for scanner_index in range(len(SCANNERS)):
            if scanner_index == target_index:
                continue
            rungs = apply_rungs(
                torch.from_numpy(block[scanner_index]),
                train_sum[scanner_index],
                train_gram[scanner_index],
                train_sum[target_index],
                train_gram[target_index],
                train_count,
                shrinkage,
            )
            for rung_index, rung in enumerate(RUNGS):
                value = rungs[rung]
                if not torch.isfinite(value).all():
                    raise ValueError(f"{slide_id}/{SCANNERS[scanner_index]}/{rung}: non-finite")
                corrected[rung_index, scanner_index] = value.float()

        path = output_root / f"{slide_id}.h5"
        temporary = output_root / f".{slide_id}.{os.getpid()}.tmp.h5"
        with h5py.File(temporary, "w") as sink:
            sink.create_dataset(
                "features", data=corrected.numpy(), compression="lzf", shuffle=True
            )
            sink.create_dataset(
                "condition", data=np.asarray(RUNGS, dtype=h5py.string_dtype("utf-8"))
            )
            sink.create_dataset(
                "scanner", data=np.asarray(SCANNERS, dtype=h5py.string_dtype("utf-8"))
            )
            sink.attrs["analysis"] = "e8_moment_ladder_shard"
            sink.attrs["e8_version"] = E8_VERSION
            sink.attrs["encoder_id"] = model_id
            sink.attrs["slide_id"] = slide_id
            sink.attrs["target"] = args.target
            sink.attrs["input_root"] = str(args.input)
            sink.attrs["input_condition"] = args.input_condition
        temporary.replace(path)
        written.append(path)
        print(f"[{index + 1}/{SLIDES}] {model_id} {slide_id} {args.label}", flush=True)

    summary = {
        "analysis": "e8_moment_ladder",
        "e8_version": E8_VERSION,
        "status": "POST_CORE_EXPLORATORY_EXTENSION",
        "note": (
            "Nested feature-space corrections sharing one set of LOSO training "
            "moments, so the rungs differ only in how much of the second-moment "
            "structure they match. Does not enter the locked five-method ranking."
        ),
        "encoder_id": model_id,
        "label": args.label,
        "input_root": str(args.input),
        "input_condition": args.input_condition,
        "target": args.target,
        "rungs": list(RUNGS),
        "coral_shrinkage": shrinkage,
        "coral_shrinkage_provenance": "reused from locked E5 input-only stability manifest",
        "fit": "exact_leave_one_physical_slide_out",
        "loso_train_count": LOSO_TRAIN_COUNT,
        "slides": len(written),
        "device": "cpu",
        "shard_sha256": {path.name: sha256(path) for path in written[:3]},
    }
    (Path(args.output) / args.label / model_id / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
