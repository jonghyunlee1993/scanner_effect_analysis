"""Cross-moment statistics for feature harmonization toward a non-AT2 target.

The locked E5 feature harmonization is AT2-only. That is not a stated design
choice; it follows from the statistics file, which stores `cross_to_at2` and
nothing else, and from `SCANNERS[0]` being read as the target throughout. So the
report compares an image correction aimed at GT450 against a feature correction
aimed at AT2, and the destination result of section 9 says that difference is
not innocent.

Retargeting CORAL needs nothing new: it reads per-scanner `sum` and `gram`, both
already target-agnostic. Orthogonal Procrustes needs the paired cross-product
between each source scanner and the target, which exists only for AT2. This
script computes the missing `cross_to_<target>` blocks from the frozen raw
features and writes them beside the existing statistics.

Everything else is deliberately reused rather than recomputed: `sum`, `gram` and
the LOSO bookkeeping come from the locked file, so the only new quantity is the
cross term. The frozen CORAL shrinkage is likewise reused, not reselected -- it
was chosen input-only on a calibration slide, so carrying it over keeps the new
destinations comparable to the locked one instead of giving them a second fit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from e5_comparator_population import SCANNERS
from fetch_e0_pfm_checkpoints import sha256


E8_VERSION = "e8_target_statistics_v1"
NEW_TARGETS = ("gt450", "s60")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--encoder-index", type=int, required=True)
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--targets", default=",".join(NEW_TARGETS))
    parser.add_argument("--output", default="outputs/e8_target_statistics")
    return parser.parse_args()


def cross_to_target(features: torch.Tensor, target_index: int) -> torch.Tensor:
    """Source-to-target cross-products for the five non-target scanners.

    Row order follows the scanners with the target removed, kept ascending, so
    it is reproducible from `SCANNERS` and the target name alone.
    """
    if features.ndim != 3 or features.shape[0] != len(SCANNERS):
        raise ValueError(f"expected 6xNxD features, got {tuple(features.shape)}")
    value = features.double()
    sources = [index for index in range(len(SCANNERS)) if index != target_index]
    return torch.stack([value[index].T @ value[target_index] for index in sources], dim=0)


def source_order(target_index: int) -> list[str]:
    return [name for index, name in enumerate(SCANNERS) if index != target_index]


def main():
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text())
    model = contract["models"][args.encoder_index]
    model_id = model["encoder_id"]
    feature_dim = int(model["feature_dim"])
    targets = [name.strip() for name in args.targets.split(",") if name.strip()]
    for target in targets:
        if target not in SCANNERS:
            raise ValueError(f"unknown target {target!r}")

    paths = sorted((Path(args.raw) / model_id / "shards").glob("*.h5"))
    if len(paths) != 109:
        raise ValueError(f"{model_id}: expected 109 raw feature shards, got {len(paths)}")

    accumulators = {target: None for target in targets}
    count = 0
    slide_ids = []
    for index, path in enumerate(paths):
        with h5py.File(path, "r") as source:
            slide_id = str(source.attrs["slide_id"])
            values = np.asarray(source["features"][:], dtype=np.float32)
        if slide_id != path.stem or values.shape != (len(SCANNERS), 100, feature_dim):
            raise ValueError(f"{model_id}/{path.stem}: raw identity or shape mismatch")
        tensor = torch.from_numpy(values)
        for target in targets:
            local = cross_to_target(tensor, SCANNERS.index(target))
            if accumulators[target] is None:
                accumulators[target] = torch.zeros_like(local)
            accumulators[target] += local
        count += values.shape[1]
        slide_ids.append(slide_id)
        del values, tensor
        print(f"[{index + 1}/109] {model_id} {slide_id}", flush=True)

    if count != 10_900:
        raise RuntimeError(f"{model_id}: accumulated {count} locations, expected 10900")
    for target, value in accumulators.items():
        if value is None or not torch.isfinite(value).all():
            raise RuntimeError(f"{model_id}/{target}: non-finite cross statistics")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{model_id}.h5"
    temporary = output_root / f".{model_id}.{os.getpid()}.tmp.h5"
    with h5py.File(temporary, "w") as sink:
        for target, value in accumulators.items():
            sink.create_dataset(
                f"cross_to_{target}",
                data=value.numpy(),
                compression="lzf",
                shuffle=True,
            )
            sink.create_dataset(
                f"source_order_{target}",
                data=np.asarray(
                    source_order(SCANNERS.index(target)),
                    dtype=h5py.string_dtype("utf-8"),
                ),
            )
        sink.create_dataset(
            "slide_id", data=np.asarray(slide_ids, dtype=h5py.string_dtype("utf-8"))
        )
        sink.attrs["analysis"] = "e8_target_sufficient_statistics"
        sink.attrs["e8_version"] = E8_VERSION
        sink.attrs["encoder_id"] = model_id
        sink.attrs["feature_dim"] = feature_dim
        sink.attrs["sample_count_per_scanner"] = count
        sink.attrs["targets"] = ",".join(targets)
    temporary.replace(output_path)

    summary = {
        "analysis": "e8_target_sufficient_statistics",
        "e8_version": E8_VERSION,
        "status": "POST_CORE_EXPLORATORY_EXTENSION",
        "note": (
            "Adds source-to-target cross-products for destinations other than "
            "AT2. Per-scanner sums and Grams are target-agnostic and are read "
            "from the locked E5 statistics rather than recomputed. This does not "
            "enter the locked five-method comparator ranking."
        ),
        "encoder_id": model_id,
        "feature_dim": feature_dim,
        "targets": targets,
        "source_order": {
            target: source_order(SCANNERS.index(target)) for target in targets
        },
        "sample_count_per_scanner": count,
        "slides": len(slide_ids),
        "device": "cpu",
        "output_sha256": sha256(output_path),
    }
    (output_root / f"{model_id}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
