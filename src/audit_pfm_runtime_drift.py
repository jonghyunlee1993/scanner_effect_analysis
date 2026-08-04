"""Measure whether the current runtime still reproduces the locked raw PFM features.

The frozen PFM contract pins `torch`, `torchvision`, `timm`, `huggingface_hub`
and `conch`, and `verify_runtime_contract` refuses to extract when the installed
versions differ.  That guard is correct, but it is a version comparison, not a
measurement: it cannot say whether the drift actually changes an embedding.

This audit answers that empirically.  It re-encodes one slide's audited crops
with the encoders as currently installed and compares against the stored raw
features, reporting maximum and mean absolute deviation, the worst cosine
similarity and the relative L2 error.  A near-identity result means the pinned
population can still be used as the comparison baseline; a material difference
means any new condition must be extracted together with a fresh raw baseline.

Encoders that cannot be constructed at all, such as `conch_v1` without the
`conch` package, are reported as unavailable rather than skipped silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from extract_e0_pfm_features import BATCH_SIZE, centered_crop, transformed_batch
from fetch_e0_pfm_checkpoints import sha256
from smoke_e0_pfm_encoders import encoder_kwargs, package_version


COSINE_IDENTITY_TOLERANCE = 1e-4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--slide-indices", type=int, nargs="+", default=[0, 54, 108])
    parser.add_argument(
        "--trident-root", default="/mnt/isilon/oldridge_lab/irinaz/github/env/trident"
    )
    parser.add_argument("--output", default="outputs/rf1u_multitarget/runtime_drift")
    return parser.parse_args()


def observed_runtime(expected: dict) -> dict:
    return {name: package_version(name) for name in expected}


def compare(stored: np.ndarray, current: np.ndarray) -> dict:
    if stored.shape != current.shape:
        raise ValueError("stored and re-encoded feature shapes differ")
    difference = np.abs(current - stored)
    unit_stored = stored / np.linalg.norm(stored, axis=-1, keepdims=True)
    unit_current = current / np.linalg.norm(current, axis=-1, keepdims=True)
    cosine = (unit_stored * unit_current).sum(axis=-1)
    return {
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "min_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
        "relative_l2": float(
            np.linalg.norm(current - stored) / np.linalg.norm(stored)
        ),
        "reproduces_locked_features": bool(
            cosine.min() >= 1.0 - COSINE_IDENTITY_TOLERANCE
        ),
    }


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("runtime drift audit requires a CUDA device")
    contract = json.loads(Path(args.contract).read_text())
    expected = contract["runtime_distributions"]
    observed = observed_runtime(expected)

    sys.path.insert(0, str(Path(args.trident_root).resolve()))
    from trident.patch_encoder_models.load import encoder_factory

    grid_paths = sorted(Path(args.grid).glob("*.h5"))
    selected = [grid_paths[index] for index in args.slide_indices]
    rows = []
    for model in contract["models"]:
        encoder_id = model["encoder_id"]
        try:
            encoder = encoder_factory(
                encoder_id, **encoder_kwargs(encoder_id, model["checkpoint_path"])
            ).eval().cuda()
            encoder = (
                encoder.half() if encoder.precision == torch.float16 else encoder.float()
            )
        except Exception as error:  # noqa: BLE001 - report, do not mask
            rows.append(
                {
                    "encoder_id": encoder_id,
                    "available": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"{encoder_id}: unavailable", flush=True)
            continue

        fov = int(model["native_fov_px"])
        feature_dim = int(model["feature_dim"])
        batch_size = BATCH_SIZE[encoder_id]
        for grid_path in selected:
            raw_path = Path(args.raw) / encoder_id / "shards" / grid_path.name
            with h5py.File(raw_path, "r") as source:
                stored = np.asarray(source["features"][:], dtype=np.float32)
            current = np.empty_like(stored)
            with h5py.File(grid_path, "r") as grid, torch.inference_mode():
                for scanner_index in range(stored.shape[0]):
                    for start in range(0, 100, batch_size):
                        stop = min(start + batch_size, 100)
                        rgb = centered_crop(grid["rgb"][scanner_index, start:stop], fov)
                        batch = transformed_batch(rgb, encoder.eval_transforms).cuda(
                            non_blocking=True
                        )
                        batch = (
                            batch.half()
                            if encoder.precision == torch.float16
                            else batch.float()
                        )
                        current[scanner_index, start:stop] = (
                            encoder(batch).float().cpu().numpy()
                        )
            rows.append(
                {
                    "encoder_id": encoder_id,
                    "slide_id": grid_path.stem,
                    "available": True,
                    "feature_dim": feature_dim,
                    "native_fov_px": fov,
                    "raw_feature_sha256": sha256(raw_path),
                    **compare(stored, current),
                }
            )
            print(f"{encoder_id} {grid_path.stem}: {rows[-1]['max_abs_difference']:.3e}", flush=True)
        del encoder
        torch.cuda.empty_cache()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    available = [row for row in rows if row.get("available")]
    summary = {
        "analysis": "pfm_runtime_drift_audit",
        "slide_ids": [path.stem for path in selected],
        "features_compared": int(sum(6 * 100 for _ in available)),
        "contract_runtime": expected,
        "observed_runtime": observed,
        "runtime_matches_contract": observed == expected,
        "encoders_available": sorted({row["encoder_id"] for row in available}),
        "bit_identical_cells": int(
            sum(row["max_abs_difference"] == 0.0 for row in available)
        ),
        "cells": len(available),
        "encoders_unavailable": sorted(
            {row["encoder_id"] for row in rows if not row.get("available")}
        ),
        "all_available_reproduce_locked_features": bool(
            available and all(row["reproduces_locked_features"] for row in available)
        ),
        "worst_min_cosine": (
            min(row["min_cosine"] for row in available) if available else None
        ),
        "cosine_identity_tolerance": COSINE_IDENTITY_TOLERANCE,
        "encoders": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "encoders"}, indent=2))


if __name__ == "__main__":
    main()
