"""Build frozen all-slide HF sufficient statistics for exact LOSO E4 controls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from e4_control_population import (
    CONTROL_VERSION,
    FOVS,
    PYRAMID_LEVELS,
    centered_crop,
    decompose_scanner_tensor,
    slide_band_sums,
    uint8_to_analysis_tensor,
)
from fetch_e0_pfm_checkpoints import sha256
from prenorm.exp01.frequency import FixedLaplacianPyramid


def parse_args():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--fov", type=int)
    selection.add_argument("--fov-index", type=int)
    parser.add_argument(
        "--grid-audit", default="outputs/e0_native_aa_grid/audit/summary.json"
    )
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--output", default="outputs/e4_control_references")
    return parser.parse_args()


def selected_fov(fov: int | None, fov_index: int | None) -> int:
    if fov_index is not None:
        if fov_index < 0 or fov_index >= len(FOVS):
            raise IndexError(f"FOV index {fov_index} outside 0..{len(FOVS) - 1}")
        value = FOVS[fov_index]
    else:
        value = int(fov)
    if value not in FOVS:
        raise ValueError(f"FOV is outside the frozen E4 contract: {value}")
    return value


def existing_output_passes(path: Path, summary_path: Path, audit_sha256: str, fov: int):
    if not path.exists() or not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        return bool(
            summary.get("analysis") == "e4_global_hf_reference"
            and summary.get("control_version") == CONTROL_VERSION
            and summary.get("fov") == fov
            and summary.get("pyramid_levels") == PYRAMID_LEVELS
            and summary.get("grid_audit_sha256") == audit_sha256
            and summary.get("output_sha256") == sha256(path)
            and summary.get("reference_gate_pass") is True
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E4 global HF reference requires a visible CUDA device")
    fov = selected_fov(args.fov, args.fov_index)
    audit_path = Path(args.grid_audit)
    audit = json.loads(audit_path.read_text())
    if audit.get("grid_gate_pass") is not True or audit.get("patches_observed") != 65400:
        raise RuntimeError("native-AA population grid has not passed its frozen audit")
    audit_sha = sha256(audit_path)

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"fov_{fov}.npz"
    summary_path = output_root / f"fov_{fov}.summary.json"
    if existing_output_passes(output_path, summary_path, audit_sha, fov):
        print(summary_path.read_text(), end="")
        return

    grid_paths = sorted(Path(args.grid).glob("*.h5"))
    if len(grid_paths) != 109:
        raise ValueError(f"expected 109 native-AA shards, got {len(grid_paths)}")
    pyramid = FixedLaplacianPyramid(PYRAMID_LEVELS).eval().cuda()
    totals = None
    images_seen = 0
    slide_ids = []
    for index, grid_path in enumerate(grid_paths):
        with h5py.File(grid_path, "r") as source:
            if source["rgb"].shape != (6, 100, 512, 512, 3):
                raise ValueError(f"{grid_path}: invalid RGB shape")
            slide_id = str(source.attrs["slide_id"])
            if slide_id != grid_path.stem:
                raise ValueError(f"{grid_path}: slide ID/path mismatch")
            rgb = centered_crop(source["rgb"][:], fov)
        images = uint8_to_analysis_tensor(rgb, device="cuda")
        _, bands = decompose_scanner_tensor(pyramid, images)
        sums = slide_band_sums(bands)
        if totals is None:
            totals = [np.zeros_like(value, dtype=np.float64) for value in sums]
        for total, value in zip(totals, sums):
            total += value
        images_seen += int(images.shape[0] * images.shape[1])
        slide_ids.append(slide_id)
        del images, bands
        print(f"[{index + 1}/109] FOV {fov} {slide_id}", flush=True)

    if totals is None or len(totals) != PYRAMID_LEVELS or images_seen != 65400:
        raise RuntimeError("incomplete E4 global sufficient statistics")
    if not all(np.isfinite(value).all() for value in totals):
        raise RuntimeError("non-finite E4 global sufficient statistics")

    temporary = output_root / f".fov_{fov}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        temporary,
        **{f"band_sum_{index}": value for index, value in enumerate(totals)},
        image_count=np.asarray(images_seen, dtype=np.int64),
        slide_ids=np.asarray(slide_ids),
    )
    os.replace(temporary, output_path)
    summary = {
        "analysis": "e4_global_hf_reference",
        "control_version": CONTROL_VERSION,
        "fov": fov,
        "pyramid_levels": PYRAMID_LEVELS,
        "slides": len(slide_ids),
        "images": images_seen,
        "band_shapes": [list(value.shape) for value in totals],
        "grid_audit": str(audit_path.resolve()),
        "grid_audit_sha256": audit_sha,
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "reference_gate_pass": True,
        "device": torch.cuda.get_device_name(0),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

