"""Build compact all-slide sufficient statistics for exact LOSO E5 image methods."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from e5_comparator_population import (
    E5_VERSION,
    FOVS,
    RADIAL_BINS,
    SCANNERS,
    batch_radial_power,
    centered_crop,
    moments,
    od_affine_statistics,
    radial_geometry,
    rgb01_to_lab,
    rgb8_to_rgb01,
    sampled_pixels,
)
from fetch_e0_pfm_checkpoints import sha256


def parse_args():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--fov", type=int)
    selection.add_argument("--fov-index", type=int)
    parser.add_argument("--grid-audit", default="outputs/e0_native_aa_grid/audit/summary.json")
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--output", default="outputs/e5_image_statistics")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def selected_fov(fov: int | None, index: int | None):
    value = FOVS[index] if index is not None else int(fov)
    if value not in FOVS:
        raise ValueError(f"unsupported E5 FOV: {value}")
    return value


def existing_output_passes(path: Path, summary_path: Path, audit_sha: str, fov: int):
    if not path.exists() or not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        return bool(
            summary.get("analysis") == "e5_image_sufficient_statistics"
            and summary.get("e5_version") == E5_VERSION
            and summary.get("fov") == fov
            and summary.get("grid_audit_sha256") == audit_sha
            and summary.get("output_sha256") == sha256(path)
            and summary.get("statistics_gate_pass") is True
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E5 image sufficient statistics require a visible CUDA device")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    fov = selected_fov(args.fov, args.fov_index)
    audit_path = Path(args.grid_audit)
    audit = json.loads(audit_path.read_text())
    if audit.get("grid_gate_pass") is not True or audit.get("patches_observed") != 65_400:
        raise RuntimeError("native-AA population grid has not passed its frozen audit")
    audit_sha = sha256(audit_path)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"fov_{fov}.npz"
    summary_path = output_root / f"fov_{fov}.summary.json"
    if existing_output_passes(output_path, summary_path, audit_sha, fov):
        print(summary_path.read_text(), end="")
        return

    paths = sorted(Path(args.grid).glob("*.h5"))
    if len(paths) != 109:
        raise ValueError(f"expected 109 native-AA shards, got {len(paths)}")
    geometry = radial_geometry(fov)
    shape = (len(paths), len(SCANNERS))
    lab_sum = np.zeros((*shape, 3), dtype=np.float64)
    lab_square_sum = np.zeros_like(lab_sum)
    lab_count = np.zeros(shape, dtype=np.int64)
    od_xtx = np.zeros((len(paths), 5, 4, 4), dtype=np.float64)
    od_xty = np.zeros((len(paths), 5, 4, 3), dtype=np.float64)
    od_count = np.zeros((len(paths), 5), dtype=np.int64)
    radial_power = np.zeros((*shape, RADIAL_BINS), dtype=np.float64)
    radial_image_count = np.zeros(shape, dtype=np.int64)
    slide_ids = []

    for slide_index, path in enumerate(paths):
        with h5py.File(path, "r") as source:
            slide_id = str(source.attrs["slide_id"])
            scanners = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in source["scanner"][:]
            ]
            if slide_id != path.stem or scanners != list(SCANNERS):
                raise ValueError(f"{path}: slide/scanner identity mismatch")
            for start in range(0, 100, args.batch_size):
                stop = min(start + args.batch_size, 100)
                rgb8 = centered_crop(source["rgb"][:, start:stop], fov)
                rgb = rgb8_to_rgb01(rgb8, device="cuda")
                sample = sampled_pixels(rgb)
                lab = rgb01_to_lab(sample)
                for scanner_index in range(6):
                    total, square, count = moments(lab[scanner_index])
                    lab_sum[slide_index, scanner_index] += total.cpu().numpy()
                    lab_square_sum[slide_index, scanner_index] += square.cpu().numpy()
                    lab_count[slide_index, scanner_index] += count
                    radial_power[slide_index, scanner_index] += batch_radial_power(
                        rgb[scanner_index], geometry
                    ).cpu().numpy()
                    radial_image_count[slide_index, scanner_index] += stop - start
                for scanner_index in range(1, 6):
                    xtx, xty, count = od_affine_statistics(
                        sample[scanner_index], sample[0]
                    )
                    od_xtx[slide_index, scanner_index - 1] += xtx.cpu().numpy()
                    od_xty[slide_index, scanner_index - 1] += xty.cpu().numpy()
                    od_count[slide_index, scanner_index - 1] += count
                del rgb, sample, lab
        slide_ids.append(slide_id)
        print(f"[{slide_index + 1}/109] FOV {fov} {slide_id}", flush=True)

    expected_side = len(range(4, fov, 8))
    expected_sample_count = 100 * expected_side**2
    if not (
        np.all(lab_count == expected_sample_count)
        and np.all(od_count == expected_sample_count)
        and np.all(radial_image_count == 100)
        and np.isfinite(lab_sum).all()
        and np.isfinite(lab_square_sum).all()
        and np.isfinite(od_xtx).all()
        and np.isfinite(od_xty).all()
        and np.isfinite(radial_power).all()
        and np.all(radial_power >= 0)
    ):
        raise RuntimeError("E5 image sufficient-statistic population is incomplete")

    temporary = output_root / f".fov_{fov}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        temporary,
        slide_ids=np.asarray(slide_ids),
        scanners=np.asarray(SCANNERS),
        lab_sum=lab_sum,
        lab_square_sum=lab_square_sum,
        lab_count=lab_count,
        od_xtx=od_xtx,
        od_xty=od_xty,
        od_count=od_count,
        radial_power=radial_power,
        radial_image_count=radial_image_count,
        radial_frequency=geometry.frequency,
        radial_counts=geometry.counts,
    )
    os.replace(temporary, output_path)
    summary = {
        "analysis": "e5_image_sufficient_statistics",
        "e5_version": E5_VERSION,
        "fov": fov,
        "slides": len(slide_ids),
        "scanners": len(SCANNERS),
        "locations_per_scanner_slide": 100,
        "sampled_pixels_per_scanner_slide": expected_sample_count,
        "radial_bins": RADIAL_BINS,
        "grid_audit": str(audit_path.resolve()),
        "grid_audit_sha256": audit_sha,
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "statistics_gate_pass": True,
        "device": torch.cuda.get_device_name(0),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

