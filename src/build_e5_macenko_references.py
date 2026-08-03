"""Estimate per-slide AT2 Macenko references for exact LOSO Supplement fitting."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from e5_comparator_population import E5_VERSION, FOVS, centered_crop, rgb01_to_od, rgb8_to_rgb01
from e5_macenko_supplement import ALPHA, BETA, MACENKO_VERSION, macenko_parameters
from fetch_e0_pfm_checkpoints import sha256


def parse_args():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--fov", type=int)
    selection.add_argument("--fov-index", type=int)
    parser.add_argument("--grid-audit", default="outputs/e0_native_aa_grid/audit/summary.json")
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--output", default="outputs/e5_macenko_references")
    return parser.parse_args()


def selected_fov(fov, index):
    value = FOVS[index] if index is not None else int(fov)
    if value not in FOVS:
        raise ValueError(f"unsupported Macenko FOV: {value}")
    return value


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Macenko reference estimation requires a visible CUDA device")
    fov = selected_fov(args.fov, args.fov_index)
    audit_path = Path(args.grid_audit)
    audit = json.loads(audit_path.read_text())
    if audit.get("grid_gate_pass") is not True or audit.get("patches_observed") != 65_400:
        raise RuntimeError("native-AA grid audit has not passed")
    paths = sorted(Path(args.grid).glob("*.h5"))
    if len(paths) != 109:
        raise ValueError(f"expected 109 grids, got {len(paths)}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    output_path = output / f"fov_{fov}.npz"
    summary_path = output / f"fov_{fov}.summary.json"
    stains = np.full((109, 3, 2), np.nan, dtype=np.float32)
    maximum = np.full((109, 2), np.nan, dtype=np.float32)
    eligible = np.zeros(109, dtype=np.int64)
    slide_ids = []
    for index, path in enumerate(paths):
        with h5py.File(path, "r") as source:
            slide_id = str(source.attrs["slide_id"])
            rgb8 = centered_crop(source["rgb"][0], fov)
        rgb = rgb8_to_rgb01(rgb8, device="cuda")
        sampled = rgb[:, 4::8, 4::8].reshape(-1, 3)
        od = rgb01_to_od(sampled)
        eligible[index] = int((od.sum(dim=1) > BETA).sum().cpu())
        parameters = macenko_parameters(od)
        if parameters is None:
            raise RuntimeError(f"{slide_id}: invalid AT2 Macenko reference")
        stains[index] = parameters[0].cpu().numpy()
        maximum[index] = parameters[1].cpu().numpy()
        slide_ids.append(slide_id)
        print(f"[{index + 1}/109] FOV {fov} {slide_id}", flush=True)
    if not np.isfinite(stains).all() or not np.isfinite(maximum).all() or np.any(eligible < 50):
        raise RuntimeError("incomplete Macenko reference population")
    temporary = output / f".fov_{fov}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        temporary,
        slide_ids=np.asarray(slide_ids),
        stains=stains,
        maximum=maximum,
        eligible_pixels=eligible,
    )
    os.replace(temporary, output_path)
    summary = {
        "analysis": "e5_macenko_reference_population",
        "e5_version": E5_VERSION,
        "macenko_version": MACENKO_VERSION,
        "fov": fov,
        "slides": len(slide_ids),
        "beta": BETA,
        "alpha_percent": ALPHA,
        "eligible_pixels_min": int(eligible.min()),
        "eligible_pixels_max": int(eligible.max()),
        "grid_audit_sha256": sha256(audit_path),
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "reference_gate_pass": True,
        "device": torch.cuda.get_device_name(0),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

