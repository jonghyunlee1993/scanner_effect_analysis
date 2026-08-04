"""Render a side-by-side image gallery for each RF1U target.

For a deterministically chosen physical location the gallery shows the raw
source, the raw target acquisition of the same spot, three comparator
normalizations and RF1U, plus an amplified difference against Reinhard so the
reader can see what the band correction actually adds.

Every transform is fitted on the training folds of the slide being shown, so no
panel sees its own slide. Panels are written as PNG and also emitted as base64
so a single self-contained page can embed them.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import h5py
import numpy as np

from analyze_rf1_multiscale import shared_od_multiscale
from build_rf1m_cell import load_e5_statistics
from build_rf1m_slide_band_energy import fold_lab_statistics
from e5_comparator_population import (
    SCANNERS,
    centered_crop,
    od_affine_statistics,
    paired_od_affine,
    reinhard_lab,
    rgb01_to_od,
    rgb8_to_rgb01,
    solve_od_affine,
    uint8_from_rgb01,
)
from e5_macenko_supplement import macenko_normalize_batch, macenko_parameters
from e5_reinhard_residual_frequency import RF1_FOLDS, fold_assignments
from extract_rf1u_features import fold_gains, load_band_energy
from fetch_e0_pfm_checkpoints import sha256
from rf1m_combined import RF1M_SIGMAS
from rf1u_unpaired import RF1U_TARGETS, RF1U_VERSION, source_indices, target_index


GALLERY_FOV = 256
GALLERY_SALT = "e5_rf1u_gallery_v1:"
PANELS = (
    "raw_source",
    "raw_target",
    "reinhard",
    "macenko",
    "paired_od_affine",
    "rf1u",
    "rf1u_minus_reinhard_x5",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-audit", default="outputs/e0_native_aa_grid/audit/summary.json")
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--e5-statistics", default="outputs/e5_image_statistics")
    parser.add_argument("--energy", default="outputs/rf1u_multitarget/energy")
    parser.add_argument("--output", default="outputs/rf1u_multitarget/gallery")
    parser.add_argument("--sources-per-target", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def deterministic_choice(slide_ids: list[str], salt: str) -> tuple[str, int]:
    """Outcome-blind slide and location, fixed by a salted hash of the cohort."""
    import hashlib

    digest = hashlib.sha256((salt + "|".join(slide_ids)).encode()).digest()
    slide = slide_ids[int.from_bytes(digest[:4], "big") % len(slide_ids)]
    location = int.from_bytes(digest[4:8], "big") % 100
    return slide, location


def png_bytes(rgb8: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(rgb8).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("gallery rendering requires a CUDA device")
    audit = json.loads(Path(args.grid_audit).read_text())
    if audit.get("grid_gate_pass") is not True:
        raise RuntimeError("native-AA grid audit has not passed")
    audit_sha = sha256(Path(args.grid_audit))

    _, _, statistics = load_e5_statistics(Path(args.e5_statistics), GALLERY_FOV, audit_sha)
    slide_ids = [str(value) for value in statistics["slide_ids"]]
    assignments = fold_assignments(slide_ids)
    lab_mean, lab_std = fold_lab_statistics(statistics, slide_ids, assignments)
    slide_id, location = deterministic_choice(slide_ids, GALLERY_SALT)
    fold = assignments[slide_id]
    train_ids = [value for value in slide_ids if assignments[value] != fold]
    print(f"gallery slide={slide_id} location={location} heldout_fold={fold}", flush=True)

    output = Path(args.output)
    (output / "panels").mkdir(parents=True, exist_ok=True)
    grid_root = Path(args.grid)
    mean_tensor = torch.as_tensor(lab_mean[fold], dtype=torch.float32, device="cuda")
    std_tensor = torch.as_tensor(lab_std[fold], dtype=torch.float32, device="cuda")

    def crop(slide: str, scanner_index: int, start: int, stop: int):
        with h5py.File(grid_root / f"{slide}.h5", "r") as source:
            return rgb8_to_rgb01(
                centered_crop(source["rgb"][scanner_index, start:stop], GALLERY_FOV),
                device="cuda",
            )

    rows = []
    embedded = {}
    for target in RF1U_TARGETS:
        reference = target_index(target)
        sources = source_indices(target)
        energy_path, _, energy = load_band_energy(Path(args.energy), target, GALLERY_FOV)
        gains = fold_gains(energy, fold)
        order = np.argsort(-np.abs(gains["raw_log_gain"]).mean(axis=1))
        chosen = [int(index) for index in order[: args.sources_per_target]]

        with torch.inference_mode():
            target_patch = crop(slide_id, reference, location, location + 1)

            # Macenko: one stain matrix per training slide of the target scanner,
            # aggregated exactly as the locked supplement does.
            stain_list, maximum_list = [], []
            for train_slide in train_ids:
                batch = crop(train_slide, reference, 0, 100)
                sampled = rgb01_to_od(batch[:, 4::8, 4::8].reshape(-1, 3))
                parameters = macenko_parameters(sampled)
                if parameters is None:
                    continue
                stain_list.append(parameters[0].cpu().numpy())
                maximum_list.append(parameters[1].cpu().numpy())
            if len(stain_list) < 50:
                raise RuntimeError(f"{target}: too few valid Macenko references")
            target_stain = np.mean(np.stack(stain_list), axis=0)
            target_stain /= np.maximum(
                np.linalg.norm(target_stain, axis=0, keepdims=True), 1e-8
            )
            target_maximum = np.exp(
                np.log(np.maximum(np.stack(maximum_list), 1e-8)).mean(axis=0)
            )
            stain_tensor = torch.as_tensor(target_stain, dtype=torch.float32, device="cuda")
            maximum_tensor = torch.as_tensor(
                target_maximum, dtype=torch.float32, device="cuda"
            )

            # Paired OD affine: the design matrix is built from the source
            # scanner's own OD, so the normal equations accumulate per source.
            normal = {
                position: [np.zeros((4, 4)), np.zeros((4, 3))] for position in chosen
            }
            for train_slide in train_ids:
                for start in range(0, 100, args.batch_size):
                    stop = min(start + args.batch_size, 100)
                    reference_batch = crop(train_slide, reference, start, stop)
                    for position in chosen:
                        source_batch = crop(train_slide, sources[position], start, stop)
                        left, right, _ = od_affine_statistics(source_batch, reference_batch)
                        normal[position][0] += left.cpu().numpy()
                        normal[position][1] += right.cpu().numpy()

            for position in chosen:
                scanner_index = sources[position]
                scanner = SCANNERS[scanner_index]
                source_patch = crop(slide_id, scanner_index, location, location + 1)
                base = reinhard_lab(
                    source_patch,
                    mean_tensor[scanner_index].reshape(1, 1, 1, 3),
                    std_tensor[scanner_index].reshape(1, 1, 1, 3),
                    mean_tensor[reference].reshape(1, 1, 1, 3),
                    std_tensor[reference].reshape(1, 1, 1, 3),
                )["output"]
                ours = shared_od_multiscale(base, gains["gain"][position], RF1M_SIGMAS)["output"]
                macenko = macenko_normalize_batch(
                    source_patch, stain_tensor, maximum_tensor
                )["output"]
                coefficient = torch.as_tensor(
                    solve_od_affine(*normal[position]), dtype=torch.float32, device="cuda"
                )
                affine = paired_od_affine(source_patch, coefficient)["output"]
                residual = (0.5 + 5.0 * (ours - base)).clamp(0.0, 1.0)

                panels = {
                    "raw_source": source_patch,
                    "raw_target": target_patch,
                    "reinhard": base,
                    "macenko": macenko,
                    "paired_od_affine": affine,
                    "rf1u": ours,
                    "rf1u_minus_reinhard_x5": residual,
                }
                row = {
                    "target": target,
                    "source": scanner,
                    "slide_id": slide_id,
                    "location": location,
                    "heldout_fold": fold,
                    **{
                        f"gain_sigma{band + 1}": float(gains["gain"][position, band])
                        for band in range(len(RF1M_SIGMAS))
                    },
                    **{
                        f"alpha_sigma{band + 1}": float(gains["alpha"][position, band])
                        for band in range(len(RF1M_SIGMAS))
                    },
                }
                for name, image in panels.items():
                    array = uint8_from_rgb01(image)[0]
                    data = png_bytes(array)
                    key = f"{target}__{scanner}__{name}"
                    (output / "panels" / f"{key}.png").write_bytes(data)
                    embedded[key] = "data:image/png;base64," + base64.b64encode(data).decode()
                rows.append(row)
                print(f"  rendered {target} <- {scanner}", flush=True)


    (output / "gallery.json").write_text(
        json.dumps({"rows": rows, "panels": list(PANELS), "images": embedded}, indent=1) + "\n"
    )
    summary = {
        "analysis": "rf1u_gallery",
        "rf1u_version": RF1U_VERSION,
        "outcome_access": False,
        "slide_id": slide_id,
        "location": location,
        "heldout_fold": fold,
        "fov": GALLERY_FOV,
        "salt": GALLERY_SALT,
        "targets": list(RF1U_TARGETS),
        "panels": list(PANELS),
        "rows": len(rows),
        "training_slides": len(train_ids),
        "note": (
            "Every transform is fitted on the training folds only; the displayed "
            "slide is held out of all of them."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
