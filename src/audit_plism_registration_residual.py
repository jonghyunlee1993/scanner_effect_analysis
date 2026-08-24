"""Residual misregistration of one PLISM section, measured per location.

The similarity transform is fitted once per scanner at 16 um/px, so whatever it
leaves behind is a sub-thumbnail-pixel error that only shows up at full
resolution.  This measures it directly: read the AT2 patch and the scanner patch
at the locations the transform maps to each other, and phase-correlate them.

Two things are deliberately separated.

*Orientation.*  The transform maps coordinates, not pixels, so a scanner whose
slide sits rotated -- Philips, by 180 degrees in every section -- lands on the
right tissue with the patch content upside down.  Both orientations are scored
and the winner is recorded, so the audit diagnoses rather than assumes.

*Bias against jitter.*  The offsets are regressed on the location's position in
the reference frame.  A constant term is a residual translation; a position
dependent term is residual rotation or scale that the similarity fit did not
remove.

Resampling appears here only to bring two patches onto a common grid for the
correlation.  It never touches the spectral estimate.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE = "AT2"

from extract_plism_native_psd import PATCH_UM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", default="outputs/plism_section_registration")
    parser.add_argument("--task-index", type=int, required=True, help="stain index, 0-12")
    parser.add_argument("--wsi-dir", default="data/PLISM_dataset/original_wsi")
    parser.add_argument("--output", default="outputs/plism_registration_residual")
    parser.add_argument("--locations", type=int, default=80)
    return parser.parse_args()


def open_full(path: str):
    import pyvips

    image = pyvips.Image.new_from_file(path, access="random")
    if image.hasalpha():
        image = image.flatten(background=255)
    return image


def read_patch(image, centre_xy, mpp: float) -> np.ndarray | None:
    side = int(round(PATCH_UM / mpp))
    x = int(round(centre_xy[0])) - side // 2
    y = int(round(centre_xy[1])) - side // 2
    if x < 0 or y < 0 or x + side > image.width or y + side > image.height:
        return None
    patch = image.crop(x, y, side, side).numpy()
    return -np.log10(np.clip(patch.astype(np.float32), 1.0, 255.0) / 255.0).mean(axis=2)


def correlate(reference: np.ndarray, moving: np.ndarray, window: np.ndarray):
    """Sub-pixel shift aligning `moving` to `reference`, in reference pixels."""
    import cv2

    if moving.shape != reference.shape:
        moving = cv2.resize(moving, reference.shape[::-1], interpolation=cv2.INTER_AREA)
    a = np.ascontiguousarray(reference - reference.mean(), dtype=np.float64)
    b = np.ascontiguousarray(moving - moving.mean(), dtype=np.float64)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    return dx, dy, response


def main() -> None:
    args = parse_args()
    registration_dir = Path(args.registration)
    stains = sorted(p.stem for p in registration_dir.glob("*.json"))
    stain = stains[args.task_index]
    section = json.loads((registration_dir / f"{stain}.json").read_text())

    wsi_dir = Path(args.wsi_dir)
    thumb_um = float(section["thumb_um_per_px"])
    locations = np.array(section["locations_reference_thumb_xy"], dtype=np.float64)
    locations = locations[: args.locations]

    reference_meta = section["scanners"][REFERENCE]
    reference_image = open_full(str(wsi_dir / reference_meta["name"]))
    reference_mpp = reference_meta["mpp"]
    side = int(round(PATCH_UM / reference_mpp))
    window = np.outer(np.hanning(side), np.hanning(side))

    references = {}
    for index, (ux, uy) in enumerate(locations):
        centre = np.array([ux, uy]) * thumb_um / reference_mpp
        patch = read_patch(reference_image, centre, reference_mpp)
        if patch is not None:
            references[index] = patch

    rows = []
    for scanner, meta in section["scanners"].items():
        if scanner == REFERENCE:
            continue
        transform = section["transforms"][scanner]
        inverse = np.array(transform["inverse"], dtype=np.float64)
        image = open_full(str(wsi_dir / meta["name"]))
        for index, (ux, uy) in enumerate(locations):
            if index not in references:
                continue
            centre = (np.array([[ux, uy]]) @ inverse[:, :2].T + inverse[:, 2])[0]
            centre = centre * thumb_um / meta["mpp"]
            moving = read_patch(image, centre, meta["mpp"])
            if moving is None:
                continue
            upright = correlate(references[index], moving, window)
            flipped = correlate(references[index], np.rot90(moving, 2), window)
            best, orientation = (
                (upright, "upright") if upright[2] >= flipped[2] else (flipped, "rot180")
            )
            dx, dy, response = best
            rows.append(
                {
                    "stain": stain,
                    "scanner": scanner,
                    "location": index,
                    "ref_x": ux,
                    "ref_y": uy,
                    "dx_px": dx,
                    "dy_px": dy,
                    "offset_um": float(np.hypot(dx, dy) * reference_mpp),
                    "response": response,
                    "orientation": orientation,
                    "response_upright": upright[2],
                    "response_rot180": flipped[2],
                }
            )
        block = [r for r in rows if r["scanner"] == scanner]
        median = np.median([r["offset_um"] for r in block]) if block else float("nan")
        rot = sum(r["orientation"] == "rot180" for r in block)
        print(f"  {scanner:6s} n={len(block):4d} median offset={median:6.2f} um  rot180={rot}/{len(block)}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{stain}.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(f"{stain}: {len(rows)} pairs -> {destination}")


if __name__ == "__main__":
    main()
