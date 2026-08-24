"""Which transform model the PLISM scanners actually need.

The similarity fit has four degrees of freedom: translation, rotation, one
isotropic scale.  The residual audit showed that removing a global linear model
from its errors drops the spread from 5-9 um to 2-4 um for every scanner, which
says the leftover is systematic and a richer model should absorb it.

This fits three models per scanner and measures the residual each one leaves, at
full resolution, by phase correlation against the reference patch:

    similarity   4 DOF   translation, rotation, isotropic scale
    affine       6 DOF   adds anisotropic scale and shear
    homography   8 DOF   adds perspective

The decision is made on measured residual, not on argument.  Every model is used
the same way -- to choose where to read -- and analysed pixels are still taken by
integer crop at level 0 with no interpolation.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_plism_section_registration import REFERENCE, THUMB_UM, thumbnail
from extract_plism_native_psd import PATCH_UM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/PLISM_dataset/manifest_original.csv")
    parser.add_argument("--registration", default="outputs/plism_section_registration")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--wsi-dir", default="data/PLISM_dataset/original_wsi")
    parser.add_argument("--output", default="outputs/plism_registration_models")
    parser.add_argument("--locations", type=int, default=60)
    return parser.parse_args()


def match_points(moving: np.ndarray, fixed: np.ndarray):
    import cv2

    detector = cv2.AKAZE_create()
    kp_move, desc_move = detector.detectAndCompute(moving, None)
    kp_fix, desc_fix = detector.detectAndCompute(fixed, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_move, desc_fix, k=2)
    good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
    source = np.float32([kp_move[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    target = np.float32([kp_fix[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    return source, target


def fit_models(source, target) -> dict:
    import cv2

    models = {}
    similarity, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if similarity is not None:
        models["similarity"] = (similarity, int(inliers.sum()), "affine")
    affine, inliers = cv2.estimateAffine2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if affine is not None:
        models["affine"] = (affine, int(inliers.sum()), "affine")
    homography, inliers = cv2.findHomography(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if homography is not None:
        models["homography"] = (homography, int(inliers.sum()), "homography")
    return models


def invert(matrix: np.ndarray, kind: str) -> np.ndarray:
    import cv2

    if kind == "affine":
        return cv2.invertAffineTransform(matrix)
    return np.linalg.inv(matrix)


def map_points(matrix: np.ndarray, kind: str, points: np.ndarray) -> np.ndarray:
    if kind == "affine":
        return points @ matrix[:, :2].T + matrix[:, 2]
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def open_full(path: str):
    import pyvips

    image = pyvips.Image.new_from_file(path, access="random")
    if image.hasalpha():
        image = image.flatten(background=255)
    return image


def read_patch(image, centre, mpp: float):
    side = int(round(PATCH_UM / mpp))
    x = int(round(centre[0])) - side // 2
    y = int(round(centre[1])) - side // 2
    if x < 0 or y < 0 or x + side > image.width or y + side > image.height:
        return None
    patch = image.crop(x, y, side, side).numpy()
    return -np.log10(np.clip(patch.astype(np.float32), 1.0, 255.0) / 255.0).mean(axis=2)


def residual(reference, moving, window, allow_flip: bool):
    import cv2

    if moving.shape != reference.shape:
        moving = cv2.resize(moving, reference.shape[::-1], interpolation=cv2.INTER_AREA)
    best = None
    candidates = [(moving, "upright")]
    if allow_flip:
        candidates.append((np.rot90(moving, 2), "rot180"))
    for candidate, tag in candidates:
        a = np.ascontiguousarray(reference - reference.mean(), dtype=np.float64)
        b = np.ascontiguousarray(candidate - candidate.mean(), dtype=np.float64)
        (dx, dy), response = cv2.phaseCorrelate(a, b, window)
        if best is None or response > best[3]:
            best = (dx, dy, tag, response)
    return best


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    stains = sorted(manifest["stain"].unique())
    stain = stains[args.task_index]
    section = json.loads((Path(args.registration) / f"{stain}.json").read_text())
    wsi_dir = Path(args.wsi_dir)

    thumbs, meta = {}, {}
    for scanner, entry in section["scanners"].items():
        meta[scanner] = entry
        thumbs[scanner] = thumbnail(str(wsi_dir / entry["name"]), entry["mpp"])
    fixed = thumbs[REFERENCE]

    locations = np.array(section["locations_reference_thumb_xy"], dtype=np.float64)[: args.locations]
    reference_mpp = meta[REFERENCE]["mpp"]
    side = int(round(PATCH_UM / reference_mpp))
    window = np.outer(np.hanning(side), np.hanning(side))

    reference_image = open_full(str(wsi_dir / meta[REFERENCE]["name"]))
    references = {}
    for index, point in enumerate(locations):
        patch = read_patch(reference_image, point * THUMB_UM / reference_mpp, reference_mpp)
        if patch is not None:
            references[index] = patch

    rows = []
    for scanner in sorted(thumbs):
        if scanner == REFERENCE:
            continue
        source, target = match_points(thumbs[scanner], fixed)
        models = fit_models(source, target)
        image = open_full(str(wsi_dir / meta[scanner]["name"]))
        allow_flip = scanner == "P"
        for name, (matrix, inliers, kind) in models.items():
            inverse = invert(matrix, kind)
            mapped = map_points(inverse, kind, locations) * THUMB_UM / meta[scanner]["mpp"]
            offsets = []
            for index, centre in enumerate(mapped):
                if index not in references:
                    continue
                patch = read_patch(image, centre, meta[scanner]["mpp"])
                if patch is None:
                    continue
                dx, dy, tag, response = residual(references[index], patch, window, allow_flip)
                offsets.append((float(np.hypot(dx, dy) * reference_mpp), response, tag))
            if not offsets:
                continue
            values = np.array([o[0] for o in offsets])
            rows.append({
                "stain": stain, "scanner": scanner, "model": name, "dof": {"similarity": 4, "affine": 6, "homography": 8}[name],
                "inliers": inliers, "n": len(values),
                "median_um": float(np.median(values)),
                "p90_um": float(np.percentile(values, 90)),
                "response": float(np.median([o[1] for o in offsets])),
                "rot180_frac": float(np.mean([o[2] == "rot180" for o in offsets])),
            })
            print(f"  {scanner:6s} {name:11s} inliers={inliers:5d} "
                  f"median={rows[-1]['median_um']:6.2f} um  p90={rows[-1]['p90_um']:6.2f}  "
                  f"resp={rows[-1]['response']:.2f}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{stain}.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(f"{stain} -> {destination}")


if __name__ == "__main__":
    main()
