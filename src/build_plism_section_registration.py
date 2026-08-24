"""Coarse registration and a shared location set for one PLISM stain section.

The seven scanners of a stain image the same physical section, but each WSI has
its own origin, and the Philips unit scans the slide rotated by 180 degrees.  A
similarity transform per scanner is estimated once, at 16 um/px, and is used
**only to decide where to read**.  Patches themselves are later read by integer
crop at level 0 on each scanner's own pixel grid, so no analysed pixel is ever
interpolated.

Locations are chosen once, on the section's AT2 reference, and mapped outward.
Choosing them independently per WSI made tissue selection scanner-dependent and
corrupted the estimate; see Amendment 1 of the contract.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import ndimage

from extract_plism_native_psd import PATCH_UM, TISSUE_OD_FLOOR, otsu_threshold

REFERENCE = "AT2"
THUMB_UM = 16.0
MIN_INLIERS = 100
CORE_MIN_AREA = 1500
SCALE_TOLERANCE = (0.98, 1.02)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/PLISM_dataset/manifest_original.csv")
    parser.add_argument("--task-index", type=int, required=True, help="stain index, 0-12")
    parser.add_argument("--wsi-dir", default="data/PLISM_dataset/original_wsi")
    parser.add_argument("--output", default="outputs/plism_section_registration")
    parser.add_argument("--locations", type=int, default=200,
                        help="minimum shared positions required before stratifying")
    parser.add_argument("--per-core", type=int, default=12,
                        help="locations sampled per TMA core; the REML block size")
    parser.add_argument("--seed-base", type=int, default=20260808)
    return parser.parse_args()


def thumbnail(path: str, mpp: float) -> np.ndarray:
    """Grayscale view of the whole slide at exactly THUMB_UM per pixel."""
    import cv2
    import pyvips

    probe = pyvips.Image.new_from_file(path, access="sequential")
    count = int(probe.get("openslide.level-count"))
    downsamples = [float(probe.get(f"openslide.level[{i}].downsample")) for i in range(count)]
    level = int(np.argmin([abs(np.log(d * mpp / THUMB_UM)) for d in downsamples]))

    image = pyvips.Image.new_from_file(path, level=level, access="sequential")
    if image.hasalpha():
        image = image.flatten(background=255)
    array = image.colourspace("b-w").numpy()
    if array.ndim == 3:
        array = array[..., 0]
    scale = (downsamples[level] * mpp) / THUMB_UM
    return cv2.resize(array.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def register(moving: np.ndarray, fixed: np.ndarray) -> dict:
    """Similarity transform taking `moving` thumbnail coordinates into `fixed`."""
    import cv2

    detector = cv2.AKAZE_create()
    kp_move, desc_move = detector.detectAndCompute(moving, None)
    kp_fix, desc_fix = detector.detectAndCompute(fixed, None)
    if desc_move is None or desc_fix is None:
        return {"ok": False, "reason": "no descriptors"}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_move, desc_fix, k=2)
    good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
    if len(good) < 8:
        return {"ok": False, "reason": f"only {len(good)} matches"}

    source = np.float32([kp_move[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    target = np.float32([kp_fix[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=2.0
    )
    if matrix is None:
        return {"ok": False, "reason": "RANSAC failed"}

    scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    rotation = float(np.degrees(np.arctan2(matrix[0, 1], matrix[0, 0])))
    count = int(inliers.sum())
    ok = count >= MIN_INLIERS and SCALE_TOLERANCE[0] <= scale <= SCALE_TOLERANCE[1]
    return {
        "ok": ok,
        "reason": "" if ok else f"inliers={count} scale={scale:.4f}",
        "matrix": matrix.tolist(),
        "inverse": cv2.invertAffineTransform(matrix).tolist(),
        "inliers": count,
        "matches": len(good),
        "scale": scale,
        "rotation_deg": rotation,
    }


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    stains = sorted(manifest["stain"].unique())
    if not 0 <= args.task_index < len(stains):
        raise IndexError(f"stain index outside 0..{len(stains) - 1}")
    stain = stains[args.task_index]
    block = manifest.loc[manifest["stain"] == stain].set_index("scanner")
    if REFERENCE not in block.index:
        raise RuntimeError(f"{stain}: no {REFERENCE} slide")

    wsi_dir = Path(args.wsi_dir)
    thumbs, meta = {}, {}
    for scanner, row in block.iterrows():
        path = str(wsi_dir / str(row["name"]))
        import pyvips

        probe = pyvips.Image.new_from_file(path, access="sequential")
        mpp = float(probe.get("openslide.mpp-x"))
        thumbs[scanner] = thumbnail(path, mpp)
        meta[scanner] = {
            "name": str(row["name"]),
            "mpp": mpp,
            "width": int(probe.width),
            "height": int(probe.height),
            "thumb_shape": list(thumbs[scanner].shape),
        }

    fixed = thumbs[REFERENCE]
    transforms = {REFERENCE: {"ok": True, "matrix": [[1, 0, 0], [0, 1, 0]],
                              "inverse": [[1, 0, 0], [0, 1, 0]], "inliers": -1,
                              "matches": -1, "scale": 1.0, "rotation_deg": 0.0, "reason": ""}}
    for scanner, thumb in thumbs.items():
        if scanner == REFERENCE:
            continue
        transforms[scanner] = register(thumb, fixed)

    failed = [s for s, t in transforms.items() if not t["ok"]]

    # Locations: one Otsu mask on the reference only, eroded so the whole patch is tissue.
    from scipy.ndimage import minimum_filter

    white = max(float(np.percentile(fixed, 99.0)), 1.0)
    od = -np.log10(np.clip(fixed.astype(np.float32), 1.0, None) / white)
    threshold = max(otsu_threshold(od), TISSUE_OD_FLOOR)
    mask = od > threshold
    footprint = int(np.ceil(PATCH_UM / THUMB_UM))
    interior = minimum_filter(mask.astype(np.uint8), size=footprint, mode="constant") > 0
    rows, cols = np.nonzero(interior)
    candidates = np.stack([cols, rows], axis=1).astype(np.float64)  # (x, y) in reference thumb

    # Keep only locations whose full patch is in bounds on every passing scanner.
    keep = np.ones(len(candidates), dtype=bool)
    for scanner, transform in transforms.items():
        if not transform["ok"]:
            continue
        inverse = np.array(transform["inverse"], dtype=np.float64)
        mapped = apply_affine(inverse, candidates) * THUMB_UM / meta[scanner]["mpp"]
        side = int(round(PATCH_UM / meta[scanner]["mpp"]))
        keep &= (
            (mapped[:, 0] - side / 2 >= 0)
            & (mapped[:, 1] - side / 2 >= 0)
            & (mapped[:, 0] + side / 2 < meta[scanner]["width"])
            & (mapped[:, 1] + side / 2 < meta[scanner]["height"])
        )
    candidates = candidates[keep]
    if len(candidates) < args.locations:
        raise RuntimeError(
            f"{stain}: {len(candidates)} shared locations for {args.locations} requested"
        )

    # Stratify by TMA core.  The nested REML downstream needs a balanced replicate
    # count inside every core, and uniform sampling over the whole section does not
    # give one -- large cores would swamp small ones and the block structure would
    # be unbalanced.
    components, n_components = ndimage.label(interior)
    core_of = components[candidates[:, 1].astype(int), candidates[:, 0].astype(int)]
    areas = np.bincount(components.ravel(), minlength=n_components + 1)
    usable = [c for c in range(1, n_components + 1)
              if areas[c] >= CORE_MIN_AREA and (core_of == c).sum() >= args.per_core]
    if not usable:
        raise RuntimeError(f"{stain}: no core holds {args.per_core} shared locations")

    digest = hashlib.sha256(stain.encode()).hexdigest()
    rng = np.random.default_rng(args.seed_base + int(digest[:8], 16))
    chosen, core_ids = [], []
    for rank, component in enumerate(sorted(usable, key=lambda c: -areas[c])):
        pool = candidates[core_of == component]
        take = pool[rng.choice(len(pool), size=args.per_core, replace=False)]
        chosen.append(take)
        core_ids += [rank + 1] * args.per_core
    picked = np.concatenate(chosen)
    core_ids = np.array(core_ids)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stain": stain,
        "reference": REFERENCE,
        "thumb_um_per_px": THUMB_UM,
        "patch_um": PATCH_UM,
        "mask_threshold_od": threshold,
        "reference_tissue_fraction": float(mask.mean()),
        "shared_candidates": int(len(candidates)),
        "locations_reference_thumb_xy": picked.tolist(),
        "location_core": core_ids.tolist(),
        "cores": int(len(set(core_ids.tolist()))),
        "per_core": int(args.per_core),
        "scanners": meta,
        "transforms": transforms,
        "failed_scanners": failed,
        "seed": args.seed_base + int(digest[:8], 16),
    }
    destination = output_dir / f"{stain}.json"
    destination.write_text(json.dumps(payload, indent=2))

    print(f"{stain}: reference {REFERENCE}, tissue {mask.mean():.3f}, "
          f"{len(candidates)} shared candidates, {len(set(core_ids.tolist()))} cores x "
          f"{args.per_core} = {len(picked)} locations")
    for scanner in sorted(transforms):
        t = transforms[scanner]
        flag = "ok " if t["ok"] else "FAIL"
        print(f"  {flag} {scanner:6s} inliers={t.get('inliers', 0):5d} "
              f"scale={t.get('scale', float('nan')):.4f} rot={t.get('rotation_deg', float('nan')):+8.2f} "
              f"{t.get('reason', '')}")
    print(f"  -> {destination}")


if __name__ == "__main__":
    main()
