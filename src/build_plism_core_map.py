"""Attach PLISM's 46 tissue-type names to the TMA cores of one native section.

PLISM ships a tissue label for every patch of its *registered* subset, but those
coordinates live in the authors' Elastix canvas, not on any scanner's pixel grid.
The earlier pass therefore labelled cores by position and left the names on the
table (see the docstring of build_plism_registration_report_assets.py).

Nothing here reverses that decision.  The canvas is used only to answer *which
core is this*, and a core is 2.5 mm across on a 3.6 mm pitch, so an assignment
that is right to within a millimetre is right.  A single global similarity --
one scale, one rotation, one translation -- is fitted between the canvas
occupancy lattice and this section's AT2 tissue mask at 16 um/px, and it is used
to carry *labels* inward, never to place a pixel.  Every analysed patch is still
read by integer crop at level 0 on its own scanner's grid.

The fit is scored by Dice on the tissue masks and the score is stored, so the
labelling can be gated on it downstream rather than assumed.

Output is schema-compatible with build_plism_section_registration.py, so the
existing refinement and extraction read it unchanged; `location_core` is the
official core index 1..46 and `location_tissue` carries the name.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_plism_section_registration import (
    REFERENCE, THUMB_UM, apply_affine, register, thumbnail,
)
from plism_dataset_corrections import EVIDENCE, corrected_name
from extract_plism_native_psd import PATCH_UM, TISSUE_OD_FLOOR

# The registered subset is a 512 px patch every 1024 canvas units, so 1024 units
# is the finest lattice the annotation actually resolves.
CANVAS_CELL = 1024
CANVAS_PATCH = 512

# Thumbnail pixels per lattice cell.  A canvas unit is one pixel of the authors'
# reference grid, near 0.22 um, so a 1024-unit cell is about 225 um and lands on
# roughly 14 pixels of a 16 um/px thumbnail.
CELL_PX_COARSE = np.linspace(11.0, 18.0, 29)
ROTATION_COARSE = np.concatenate([np.arange(-8.0, 8.01, 1.0), np.arange(172.0, 188.01, 1.0)])
CELL_PX_REFINE = 0.05
ROTATION_REFINE = 0.25

# A core assignment further than this from a core centroid is not an assignment.
ASSIGN_RADIUS_UM = 1600.0
# A PLISM core is about 2.5 mm across; the contrast window stays inside one.
CORE_RADIUS_UM = 1250.0
MIN_DICE = 0.70

# Tissue is mean optical density above a fixed floor, exactly the contract's rule.
# Otsu is deliberately *not* used here: on a TMA it splits dark tissue from pale
# tissue rather than tissue from glass, and the pale cores -- penis, renal cortex,
# lung, collagenous fibre -- fall on the glass side and are never sampled.
# A per-core report cannot afford to lose ten of its forty-six units that way.
#
# A location is kept when most of its patch footprint is tissue rather than all of
# it, so reticular tissue (lung, cartilage, collagen) is reachable at all.
FOOTPRINT_TISSUE = 0.70


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="data/PLISM_dataset/PLISM_wsi_en.csv")
    parser.add_argument("--registration", default="outputs/plism_section_registration")
    parser.add_argument("--task-index", type=int, required=True, help="stain index, 0-12")
    parser.add_argument("--wsi-dir", default="data/PLISM_dataset/original_wsi")
    parser.add_argument("--output", default="outputs/plism_core_registration")
    parser.add_argument("--qc-dir", default="outputs/plism_core_map_qc")
    parser.add_argument("--per-core", type=int, default=0,
                        help="cap on locations per core; 0 keeps the whole lattice")
    parser.add_argument("--pitch", type=float, default=1.0,
                        help="lattice step as a multiple of the patch side; 1.0 tiles "
                             "each core without overlap and without gaps")
    parser.add_argument("--no-file-corrections", action="store_true",
                        help="use PLISM's file labels as published, swaps included")
    parser.add_argument("--resample", action="store_true",
                        help="reuse the stored canvas fit and only redo the sampling")
    parser.add_argument("--seed-base", type=int, default=20260821)
    return parser.parse_args()


def canvas_cores(annotation: str, stain: str) -> tuple[pd.DataFrame, np.ndarray, tuple]:
    """Per-core centroids and an occupancy lattice, both in canvas units."""
    table = pd.read_csv(annotation)
    block = table.loc[(table["stain"] == stain) & (table["device"] == REFERENCE)].copy()
    if block.empty:
        raise RuntimeError(f"{stain}: no {REFERENCE} rows in the annotation")
    coords = block["coordinate"].str.split("_", expand=True).astype(int)
    # coordinates name the patch corner; work with its centre
    block["x"] = coords[0] + CANVAS_PATCH / 2
    block["y"] = coords[1] + CANVAS_PATCH / 2

    cores = (
        block.groupby("tissue_type")
        .agg(x=("x", "mean"), y=("y", "mean"), patches=("x", "size"))
        .reset_index()
        .sort_values("tissue_type")
        .reset_index(drop=True)
    )
    cores["core"] = cores.index + 1

    origin = (float(block["x"].min()), float(block["y"].min()))
    columns = int((block["x"].max() - origin[0]) // CANVAS_CELL) + 1
    rows = int((block["y"].max() - origin[1]) // CANVAS_CELL) + 1
    lattice = np.zeros((rows, columns), dtype=np.float32)
    ix = ((block["x"] - origin[0]) // CANVAS_CELL).astype(int)
    iy = ((block["y"] - origin[1]) // CANVAS_CELL).astype(int)
    lattice[iy, ix] = 1.0
    return cores, lattice, origin


def tissue_mask(thumb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    white = max(float(np.percentile(thumb, 99.0)), 1.0)
    od = -np.log10(np.clip(thumb.astype(np.float32), 1.0, None) / white)
    return od > TISSUE_OD_FLOOR, od, float(TISSUE_OD_FLOOR)


def warp_lattice(lattice: np.ndarray, cell_px: float, rotation: float) -> np.ndarray:
    """Lattice resampled so one 1024-unit cell covers `cell_px` thumbnail pixels."""
    import cv2

    height = max(int(round(lattice.shape[0] * cell_px)), 1)
    width = max(int(round(lattice.shape[1] * cell_px)), 1)
    grown = cv2.resize(lattice, (width, height), interpolation=cv2.INTER_NEAREST)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -rotation, 1.0)
    # rotate into a canvas large enough that no tissue leaves the frame
    diagonal = int(np.ceil(np.hypot(width, height)))
    matrix[0, 2] += (diagonal - width) / 2
    matrix[1, 2] += (diagonal - height) / 2
    rotated = cv2.warpAffine(grown, matrix, (diagonal, diagonal), flags=cv2.INTER_NEAREST)
    return rotated


def best_shift(moving: np.ndarray, fixed_spectrum: dict) -> tuple[np.ndarray, float]:
    """Translation maximising binary overlap, by FFT cross-correlation."""
    height, width = fixed_spectrum["shape"]
    b = np.zeros((height, width), dtype=np.float32)
    b[: moving.shape[0], : moving.shape[1]] = moving
    correlation = np.fft.irfft2(fixed_spectrum["fft"] * np.conj(np.fft.rfft2(b)), s=(height, width))
    peak_y, peak_x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    overlap = float(correlation[peak_y, peak_x])
    dy = peak_y - height if peak_y > height // 2 else peak_y
    dx = peak_x - width if peak_x > width // 2 else peak_x
    total = fixed_spectrum["sum"] + float(b.sum())
    dice = 2.0 * overlap / total if total > 0 else 0.0
    return np.array([dx, dy], dtype=np.float64), dice


def fit_canvas_to_thumb(lattice: np.ndarray, mask: np.ndarray) -> dict:
    """Similarity taking canvas lattice coordinates into thumbnail pixels."""
    fixed = mask.astype(np.float32)
    # Only the lattice moves, so the reference spectrum is computed once per pad
    # size.  The scale sweep uses a handful of sizes, not one per evaluation.
    cache: dict[tuple[int, int], dict] = {}

    def spectrum_for(shape: tuple[int, int]) -> dict:
        height = int(2 ** np.ceil(np.log2(max(shape[0], fixed.shape[0]) * 2)))
        width = int(2 ** np.ceil(np.log2(max(shape[1], fixed.shape[1]) * 2)))
        if (height, width) not in cache:
            padded = np.zeros((height, width), dtype=np.float32)
            padded[: fixed.shape[0], : fixed.shape[1]] = fixed
            cache[(height, width)] = {"fft": np.fft.rfft2(padded), "shape": (height, width),
                                      "sum": float(fixed.sum())}
        return cache[(height, width)]

    def evaluate(cell_px: float, rotation: float) -> dict:
        moving = warp_lattice(lattice, cell_px, rotation)
        shift, dice = best_shift(moving, spectrum_for(moving.shape))
        return {"cell_px": float(cell_px), "rotation": float(rotation),
                "shift": shift, "dice": dice, "moving_shape": tuple(moving.shape)}

    best = max((evaluate(c, r) for c in CELL_PX_COARSE for r in ROTATION_COARSE),
               key=lambda e: e["dice"])
    cells = np.arange(best["cell_px"] - 0.25, best["cell_px"] + 0.2501, CELL_PX_REFINE)
    rotations = np.arange(best["rotation"] - 1.0, best["rotation"] + 1.001, ROTATION_REFINE)
    best = max((evaluate(c, r) for c in cells for r in rotations), key=lambda e: e["dice"])
    return best


def canvas_matrix(fit: dict, lattice_shape: tuple, origin: tuple) -> np.ndarray:
    """2x3 matrix mapping canvas unit coordinates to thumbnail pixels."""
    import cv2

    factor = fit["cell_px"]
    height = max(int(round(lattice_shape[0] * factor)), 1)
    width = max(int(round(lattice_shape[1] * factor)), 1)
    rotate = cv2.getRotationMatrix2D((width / 2, height / 2), -fit["rotation"], 1.0)
    diagonal = int(np.ceil(np.hypot(width, height)))
    rotate[0, 2] += (diagonal - width) / 2 + fit["shift"][0]
    rotate[1, 2] += (diagonal - height) / 2 + fit["shift"][1]

    # canvas unit -> lattice cell -> grown pixel, then the rotation above
    pre = np.array([[factor / CANVAS_CELL, 0.0, -origin[0] * factor / CANVAS_CELL],
                    [0.0, factor / CANVAS_CELL, -origin[1] * factor / CANVAS_CELL]])
    full = np.vstack([rotate, [0, 0, 1]]) @ np.vstack([pre, [0, 0, 1]])
    return full[:2]


def main() -> None:
    args = parse_args()
    registration_dir = Path(args.registration)
    stains = sorted(p.stem for p in registration_dir.glob("*.json"))
    if not 0 <= args.task_index < len(stains):
        raise IndexError(f"stain index outside 0..{len(stains) - 1}")
    stain = stains[args.task_index]
    section = json.loads((registration_dir / f"{stain}.json").read_text())
    wsi_dir = Path(args.wsi_dir)

    # Two SQ files hold each other's section; see plism_dataset_corrections.
    # The scanner's metadata and its transform both have to be redone, because
    # both were derived from the wrong file.
    corrections = []
    if not args.no_file_corrections:
        import pyvips

        reference_thumb = thumbnail(
            str(wsi_dir / section["scanners"][REFERENCE]["name"]),
            section["scanners"][REFERENCE]["mpp"])
        for scanner, meta in section["scanners"].items():
            name, changed = corrected_name(stain, scanner, meta["name"])
            if not changed:
                continue
            probe = pyvips.Image.new_from_file(str(wsi_dir / name), access="sequential")
            mpp = float(probe.get("openslide.mpp-x"))
            meta.update(name=name, mpp=mpp, width=int(probe.width), height=int(probe.height))
            moving = thumbnail(str(wsi_dir / name), mpp)
            meta["thumb_shape"] = list(moving.shape)
            section["transforms"][scanner] = register(moving, reference_thumb)
            corrections.append({"scanner": scanner, "published": f"{stain}_{scanner}",
                                "actual_file": name, "evidence": EVIDENCE,
                                "refitted_ok": bool(section["transforms"][scanner]["ok"])})
            print(f"{stain}: {scanner} corrected to {name}, transform refitted "
                  f"(ok={section['transforms'][scanner]['ok']}, "
                  f"inliers={section['transforms'][scanner].get('inliers')})")

    reference_meta = section["scanners"][REFERENCE]
    thumb = thumbnail(str(wsi_dir / reference_meta["name"]), reference_meta["mpp"])
    mask, optical_density, threshold = tissue_mask(thumb)

    cores, lattice, origin = canvas_cores(args.annotation, stain)
    stored = Path(args.output) / f"{stain}.json"
    if args.resample and stored.exists():
        # The canvas fit is the expensive half and does not depend on sampling.
        previous = json.loads(stored.read_text())["canvas_fit"]
        fit = {"dice": previous["dice"], "cell_px": previous["cell_px"],
               "rotation": previous["rotation_deg"],
               "shift": np.array(previous["shift_px"], dtype=np.float64)}
        matrix = np.array(previous["matrix_canvas_to_thumb"], dtype=np.float64)
        print(f"{stain}: reusing stored canvas fit, dice={fit['dice']:.3f}")
    else:
        fit = fit_canvas_to_thumb(lattice, mask)
        matrix = canvas_matrix(fit, lattice.shape, origin)
    centres = apply_affine(matrix, cores[["x", "y"]].to_numpy(dtype=np.float64))
    cores["thumb_x"] = centres[:, 0]
    cores["thumb_y"] = centres[:, 1]

    print(f"{stain}: canvas dice={fit['dice']:.3f} cell={fit['cell_px']:.2f} px "
          f"({fit['cell_px'] * THUMB_UM / CANVAS_CELL:.4f} um/unit) "
          f"rot={fit['rotation']:+.2f} deg shift={fit['shift'].tolist()}")
    if fit["dice"] < MIN_DICE:
        raise RuntimeError(f"{stain}: canvas fit dice {fit['dice']:.3f} below {MIN_DICE}")

    # Candidate locations: mostly-tissue patches that are in bounds on every scanner.
    from scipy.ndimage import uniform_filter

    footprint = int(np.ceil(PATCH_UM / THUMB_UM))
    covered = uniform_filter(mask.astype(np.float32), size=footprint, mode="constant")
    interior = covered >= FOOTPRINT_TISSUE
    rows, columns = np.nonzero(interior)
    candidates = np.stack([columns, rows], axis=1).astype(np.float64)

    keep = np.ones(len(candidates), dtype=bool)
    for scanner, transform in section["transforms"].items():
        if not transform["ok"]:
            continue
        inverse = np.array(transform["inverse"], dtype=np.float64)
        mapped = apply_affine(inverse, candidates) * THUMB_UM / section["scanners"][scanner]["mpp"]
        side = int(round(PATCH_UM / section["scanners"][scanner]["mpp"]))
        keep &= (
            (mapped[:, 0] - side / 2 >= 0)
            & (mapped[:, 1] - side / 2 >= 0)
            & (mapped[:, 0] + side / 2 < section["scanners"][scanner]["width"])
            & (mapped[:, 1] + side / 2 < section["scanners"][scanner]["height"])
        )
    candidates = candidates[keep]

    # Nearest official core, rejected beyond one core radius.
    radius_px = ASSIGN_RADIUS_UM / THUMB_UM
    delta = candidates[:, None, :] - centres[None, :, :]
    distance = np.hypot(delta[..., 0], delta[..., 1])
    nearest = np.argmin(distance, axis=1)
    within = distance[np.arange(len(candidates)), nearest] <= radius_px

    # Locations are a lattice at the patch pitch, not a random sample.  Random
    # sampling of overlapping candidate positions re-measures the same pixels and
    # leaves most of the section untouched: twelve per core covered 6% of the
    # tissue, while a pitch-1.0 lattice covers about 90% with no overlap at all.
    step = max(args.pitch * PATCH_UM / THUMB_UM, 1.0)
    digest = hashlib.sha256(f"{stain}-core".encode()).hexdigest()
    seed = args.seed_base + int(digest[:8], 16)
    rng = np.random.default_rng(seed)
    # One shared sub-pixel phase per section, so the lattice is not aligned to the
    # image border in a way that could favour one part of every core.
    phase = rng.uniform(0.0, step, size=2)
    on_lattice = np.all(np.abs(
        ((candidates - phase + step / 2) % step) - step / 2) < 0.5, axis=1)

    picked, core_ids, tissues = [], [], []
    coverage = []
    for index, row in cores.iterrows():
        pool = candidates[within & (nearest == index) & on_lattice]
        take = len(pool) if args.per_core <= 0 else min(args.per_core, len(pool))
        # Reference contrast, so the report can tell a core that is hard to align
        # from a core that carries too little signal to align *or* to check.
        half = int(round(CORE_RADIUS_UM * 0.8 / THUMB_UM))
        window = optical_density[
            max(int(row["thumb_y"]) - half, 0):int(row["thumb_y"]) + half,
            max(int(row["thumb_x"]) - half, 0):int(row["thumb_x"]) + half]
        coverage.append({"core": int(row["core"]), "tissue_type": row["tissue_type"],
                         "available": int(len(pool)), "sampled": int(take),
                         "canvas_patches": int(row["patches"]),
                         "od_median": float(np.median(window)) if window.size else float("nan"),
                         "od_std": float(window.std()) if window.size else float("nan"),
                         "thumb_x": float(row["thumb_x"]), "thumb_y": float(row["thumb_y"])})
        if take == 0:
            continue
        chosen = pool if take == len(pool) else pool[rng.choice(len(pool), size=take,
                                                                replace=False)]
        chosen = chosen[np.lexsort((chosen[:, 0], chosen[:, 1]))]
        picked.append(chosen)
        core_ids += [int(row["core"])] * take
        tissues += [row["tissue_type"]] * take

    if not picked:
        raise RuntimeError(f"{stain}: no core received a location")
    picked = np.concatenate(picked)
    empty = [c["tissue_type"] for c in coverage if c["sampled"] == 0]
    # Actual tissue reached, not patch area: a location is kept when 70% of its
    # footprint is tissue, so counting whole patches would overstate the coverage.
    stamp = np.zeros(mask.shape, dtype=np.float32)
    stamp[picked[:, 1].astype(int), picked[:, 0].astype(int)] = 1.0
    reached = (uniform_filter(stamp, size=footprint, mode="constant") > 0) & mask
    tissue_mm2 = float(mask.sum()) * (THUMB_UM / 1000.0) ** 2
    covered_mm2 = float(reached.sum()) * (THUMB_UM / 1000.0) ** 2
    print(f"  {len(cores)} official cores, {len(picked)} locations on a "
          f"{args.pitch:.2f}x patch-pitch lattice, {len(cores) - len(empty)} cores covered")
    print(f"  tissue {tissue_mm2:.1f} mm2, patches reach {covered_mm2:.1f} mm2 "
          f"({covered_mm2 / tissue_mm2 * 100:.0f}%)")
    if empty:
        print(f"  no tissue candidate: {', '.join(empty)}")

    payload = dict(section)
    payload.update({
        "stain": stain,
        "reference": REFERENCE,
        "thumb_um_per_px": THUMB_UM,
        "patch_um": PATCH_UM,
        "mask_threshold_od": threshold,
        "footprint_tissue_fraction": FOOTPRINT_TISSUE,
        "reference_tissue_fraction": float(mask.mean()),
        "shared_candidates": int(len(candidates)),
        "locations_reference_thumb_xy": picked.tolist(),
        "location_core": core_ids,
        "location_tissue": tissues,
        "cores": int(len(set(core_ids))),
        "per_core": int(args.per_core),
        "lattice_pitch": float(args.pitch),
        "lattice_step_px": float(step),
        "tissue_mm2": tissue_mm2,
        "tissue_reached_mm2": covered_mm2,
        "core_labels": coverage,
        "canvas_fit": {"dice": fit["dice"], "cell_px": fit["cell_px"],
                       "canvas_um_per_unit": fit["cell_px"] * THUMB_UM / CANVAS_CELL,
                       "rotation_deg": fit["rotation"], "shift_px": fit["shift"].tolist(),
                       "matrix_canvas_to_thumb": matrix.tolist(),
                       "min_dice": MIN_DICE, "assign_radius_um": ASSIGN_RADIUS_UM},
        "seed": seed,
        "file_corrections": corrections,
    })

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{stain}.json"
    destination.write_text(json.dumps(payload, indent=2))
    print(f"  -> {destination}")

    qc_dir = Path(args.qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    write_qc(qc_dir / f"{stain}.png", thumb, mask, cores, picked, fit)
    print(f"  -> {qc_dir / f'{stain}.png'}")


def write_qc(path: Path, thumb, mask, cores, picked, fit) -> None:
    """Thumbnail with the fitted core centroids and the sampled locations drawn on."""
    import cv2

    canvas = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
    canvas[mask] = (0.75 * canvas[mask] + 0.25 * np.array([255, 120, 120])).astype(np.uint8)
    for point in picked:
        cv2.circle(canvas, (int(point[0]), int(point[1])), 2, (30, 200, 60), -1)
    for _, row in cores.iterrows():
        centre = (int(row["thumb_x"]), int(row["thumb_y"]))
        cv2.circle(canvas, centre, int(round(CORE_RADIUS_UM / THUMB_UM)), (255, 60, 60), 1)
        cv2.putText(canvas, row["tissue_type"].split("_")[0], (centre[0] - 12, centre[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 220), 2)
    cv2.putText(canvas, f"dice={fit['dice']:.3f} rot={fit['rotation']:+.2f}",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.imwrite(str(path), canvas)


if __name__ == "__main__":
    main()
