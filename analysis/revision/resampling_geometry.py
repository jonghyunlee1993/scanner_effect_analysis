#!/usr/bin/env python3
"""RV10 Stage A: map every set20 registered patch back to its native WSI.

For each slide, scanner and ``set20`` location the registered (pipeline) patch is the
cached 256x256 crop read at ``final_coords_xy`` from the VALIS-registered image in the
AT2 frame (0.5052 um/px).  The historical E0 native-geometry manifest supplies one
native-to-registered similarity per slide and scanner.  Because the current registered
images need not share E0's registered frame (AKOYA was re-registered), the transform is
re-estimated here from the set20 locations themselves:

1. pass 1: render a wide native region with the E0 transform (Gaussian pre-filter then
   bicubic), find the cached patch by coarse-to-fine NCC on blurred optical density and
   refine a local affine by ECC;
2. fit one affine per slide and scanner from the confident pass-1 correspondences
   (iteratively trimmed least squares);
3. pass 2: repeat the local match with the refitted transform and store, per location,
   the affine that maps patch pixels to native level-0 pixels.

A location is *confirmed* when the pass-2 ECC correlation is >= 0.95 and the integer
peak is not on the search boundary.  Stage A passes when >= 95% of set20 locations are
confirmed for each of the five non-reference scanners.  The per-location affines are
the geometry used by Stage B (``resampling_render.py``).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import openslide
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "results/resampling_robustness"
GEOMETRY = OUTPUT / "geometry"
SET20 = Path(__file__).resolve().parent / "results/location_sets/set20.csv"
COHORT = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17/00_contract/cohort.csv"
MANIFEST = PROJECT / "data/provenance/pannormal_native_geometry_manifest.csv"
RAW_ROOT = Path("/mnt/isilon/oldridge_lab/batch_effects/pan_normal/raw/raw_images")
REGISTERED_ROOT = Path(
    "/mnt/isilon/oldridge_lab/batch_effects/pan_normal/registered_ref_at2_all/registered_images"
)
SCANNERS = ("at2", "versa", "akoya", "gt450", "s360", "s60")  # cache order
TARGETS = ("versa", "akoya", "gt450", "s360", "s60")
EXTENSIONS = {"at2": "svs", "versa": "svs", "akoya": "qptiff", "gt450": "svs",
              "s360": "ndpi", "s60": "ndpi"}
NATIVE_MPP = {"at2": 0.5052, "versa": 0.2742, "akoya": 0.4999, "gt450": 0.262407,
              "s360": 0.4603, "s60": 0.4426}
TARGET_MPP = 0.5052
PATCH = 256
PASS1_SEARCH = 256
PASS2_SEARCH = 64
COARSE_FACTOR = 4
BLUR_SIGMA = 1.5
ECC_MIN = 0.95
FIT_MIN_CC = 0.95
FIT_TRIM_PX = 3.0
STAGE_A_MIN_FRACTION = 0.95
NATIVE_PAD = 16


def od(rgb: np.ndarray) -> np.ndarray:
    """Mean optical density, as in the patch cache and the paper's spectra."""
    value = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 255.0)
    return -np.log((value + 1.0) / 256.0).mean(axis=-1)


def blurred_od(rgb: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(od(rgb), (0, 0), BLUR_SIGMA)


def native_path(scanner: str, slide_id: str) -> Path:
    return RAW_ROOT / scanner / f"{slide_id}.{EXTENSIONS[scanner]}"


def to3(affine: np.ndarray) -> np.ndarray:
    return np.vstack([np.asarray(affine, dtype=np.float64)[:2], [0.0, 0.0, 1.0]])


def e0_native_to_target(manifest: pd.DataFrame, slide_id: str, scanner: str) -> np.ndarray:
    rows = manifest[(manifest.slide_id == slide_id) & (manifest.scanner == scanner)]
    if rows.empty:
        raise KeyError(f"{slide_id}/{scanner}: no E0 native geometry")
    names = [f"native_to_target_m{index}" for index in ("00", "01", "02", "10", "11", "12")]
    values = rows[names].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"{slide_id}/{scanner}: E0 transform is not unique")
    return to3(values.to_numpy(dtype=np.float64).reshape(2, 3))


def read_native_for(handle: openslide.OpenSlide, grid_to_native: np.ndarray,
                    width: int, height: int, pad: int = NATIVE_PAD):
    """Read the native level-0 box covering an output grid; return crop and offset."""
    corners = np.array([[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1],
                        [0, height - 1, 1]], dtype=np.float64).T
    native = grid_to_native[:2] @ corners
    x0 = int(np.floor(native[0].min())) - pad
    y0 = int(np.floor(native[1].min())) - pad
    x1 = int(np.ceil(native[0].max())) + pad + 1
    y1 = int(np.ceil(native[1].max())) + pad + 1
    full_w, full_h = handle.dimensions
    inside = x0 >= 0 and y0 >= 0 and x1 <= full_w and y1 <= full_h
    region = handle.read_region((x0, y0), 0, (x1 - x0, y1 - y0))
    crop = np.asarray(region.convert("RGB"), dtype=np.uint8)
    return crop, (x0, y0), inside


def render_matching(crop: np.ndarray, offset: tuple[int, int], grid_to_native: np.ndarray,
                    width: int, height: int) -> np.ndarray:
    """Geometry-only render (Gaussian pre-filter + bicubic) used for matching."""
    affine = grid_to_native[:2].copy()
    affine[:, 2] -= offset
    scale = float(np.sqrt(abs(np.linalg.det(affine[:, :2]))))
    source = crop.astype(np.float32)
    if scale > 1.05:
        source = cv2.GaussianBlur(source, (0, 0), 0.5 * scale)
    rendered = cv2.warpAffine(source, affine, (width, height),
                              flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REFLECT)
    return np.clip(rendered, 0.0, 255.0)


def ncc_peak(image: np.ndarray, template: np.ndarray) -> tuple[int, int, float, bool]:
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    result = np.nan_to_num(result, nan=-1.0)
    row, col = np.unravel_index(int(np.argmax(result)), result.shape)
    boundary = row in (0, result.shape[0] - 1) or col in (0, result.shape[1] - 1)
    return int(col), int(row), float(result[row, col]), bool(boundary)


def locate(handle: openslide.OpenSlide, native_to_target: np.ndarray,
           top_left: np.ndarray, patch_rgb: np.ndarray, search: int) -> dict:
    """Find the cached patch inside a native render around its registered position."""
    size = PATCH + 2 * search
    grid_to_target = np.array([[1.0, 0.0, top_left[0] - search],
                               [0.0, 1.0, top_left[1] - search], [0.0, 0.0, 1.0]])
    grid_to_native = np.linalg.inv(native_to_target) @ grid_to_target
    crop, offset, inside = read_native_for(handle, grid_to_native, size, size)
    region = blurred_od(render_matching(crop, offset, grid_to_native, size, size))
    template = blurred_od(patch_rgb)
    small_region = cv2.resize(region, (size // COARSE_FACTOR, size // COARSE_FACTOR),
                              interpolation=cv2.INTER_AREA)
    small_template = cv2.resize(template, (PATCH // COARSE_FACTOR, PATCH // COARSE_FACTOR),
                                interpolation=cv2.INTER_AREA)
    cx, cy, coarse_ncc, _ = ncc_peak(small_region, small_template)
    cx, cy = cx * COARSE_FACTOR, cy * COARSE_FACTOR
    fine = 2 * COARSE_FACTOR
    x0, y0 = max(cx - fine, 0), max(cy - fine, 0)
    x1, y1 = min(cx + fine + PATCH, size), min(cy + fine + PATCH, size)
    fx, fy, fine_ncc, _ = ncc_peak(region[y0:y1, x0:x1], template)
    px, py = x0 + fx, y0 + fy
    boundary = px <= 0 or py <= 0 or px >= size - PATCH or py >= size - PATCH
    warp = np.array([[1.0, 0.0, px], [0.0, 1.0, py]], dtype=np.float32)
    try:
        ecc, warp = cv2.findTransformECC(
            template.astype(np.float32), region.astype(np.float32), warp, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error:
        ecc = float("nan")
    patch_to_native = grid_to_native @ to3(warp)
    return {
        "native_inside": inside,
        "coarse_ncc": coarse_ncc,
        "int_dx": px - search,
        "int_dy": py - search,
        "int_ncc": fine_ncc,
        "int_boundary": bool(boundary),
        "ecc": float(ecc),
        "ecc_dx": float(warp[0, 2] - search),
        "ecc_dy": float(warp[1, 2] - search),
        "local_linear": np.asarray(warp[:, :2], dtype=np.float64),
        "patch_to_native": patch_to_native,
    }


def correspondences(result: dict, top_left: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.array([[0, 0], [PATCH - 1, 0], [PATCH - 1, PATCH - 1], [0, PATCH - 1],
                       [PATCH / 2, PATCH / 2]], dtype=np.float64)
    native = (result["patch_to_native"] @ np.c_[points, np.ones(len(points))].T)[:2].T
    return native, points + top_left[None, :]


def fit_affine(native: np.ndarray, target: np.ndarray, groups: np.ndarray):
    """Trimmed least-squares native->target affine; returns matrix, kept groups, rms."""
    keep = np.unique(groups)
    matrix = None
    for _ in range(4):
        selected = np.isin(groups, keep)
        design = np.c_[native[selected], np.ones(int(selected.sum()))]
        coefficients, *_ = np.linalg.lstsq(design, target[selected], rcond=None)
        matrix = to3(coefficients.T)
        residual = np.linalg.norm((np.c_[native, np.ones(len(native))] @ coefficients) - target, axis=1)
        per_group = pd.Series(residual).groupby(groups).max()
        new_keep = per_group.index[per_group <= FIT_TRIM_PX].to_numpy()
        if len(new_keep) < 5 or set(new_keep) == set(keep):
            break
        keep = new_keep
    selected = np.isin(groups, keep)
    residual = np.linalg.norm((np.c_[native, np.ones(len(native))] @ matrix[:2].T) - target, axis=1)
    return matrix, keep, float(np.sqrt(np.mean(residual[selected] ** 2)))


def aligned_correlation(handle: openslide.OpenSlide, patch_to_native: np.ndarray,
                        patch_rgb: np.ndarray) -> float:
    crop, offset, _ = read_native_for(handle, patch_to_native, PATCH, PATCH)
    rendered = od(render_matching(crop, offset, patch_to_native, PATCH, PATCH)).ravel()
    reference = od(patch_rgb).ravel()
    return float(np.corrcoef(rendered, reference)[0, 1])


def slide_task(slide_index: int) -> None:
    cohort = pd.read_csv(COHORT, dtype={"slide_id": str})
    row = cohort.iloc[slide_index]
    slide_id = str(row.slide_id)
    shard_dir = GEOMETRY / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    set20 = pd.read_csv(SET20, dtype={"slide_id": str})
    locations = set20[set20.slide_id == slide_id].sort_values("location_index")
    if len(locations) != 20:
        raise ValueError(f"{slide_id}: expected 20 set20 locations")
    location_index = locations.location_index.to_numpy(dtype=int)
    manifest = pd.read_csv(MANIFEST, dtype={"slide_id": str}, low_memory=False,
                           usecols=lambda c: c in {"slide_id", "scanner", "native_path"}
                           or c.startswith("native_to_target_m"))
    with h5py.File(row.cache_path, "r") as store:
        names = [value.decode() for value in store["scanner_names"][:]]
        if tuple(names) != SCANNERS:
            raise ValueError("unexpected cache scanner order")
        source_index = np.asarray(store["source_index"][:], dtype=np.int64)[location_index]
        final_xy = np.asarray(store["final_coords_xy"][:], dtype=np.float64)[location_index]
        images = np.asarray(store["images"][sorted(location_index.tolist())], dtype=np.uint8)
    if not np.array_equal(source_index, locations.source_index.to_numpy()):
        raise ValueError(f"{slide_id}: set20 source_index disagrees with the cache")

    records, transforms = [], {}
    for scanner_index, scanner in enumerate(SCANNERS):
        path = native_path(scanner, slide_id)
        e0_path = manifest[(manifest.slide_id == slide_id) & (manifest.scanner == scanner)].native_path
        same_file = bool(len(e0_path) and os.path.realpath(path) == os.path.realpath(e0_path.iloc[0]))
        handle = openslide.OpenSlide(str(path))
        m0 = e0_native_to_target(manifest, slide_id, scanner)
        pass1 = [locate(handle, m0, final_xy[k, scanner_index], images[k, scanner_index],
                        PASS1_SEARCH) for k in range(20)]
        native_pts, target_pts, groups = [], [], []
        for k, result in enumerate(pass1):
            if result["ecc"] >= FIT_MIN_CC and not result["int_boundary"]:
                native, target = correspondences(result, final_xy[k, scanner_index])
                native_pts.append(native)
                target_pts.append(target)
                groups.extend([k] * len(native))
        if len(set(groups)) >= 5:
            m1, kept, rms = fit_affine(np.concatenate(native_pts), np.concatenate(target_pts),
                                       np.asarray(groups))
            fit_status = "refit"
        else:
            m1, kept, rms, fit_status = m0, np.array([], dtype=int), float("nan"), "e0_fallback"
        singular = np.linalg.svd(m1[:2, :2], compute_uv=False)
        transforms[scanner] = {
            "native_path": str(path), "native_realpath": os.path.realpath(path),
            "same_file_as_e0_manifest": same_file,
            "e0_native_to_target": m0[:2].tolist(), "refit_native_to_target": m1[:2].tolist(),
            "fit_status": fit_status, "fit_locations": int(len(kept)), "fit_rms_px": rms,
            "refit_singular_values": singular.tolist(),
            "expected_scale": NATIVE_MPP[scanner] / TARGET_MPP,
            "pass1_confident": int(sum(r["ecc"] >= FIT_MIN_CC and not r["int_boundary"] for r in pass1)),
        }
        for k in range(20):
            result = locate(handle, m1, final_xy[k, scanner_index], images[k, scanner_index],
                            PASS2_SEARCH)
            linear = result["local_linear"]
            local_sv = np.linalg.svd(linear, compute_uv=False)
            confirmed = bool(result["ecc"] >= ECC_MIN and not result["int_boundary"])
            correlation = aligned_correlation(handle, result["patch_to_native"],
                                              images[k, scanner_index])
            affine = result["patch_to_native"][:2]
            record = {
                "slide_id": slide_id, "tissue_type": str(row.tissue_type),
                "location_index": int(location_index[k]), "source_index": int(source_index[k]),
                "scanner": scanner, "registered_x": int(final_xy[k, scanner_index, 0]),
                "registered_y": int(final_xy[k, scanner_index, 1]),
                "pass1_int_dx": pass1[k]["int_dx"], "pass1_int_dy": pass1[k]["int_dy"],
                "pass1_ecc": pass1[k]["ecc"],
                "native_inside": result["native_inside"], "coarse_ncc": result["coarse_ncc"],
                "int_dx": result["int_dx"], "int_dy": result["int_dy"],
                "int_ncc": result["int_ncc"], "int_boundary": result["int_boundary"],
                "ecc": result["ecc"], "ecc_dx": result["ecc_dx"], "ecc_dy": result["ecc_dy"],
                "local_sv_max": float(local_sv[0]), "local_sv_min": float(local_sv[1]),
                "local_rotation_deg": float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))),
                "aligned_od_correlation": correlation,
                "confirmed": confirmed,
            }
            record.update({f"patch_to_native_{name}": float(value) for name, value in
                           zip(("a00", "a01", "a02", "a10", "a11", "a12"), affine.ravel())})
            records.append(record)
        handle.close()
        print(json.dumps({"slide_id": slide_id, "scanner": scanner, **{
            key: transforms[scanner][key] for key in ("fit_status", "fit_locations", "fit_rms_px",
                                                      "pass1_confident")}}), flush=True)

    frame = pd.DataFrame(records)
    temporary = shard_dir / f".{slide_id}.csv.{os.getpid()}.tmp"
    frame.to_csv(temporary, index=False)
    temporary.replace(shard_dir / f"{slide_id}.csv")
    payload = {"slide_id": slide_id, "status": "pass", "transforms": transforms,
               "confirmed": frame.groupby("scanner").confirmed.sum().astype(int).to_dict()}
    (shard_dir / f"{slide_id}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["confirmed"]))


def aggregate() -> None:
    cohort = pd.read_csv(COHORT, dtype={"slide_id": str})
    frames, transforms = [], []
    for slide_id in cohort.slide_id:
        path = GEOMETRY / "shards" / f"{slide_id}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path, dtype={"slide_id": str}))
        payload = json.loads(path.with_suffix(".json").read_text())
        for scanner, item in payload["transforms"].items():
            sv = item["refit_singular_values"]
            transforms.append({
                "slide_id": slide_id, "scanner": scanner, "fit_status": item["fit_status"],
                "fit_locations": item["fit_locations"], "fit_rms_px": item["fit_rms_px"],
                "pass1_confident": item["pass1_confident"],
                "same_file_as_e0_manifest": item["same_file_as_e0_manifest"],
                "scale_mean": float(np.mean(sv)), "scale_anisotropy": float(sv[0] / sv[1]),
                "expected_scale": item["expected_scale"],
                "native_realpath": item["native_realpath"],
            })
    table = pd.concat(frames, ignore_index=True)
    transform_table = pd.DataFrame(transforms)
    table.to_csv(GEOMETRY / "location_geometry.csv.gz", index=False)
    transform_table.to_csv(GEOMETRY / "slide_transforms.csv", index=False)

    per_scanner = []
    for scanner in SCANNERS:
        part = table[table.scanner == scanner]
        tf = transform_table[transform_table.scanner == scanner]
        per_scanner.append({
            "scanner": scanner,
            "locations": int(len(part)),
            "confirmed": int(part.confirmed.sum()),
            "confirmed_fraction": float(part.confirmed.mean()),
            "ecc_median": float(part.ecc.median()),
            "ecc_p05": float(part.ecc.quantile(0.05)),
            "aligned_od_correlation_median": float(part.aligned_od_correlation.median()),
            "aligned_od_correlation_p05": float(part.aligned_od_correlation.quantile(0.05)),
            "abs_ecc_shift_p95_px": float(np.hypot(part.ecc_dx, part.ecc_dy).quantile(0.95)),
            "local_scale_deviation_p95": float(
                np.maximum(np.abs(part.local_sv_max - 1), np.abs(part.local_sv_min - 1)).quantile(0.95)),
            "slides_refit": int((tf.fit_status == "refit").sum()),
            "fit_rms_px_median": float(tf.fit_rms_px.median()),
            "fit_rms_px_max": float(tf.fit_rms_px.max()),
            "scale_mean_median": float(tf.scale_mean.median()),
            "expected_scale": float(tf.expected_scale.iloc[0]),
            "scale_relative_error_max": float((tf.scale_mean / tf.expected_scale - 1).abs().max()),
            "same_file_as_e0_all": bool(tf.same_file_as_e0_manifest.all()),
            "native_inside_all": bool(part.native_inside.all()),
        })
    per_scanner = pd.DataFrame(per_scanner)
    per_scanner.to_csv(GEOMETRY / "scanner_summary.csv", index=False)
    targets = per_scanner[per_scanner.scanner.isin(TARGETS)]
    verdict = bool((targets.confirmed_fraction >= STAGE_A_MIN_FRACTION).all())
    summary = {
        "analysis": "RV10 Stage A native-geometry feasibility",
        "slides": int(table.slide_id.nunique()),
        "locations_per_scanner": 20 * int(table.slide_id.nunique()),
        "gate": {"ecc_min": ECC_MIN, "pass_fraction_min": STAGE_A_MIN_FRACTION,
                 "blur_sigma_px": BLUR_SIGMA, "pass2_search_px": PASS2_SEARCH},
        "stage_a_pass": verdict,
        "per_scanner": per_scanner.to_dict(orient="records"),
    }
    (GEOMETRY / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("slide", "aggregate"))
    parser.add_argument("--slide-index", type=int,
                        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")))
    args = parser.parse_args()
    if args.command == "slide":
        slide_task(args.slide_index)
    else:
        aggregate()


if __name__ == "__main__":
    main()
