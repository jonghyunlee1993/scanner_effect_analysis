"""Run one slide of the frozen E0 registration/ERT audit.

The audit compares three contracts at the exact Exp05 locations:

* ``current_old_local16``: the zero-centred +/-16 px search used by Exp05;
* ``current_corrected``: a cross-fitted slide prior plus local integer crop;
* ``valis_center`` / ``valis_corrected``: existing VALIS rigid outputs before and
  after the same residual integer-crop refinement.

No sub-pixel or non-rigid warp is introduced here.  The five pre-existing patch
replicate groups are used as folds so a tile never contributes to its own slide prior.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import openslide
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import align, qc


SCANNERS = ("gt450", "versa", "akoya", "s60", "s360")
BANDS = {
    "low_mid": (0.10, 0.30),
    "mid": (0.30, 0.60),
    "high": (0.60, 0.90),
}
THRESHOLDS = {
    "probe_min_ncc": 0.60,
    "tile_min_ncc": 0.50,
    "max_distance_from_prior": 48,
    "max_residual_shift": 8.0,
    "max_pad_fraction": 0.01,
    "min_common_locations": 80,
    "cell_geometry_pass_fraction": 0.90,
    "search_margin": 96,
    "refine_radius": 16,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide-id", required=True)
    parser.add_argument(
        "--registry",
        default=(
            "/mnt/isilon/oldridge_lab/batch_effects/pan_normal/"
            "registered_ref_at2_all/registered_images/qc_registered_eval/"
            "registered_pair_summary.csv"
        ),
    )
    parser.add_argument(
        "--selected-patches",
        default="outputs/exp05_spectral_cohort_109/selected_patches.csv",
    )
    parser.add_argument(
        "--valis-root",
        default=(
            "/mnt/isilon/oldridge_lab/batch_effects/pan_normal/"
            "registered_ref_at2_all/registered_images"
        ),
    )
    parser.add_argument("--output-root", default="outputs/e0_registration_sentinel/shards")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument(
        "--scanners", nargs="+", choices=SCANNERS, default=list(SCANNERS)
    )
    parser.add_argument(
        "--route",
        choices=("both", "current", "valis"),
        default="both",
        help="Audit both routes for sentinels or one route for population expansion.",
    )
    parser.add_argument("--search-margin", type=int, default=THRESHOLDS["search_margin"])
    parser.add_argument("--refine-radius", type=int, default=THRESHOLDS["refine_radius"])
    return parser.parse_args()


def frequency_geometry(size: int, mpp: float, bins: int):
    one_d = np.fft.fftfreq(size) / mpp
    radius = np.sqrt(one_d[:, None] ** 2 + one_d[None, :] ** 2)
    nyquist = 1.0 / (2.0 * mpp)
    edges = np.linspace(0.0, nyquist, bins + 1)
    index = np.digitize(radius.ravel(), edges) - 1
    valid = (index >= 0) & (index < bins)
    counts = np.bincount(index[valid], minlength=bins).astype(np.float64)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return index, valid, counts, centres


def radial_power(rgb, window, index, valid, counts):
    value = -np.log((rgb.astype(np.float32) + 1.0) / 256.0).mean(axis=2)
    value -= value.mean()
    power = np.abs(np.fft.fft2(value * window)) ** 2
    return np.bincount(
        index[valid], weights=power.ravel()[valid], minlength=len(counts)
    ) / np.maximum(counts, 1.0)


def normalized_transfer(power, reference, frequency):
    ratio = np.sqrt(np.maximum(power, 1e-20) / np.maximum(reference, 1e-20))
    anchor = (frequency >= 0.03) & (frequency <= 0.10)
    scale = np.exp(np.mean(np.log(np.maximum(ratio[anchor], 1e-20))))
    return ratio / scale


def geometric_band(curve, frequency, bounds):
    selected = (frequency >= bounds[0]) & (frequency < bounds[1])
    return float(np.exp(np.mean(np.log(np.maximum(curve[selected], 1e-20)))))


def read_ext(handle, x, y, size, margin):
    return np.asarray(
        handle.read_region(
            (int(x) - margin, int(y) - margin),
            0,
            (size + 2 * margin, size + 2 * margin),
        ).convert("RGB"),
        dtype=np.uint8,
    )


def read_center(handle, x, y, size):
    return np.asarray(
        handle.read_region((int(x), int(y)), 0, (size, size)).convert("RGB"),
        dtype=np.uint8,
    )


def valis_path(root: Path, scanner: str, slide_id: str):
    branch = "rigid_akoya" if scanner == "akoya" else "rigid_all"
    return root / branch / scanner / f"{slide_id}.ome.tiff"


def _fit_priors(base: pd.DataFrame, refine_radius: int, margin: int):
    priors = {}
    fits = []
    for fold in range(5):
        train = base[
            base["replicate"].ne(fold)
            & base["global_ncc"].ge(THRESHOLDS["probe_min_ncc"])
            & ~base["global_boundary"]
        ]
        if train.empty:
            median = (0, 0)
            coefficient = None
            fit = {
                "n": 0,
                "r2_dy": np.nan,
                "r2_dx": np.nan,
                "resid_std_dy": np.nan,
                "resid_std_dx": np.nan,
                "affine": False,
            }
        else:
            median = (
                int(np.median(train["global_dy"])),
                int(np.median(train["global_dx"])),
            )
            probe = train[["x", "y", "global_dy", "global_dx", "global_ncc"]].to_numpy(
                dtype=float
            )
            coefficient, fit = align.fit_offset_field(probe, min_n=12, min_r2=0.5)
        for row in base[base["replicate"].eq(fold)].itertuples(index=False):
            dy, dx = align.prior_at(row.x, row.y, median, coefficient)
            clipped_dy = int(np.clip(dy, -margin + refine_radius, margin - refine_radius))
            clipped_dx = int(np.clip(dx, -margin + refine_radius, margin - refine_radius))
            priors[int(row.patch_index)] = (
                clipped_dy,
                clipped_dx,
                bool(clipped_dy != dy or clipped_dx != dx),
            )
        fits.append(
            {
                "fold": fold,
                "n_confident_train": int(len(train)),
                "median_dy": median[0],
                "median_dx": median[1],
                "model": "affine" if fit["affine"] else "constant",
                **fit,
            }
        )
    return priors, fits


def process_branch(
    slide_id,
    scanner,
    branch,
    handle,
    refs,
    ref_powers,
    coords,
    size,
    margin,
    refine_radius,
    spectral_geometry,
    window,
):
    index, valid_frequency, counts, _ = spectral_geometry
    dimensions = handle.dimensions
    ext_by_patch = {}
    nmap_by_patch = {}
    base_rows = []
    for row in coords.itertuples(index=False):
        patch_index = int(row.patch_index)
        x, y = int(row.x), int(row.y)
        in_frame = align.in_bounds(x, y, size, margin, dimensions[0], dimensions[1])
        if not in_frame:
            base_rows.append(
                {
                    "patch_index": patch_index,
                    "x": x,
                    "y": y,
                    "replicate": patch_index % 5,
                    "in_bounds": False,
                    "zero_ncc": np.nan,
                    "global_dy": np.nan,
                    "global_dx": np.nan,
                    "global_ncc": np.nan,
                    "global_boundary": True,
                    "old_dy": np.nan,
                    "old_dx": np.nan,
                    "old_ncc": np.nan,
                    "old_boundary": True,
                }
            )
            continue
        ext = read_ext(handle, x, y, size, margin)
        nmap = align.ncc_map(ext, refs[patch_index])
        gy, gx, gv = align.peak(nmap, margin)
        oy, ox, ov = align.peak(nmap, margin, around=(0, 0), radius=16)
        ext_by_patch[patch_index] = ext
        nmap_by_patch[patch_index] = nmap
        base_rows.append(
            {
                "patch_index": patch_index,
                "x": x,
                "y": y,
                "replicate": patch_index % 5,
                "in_bounds": True,
                "zero_ncc": float(nmap[margin, margin]),
                "global_dy": gy,
                "global_dx": gx,
                "global_ncc": gv,
                "global_boundary": bool(abs(gy) >= margin or abs(gx) >= margin),
                "old_dy": oy,
                "old_dx": ox,
                "old_ncc": ov,
                "old_boundary": bool(abs(oy) >= 16 or abs(ox) >= 16),
            }
        )
    base = pd.DataFrame(base_rows)
    trainable = base[base["in_bounds"]].copy()
    priors, fits = _fit_priors(trainable, refine_radius, margin)

    alignment_rows = []
    powers = {
        "center": np.full((len(coords), len(counts)), np.nan, dtype=np.float64),
        "old_local16": np.full((len(coords), len(counts)), np.nan, dtype=np.float64),
        "corrected": np.full((len(coords), len(counts)), np.nan, dtype=np.float64),
    }
    condition_valid = {
        key: np.zeros(len(coords), dtype=bool) for key in powers
    }
    for row in base.itertuples(index=False):
        patch_index = int(row.patch_index)
        common = {
            "slide_id": slide_id,
            "scanner": scanner,
            "branch": branch,
            "patch_index": patch_index,
            "x": int(row.x),
            "y": int(row.y),
            "replicate": int(row.replicate),
            "in_bounds": bool(row.in_bounds),
            "zero_ncc": row.zero_ncc,
            "global_dy": row.global_dy,
            "global_dx": row.global_dx,
            "global_ncc": row.global_ncc,
            "global_boundary": bool(row.global_boundary),
            "old_dy": row.old_dy,
            "old_dx": row.old_dx,
            "old_ncc": row.old_ncc,
            "old_boundary": bool(row.old_boundary),
        }
        if not row.in_bounds:
            alignment_rows.append(
                {
                    **common,
                    "prior_dy": np.nan,
                    "prior_dx": np.nan,
                    "prior_clipped": False,
                    "corrected_dy": np.nan,
                    "corrected_dx": np.nan,
                    "corrected_ncc": np.nan,
                    "corrected_mode": "out_of_bounds",
                    "global_far_from_prior": False,
                    "corrected_boundary": True,
                    "q_reg": np.nan,
                    "ncc_lp": np.nan,
                    "phase_response": np.nan,
                    "residual_shift": np.nan,
                    "pad_fraction": np.nan,
                    "content_ok": False,
                    "geometry_pass": False,
                }
            )
            continue

        ext = ext_by_patch[patch_index]
        nmap = nmap_by_patch[patch_index]
        prior_dy, prior_dx, prior_clipped = priors[patch_index]
        global_far = bool(
            abs(int(row.global_dy) - prior_dy) + abs(int(row.global_dx) - prior_dx)
            > THRESHOLDS["max_distance_from_prior"]
        )
        if row.global_ncc >= THRESHOLDS["tile_min_ncc"] and not global_far:
            dy, dx, score = int(row.global_dy), int(row.global_dx), float(row.global_ncc)
            corrected_mode = "global"
            corrected_boundary = bool(abs(dy) >= margin or abs(dx) >= margin)
        else:
            dy, dx, score = align.peak(
                nmap,
                margin,
                around=(prior_dy, prior_dx),
                radius=refine_radius,
            )
            corrected_mode = "local"
            corrected_boundary = bool(
                abs(dy) >= margin
                or abs(dx) >= margin
                or abs(dy - prior_dy) >= refine_radius
                or abs(dx - prior_dx) >= refine_radius
            )
        crops = {
            "center": align.crop_at(ext, 0, 0, size, margin),
            "old_local16": align.crop_at(
                ext, int(row.old_dy), int(row.old_dx), size, margin
            ),
            "corrected": align.crop_at(ext, dy, dx, size, margin),
        }
        quality = qc.registration_quality(refs[patch_index], crops["corrected"], sigma=3.0)
        content_ok, pad_fraction = align.content_ok(
            crops["corrected"],
            refs[patch_index],
            max_pad=THRESHOLDS["max_pad_fraction"],
        )
        geometry_pass = bool(
            content_ok
            and not corrected_boundary
            and quality["residual_shift"] <= THRESHOLDS["max_residual_shift"]
        )
        alignment_rows.append(
            {
                **common,
                "prior_dy": prior_dy,
                "prior_dx": prior_dx,
                "prior_clipped": prior_clipped,
                "corrected_dy": dy,
                "corrected_dx": dx,
                "corrected_ncc": score,
                "corrected_mode": corrected_mode,
                "global_far_from_prior": global_far,
                "corrected_boundary": corrected_boundary,
                **quality,
                "pad_fraction": pad_fraction,
                "content_ok": content_ok,
                "geometry_pass": geometry_pass,
            }
        )
        for condition, crop in crops.items():
            powers[condition][patch_index] = radial_power(
                crop, window, index, valid_frequency, counts
            )
            if condition == "corrected":
                condition_valid[condition][patch_index] = geometry_pass
            else:
                ok, pad = align.content_ok(
                    crop, refs[patch_index], max_pad=THRESHOLDS["max_pad_fraction"]
                )
                condition_valid[condition][patch_index] = bool(ok and pad <= 0.01)

    fit_rows = [
        {"slide_id": slide_id, "scanner": scanner, "branch": branch, **row}
        for row in fits
    ]
    return pd.DataFrame(alignment_rows), pd.DataFrame(fit_rows), powers, condition_valid


def main():
    args = parse_args()
    slide_id = str(args.slide_id)
    registry = pd.read_csv(args.registry, dtype={"slide_id": str})
    selected = registry[registry["slide_id"].eq(slide_id)]
    if len(selected) != 1:
        raise ValueError(f"expected one registry row for {slide_id}, got {len(selected)}")
    registry_row = selected.iloc[0]
    coords = pd.read_csv(args.selected_patches, dtype={"slide_id": str})
    if "patch_index" not in coords.columns:
        if "location_id" not in coords.columns:
            raise ValueError("coordinate manifest needs patch_index or location_id")
        coords = coords.copy()
        coords["patch_index"] = coords["location_id"].astype(int)
    coords = coords[coords["slide_id"].eq(slide_id)].sort_values("patch_index")
    if len(coords) != 100 or coords["patch_index"].tolist() != list(range(100)):
        raise ValueError(f"{slide_id}: expected patch_index 0..99, got {len(coords)} rows")

    size = int(args.patch_size)
    margin = int(args.search_margin)
    refine_radius = int(args.refine_radius)
    ref_handle = openslide.OpenSlide(str(registry_row["at2_path"]))
    ref_dimensions = ref_handle.dimensions
    mpp = float(ref_handle.properties["openslide.mpp-x"])
    spectral_geometry = frequency_geometry(size, mpp, int(args.bins))
    frequency = spectral_geometry[3]
    window_1d = np.hanning(size).astype(np.float32)
    window = window_1d[:, None] * window_1d[None, :]
    refs = {
        int(row.patch_index): read_center(ref_handle, int(row.x), int(row.y), size)
        for row in coords.itertuples(index=False)
    }
    ref_powers = np.stack(
        [
            radial_power(
                refs[index],
                window,
                spectral_geometry[0],
                spectral_geometry[1],
                spectral_geometry[2],
            )
            for index in range(100)
        ]
    )
    ref_handle.close()

    valis_root = Path(args.valis_root)
    all_alignment = []
    all_fits = []
    spectra_rows = []
    requested_branches = (
        ("current", "valis") if args.route == "both" else (args.route,)
    )
    for scanner in args.scanners:
        available_paths = {
            "current": Path(registry_row[f"{scanner}_path"]),
            "valis": valis_path(valis_root, scanner, slide_id),
        }
        paths = {branch: available_paths[branch] for branch in requested_branches}
        branch_result = {}
        for branch, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(path)
            handle = openslide.OpenSlide(str(path))
            if handle.dimensions != ref_dimensions:
                handle.close()
                raise ValueError(f"dimension mismatch: {slide_id}/{scanner}/{branch}")
            alignment_frame, fit_frame, powers, valid_conditions = process_branch(
                slide_id,
                scanner,
                branch,
                handle,
                refs,
                ref_powers,
                coords,
                size,
                margin,
                refine_radius,
                spectral_geometry,
                window,
            )
            handle.close()
            all_alignment.append(alignment_frame)
            all_fits.append(fit_frame)
            branch_result[branch] = (powers, valid_conditions)

        condition_map = {}
        if "current" in branch_result:
            current_powers, current_valid = branch_result["current"]
            condition_map.update(
                {
                    "current_center": (
                        current_powers["center"],
                        current_valid["center"],
                        current_valid["corrected"],
                    ),
                    "current_old_local16": (
                        current_powers["old_local16"],
                        current_valid["old_local16"],
                        current_valid["corrected"],
                    ),
                    "current_corrected": (
                        current_powers["corrected"],
                        current_valid["corrected"],
                        current_valid["corrected"],
                    ),
                }
            )
        if "valis" in branch_result:
            valis_powers, valis_valid = branch_result["valis"]
            condition_map.update(
                {
                    "valis_center": (
                        valis_powers["center"],
                        valis_valid["center"],
                        valis_valid["corrected"],
                    ),
                    "valis_corrected": (
                        valis_powers["corrected"],
                        valis_valid["corrected"],
                        valis_valid["corrected"],
                    ),
                }
            )
        if args.route == "both":
            common = current_valid["corrected"] & valis_valid["corrected"]
            condition_map.update(
                {
                    "current_corrected_route_common": (
                        current_powers["corrected"],
                        current_valid["corrected"],
                        common,
                    ),
                    "valis_corrected_route_common": (
                        valis_powers["corrected"],
                        valis_valid["corrected"],
                        common,
                    ),
                }
            )
        for condition, (power, own_valid, comparison_mask) in condition_map.items():
            mask = comparison_mask & own_valid
            if not mask.any():
                continue
            transfer = normalized_transfer(
                np.nanmean(power[mask], axis=0),
                np.nanmean(ref_powers[mask], axis=0),
                frequency,
            )
            for band, bounds in BANDS.items():
                value = geometric_band(transfer, frequency, bounds)
                spectra_rows.append(
                    {
                        "slide_id": slide_id,
                        "scanner": scanner,
                        "condition": condition,
                        "band": band,
                        "n_common": int(mask.sum()),
                        "relative_transfer": value,
                        "log2_relative_transfer": float(np.log2(value)),
                    }
                )

    alignment_frame = pd.concat(all_alignment, ignore_index=True)
    fit_frame = pd.concat(all_fits, ignore_index=True)
    spectral_frame = pd.DataFrame(spectra_rows)
    alignment_summary = (
        alignment_frame.groupby(["slide_id", "scanner", "branch"], as_index=False)
        .agg(
            n_locations=("patch_index", "size"),
            n_in_bounds=("in_bounds", "sum"),
            zero_ncc_median=("zero_ncc", "median"),
            global_ncc_median=("global_ncc", "median"),
            global_boundary_fraction=("global_boundary", "mean"),
            old_ncc_median=("old_ncc", "median"),
            old_boundary_fraction=("old_boundary", "mean"),
            corrected_ncc_median=("corrected_ncc", "median"),
            corrected_boundary_fraction=("corrected_boundary", "mean"),
            residual_shift_median=("residual_shift", "median"),
            residual_shift_q95=("residual_shift", lambda value: value.quantile(0.95)),
            pad_fraction_max=("pad_fraction", "max"),
            content_pass_fraction=("content_ok", "mean"),
            geometry_pass_fraction=("geometry_pass", "mean"),
        )
    )
    alignment_summary["cell_pass"] = (
        alignment_summary["n_in_bounds"].ge(THRESHOLDS["min_common_locations"])
        & alignment_summary["geometry_pass_fraction"].ge(
            THRESHOLDS["cell_geometry_pass_fraction"]
        )
    )

    output = Path(args.output_root) / slide_id
    output.mkdir(parents=True, exist_ok=True)
    alignment_frame.to_csv(output / "alignment_patch_metrics.csv", index=False)
    alignment_summary.to_csv(output / "alignment_summary.csv", index=False)
    fit_frame.to_csv(output / "prior_fits.csv", index=False)
    spectral_frame.to_csv(output / "spectral_bands.csv", index=False)
    summary = {
        "analysis": "e0_registration_sentinel",
        "slide_id": slide_id,
        "patches": int(len(coords)),
        "scanners": list(args.scanners),
        "thresholds": {
            **THRESHOLDS,
            "search_margin": margin,
            "refine_radius": refine_radius,
        },
        "integer_alignment_only": True,
        "crossfit_folds": 5,
        "route": args.route,
        "files": {
            "alignment_patch_metrics": "alignment_patch_metrics.csv",
            "alignment_summary": "alignment_summary.csv",
            "prior_fits": "prior_fits.csv",
            "spectral_bands": "spectral_bands.csv",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(alignment_summary.to_string(index=False))
    print(f"wrote E0 registration audit shard -> {output}")


if __name__ == "__main__":
    main()
