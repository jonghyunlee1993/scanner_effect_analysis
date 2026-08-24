"""Native-resolution radial power spectrum for one PLISM WSI.

No resampling.  Every patch is read at level 0 on the scanner's own pixel grid by
integer crop, and its spectrum is expressed in physical frequency (cycles per
micrometre) using that scanner's native MPP.  This is the measurement our own
cohort cannot make: there AT2 runs at 0.5052 um/px and GT450 at 0.2624, so a
common grid forces a resampling step onto the exact quantity under study.

Locations come from ``build_plism_section_registration.py``: one set per stain
section, defined on its AT2 reference and mapped outward by a coarse similarity
transform.  Registration decides *where to read* and nothing else -- no analysed
pixel is interpolated.  Choosing locations independently per WSI made tissue
selection scanner-dependent and corrupted the estimate; see Amendment 1.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Patch side in micrometres: 256 px at the study's 0.5052 um/px target grid, so
# the physical field of view matches the ResNet50 / UNI crop while the pixel
# count is whatever the scanner's native grid gives.
PATCH_UM = 256 * 0.5052

# Frozen bands in cycles per micrometre (contract section 2).
BANDS = {
    "low": (0.02, 0.10),
    "high": (0.10, 0.99),
    "extended": (0.99, 1.90),
}

# Radial curve reported for diagnostics, common to every scanner.
CURVE_EDGES = np.arange(0.0, 1.9001, 0.025)

# Tissue mask resolution is fixed in micrometres, not in pixels, so the seven
# scanners of a section mask at the same physical scale despite differing MPP.
MASK_TARGET_UM = 16.0
# Floor on the Otsu threshold, guarding against a slide with no bimodal split.
TISSUE_OD_FLOOR = 0.05


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/PLISM_dataset/manifest_original.csv")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--wsi-dir", default="data/PLISM_dataset/original_wsi")
    parser.add_argument("--output", default="outputs/plism_native_psd")
    parser.add_argument("--refinement", default="outputs/plism_core_refinement")
    parser.add_argument("--patches", type=int, default=0,
                        help="cap on locations, 0 uses every location in the registration")
    return parser.parse_args()


def open_level(path: str, level: int):
    import pyvips

    image = pyvips.Image.new_from_file(path, level=level, access="random")
    if image.hasalpha():
        image = image.flatten(background=255)
    return image


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's between-class variance threshold, implemented here to avoid a new dep."""
    counts, edges = np.histogram(values, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    weight0 = np.cumsum(counts)
    weight1 = counts.sum() - weight0
    valid = (weight0 > 0) & (weight1 > 0)
    cumulative = np.cumsum(counts * centres)
    mean0 = np.divide(cumulative, weight0, out=np.zeros_like(cumulative), where=weight0 > 0)
    mean1 = np.divide(
        cumulative[-1] - cumulative, weight1, out=np.zeros_like(cumulative), where=weight1 > 0
    )
    between = weight0 * weight1 * (mean0 - mean1) ** 2
    between[~valid] = -np.inf
    return float(centres[int(np.argmax(between))])


def radial_power(patch: np.ndarray, mpp: float) -> tuple[np.ndarray, np.ndarray]:
    """Mode power and matching radial frequency for one patch.

    Power is |X|^2 / N^4, which sums to the patch variance and so does not depend
    on the pixel count -- necessary here because N differs by scanner while the
    physical patch size does not.
    """
    rgb = patch.astype(np.float32)
    od = -np.log10(np.clip(rgb, 1.0, 255.0) / 255.0).mean(axis=2)
    od = od - od.mean()

    n = od.shape[0]
    window = np.hanning(n)
    window2d = np.outer(window, window)
    od = od * window2d
    # restore the power the window removed, so patches of different N compare
    od = od / np.sqrt((window2d**2).mean())

    spectrum = np.abs(np.fft.fft2(od)) ** 2 / (n**4)
    freq = np.fft.fftfreq(n, d=mpp)
    radius = np.hypot(*np.meshgrid(freq, freq, indexing="ij"))
    return spectrum.ravel(), radius.ravel()


def summarise(power: np.ndarray, radius: np.ndarray) -> dict:
    """Mean mode power inside each frozen band and each diagnostic radial bin."""
    out = {}
    for name, (lo, hi) in BANDS.items():
        selection = (radius >= lo) & (radius < hi)
        out[name] = float(power[selection].mean()) if selection.any() else np.nan
        out[f"{name}_modes"] = int(selection.sum())

    index = np.digitize(radius, CURVE_EDGES) - 1
    valid = (index >= 0) & (index < len(CURVE_EDGES) - 1)
    totals = np.bincount(index[valid], weights=power[valid], minlength=len(CURVE_EDGES) - 1)
    counts = np.bincount(index[valid], minlength=len(CURVE_EDGES) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["curve"] = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    return out


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    matches = manifest.loc[manifest["task_index"] == args.task_index]
    if len(matches) != 1:
        raise KeyError(f"task index {args.task_index} absent from {args.manifest}")
    row = matches.iloc[0]
    stain, scanner = str(row["stain"]), str(row["scanner"])

    refinement = pd.read_csv(Path(args.refinement) / f"{stain}.csv")
    plan = refinement.loc[refinement["scanner"] == scanner].sort_values("location")
    if plan.empty:
        raise RuntimeError(f"{stain}/{scanner}: no refined locations")
    if args.patches:
        plan = plan.head(args.patches)

    from plism_dataset_corrections import corrected_name

    actual, swapped = corrected_name(stain, scanner, str(row["name"]))
    path = str(Path(args.wsi_dir) / actual)
    stem = str(row["name"]).rsplit(".", 1)[0]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{stem}.npz"

    import pyvips

    probe = pyvips.Image.new_from_file(path, access="sequential")
    mpp_x = float(probe.get("openslide.mpp-x"))
    mpp_y = float(probe.get("openslide.mpp-y"))
    if not np.isfinite(mpp_x) or abs(mpp_x - mpp_y) > 0.01 * mpp_x:
        raise ValueError(f"{stem}: anisotropic or missing MPP ({mpp_x}, {mpp_y})")
    if abs(mpp_x - float(plan["mpp"].iloc[0])) > 1e-9:
        raise ValueError(f"{stem}: MPP disagrees with the refinement table")

    side = int(round(PATCH_UM / mpp_x))

    full = open_level(path, 0)
    # band_per_patch is a float matrix, so the tissue name travels as a code and
    # the names themselves as a separate array.
    tissue_names = sorted(set(plan["tissue_type"].astype(str)))
    tissue_code = {name: index for index, name in enumerate(tissue_names)}
    band_rows = []
    curve_total = np.zeros(len(CURVE_EDGES) - 1)
    curve_count = 0
    skipped = 0

    for record in plan.itertuples():
        if not record.ok:
            skipped += 1
            continue
        x = int(round(record.centre_x)) - side // 2
        y = int(round(record.centre_y)) - side // 2
        if x < 0 or y < 0 or x + side > full.width or y + side > full.height:
            skipped += 1
            continue
        patch = full.crop(x, y, side, side).numpy()
        optical = -np.log10(np.clip(patch.astype(np.float32), 1.0, 255.0) / 255.0).mean(axis=2)
        power, radius = radial_power(patch, mpp_x)
        stats = summarise(power, radius)
        entry = {name: stats[name] for name in BANDS}
        entry["od_mean"] = float(optical.mean())
        entry["od_std"] = float(optical.std())
        # carried so the analysis can gate on alignment and block on core
        entry["residual_um"] = float(record.residual_um)
        entry["response"] = float(record.response)
        entry["core"] = float(record.core)
        entry["tissue"] = float(tissue_code[record.tissue_type])
        entry["replicate"] = float(record.replicate)
        entry["location"] = float(record.location)
        band_rows.append(entry)
        curve = stats["curve"]
        if np.isfinite(curve).all():
            curve_total += curve
            curve_count += 1

    if not band_rows:
        raise RuntimeError(f"{stem}: every location fell outside the slide")

    bands = pd.DataFrame(band_rows)
    np.savez_compressed(
        destination,
        curve_edges=CURVE_EDGES,
        tissue_names=np.array(tissue_names),
        curve_mean=curve_total / max(curve_count, 1),
        curve_patches=curve_count,
        band_means=bands.mean().to_numpy(),
        band_names=np.array(list(bands.columns)),
        band_per_patch=bands.to_numpy(),
        meta=json.dumps(
            {
                "slide": stem,
                "stain": stain,
                "scanner": scanner,
                "native_mpp": mpp_x,
                "patch_px": side,
                "patch_um": PATCH_UM,
                "locations": int(len(plan)),
                "patches": int(len(bands)),
                "skipped_out_of_bounds": int(skipped),
                "file_read": actual,
                "file_swapped": bool(swapped),
                "model": str(plan["model"].iloc[0]),
                "flip": bool(plan["flip"].iloc[0]),
                "residual_median_um": float(plan.loc[plan["ok"], "residual_um"].median()),
                "nyquist_cyc_um": 0.5 / mpp_x,
            }
        ),
    )

    print(
        f"{stem}: scanner={scanner} stain={stain} mpp={mpp_x:.5f} patch={side}px "
        f"nyquist={0.5 / mpp_x:.3f} cyc/um patches={len(bands)} skipped={skipped} "
        f"residual={plan.loc[plan['ok'], 'residual_um'].median():.3f} um"
    )
    for name in BANDS:
        print(f"  {name:9s} {bands[name].mean():.6e}")
    print(f"  od_mean   {bands['od_mean'].mean():.4f}")
    print(f"  -> {destination}")


if __name__ == "__main__":
    main()
