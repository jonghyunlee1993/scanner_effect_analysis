#!/usr/bin/env python3
"""RV10 Stage B: re-render set20 target patches from native WSIs with three resamplers.

Geometry comes from Stage A (``resampling_geometry.py``): for every confirmed location
and scanner, an affine that maps registered-patch pixel centres (u, v) to native level-0
pixel centres.  The same geometry is used for every variant, so variants differ only in
the resampling kernel:

``bicubic``     point sampling at the mapped centre with the Catmull-Rom cubic
                (a = -0.5, the libvips ``bicubic`` interpolator); no pre-filter.
``lanczos_aa``  Lanczos-3 kernel defined in output-pixel units (support widened by the
                native/output scale), normalised; the anti-aliased reduction.
``box``         area average of the piecewise-constant native image over the output
                pixel footprint (the AT2 pixel square mapped to native), by 16 x 16
                supersampling.

AT2 is read natively at the registered coordinates (scale 1, integer shift: all three
variants reduce to a copy) and checked against the cached AT2 patch.

Per patch the shard stores the rendered images, the radial OD power spectrum (the
paper's Hann-windowed, mean-removed mean-OD FFT) and the similarity of each variant to
the registered pipeline patch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from final_image_study import (  # noqa: E402
    BANDS_CYC_PER_UM, PROVISIONAL_MPP, SPECTRAL_BINS, frequency_geometry, radial_mean)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resampling_geometry import (  # noqa: E402
    COHORT, GEOMETRY, OUTPUT, PATCH, SCANNERS, TARGETS, native_path, od)


VARIANTS = ("bicubic", "lanczos_aa", "box")
RENDER = OUTPUT / "render"
BOX_SUPERSAMPLE = 16
LANCZOS_A = 3.0
CUBIC_A = -0.5
NATIVE_MARGIN = 4


def cubic(x: np.ndarray, a: float = CUBIC_A) -> np.ndarray:
    x = np.abs(x)
    return np.where(
        x <= 1.0, (a + 2.0) * x**3 - (a + 3.0) * x**2 + 1.0,
        np.where(x < 2.0, a * x**3 - 5.0 * a * x**2 + 8.0 * a * x - 4.0 * a, 0.0))


def lanczos(x: np.ndarray, a: float = LANCZOS_A) -> np.ndarray:
    return np.where(np.abs(x) < a, np.sinc(x) * np.sinc(x / a), 0.0)


def patch_grid() -> tuple[np.ndarray, np.ndarray]:
    v, u = np.mgrid[0:PATCH, 0:PATCH].astype(np.float64)
    return u.ravel(), v.ravel()


def gather(crop: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    if ix.min() < 0 or iy.min() < 0 or ix.max() >= crop.shape[1] or iy.max() >= crop.shape[0]:
        raise IndexError("resampling support leaves the native crop")
    return crop[iy, ix]


def resample(crop: np.ndarray, affine: np.ndarray, variant: str) -> np.ndarray:
    """Render a 256x256 patch; ``affine`` maps patch (u, v) to crop pixel centres."""
    image = crop.astype(np.float64)
    linear, shift = affine[:, :2], affine[:, 2]
    u, v = patch_grid()
    px = linear[0, 0] * u + linear[0, 1] * v + shift[0]
    py = linear[1, 0] * u + linear[1, 1] * v + shift[1]
    out = np.zeros((u.size, 3), dtype=np.float64)
    if variant == "bicubic":
        ix0, iy0 = np.floor(px).astype(int), np.floor(py).astype(int)
        fx, fy = px - ix0, py - iy0
        for dy in range(-1, 3):
            wy = cubic(dy - fy)
            for dx in range(-1, 3):
                w = wy * cubic(dx - fx)
                out += w[:, None] * gather(image, ix0 + dx, iy0 + dy)
    elif variant == "lanczos_aa":
        inverse = np.linalg.inv(linear)
        # Kernel support is |du|, |dv| < 3 output pixels; bound it in native pixels.
        half = int(np.ceil(LANCZOS_A * np.abs(linear).sum(axis=1).max())) + 1
        ix0, iy0 = np.rint(px).astype(int), np.rint(py).astype(int)
        weight_sum = np.zeros(u.size, dtype=np.float64)
        for dy in range(-half, half + 1):
            qy = iy0 + dy - py
            for dx in range(-half, half + 1):
                qx = ix0 + dx - px
                du = inverse[0, 0] * qx + inverse[0, 1] * qy
                dv = inverse[1, 0] * qx + inverse[1, 1] * qy
                w = lanczos(du) * lanczos(dv)
                if not np.any(w):
                    continue
                out += w[:, None] * gather(image, ix0 + dx, iy0 + dy)
                weight_sum += w
        out /= weight_sum[:, None]
    elif variant == "box":
        offsets = (np.arange(BOX_SUPERSAMPLE) + 0.5) / BOX_SUPERSAMPLE - 0.5
        for b in offsets:
            for a in offsets:
                sx = px + linear[0, 0] * a + linear[0, 1] * b
                sy = py + linear[1, 0] * a + linear[1, 1] * b
                out += gather(image, np.rint(sx).astype(int), np.rint(sy).astype(int))
        out /= BOX_SUPERSAMPLE**2
    else:
        raise ValueError(variant)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8).reshape(PATCH, PATCH, 3)


def read_crop(handle: openslide.OpenSlide, patch_to_native: np.ndarray, half_support: float):
    corners = np.array([[-0.5, -0.5, 1], [PATCH - 0.5, -0.5, 1], [PATCH - 0.5, PATCH - 0.5, 1],
                        [-0.5, PATCH - 0.5, 1]], dtype=np.float64).T
    native = (patch_to_native @ corners)[:2]
    pad = int(np.ceil(half_support)) + NATIVE_MARGIN
    x0 = int(np.floor(native[0].min())) - pad
    y0 = int(np.floor(native[1].min())) - pad
    x1 = int(np.ceil(native[0].max())) + pad + 1
    y1 = int(np.ceil(native[1].max())) + pad + 1
    width, height = handle.dimensions
    inside = x0 >= 0 and y0 >= 0 and x1 <= width and y1 <= height
    region = handle.read_region((x0, y0), 0, (x1 - x0, y1 - y0))
    crop = np.asarray(region.convert("RGB"), dtype=np.uint8)
    affine = patch_to_native[:2].copy()
    affine[:, 2] -= (x0, y0)
    return crop, affine, inside


class Spectrum:
    """The paper's per-patch spectrum: mean-removed mean-OD, Hann window, radial power."""

    def __init__(self) -> None:
        self.index, self.valid, self.counts, self.frequency_px = frequency_geometry(PATCH, SPECTRAL_BINS)
        self.frequency_um = self.frequency_px / PROVISIONAL_MPP
        window = np.hanning(PATCH)
        self.window = window[:, None] * window[None, :]
        self.one_d = np.fft.fftfreq(PATCH)
        radius = np.sqrt(self.one_d[:, None] ** 2 + self.one_d[None, :] ** 2) / PROVISIONAL_MPP
        low, high = BANDS_CYC_PER_UM["high"]
        self.high_mask = (radius >= low) & (radius < high)

    def fft(self, rgb: np.ndarray) -> np.ndarray:
        value = od(rgb).astype(np.float64)
        value -= value.mean()
        return np.fft.fft2(value * self.window)

    def radial(self, spectrum: np.ndarray) -> np.ndarray:
        return radial_mean(np.abs(spectrum) ** 2, self.index, self.valid, self.counts)


def similarity(candidate: np.ndarray, reference: np.ndarray, spectrum: Spectrum,
               reference_fft: np.ndarray) -> dict:
    a, b = od(candidate).ravel().astype(np.float64), od(reference).ravel().astype(np.float64)
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    candidate_fft = spectrum.fft(candidate)
    residual = np.abs(candidate_fft - reference_fft) ** 2
    reference_power = np.abs(reference_fft) ** 2
    candidate_power = np.abs(candidate_fft) ** 2
    mask = spectrum.high_mask
    return {
        "od_correlation": float(np.corrcoef(a, b)[0, 1]),
        "rgb_rmse": float(np.sqrt(np.mean(difference**2))),
        "rgb_mae": float(np.mean(np.abs(difference))),
        "high_log2_amplitude_ratio_to_registered": float(
            0.5 * np.log2(candidate_power[mask].sum() / reference_power[mask].sum())),
        "high_residual_fraction": float(residual[mask].sum() / reference_power[mask].sum()),
        "all_residual_fraction": float(residual.sum() / reference_power.sum()),
    }


def slide_task(slide_index: int) -> None:
    cohort = pd.read_csv(COHORT, dtype={"slide_id": str})
    row = cohort.iloc[slide_index]
    slide_id = str(row.slide_id)
    geometry = pd.read_csv(GEOMETRY / "shards" / f"{slide_id}.csv", dtype={"slide_id": str})
    location_index = np.sort(geometry.location_index.unique())
    if len(location_index) != 20:
        raise ValueError(f"{slide_id}: Stage A geometry does not hold 20 locations")
    with h5py.File(row.cache_path, "r") as store:
        cached = np.asarray(store["images"][location_index.tolist()], dtype=np.uint8)
        final_xy = np.asarray(store["final_coords_xy"][:], dtype=np.int64)[location_index]
    spectrum = Spectrum()
    n = len(location_index)
    rendered = np.zeros((n, len(TARGETS), len(VARIANTS), PATCH, PATCH, 3), dtype=np.uint8)
    native_at2 = np.zeros((n, PATCH, PATCH, 3), dtype=np.uint8)
    rendered_ok = np.zeros((n, len(TARGETS)), dtype=bool)
    power = {name: np.full((n, len(SCANNERS), SPECTRAL_BINS), np.nan) for name in
             ("pipeline", "native_at2", *VARIANTS)}
    rows = []

    at2 = openslide.OpenSlide(str(native_path("at2", slide_id)))
    for k in range(n):
        x, y = final_xy[k, 0]
        native_at2[k] = np.asarray(at2.read_region((int(x), int(y)), 0, (PATCH, PATCH)).convert("RGB"))
        for scanner_index in range(len(SCANNERS)):
            power["pipeline"][k, scanner_index] = spectrum.radial(spectrum.fft(cached[k, scanner_index]))
        at2_power = spectrum.radial(spectrum.fft(native_at2[k]))
        power["native_at2"][k, 0] = at2_power
        for variant in VARIANTS:
            power[variant][k, 0] = at2_power
    at2.close()
    at2_identical = bool(np.array_equal(native_at2, cached[:, 0]))
    at2_max_abs = int(np.abs(native_at2.astype(int) - cached[:, 0].astype(int)).max())

    for target_offset, scanner in enumerate(TARGETS):
        scanner_index = SCANNERS.index(scanner)
        handle = openslide.OpenSlide(str(native_path(scanner, slide_id)))
        part = geometry[geometry.scanner == scanner].set_index("location_index")
        for k, location in enumerate(location_index):
            record = part.loc[int(location)]
            base = {"slide_id": slide_id, "tissue_type": str(row.tissue_type),
                    "location_index": int(location), "scanner": scanner,
                    "confirmed": bool(record.confirmed), "ecc": float(record.ecc)}
            if not bool(record.confirmed):
                rows.append({**base, "variant": "none"})
                continue
            patch_to_native = np.array([[record.patch_to_native_a00, record.patch_to_native_a01,
                                         record.patch_to_native_a02],
                                        [record.patch_to_native_a10, record.patch_to_native_a11,
                                         record.patch_to_native_a12]], dtype=np.float64)
            scale = float(np.abs(patch_to_native[:, :2]).sum(axis=1).max())
            crop, affine, inside = read_crop(handle, np.vstack([patch_to_native, [0, 0, 1]]),
                                             LANCZOS_A * scale + 2)
            if not inside:
                rows.append({**base, "variant": "outside_native"})
                continue
            reference = cached[k, scanner_index]
            reference_fft = spectrum.fft(reference)
            for variant_index, variant in enumerate(VARIANTS):
                image = resample(crop, affine, variant)
                rendered[k, target_offset, variant_index] = image
                power[variant][k, scanner_index] = spectrum.radial(spectrum.fft(image))
                rows.append({**base, "variant": variant, "native_scale": scale,
                             **similarity(image, reference, spectrum, reference_fft)})
            rendered_ok[k, target_offset] = True
        handle.close()
        print(json.dumps({"slide_id": slide_id, "scanner": scanner,
                          "rendered": int(rendered_ok[:, target_offset].sum())}), flush=True)

    out_dir = RENDER / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    temporary = out_dir / f".{slide_id}.csv.{os.getpid()}.tmp"
    frame.to_csv(temporary, index=False)
    temporary.replace(out_dir / f"{slide_id}.csv")
    h5_path = out_dir / f"{slide_id}.h5"
    temporary = out_dir / f".{slide_id}.h5.{os.getpid()}.tmp"
    with h5py.File(temporary, "w") as store:
        store.attrs["slide_id"] = slide_id
        store.attrs["tissue_type"] = str(row.tissue_type)
        store.attrs["at2_native_identical_to_cache"] = at2_identical
        store.create_dataset("location_index", data=location_index)
        store.create_dataset("target_scanners", data=np.asarray(TARGETS, dtype="S8"))
        store.create_dataset("variants", data=np.asarray(VARIANTS, dtype="S16"))
        store.create_dataset("rendered", data=rendered, compression="gzip", compression_opts=4,
                             chunks=(1, 1, 1, PATCH, PATCH, 3))
        store.create_dataset("rendered_ok", data=rendered_ok)
        store.create_dataset("native_at2", data=native_at2, compression="gzip", compression_opts=4)
        store.create_dataset("scanner_names", data=np.asarray(SCANNERS, dtype="S8"))
        store.create_dataset("frequency_cyc_per_um", data=spectrum.frequency_um)
        for name, values in power.items():
            store.create_dataset(f"radial_power/{name}", data=values)
    temporary.replace(h5_path)
    summary = {"slide_id": slide_id, "status": "pass", "at2_native_identical_to_cache": at2_identical,
               "at2_native_max_abs_difference": at2_max_abs,
               "rendered": {scanner: int(rendered_ok[:, i].sum()) for i, scanner in enumerate(TARGETS)}}
    (out_dir / f"{slide_id}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide-index", type=int,
                        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")))
    slide_task(parser.parse_args().slide_index)


if __name__ == "__main__":
    main()
