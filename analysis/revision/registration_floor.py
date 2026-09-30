#!/usr/bin/env python3
"""RV08: registration noise floor of PFM, SSIM and LPIPS distances.

Question: how much target distance remains when two images differ only by a
misregistration of the size allowed by quality control?  For every ``set20``
location the cached AT2 patch is compared with the same AT2 field sampled
0.25 and 0.5 px away (ideal Fourier shift of a padded WSI window, centre-cropped
so that wrap-around never reaches the patch) and 1 and 2 px away (read directly
from the WSI), in four directions (+x, -x, +y, -y).

Coordinate convention (src/preprocessing_patch_extraction_matching_and_QC.py):
``final_coords_xy[:, 0]`` is the AT2 level-0 top-left (x, y) handed to
``openslide.read_region(..., level=0, size=(256, 256))`` on the registered AT2
image.  A shift (dx, dy) samples the field at (x + dx, y + dy); image content
therefore moves by (-dx, -dy).

Distances: cosine distance in UNI v1, UNI2-h, Virchow2 and H-optimus-1 with the
paper's embedding code (scripts/features/review_multiencoder_scanner.load_encoder
and review_feature_crossencoder_extract.images_to_features); SSIM and
LPIPS-VGG16 with the Table 2 contract (scanner_gan.evaluate_images.image_metrics
and analysis/paper/table2_benchmark.lpips_* on the 252 x 252 interior, i.e. the
2-pixel boundary excluded).

Stages (submitted from the repository root, see the matching sbatch files):
  identity   CPU gate. Re-reads each set20 AT2 patch from the registered AT2 WSI
             at the stored coordinates and requires bit-identity with the paired
             cache. Exits non-zero otherwise, so the dependent stages never run.
  embed      GPU array. Shifted images, image QC, SSIM, LPIPS, four PFMs.
  aggregate  CPU. Slide means, slide bootstrap CIs, floor as a fraction of the
             raw AT2-target distance, fraction of achievable reduction,
             summary.md.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results/registration_floor"
SET20 = HERE / "results/location_sets/set20.csv"
STUDY = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17"
COHORT = STUDY / "00_contract/cohort.csv"
FOLDS = PROJECT / "outputs/augmentation_ood_v1/00_contract/slide_folds.csv"
PAN_NORMAL = Path("/mnt/isilon/oldridge_lab/batch_effects/pan_normal")
REGISTERED_AT2 = PAN_NORMAL / "registered_ref_at2_all/registered_images/at2"
RAW_AT2 = PAN_NORMAL / "raw/raw_images/at2"
HF_CACHE = PROJECT / "outputs/encoder_review_2026-09-25/hf_cache"
STORED_UNI = STUDY / "03_uni/shards"
STORED_OTHER = PROJECT / "outputs/feature_crossencoder_review_2026-09-25/raw/internal"
LPIPS_CONTRACT = STUDY / "12_manuscript_completion/09_table2_benchmark/lpips_contract.json"
TABLE4_UNITS = (PROJECT / "outputs/table4_color_frequency_crossencoder_2026-09-25/"
                "pannormal_unit_distances_and_gains.csv.gz")
GAN_SLIDES = PROJECT / "outputs/gan_encoder_review_2026-09-25/gan_per_slide.csv"
FEATURE_UNI = PROJECT / "outputs/discussion_followup_2026-09-25/feature40_forward/per_slide.csv"
FEATURE_OTHER = PROJECT / "outputs/feature_crossencoder_review_2026-09-25/results"
TABLE2_SHARDS = STUDY / "12_manuscript_completion/09_table2_benchmark/pannormal/shards"
TABLE2_BASELINE = PROJECT / "analysis/paper/table2_baseline_summary.csv"

PATCH = 256
PAD = 128          # Fourier window margin on every side (window 512 x 512)
PAD_CHECK = 64     # smaller margin, used only to show the margin does not matter
BORDER = 2         # Table 2 boundary exclusion for SSIM and LPIPS
SUBPIXEL = (0.25, 0.5)
INTEGER = (1.0, 2.0)
SHIFTS = SUBPIXEL + INTEGER
DIRECTIONS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
CONDITIONS = [("original", "cache", 0.0, "none"), ("zero", "fourier", 0.0, "none")] + [
    (f"{'fourier' if shift in SUBPIXEL else 'wsi'}_{shift:g}px_{direction}",
     "fourier" if shift in SUBPIXEL else "wsi", shift, direction)
    for shift in SHIFTS for direction in DIRECTIONS
]
PFMS = ("uni_v1", "uni2", "virchow2", "hoptimus1")
PFM_LABELS = {"uni_v1": "UNI v1", "uni2": "UNI2-h", "virchow2": "Virchow2",
              "hoptimus1": "H-optimus-1"}
IMAGE_METRICS = ("ssim", "lpips_vgg", "l1_unit", "psnr_db")
METRICS = PFMS + IMAGE_METRICS
TARGETS = ("versa", "akoya", "gt450", "s360", "s60")
METHODS = ("reinhard", "combined", "pix2pix", "cyclegan", "featmap_ridge", "combat")
METHOD_LABELS = {"raw": "Raw", "reinhard": "Reinhard", "combined": "Color + frequency",
                 "pix2pix": "Pix2Pix", "cyclegan": "CycleGAN",
                 "featmap_ridge": "Ridge affine", "combat": "ComBat"}
# PanNormal rows of 00_manuscript/tables/table_cross_pfm_correction.tex (used only to
# check that the per-slide evidence files reproduce the published table).
TABLE_PANNORMAL = {
    "raw": (0.2136, 0.2795, 0.0928, 0.1434),
    "reinhard": (0.1798, 0.2476, 0.0776, 0.1341),
    "combined": (0.1794, 0.2328, 0.0730, 0.1350),
    "pix2pix": (0.2704, 0.2611, 0.2369, 0.1914),
    "cyclegan": (0.2015, 0.1893, 0.1212, 0.1618),
    "featmap_ridge": (0.1403, 0.1776, 0.0744, 0.1120),
    "combat": (0.1546, 0.2208, 0.0679, 0.1074),
}
FAR_FLOOR_SHIFT = 1.0
BOOTSTRAP = 2000
SEED = 20260929
PARITY_MIN = 0.999


# --------------------------------------------------------------------------- utils
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(json_ready(value), indent=2) + "\n")
    temporary.replace(path)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False,
                 compression="gzip" if path.suffix == ".gz" else None)
    temporary.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def load_locations() -> tuple[pd.DataFrame, list[str]]:
    """Locked set20 locations with cache paths, in cohort order."""
    set20 = pd.read_csv(SET20, dtype={"slide_id": str})
    cohort = pd.read_csv(COHORT, dtype={"slide_id": str}).sort_values("array_index")
    folds = pd.read_csv(FOLDS, dtype={"slide_id": str}).set_index("slide_id")["fold"]
    order = cohort.slide_id.tolist()
    if len(set20) != 2060 or set20.slide_id.nunique() != 103:
        raise ValueError("set20 must hold 2,060 locations on 103 slides")
    if not set20.groupby("slide_id").size().eq(20).all():
        raise ValueError("set20 must hold 20 locations per slide")
    if set(order) != set(set20.slide_id) or len(order) != 103:
        raise ValueError("cohort and set20 slides differ")
    if not np.array_equal(set20.groupby("slide_id").fold.first().reindex(order).to_numpy(),
                          folds.reindex(order).to_numpy()):
        raise ValueError("set20 folds disagree with the locked fold file")
    table = set20.merge(cohort[["slide_id", "cache_path"]], on="slide_id",
                        validate="many_to_one")
    rank = {slide: index for index, slide in enumerate(order)}
    table = table.assign(_rank=table.slide_id.map(rank)).sort_values(
        ["_rank", "location_index"]).drop(columns="_rank").reset_index(drop=True)
    return table, order


def read_cache(cache_path: str, locations: np.ndarray) -> dict[str, np.ndarray]:
    with h5py.File(cache_path, "r") as store:
        names = tuple(value.decode().lower() for value in store["scanner_names"][:])
        if names[0] != "at2":
            raise ValueError(f"{cache_path}: AT2 is not scanner 0: {names}")
        return {
            "image": np.asarray(store["images"][locations, 0], dtype=np.uint8),
            "final": np.asarray(store["final_coords_xy"][locations, 0], dtype=np.int64),
            "phase": np.asarray(store["phase_coords_xy"][locations, 0], dtype=np.int64),
            "applied": np.asarray(store["applied_shift_xy"][locations, 0], dtype=np.int64),
            "source_index": np.asarray(store["source_index"][locations], dtype=np.int64),
        }


# ------------------------------------------------------------------ identity stage
def identity_slide(task: tuple[str, str, list[int], list[int]]) -> tuple[list[dict], dict]:
    import openslide
    from preprocessing_patch_extraction_matching_and_QC import in_bounds, read_rgb

    slide_id, cache_path, locations, expected_source = task
    cache = read_cache(cache_path, np.asarray(locations, dtype=np.int64))
    registered_path = REGISTERED_AT2 / f"{slide_id}.tiff"
    raw_path = RAW_AT2 / f"{slide_id}.svs"
    registered = openslide.OpenSlide(str(registered_path))
    raw = openslide.OpenSlide(str(raw_path)) if raw_path.exists() else None
    slide_info = {
        "slide_id": slide_id,
        "registered_dimensions": list(registered.dimensions),
        "registered_level_count": registered.level_count,
        "registered_mpp_x": registered.properties.get("openslide.mpp-x"),
        "raw_available": raw is not None,
        "raw_dimensions": list(raw.dimensions) if raw is not None else None,
        "raw_mpp_x": raw.properties.get("openslide.mpp-x") if raw is not None else None,
    }
    rows = []
    for offset, location in enumerate(locations):
        x, y = map(int, cache["final"][offset])
        cached = cache["image"][offset]
        reread = read_rgb(registered, x, y, PATCH)
        difference = np.abs(reread.astype(np.int16) - cached.astype(np.int16))
        row = {
            "slide_id": slide_id,
            "location_index": int(location),
            "source_index": int(cache["source_index"][offset]),
            "source_index_matches_set20": int(cache["source_index"][offset]) == int(expected_source[offset]),
            "x": x,
            "y": y,
            "at2_final_equals_phase": bool(np.array_equal(cache["final"][offset], cache["phase"][offset])),
            "at2_applied_shift_zero": bool(not cache["applied"][offset].any()),
            "registered_identical": bool(np.array_equal(reread, cached)),
            "registered_max_abs_diff": int(difference.max()),
            "window_in_bounds": bool(in_bounds(registered, x - PAD, y - PAD, PATCH + 2 * PAD)),
        }
        if raw is not None:
            raw_patch = read_rgb(raw, x, y, PATCH)
            row["raw_svs_identical"] = bool(np.array_equal(raw_patch, cached))
            row["raw_svs_max_abs_diff"] = int(np.abs(raw_patch.astype(np.int16)
                                                     - cached.astype(np.int16)).max())
        else:
            row["raw_svs_identical"] = False
            row["raw_svs_max_abs_diff"] = -1
        rows.append(row)
    registered.close()
    if raw is not None:
        raw.close()
    return rows, slide_info


def identity(args: argparse.Namespace) -> None:
    table, order = load_locations()
    slides = args.slides or order
    output = Path(args.output_dir) / "identity"
    tasks = []
    for slide_id in slides:
        part = table[table.slide_id.eq(slide_id)]
        tasks.append((slide_id, str(part.cache_path.iloc[0]),
                      part.location_index.astype(int).tolist(),
                      part.source_index.astype(int).tolist()))
    rows, slide_infos = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for slide_rows, info in pool.map(identity_slide, tasks):
            rows.extend(slide_rows)
            slide_infos.append(info)
            print(f"identity {info['slide_id']}: "
                  f"{sum(row['registered_identical'] for row in slide_rows)}/{len(slide_rows)}",
                  flush=True)
    frame = pd.DataFrame(rows)
    slides_frame = pd.DataFrame(slide_infos)
    write_frame(output / "identity_check.csv.gz", frame)
    write_frame(output / "identity_slides.csv", slides_frame)
    geometry_equal = bool(
        slides_frame.raw_available.all()
        and (slides_frame.registered_dimensions.map(tuple)
             == slides_frame.raw_dimensions.map(lambda v: tuple(v) if v else None)).all()
    )
    checks = {
        "registered_identical_all": bool(frame.registered_identical.all()),
        "source_index_matches_set20_all": bool(frame.source_index_matches_set20.all()),
        "at2_final_equals_phase_all": bool(frame.at2_final_equals_phase.all()),
        "at2_applied_shift_zero_all": bool(frame.at2_applied_shift_zero.all()),
        "fourier_window_in_bounds_all": bool(frame.window_in_bounds.all()),
    }
    status = "pass" if all(checks.values()) and (args.slides or len(frame) == 2060) else "fail"
    summary = {
        "stage": "identity",
        "status": status,
        "created_utc": utc_now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slides": int(frame.slide_id.nunique()),
        "locations": int(len(frame)),
        "file_used_for_shifts": str(REGISTERED_AT2 / "<slide_id>.tiff"),
        "coordinate_convention": "cache final_coords_xy[:, 0] = AT2 level-0 top-left (x, y); "
                                 "openslide read_region(level 0, 256 x 256), RGBA->RGB",
        "checks": checks,
        "registered_identical": int(frame.registered_identical.sum()),
        "registered_max_abs_diff": int(frame.registered_max_abs_diff.max()),
        "raw_svs_identical": int(frame.raw_svs_identical.sum()),
        "raw_svs_max_abs_diff": int(frame.raw_svs_max_abs_diff.max()),
        "raw_svs_same_dimensions_as_registered_all_slides": geometry_equal,
        "registered_mpp_x": sorted(set(slides_frame.registered_mpp_x.astype(str))),
        "raw_mpp_x": sorted(set(slides_frame.raw_mpp_x.astype(str))),
        "fourier_window": {"pad_px": PAD, "size_px": PATCH + 2 * PAD},
    }
    write_json(output / "identity_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if status != "pass":
        raise SystemExit("identity check failed: shifts must not be computed")


# --------------------------------------------------------------------- embed stage
@functools.lru_cache(maxsize=64)
def phase_ramp(height: int, width: int, dx: float, dy: float) -> np.ndarray:
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    return np.exp(2j * np.pi * (fx * dx + fy * dy))[..., None]


def fourier_sample(spectrum: np.ndarray, dx: float, dy: float, workers: int) -> np.ndarray:
    """Band-limited resampling of the window at (n + dx, n + dy): out[n] = in[n + d]."""
    from scipy import fft as sp_fft

    ramp = phase_ramp(spectrum.shape[0], spectrum.shape[1], float(dx), float(dy))
    return np.real(sp_fft.ifft2(spectrum * ramp, axes=(0, 1), workers=workers))


def centre_uint8(values: np.ndarray, pad: int) -> np.ndarray:
    return np.clip(np.rint(values[pad:pad + PATCH, pad:pad + PATCH]), 0, 255).astype(np.uint8)


def shifted_images(handle, original: np.ndarray, x: int, y: int,
                   workers: int) -> tuple[np.ndarray, dict]:
    from scipy import fft as sp_fft
    from preprocessing_patch_extraction_matching_and_QC import read_rgb

    window = read_rgb(handle, x - PAD, y - PAD, PATCH + 2 * PAD)
    spectrum = sp_fft.fft2(window.astype(np.float64), axes=(0, 1), workers=workers)
    inner = window[PAD - PAD_CHECK:PAD + PATCH + PAD_CHECK,
                   PAD - PAD_CHECK:PAD + PATCH + PAD_CHECK]
    inner_spectrum = sp_fft.fft2(inner.astype(np.float64), axes=(0, 1), workers=workers)
    images = np.empty((len(CONDITIONS), PATCH, PATCH, 3), dtype=np.uint8)
    images[0] = original
    images[1] = centre_uint8(fourier_sample(spectrum, 0.0, 0.0, workers), PAD)
    qc = {
        "window_centre_equals_cache": bool(np.array_equal(
            window[PAD:PAD + PATCH, PAD:PAD + PATCH], original)),
        "fourier_zero_equals_cache": bool(np.array_equal(images[1], original)),
        "fourier_integer_equals_wsi": 0,
        "window_crop_equals_wsi": 0,
        "pad_check_max_abs": 0,
        "pad_check_mean_abs": 0.0,
    }
    pad_means = []
    for index, (_, method, shift, direction) in enumerate(CONDITIONS[2:], start=2):
        ux, uy = DIRECTIONS[direction]
        dx, dy = shift * ux, shift * uy
        if method == "fourier":
            images[index] = centre_uint8(fourier_sample(spectrum, dx, dy, workers), PAD)
            if shift == 0.5:
                check = centre_uint8(fourier_sample(inner_spectrum, dx, dy, workers), PAD_CHECK)
                difference = np.abs(check.astype(np.int16) - images[index].astype(np.int16))
                qc["pad_check_max_abs"] = max(qc["pad_check_max_abs"], int(difference.max()))
                pad_means.append(float(difference.mean()))
        else:
            ix, iy = int(round(dx)), int(round(dy))
            images[index] = read_rgb(handle, x + ix, y + iy, PATCH)
            fourier = centre_uint8(fourier_sample(spectrum, dx, dy, workers), PAD)
            qc["fourier_integer_equals_wsi"] += int(np.array_equal(fourier, images[index]))
            crop = window[PAD + iy:PAD + iy + PATCH, PAD + ix:PAD + ix + PATCH]
            qc["window_crop_equals_wsi"] += int(np.array_equal(crop, images[index]))
    qc["pad_check_mean_abs"] = float(np.mean(pad_means))
    return images, qc


def stored_features(model: str, slide_id: str, fold: int, locations: np.ndarray) -> np.ndarray:
    if model == "uni_v1":
        path = STORED_UNI / f"{slide_id}.h5"
        with h5py.File(path, "r") as store:
            names = [value.decode() for value in store["condition_names"][:]]
            observed = np.asarray(store["location_index"], dtype=np.int64)
            features = np.asarray(store["features"][:, names.index("source")], dtype=np.float32)
    else:
        path = STORED_OTHER / model / f"fold_{fold}" / f"{slide_id}.h5"
        with h5py.File(path, "r") as store:
            names = [value.decode().lower() for value in store["scanner_names"][:]]
            observed = np.asarray(store["location_index"], dtype=np.int64)
            features = np.asarray(store["features"][:, names.index("at2")], dtype=np.float32)
    lookup = {int(value): index for index, value in enumerate(observed)}
    return features[[lookup[int(value)] for value in locations]]


def embed(args: argparse.Namespace) -> None:
    import openslide
    import torch

    sys.path.insert(0, str(PROJECT / "analysis/paper"))
    from scanner_gan.evaluate_images import image_metrics
    from scripts.features.review_feature_crossencoder_extract import images_to_features
    from scripts.features.review_multiencoder_scanner import load_encoder
    from table2_benchmark import lpips_model, lpips_scores

    gate = json.loads((OUTPUT / "identity/identity_summary.json").read_text())
    if gate.get("status") != "pass" or gate.get("locations") != 2060:
        raise SystemExit("identity gate has not passed for all 2,060 locations")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    torch.set_num_threads(workers)
    table, order = load_locations()
    slides = args.slides or order[args.task_index::args.task_count]
    output = Path(args.output_dir)
    shard_dir = output / "shards"
    if not args.overwrite:
        slides = [slide for slide in slides
                  if not (shard_dir / f"{slide}.csv.gz").exists()
                  or not (shard_dir / f"{slide}.qc.csv.gz").exists()]
    print(json.dumps({"task_index": args.task_index, "slides": slides}), flush=True)
    if not slides:
        return

    # 1. images and image-level QC (CPU)
    data: dict[str, dict] = {}
    for slide_id in slides:
        part = table[table.slide_id.eq(slide_id)].reset_index(drop=True)
        locations = part.location_index.to_numpy(dtype=np.int64)
        cache = read_cache(str(part.cache_path.iloc[0]), locations)
        if not np.array_equal(cache["source_index"], part.source_index.to_numpy()):
            raise ValueError(f"{slide_id}: source_index differs from set20")
        handle = openslide.OpenSlide(str(REGISTERED_AT2 / f"{slide_id}.tiff"))
        images, qcs = [], []
        for offset in range(len(part)):
            x, y = map(int, cache["final"][offset])
            stack, qc = shifted_images(handle, cache["image"][offset], x, y, workers)
            images.append(stack)
            qcs.append(qc)
        handle.close()
        images = np.stack(images)                                   # (20, C, 256, 256, 3)
        interior = images[:, :, BORDER:PATCH - BORDER, BORDER:PATCH - BORDER]
        reference = np.repeat(interior[:, 0], len(CONDITIONS) - 1, axis=0)
        compared = interior[:, 1:].reshape(-1, *interior.shape[2:])
        metrics = image_metrics(reference, compared, reference)
        data[slide_id] = {"part": part, "images": images, "qc": qcs,
                          "reference": reference, "compared": compared,
                          "metrics": metrics}
        print(f"images {slide_id}: {images.shape}", flush=True)

    # 2. LPIPS-VGG16 (Table 2 contract)
    device = torch.device("cuda")
    contract = json.loads(LPIPS_CONTRACT.read_text())
    torch_home = Path(os.environ.get("TORCH_HOME", ""))
    vgg = torch_home / "hub/checkpoints/vgg16-397923af.pth"
    vgg_hash = sha256(vgg) if vgg.exists() else None
    expected_hash = contract["weight_files"][0]["sha256"]
    model = lpips_model(device)
    identical = float(lpips_scores(model, np.zeros((1, 64, 64, 3), np.uint8),
                                   np.zeros((1, 64, 64, 3), np.uint8), device, 1)[0])
    for slide_id, item in data.items():
        item["lpips"] = lpips_scores(model, item["compared"], item["reference"], device,
                                     args.batch_size)
    del model
    torch.cuda.empty_cache()

    # 3. PFM embeddings; condition-major order so that original and zero-shift copies
    #    sit in different batch positions (numerical reproducibility check).
    for name in PFMS:
        encoder, size, mean, std, variants = load_encoder(name, HF_CACHE)
        variant = variants[0]
        for slide_id, item in data.items():
            images = item["images"]
            flat = np.ascontiguousarray(images.swapaxes(0, 1)).reshape(-1, PATCH, PATCH, 3)
            features = images_to_features(encoder, variant, name, flat, size, mean, std,
                                          args.batch_size)
            features = features.reshape(len(CONDITIONS), len(images), -1).swapaxes(0, 1)
            original = features[:, 0]
            item[name] = 1.0 - np.clip((features[:, 1:] * original[:, None]).sum(-1), -1.0, 1.0)
            part = item["part"]
            stored = stored_features(name, slide_id, int(part.fold.iloc[0]),
                                     part.location_index.to_numpy())
            item[f"parity_{name}"] = (original * stored).sum(-1) / (
                np.linalg.norm(original, axis=-1) * np.linalg.norm(stored, axis=-1))
            print(f"{name} {slide_id}: max 1px distance "
                  f"{item[name][:, [i - 1 for i, c in enumerate(CONDITIONS) if c[2] == 1.0]].max():.4f}, "
                  f"min parity {item[f'parity_{name}'].min():.6f}", flush=True)
        del encoder
        torch.cuda.empty_cache()

    # 4. shards
    for slide_id, item in data.items():
        part = item["part"]
        rows, qc_rows = [], []
        metrics = item["metrics"]
        for offset, record in enumerate(part.itertuples(index=False)):
            for position, (condition, method, shift, direction) in enumerate(CONDITIONS[1:]):
                flat = offset * (len(CONDITIONS) - 1) + position
                row = {
                    "slide_id": slide_id, "tissue_type": record.tissue_type,
                    "fold": int(record.fold), "location_index": int(record.location_index),
                    "source_index": int(record.source_index), "condition": condition,
                    "method": method, "shift_px": shift, "direction": direction,
                    "ssim": float(metrics.target_ssim.iloc[flat]),
                    "lpips_vgg": float(item["lpips"][flat]),
                    "l1_unit": float(metrics.target_l1_unit.iloc[flat]),
                    "psnr_db": float(metrics.target_psnr_db.iloc[flat]),
                }
                for name in PFMS:
                    row[name] = float(item[name][offset, position])
                rows.append(row)
            qc = {"slide_id": slide_id, "location_index": int(record.location_index),
                  **item["qc"][offset]}
            for name in PFMS:
                qc[f"parity_{name}"] = float(item[f"parity_{name}"][offset])
            qc["lpips_identical_smoke"] = identical
            qc["vgg16_sha256_matches_contract"] = vgg_hash == expected_hash
            qc["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
            qc_rows.append(qc)
        write_frame(shard_dir / f"{slide_id}.csv.gz", pd.DataFrame(rows))
        write_frame(shard_dir / f"{slide_id}.qc.csv.gz", pd.DataFrame(qc_rows))
        print(f"wrote {slide_id}", flush=True)


# ----------------------------------------------------------------- aggregate stage
def bootstrap_index(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, n, size=(BOOTSTRAP, n))


def interval(draws: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def reference_distances(slides: list[str]) -> tuple[pd.DataFrame, dict]:
    """Per-slide raw and corrected PanNormal target distances behind Table 3."""
    frames = []
    units = pd.read_csv(TABLE4_UNITS, dtype={"unit": str})
    units = units[units.dataset.eq("PanNormal")].rename(columns={"unit": "slide_id"})
    for method in ("raw", "reinhard", "combined"):
        part = units[units.scanner.isin(TARGETS)]
        frames.append(pd.DataFrame({
            "model": part.model, "scanner": part.scanner, "slide_id": part.slide_id,
            "method": method, "raw": part.raw, "corrected": part[method],
            "source": "table4_color_frequency"}))
    gan = pd.read_csv(GAN_SLIDES, dtype={"slide_id": str})
    gan_part = gan[gan.scanner.isin(TARGETS)]
    for method in ("pix2pix", "cyclegan"):
        frames.append(pd.DataFrame({
            "model": gan_part.model, "scanner": gan_part.scanner,
            "slide_id": gan_part.slide_id, "method": method,
            "raw": gan_part.raw_distance, "corrected": gan_part[f"{method}_distance"],
            "source": "gan_encoder_review"}))
    feature = [pd.read_csv(FEATURE_UNI, dtype={"slide_id": str}).assign(model="uni_v1")]
    for model in PFMS[1:]:
        for scanner in TARGETS:
            feature.append(pd.read_csv(FEATURE_OTHER / model / scanner / "per_slide.csv",
                                       dtype={"slide_id": str}))
    feature = pd.concat(feature, ignore_index=True)
    feature = feature[feature.method.isin(("featmap_ridge", "combat"))]
    frames.append(pd.DataFrame({
        "model": feature.model, "scanner": feature.scanner, "slide_id": feature.slide_id,
        "method": feature.method, "raw": feature.raw_distance,
        "corrected": feature.corrected_distance, "source": "feature_crossencoder"}))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame.slide_id.isin(slides)]
    counts = frame.groupby(["model", "method", "scanner"]).slide_id.nunique()
    if frame.duplicated(["model", "method", "scanner", "slide_id"]).any():
        raise ValueError("duplicate reference distances")
    if len(counts) != len(PFMS) * (len(METHODS) + 1) * len(TARGETS) or not counts.eq(len(slides)).all():
        raise ValueError("incomplete reference distances")
    pooled = (frame.groupby(["model", "method", "slide_id"], as_index=False)[["raw", "corrected"]]
              .mean().assign(scanner="all"))
    frame = pd.concat([frame.drop(columns="source"), pooled], ignore_index=True)

    qc: dict = {}
    # raw distance agreement across the three evidence sources
    raw = frame[frame.scanner.ne("all")].pivot_table(
        index=["model", "scanner", "slide_id"], columns="method", values="raw")
    qc["raw_max_abs_diff_across_sources"] = float(
        raw.max(axis=1).sub(raw.min(axis=1)).abs().max())
    # per model and method: disagreement with the table4 raw (the floor-fraction raw)
    qc["raw_max_abs_diff_vs_table4"] = {
        f"{model}:{method}": float(value)
        for (model, method), value in raw.sub(raw["reinhard"], axis=0).abs()
        .groupby(level="model").max().stack().items()
        if method not in ("raw", "reinhard", "combined")}
    # provided pooled rows agree with the scanner mean
    given = units[units.scanner.eq("all")].set_index(["model", "slide_id"])
    mine = pooled[pooled.method.eq("combined")].set_index(["model", "slide_id"])
    qc["table4_pooled_rows_max_abs_diff"] = float(
        (given.loc[mine.index, "combined"] - mine.corrected).abs().max())
    given_gan = gan[gan.scanner.eq("all")].set_index(["model", "slide_id"])
    mine_gan = pooled[pooled.method.eq("pix2pix")].set_index(["model", "slide_id"])
    qc["gan_pooled_rows_max_abs_diff"] = float(
        (given_gan.loc[mine_gan.index, "pix2pix_distance"] - mine_gan.corrected).abs().max())
    # published table reproduction (PanNormal)
    reproduction = []
    for method, values in TABLE_PANNORMAL.items():
        source_method = "reinhard" if method == "raw" else method
        for model, published in zip(PFMS, values):
            subset = pooled[pooled.model.eq(model) & pooled.method.eq(source_method)]
            value = subset.raw.mean() if method == "raw" else subset.corrected.mean()
            reproduction.append({"method": method, "model": model, "published": published,
                                 "recomputed": float(value),
                                 "abs_diff": float(abs(value - published))})
    reproduction = pd.DataFrame(reproduction)
    qc["table_reproduction"] = reproduction.to_dict(orient="records")
    qc["table_reproduction_max_abs_diff"] = float(reproduction.abs_diff.max())
    qc["table_reproduction_within_rounding"] = bool(
        (reproduction.abs_diff <= 0.00005 + 1e-9).all())
    return frame, qc


def raw_image_metrics(slides: list[str], set20: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frames = []
    for slide_id in slides:
        part = pd.read_csv(TABLE2_SHARDS / f"{slide_id}.csv.gz", dtype={"slide_id": str})
        frames.append(part[part.method.eq("raw")])
    frame = pd.concat(frames, ignore_index=True)
    frame["scanner"] = frame.scanner.str.lower()
    expected = set(map(tuple, set20[["slide_id", "location_index"]].itertuples(index=False)))
    observed = set(map(tuple, frame[["slide_id", "location_index"]].drop_duplicates()
                       .itertuples(index=False)))
    per_scanner = frame.groupby(["slide_id", "scanner"], as_index=False)[
        ["target_ssim", "lpips_vgg"]].mean()
    pooled = per_scanner.groupby("slide_id", as_index=False)[["target_ssim", "lpips_vgg"]] \
        .mean().assign(scanner="all")
    result = pd.concat([per_scanner, pooled], ignore_index=True).rename(
        columns={"target_ssim": "ssim"})
    baseline = pd.read_csv(TABLE2_BASELINE)
    baseline = baseline[baseline.dataset.eq("PanNormal") & baseline.method.eq("raw")].iloc[0]
    qc = {
        "table2_locations_equal_set20": observed == expected,
        "table2_rows_per_slide_scanner": int(frame.groupby(["slide_id", "scanner"]).size().min()),
        "raw_ssim_recomputed": float(pooled.target_ssim.mean()),
        "raw_ssim_published": float(baseline.target_ssim_mean),
        "raw_lpips_recomputed": float(pooled.lpips_vgg.mean()),
        "raw_lpips_published": float(baseline.lpips_vgg_mean),
    }
    qc["raw_image_metrics_reproduced"] = bool(
        abs(qc["raw_ssim_recomputed"] - qc["raw_ssim_published"]) < 1e-6
        and abs(qc["raw_lpips_recomputed"] - qc["raw_lpips_published"]) < 1e-6)
    return result, qc


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def fmt_ci(mean: float, low: float, high: float, digits: int = 4, scale: float = 1.0) -> str:
    return (f"{mean * scale:.{digits}f} [{low * scale:.{digits}f}, "
            f"{high * scale:.{digits}f}]")


def aggregate(args: argparse.Namespace) -> None:
    table, order = load_locations()
    output = Path(args.output_dir)
    shard_dir = Path(args.shard_dir) if args.shard_dir else output / "shards"
    frames, qc_frames = [], []
    for slide_id in order:
        path = shard_dir / f"{slide_id}.csv.gz"
        qc_path = shard_dir / f"{slide_id}.qc.csv.gz"
        if not path.exists() or not qc_path.exists():
            if args.allow_partial:
                continue
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path, dtype={"slide_id": str}))
        qc_frames.append(pd.read_csv(qc_path, dtype={"slide_id": str}))
    frame = pd.concat(frames, ignore_index=True)
    location_qc = pd.concat(qc_frames, ignore_index=True)
    slides = [slide for slide in order if slide in set(frame.slide_id)]
    expected = table[table.slide_id.isin(slides)]
    if len(frame) != len(expected) * (len(CONDITIONS) - 1):
        raise ValueError("unexpected number of shard rows")
    if frame.duplicated(["slide_id", "location_index", "condition"]).any():
        raise ValueError("duplicate shard rows")
    if set(map(tuple, frame[["slide_id", "location_index"]].drop_duplicates()
               .itertuples(index=False))) != set(map(tuple, expected[
                   ["slide_id", "location_index"]].itertuples(index=False))):
        raise ValueError("shard locations differ from set20")
    if not args.allow_partial and len(slides) != 103:
        raise ValueError("expected 103 slides")
    write_frame(output / "location_distances.csv.gz", frame)
    write_frame(output / "location_qc.csv.gz", location_qc)

    n = len(slides)
    index = bootstrap_index(n)
    rank = {slide: position for position, slide in enumerate(slides)}

    def vector(series: pd.Series) -> np.ndarray:
        values = series.reindex(slides)
        if values.isna().any():
            raise ValueError(f"missing slide values for {series.name}")
        return values.to_numpy(dtype=float)

    shifted = frame[frame.condition.ne("zero")]
    slide_means = shifted.groupby(["shift_px", "slide_id"])[list(METRICS)].mean()
    slide_table = slide_means.reset_index()
    slide_table = slide_table.assign(_r=slide_table.slide_id.map(rank)).sort_values(
        ["shift_px", "_r"]).drop(columns="_r")
    write_frame(output / "slide_floor.csv", slide_table)

    # floor per metric and shift (directions pooled) and by direction
    floor_rows = []
    for shift in SHIFTS:
        for metric in METRICS:
            values = vector(slide_means.loc[shift][metric])
            low, high = interval(values[index].mean(axis=1))
            floor_rows.append({"metric": metric, "shift_px": shift,
                               "method": "fourier" if shift in SUBPIXEL else "wsi",
                               "direction": "all", "mean": values.mean(),
                               "sd_slides": values.std(ddof=1) if n > 1 else np.nan,
                               "ci_low": low, "ci_high": high, "slides": n})
    by_direction = shifted.groupby(["shift_px", "direction", "slide_id"])[list(METRICS)].mean()
    for (shift, direction), part in by_direction.groupby(level=[0, 1]):
        part = part.droplevel([0, 1])
        for metric in METRICS:
            values = vector(part[metric])
            low, high = interval(values[index].mean(axis=1))
            floor_rows.append({"metric": metric, "shift_px": shift,
                               "method": "fourier" if shift in SUBPIXEL else "wsi",
                               "direction": direction, "mean": values.mean(),
                               "sd_slides": values.std(ddof=1) if n > 1 else np.nan,
                               "ci_low": low, "ci_high": high, "slides": n})
    zero = frame[frame.condition.eq("zero")].groupby("slide_id")[list(METRICS)].mean()
    for metric in METRICS:
        values = vector(zero[metric])
        floor_rows.append({"metric": metric, "shift_px": 0.0, "method": "fourier_zero_check",
                           "direction": "none", "mean": values.mean(),
                           "sd_slides": values.std(ddof=1) if n > 1 else np.nan,
                           "ci_low": np.nan, "ci_high": np.nan, "slides": n})
    floor = pd.DataFrame(floor_rows)
    write_frame(output / "floor_summary.csv", floor)

    # floor as a fraction of the raw AT2-target distance; fraction of achievable reduction
    reference, reference_qc = reference_distances(slides)
    write_frame(output / "reference_distances_used.csv.gz", reference)
    fraction_rows, far_rows = [], []
    for model in PFMS:
        for scanner in ("all", *TARGETS):
            raw = reference[reference.model.eq(model) & reference.method.eq("reinhard")
                            & reference.scanner.eq(scanner)].set_index("slide_id").raw
            raw_v = vector(raw)
            raw_draw = raw_v[index].mean(axis=1)
            for shift in SHIFTS:
                floor_v = vector(slide_means.loc[shift][model])
                floor_draw = floor_v[index].mean(axis=1)
                ratio = floor_v.mean() / raw_v.mean()
                low, high = interval(floor_draw / raw_draw)
                fraction_rows.append({
                    "model": model, "scanner": scanner, "shift_px": shift,
                    "floor": floor_v.mean(), "raw": raw_v.mean(), "floor_fraction_of_raw": ratio,
                    "ci_low": low, "ci_high": high,
                    "max_achievable_reduction_pct": 100.0 * (1.0 - ratio),
                    "slides": n})
            for method in METHODS:
                part = reference[reference.model.eq(model) & reference.method.eq(method)
                                 & reference.scanner.eq(scanner)].set_index("slide_id")
                r_v, c_v = vector(part.raw), vector(part.corrected)
                r_d, c_d = r_v[index].mean(axis=1), c_v[index].mean(axis=1)
                pct = 100.0 * (r_v.mean() - c_v.mean()) / r_v.mean()
                pct_low, pct_high = interval(100.0 * (r_d - c_d) / r_d)
                for shift in SHIFTS:
                    floor_v = vector(slide_means.loc[shift][model])
                    f_d = floor_v[index].mean(axis=1)
                    far = (r_v.mean() - c_v.mean()) / (r_v.mean() - floor_v.mean())
                    far_low, far_high = interval((r_d - c_d) / (r_d - f_d))
                    far_rows.append({
                        "model": model, "scanner": scanner, "method": method,
                        "floor_shift_px": shift, "raw": r_v.mean(), "corrected": c_v.mean(),
                        "floor": floor_v.mean(), "pct_reduction": pct,
                        "pct_reduction_ci_low": pct_low, "pct_reduction_ci_high": pct_high,
                        "fraction_achievable_reduction": far,
                        "far_ci_low": far_low, "far_ci_high": far_high,
                        "floor_over_corrected": floor_v.mean() / c_v.mean(),
                        "slides": n})
    fractions = pd.DataFrame(fraction_rows)
    far = pd.DataFrame(far_rows)
    write_frame(output / "floor_fraction.csv", fractions)
    write_frame(output / "achievable_reduction.csv", far)

    # image counterparts: floor SSIM / LPIPS next to raw AT2-target SSIM / LPIPS
    raw_image, raw_image_qc = raw_image_metrics(slides, expected)
    image_rows = []
    for scanner in ("all", *TARGETS):
        raw_part = raw_image[raw_image.scanner.eq(scanner)].set_index("slide_id")
        for metric in ("ssim", "lpips_vgg"):
            raw_v = vector(raw_part[metric])
            raw_d = raw_v[index].mean(axis=1)
            raw_low, raw_high = interval(raw_d)
            for shift in SHIFTS:
                floor_v = vector(slide_means.loc[shift][metric])
                floor_d = floor_v[index].mean(axis=1)
                floor_low, floor_high = interval(floor_d)
                if metric == "ssim":
                    ratio = (1 - floor_v.mean()) / (1 - raw_v.mean())
                    ratio_d = (1 - floor_d) / (1 - raw_d)
                    ratio_name = "(1-SSIM floor)/(1-SSIM raw)"
                else:
                    ratio = floor_v.mean() / raw_v.mean()
                    ratio_d = floor_d / raw_d
                    ratio_name = "LPIPS floor/LPIPS raw"
                low, high = interval(ratio_d)
                image_rows.append({
                    "metric": metric, "scanner": scanner, "shift_px": shift,
                    "floor": floor_v.mean(), "floor_ci_low": floor_low,
                    "floor_ci_high": floor_high, "raw": raw_v.mean(),
                    "raw_ci_low": raw_low, "raw_ci_high": raw_high,
                    "ratio_definition": ratio_name, "ratio": ratio,
                    "ratio_ci_low": low, "ratio_ci_high": high, "slides": n})
    image = pd.DataFrame(image_rows)
    write_frame(output / "image_counterparts.csv", image)

    # QC record
    identity_summary = json.loads((OUTPUT / "identity/identity_summary.json").read_text())
    locations = len(location_qc)
    parity = {}
    for model in PFMS:
        values = location_qc[f"parity_{model}"]
        parity[model] = {"min": float(values.min()), "p01": float(values.quantile(0.01)),
                         "median": float(values.median()),
                         "fraction_ge_0.999": float((values >= PARITY_MIN).mean())}
    zero_max = {model: float(frame.loc[frame.condition.eq("zero"), model].max())
                for model in PFMS}
    qc = {
        "created_utc": utc_now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "job_chain": os.environ.get("RV08_JOB_IDS"),
        "slides": n,
        "locations": locations,
        "identity_gate": {key: identity_summary[key] for key in (
            "status", "locations", "registered_identical", "registered_max_abs_diff",
            "raw_svs_identical", "checks")},
        "image_qc": {
            "window_centre_equals_cache": int(location_qc.window_centre_equals_cache.sum()),
            "fourier_zero_equals_cache": int(location_qc.fourier_zero_equals_cache.sum()),
            "fourier_integer_equals_wsi_all8": int(location_qc.fourier_integer_equals_wsi.eq(8).sum()),
            "window_crop_equals_wsi_all8": int(location_qc.window_crop_equals_wsi.eq(8).sum()),
            "pad_check_max_abs_levels": int(location_qc.pad_check_max_abs.max()),
            "pad_check_mean_abs_levels": float(location_qc.pad_check_mean_abs.mean()),
            "lpips_identical_smoke": float(location_qc.lpips_identical_smoke.max()),
            "vgg16_sha256_matches_contract": bool(location_qc.vgg16_sha256_matches_contract.all()),
        },
        "zero_shift_image_metrics": {
            "ssim_min": float(frame.loc[frame.condition.eq("zero"), "ssim"].min()),
            "lpips_max": float(frame.loc[frame.condition.eq("zero"), "lpips_vgg"].max()),
        },
        "zero_shift_pfm_distance_max": zero_max,
        "embedding_parity_vs_stored_at2": parity,
        "reference_distances": reference_qc,
        "raw_image_metrics": raw_image_qc,
        "bootstrap": {"resamples": BOOTSTRAP, "seed": SEED, "unit": "slide",
                      "interval": "percentile 2.5-97.5"},
    }
    qc["image_qc"]["all_pass"] = bool(
        qc["image_qc"]["window_centre_equals_cache"] == locations
        and qc["image_qc"]["fourier_zero_equals_cache"] == locations
        and qc["image_qc"]["fourier_integer_equals_wsi_all8"] == locations
        and qc["image_qc"]["window_crop_equals_wsi_all8"] == locations
        and qc["image_qc"]["vgg16_sha256_matches_contract"])
    write_json(output / "qc.json", qc)
    write_text(output / "summary.md", summary_markdown(
        floor, fractions, far, image, qc, n, locations, args.allow_partial))
    print(json.dumps({"slides": n, "locations": locations,
                      "image_qc_pass": qc["image_qc"]["all_pass"]}, indent=2))


def summary_markdown(floor: pd.DataFrame, fractions: pd.DataFrame, far: pd.DataFrame,
                     image: pd.DataFrame, qc: dict, n: int, locations: int,
                     partial: bool) -> str:
    identity = qc["identity_gate"]
    image_qc = qc["image_qc"]
    lines = [
        "# RV08 Registration noise floor",
        "",
        f"Generated {qc['created_utc']} by `analysis/revision/registration_floor.py aggregate`"
        + (f" (SLURM job {qc['slurm_job_id']})" if qc["slurm_job_id"] else "") + ".",
        "Protocol: `analysis/revision/README.md`, section RV08.",
    ]
    if qc.get("job_chain"):
        lines.append(f"SLURM chain: {qc['job_chain']}.")
    lines.append("")
    if partial:
        lines += [f"**PARTIAL RUN (smoke test): {n} slides only. Not the pre-specified "
                  "analysis.**", ""]
    lines += [
        "## What ran",
        "",
        f"- {n} PanNormal slides, {locations} `set20` locations. The cached AT2 patch is compared "
        "with the same AT2 field sampled 0.25 and 0.5 px away (ideal Fourier shift of a "
        f"{PATCH + 2 * PAD}-px WSI window, centre-cropped to 256 px) and 1 and 2 px away (read "
        "from the registered AT2 WSI), in four directions (+x, -x, +y, -y).",
        "- PFM cosine distance for UNI v1, UNI2-h, Virchow2 and H-optimus-1 with the paper's "
        "embedding code (`load_encoder`, `images_to_features`); SSIM and LPIPS-VGG16 with the "
        "Table 2 contract (252 x 252 interior, 2-pixel boundary excluded).",
        "- Location values are averaged over the four directions and 20 locations per slide, "
        f"then over slides. 95% CIs: {BOOTSTRAP:,} slide bootstrap resamples (seed {SEED}); the "
        "same resamples are used for floor, raw and corrected distances, so ratios are paired.",
        "- Raw and corrected target distances are the per-slide evidence files behind "
        "`table_cross_pfm_correction.tex` (PanNormal, five directions equally weighted).",
        "",
        "## QC",
        "",
        f"- Identity gate (run before any shift): {identity['registered_identical']}/"
        f"{identity['locations']} cached AT2 patches reproduced bit-exactly by re-reading the "
        "registered AT2 WSI at `final_coords_xy[:, 0]` (max abs difference "
        f"{identity['registered_max_abs_diff']}); status **{identity['status']}**. The raw AT2 "
        f"SVS reproduced {identity['raw_svs_identical']}/{identity['locations']}.",
        f"- Fourier window centre equals cache: {image_qc['window_centre_equals_cache']}/{locations}; "
        f"zero-shift Fourier round trip equals cache: {image_qc['fourier_zero_equals_cache']}/{locations}.",
        f"- Integer Fourier shifts (±1, ±2 px) equal the direct WSI reads in all 8 cases: "
        f"{image_qc['fourier_integer_equals_wsi_all8']}/{locations} locations (confirms direction "
        "convention and absence of wrap-around in the 256-px crop).",
        f"- Window margin sensitivity (0.5 px shifts, margin {PAD} vs {PAD_CHECK} px): max "
        f"{image_qc['pad_check_max_abs_levels']} grey level(s), mean "
        f"{image_qc['pad_check_mean_abs_levels']:.4f}.",
        f"- LPIPS VGG16 weights match the Table 2 contract hash: "
        f"{image_qc['vgg16_sha256_matches_contract']}; identical-image LPIPS "
        f"{image_qc['lpips_identical_smoke']:.3g}.",
        "- Zero-shift re-embedding (identical pixels, different batch position), max PFM "
        "distance: " + ", ".join(f"{PFM_LABELS[m]} {v:.2e}"
                                 for m, v in qc["zero_shift_pfm_distance_max"].items()) + ".",
        "- Parity of the unshifted AT2 embedding with the stored paper embeddings (cosine; "
        "UNI v1 vs `03_uni`, others vs `feature_crossencoder_review` raw): "
        + "; ".join(f"{PFM_LABELS[m]} min {v['min']:.5f}, median {v['median']:.6f}, "
                    f"{100 * v['fraction_ge_0.999']:.1f}% >= 0.999"
                    for m, v in qc["embedding_parity_vs_stored_at2"].items()) + ".",
        f"- Evidence files reproduce the published PanNormal rows of the cross-PFM table: "
        f"max abs difference {qc['reference_distances']['table_reproduction_max_abs_diff']:.2e} "
        f"(within rounding: {qc['reference_distances']['table_reproduction_within_rounding']}); "
        "raw distance max disagreement across sources (per slide x scanner) "
        f"{qc['reference_distances']['raw_max_abs_diff_across_sources']:.2e}; sources above "
        "1e-3: " + (", ".join(key for key, value in
                             qc["reference_distances"]["raw_max_abs_diff_vs_table4"].items()
                             if value > 1e-3) or "none")
        + " (each method's reduction and fraction of achievable reduction use the raw "
        "distance from that method's own evidence file, as the published table does; the "
        "floor fraction uses the colour/frequency file's raw).",
        f"- Raw AT2-target SSIM/LPIPS from the Table 2 shards: "
        f"{qc['raw_image_metrics']['raw_ssim_recomputed']:.4f}/"
        f"{qc['raw_image_metrics']['raw_lpips_recomputed']:.4f} (published "
        f"{qc['raw_image_metrics']['raw_ssim_published']:.4f}/"
        f"{qc['raw_image_metrics']['raw_lpips_published']:.4f}); locations equal set20: "
        f"{qc['raw_image_metrics']['table2_locations_equal_set20']}.",
        "",
        "## Primary endpoint: PFM floor as a fraction of the raw AT2-target distance",
        "",
        "Mean original-vs-shifted cosine distance [95% CI] and floor / raw distance [95% CI] "
        "(raw pooled over the five target scanners).",
        "",
        "| PFM | Raw distance | " + " | ".join(f"{s:g} px floor" for s in SHIFTS) + " | "
        + " | ".join(f"{s:g} px / raw" for s in SHIFTS) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in SHIFTS) + " | "
        + " | ".join("---:" for _ in SHIFTS) + " |",
    ]
    for model in PFMS:
        cells, ratios = [], []
        raw_value = None
        for shift in SHIFTS:
            row = floor[floor.metric.eq(model) & floor.shift_px.eq(shift)
                        & floor.direction.eq("all")].iloc[0]
            cells.append(fmt_ci(row["mean"], row.ci_low, row.ci_high))
            frac = fractions[fractions.model.eq(model) & fractions.scanner.eq("all")
                             & fractions.shift_px.eq(shift)].iloc[0]
            raw_value = frac.raw
            ratios.append(fmt_ci(frac.floor_fraction_of_raw, frac.ci_low, frac.ci_high, 3))
        lines.append(f"| {PFM_LABELS[model]} | {fmt(raw_value)} | " + " | ".join(cells)
                     + " | " + " | ".join(ratios) + " |")
    lines += [
        "",
        "Per target scanner, 1-px floor / raw distance [95% CI]:",
        "",
        "| PFM | " + " | ".join(scanner.upper() for scanner in TARGETS) + " |",
        "| --- | " + " | ".join("---:" for _ in TARGETS) + " |",
    ]
    for model in PFMS:
        cells = []
        for scanner in TARGETS:
            frac = fractions[fractions.model.eq(model) & fractions.scanner.eq(scanner)
                             & fractions.shift_px.eq(FAR_FLOOR_SHIFT)].iloc[0]
            cells.append(fmt_ci(frac.floor_fraction_of_raw, frac.ci_low, frac.ci_high, 3))
        lines.append(f"| {PFM_LABELS[model]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Image counterparts",
        "",
        "Original-vs-shifted SSIM and LPIPS-VGG16 [95% CI], next to raw AT2-target values "
        "on the same locations (pooled over scanners).",
        "",
        "| Metric | Raw AT2-target | " + " | ".join(f"{s:g} px" for s in SHIFTS) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in SHIFTS) + " |",
    ]
    for metric, label in (("ssim", "SSIM"), ("lpips_vgg", "LPIPS-VGG16")):
        part = image[image.metric.eq(metric) & image.scanner.eq("all")]
        raw_row = part.iloc[0]
        cells = [fmt_ci(row.floor, row.floor_ci_low, row.floor_ci_high, 3)
                 for row in part.sort_values("shift_px").itertuples()]
        lines.append(f"| {label} | {fmt_ci(raw_row.raw, raw_row.raw_ci_low, raw_row.raw_ci_high, 3)} | "
                     + " | ".join(cells) + " |")
    lines += [
        "",
        "## Fraction of achievable reduction (1-px floor)",
        "",
        "(raw - corrected) / (raw - floor) next to the percentage reduction (raw - corrected) / "
        "raw; PanNormal, pooled over target scanners, [95% CI]. Negative values mean the "
        "correction moved embeddings away from the real target. 'Floor / corrected' is the "
        "share of the corrected distance matched by the 1-px floor.",
        "",
        "| PFM | Method | Raw | Corrected | Reduction (%) | Fraction of achievable reduction | Floor / corrected |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    headline = far[far.scanner.eq("all") & far.floor_shift_px.eq(FAR_FLOOR_SHIFT)]
    for model in PFMS:
        for method in METHODS:
            row = headline[headline.model.eq(model) & headline.method.eq(method)].iloc[0]
            lines.append(
                f"| {PFM_LABELS[model]} | {METHOD_LABELS[method]} | {fmt(row.raw)} | "
                f"{fmt(row.corrected)} | "
                f"{fmt_ci(row.pct_reduction, row.pct_reduction_ci_low, row.pct_reduction_ci_high, 1)} | "
                f"{fmt_ci(row.fraction_achievable_reduction, row.far_ci_low, row.far_ci_high, 3)} | "
                f"{row.floor_over_corrected:.3f} |")
    lines += [
        "",
        "## Notes",
        "",
        "- The floor compares AT2 with AT2, so it contains misregistration only; the raw "
        "AT2-target distance contains scanner differences plus any residual misregistration "
        "(QC allowed up to 2 px). The pre-specified fraction-of-achievable-reduction uses the "
        "1-px floor; the other floors are in `achievable_reduction.csv`.",
        "- Fourier shifts are band-limited (ideal sinc) resampling followed by rounding to "
        "uint8; the 1- and 2-px shifts involve no interpolation.",
        "",
        "## Files",
        "",
        "- `identity/identity_check.csv.gz`, `identity/identity_summary.json`: identity gate.",
        "- `shards/<slide>.csv.gz`, `location_distances.csv.gz`: per location x shift x "
        "direction distances; `shards/<slide>.qc.csv.gz`, `location_qc.csv.gz`: image QC and "
        "embedding parity per location.",
        "- `slide_floor.csv`: slide means per shift. `floor_summary.csv`: floor per metric, "
        "shift (and direction) with CIs.",
        "- `floor_fraction.csv`: floor / raw per PFM x shift x scanner. "
        "`achievable_reduction.csv`: per PFM x method x scanner x floor shift. "
        "`image_counterparts.csv`: SSIM/LPIPS floors next to raw.",
        "- `reference_distances_used.csv.gz`: per-slide raw and corrected distances used. "
        "`qc.json`: all QC values.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    first = sub.add_parser("identity")
    first.add_argument("--workers", type=int,
                       default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    first.add_argument("--slides", nargs="*")
    first.add_argument("--output-dir", default=str(OUTPUT))
    second = sub.add_parser("embed")
    second.add_argument("--task-index", type=int,
                        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    second.add_argument("--task-count", type=int, default=4)
    second.add_argument("--slides", nargs="*")
    second.add_argument("--batch-size", type=int, default=32)
    second.add_argument("--output-dir", default=str(OUTPUT))
    second.add_argument("--overwrite", action="store_true")
    third = sub.add_parser("aggregate")
    third.add_argument("--output-dir", default=str(OUTPUT))
    third.add_argument("--shard-dir")
    third.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {"identity": identity, "embed": embed, "aggregate": aggregate}[args.stage](args)


if __name__ == "__main__":
    main()
