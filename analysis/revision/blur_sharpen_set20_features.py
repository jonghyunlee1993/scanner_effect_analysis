#!/usr/bin/env python3
"""RV01: UNI v1 blur/sharpen embeddings of all six scanners at the 20 `set20` locations.

Re-runs the paper's single-factor trajectories
(`analysis/paper/augmentation_common_uni_features.py`, `augmentation_strong_blur_features.py`)
on every `set20` location instead of three per slide. Operations and strengths follow the
protocol: Gaussian blur sigma = 0.5, 1, 2, 3, 6 (cv2, as `prenorm.augmentation.transform`)
and unsharp mask alpha = 0.25, 0.5, 1, 2 (alpha * (I - G_{sigma=1}(I)), clipped). As in the
paper, the untransformed state uses the stored `augmentation_ood_v1` UNI features; the raw
images are re-embedded as well so that extraction parity can be checked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "analysis/paper"))
from augmentation_ood_uni import embed_images  # noqa: E402
from augmentation_strong_blur_features import score_detail  # noqa: E402
from prenorm.augmentation import transform  # noqa: E402
from prenorm.embedding import load_uni  # noqa: E402


COHORT = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17/00_contract/cohort.csv"
STORED_UNI = PROJECT / "outputs/augmentation_ood_v1/03_pfm_uni/shards"
SET20 = PROJECT / "analysis/revision/results/location_sets/set20.csv"
OUTPUT = Path(__file__).resolve().parent / "results/blur_sharpen_set20"
SCANNERS = ("at2", "versa", "akoya", "gt450", "s360", "s60")
BLUR_SIGMAS = (0.0, 0.5, 1.0, 2.0, 3.0, 6.0)
SHARPEN_ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)
BATCH_SIZE = 32


def extract_slide(slide, locations: np.ndarray, probe: np.ndarray, model, size, mean, std) -> None:
    slide_id = str(slide.slide_id)
    path = OUTPUT / "shards" / f"{slide_id}.h5"
    if path.exists():
        print(f"skip {slide_id}", flush=True)
        return
    with h5py.File(STORED_UNI / f"{slide_id}.h5", "r") as store:
        stored_locations = np.asarray(store["location_index"], dtype=int)
        names = [value.decode() for value in store["condition_names"][:]]
        stored = np.asarray(store["features"], dtype=np.float32)
    if not np.array_equal(stored_locations, locations):
        raise ValueError(f"{slide_id}: stored UNI locations differ from set20")
    raw = np.stack([
        stored[:, names.index("source" if scanner == "at2" else f"target:{scanner}")]
        for scanner in SCANNERS
    ], axis=1)
    with h5py.File(str(slide.cache_path), "r") as store:
        cache_scanners = tuple(value.decode().lower() for value in store["scanner_names"][:])
        images = np.asarray(store["images"][locations], dtype=np.uint8)
    n_loc = len(locations)
    if cache_scanners != SCANNERS or images.shape != (n_loc, 6, 256, 256, 3) or raw.shape != (n_loc, 6, 1024):
        raise ValueError(f"{slide_id}: invalid paired data")

    generated, lookup = [], []
    blur_detail = np.zeros((n_loc, 6, len(BLUR_SIGMAS), 2), dtype=np.float32)
    sharpen_detail = np.zeros((n_loc, 6, len(SHARPEN_ALPHAS), 2), dtype=np.float32)
    blur_detail[:, :, 0] = (1.0, 1.0)
    sharpen_detail[:, :, 0] = (1.0, 1.0)
    for loc in range(n_loc):
        for scanner in range(6):
            source = images[loc, scanner]
            generated.append(source)
            lookup.append(("raw", loc, scanner, 0))
            for step, sigma in enumerate(BLUR_SIGMAS[1:], start=1):
                changed = transform(source, "blur_sigma", sigma)
                blur_detail[loc, scanner, step] = score_detail(source, changed)
                generated.append(changed)
                lookup.append(("blur", loc, scanner, step))
            for step, alpha in enumerate(SHARPEN_ALPHAS[1:], start=1):
                changed = transform(source, "sharpen_amount", alpha)
                sharpen_detail[loc, scanner, step] = score_detail(source, changed)
                generated.append(changed)
                lookup.append(("sharpen", loc, scanner, step))
    vectors = embed_images(model, np.stack(generated), size, mean, std, torch.device("cuda"), BATCH_SIZE)
    recomputed = np.empty((n_loc, 6, 1024), dtype=np.float32)
    blur = np.empty((n_loc, 6, len(BLUR_SIGMAS), 1024), dtype=np.float32)
    sharpen = np.empty((n_loc, 6, len(SHARPEN_ALPHAS), 1024), dtype=np.float32)
    blur[:, :, 0] = raw
    sharpen[:, :, 0] = raw
    for (kind, loc, scanner, step), vector in zip(lookup, vectors):
        if kind == "raw":
            recomputed[loc, scanner] = vector
        elif kind == "blur":
            blur[loc, scanner, step] = vector
        else:
            sharpen[loc, scanner, step] = vector
    arrays = (recomputed, blur, sharpen, blur_detail, sharpen_detail)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{slide_id}: nonfinite output")
    parity = np.einsum("lsd,lsd->ls", recomputed, raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with h5py.File(temporary, "w") as store:
        store.attrs["slide_id"] = slide_id
        store.attrs["tissue_type"] = str(slide.tissue_type)
        store.attrs["fold"] = int(slide.fold)
        store.create_dataset("location_index", data=locations)
        store.create_dataset("paper_blur_probe", data=probe)
        store.create_dataset("scanner_names", data=np.asarray(SCANNERS, dtype="S10"))
        store.create_dataset("blur_sigmas", data=np.asarray(BLUR_SIGMAS, dtype=np.float32))
        store.create_dataset("sharpen_alphas", data=np.asarray(SHARPEN_ALPHAS, dtype=np.float32))
        store.create_dataset("raw_features", data=raw, compression="lzf")
        store.create_dataset("raw_recomputed", data=recomputed, compression="lzf")
        store.create_dataset("blur_features", data=blur, compression="lzf")
        store.create_dataset("sharpen_features", data=sharpen, compression="lzf")
        store.create_dataset("blur_detail", data=blur_detail, compression="lzf")
        store.create_dataset("sharpen_detail", data=sharpen_detail, compression="lzf")
    temporary.replace(path)
    print(f"saved {slide_id}: {len(generated)} images; raw parity min {parity.min():.5f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--max-slides", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not 0 <= args.task_index < args.task_count:
        raise RuntimeError("valid CUDA task index required")
    cohort = pd.read_csv(COHORT, dtype={"slide_id": str})
    set20 = pd.read_csv(SET20, dtype={"slide_id": str})
    slides = cohort.iloc[args.task_index::args.task_count]
    if args.max_slides:
        slides = slides.iloc[:args.max_slides]
    model, size, mean, std = load_uni(torch.device("cuda"))
    for slide in slides.itertuples(index=False):
        chosen = set20[set20.slide_id.eq(str(slide.slide_id))].sort_values("location_index")
        if len(chosen) != 20:
            raise ValueError(f"{slide.slide_id}: set20 must have 20 locations")
        extract_slide(slide, chosen.location_index.to_numpy(int),
                      chosen.paper_blur_probe.to_numpy(bool), model, size, mean, std)


if __name__ == "__main__":
    main()
