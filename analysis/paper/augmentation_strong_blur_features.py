#!/usr/bin/env python3
"""Extend Gaussian blur from every scanner into an extreme image-loss range."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from augmentation_ood_uni import embed_images  # noqa: E402
from prenorm.embedding import load_uni  # noqa: E402


AUGMENTATION = PROJECT / "outputs/augmentation_ood_v1"
COMMON = Path(__file__).resolve().parent / "results/augmentation_common_uni_space"
OUTPUT = Path(__file__).resolve().parent / "results/augmentation_strong_blur"
SIGMAS = (0.0, 3.0, 6.0, 12.0, 24.0, 48.0)


def gradients(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gy, gx = np.gradient(gray)
    return np.stack((gx, gy), axis=-1).reshape(-1)


def score_detail(source: np.ndarray, changed: np.ndarray) -> tuple[float, float]:
    left = gradients(source)
    right = gradients(changed)
    energy_ratio = float((right @ right) / max(float(left @ left), 1e-12))
    left -= left.mean()
    right -= right.mean()
    ncc = float((left @ right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12))
    return energy_ratio, ncc


def process_slide(slide, model, size, mean, std) -> None:
    slide_id = str(slide.slide_id)
    path = OUTPUT / "shards" / f"{slide_id}.h5"
    if path.exists():
        print(f"skip {slide_id}", flush=True)
        return
    with h5py.File(COMMON / "shards" / f"{slide_id}.h5", "r") as store:
        locations = np.asarray(store["location_index"], dtype=int)
        raw = np.asarray(store["raw_features"], dtype=np.float32)
    with h5py.File(str(slide.cache_path), "r") as store:
        images = np.asarray(store["images"][locations], dtype=np.uint8)
    if images.shape != (3, 6, 256, 256, 3) or raw.shape != (3, 6, 1024):
        raise ValueError(f"incomplete paired slide: {slide_id}")
    candidates = []
    lookup = []
    detail = np.zeros((3, 6, len(SIGMAS), 2), dtype=np.float32)
    detail[:, :, 0] = (1.0, 1.0)
    for location in range(3):
        for scanner in range(6):
            source = images[location, scanner]
            for step, sigma in enumerate(SIGMAS[1:], start=1):
                blurred = cv2.GaussianBlur(source, (0, 0), sigma)
                detail[location, scanner, step] = score_detail(source, blurred)
                candidates.append(blurred)
                lookup.append((location, scanner, step))
    embeddings = embed_images(
        model, np.stack(candidates), size, mean, std, torch.device("cuda"), 32
    )
    features = np.empty((3, 6, len(SIGMAS), 1024), dtype=np.float32)
    features[:, :, 0] = raw
    for index, value in zip(lookup, embeddings):
        features[index] = value
    if not np.isfinite(features).all() or not np.isfinite(detail).all():
        raise ValueError(f"nonfinite strong-blur output: {slide_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with h5py.File(temporary, "w") as store:
        store.attrs["slide_id"] = slide_id
        store.attrs["tissue_type"] = str(slide.tissue_type)
        store.create_dataset("location_index", data=locations)
        store.create_dataset("sigmas", data=np.asarray(SIGMAS, dtype=np.float32))
        store.create_dataset("features", data=features, compression="lzf")
        store.create_dataset("detail_metrics", data=detail, compression="lzf")
    temporary.replace(path)
    print(f"saved {slide_id}: {len(candidates)} strong-blur images", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--max-slides", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not 0 <= args.task_index < args.task_count:
        raise RuntimeError("valid CUDA task required")
    cohort = pd.read_csv(AUGMENTATION / "00_contract/cohort.csv", dtype={"slide_id": str})
    slides = cohort.iloc[args.task_index::args.task_count]
    if args.max_slides:
        slides = slides.iloc[:args.max_slides]
    model, size, mean, std = load_uni(torch.device("cuda"))
    for slide in slides.itertuples(index=False):
        process_slide(slide, model, size, mean, std)


if __name__ == "__main__":
    main()
