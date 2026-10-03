#!/usr/bin/env python3
"""Embed three one-parameter sweeps from every PanNormal scanner image."""

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
from augmentation_ood_study import SCANNERS  # noqa: E402
from augmentation_ood_uni import embed_images  # noqa: E402
from prenorm.embedding import load_uni  # noqa: E402
from prenorm.augmentation import SWEEPS, transform  # noqa: E402


AUGMENTATION = PROJECT / "outputs/augmentation_ood_v1"
OUTPUT = Path(__file__).resolve().parent / "results/augmentation_common_uni_space"
LOCATION_OFFSETS = (0, 9, 19)


def extract_slide(slide, model, size, mean, std) -> None:
    slide_id = str(slide.slide_id)
    path = OUTPUT / "shards" / f"{slide_id}.h5"
    if path.exists():
        print(f"skip {slide_id}", flush=True)
        return
    with h5py.File(AUGMENTATION / "03_pfm_uni/shards" / f"{slide_id}.h5", "r") as store:
        available = np.asarray(store["location_index"], dtype=int)
        names = [value.decode() for value in store["condition_names"][:]]
        locations = available[list(LOCATION_OFFSETS)]
        previous = np.asarray(store["features"][list(LOCATION_OFFSETS)], dtype=np.float32)
    if len(available) != 20 or not np.array_equal(locations, np.sort(locations)):
        raise ValueError(f"invalid UNI location selection: {slide_id}")
    name_to_index = {name: index for index, name in enumerate(names)}
    raw = np.stack([
        previous[:, name_to_index["source" if scanner == "at2" else f"target:{scanner}"]]
        for scanner in SCANNERS
    ], axis=1)
    with h5py.File(str(slide.cache_path), "r") as store:
        images = np.asarray(store["images"][locations], dtype=np.uint8)
    if images.shape != (3, 6, 256, 256, 3) or raw.shape != (3, 6, 1024):
        raise ValueError(f"invalid paired data: {slide_id}")

    generated_images = []
    indices = []
    for loc_index in range(3):
        for scanner_index in range(6):
            source = images[loc_index, scanner_index]
            for operation_index, (_, parameter, strengths, _) in enumerate(SWEEPS):
                for step in range(1, 5):
                    generated_images.append(transform(source, parameter, float(strengths[step])))
                    indices.append((loc_index, scanner_index, operation_index, step))
    vectors = embed_images(
        model, np.stack(generated_images), size, mean, std, torch.device("cuda"), 32
    )
    result = np.empty((3, 6, 3, 5, 1024), dtype=np.float32)
    result[:, :, :, 0] = raw[:, :, None, :]
    for index, feature in zip(indices, vectors):
        result[index] = feature
    if not np.isfinite(result).all():
        raise ValueError(f"nonfinite embedding: {slide_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with h5py.File(temporary, "w") as store:
        store.attrs["slide_id"] = slide_id
        store.attrs["tissue_type"] = str(slide.tissue_type)
        store.create_dataset("location_index", data=locations)
        store.create_dataset("scanner_names", data=np.asarray(SCANNERS, dtype="S10"))
        store.create_dataset("operation_names", data=np.asarray([row[0] for row in SWEEPS], dtype="S20"))
        store.create_dataset("raw_features", data=raw, compression="lzf")
        store.create_dataset("augmented_features", data=result, compression="lzf")
    temporary.replace(path)
    print(f"saved {slide_id}: {len(generated_images)} augmented images", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--max-slides", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not 0 <= args.task_index < args.task_count:
        raise RuntimeError("valid CUDA task index required")
    cohort = pd.read_csv(AUGMENTATION / "00_contract/cohort.csv", dtype={"slide_id": str})
    slides = cohort.iloc[args.task_index::args.task_count]
    if args.max_slides:
        slides = slides.iloc[:args.max_slides]
    model, size, mean, std = load_uni(torch.device("cuda"))
    for slide in slides.itertuples(index=False):
        extract_slide(slide, model, size, mean, std)


if __name__ == "__main__":
    main()
