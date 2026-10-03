#!/usr/bin/env python3
"""Extract locked PanNormal and PLISM raw features for cross-encoder correction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from scripts.features.review_multiencoder_scanner import embed, load_encoder


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "outputs/scanner_batch_effect_analysis_2026-09-17"
GAN_REVIEW = ROOT / "outputs/gan_encoder_review_2026-09-25"
OUTPUT = ROOT / "outputs/feature_crossencoder_review_2026-09-25"
SCANNERS = ("at2", "versa", "akoya", "gt450", "s360", "s60")
EXTERNAL_SCANNERS = ("at2", "gt450", "s360", "s60")
OLD_RENDER = STUDY / "08_plism_external/pix2pix_gt450_to_at2/01_rendered_inputs"
NEW_RENDER = STUDY / "12_manuscript_completion/05_plism_external_correction/01_rendered_inputs"


def images_to_features(model, variant: str, name: str, images: np.ndarray,
                       size: int, mean: tuple[float, ...], std: tuple[float, ...],
                       batch_size: int) -> np.ndarray:
    images = np.asarray(images, dtype=np.uint8)
    if images.ndim != 4 or images.shape[1:] != (256, 256, 3):
        raise ValueError(f"unexpected image shape: {images.shape}")
    block = [image.astype(np.float32) / 255.0 for image in images]
    return embed(model, name, block, size, mean, std, batch_size)[variant]


def write_h5(path: Path, features: np.ndarray, location: np.ndarray,
             scanners: tuple[str, ...], **attributes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with h5py.File(temporary, "w") as store:
        store.create_dataset("features", data=features.astype(np.float32), compression="lzf")
        store.create_dataset("location_index", data=location.astype(np.int64))
        store.create_dataset("scanner_names", data=np.asarray(scanners, dtype="S12"))
        for key, value in attributes.items():
            store.attrs[key] = value
    temporary.replace(path)


def internal(model_name: str, fold: int, model, variant: str, size: int,
             mean: tuple[float, ...], std: tuple[float, ...],
             batch_size: int, max_slides: int | None) -> None:
    audit = json.loads((GAN_REVIEW / "audit.json").read_text())
    if audit.get("status") != "pass" or audit.get("direction_slide_pairs") != 515:
        raise ValueError("full GAN prediction audit did not pass")
    pairs = pd.read_csv(GAN_REVIEW / "prediction_pairs.csv", dtype={"slide_id": str})
    cohort = pd.read_csv(STUDY / "00_contract/cohort.csv", dtype={"slide_id": str})
    selected = cohort[cohort.fold == fold]
    if max_slides is not None:
        selected = selected.head(max_slides)
    for slide in selected.itertuples(index=False):
        source = None
        location = None
        targets = []
        for scanner in SCANNERS[1:]:
            row = pairs[(pairs.slide_id == str(slide.slide_id)) & (pairs.scanner == scanner)]
            if len(row) != 1 or int(row.iloc[0].fold) != fold:
                raise ValueError(f"prediction pair missing: {slide.slide_id} {scanner}")
            with h5py.File(row.iloc[0].pix2pix_path, "r") as store:
                observed = np.asarray(store["metadata/location_index"], dtype=np.int64)
                if len(observed) != 40:
                    raise ValueError(f"40-location training panel changed: {slide.slide_id}")
                if location is None:
                    location = observed
                    source = np.asarray(store["images/raw_source"], dtype=np.uint8)
                elif not np.array_equal(location, observed):
                    raise ValueError(f"location panel differs across scanners: {slide.slide_id}")
                target = np.asarray(store["images/real_target"], dtype=np.uint8)
                for index in (0, len(observed) // 2, -1):
                    if not np.array_equal(source[index], store["images/raw_source"][index]):
                        raise ValueError(f"AT2 source changed: {slide.slide_id} {scanner}")
            targets.append(target)
        with h5py.File(STUDY / "03_uni/shards" / f"{slide.slide_id}.h5", "r") as panel:
            locked = np.asarray(panel["location_index"], dtype=np.int64)
        if len(locked) != 20 or not set(locked).issubset(set(location)):
            raise ValueError(f"20-location held-out panel differs: {slide.slide_id}")
        blocks = [source, *targets]
        vectors = [images_to_features(model, variant, model_name, images,
                                     size, mean, std, batch_size) for images in blocks]
        feature_matrix = np.stack(vectors, axis=1)
        path = OUTPUT / "raw/internal" / variant / f"fold_{fold}" / f"{slide.slide_id}.h5"
        write_h5(path, feature_matrix, location, SCANNERS,
                 slide_id=str(slide.slide_id), tissue_type=slide.tissue_type,
                 fold=fold, cohort="PanNormal")
        print(f"{variant} internal fold {fold}: {slide.slide_id}", flush=True)


def render_path(scanner: str, section: str) -> Path:
    root = OLD_RENDER if scanner in {"at2", "gt450"} else NEW_RENDER
    return root / scanner.upper() / f"{section}.h5"


def external(model_name: str, model, variant: str, size: int,
             mean: tuple[float, ...], std: tuple[float, ...], batch_size: int) -> None:
    reference = STUDY / (
        "12_manuscript_completion/05_plism_external_correction/"
        "04_external_uni/features/gt450"
    )
    sections = sorted(path.stem for path in reference.glob("*.h5"))
    if len(sections) != 13:
        raise ValueError(f"expected 13 external sections, got {sections}")
    total = 0
    for section in sections:
        blocks = []
        location = None
        for scanner in EXTERNAL_SCANNERS:
            path = render_path(scanner, section)
            with h5py.File(path, "r") as store:
                observed = np.asarray(store["location"], dtype=np.int64)
                if location is None:
                    location = observed
                elif not np.array_equal(location, observed):
                    raise ValueError(f"external locations differ: {section} {scanner}")
                blocks.append(np.asarray(store["images"], dtype=np.uint8))
        with h5py.File(reference / f"{section}.h5", "r") as uni:
            if not np.array_equal(location, np.asarray(uni["location"], dtype=np.int64)):
                raise ValueError(f"external location index differs from UNI audit: {section}")
        vectors = [images_to_features(model, variant, model_name, images,
                                     size, mean, std, batch_size) for images in blocks]
        feature_matrix = np.stack(vectors, axis=1)
        path = OUTPUT / "raw/external" / variant / f"{section}.h5"
        write_h5(path, feature_matrix, location, EXTERNAL_SCANNERS,
                 section=section, cohort="PLISM")
        total += len(location)
        print(f"{variant} external: {section} n={len(location)}", flush=True)
    if total != 2387:
        raise ValueError(f"external location count changed: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=("uni2", "virchow2", "hoptimus1"))
    parser.add_argument("--task-index", required=True, type=int, choices=range(6))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-slides", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    model, size, mean, std, variants = load_encoder(
        args.model, ROOT / "outputs/encoder_review_2026-09-25/hf_cache"
    )
    if len(variants) != 1:
        raise ValueError("raw feature extraction expects one encoder output")
    variant = variants[0]
    if args.task_index == 5:
        external(args.model, model, variant, size, mean, std, args.batch_size)
    else:
        internal(args.model, args.task_index, model, variant, size, mean, std,
                 args.batch_size, args.max_slides)
