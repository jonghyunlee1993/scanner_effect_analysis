#!/usr/bin/env python3
"""Evaluate locked Pix2Pix/CycleGAN outputs with frozen image encoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from scripts.features.review_multiencoder_scanner import embed, load_encoder, cosine_distance


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "outputs/scanner_batch_effect_analysis_2026-09-17"
OUTPUT = ROOT / "outputs/gan_encoder_review_2026-09-25"
SCANNERS = ("versa", "akoya", "gt450", "s360", "s60")
ARMS = ("source", "target", "pix2pix", "cyclegan")


def unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-12)


def run(model_name: str, fold: int, batch_size: int, max_slides: int | None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    audit = json.loads((OUTPUT / "audit.json").read_text())
    if audit.get("status") != "pass" or audit.get("direction_slide_pairs") != 515:
        raise ValueError("full paired-prediction audit did not pass")
    pairs = pd.read_csv(OUTPUT / "prediction_pairs.csv", dtype={"slide_id": str})
    cohort = pd.read_csv(STUDY / "00_contract/cohort.csv", dtype={"slide_id": str})
    selected = cohort[cohort.fold == fold]
    if max_slides is not None:
        selected = selected.head(max_slides)
    model, size, mean, std, variants = load_encoder(
        model_name, ROOT / "outputs/encoder_review_2026-09-25/hf_cache"
    )
    if len(variants) != 1 and model_name != "conch":
        raise ValueError("review GAN endpoint expects one embedding per model")
    variant = "conch_pre" if model_name == "conch" else variants[0]
    rows = []
    feature_root = OUTPUT / "slide_means" / variant / f"fold_{fold}"
    feature_root.mkdir(parents=True, exist_ok=True)
    for slide in selected.itertuples(index=False):
        with h5py.File(STUDY / "03_uni/shards" / f"{slide.slide_id}.h5", "r") as panel:
            locked = np.asarray(panel["location_index"], dtype=np.int64)
        if len(locked) != 20:
            raise ValueError(f"locked 20-location panel changed for {slide.slide_id}")
        slide_means = {arm: [] for arm in ARMS}
        for scanner in SCANNERS:
            record = pairs[(pairs.slide_id == str(slide.slide_id)) &
                           (pairs.scanner == scanner)]
            if len(record) != 1 or int(record.iloc[0].fold) != fold:
                raise ValueError(f"missing audited pair: {slide.slide_id} {scanner}")
            record = record.iloc[0]
            with h5py.File(record.pix2pix_path, "r") as pix, h5py.File(record.cyclegan_path, "r") as cyc:
                locations = np.asarray(pix["metadata/location_index"], dtype=np.int64)
                other_locations = np.asarray(cyc["metadata/location_index"], dtype=np.int64)
                if not np.array_equal(locations, other_locations):
                    raise ValueError(f"GAN location order changed: {slide.slide_id} {scanner}")
                lookup = {int(value): index for index, value in enumerate(locations)}
                indices = np.asarray([lookup[int(location)] for location in locked], dtype=np.int64)
                arrays = {
                    "source": np.asarray(pix["images/raw_source"])[indices],
                    "target": np.asarray(pix["images/real_target"])[indices],
                    "pix2pix": np.asarray(pix["images/generated"])[indices],
                    "cyclegan": np.asarray(cyc["images/generated"])[indices],
                }
                if (not np.array_equal(arrays["source"],
                                       np.asarray(cyc["images/raw_source"])[indices]) or
                    not np.array_equal(arrays["target"],
                                       np.asarray(cyc["images/real_target"])[indices])):
                    raise ValueError(f"source/target image mismatch: {slide.slide_id} {scanner}")
            images = []
            slots = {}
            for position, location in enumerate(locked):
                for arm in ARMS:
                    slots[(int(location), arm)] = len(images)
                    images.append(arrays[arm][position].astype(np.float32) / 255.0)
            features = embed(model, model_name, images, size, mean, std, batch_size)[variant]
            for arm in ARMS:
                vectors = np.stack([features[slots[(int(location), arm)]] for location in locked])
                slide_means[arm].append(unit(vectors.mean(axis=0)))
            for location in locked:
                source = features[slots[(int(location), "source")]]
                target = features[slots[(int(location), "target")]]
                pix = features[slots[(int(location), "pix2pix")]]
                cyc = features[slots[(int(location), "cyclegan")]]
                raw_distance = cosine_distance(source, target)
                pix_distance = cosine_distance(pix, target)
                cyc_distance = cosine_distance(cyc, target)
                rows.append({
                    "model": variant, "slide_id": str(slide.slide_id),
                    "tissue_type": slide.tissue_type, "fold": fold,
                    "scanner": scanner, "location_index": int(location),
                    "raw_distance": raw_distance,
                    "pix2pix_distance": pix_distance,
                    "cyclegan_distance": cyc_distance,
                    "pix2pix_gain": raw_distance - pix_distance,
                    "cyclegan_gain": raw_distance - cyc_distance,
                })
        np.savez_compressed(
            feature_root / f"{slide.slide_id}.npz",
            slide_id=str(slide.slide_id), tissue_type=slide.tissue_type,
            scanners=np.array(SCANNERS),
            **{arm: np.stack(values).astype(np.float32)
               for arm, values in slide_means.items()},
        )
        print(f"{variant} fold {fold}: {slide.slide_id} complete", flush=True)
    shard_root = OUTPUT / "shards" / variant
    shard_root.mkdir(parents=True, exist_ok=True)
    path = shard_root / f"fold_{fold}.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    print(json.dumps({"path": str(path), "model": variant,
                      "slides": len(selected), "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=("uni_v1", "uni2", "virchow2", "hoptimus1", "conch"))
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-slides", type=int)
    args = parser.parse_args()
    run(args.model, args.fold, args.batch_size, args.max_slides)
