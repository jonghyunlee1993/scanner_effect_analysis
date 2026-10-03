#!/usr/bin/env python3
"""Repeat the UNI-v1 Figure 1 band experiment for three additional frozen PFMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from scripts.frequency.bandwise_uni_experiment import (
    BANDS,
    DOSES,
    apply_reinhard,
    components,
    render_at_dose,
    target_like_sign,
)
from scripts.features.review_multiencoder_scanner import embed, load_encoder


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "outputs/scanner_batch_effect_analysis_2026-09-17"
PRIOR = ROOT / "outputs/encoder_review_2026-09-25"
OUTPUT = ROOT / "outputs/bandwise_crosspfm_2026-09-25"
MODELS = ("uni2", "virchow2", "hoptimus1")
TARGETS = ("versa", "akoya", "gt450", "s360", "s60")
SCANNERS = ("at2", *TARGETS)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(a, b), -1.0, 1.0))


def slide_images(slide, fitted: dict) -> tuple[list[np.ndarray], list[dict]]:
    slide_id = str(slide.slide_id)
    with h5py.File(STUDY / "03_uni/shards" / f"{slide_id}.h5", "r") as panel:
        locked = np.asarray(panel["location_index"], dtype=np.int64)
    if len(locked) != 20:
        raise ValueError(f"{slide_id}: expected 20 locked locations")
    locations = locked[[0, len(locked) // 2, -1]]
    with h5py.File(slide.cache_path, "r") as cache:
        names = tuple(value.decode().lower() for value in cache["scanner_names"][:])
        if names != SCANNERS:
            raise ValueError(f"{slide_id}: scanner order changed")
        paired = np.asarray(cache["images"][locations], dtype=np.uint8)

    images: list[np.ndarray] = []
    records: list[dict] = []
    for location_pos, location in enumerate(locations):
        source = paired[location_pos, 0]
        for scanner in TARGETS:
            color = apply_reinhard(source, fitted[scanner])
            target = paired[location_pos, names.index(scanner)]
            slot = {
                "location_index": int(location),
                "scanner": scanner,
                "base": len(images),
                "target": len(images) + 1,
                "perturbations": [],
            }
            images.extend((color.astype(np.float32) / 255.0,
                           target.astype(np.float32) / 255.0))
            parts = components(color)
            minimum_rms = min(float(np.sqrt(np.mean(part * part)))
                              for part in parts.values())
            for dose in DOSES:
                target_rms = dose * minimum_rms
                for band, part in parts.items():
                    like_sign = target_like_sign(fitted[scanner], BANDS[band])
                    for sign in (-1, 1):
                        changed, achieved, coefficient = render_at_dose(
                            color, part, sign, target_rms
                        )
                        slot["perturbations"].append({
                            "index": len(images), "dose_fraction": dose,
                            "band": band, "sign": sign,
                            "target_like": sign == like_sign,
                            "target_rms_od": target_rms,
                            "achieved_rms_od": achieved,
                            "coefficient": coefficient,
                        })
                        images.append(changed)
            records.append(slot)
    return images, records


def check_existing_baseline(model_name: str, fold: int, slide_id: str,
                            features: np.ndarray, records: list[dict]) -> None:
    job_name = "uni2_pair" if model_name == "uni2" else model_name
    path = PRIOR / "features" / job_name / f"fold_{fold}" / f"{slide_id}_{model_name}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as previous:
        for row, slot in enumerate(records):
            if (int(previous["location_index"][row]) != slot["location_index"]
                    or str(previous["scanner"][row]) != slot["scanner"]):
                raise ValueError(f"{slide_id}: baseline location mismatch")
            for arm, key in (("reinhard", "base"), ("target", "target")):
                agreement = float(np.dot(previous[arm][row], features[slot[key]]))
                if agreement < 0.995:
                    raise ValueError(
                        f"{slide_id} {slot['scanner']} {arm}: "
                        f"frozen baseline agreement {agreement:.5f}"
                    )


def run(task_index: int, batch_size: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    model_name = MODELS[task_index // 5]
    fold = task_index % 5
    torch.set_num_threads(2)
    model, size, mean, std, variants = load_encoder(model_name, PRIOR / "hf_cache")
    if variants != (model_name,):
        raise ValueError(f"unexpected encoder variants: {variants}")
    cohort = pd.read_csv(STUDY / "00_contract/cohort.csv", dtype={"slide_id": str})
    slides = cohort[cohort.fold == fold]
    fitted = json.loads((STUDY / "02_correction/parameters.json").read_text())["fold"][str(fold)]
    rows: list[dict] = []
    for slide in slides.itertuples(index=False):
        images, records = slide_images(slide, fitted)
        vectors = embed(model, model_name, images, size, mean, std, batch_size)[model_name]
        check_existing_baseline(model_name, fold, str(slide.slide_id), vectors, records)
        for slot in records:
            base, target = vectors[slot["base"]], vectors[slot["target"]]
            baseline_distance = distance(base, target)
            for item in slot["perturbations"]:
                changed = vectors[item["index"]]
                changed_distance = distance(changed, target)
                rows.append({
                    "model": model_name, "slide_id": str(slide.slide_id),
                    "tissue_type": slide.tissue_type, "fold": fold,
                    "location_index": slot["location_index"],
                    "scanner": slot["scanner"],
                    **{key: value for key, value in item.items() if key != "index"},
                    "embedding_displacement": distance(base, changed),
                    "baseline_target_distance": baseline_distance,
                    "changed_target_distance": changed_distance,
                    "target_gain": baseline_distance - changed_distance,
                })
        print(f"{model_name} fold {fold}: {slide.slide_id}, {len(rows)} rows", flush=True)
    out_dir = OUTPUT / "shards" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fold_{fold}.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    print(json.dumps({"path": str(path), "model": model_name, "fold": fold,
                      "slides": len(slides), "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True, choices=range(15))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    run(args.task_index, args.batch_size)
