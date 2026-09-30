#!/usr/bin/env python3
"""RV02: frequency-band manipulation on every `set20` location in four frozen PFMs.

Re-runs the paper's band experiment (`scripts/frequency/bandwise_uni_experiment.py` for
UNI v1 and `scripts/frequency/review_bandwise_crosspfm.py` for UNI2-h, Virchow2 and
H-optimus-1) on the 20 `set20` locations per slide instead of positions 0, 10, 19.
The procedure is imported unchanged: AT2 is Reinhard-matched to each target with the
held-out slide's fold parameters; OD amplitude in the low-mid (0.10-0.30), mid (0.30-0.60)
and high (0.60-0.90 cycles/um) band is increased and decreased at doses 0.25 and 0.50 of
the smallest band OD RMS, matched in the 256x256 image before model resizing; the target
direction is the sign of the band-mean train-fold scanner gain. Each image set is rendered
once per slide and embedded by all four PFMs with each PFM's original embedding path and
batch size. Per-slide rows go to `shards/<model>/<slide>.csv.gz`, frozen-baseline agreement
checks to `shards/qc/<slide>.csv`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))
from scripts.frequency.bandwise_uni_experiment import (  # noqa: E402
    BANDS,
    DOSES,
    apply_reinhard,
    components,
    embed as embed_uni_v1,
    render_at_dose,
    target_like_sign,
)
from scripts.features.review_multiencoder_scanner import embed as embed_pfm, load_encoder  # noqa: E402


STUDY = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17"
COHORT = STUDY / "00_contract/cohort.csv"
PARAMETERS = STUDY / "02_correction/parameters.json"
UNI_PANEL = STUDY / "03_uni/shards"
PRIOR = PROJECT / "outputs/encoder_review_2026-09-25"
RAW40 = PROJECT / "outputs/feature_crossencoder_review_2026-09-25/raw/internal"
SET20 = PROJECT / "analysis/revision/results/location_sets/set20.csv"
OUTPUT = Path(__file__).resolve().parent / "results/band_manipulation_set20"
MODELS = ("uni_v1", "uni2", "virchow2", "hoptimus1")
BATCH_SIZE = {"uni_v1": 32, "uni2": 8, "virchow2": 8, "hoptimus1": 8}
TARGETS = ("versa", "akoya", "gt450", "s360", "s60")
SCANNERS = ("at2", *TARGETS)
AGREEMENT_MIN = 0.995  # threshold of the original frozen-baseline checks


def render_location(job: tuple[int, np.ndarray, dict]) -> tuple[list[np.ndarray], list[dict]]:
    """Images for one location, in the order of the original scripts."""
    location, paired, fitted = job
    source = paired[0]
    images: list[np.ndarray] = []
    slots: list[dict] = []
    for scanner in TARGETS:
        color = apply_reinhard(source, fitted[scanner])
        target = paired[SCANNERS.index(scanner)]
        slot = {"location_index": int(location), "scanner": scanner,
                "base": len(images), "target": len(images) + 1, "perturbations": []}
        images.extend((color.astype(np.float32) / 255.0, target.astype(np.float32) / 255.0))
        parts = components(color)
        minimum_rms = min(float(np.sqrt(np.mean(part * part))) for part in parts.values())
        for dose in DOSES:
            target_rms = dose * minimum_rms
            for band, part in parts.items():
                like_sign = target_like_sign(fitted[scanner], BANDS[band])
                for sign in (-1, 1):
                    changed, achieved, coefficient = render_at_dose(color, part, sign, target_rms)
                    slot["perturbations"].append({
                        "index": len(images), "dose_fraction": dose, "band": band,
                        "sign": sign, "target_like": sign == like_sign,
                        "target_rms_od": target_rms, "achieved_rms_od": achieved,
                        "coefficient": coefficient,
                    })
                    images.append(changed)
        slots.append(slot)
    return images, slots


def render_slide(pool, slide, locations: np.ndarray, fitted: dict) -> tuple[list[np.ndarray], list[dict]]:
    with h5py.File(str(slide.cache_path), "r") as cache:
        names = tuple(value.decode().lower() for value in cache["scanner_names"][:])
        if names != SCANNERS:
            raise ValueError(f"{slide.slide_id}: scanner order changed")
        paired = np.asarray(cache["images"][locations], dtype=np.uint8)
    jobs = [(int(location), paired[position], fitted) for position, location in enumerate(locations)]
    images: list[np.ndarray] = []
    records: list[dict] = []
    for part_images, part_slots in pool.imap(render_location, jobs):
        offset = len(images)
        for slot in part_slots:
            slot["base"] += offset
            slot["target"] += offset
            for item in slot["perturbations"]:
                item["index"] += offset
        images.extend(part_images)
        records.extend(part_slots)
    return images, records


def load_models(names: tuple[str, ...]) -> dict:
    loaded = {}
    for name in names:
        if name == "uni_v1":
            from prenorm.embedding import load_uni

            model, size, mean, std = load_uni(torch.device("cuda"))
        else:
            model, size, mean, std, variants = load_encoder(name, PRIOR / "hf_cache")
            if variants != (name,):
                raise ValueError(f"unexpected encoder variants: {variants}")
        loaded[name] = (model, size, mean, std)
        print(f"loaded {name}", flush=True)
    return loaded


def embed(name: str, loaded: dict, images: list[np.ndarray]) -> np.ndarray:
    model, size, mean, std = loaded[name]
    if name == "uni_v1":
        return embed_uni_v1(model, images, size, mean, std, torch.device("cuda"), BATCH_SIZE[name])
    return embed_pfm(model, name, images, size, mean, std, BATCH_SIZE[name])[name]


def distance(name: str, a: np.ndarray, b: np.ndarray) -> float:
    # UNI v1 script used 1 - dot; the cross-PFM script clipped the dot product.
    value = float(np.dot(a, b))
    return 1.0 - (value if name == "uni_v1" else float(np.clip(value, -1.0, 1.0)))


def baseline_checks(name: str, slide, vectors: np.ndarray, records: list[dict]) -> list[dict]:
    slide_id = str(slide.slide_id)
    rows = []

    def add(record: dict, check: str, reference: np.ndarray, key: str, fatal: bool) -> None:
        rows.append({"model": name, "slide_id": slide_id, "location_index": record["location_index"],
                     "scanner": record["scanner"], "check": check, "fatal": fatal,
                     "cosine": float(np.dot(reference, vectors[record[key]]))})

    if name == "uni_v1":
        with h5py.File(UNI_PANEL / f"{slide_id}.h5", "r") as panel:
            location = [int(value) for value in panel["location_index"]]
            conditions = [value.decode() for value in panel["condition_names"][:]]
            features = np.asarray(panel["features"], dtype=np.float32)
        for record in records:
            row = location.index(record["location_index"])
            add(record, "reinhard_vs_03uni", features[row, conditions.index(f"reinhard:{record['scanner']}")], "base", True)
            add(record, "target_vs_03uni", features[row, conditions.index(f"target:{record['scanner']}")], "target", True)
        return rows
    job = "uni2_pair" if name == "uni2" else name
    path = PRIOR / "features" / job / f"fold_{int(slide.fold)}" / f"{slide_id}_{name}.npz"
    with np.load(path, allow_pickle=False) as previous:
        keys = {(int(loc), str(scanner)): row for row, (loc, scanner)
                in enumerate(zip(previous["location_index"], previous["scanner"]))}
        reinhard = np.asarray(previous["reinhard"], dtype=np.float32)
        target = np.asarray(previous["target"], dtype=np.float32)
    matched = 0
    for record in records:
        row = keys.get((record["location_index"], record["scanner"]))
        if row is not None:
            matched += 1
            add(record, "reinhard_vs_prior3", reinhard[row], "base", True)
            add(record, "target_vs_prior3", target[row], "target", True)
    if matched != len(keys):
        raise ValueError(f"{slide_id} {name}: prior 3-location baseline not covered ({matched}/{len(keys)})")
    with h5py.File(RAW40 / name / f"fold_{int(slide.fold)}" / f"{slide_id}.h5", "r") as store:
        location = [int(value) for value in store["location_index"]]
        scanners = tuple(value.decode() for value in store["scanner_names"][:])
        features = np.asarray(store["features"], dtype=np.float32)
    for record in records:
        add(record, "target_vs_raw40", features[location.index(record["location_index"]),
                                                 scanners.index(record["scanner"])], "target", False)
    return rows


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    temporary.replace(path)


def process_slide(pool, slide, chosen: pd.DataFrame, fitted: dict, loaded: dict) -> None:
    slide_id = str(slide.slide_id)
    pending = [name for name in loaded if not (OUTPUT / "shards" / name / f"{slide_id}.csv.gz").exists()]
    if not pending:
        print(f"skip {slide_id}", flush=True)
        return
    locations = chosen.location_index.to_numpy(int)
    probe = dict(zip(chosen.location_index.astype(int), chosen.paper_band_probe.astype(bool)))
    start = time.time()
    images, records = render_slide(pool, slide, locations, fitted)
    expected = len(locations) * len(TARGETS) * (2 + len(DOSES) * len(BANDS) * 2)
    if len(images) != expected:
        raise ValueError(f"{slide_id}: expected {expected} images, got {len(images)}")
    timing = {"render_s": round(time.time() - start, 1)}
    qc_rows = []
    for name in pending:
        start = time.time()
        vectors = embed(name, loaded, images)
        timing[f"{name}_embed_s"] = round(time.time() - start, 1)
        if vectors.shape[0] != len(images) or not np.isfinite(vectors).all():
            raise ValueError(f"{slide_id} {name}: invalid embeddings {vectors.shape}")
        checks = baseline_checks(name, slide, vectors, records)
        qc_rows.extend(checks)
        failed = [row for row in checks if row["fatal"] and row["cosine"] < AGREEMENT_MIN]
        if failed:
            write_frame(pd.DataFrame(qc_rows), OUTPUT / "shards/qc" / f"{slide_id}.failed.csv")
            raise ValueError(f"{slide_id} {name}: frozen baseline agreement below {AGREEMENT_MIN}: {failed[:3]}")
        rows = []
        for slot in records:
            base, target = vectors[slot["base"]], vectors[slot["target"]]
            baseline_distance = distance(name, base, target)
            for item in slot["perturbations"]:
                changed = vectors[item["index"]]
                changed_distance = distance(name, changed, target)
                rows.append({
                    "model": name, "slide_id": slide_id, "tissue_type": slide.tissue_type,
                    "fold": int(slide.fold), "location_index": slot["location_index"],
                    "paper_band_probe": probe[slot["location_index"]], "scanner": slot["scanner"],
                    **{key: value for key, value in item.items() if key != "index"},
                    "embedding_displacement": distance(name, base, changed),
                    "baseline_target_distance": baseline_distance,
                    "changed_target_distance": changed_distance,
                    "target_gain": baseline_distance - changed_distance,
                })
        write_frame(pd.DataFrame(rows), OUTPUT / "shards" / name / f"{slide_id}.csv.gz")
    qc_path = OUTPUT / "shards/qc" / f"{slide_id}.csv"
    if qc_path.exists():
        previous = pd.read_csv(qc_path, dtype={"slide_id": str})
        qc_rows = previous[~previous.model.isin(pending)].to_dict("records") + qc_rows
    write_frame(pd.DataFrame(qc_rows), qc_path)
    print(f"saved {slide_id}: {len(images)} images, models {pending}, {json.dumps(timing)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--max-slides", type=int, default=0)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = parser.parse_args()
    models = tuple(args.models.split(","))
    if not set(models) <= set(MODELS):
        parser.error(f"models must be among {MODELS}")
    if not torch.cuda.is_available() or not 0 <= args.task_index < args.task_count:
        raise RuntimeError("valid CUDA task index required")
    cohort = pd.read_csv(COHORT, dtype={"slide_id": str})
    set20 = pd.read_csv(SET20, dtype={"slide_id": str})
    parameters = json.loads(PARAMETERS.read_text())["fold"]
    slides = cohort.iloc[args.task_index::args.task_count]
    if args.max_slides:
        slides = slides.iloc[:args.max_slides]
    # Fork the rendering workers before any CUDA context exists.
    pool = mp.get_context("fork").Pool(max(1, args.workers))
    loaded = load_models(models)
    for slide in slides.itertuples(index=False):
        chosen = set20[set20.slide_id.eq(str(slide.slide_id))].sort_values("location_index")
        if len(chosen) != 20 or int(chosen.fold.iloc[0]) != int(slide.fold):
            raise ValueError(f"{slide.slide_id}: set20 locations or fold inconsistent")
        process_slide(pool, slide, chosen, parameters[str(int(slide.fold))], loaded)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
