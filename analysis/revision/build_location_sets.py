#!/usr/bin/env python3
"""RV-P0a: write the locked 40- and 20-location sets used by every revision analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
BATCH = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17"
PHENOTYPES = PROJECT / "outputs/final_image_study_v1/02_image_phenotypes/shards"
SET40 = BATCH / "06_learned_baselines/00_contract/locked_image_evaluation_index.csv.gz"
FOLDS = PROJECT / "outputs/augmentation_ood_v1/00_contract/slide_folds.csv"
UNI_SHARDS = BATCH / "03_uni/shards"
OUTPUT = Path(__file__).resolve().parent / "results/location_sets"
# Positions within the sorted 20 used by the paper's three-location probes.
BLUR_OFFSETS = (0, 9, 19)
BAND_OFFSETS = (0, 10, 19)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replicate_groups(slide_id: str) -> np.ndarray:
    with h5py.File(PHENOTYPES / slide_id / "location_metrics.h5", "r") as store:
        return np.asarray(store["replicate_id"], dtype=int)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    set40 = pd.read_csv(SET40, dtype={"slide_id": str})
    folds = pd.read_csv(FOLDS, dtype={"slide_id": str}).set_index("slide_id")["fold"]
    if set40.slide_id.nunique() != 103 or not (set40.groupby("slide_id").size() == 40).all():
        raise ValueError("set40 must hold 40 locations for each of 103 slides")
    if not (set40.set_index("slide_id").fold.groupby(level=0).first() == folds).all():
        raise ValueError("set40 folds disagree with the locked fold file")

    rows = []
    for slide_id, group in set40.groupby("slide_id", sort=True):
        replicate = replicate_groups(slide_id)
        available = sorted(int(index) for index in group.location_index)
        chosen = []
        for replicate_id in range(10):
            chosen.extend([index for index in available if replicate[index] == replicate_id][:2])
        chosen = sorted(chosen)
        with h5py.File(UNI_SHARDS / f"{slide_id}.h5", "r") as store:
            stored = sorted(int(index) for index in store["location_index"])
        if chosen != stored:
            raise ValueError(f"{slide_id}: set20 rule disagrees with the stored UNI locations")
        blur = {chosen[offset] for offset in BLUR_OFFSETS}
        band = {chosen[offset] for offset in BAND_OFFSETS}
        for record in group.sort_values("location_index").itertuples():
            index = int(record.location_index)
            rows.append({
                "slide_id": slide_id,
                "tissue_type": record.tissue_type,
                "fold": int(record.fold),
                "location_index": index,
                "source_index": int(record.source_index),
                "replicate_group": int(replicate[index]),
                "in_set20": index in chosen,
                "paper_blur_probe": index in blur,
                "paper_band_probe": index in band,
            })

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "set40.csv", index=False)
    table[table.in_set20].drop(columns="in_set20").to_csv(OUTPUT / "set20.csv", index=False)
    summary = {
        "slides": int(table.slide_id.nunique()),
        "tissue_types": int(table.tissue_type.nunique()),
        "set40_locations": int(len(table)),
        "set20_locations": int(table.in_set20.sum()),
        "paper_blur_probe_locations": int(table.paper_blur_probe.sum()),
        "paper_band_probe_locations": int(table.paper_band_probe.sum()),
        "blur_band_probe_overlap": int((table.paper_blur_probe & table.paper_band_probe).sum()),
        "set20_matches_stored_uni_locations": True,
        "sources": {"set40": str(SET40.relative_to(PROJECT)),
                    "folds": str(FOLDS.relative_to(PROJECT)),
                    "uni_shards": str(UNI_SHARDS.relative_to(PROJECT))},
        "sha256": {name: sha256(OUTPUT / name) for name in ("set40.csv", "set20.csv")},
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
