#!/usr/bin/env python3
"""Audit paired PanNormal GAN predictions before frozen-encoder reuse."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "outputs/scanner_batch_effect_analysis_2026-09-17"
BASE = STUDY / "12_manuscript_completion/02_baseline_benchmark"
OUTPUT = ROOT / "outputs/gan_encoder_review_2026-09-25"
SCANNERS = ("versa", "akoya", "gt450", "s360", "s60")


def prediction_path(method: str, scanner: str, fold: int, slide_id: str) -> Path:
    if scanner == "gt450":
        root = STUDY / "06_learned_baselines" / (
            "09_bidirectional_full_training" if method == "pix2pix"
            else "11_cyclegan_full_training"
        ) / "02_predictions"
    else:
        root = BASE / ("03_pix2pix" if method == "pix2pix" else "04_cyclegan") / "02_predictions"
    matches = list((root / f"at2_to_{scanner}" / f"fold_{fold}" / "slides").glob(
        f"{slide_id}__*.h5"
    ))
    if len(matches) != 1:
        raise FileNotFoundError(f"{method} {scanner} {slide_id} fold {fold}: {matches}")
    return matches[0]


def main() -> None:
    cohort = pd.read_csv(STUDY / "00_contract/cohort.csv", dtype={"slide_id": str})
    if len(cohort) != 103 or cohort.slide_id.nunique() != 103:
        raise ValueError("manuscript cohort changed")
    rows = []
    for slide in cohort.itertuples(index=False):
        with h5py.File(STUDY / "03_uni/shards" / f"{slide.slide_id}.h5", "r") as panel:
            locked = np.asarray(panel["location_index"], dtype=np.int64)
        if len(locked) != 20 or len(set(locked)) != 20:
            raise ValueError(f"{slide.slide_id}: locked UNI locations changed")
        for scanner in SCANNERS:
            pix = prediction_path("pix2pix", scanner, slide.fold, str(slide.slide_id))
            cyc = prediction_path("cyclegan", scanner, slide.fold, str(slide.slide_id))
            with h5py.File(pix, "r") as a, h5py.File(cyc, "r") as b:
                locations_a = np.asarray(a["metadata/location_index"], dtype=np.int64)
                locations_b = np.asarray(b["metadata/location_index"], dtype=np.int64)
                if (len(locations_a) != 40 or len(locations_b) != 40 or
                    not np.array_equal(locations_a, locations_b) or
                    not set(locked).issubset(set(locations_a))):
                    raise ValueError(f"sample-index mismatch: {slide.slide_id} {scanner}")
                lookup = {int(value): index for index, value in enumerate(locations_a)}
                for location in locked[[0, len(locked) // 2, -1]]:
                    index = lookup[int(location)]
                    for arm in ("raw_source", "real_target"):
                        if not np.array_equal(a[f"images/{arm}"][index],
                                              b[f"images/{arm}"][index]):
                            raise ValueError(
                                f"{arm} differs between GANs: {slide.slide_id} {scanner} {location}"
                            )
                rows.append({"slide_id": str(slide.slide_id), "fold": int(slide.fold),
                             "tissue_type": slide.tissue_type, "scanner": scanner,
                             "pix2pix_path": str(pix), "cyclegan_path": str(cyc),
                             "locked_locations": len(locked)})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "prediction_pairs.csv", index=False)
    (OUTPUT / "audit.json").write_text(json.dumps({
        "status": "pass", "slides": 103, "scanner_directions": len(SCANNERS),
        "direction_slide_pairs": len(rows), "locked_locations_per_pair": 20,
        "source_target_sentinel_equality": "first, middle, last locked location in every pair",
    }, indent=2) + "\n")
    print(OUTPUT / "audit.json", flush=True)


if __name__ == "__main__":
    main()
