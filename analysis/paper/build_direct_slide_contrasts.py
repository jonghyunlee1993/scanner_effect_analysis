#!/usr/bin/env python3
"""Build one paired contrast per physical slide, target scanner, and endpoint.

Scalar measurements are averaged over every paired location in a slide before
the target/reference contrast is formed. Frequency endpoints use the existing
whole-slide spectra, which were computed from every paired location.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.final_image_study import MODEL_ENDPOINTS, REFERENCE, SCANNERS  # noqa: E402

FREQUENCY_LABELS = {
    "low_mid": "Low–mid frequency transfer",
    "mid": "Mid-frequency transfer",
    "high": "High-frequency transfer",
}


def contrast(target: float, reference: float, transform: str) -> float:
    if transform == "difference":
        return target - reference
    if transform == "log2_ratio":
        return float(np.log2(max(target, 1e-12) / max(reference, 1e-12)))
    if transform == "identity":
        return target
    if transform == "one_minus":
        return 1.0 - target
    raise ValueError(f"unknown endpoint transform: {transform}")


def build(cohort_path: Path, metrics_root: Path, bands_path: Path) -> pd.DataFrame:
    cohort = pd.read_csv(cohort_path, dtype={"slide_id": str})
    bands = pd.read_csv(bands_path, dtype={"slide_id": str})
    if len(cohort) != 103 or cohort["slide_id"].duplicated().any():
        raise ValueError("expected 103 unique physical slides")
    band_key = ["slide_id", "scanner", "band"]
    if bands.duplicated(band_key).any():
        raise ValueError("duplicate whole-slide frequency band")
    bands = bands.set_index(band_key)
    rows = []
    for slide in cohort.itertuples(index=False):
        slide_id = str(slide.slide_id)
        tissue = str(slide.tissue_type)
        with h5py.File(metrics_root / slide_id / "location_metrics.h5", "r") as store:
            scanners = [item.decode() for item in store["scanner_names"][:]]
            metric_names = [item.decode() for item in store["metric_names"][:]]
            if scanners != list(SCANNERS):
                raise ValueError(f"{slide_id}: unexpected scanner order")
            values = store["metrics"][:]
        if values.ndim != 3 or values.shape[1:] != (len(SCANNERS), len(metric_names)):
            raise ValueError(f"{slide_id}: unexpected metric shape")
        if values.shape[0] != int(slide.pair_count) or not np.isfinite(values).all():
            raise ValueError(f"{slide_id}: incomplete paired locations")
        patch_count = int(values.shape[0])
        means = values.mean(axis=0, dtype=np.float64)
        metric_index = {name: index for index, name in enumerate(metric_names)}
        for scanner_index, scanner in enumerate(scanners):
            if scanner == REFERENCE:
                continue
            base = {
                "slide_id": slide_id,
                "tissue_type": tissue,
                "scanner": scanner,
                "patch_count": patch_count,
            }
            for endpoint, specification in MODEL_ENDPOINTS.items():
                index = metric_index[specification["source"]]
                rows.append({
                    **base,
                    "endpoint": endpoint,
                    "endpoint_label": specification["label"],
                    "family": specification["family"],
                    "value": contrast(
                        float(means[scanner_index, index]),
                        float(means[0, index]),
                        specification["transform"],
                    ),
                })
            for band, label in FREQUENCY_LABELS.items():
                rows.append({
                    **base,
                    "endpoint": f"frequency_{band}",
                    "endpoint_label": label,
                    "family": "spatial frequency",
                    "value": float(bands.loc[(slide_id, scanner, band), "log2_relative_transfer"]),
                })
    result = pd.DataFrame(rows)
    if len(result) != 103 * 5 * 13 or not np.isfinite(result["value"]).all():
        raise ValueError("incomplete direct slide-level contrast table")
    observed_locations = result.drop_duplicates("slide_id")["patch_count"].sum()
    if observed_locations != cohort["pair_count"].sum():
        raise ValueError("paired location count does not match the frozen cohort")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--bands", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = build(args.cohort, args.metrics_root, args.bands)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote {len(frame)} contrasts from {frame['slide_id'].nunique()} slides")
