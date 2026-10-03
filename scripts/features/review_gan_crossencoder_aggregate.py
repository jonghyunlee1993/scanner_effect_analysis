#!/usr/bin/env python3
"""Aggregate paired GAN target gains and tissue retrieval for frozen encoders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/gan_encoder_review_2026-09-25"
MODELS = ("uni_v1", "uni2", "virchow2", "hoptimus1")
SCANNERS = ("versa", "akoya", "gt450", "s360", "s60")
ARMS = ("source", "target", "pix2pix", "cyclegan")
N_SLIDES = 103


def unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value / max(float(np.linalg.norm(value)), 1e-12)


def bootstrap(values: np.ndarray, seed: int = 20260925) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), (4000, len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_locations() -> pd.DataFrame:
    frames = []
    for model in MODELS:
        for fold in range(5):
            path = OUTPUT / "shards" / model / f"fold_{fold}.csv.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
            frames.append(pd.read_csv(path, dtype={"slide_id": str}))
    frame = pd.concat(frames, ignore_index=True)
    for model, group in frame.groupby("model"):
        if (len(group) != N_SLIDES * 5 * 20 or
            group.slide_id.nunique() != N_SLIDES or
            group.groupby(["slide_id", "scanner"]).location_index.nunique().ne(20).any()):
            raise ValueError(f"{model}: incomplete locked 20-location panel")
    if frame.duplicated(["model", "slide_id", "scanner", "location_index"]).any():
        raise ValueError("duplicate model/scanner/location row")
    return frame


def summarize_gain(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    slide_rows = []
    for model, subset in frame.groupby("model"):
        for scanner in ("all", *SCANNERS):
            selected = subset if scanner == "all" else subset[subset.scanner == scanner]
            slide_scanner = selected.groupby(["slide_id", "scanner"])[
                ["raw_distance", "pix2pix_distance", "cyclegan_distance",
                 "pix2pix_gain", "cyclegan_gain"]
            ].mean().reset_index()
            slide_scanner["pix2pix_fraction"] = (
                slide_scanner.pix2pix_gain / slide_scanner.raw_distance
            )
            slide_scanner["cyclegan_fraction"] = (
                slide_scanner.cyclegan_gain / slide_scanner.raw_distance
            )
            slide_scanner["pix_minus_cycle_gain"] = (
                slide_scanner.pix2pix_gain - slide_scanner.cyclegan_gain
            )
            by_slide = slide_scanner.groupby("slide_id")[
                ["raw_distance", "pix2pix_distance", "cyclegan_distance",
                 "pix2pix_gain", "cyclegan_gain", "pix2pix_fraction",
                 "cyclegan_fraction", "pix_minus_cycle_gain"]
            ].mean()
            if len(by_slide) != N_SLIDES:
                raise ValueError(f"{model} {scanner}: missing slides")
            for metric in by_slide.columns:
                mean, low, high = bootstrap(by_slide[metric].to_numpy())
                rows.append({"model": model, "scanner": scanner, "metric": metric,
                             "mean": mean, "ci_low": low, "ci_high": high,
                             "slides": len(by_slide)})
            for slide_id, values in by_slide.iterrows():
                slide_rows.append({"model": model, "scanner": scanner,
                                   "slide_id": slide_id, **values.to_dict()})
    return pd.DataFrame(rows), pd.DataFrame(slide_rows)


def tissue_retrieval() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        paths = sorted((OUTPUT / "slide_means" / model).glob("fold_*/*.npz"))
        if len(paths) != N_SLIDES:
            raise ValueError(f"{model}: expected {N_SLIDES} slide means, got {len(paths)}")
        slides = {}
        for path in paths:
            with np.load(path, allow_pickle=False) as item:
                slide_id = str(item["slide_id"].item())
                slides[slide_id] = {
                    "tissue_type": str(item["tissue_type"].item()),
                    **{arm: np.asarray(item[arm]) for arm in ARMS},
                }
        ids = sorted(slides)
        labels = np.asarray([slides[slide]["tissue_type"] for slide in ids])
        counts = pd.Series(labels).value_counts()
        references = np.stack([unit(slides[slide]["source"].mean(axis=0))
                               for slide in ids])
        for arm in ARMS:
            per_slide = []
            for i, slide in enumerate(ids):
                if counts[labels[i]] < 2:
                    continue
                queries = slides[slide][arm]
                similarity = queries @ references.T
                similarity[:, i] = -np.inf
                correct = labels[similarity.argmax(axis=1)] == labels[i]
                per_slide.append({"tissue_type": labels[i],
                                  "recall": float(correct.mean())})
            result = pd.DataFrame(per_slide)
            rows.append({"model": model, "arm": arm,
                         "macro_recall": float(result.groupby("tissue_type").recall.mean().mean()),
                         "slide_mean_recall": float(result.recall.mean()),
                         "tissue_types": int(result.tissue_type.nunique()),
                         "slides": len(result)})
    return pd.DataFrame(rows)


def main() -> None:
    audit = json.loads((OUTPUT / "audit.json").read_text())
    if audit.get("status") != "pass":
        raise ValueError("prediction audit did not pass")
    frame = load_locations()
    summary, slides = summarize_gain(frame)
    retrieval = tissue_retrieval()
    summary.to_csv(OUTPUT / "gan_target_gain_summary.csv", index=False)
    slides.to_csv(OUTPUT / "gan_per_slide.csv", index=False)
    retrieval.to_csv(OUTPUT / "gan_tissue_retrieval.csv", index=False)
    print(json.dumps({"status": "complete", "models": list(MODELS),
                      "slides": N_SLIDES, "locations": len(frame),
                      "summary": str(OUTPUT / "gan_target_gain_summary.csv")}))


if __name__ == "__main__":
    main()
