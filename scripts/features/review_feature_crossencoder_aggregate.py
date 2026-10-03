#!/usr/bin/env python3
"""Summarize cross-fitted feature correction across frozen encoders."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/feature_crossencoder_review_2026-09-25"
UNI = Path(os.environ.get(
    "PAPER_UNI_FEATURE_ROOT", str(ROOT / "outputs/discussion_followup_2026-09-25")
))
SUMMARY_OUTPUT = Path(os.environ.get("PAPER_FEATURE_SUMMARY_ROOT", str(OUTPUT / "summary")))
GAN = ROOT / "outputs/gan_encoder_review_2026-09-25/gan_per_slide.csv"
MODELS = ("uni_v1", "uni2", "virchow2", "hoptimus1")
SCANNERS = ("versa", "akoya", "gt450", "s360", "s60")
METHODS = ("raw", "featmap_ridge", "combat")


def bootstrap(values: np.ndarray, seed: int = 20260925) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(4000, len(values)))
    means = values[draws].mean(axis=1)
    return (float(values.mean()), float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)))


def load_internal() -> pd.DataFrame:
    frames = []
    original = pd.read_csv(UNI / "feature40_forward/per_slide.csv", dtype={"slide_id": str})
    frames.append(original[original.method.isin(METHODS)].assign(model="uni_v1"))
    for model in MODELS[1:]:
        for scanner in SCANNERS:
            path = OUTPUT / "results" / model / scanner / "per_slide.csv"
            frames.append(pd.read_csv(path, dtype={"slide_id": str}))
    frame = pd.concat(frames, ignore_index=True)
    counts = frame.groupby(["model", "scanner", "method"]).slide_id.nunique()
    if len(counts) != len(MODELS) * len(SCANNERS) * len(METHODS) or not counts.eq(103).all():
        raise ValueError("incomplete internal feature results")
    if frame.duplicated(["model", "scanner", "method", "slide_id"]).any():
        raise ValueError("duplicate internal feature results")
    return frame


def summarize_internal(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for scanner in ("all", *SCANNERS):
            part = frame[frame.model.eq(model)]
            if scanner != "all":
                part = part[part.scanner.eq(scanner)]
            for method in METHODS:
                subset = part[part.method.eq(method)]
                slide = subset.groupby("slide_id")[["raw_distance", "corrected_distance",
                                                    "target_gain"]].mean()
                for metric in slide.columns:
                    mean, low, high = bootstrap(slide[metric].to_numpy())
                    rows.append({"model": model, "scanner": scanner, "method": method,
                                 "metric": metric, "mean": mean, "ci_low": low,
                                 "ci_high": high, "slides": len(slide)})
    return pd.DataFrame(rows)


def load_external() -> pd.DataFrame:
    frames = []
    original = pd.read_csv(UNI / "feature40_external_forward/per_section_fold.csv")
    frames.append(original[original.method.isin(METHODS)].assign(model="uni_v1"))
    for model in MODELS[1:]:
        for scanner in ("gt450", "s360", "s60"):
            path = OUTPUT / "results" / model / scanner / "external_per_section_fold.csv"
            frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    counts = frame.groupby(["model", "scanner", "method", "section"]).fold.nunique()
    if len(counts) != len(MODELS) * 3 * len(METHODS) * 13 or not counts.eq(5).all():
        raise ValueError("incomplete external feature results")
    return frame


def summarize_external(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for scanner in ("all", "gt450", "s360", "s60"):
            part = frame[frame.model.eq(model)]
            if scanner != "all":
                part = part[part.scanner.eq(scanner)]
            for method in METHODS:
                subset = part[part.method.eq(method)]
                section = subset.groupby("section")[["raw_distance", "corrected_distance",
                                                   "target_gain"]].mean()
                for metric in section.columns:
                    mean, low, high = bootstrap(section[metric].to_numpy())
                    rows.append({"model": model, "scanner": scanner, "method": method,
                                 "metric": metric, "mean": mean, "ci_low": low,
                                 "ci_high": high, "sections": len(section)})
    return pd.DataFrame(rows)


def compare_image(frame: pd.DataFrame) -> pd.DataFrame:
    feature = frame[frame.method.ne("raw")][["model", "scanner", "slide_id", "method",
                                             "target_gain"]]
    gan = pd.read_csv(GAN, dtype={"slide_id": str})
    gan = gan[gan.scanner.isin(SCANNERS)]
    joined = feature.merge(gan[["model", "scanner", "slide_id", "pix2pix_gain",
                                "cyclegan_gain"]], on=["model", "scanner", "slide_id"],
                           validate="many_to_one")
    if len(joined) != len(MODELS) * len(SCANNERS) * 103 * 2:
        raise ValueError("feature/image comparison has missing pairs")
    rows = []
    for (model, method), subset in joined.groupby(["model", "method"]):
        for scanner in ("all", *SCANNERS):
            part = subset if scanner == "all" else subset[subset.scanner.eq(scanner)]
            slide = part.groupby("slide_id").mean(numeric_only=True)
            for comparator in ("pix2pix", "cyclegan"):
                delta = slide.target_gain - slide[f"{comparator}_gain"]
                mean, low, high = bootstrap(delta.to_numpy())
                rows.append({"model": model, "scanner": scanner, "feature_method": method,
                             "image_method": comparator, "gain_difference": mean,
                             "ci_low": low, "ci_high": high, "slides": len(slide)})
    return pd.DataFrame(rows)


def summarize_content() -> pd.DataFrame:
    frames = []
    original = pd.read_csv(UNI / "feature40_forward/content_summary.csv")
    frames.append(original[original.method.isin(METHODS)].assign(model="uni_v1"))
    for model in MODELS[1:]:
        for scanner in SCANNERS:
            frames.append(pd.read_csv(OUTPUT / "results" / model / scanner / "content.csv"))
    frame = pd.concat(frames, ignore_index=True)
    summary = frame.groupby(["model", "method"])[["macro_tissue_retrieval",
                                                    "target_self_retrieval",
                                                    "variance_trace_ratio"]].mean().reset_index()
    frame.to_csv(SUMMARY_OUTPUT / "content_by_scanner.csv", index=False)
    return summary


def main() -> None:
    out = SUMMARY_OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    internal = load_internal()
    external = load_external()
    summarize_internal(internal).to_csv(out / "feature_internal_summary.csv", index=False)
    summarize_external(external).to_csv(out / "feature_external_summary.csv", index=False)
    compare_image(internal).to_csv(out / "feature_minus_image_gain.csv", index=False)
    summarize_content().to_csv(out / "feature_content_summary.csv", index=False)
    print(f"complete: {len(internal)} internal and {len(external)} external rows")


if __name__ == "__main__":
    main()
