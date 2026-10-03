#!/usr/bin/env python3
"""Cross-fitted paired ridge and ComBat on frozen pathology encoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from prenorm.feature_correction import (
    apply_affine,
    fit_affine,
    fit_combat,
    macro_retrieval,
    paired_distance,
    unit,
)


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "outputs/scanner_batch_effect_analysis_2026-09-17"
RAW = ROOT / "outputs/feature_crossencoder_review_2026-09-25/raw"
OUTPUT = ROOT / "outputs/feature_crossencoder_review_2026-09-25/results"
SCANNERS = ("at2", "versa", "akoya", "gt450", "s360", "s60")
EXTERNAL_SCANNERS = ("at2", "gt450", "s360", "s60")
METHODS = ("raw", "featmap_ridge", "combat")
ALPHAS = (0.01, 0.1, 1.0)


def load_internal(model: str, scanner: str, cohort: pd.DataFrame):
    scanner_index = SCANNERS.index(scanner)
    slides = cohort.slide_id.astype(str).tolist()
    source40 = []
    target40 = []
    eval_source = []
    eval_target = []
    for slide in cohort.itertuples(index=False):
        path = RAW / "internal" / model / f"fold_{slide.fold}" / f"{slide.slide_id}.h5"
        with h5py.File(path, "r") as store:
            locations = np.asarray(store["location_index"], dtype=np.int64)
            names = tuple(x.decode() for x in store["scanner_names"][:])
            if (len(locations) != 40 or names != SCANNERS or
                str(store.attrs["slide_id"]) != str(slide.slide_id)):
                raise ValueError(f"raw internal feature contract changed: {path}")
            x = np.asarray(store["features"][:, 0, :], dtype=np.float32)
            y = np.asarray(store["features"][:, scanner_index, :], dtype=np.float32)
        with h5py.File(STUDY / "03_uni/shards" / f"{slide.slide_id}.h5", "r") as panel:
            locked = np.asarray(panel["location_index"], dtype=np.int64)
        if len(locked) != 20 or not set(locked).issubset(set(locations)):
            raise ValueError(f"held-out location panel changed: {slide.slide_id}")
        lookup = {int(value): index for index, value in enumerate(locations)}
        eval_indices = [lookup[int(value)] for value in locked]
        source40.append(x)
        target40.append(y)
        eval_source.append(x[eval_indices])
        eval_target.append(y[eval_indices])
    return (np.stack(source40), np.stack(target40),
            np.stack(eval_source), np.stack(eval_target))


def load_external(model: str, scanner: str):
    index = EXTERNAL_SCANNERS.index(scanner)
    paths = sorted((RAW / "external" / model).glob("*.h5"))
    if len(paths) != 13:
        raise ValueError(f"expected 13 external sections for {model}, got {len(paths)}")
    sections = []
    for path in paths:
        with h5py.File(path, "r") as store:
            names = tuple(x.decode() for x in store["scanner_names"][:])
            if names != EXTERNAL_SCANNERS or str(store.attrs["section"]) != path.stem:
                raise ValueError(f"external raw feature contract changed: {path}")
            source = np.asarray(store["features"][:, 0, :], dtype=np.float64)
            target = np.asarray(store["features"][:, index, :], dtype=np.float64)
            n = len(store["location_index"])
        if len(source) != n or len(target) != n:
            raise ValueError(f"external shape mismatch: {path}")
        sections.append((path.stem, source, target))
    if sum(len(source) for _, source, _ in sections) != 2387:
        raise ValueError("PLISM external location count changed")
    return sections


def main(model: str, scanner: str) -> None:
    cohort = pd.read_csv(STUDY / "00_contract/cohort.csv", dtype={"slide_id": str})
    if len(cohort) != 103 or cohort.slide_id.nunique() != 103:
        raise ValueError("manuscript cohort changed")
    if scanner not in SCANNERS[1:]:
        raise ValueError(scanner)
    source40, target40, source, target = load_internal(model, scanner, cohort)
    dim = source.shape[-1]
    folds = cohort.fold.to_numpy(dtype=int)
    tissues = cohort.tissue_type.to_numpy()
    predictions = {"raw": source.astype(np.float64),
                   "featmap_ridge": np.empty_like(source, dtype=np.float64),
                   "combat": np.empty_like(source, dtype=np.float64)}
    external_sections = load_external(model, scanner) if scanner in EXTERNAL_SCANNERS[1:] else []
    external_source = (np.concatenate([item[1] for item in external_sections])
                       if external_sections else np.empty((0, dim), dtype=np.float64))
    external_target = (np.concatenate([item[2] for item in external_sections])
                       if external_sections else np.empty((0, dim), dtype=np.float64))
    external_raw = paired_distance(external_source, external_target) if len(external_source) else np.empty(0)
    section_slices = {}
    offset = 0
    for section, block, _ in external_sections:
        section_slices[section] = slice(offset, offset + len(block))
        offset += len(block)
    alpha_rows = []
    external_rows = []
    for outer_fold in range(5):
        train = folds != outer_fold
        test = folds == outer_fold
        validation = folds == (outer_fold + 1) % 5
        inner_train = train & ~validation
        xin = source40[inner_train].reshape(-1, dim).astype(np.float64)
        yin = target40[inner_train].reshape(-1, dim).astype(np.float64)
        xval = source[validation].reshape(-1, dim).astype(np.float64)
        yval = target[validation].reshape(-1, dim).astype(np.float64)
        losses = [float(paired_distance(
            apply_affine(fit_affine(xin, yin, alpha), xval), yval
        ).mean()) for alpha in ALPHAS]
        alpha = ALPHAS[int(np.argmin(losses))]
        alpha_rows.append({"model": model, "scanner": scanner, "fold": outer_fold,
                           "chosen_relative_alpha": alpha,
                           "inner_losses": json.dumps(losses)})
        xtrain = source40[train].reshape(-1, dim).astype(np.float64)
        ytrain = target40[train].reshape(-1, dim).astype(np.float64)
        xtest = source[test].reshape(-1, dim).astype(np.float64)
        test_shape = source[test].shape
        ridge = fit_affine(xtrain, ytrain, alpha)
        mapped_ridge = apply_affine(ridge, xtest)
        predictions["featmap_ridge"][test] = mapped_ridge.reshape(test_shape)
        both_test = np.concatenate([xtest, external_source], axis=0)
        mapped_combat = fit_combat(xtrain, ytrain, both_test)
        predictions["combat"][test] = mapped_combat[:len(xtest)].reshape(test_shape)
        if external_sections:
            mapped_external = {
                "raw": external_source,
                "featmap_ridge": apply_affine(ridge, external_source),
                "combat": mapped_combat[len(xtest):],
            }
            for section, _, _ in external_sections:
                sl = section_slices[section]
                for method, vectors in mapped_external.items():
                    d = paired_distance(vectors[sl], external_target[sl])
                    external_rows.append({
                        "model": model, "scanner": scanner, "section": section,
                        "fold": outer_fold, "method": method,
                        "locations": sl.stop - sl.start,
                        "raw_distance": float(external_raw[sl].mean()),
                        "corrected_distance": float(d.mean()),
                        "target_gain": float((external_raw[sl] - d).mean()),
                    })
        print(f"{model} {scanner}: fold {outer_fold} complete", flush=True)
    raw_distance = paired_distance(source, target).mean(axis=1)
    target_slide = unit(target.mean(axis=1))
    rows = []
    content_rows = []
    for method, mapped in predictions.items():
        distances = paired_distance(mapped, target).mean(axis=1)
        mapped_slide = unit(mapped.mean(axis=1))
        content_rows.append({
            "model": model, "scanner": scanner, "method": method,
            "macro_tissue_retrieval": macro_retrieval(mapped_slide, target_slide, tissues),
            "target_self_retrieval": macro_retrieval(target_slide, target_slide, tissues),
            "variance_trace_ratio": float(np.var(mapped_slide, axis=0).sum() /
                                          np.var(target_slide, axis=0).sum()),
        })
        for i, slide in enumerate(cohort.itertuples(index=False)):
            rows.append({"model": model, "scanner": scanner, "method": method,
                         "slide_id": str(slide.slide_id), "tissue_type": slide.tissue_type,
                         "fold": int(slide.fold),
                         "raw_distance": float(raw_distance[i]),
                         "corrected_distance": float(distances[i]),
                         "target_gain": float(raw_distance[i] - distances[i])})
    out = OUTPUT / model / scanner
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "per_slide.csv", index=False)
    pd.DataFrame(content_rows).to_csv(out / "content.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(out / "ridge_selection.csv", index=False)
    if external_rows:
        pd.DataFrame(external_rows).to_csv(out / "external_per_section_fold.csv", index=False)
    print(json.dumps({"model": model, "scanner": scanner,
                      "internal_slides": 103, "external_sections": len(external_sections),
                      "dimension": dim, "methods": METHODS}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=("uni2", "virchow2", "hoptimus1"))
    parser.add_argument("--scanner", required=True, choices=SCANNERS[1:])
    args = parser.parse_args()
    main(args.model, args.scanner)
