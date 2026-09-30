#!/usr/bin/env python3
"""SMOKE TESTS ONLY: present older embedding stores in the RV03 in-memory structure.

Used by RV04, RV05 and RV09 only when ``--source smoke`` is passed, before RV03 exists.
Production runs never import this module.

* PanNormal / ``uni_v1``: ``03_uni/shards/<slide>.h5`` (features [20, 41, 1024]); the
  conditions ``source``, ``target:<s>``, ``reinhard:<s>``, ``frequency:<s>`` and
  ``combined:<s>`` are renamed to ``source_at2``, ``target_<s>``, ``<method>_to_<s>``.
  The supplementary oracle arms of that store are not exposed.
* PLISM / ``uni2``: the raw external features of the cross-PFM review
  (``outputs/feature_crossencoder_review_2026-09-25/raw/external/uni2``) give
  ``source_at2`` and ``target_<s>``; core and tissue come from the frozen PLISM benchmark
  location table. A test fixture ``identity_to_<s>_fold<k>`` (k = 0..4, a copy of the
  source) exercises the fold-variant averaging: every metric of ``identity`` must equal
  the raw value exactly.

``write_fixture`` writes a few shards in the RV03 file format so that the production
reader can be round-trip tested against this adapter.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from corrected_embedding_reader import (
    BENCHMARK,
    PROJECT,
    TARGETS,
    EmbeddingSet,
    decode,
    load_cohort,
    load_set20,
    plism_section_folds,
)


UNI_SHARDS = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17/03_uni/shards"
PLISM_RAW = PROJECT / "outputs/feature_crossencoder_review_2026-09-25/raw/external"
PANNORMAL_METHODS = ("reinhard", "frequency", "combined")
SMOKE_PFM = {"pannormal": "uni_v1", "plism": "uni2"}


def smoke_conditions(dataset: str, pfm: str) -> list[str]:
    scanners = TARGETS[dataset]
    names = ["source_at2"] + [f"target_{scanner}" for scanner in scanners]
    if dataset == "pannormal":
        names += [f"{method}_to_{scanner}" for method in PANNORMAL_METHODS for scanner in scanners]
    else:
        names += [f"identity_to_{scanner}_fold{fold}" for scanner in scanners for fold in range(5)]
    return names


def load_smoke(dataset: str, pfm: str, methods=None) -> EmbeddingSet:
    if pfm != SMOKE_PFM[dataset]:
        raise ValueError(f"smoke adapter provides {dataset}/{SMOKE_PFM[dataset]} only")
    if dataset == "pannormal":
        return _pannormal(methods)
    return _plism(methods)


def _pannormal(methods) -> EmbeddingSet:
    cohort = load_cohort()
    set20 = load_set20()
    scanners = TARGETS["pannormal"]
    wanted = {"source_at2": "source"}
    wanted.update({f"target_{scanner}": f"target:{scanner}" for scanner in scanners})
    for method in PANNORMAL_METHODS:
        if methods is None or method in methods:
            wanted.update({f"{method}_to_{scanner}": f"{method}:{scanner}" for scanner in scanners})
    blocks = {name: [] for name in wanted}
    rows = []
    for row in cohort.itertuples():
        path = UNI_SHARDS / f"{row.slide_id}.h5"
        with h5py.File(path, "r") as store:
            names = decode(store["condition_names"][:])
            features = np.asarray(store["features"][:], dtype=np.float32)
            locations = np.asarray(store["location_index"][:], dtype=np.int64)
        expected = set20.loc[set20.slide_id == row.slide_id].sort_values("location_index")
        if locations.tolist() != expected.location_index.tolist():
            raise ValueError(f"{path}: locations differ from set20")
        lookup = {name: index for index, name in enumerate(names)}
        for new, old in wanted.items():
            blocks[new].append(features[:, lookup[old], :])
        for record in expected.itertuples():
            rows.append({"unit": row.slide_id, "tissue": str(row.tissue_type), "fold": int(row.fold),
                         "location_id": int(record.location_index), "core": "",
                         "source_index": int(record.source_index)})
    conditions = {name: np.concatenate(block) for name, block in blocks.items()}
    return EmbeddingSet("pannormal", "uni_v1", "smoke_adapter", pd.DataFrame(rows), conditions,
                        tuple(smoke_conditions("pannormal", "uni_v1")),
                        ["SMOKE: 03_uni shards via smoke_embedding_adapter"])


def plism_keys() -> pd.DataFrame:
    frames =[pd.read_csv(path, usecols=["location", "section", "core", "tissue_type", "method"])
              for path in sorted((BENCHMARK / "plism_conventional/shards").glob("S360_*.csv.gz"))]
    keys = pd.concat(frames, ignore_index=True)
    keys = keys.loc[keys.method.eq("raw")].drop(columns="method").drop_duplicates()
    keys["section"] = keys["section"].astype(str)
    return keys


def _plism(methods) -> EmbeddingSet:
    keys = plism_keys().set_index(["section", "location"])
    scanners = TARGETS["plism"]
    paths = sorted((PLISM_RAW / "uni2").glob("*.h5"))
    folds = plism_section_folds([path.stem for path in paths])
    source, targets, rows = [], {scanner: [] for scanner in scanners}, []
    for path in paths:
        with h5py.File(path, "r") as store:
            names = decode(store["scanner_names"][:])
            features = np.asarray(store["features"][:], dtype=np.float32)
            locations = np.asarray(store["location_index"][:], dtype=np.int64)
        meta = keys.loc[[(path.stem, int(location)) for location in locations]]
        order = np.lexsort((locations, meta["core"].astype(str).to_numpy()))
        source.append(features[order, names.index("at2")])
        for scanner in scanners:
            targets[scanner].append(features[order, names.index(scanner)])
        for offset in order:
            record = meta.iloc[offset]
            rows.append({"unit": path.stem, "tissue": str(record.tissue_type), "fold": folds[path.stem],
                         "location_id": int(locations[offset]), "core": str(int(record.core)),
                         "source_index": -1})
    conditions = {"source_at2": np.concatenate(source)}
    for scanner in scanners:
        conditions[f"target_{scanner}"] = np.concatenate(targets[scanner])
        if methods is None or "identity" in methods:
            for fold in range(5):
                conditions[f"identity_to_{scanner}_fold{fold}"] = conditions["source_at2"]
    return EmbeddingSet("plism", "uni2", "smoke_adapter", pd.DataFrame(rows), conditions,
                        tuple(smoke_conditions("plism", "uni2")),
                        ["SMOKE: cross-PFM raw external UNI2-h features + identity fixture"])


def write_fixture(destination: Path, slides: int = 3, sections: int = 2) -> dict:
    """Write a few adapter shards in the RV03 file format (reader round-trip test)."""
    written = {"pannormal": [], "plism": []}
    pan = _pannormal(None)
    names = list(pan.conditions)
    for slide in pan.units[:slides]:
        mask = (pan.locations.unit == slide).to_numpy()
        features = np.stack([pan.conditions[name][mask] for name in names], axis=1)
        meta = pan.locations.loc[mask]
        path = destination / "pannormal/uni_v1" / f"{slide}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Reverse the location order to check that the reader sorts by location index.
        with h5py.File(path, "w") as store:
            store.attrs.update({"pfm": "uni_v1", "slide_id": slide, "tissue_type": meta.tissue.iloc[0],
                                "fold": int(meta.fold.iloc[0])})
            store.create_dataset("features", data=features[::-1])
            store.create_dataset("condition_names", data=np.asarray(names, dtype="S64"))
            store.create_dataset("location_index", data=meta.location_id.to_numpy()[::-1])
            store.create_dataset("source_index", data=meta.source_index.to_numpy()[::-1])
        written["pannormal"].append(str(path))
    plism = _plism(None)
    names = list(plism.conditions)
    for section in plism.units[:sections]:
        mask = (plism.locations.unit == section).to_numpy()
        features = np.stack([plism.conditions[name][mask] for name in names], axis=1)
        meta = plism.locations.loc[mask]
        path = destination / "plism/uni2" / f"{section}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as store:
            store.attrs.update({"pfm": "uni2", "section": section})
            store.create_dataset("features", data=features[::-1])
            store.create_dataset("condition_names", data=np.asarray(names, dtype="S64"))
            store.create_dataset("core", data=meta.core.astype(int).to_numpy()[::-1])
            store.create_dataset("location", data=meta.location_id.to_numpy()[::-1])
            store.create_dataset("tissue_type", data=np.asarray(meta.tissue.tolist()[::-1], dtype="S64"))
        written["plism"].append(str(path))
    return written
