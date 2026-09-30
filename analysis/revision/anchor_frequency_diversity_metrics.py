#!/usr/bin/env python3
"""RV13 stage 2 (CPU): scale-free scanner-sensitivity metrics for 13 embedding streams.

Models: the four original PFMs (UNI v1, UNI2-h, Virchow2, H-optimus-1; RV03 embeddings via
``corrected_embedding_reader`` and RV02 band rows) and the nine RV13 streams
(``anchor_frequency_diversity_embed.py``). PanNormal `set20`, 103 slides x 20 locations.

Metrics (README, RV13):
1. normalized scanner distance: raw AT2-target cosine distance (location -> slide -> five
   scanners equally weighted) / mean between-tissue cosine distance among raw AT2 embeddings
   (all location pairs of different tissue types; recomputed in every resample, as RV04);
2. detectability of raw AT2 vs raw target, pooled over scanners, locked slide-fold 5-fold CV,
   L2-normalized inputs, the three RV04b classifiers (linear, MLP, k-NN); accuracy per location
   -> slide -> mean (RV04); best probe = the maximum of the three means within each resample;
   six-way scanner probe = RV01's (``blur_sharpen_set20.scanner_probe``: StandardScaler +
   class-balanced multinomial logistic regression, C in {0.01, 0.1, 1} by inner locked folds;
   balanced accuracy);
3. colour share (raw - Reinhard) / raw, frequency-alone share (raw - frequency) / raw,
   frequency share (Reinhard - colour + frequency) / raw, on pooled target distances;
4. band sensitivity: representation shift (cosine distance manipulated vs colour-matched; both
   signs) per band and dose / between-tissue distance; high / low-mid ratio; target-direction
   high-band target gain / between-tissue distance (RV02 definitions and aggregation);
5. tissue retrieval: top-1 same-tissue macro recall among raw AT2 slide means (other slides as
   bank; 36 tissues / 102 slides; ``prenorm.feature_correction.macro_retrieval``);
6. PathoROB robustness index (amendment): all six raw scanners pooled (12,360 vectors),
   L2-normalized, Euclidean k-NN with a margin of 120 (the largest slide), neighbours from the
   anchor's own slide removed; SO = same tissue / other scanner, OS = other tissue / same
   scanner; RI_k = cumulative SO / cumulative (SO + OS) over ranks 1..k and anchors (the 36
   tissue types with >= 2 slides); primary k maximizes k-NN tissue balanced accuracy over
   {5, 10, 15, 20, 25, 30, 40, 50} (majority vote, ties to the lowest label as
   ``scipy.stats.mode``); curve k = 1..50; exact chance level from the label counts.
CIs: 2,000 slide-bootstrap resamples from ``corrected_embedding_reader.bootstrap_weights``
(shared by every metric and model, so contrasts are paired); percentile intervals.

Commands: ``list-tasks``; ``task --task-index i`` (model x {detect, probe6, ri}); ``aggregate``
(everything else, QC, H1-H4 read-outs, ``summary.md``).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import glob
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
REVISION = Path(__file__).resolve().parent
for _path in (PROJECT, PROJECT / "src", PROJECT / "scripts", REVISION):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from corrected_embedding_reader import (  # noqa: E402
    bootstrap_weights, load, load_cohort, load_set20, summarize, unit, weighted_macro, weighted_mean,
    write_frame, write_json, write_text,
)

OUTPUT = REVISION / "results/anchor_frequency_diversity"
EMBEDDINGS = OUTPUT / "embeddings"
TASKS = OUTPUT / "tasks"
RV02 = REVISION / "results/band_manipulation_set20"
RV04_CELLS = REVISION / "results/three_axis_evaluation/cell_statistics.csv"
RV01_PROBE = REVISION / "results/blur_sharpen_set20/scanner_probe_summary.csv"

ORIGINAL = ("uni_v1", "uni2", "virchow2", "hoptimus1")
NEW = ("exaonepath", "exaonepath_raw", "seal_uni2", "seal_conch_pre", "seal_conch_projected",
       "conch_pre", "conch_projected", "plip", "dinov2")
MODELS = ORIGINAL + NEW
LABELS = {"uni_v1": "UNI v1", "uni2": "UNI2-h", "virchow2": "Virchow2", "hoptimus1": "H-optimus-1",
          "exaonepath": "EXAONEPath (as released)", "exaonepath_raw": "EXAONEPath (raw input, off-label)",
          "seal_uni2": "SEAL-UNI2", "seal_conch_pre": "SEAL-CONCH (pre-projection)",
          "seal_conch_projected": "SEAL-CONCH (projected)", "conch_pre": "CONCH (pre-projection)",
          "conch_projected": "CONCH (projected)", "plip": "PLIP", "dinov2": "DINOv2 ViT-L/14"}
CELL = {"uni_v1": "image-only / narrow", "uni2": "image-only / narrow", "virchow2": "image-only / narrow",
        "hoptimus1": "image-only / narrow", "exaonepath": "image-only / narrow (+ stain norm.)",
        "exaonepath_raw": "off-label", "seal_uni2": "anchored / narrow", "seal_conch_pre": "anchored / narrow",
        "seal_conch_projected": "anchored / narrow (secondary)", "conch_pre": "anchored / heterogeneous",
        "conch_projected": "anchored / heterogeneous (secondary)", "plip": "anchored / heterogeneous",
        "dinov2": "image-only / heterogeneous (reference)"}
IMAGE_ONLY_WSI = ("uni_v1", "uni2", "virchow2", "hoptimus1", "exaonepath")
HETEROGENEOUS = ("conch_pre", "plip", "dinov2")
NARROW = ("uni_v1", "uni2", "virchow2", "hoptimus1", "exaonepath", "seal_uni2", "seal_conch_pre")
ANCHORED = ("seal_uni2", "seal_conch_pre", "conch_pre", "plip")
IMAGE_ONLY = ("uni_v1", "uni2", "virchow2", "hoptimus1", "exaonepath", "dinov2")
OTHER_WSI = ("uni_v1", "uni2", "virchow2", "hoptimus1")
TARGETS = ("versa", "akoya", "gt450", "s360", "s60")
SCANNERS = ("at2",) + TARGETS
BANDS = ("low_mid", "mid", "high")
DOSES = (0.25, 0.5)
CLASSIFIERS = ("linear", "mlp", "knn")
PARTS = ("detect", "probe6", "ri")
K_GRID = (5, 10, 15, 20, 25, 30, 40, 50)
K_MAX = 50
PRIMARY_DOSE = 0.25
PARITY_MIN = 0.999


# ----------------------------------------------------------------------------- loading

class Model:
    """Unit-normalized embeddings of one stream, rows = set20 locations sorted by slide, location."""

    def __init__(self, name: str, conditions: dict[str, np.ndarray], locations: pd.DataFrame):
        self.name = name
        self.conditions = conditions
        self.locations = locations.reset_index(drop=True)
        self.slides = tuple(sorted(self.locations.slide_id.unique()))
        code = {s: i for i, s in enumerate(self.slides)}
        self.slide_code = self.locations.slide_id.map(code).to_numpy()
        self.n = len(self.locations)

    def slide_means(self, values: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
        """Mean per slide; locations with ``keep == False`` are left out (NaN if a slide has none)."""
        keep = np.ones(self.n, dtype=bool) if keep is None else keep
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.bincount(self.slide_code[keep], weights=values[keep], minlength=len(self.slides)) / np.bincount(
                self.slide_code[keep], minlength=len(self.slides))


def load_model(name: str, root: Path = EMBEDDINGS, band: bool = False) -> Model:
    cohort = load_cohort()
    if name in ORIGINAL:
        emb = load("pannormal", name, methods=["reinhard", "frequency", "combined"])
        locations = emb.locations.rename(columns={"unit": "slide_id", "location_id": "location_index", "tissue": "tissue_type"})
        conditions = {key: unit(value).astype(np.float32) for key, value in emb.conditions.items()}
        return Model(name, conditions, locations[["slide_id", "tissue_type", "fold", "location_index"]])
    set20 = load_set20()
    blocks: dict[str, list[np.ndarray]] = {}
    rows = []
    for slide in cohort.itertuples(index=False):
        path = root / name / f"{slide.slide_id}.h5"
        with h5py.File(path, "r") as store:
            names = [v.decode() if isinstance(v, bytes) else str(v) for v in store["condition_names"][:]]
            locations = np.asarray(store["location_index"], dtype=np.int64)
            features = np.asarray(store["features"], dtype=np.float32)
            if str(store.attrs["slide_id"]) != slide.slide_id or int(store.attrs["fold"]) != int(slide.fold):
                raise ValueError(f"{path}: attributes disagree with cohort")
        expected = set20.loc[set20.slide_id.eq(slide.slide_id)].location_index.sort_values().tolist()
        if sorted(locations.tolist()) != expected or not np.isfinite(features).all():
            raise ValueError(f"{path}: locations differ from set20 or non-finite features")
        order = np.argsort(locations)
        for index, condition in enumerate(names):
            if band or not condition.startswith("band_"):
                blocks.setdefault(condition, []).append(features[order, index])
        rows.extend({"slide_id": slide.slide_id, "tissue_type": slide.tissue_type, "fold": int(slide.fold),
                     "location_index": int(v)} for v in locations[order])
    conditions = {key: unit(np.concatenate(value)).astype(np.float32) for key, value in blocks.items()}
    return Model(name, conditions, pd.DataFrame(rows))


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - np.clip(np.sum(a.astype(np.float64) * b.astype(np.float64), axis=1), -1.0, 1.0)


# ----------------------------------------------------------------------------- tasks

def task_list() -> list[dict]:
    return [{"model": model, "part": part} for model in MODELS for part in PARTS]


def make_classifier(name: str):
    from detectability_within_fold import make_classifier as rv04b

    return rv04b(name)


def run_detect(model: Model, out: Path) -> dict:
    """RV04 design (locked outer folds) with the RV04b classifiers; pooled over scanners."""
    folds = model.locations.fold.to_numpy(int)
    source = model.conditions["source_at2"]
    generated = unit(np.concatenate([source] * len(TARGETS))).astype(np.float32)
    target = unit(np.concatenate([model.conditions[f"target_{s}"] for s in TARGETS])).astype(np.float32)
    fold = np.tile(folds, len(TARGETS))
    codes = np.tile(model.slide_code, len(TARGETS))
    scanner_index = np.repeat(np.arange(len(TARGETS)), model.n)
    rows = []
    for held_out in range(5):
        train, test = fold != held_out, fold == held_out
        x_train = np.concatenate([generated[train], target[train]])
        y_train = np.concatenate([np.zeros(train.sum()), np.ones(train.sum())])
        x_test = np.concatenate([generated[test], target[test]])
        truth = np.concatenate([np.zeros(test.sum()), np.ones(test.sum())])
        for name in CLASSIFIERS:
            started = time.time()
            classifier = make_classifier(name)
            classifier.fit(x_train, y_train)
            correct = (classifier.predict(x_test) == truth).astype(int)
            frame = pd.DataFrame({"slide_code": np.concatenate([codes[test], codes[test]]),
                                  "scanner": np.asarray(TARGETS)[np.concatenate([scanner_index[test], scanner_index[test]])],
                                  "cls": np.where(truth == 1, "real_target", "source_at2"), "correct": correct})
            grouped = frame.groupby(["slide_code", "scanner", "cls"], as_index=False).agg(
                n=("correct", "size"), correct=("correct", "sum"))
            grouped["classifier"] = name
            grouped["cv_fold"] = held_out
            rows.append(grouped)
            print(f"{model.name} detect fold {held_out} {name}: {correct.mean():.4f} ({time.time() - started:.0f}s)", flush=True)
    result = pd.concat(rows, ignore_index=True)
    result.insert(0, "slide_id", np.asarray(model.slides)[result.pop("slide_code").to_numpy()])
    write_frame(result, out / "detect.csv.gz")
    return {"rows": len(result)}


def six_scanner_matrix(model: Model) -> tuple[np.ndarray, pd.DataFrame]:
    """[n * 6, d] unit vectors ordered location-major, scanner-minor (as RV01)."""
    stacked = np.stack([model.conditions["source_at2"]] + [model.conditions[f"target_{s}"] for s in TARGETS], axis=1)
    meta = pd.DataFrame({"slide_id": np.repeat(model.locations.slide_id.to_numpy(), 6),
                         "tissue_type": np.repeat(model.locations.tissue_type.to_numpy(), 6),
                         "fold": np.repeat(model.locations.fold.to_numpy(int), 6),
                         "location_index": np.repeat(model.locations.location_index.to_numpy(), 6),
                         "scanner": np.tile(SCANNERS, model.n)})
    return unit(stacked.reshape(-1, stacked.shape[-1])).astype(np.float32), meta


def run_probe6(model: Model, out: Path) -> dict:
    from blur_sharpen_set20 import scanner_probe

    x, meta = six_scanner_matrix(model)
    predictions = scanner_probe({"raw": x}, meta, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    write_frame(predictions[["slide_id", "fold", "location_index", "scanner", "predicted", "correct", "selected_c"]],
                out / "probe6.csv.gz")
    return {"rows": len(predictions), "selected_c": sorted(predictions.selected_c.unique().tolist())}


def run_ri(model: Model, out: Path) -> dict:
    """PathoROB robustness index (robustness_index_utils.py semantics; see module docstring)."""
    from scipy.stats import mode
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import LabelEncoder, Normalizer

    x, meta = six_scanner_matrix(model)
    x = Normalizer(norm="l2").fit_transform(x)
    slide = meta.slide_id.to_numpy()
    tissue = LabelEncoder().fit_transform(meta.tissue_type.to_numpy())
    scanner = meta.scanner.to_numpy()
    margin = int(pd.Series(slide).value_counts().max())
    search = min(K_MAX + margin, len(x))
    _, index = NearestNeighbors(n_neighbors=search, metric="euclidean", algorithm="brute").fit(x).kneighbors(x)
    kept = np.empty((len(x), K_MAX), dtype=np.int64)
    for row in range(len(x)):
        others = index[row][slide[index[row]] != slide[row]]
        if len(others) < K_MAX:
            raise ValueError(f"only {len(others)} other-slide neighbours at row {row}")
        kept[row] = others[:K_MAX]
    slides_per_tissue = meta.drop_duplicates("slide_id").groupby("tissue_type").size()
    anchor = meta.tissue_type.map(slides_per_tissue).ge(2).to_numpy()
    neighbour_tissue = tissue[kept]
    same_tissue = neighbour_tissue == tissue[:, None]
    same_scanner = scanner[kept] == scanner[:, None]
    so = (same_tissue & ~same_scanner)[anchor].astype(np.int8)
    os_ = (~same_tissue & same_scanner)[anchor].astype(np.int8)
    accuracy = {}
    for k in K_GRID:
        predicted = mode(neighbour_tissue[anchor, :k], axis=1, keepdims=False).mode
        accuracy[k] = float(balanced_accuracy_score(tissue[anchor], predicted))
    k_opt = max(K_GRID, key=lambda k: (accuracy[k], -K_GRID.index(k)))  # first maximum, as np.argmax
    # exact chance level: random neighbours from the pool without the anchor's slide
    pool = len(x) - pd.Series(slide).map(pd.Series(slide).value_counts()).to_numpy()
    frame = pd.DataFrame({"slide": slide, "tissue": tissue, "scanner": scanner})
    n_ts = frame.groupby(["tissue", "scanner"]).size()
    n_t = frame.groupby("tissue").size()
    n_s = frame.groupby("scanner").size()
    n_slide_s = frame.groupby(["slide", "scanner"]).size()
    n_slide = frame.groupby("slide").size()
    same_t = n_t.reindex(tissue).to_numpy() - n_slide.reindex(slide).to_numpy()  # same tissue, other slide
    same_t_same_s = n_ts.reindex(pd.MultiIndex.from_arrays([tissue, scanner])).to_numpy() - \
        n_slide_s.reindex(pd.MultiIndex.from_arrays([slide, scanner])).to_numpy()
    expected_so = (same_t - same_t_same_s) / pool
    same_s_other_t = n_s.reindex(scanner).to_numpy() - n_ts.reindex(pd.MultiIndex.from_arrays([tissue, scanner])).to_numpy()
    expected_os = same_s_other_t / pool
    chance = float(expected_so[anchor].sum() / (expected_so[anchor].sum() + expected_os[anchor].sum()))
    np.savez_compressed(out / "ri.npz", so=so, os=os_, anchor_slide=slide[anchor].astype(str),
                        anchor_tissue=meta.tissue_type.to_numpy()[anchor].astype(str), anchor_scanner=scanner[anchor].astype(str),
                        anchor_location=meta.location_index.to_numpy()[anchor], neighbours=kept.astype(np.int32),
                        k_grid=np.asarray(K_GRID), balanced_accuracy=np.asarray([accuracy[k] for k in K_GRID]),
                        k_opt=k_opt, chance=chance, margin=margin, search=search)
    return {"anchors": int(anchor.sum()), "pool": len(x), "k_opt": int(k_opt), "chance": chance,
            "balanced_accuracy": accuracy, "margin": margin}


def run_task(task: dict) -> None:
    out = TASKS / task["model"]
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = load_model(task["model"])
    info = {"detect": run_detect, "probe6": run_probe6, "ri": run_ri}[task["part"]](model, out)
    write_json({"status": "pass", "task": task, "n_locations": model.n, "seconds": round(time.time() - started, 1),
                **info}, out / f"{task['part']}.json")
    print(json.dumps(info, default=str), flush=True)


# ----------------------------------------------------------------------------- aggregate helpers

def between_tissue(weights: np.ndarray, pairs: np.ndarray, counts: np.ndarray) -> np.ndarray:
    from three_axis_evaluation import between_tissue as rv04

    return rv04(weights, pairs, counts)


def pair_matrices(model: Model) -> tuple[np.ndarray, np.ndarray]:
    tissue = model.locations.tissue_type.to_numpy()
    different = (tissue[:, None] != tissue[None, :]).astype(np.float64)
    membership = np.zeros((len(model.slides), model.n))
    membership[model.slide_code, np.arange(model.n)] = 1.0
    x = model.conditions["source_at2"].astype(np.float64)
    pairs = membership @ ((1.0 - x @ x.T) * different) @ membership.T
    counts = membership @ different @ membership.T
    return pairs, counts


def distance_vectors(model: Model, keep: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Slide-level target distances [slides, 5 scanners] for raw and the three corrections."""
    out = {}
    for method in ("raw", "reinhard", "frequency", "combined"):
        columns = []
        for scanner in TARGETS:
            query = model.conditions["source_at2" if method == "raw" else f"{method}_to_{scanner}"]
            columns.append(model.slide_means(distance(query, model.conditions[f"target_{scanner}"]), keep))
        out[method] = np.stack(columns, axis=1)
    return out


def band_rows_new(model: Model) -> pd.DataFrame:
    meta = pd.concat([pd.read_csv(path, dtype={"slide_id": str}) for path in
                      sorted((OUTPUT / "render").glob("*_band.csv.gz"))], ignore_index=True)
    position = {(s, int(l)): i for i, (s, l) in enumerate(zip(model.locations.slide_id, model.locations.location_index))}
    rows = meta.copy()
    index = np.asarray([position[(s, int(l))] for s, l in zip(rows.slide_id, rows.location_index)])
    displacement = np.empty(len(rows))
    gain = np.empty(len(rows))
    for condition, group in rows.groupby("condition"):
        scanner = group.scanner.iloc[0]
        at = index[group.index.to_numpy()]
        base = model.conditions[f"reinhard_to_{scanner}"][at]
        target = model.conditions[f"target_{scanner}"][at]
        changed = model.conditions[condition][at]
        displacement[group.index.to_numpy()] = distance(base, changed)
        gain[group.index.to_numpy()] = distance(base, target) - distance(changed, target)
    rows["embedding_displacement"] = displacement
    rows["target_gain"] = gain
    return rows


def band_rows_original(name: str) -> pd.DataFrame:
    paths = sorted((RV02 / "shards" / name).glob("*.csv.gz"))
    if len(paths) != 103:
        raise FileNotFoundError(f"RV02 {name}: {len(paths)} shards")
    return pd.concat([pd.read_csv(path, dtype={"slide_id": str}) for path in paths], ignore_index=True)


def band_vectors(rows: pd.DataFrame, slides: tuple) -> dict:
    """RV02 slide vectors: mean within slide x scanner, then scanner-equal mean per slide."""

    def vector(frame: pd.DataFrame, value: str) -> np.ndarray:
        by = frame.groupby(["slide_id", "scanner"])[value].mean()
        counts = by.groupby(level="slide_id").size().reindex(slides)
        if counts.isna().any() or (counts != len(TARGETS)).any():
            raise ValueError("band rows incomplete")
        return by.groupby(level="slide_id").mean().reindex(slides).to_numpy(float)

    out = {}
    if len(rows) != len(slides) * 20 * len(TARGETS) * len(BANDS) * len(DOSES) * 2:
        raise ValueError(f"band rows: {len(rows)}")
    for dose in DOSES:
        at = rows[np.isclose(rows.dose_fraction, dose)]
        for band in BANDS:
            part = at[at.band.eq(band)]
            out[("shift", band, dose)] = vector(part, "embedding_displacement")
            out[("gain_target", band, dose)] = vector(part[part.target_like.astype(bool)], "target_gain")
    return out


def retrieval_vector(model: Model) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    """Per-slide top-1 correctness among raw AT2 slide means (other slides as bank)."""
    from prenorm.feature_correction import macro_retrieval

    x = model.conditions["source_at2"].astype(np.float64)
    total = np.zeros((len(model.slides), x.shape[1]))
    np.add.at(total, model.slide_code, x)
    query = unit(total / np.bincount(model.slide_code)[:, None])
    tissue = model.locations.groupby("slide_id").tissue_type.first().reindex(model.slides).to_numpy()
    scores = query @ query.T
    np.fill_diagonal(scores, -np.inf)
    correct = (tissue[np.argmax(scores, axis=1)] == tissue).astype(float)
    reference = macro_retrieval(query, query, tissue)
    counts = pd.Series(tissue).value_counts()
    eligible = pd.Series(tissue).map(counts).ge(2).to_numpy()
    names = sorted(set(tissue[eligible]))
    ours = pd.Series(correct[eligible]).groupby(tissue[eligible]).mean().mean()
    if abs(ours - reference) > 1e-12:
        raise ValueError(f"retrieval disagrees with macro_retrieval: {ours} vs {reference}")
    code = np.asarray([names.index(t) if e else -1 for t, e in zip(tissue, eligible)])
    return correct, code, names, float(reference)


# ----------------------------------------------------------------------------- aggregate

def model_draws(name: str, weights: np.ndarray) -> tuple[dict, dict]:
    """Bootstrap draws [B + 1] per statistic for one model, and per-slide/other info."""
    started = time.time()
    model = load_model(name, band=name in NEW)
    if model.slides != tuple(load_cohort().slide_id):
        raise ValueError(f"{name}: slide order differs from the cohort")
    draws: dict[str, np.ndarray] = {}
    info: dict = {"model": name}
    pairs, counts = pair_matrices(model)
    b = between_tissue(weights, pairs, counts)
    draws["between_tissue_distance"] = b
    vectors = distance_vectors(model)
    pooled = {m: weighted_mean(weights, v.mean(axis=1)) for m, v in vectors.items()}
    for method, value in pooled.items():
        draws[f"target_distance_{method}"] = value
    draws["normalized_distance"] = pooled["raw"] / b
    for j, scanner in enumerate(TARGETS):
        draws[f"normalized_distance_{scanner}"] = weighted_mean(weights, vectors["raw"][:, j]) / b
    draws["colour_share"] = (pooled["raw"] - pooled["reinhard"]) / pooled["raw"]
    draws["frequency_alone_share"] = (pooled["raw"] - pooled["frequency"]) / pooled["raw"]
    draws["frequency_share"] = (pooled["reinhard"] - pooled["combined"]) / pooled["raw"]
    draws["colour_frequency_share"] = (pooled["raw"] - pooled["combined"]) / pooled["raw"]
    rows = band_rows_new(model) if name in NEW else band_rows_original(name)
    bands = band_vectors(rows, model.slides)
    for dose in DOSES:
        tag = f"d{dose:g}"
        for band in BANDS:
            shift = weighted_mean(weights, bands[("shift", band, dose)])
            draws[f"shift_{band}_{tag}"] = shift
            draws[f"normalized_shift_{band}_{tag}"] = shift / b
            gain = weighted_mean(weights, bands[("gain_target", band, dose)])
            draws[f"target_gain_{band}_{tag}"] = gain
            draws[f"normalized_target_gain_{band}_{tag}"] = gain / b
        draws[f"high_over_low_mid_{tag}"] = draws[f"shift_high_{tag}"] / draws[f"shift_low_mid_{tag}"]
        draws[f"normalized_high_minus_low_mid_{tag}"] = (draws[f"shift_high_{tag}"] - draws[f"shift_low_mid_{tag}"]) / b
    correct, code, names, point = retrieval_vector(model)
    eligible = code >= 0
    draws["tissue_retrieval"] = weighted_macro(weights[:, eligible], correct[eligible], code[eligible], len(names))
    info.update({"retrieval_tissues": len(names), "retrieval_slides": int(eligible.sum()), "retrieval_reference": point,
                 "band_rows": len(rows)})
    # detectability
    detect = pd.read_csv(TASKS / name / "detect.csv.gz", dtype={"slide_id": str})
    for classifier in CLASSIFIERS:
        part = detect[detect.classifier.eq(classifier)].groupby("slide_id")[["correct", "n"]].sum()
        accuracy = (part.correct / part.n).reindex(model.slides).to_numpy(float)
        draws[f"detectability_{classifier}"] = weighted_mean(weights, accuracy)
    stacked = np.stack([draws[f"detectability_{c}"] for c in CLASSIFIERS])
    draws["detectability_best"] = stacked.max(axis=0)
    info["best_probe_point"] = CLASSIFIERS[int(np.argmax(stacked[:, 0]))]
    probe = pd.read_csv(TASKS / name / "probe6.csv.gz", dtype={"slide_id": str})
    table = probe.groupby(["slide_id", "scanner"]).correct.mean().unstack("scanner").reindex(
        index=list(model.slides), columns=list(SCANNERS))
    draws["scanner_probe_6way"] = np.mean([weighted_mean(weights, table[s].to_numpy(float)) for s in SCANNERS], axis=0)
    info["probe6_selected_c"] = ";".join(str(c) for c in sorted(probe.selected_c.unique()))
    # robustness index
    ri = np.load(TASKS / name / "ri.npz")
    anchor_code = pd.Index(model.slides).get_indexer(ri["anchor_slide"].astype(str))
    so_cum = np.cumsum(ri["so"].astype(np.float64), axis=1)
    os_cum = np.cumsum(ri["os"].astype(np.float64), axis=1)
    so_slide = np.zeros((len(model.slides), K_MAX))
    os_slide = np.zeros((len(model.slides), K_MAX))
    np.add.at(so_slide, anchor_code, so_cum)
    np.add.at(os_slide, anchor_code, os_cum)
    curve = (weights @ so_slide) / (weights @ (so_slide + os_slide))  # [B + 1, K_MAX]
    k_opt = int(ri["k_opt"])
    draws["robustness_index"] = curve[:, k_opt - 1]
    info.update({"ri_k_opt": k_opt, "ri_chance": float(ri["chance"]), "ri_curve": curve,
                 "ri_balanced_accuracy": dict(zip(ri["k_grid"].tolist(), ri["balanced_accuracy"].tolist())),
                 "seconds": round(time.time() - started, 1)})
    if name == "exaonepath":
        failures = []
        for path in sorted((OUTPUT / "render").glob("*_macenko.csv.gz")):
            frame = pd.read_csv(path, dtype={"slide_id": str})
            failures.append(frame[~frame.status.eq("ok")])
        failures = pd.concat(failures, ignore_index=True) if failures else pd.DataFrame()
        info["macenko_failures"] = len(failures)
        write_frame(failures, OUTPUT / "qc/macenko_failures.csv")
        # Sensitivity: the released code substitutes the unnormalized image on failure (flagged here);
        # recompute the distance-based metrics without every location that had any failed image.
        failed = set(zip(failures.slide_id.astype(str), failures.location_index.astype(int))) if len(failures) else set()
        keep = np.asarray([(s, int(l)) not in failed for s, l in zip(model.locations.slide_id, model.locations.location_index)])
        info["macenko_failed_locations"] = int((~keep).sum())
        info["macenko_failed_slides"] = int(model.locations.slide_id[~keep].nunique())
        clean = distance_vectors(model, keep)
        clean_pooled = {m: weighted_mean(weights, np.nanmean(v, axis=1)) for m, v in clean.items()}
        draws["sensitivity_macenko_ok_normalized_distance"] = clean_pooled["raw"] / b
        draws["sensitivity_macenko_ok_colour_share"] = (clean_pooled["raw"] - clean_pooled["reinhard"]) / clean_pooled["raw"]
        draws["sensitivity_macenko_ok_frequency_alone_share"] = (clean_pooled["raw"] - clean_pooled["frequency"]) / clean_pooled["raw"]
        draws["sensitivity_macenko_ok_frequency_share"] = (clean_pooled["reinhard"] - clean_pooled["combined"]) / clean_pooled["raw"]
    return draws, info


def fmt(row, digits=3) -> str:
    if row is None or not np.isfinite(row["estimate"]):
        return "n/a"
    return f"{row['estimate']:.{digits}f} [{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]"


def qc_tables() -> dict:
    qc: dict = {}
    frames = [pd.read_csv(p, dtype={"slide_id": str}) for p in sorted((OUTPUT / "qc").glob("*_review_parity.csv"))
              if os.path.getsize(p) > 1]
    frames = [f for f in frames if len(f)]
    review = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["model", "cosine", "slide_id"])
    qc["review"] = review.groupby("model").agg(comparisons=("cosine", "size"), slides=("slide_id", "nunique"),
                                               min_cosine=("cosine", "min"), median_cosine=("cosine", "median"),
                                               below_0999=("cosine", lambda v: int((v < PARITY_MIN).sum()))).reset_index()
    uni = pd.concat([pd.read_csv(p, dtype={"slide_id": str}) for p in sorted((OUTPUT / "qc").glob("*_uni_v1_parity.csv"))],
                    ignore_index=True)
    qc["uni"] = uni.groupby("condition").cosine.agg(["size", "min", "median"]).reset_index()
    qc["uni_min"] = float(uni.cosine.min())
    qc["uni_n"] = len(uni)
    band = pd.DataFrame([json.loads(Path(p).read_text()) for p in sorted((OUTPUT / "qc").glob("*_band_parity.json"))])
    qc["band"] = band
    renders = pd.concat([pd.read_csv(p, dtype={"slide_id": str}) for p in sorted((OUTPUT / "render").glob("*_band.csv.gz"))],
                        ignore_index=True)
    renders["relative_error"] = (renders.achieved_rms_od / renders.target_rms_od - 1).abs()
    qc["dose"] = renders.groupby(["band", "dose_fraction", "sign"]).relative_error.agg(median="median", maximum="max").reset_index()
    parity = OUTPUT / "parity/parity.json"
    qc["parity"] = json.loads(parity.read_text()) if parity.exists() else None
    return qc


def original_parity(cells: pd.DataFrame) -> pd.DataFrame:
    """The four original PFMs against RV04 (same embeddings, same resamples) and RV01/RV02."""
    rows = []
    rv04 = pd.read_csv(RV04_CELLS)
    rv04 = rv04[rv04.dataset.eq("pannormal") & rv04.scanner.eq("pooled")]
    checks = (("normalized_distance", "raw", "normalized_raw_distance"),
              ("between_tissue_distance", "raw", "between_tissue_distance"),
              ("target_distance_raw", "raw", "target_distance"),
              ("target_distance_reinhard", "reinhard", "target_distance"),
              ("target_distance_frequency", "frequency", "target_distance"),
              ("target_distance_combined", "combined", "target_distance"),
              ("detectability_linear", "raw", "detectability"))
    for model in ORIGINAL:
        for ours, method, statistic in checks:
            ref = rv04[rv04.pfm.eq(model) & rv04.method.eq(method) & rv04.statistic.eq(statistic)]
            mine = cells[cells.model.eq(model) & cells.statistic.eq(ours)]
            rows.append({"reference": "RV04 cell_statistics", "model": model, "statistic": ours,
                         "reference_estimate": float(ref.estimate.iloc[0]), "reference_ci_low": float(ref.ci_low.iloc[0]),
                         "reference_ci_high": float(ref.ci_high.iloc[0]), "estimate": float(mine.estimate.iloc[0]),
                         "ci_low": float(mine.ci_low.iloc[0]), "ci_high": float(mine.ci_high.iloc[0])})
    rv02 = pd.read_csv(RV02 / "summary_statistics.csv")
    rv02 = rv02[rv02.set.eq("set20") & rv02.scanner.eq("all")]
    for model in ORIGINAL:
        for dose in DOSES:
            for band in BANDS:
                for ours, statistic, subset in ((f"shift_{band}_d{dose:g}", f"shift_{band}", "both_signs"),
                                                (f"target_gain_{band}_d{dose:g}", f"target_gain_{band}", "target_direction")):
                    ref = rv02[rv02.model.eq(model) & rv02.statistic.eq(statistic) & np.isclose(rv02.dose_fraction, dose)
                               & rv02.subset.eq(subset)]
                    mine = cells[cells.model.eq(model) & cells.statistic.eq(ours)]
                    rows.append({"reference": "RV02 summary_statistics (point estimate; RV02 used its own resamples)",
                                 "model": model, "statistic": ours, "reference_estimate": float(ref["mean"].iloc[0]),
                                 "reference_ci_low": float(ref.ci_low.iloc[0]), "reference_ci_high": float(ref.ci_high.iloc[0]),
                                 "estimate": float(mine.estimate.iloc[0]), "ci_low": float(mine.ci_low.iloc[0]),
                                 "ci_high": float(mine.ci_high.iloc[0])})
    if RV01_PROBE.exists():
        rv01 = pd.read_csv(RV01_PROBE)
        ref = rv01[rv01.level.eq("location") & rv01.condition.eq("raw")]
        mine = cells[cells.model.eq("uni_v1") & cells.statistic.eq("scanner_probe_6way")]
        rows.append({"reference": "RV01 six-way probe, raw (RV01 re-embedded UNI v1)", "model": "uni_v1",
                     "statistic": "scanner_probe_6way", "reference_estimate": float(ref.balanced_accuracy.iloc[0]),
                     "reference_ci_low": float(ref.ci_low.iloc[0]), "reference_ci_high": float(ref.ci_high.iloc[0]),
                     "estimate": float(mine.estimate.iloc[0]), "ci_low": float(mine.ci_low.iloc[0]),
                     "ci_high": float(mine.ci_high.iloc[0])})
    frame = pd.DataFrame(rows)
    frame["difference"] = frame.estimate - frame.reference_estimate
    return frame


def aggregate() -> None:
    statuses = []
    for task in task_list():
        path = TASKS / task["model"] / f"{task['part']}.json"
        statuses.append(json.loads(path.read_text()) if path.exists() else {"status": "missing", "task": task})
    missing = [s["task"] for s in statuses if s.get("status") != "pass"]
    if missing:
        raise SystemExit(f"{len(missing)} tasks missing: {missing[:5]}")
    cohort = load_cohort()
    weights = bootstrap_weights("pannormal", len(cohort))
    all_draws, infos = {}, {}
    for name in MODELS:
        all_draws[name], infos[name] = model_draws(name, weights)
        print(f"aggregated {name} in {infos[name]['seconds']}s", flush=True)
    rows = []
    for name, draws in all_draws.items():
        for statistic, values in draws.items():
            rows.append({"model": name, "label": LABELS[name], "cell": CELL[name], "statistic": statistic, **summarize(values)})
    cells = pd.DataFrame(rows)
    write_frame(cells, OUTPUT / "model_statistics.csv")
    curve_rows = []
    for name in MODELS:
        curve = infos[name]["ri_curve"]
        for k in range(1, K_MAX + 1):
            curve_rows.append({"model": name, "k": k, **summarize(curve[:, k - 1]),
                               "k_opt": infos[name]["ri_k_opt"], "chance": infos[name]["ri_chance"]})
    write_frame(pd.DataFrame(curve_rows), OUTPUT / "robustness_index_curve.csv")
    write_frame(pd.DataFrame([{"model": n, "k": k, "balanced_accuracy": v} for n in MODELS
                              for k, v in infos[n]["ri_balanced_accuracy"].items()]), OUTPUT / "robustness_index_knn_accuracy.csv")

    # contrasts and group means (shared resamples => paired)
    contrast_rows = []

    def contrast(label: str, hypothesis: str, statistic: str, left: tuple, right: tuple, role: str) -> dict:
        a = np.mean([all_draws[m][statistic] for m in left], axis=0)
        b = np.mean([all_draws[m][statistic] for m in right], axis=0)
        row = {"hypothesis": hypothesis, "contrast": label, "statistic": statistic, "role": role,
               "left": "+".join(left), "right": "+".join(right), **summarize(a - b),
               "left_estimate": float(a[0]), "right_estimate": float(b[0]),
               "left_ci_low": summarize(a)["ci_low"], "left_ci_high": summarize(a)["ci_high"],
               "right_ci_low": summarize(b)["ci_low"], "right_ci_high": summarize(b)["ci_high"]}
        contrast_rows.append(row)
        return row

    h1_stats = (("normalized_distance", "criterion"), ("detectability_best", "criterion"),
                ("normalized_shift_high_d0.25", "reported"), ("normalized_shift_high_d0.5", "reported (secondary dose)"),
                ("robustness_index", "secondary (amendment)"), ("detectability_linear", "information"),
                ("detectability_mlp", "information"), ("detectability_knn", "information"),
                ("scanner_probe_6way", "information"), ("tissue_retrieval", "information"))
    pairs = (("SEAL-UNI2 - UNI2-h", ("seal_uni2",), ("uni2",)),
             ("SEAL-CONCH - CONCH (pre-projection)", ("seal_conch_pre",), ("conch_pre",)),
             ("SEAL-CONCH - CONCH (projected; secondary)", ("seal_conch_projected",), ("conch_projected",)))
    for label, left, right in pairs:
        for statistic, role in h1_stats:
            contrast(label, "H1", statistic, left, right, role)
    for statistic, role in (("frequency_share", "primary"), ("normalized_shift_high_d0.25", "primary"),
                            ("normalized_shift_high_d0.5", "secondary dose"), ("high_over_low_mid_d0.25", "information"),
                            ("frequency_alone_share", "information")):
        contrast("image-only WSI - heterogeneous", "H2", statistic, IMAGE_ONLY_WSI, HETEROGENEOUS, role)
    for statistic, role in (("normalized_distance", "primary"), ("detectability_best", "primary"),
                            ("detectability_linear", "information"), ("robustness_index", "secondary (amendment)"),
                            ("tissue_retrieval", "alongside")):
        contrast("heterogeneous - narrow", "H3", statistic, HETEROGENEOUS, NARROW, role)
        contrast("anchored - image-only (anchor alternative)", "H3", statistic, ANCHORED, IMAGE_ONLY, role)
        contrast("anchored/narrow (SEAL) - image-only/narrow (WSI)", "H3", statistic, ("seal_uni2", "seal_conch_pre"),
                 IMAGE_ONLY_WSI, "information (2x2 cell)")
        contrast("anchored/heterogeneous (CONCH, PLIP) - image-only/heterogeneous (DINOv2)", "H3", statistic,
                 ("conch_pre", "plip"), ("dinov2",), "information (2x2 cell)")
    for statistic in ("colour_share", "frequency_share", "detectability_best", "detectability_linear", "normalized_distance"):
        contrast("EXAONEPath as released - raw input", "H4", statistic, ("exaonepath",), ("exaonepath_raw",), "controlled")
        contrast("EXAONEPath as released - other image-only WSI", "H4", statistic, ("exaonepath",), OTHER_WSI, "cross-model")
    contrasts = pd.DataFrame(contrast_rows)
    write_frame(contrasts, OUTPUT / "contrasts.csv")
    qc = qc_tables()
    parity = original_parity(cells)
    write_frame(parity, OUTPUT / "qc/original_pfm_parity.csv")
    write_frame(qc["dose"], OUTPUT / "qc/dose_audit.csv")
    write_frame(qc["review"], OUTPUT / "qc/review_parity_summary.csv")
    write_frame(qc["uni"], OUTPUT / "qc/uni_v1_parity_summary.csv")
    write_json({name: {k: v for k, v in info.items() if k != "ri_curve"} for name, info in infos.items()},
               OUTPUT / "qc/model_info.json")
    write_text(render_summary(cells, contrasts, qc, parity, infos, statuses), OUTPUT / "summary.md")
    print("summary written", flush=True)


# ----------------------------------------------------------------------------- summary

def render_summary(cells, contrasts, qc, parity, infos, statuses) -> str:
    get = cells.set_index(["model", "statistic"])

    def cell(model, statistic, digits=3):
        return fmt(get.loc[(model, statistic)], digits) if (model, statistic) in get.index else "n/a"

    def crow(hypothesis, label, statistic):
        match = contrasts[contrasts.hypothesis.eq(hypothesis) & contrasts.contrast.eq(label) & contrasts.statistic.eq(statistic)]
        return match.iloc[0] if len(match) else None

    def cfmt(row, digits=3):
        return "n/a" if row is None else f"{row.estimate:+.{digits}f} [{row.ci_low:+.{digits}f}, {row.ci_high:+.{digits}f}]"

    lines = ["# RV13 anchor, frequency or data diversity", "",
             "Status: computed; not yet discussed. Protocol: `analysis/revision/README.md`, RV13 (metric 6 added by amendment).",
             "", "## What ran", "",
             "- PanNormal `set20`: 103 slides x 20 locations; five AT2 -> target directions.",
             "- New embeddings (`anchor_frequency_diversity_embed.py`): 81 images per location rendered once "
             "(raw six scanners; Reinhard, frequency, colour + frequency as RV03; RV02 band manipulations: 3 bands x "
             "2 signs x 2 doses x 5 targets) and embedded by CONCH / SEAL-CONCH (pre-projection and projected), "
             "SEAL-UNI2, PLIP, DINOv2 ViT-L/14, EXAONEPath as released (released Macenko + resize 256 / centre "
             "crop 224 / ImageNet normalization) and EXAONEPath without Macenko (off-label).",
             "- Original PFMs (UNI v1, UNI2-h, Virchow2, H-optimus-1): RV03 embeddings and RV02 band rows, not re-embedded.",
             "- Metrics (`anchor_frequency_diversity_metrics.py`): see the module docstring; 2,000 slide-bootstrap "
             "resamples from `corrected_embedding_reader.bootstrap_weights` shared by every model and metric; "
             "percentile 95% CIs; contrasts and group means are computed within each resample (paired).",
             f"- Tasks: {len(statuses)} CPU tasks (model x detect / probe6 / ri), all passed.", "",
             "## QC and parity", ""]
    parity_json = qc["parity"]
    if parity_json:
        lines.append(f"- Loader parity with official code (`parity/parity.json`): pass = {parity_json['pass']} "
                     f"(threshold {parity_json['threshold']}).")
        for item in parity_json["min_cosine_by_check"]:
            lines.append(f"  - {item['model']} / {item['check']}: min cosine {item['cosine']:.6f}")
        mac = parity_json["exaonepath_macenko"]
        lines.append(f"  - EXAONEPath Macenko port vs released code: reference max abs diff HERef {mac['HERef_max_abs_diff']:.2e}, "
                     f"maxCRef {mac['maxCRef_max_abs_diff']:.2e}; normalized images max abs diff {mac['normalized_image_max_abs_diff']:.2e}.")
        lines.append(f"  - Released reference (fitted on the TCGA target image by `macenko_normalizer()`): HERef "
                     f"{np.round(mac['HERef'], 4).tolist()}, maxCRef {np.round(mac['maxCRef'], 4).tolist()} "
                     f"(class defaults, overwritten by the fit: {np.round(mac['class_default_HERef'], 4).tolist()}, "
                     f"{np.round(mac['class_default_maxCRef'], 4).tolist()}).")
        lines.append(f"  - PLIP / DINOv2 float-path pixel max abs diff vs released processors: "
                     f"{parity_json['pixel_max_abs_diff']}.")
    else:
        lines.append("- Loader parity: `parity/parity.json` missing.")
    review = qc["review"]
    for row in review.itertuples(index=False):
        lines.append(f"- Review features (`outputs/encoder_review_2026-09-25/features`, 3 locations/slide = `paper_band_probe`): "
                     f"{row.model}: {row.comparisons} comparisons on {row.slides} slides, min cosine {row.min_cosine:.5f}, "
                     f"below 0.999: {row.below_0999}.")
    lines.append(f"- Regenerated Reinhard / frequency / colour + frequency embedded with UNI v1 (RV03 path) vs RV03 shards: "
                 f"{qc['uni_n']} images, min cosine {qc['uni_min']:.6f} (gate 0.999).")
    band = qc["band"]
    lines.append(f"- Band renders vs RV02 shards: {int(band.identical.sum())}/{len(band)} slides identical "
                 f"(dose, achieved dose, coefficient, target direction; tolerance 1e-12 for the CSV round trip; max abs diff "
                 f"{band[['target_rms_od_max_abs_diff', 'achieved_rms_od_max_abs_diff', 'coefficient_max_abs_diff']].max().max():.1e}).")
    dose = qc["dose"]
    lines.append(f"- Dose audit: median |achieved/target - 1| max over cells {dose['median'].max():.2e} (RV02 limit 0.02); "
                 f"max {dose['maximum'].max():.2e}.")
    info = infos["exaonepath"]
    lines.append(f"- EXAONEPath Macenko failures (exception or NaN; the released code falls back to the input image, which "
                 f"is kept as released and flagged per image in `qc/macenko_failures.csv`): {info.get('macenko_failures', 'n/a')} "
                 f"of {103 * 20 * 81} images, at {info.get('macenko_failed_locations', 0)} locations on "
                 f"{info.get('macenko_failed_slides', 0)} slides.")
    if info.get("macenko_failed_locations", 0):
        lines.append("  - Sensitivity without those locations (EXAONEPath as released): normalized distance "
                     f"{cell('exaonepath', 'sensitivity_macenko_ok_normalized_distance')} (all: {cell('exaonepath', 'normalized_distance')}); "
                     f"colour share {cell('exaonepath', 'sensitivity_macenko_ok_colour_share')} (all: {cell('exaonepath', 'colour_share')}); "
                     f"frequency share {cell('exaonepath', 'sensitivity_macenko_ok_frequency_share')} (all: {cell('exaonepath', 'frequency_share')}).")
    lines += ["", "Original PFMs recomputed here vs earlier analyses (same embeddings):", "",
              "| Reference | Model | Statistic | Reference | Here | Difference |", "| --- | --- | --- | --- | --- | --- |"]
    for row in parity.itertuples(index=False):
        lines.append(f"| {row.reference} | {LABELS[row.model]} | {row.statistic} | {row.reference_estimate:.4f} | "
                     f"{row.estimate:.4f} | {row.difference:+.1e} |")

    lines += ["", "## Model table", "",
              "Normalized distance, detectability and shares: estimate [95% CI]. Detectability chance 0.5; six-way probe chance 0.167; "
              f"RI chance {infos['uni_v1']['ri_chance']:.3f} (random neighbours without the anchor's slide, from the label counts).", "",
              "| Model | Cell | Normalized distance | Detect. linear | Detect. MLP | Detect. k-NN | Best probe | Six-way probe | "
              "RI (k*) | Tissue retrieval |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in MODELS:
        lines.append(f"| {LABELS[m]} | {CELL[m]} | {cell(m, 'normalized_distance')} | {cell(m, 'detectability_linear')} | "
                     f"{cell(m, 'detectability_mlp')} | {cell(m, 'detectability_knn')} | {cell(m, 'detectability_best')} "
                     f"({infos[m]['best_probe_point']}) | {cell(m, 'scanner_probe_6way')} | {cell(m, 'robustness_index')} "
                     f"({infos[m]['ri_k_opt']}) | {cell(m, 'tissue_retrieval')} |")
    lines += ["", "| Model | Between-tissue distance | Raw target distance | Colour share | Frequency-alone share | Frequency share |",
              "| --- | --- | --- | --- | --- | --- |"]
    for m in MODELS:
        lines.append(f"| {LABELS[m]} | {cell(m, 'between_tissue_distance')} | {cell(m, 'target_distance_raw', 4)} | "
                     f"{cell(m, 'colour_share')} | {cell(m, 'frequency_alone_share')} | {cell(m, 'frequency_share')} |")
    for dose in DOSES:
        tag = f"d{dose:g}"
        lines += ["", f"Band sensitivity, dose {dose:g} (shift / between-tissue distance; high / low-mid ratio; "
                  "target-direction high-band gain / between-tissue distance):", "",
                  "| Model | Low-mid | Mid | High | High / low-mid | High-band target gain |", "| --- | --- | --- | --- | --- | --- |"]
        for m in MODELS:
            lines.append(f"| {LABELS[m]} | {cell(m, f'normalized_shift_low_mid_{tag}', 4)} | {cell(m, f'normalized_shift_mid_{tag}', 4)} | "
                         f"{cell(m, f'normalized_shift_high_{tag}', 4)} | {cell(m, f'high_over_low_mid_{tag}', 2)} | "
                         f"{cell(m, f'normalized_target_gain_high_{tag}', 4)} |")

    lines += ["", "## Read-outs (as pre-specified)", "", "### H1 (controlled, within backbone)", "",
              "Criterion: supported only if both contrasts lower normalized distance and detectability (best probe), "
              "each with CI < 0. Normalized high-band sensitivity (dose 0.25) is reported; RI is a secondary endpoint "
              "(amendment) and does not enter the criterion.", "",
              "| Contrast | Normalized distance | Best-probe detectability | Normalized high-band shift (0.25) | RI (secondary) |",
              "| --- | --- | --- | --- | --- |"]
    verdict = []
    for label in ("SEAL-UNI2 - UNI2-h", "SEAL-CONCH - CONCH (pre-projection)", "SEAL-CONCH - CONCH (projected; secondary)"):
        d = crow("H1", label, "normalized_distance")
        p = crow("H1", label, "detectability_best")
        lines.append(f"| {label} | {cfmt(d)} | {cfmt(p)} | {cfmt(crow('H1', label, 'normalized_shift_high_d0.25'), 4)} | "
                     f"{cfmt(crow('H1', label, 'robustness_index'))} |")
        if "secondary" not in label:
            verdict.append(bool(d.ci_high < 0 and p.ci_high < 0))
    lines += ["", f"H1: {'supported' if all(verdict) else 'not supported'} "
              f"(SEAL-UNI2 - UNI2-h meets criterion: {verdict[0]}; SEAL-CONCH - CONCH meets criterion: {verdict[1]}).", "",
              "### H2 (frequency reliance): image-only WSI (UNI v1, UNI2-h, Virchow2, H-optimus-1, EXAONEPath) vs "
              "heterogeneous-source (CONCH, PLIP, DINOv2)", "",
              "| Model | Frequency share | Normalized high-band shift (0.25) | Normalized high-band shift (0.50) |", "| --- | --- | --- | --- |"]
    for m in IMAGE_ONLY_WSI + HETEROGENEOUS:
        lines.append(f"| {LABELS[m]} | {cell(m, 'frequency_share')} | {cell(m, 'normalized_shift_high_d0.25', 4)} | "
                     f"{cell(m, 'normalized_shift_high_d0.5', 4)} |")
    for statistic, digits in (("frequency_share", 3), ("normalized_shift_high_d0.25", 4), ("normalized_shift_high_d0.5", 4)):
        row = crow("H2", "image-only WSI - heterogeneous", statistic)
        lines.append(f"- {statistic}: image-only WSI mean {row.left_estimate:.{digits}f} [{row.left_ci_low:.{digits}f}, "
                     f"{row.left_ci_high:.{digits}f}]; heterogeneous mean {row.right_estimate:.{digits}f} "
                     f"[{row.right_ci_low:.{digits}f}, {row.right_ci_high:.{digits}f}]; difference {cfmt(row, digits)}; "
                     f"direction as expected (> 0): {row.estimate > 0}; CI excludes 0: {row.ci_low > 0 or row.ci_high < 0}.")
    lines += ["", "### H3 (source diversity): heterogeneous (CONCH, PLIP, DINOv2) vs narrow (UNI v1, UNI2-h, Virchow2, "
              "H-optimus-1, EXAONEPath, SEAL-UNI2, SEAL-CONCH)", ""]
    for statistic in ("normalized_distance", "detectability_best", "detectability_linear", "robustness_index", "tissue_retrieval"):
        row = crow("H3", "heterogeneous - narrow", statistic)
        lines.append(f"- {statistic}: heterogeneous mean {row.left_estimate:.3f} [{row.left_ci_low:.3f}, {row.left_ci_high:.3f}]; "
                     f"narrow mean {row.right_estimate:.3f} [{row.right_ci_low:.3f}, {row.right_ci_high:.3f}]; "
                     f"difference {cfmt(row)} ({row.role}).")
    d3 = crow("H3", "heterogeneous - narrow", "normalized_distance")
    p3 = crow("H3", "heterogeneous - narrow", "detectability_best")
    r3 = crow("H3", "heterogeneous - narrow", "tissue_retrieval")
    lines.append(f"- Mechanical reading: heterogeneous lower on normalized distance (difference CI < 0): {d3.ci_high < 0}; "
                 f"lower on best-probe detectability (CI < 0): {p3.ci_high < 0}; tissue retrieval not lower "
                 f"(difference CI not entirely < 0): {not r3.ci_high < 0}.")
    da = crow("H3", "anchored - image-only (anchor alternative)", "normalized_distance")
    pa = crow("H3", "anchored - image-only (anchor alternative)", "detectability_best")
    lines.append(f"- Anchor alternative, mechanical reading: anchored lower than image-only on normalized distance "
                 f"(CI < 0): {da.ci_high < 0}; on best-probe detectability (CI < 0): {pa.ci_high < 0}.")
    lines += ["", "Anchor alternative (anchored: SEAL-UNI2, SEAL-CONCH, CONCH, PLIP; image-only: four WSI PFMs, EXAONEPath, DINOv2) "
              "and 2x2 cells:", ""]
    for label in ("anchored - image-only (anchor alternative)", "anchored/narrow (SEAL) - image-only/narrow (WSI)",
                  "anchored/heterogeneous (CONCH, PLIP) - image-only/heterogeneous (DINOv2)"):
        for statistic in ("normalized_distance", "detectability_best", "robustness_index", "tissue_retrieval"):
            row = crow("H3", label, statistic)
            lines.append(f"- {label}, {statistic}: {row.left_estimate:.3f} vs {row.right_estimate:.3f}; difference {cfmt(row)}.")
    lines += ["", "### H4 (EXAONEPath stain normalization)", "",
              "| Statistic | As released | Raw input (off-label) | Released - raw | Other image-only WSI mean | Released - others |",
              "| --- | --- | --- | --- | --- | --- |"]
    for statistic in ("colour_share", "frequency_share", "detectability_best", "detectability_linear", "normalized_distance"):
        a = crow("H4", "EXAONEPath as released - raw input", statistic)
        b = crow("H4", "EXAONEPath as released - other image-only WSI", statistic)
        lines.append(f"| {statistic} | {cell('exaonepath', statistic)} | {cell('exaonepath_raw', statistic)} | {cfmt(a)} | "
                     f"{b.right_estimate:.3f} [{b.right_ci_low:.3f}, {b.right_ci_high:.3f}] | {cfmt(b)} |")
    colour = crow("H4", "EXAONEPath as released - other image-only WSI", "colour_share")
    freq = crow("H4", "EXAONEPath as released - other image-only WSI", "frequency_share")
    best = get.loc[("exaonepath", "detectability_best")]
    lines += ["", f"- Lower colour share than the other image-only WSI models: {colour.ci_high < 0} (difference {cfmt(colour)}).",
              f"- Frequency share not lower (difference CI not entirely below 0): {not freq.ci_high < 0} "
              f"(difference {cfmt(freq)}).",
              f"- High detectability (best probe >= 0.95, the RV04 H4a level): {bool(best.estimate >= 0.95)} ({fmt(best)}).",
              "", "## Notes for the protocol log (proposed deviations / implementation choices)", ""]
    lines += NOTES
    lines += ["", "## Files", "",
              "- `model_statistics.csv`: every statistic x model, estimate and CI.",
              "- `contrasts.csv`: H1-H4 contrasts and group means (paired resamples).",
              "- `robustness_index_curve.csv` (RI, k = 1..50), `robustness_index_knn_accuracy.csv`; per-anchor SO/OS in "
              "`tasks/<model>/ri.npz`.",
              "- `tasks/<model>/`: detectability counts, six-way probe predictions, RI inputs.",
              "- `embeddings/<model>/<slide>.h5`, `render/` (band metadata, Macenko status), `qc/`, `parity/`.", ""]
    return "\n".join(lines)


NOTES = [
    "- Loaders: CONCH, SEAL-CONCH and SEAL-UNI2 reuse the review loader/embedding path "
    "(`review_multiencoder_scanner.load_encoder`/`embed`; bicubic upsampling on float tensors, clamp, fp16/bf16), "
    "as instructed, rather than CONCH's PIL transform; the difference is quantified in `parity/parity.csv` (information row).",
    "- PLIP and DINOv2 released processors quantize through PIL; they are reproduced on float tensors so that the "
    "unquantized RV02 band renders (perturbations of ~1 intensity level at dose 0.25) are not rounded; parity with the "
    "released processors on uint8 patches is in `parity/`. DINOv2's released preprocessing centre-crops 224 of the 256 field.",
    "- EXAONEPath: timm ViT-B/16 with the HF config (LayerNorm eps 1e-5 as `VisionTransformer(**config)`); the released "
    "`macenko.py` needs Python >= 3.10 (`match`), so the embedding job uses a line-by-line port and the official-path "
    "half of the parity check runs in the trident-uni2 env (Python 3.10, same torch/torchvision); torchstain 1.4.1 is "
    "vendored under `analysis/revision/third_party/torchstain` (MIT). The released `macenko_normalizer()` fits its "
    "reference to the bundled TCGA target image, replacing the class-default HERef/maxCRef; 'as released' follows the "
    "model card (fitted reference).",
    "- PLIP: `vinid/plip` ships only `pytorch_model.bin`, which the installed transformers refuses to load with "
    "torch < 2.6; the same weights are loaded with `torch.load(weights_only=True)` into `CLIPModel(config)` "
    "(only `position_ids` buffers ignored).",
    "- EXAONEPath Macenko failures (pale or degenerate images) keep the released fallback (unnormalized input) and are "
    "flagged per image; a sensitivity without the affected locations is reported for the distance-based metrics.",
    "- Detectability uses the locked outer slide folds (RV04 design; raw embeddings involve no correction fitting) with "
    "the RV04b classifiers; the AT2 class holds each AT2 embedding once per direction, as RV04. Best probe = maximum of "
    "the three pooled means within each resample (point estimate's probe named in the table).",
    "- Six-way scanner probe = RV01's (`blur_sharpen_set20.scanner_probe`, class-balanced multinomial logistic "
    "regression with C chosen by inner locked folds), not the C = 0.1 liblinear binary probe.",
    "- Band sensitivity uses dose 0.25 as primary (RV02's primary dose); 0.50 reported. Shifts average both signs.",
    "- H2-H4 have no numeric criterion in the protocol; the read-outs report group means / paired differences with "
    "CIs and whether the CI excludes 0; 'high detectability' in H4 is read against 0.95 (RV04 H4a).",
    "- Group means weight models equally; CONCH and SEAL-CONCH enter groups with their pre-projection features.",
    "- RI chance level computed exactly from the label counts is 0.089 (36 anchor tissues, on average 2.96 slides per "
    "anchor's tissue); the amendment's text gave ~0.12 as an approximation.",
    "- RI (amendment): Euclidean k-NN on L2-normalized vectors (sklearn brute force, as PathoROB's KNeighborsClassifier), "
    "margin = 120 (largest slide); neighbours truncated at 50 after removing same-slide ones; bootstrap weights "
    "anchors by their slide's resample count with neighbours fixed.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("list-tasks", "task", "aggregate"))
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    os.chdir(PROJECT)
    tasks = task_list()
    if args.command == "list-tasks":
        for index, task in enumerate(tasks):
            print(index, json.dumps(task))
    elif args.command == "task":
        if args.task_index is None or not 0 <= args.task_index < len(tasks):
            raise SystemExit(f"--task-index must be in [0, {len(tasks) - 1}]")
        print(json.dumps(tasks[args.task_index]), flush=True)
        run_task(tasks[args.task_index])
    else:
        aggregate()


if __name__ == "__main__":
    main()
