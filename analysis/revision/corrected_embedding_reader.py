#!/usr/bin/env python3
"""Shared reader and statistics helpers for RV04, RV05 and RV09 (library, no sbatch).

RV03 writes one HDF5 shard per PanNormal slide and per PLISM section:

* ``results/corrected_embeddings/pannormal/<pfm>/<slide_id>.h5``: ``features``
  [20, n_conditions, d], ``condition_names``, ``location_index`` [20], ``source_index``
  [20]; attributes ``pfm``, ``slide_id``, ``tissue_type``, ``fold``.
* ``results/corrected_embeddings/plism/<pfm>/<section>.h5``: ``features``
  [n_locations, n_conditions, d], ``condition_names``, per-location ``core``, location id
  and ``tissue_type``.
* ``results/corrected_embeddings/manifest.json`` with ``"complete": true``.

Condition names are ``source_at2``, ``target_<scanner>`` and ``<method>_to_<scanner>``;
PLISM conditions fitted per PanNormal fold carry the suffix ``_fold<k>``. Every analysis
sees the embeddings as one row per location plus a dictionary of condition matrices
(:class:`EmbeddingSet`). Fold variants of one condition are returned together by
:meth:`EmbeddingSet.variants`; analyses average their metrics over variants with equal
weight within each section, as the paper did for PLISM.

Aggregation units follow the paper: PanNormal location -> slide; PLISM location -> core ->
section. The bootstrap unit is the slide (PanNormal) or the section (PLISM); the same
2,000 resamples (fixed seeds below) are used by RV04, RV05 and RV09 so that differences
between methods, PFMs and analyses are paired.

The smoke-test adapter that presents older stores in this structure is kept separately in
``smoke_embedding_adapter.py`` and is only used when ``--source smoke`` is passed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats


PROJECT = Path(__file__).resolve().parents[2]
REVISION = PROJECT / "analysis/revision"
RESULTS = REVISION / "results"
EMBEDDINGS = RESULTS / "corrected_embeddings"
SET20 = RESULTS / "location_sets/set20.csv"
COHORT = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17/00_contract/cohort.csv"
FOLDS = PROJECT / "outputs/augmentation_ood_v1/00_contract/slide_folds.csv"
IMAGE_METRICS_SET20 = RESULTS / "set20_reaggregation/image_metrics_slide.csv"
BENCHMARK = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/09_table2_benchmark"
GT450_BENCHMARK = PROJECT / "analysis/paper/results/plism_gt450_extension"

PFMS = ("uni_v1", "uni2", "virchow2", "hoptimus1")
PFM_LABELS = {"uni_v1": "UNI v1", "uni2": "UNI2-h", "virchow2": "Virchow2", "hoptimus1": "H-optimus-1"}
DATASETS = ("pannormal", "plism")
TARGETS = {
    "pannormal": ("versa", "akoya", "gt450", "s360", "s60"),
    "plism": ("gt450", "s360", "s60"),
}
IMAGE_METHODS = ("reinhard", "macenko", "vahadane", "pix2pix", "cyclegan", "frequency", "combined")
FEATURE_METHODS = ("ridge", "combat", "ols")
METHODS = IMAGE_METHODS + FEATURE_METHODS
METHOD_LABELS = {
    "raw": "Raw", "target": "Real target", "reinhard": "Reinhard", "macenko": "Macenko",
    "vahadane": "Vahadane", "pix2pix": "Pix2Pix", "cyclegan": "CycleGAN",
    "frequency": "Frequency", "combined": "Color + frequency", "ridge": "Ridge affine",
    "combat": "ComBat", "ols": "Affine OLS",
}
PANNORMAL_SLIDES = 103
PANNORMAL_LOCATIONS_PER_SLIDE = 20
PLISM_SECTIONS = 13
PLISM_LOCATIONS = 2387
PLISM_CORES = 46
CV_FOLDS = 5
BOOTSTRAPS = 2000
BOOTSTRAP_SEEDS = {"pannormal": 20260929, "plism": 20260930}
CONDITION = re.compile(r"^(?P<method>[a-z0-9]+)_to_(?P<scanner>[a-z0-9]+)(?:_fold(?P<fold>\d+))?$")
ALIASES_METRIC = {
    "ssim": "ssim", "target_ssim": "ssim",
    "lpips": "lpips", "lpips_vgg": "lpips", "lpips_vgg16": "lpips",
    "image_residual": "image_residual", "residual": "image_residual", "rms_residual": "image_residual",
    "coverage": "coverage", "joint_coverage": "joint_coverage",
    "target_gradient_ncc": "target_gradient_ncc", "source_gradient_ncc": "source_gradient_ncc",
    "uni_distance": "uni_distance",
}
HIGHER_IS_BETTER = {
    "ssim": True, "lpips": False, "image_residual": False, "coverage": True,
    "joint_coverage": True, "target_gradient_ncc": True, "source_gradient_ncc": True,
    "uni_distance": False,
}
ALIASES_METHOD = {
    "color_frequency": "combined", "colour_frequency": "combined", "color+frequency": "combined",
    "reinhard_frequency": "combined", "color + frequency": "combined", "combined": "combined",
}


# ----------------------------------------------------------------------------- utilities

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    temporary.replace(path)


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    temporary.replace(path)


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def decode(values) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def parse_condition(name: str) -> dict | None:
    """Return method/scanner/fold for a corrected condition, or None for source/target."""
    if name == "source_at2" or name.startswith("target_"):
        return None
    match = CONDITION.match(name)
    if match is None:
        return None
    fold = match.group("fold")
    return {"method": match.group("method"), "scanner": match.group("scanner"),
            "fold": None if fold is None else int(fold)}


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT))
    except ValueError:
        return str(path)


# ----------------------------------------------------------------------------- contract

def load_cohort() -> pd.DataFrame:
    """Locked 103-slide cohort with its locked folds, sorted by slide id."""
    cohort = pd.read_csv(COHORT, dtype={"slide_id": str})
    folds = pd.read_csv(FOLDS, dtype={"slide_id": str}).set_index("slide_id")["fold"]
    if len(cohort) != PANNORMAL_SLIDES or cohort.slide_id.duplicated().any():
        raise ValueError("locked cohort changed")
    if not (cohort.set_index("slide_id")["fold"] == folds.reindex(cohort.slide_id).to_numpy()).all():
        raise ValueError("cohort folds disagree with the locked fold file")
    return cohort[["slide_id", "tissue_type", "fold"]].sort_values("slide_id").reset_index(drop=True)


def load_set20() -> pd.DataFrame:
    table = pd.read_csv(SET20, dtype={"slide_id": str})
    if len(table) != PANNORMAL_SLIDES * PANNORMAL_LOCATIONS_PER_SLIDE:
        raise ValueError("set20 must hold 2,060 locations")
    return table


def plism_section_folds(sections) -> dict[str, int]:
    """Five cross-validation folds of PLISM sections (sorted sections, index modulo 5)."""
    return {section: index % CV_FOLDS for index, section in enumerate(sorted(sections))}


def manifest_status(root: Path = EMBEDDINGS) -> dict:
    path = root / "manifest.json"
    if not path.exists():
        return {"exists": False, "complete": False}
    payload = json.loads(path.read_text())
    return {"exists": True, "complete": bool(payload.get("complete")), "sha256": sha256(path)}


# ----------------------------------------------------------------------------- embedding set

@dataclass
class EmbeddingSet:
    dataset: str
    pfm: str
    source: str
    locations: pd.DataFrame
    conditions: dict[str, np.ndarray]
    available: tuple[str, ...]
    notes: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frame = self.locations.reset_index(drop=True)
        self.units = tuple(sorted(frame["unit"].unique()))
        unit_code = {name: index for index, name in enumerate(self.units)}
        frame["unit_code"] = frame["unit"].map(unit_code).astype(int)
        if self.dataset == "pannormal":
            frame["query"] = frame["unit"]
        else:
            frame["query"] = frame["unit"].astype(str) + "|" + frame["core"].astype(str)
        queries = frame.drop_duplicates("query")[["query", "unit", "unit_code", "tissue"]]
        queries = queries.sort_values(["unit_code", "query"]).reset_index(drop=True)
        query_code = {name: index for index, name in enumerate(queries["query"])}
        frame["query_code"] = frame["query"].map(query_code).astype(int)
        self.tissues = tuple(sorted(frame["tissue"].unique()))
        tissue_code = {name: index for index, name in enumerate(self.tissues)}
        frame["tissue_code"] = frame["tissue"].map(tissue_code).astype(int)
        queries["tissue_code"] = queries["tissue"].map(tissue_code).astype(int)
        self.locations = frame
        self.queries = queries
        self.scanners = TARGETS[self.dataset]

    @property
    def n(self) -> int:
        return len(self.locations)

    def raw(self) -> np.ndarray:
        return self.conditions["source_at2"]

    def target(self, scanner: str) -> np.ndarray:
        return self.conditions[f"target_{scanner}"]

    def methods(self) -> list[str]:
        """Corrected methods present in the store, known methods first."""
        found = {parsed["method"] for parsed in map(parse_condition, self.available) if parsed}
        return [method for method in METHODS if method in found] + sorted(found - set(METHODS))

    def variant_names(self, method: str, scanner: str) -> list[tuple[str, str]]:
        """(variant label, condition name) pairs; label is 'none' or 'fold<k>'."""
        if method == "raw":
            return [("none", "source_at2")]
        if method == "target":
            return [("none", f"target_{scanner}")]
        plain = f"{method}_to_{scanner}"
        if plain in self.available:
            return [("none", plain)]
        folds = []
        for name in self.available:
            parsed = parse_condition(name)
            if parsed and parsed["method"] == method and parsed["scanner"] == scanner and parsed["fold"] is not None:
                folds.append((parsed["fold"], name))
        return [(f"fold{fold}", name) for fold, name in sorted(folds)]

    def variants(self, method: str, scanner: str) -> list[tuple[str, np.ndarray]]:
        return [(label, self.conditions[name]) for label, name in self.variant_names(method, scanner)
                if name in self.conditions]

    def unit_means(self, values: np.ndarray) -> np.ndarray:
        """PanNormal: location -> slide mean. PLISM: location -> core -> section mean."""
        query_mean = self.query_means(values)
        codes = self.queries["unit_code"].to_numpy()
        return np.bincount(codes, weights=query_mean, minlength=len(self.units)) / np.bincount(
            codes, minlength=len(self.units))

    def query_means(self, values: np.ndarray) -> np.ndarray:
        codes = self.locations["query_code"].to_numpy()
        return np.bincount(codes, weights=np.asarray(values, dtype=np.float64),
                           minlength=len(self.queries)) / np.bincount(codes, minlength=len(self.queries))

    def query_embedding_means(self, x: np.ndarray) -> np.ndarray:
        """Unit-normalized mean of the stored vectors for every query (slide / section x core)."""
        codes = self.locations["query_code"].to_numpy()
        total = np.zeros((len(self.queries), x.shape[1]), dtype=np.float64)
        np.add.at(total, codes, np.asarray(x, dtype=np.float64))
        return unit(total / np.bincount(codes, minlength=len(self.queries))[:, None])


def _wanted_conditions(available: list[str], dataset: str, methods) -> list[str]:
    wanted = ["source_at2"] + [f"target_{scanner}" for scanner in TARGETS[dataset]]
    for name in available:
        parsed = parse_condition(name)
        if parsed and parsed["scanner"] in TARGETS[dataset] and (methods is None or parsed["method"] in methods):
            wanted.append(name)
    return [name for name in dict.fromkeys(wanted) if name in available]


def _check_features(features: np.ndarray, path: Path) -> None:
    if not np.isfinite(features).all():
        raise ValueError(f"non-finite embeddings in {path}")


def load(dataset: str, pfm: str, methods=None, source: str = "rv03", root: Path = EMBEDDINGS,
         require_complete: bool = True) -> EmbeddingSet:
    """Load one dataset x PFM. ``methods`` restricts corrected conditions (None = all)."""
    if source == "smoke":
        from smoke_embedding_adapter import load_smoke

        return load_smoke(dataset, pfm, methods)
    if dataset == "pannormal":
        return _load_pannormal(pfm, methods, root, require_complete)
    if dataset == "plism":
        return _load_plism(pfm, methods, root, require_complete)
    raise ValueError(dataset)


def _load_pannormal(pfm: str, methods, root: Path, require_complete: bool) -> EmbeddingSet:
    cohort = load_cohort()
    set20 = load_set20()
    folder = root / "pannormal" / pfm
    paths = {row.slide_id: folder / f"{row.slide_id}.h5" for row in cohort.itertuples()}
    present = [slide for slide, path in paths.items() if path.exists()]
    if require_complete and len(present) != PANNORMAL_SLIDES:
        raise FileNotFoundError(f"{folder}: {PANNORMAL_SLIDES - len(present)} slide shards missing")
    if not present:
        raise FileNotFoundError(folder)
    names_by_slide = {}
    for slide in present:
        with h5py.File(paths[slide], "r") as store:
            names_by_slide[slide] = decode(store["condition_names"][:])
    common = set.intersection(*(set(names) for names in names_by_slide.values()))
    union = set.union(*(set(names) for names in names_by_slide.values()))
    first = names_by_slide[present[0]]
    available = [name for name in first if name in common]
    notes = [f"condition {name} present in only some slides; dropped" for name in sorted(union - common)]
    wanted = _wanted_conditions(available, "pannormal", methods)
    blocks = {name: [] for name in wanted}
    rows = []
    files = {}
    dim = None
    for slide in present:
        path = paths[slide]
        meta = cohort.loc[cohort.slide_id == slide].iloc[0]
        with h5py.File(path, "r") as store:
            names = names_by_slide[slide]
            features = np.asarray(store["features"][:], dtype=np.float32)
            locations = np.asarray(store["location_index"][:], dtype=np.int64)
            sources = (np.asarray(store["source_index"][:], dtype=np.int64)
                       if "source_index" in store else np.full(len(locations), -1))
            attrs = dict(store.attrs)
        expected = set20.loc[set20.slide_id == slide].sort_values("location_index")
        if features.shape[:2] != (PANNORMAL_LOCATIONS_PER_SLIDE, len(names)):
            raise ValueError(f"{path}: features {features.shape} vs {len(names)} conditions")
        if sorted(locations.tolist()) != expected.location_index.tolist():
            raise ValueError(f"{path}: locations differ from set20")
        for key, value in (("slide_id", slide), ("tissue_type", meta.tissue_type), ("fold", meta.fold)):
            if key in attrs and str(attrs[key]) != str(value):
                raise ValueError(f"{path}: attribute {key}={attrs[key]!r}, expected {value!r}")
        if "pfm" in attrs and str(attrs["pfm"]) != pfm:
            raise ValueError(f"{path}: pfm attribute {attrs['pfm']!r}")
        if dim is None:
            dim = features.shape[2]
        elif features.shape[2] != dim:
            raise ValueError(f"{path}: feature dimension changed")
        _check_features(features, path)
        order = np.argsort(locations)
        lookup = {name: index for index, name in enumerate(names)}
        for name in wanted:
            blocks[name].append(features[order, lookup[name], :])
        expected_source = expected.set_index("location_index").source_index
        for offset in order:
            location = int(locations[offset])
            if sources[offset] >= 0 and int(sources[offset]) != int(expected_source[location]):
                raise ValueError(f"{path}: source_index differs from set20 at {location}")
            rows.append({"unit": slide, "tissue": str(meta.tissue_type), "fold": int(meta.fold),
                         "location_id": location, "core": "", "source_index": int(expected_source[location])})
        files[slide] = rel(path)
    conditions = {name: np.concatenate(block) for name, block in blocks.items()}
    return EmbeddingSet("pannormal", pfm, "rv03", pd.DataFrame(rows), conditions,
                        tuple(available), notes, files)


def _dataset_first(store: h5py.File, names: tuple[str, ...]):
    for name in names:
        if name in store:
            return np.asarray(store[name][:])
    return None


def _load_plism(pfm: str, methods, root: Path, require_complete: bool) -> EmbeddingSet:
    folder = root / "plism" / pfm
    paths = sorted(folder.glob("*.h5"))
    if require_complete and len(paths) != PLISM_SECTIONS:
        raise FileNotFoundError(f"{folder}: expected {PLISM_SECTIONS} section shards, found {len(paths)}")
    if not paths:
        raise FileNotFoundError(folder)
    names_by_section = {}
    for path in paths:
        with h5py.File(path, "r") as store:
            names_by_section[path.stem] = decode(store["condition_names"][:])
    common = set.intersection(*(set(names) for names in names_by_section.values()))
    union = set.union(*(set(names) for names in names_by_section.values()))
    available = [name for name in names_by_section[paths[0].stem] if name in common]
    notes = [f"condition {name} present in only some sections; dropped" for name in sorted(union - common)]
    wanted = _wanted_conditions(available, "plism", methods)
    folds = plism_section_folds([path.stem for path in paths])
    blocks = {name: [] for name in wanted}
    rows = []
    files = {}
    dim = None
    for path in paths:
        section = path.stem
        with h5py.File(path, "r") as store:
            names = names_by_section[section]
            features = np.asarray(store["features"][:], dtype=np.float32)
            core = _dataset_first(store, ("core", "core_id"))
            location = _dataset_first(store, ("location", "location_id", "location_index"))
            tissue = _dataset_first(store, ("tissue_type", "tissue"))
            attrs = dict(store.attrs)
        if core is None or location is None:
            raise ValueError(f"{path}: missing core or location keys")
        core = np.asarray(decode(core)) if core.dtype.kind in "SOU" else core.astype(int).astype(str)
        tissue = np.asarray(decode(tissue)) if tissue is not None else core
        if features.shape[:2] != (len(core), len(names)):
            raise ValueError(f"{path}: features {features.shape} vs keys")
        if "pfm" in attrs and str(attrs["pfm"]) != pfm:
            raise ValueError(f"{path}: pfm attribute {attrs['pfm']!r}")
        if dim is None:
            dim = features.shape[2]
        elif features.shape[2] != dim:
            raise ValueError(f"{path}: feature dimension changed")
        _check_features(features, path)
        order = np.lexsort((location.astype(np.int64), core.astype(str)))
        lookup = {name: index for index, name in enumerate(names)}
        for name in wanted:
            blocks[name].append(features[order, lookup[name], :])
        for offset in order:
            rows.append({"unit": section, "tissue": str(tissue[offset]), "fold": folds[section],
                         "location_id": int(location[offset]), "core": str(core[offset]), "source_index": -1})
        files[section] = rel(path)
    locations = pd.DataFrame(rows)
    if require_complete:
        if len(locations) != PLISM_LOCATIONS:
            raise ValueError(f"PLISM: {len(locations)} locations, expected {PLISM_LOCATIONS}")
        if locations.groupby("core").tissue.nunique().max() != 1:
            raise ValueError("PLISM core maps to more than one tissue")
    conditions = {name: np.concatenate(block) for name, block in blocks.items()}
    return EmbeddingSet("plism", pfm, "rv03", locations, conditions, tuple(available), notes, files)


def discover(dataset: str, pfm: str, source: str = "rv03", root: Path = EMBEDDINGS) -> list[str]:
    """Condition names available for one dataset x PFM without loading features."""
    if source == "smoke":
        from smoke_embedding_adapter import smoke_conditions

        return smoke_conditions(dataset, pfm)
    folder = root / dataset / pfm
    paths = sorted(folder.glob("*.h5"))
    if not paths:
        return []
    names = None
    for path in paths:
        with h5py.File(path, "r") as store:
            current = set(decode(store["condition_names"][:]))
        names = current if names is None else names & current
    return sorted(names)


# ----------------------------------------------------------------------------- bootstrap

def bootstrap_weights(dataset: str, n_units: int, replicates: int = BOOTSTRAPS) -> np.ndarray:
    """Row 0 is the point estimate (all ones); rows 1.. are multinomial resample counts."""
    rng = np.random.default_rng(BOOTSTRAP_SEEDS[dataset])
    draws = rng.integers(0, n_units, size=(replicates, n_units))
    counts = np.zeros((replicates, n_units), dtype=np.float64)
    np.add.at(counts, (np.repeat(np.arange(replicates), n_units), draws.ravel()), 1.0)
    return np.vstack([np.ones((1, n_units)), counts])


def weighted_mean(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Weighted mean over units for every resample; NaN units are ignored."""
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    total = weights @ np.where(mask, values, 0.0)
    count = weights @ mask.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / count


def weighted_macro(weights_query: np.ndarray, values: np.ndarray, tissue_code: np.ndarray,
                   n_tissues: int) -> np.ndarray:
    """Macro average over tissues of weighted query means (queries weighted by their unit)."""
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    indicator = np.zeros((len(values), n_tissues))
    indicator[np.arange(len(values)), tissue_code] = 1.0
    indicator *= mask[:, None]
    total = weights_query @ (np.where(mask, values, 0.0)[:, None] * indicator)
    count = weights_query @ indicator
    with np.errstate(invalid="ignore", divide="ignore"):
        per_tissue = total / count
    return np.nanmean(per_tissue, axis=1)


def summarize(draws: np.ndarray) -> dict:
    """Point estimate (row 0) and percentile 95% CI of rows 1..."""
    draws = np.asarray(draws, dtype=np.float64)
    finite = draws[1:][np.isfinite(draws[1:])]
    if not np.isfinite(draws[0]) or len(finite) == 0:
        return {"estimate": float(draws[0]), "ci_low": np.nan, "ci_high": np.nan}
    return {"estimate": float(draws[0]), "ci_low": float(np.quantile(finite, 0.025)),
            "ci_high": float(np.quantile(finite, 0.975))}


def spearman_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation for each row pair of x and y ([R, n]); NaN columns dropped per row."""
    out = np.full(x.shape[0], np.nan)
    for row in range(x.shape[0]):
        mask = np.isfinite(x[row]) & np.isfinite(y[row])
        if mask.sum() >= 3 and np.ptp(x[row, mask]) > 0 and np.ptp(y[row, mask]) > 0:
            out[row] = stats.spearmanr(x[row, mask], y[row, mask]).statistic
    return out


# ----------------------------------------------------------------------------- image fidelity

def _canonical_method(value: str) -> str:
    value = str(value).strip().lower()
    return ALIASES_METHOD.get(value, value)


def _canonical_metric(value: str) -> str:
    value = str(value).strip().lower()
    return ALIASES_METRIC.get(value, value)


def load_image_metrics(dataset: str, source: str = "rv03") -> tuple[pd.DataFrame | None, str]:
    """Unit-level image fidelity: columns unit, scanner, method, metric, value.

    PanNormal comes from RV-P0b (``set20_reaggregation/image_metrics_slide.csv``). PLISM
    comes from the existing benchmark location metrics (location -> core -> section,
    PanNormal-fitted folds averaged), exactly as ``analysis/paper/table2_benchmark.py``.
    In smoke mode only, PanNormal falls back to the benchmark's set20 SSIM/LPIPS/NCC
    location metrics so that the join can be exercised before RV-P0b exists.
    """
    if dataset == "plism":
        return _plism_benchmark_metrics(), "existing PLISM benchmark location metrics (09_table2_benchmark + plism_gt450_extension)"
    if IMAGE_METRICS_SET20.exists():
        table = pd.read_csv(IMAGE_METRICS_SET20, dtype={"slide_id": str})
        table = table.loc[table["dataset"].astype(str).str.lower().eq("pannormal")].copy()
        table["unit"] = table["slide_id"].astype(str)
        table["scanner"] = table["scanner"].astype(str).str.lower()
        table["method"] = table["method"].map(_canonical_method)
        table["metric"] = table["metric"].map(_canonical_metric)
        table = table.groupby(["unit", "scanner", "method", "metric"], as_index=False)["value"].mean()
        return table, f"RV-P0b {rel(IMAGE_METRICS_SET20)} (sha256 {sha256(IMAGE_METRICS_SET20)[:12]})"
    if source == "smoke":
        return _pannormal_benchmark_metrics(), "SMOKE STAND-IN: 09_table2_benchmark PanNormal set20 location metrics"
    return None, f"pending: {rel(IMAGE_METRICS_SET20)} not found"


def _pannormal_benchmark_metrics() -> pd.DataFrame:
    frames = [pd.read_csv(path, dtype={"slide_id": str}) for path in sorted((BENCHMARK / "pannormal/shards").glob("*.csv.gz"))]
    data = pd.concat(frames, ignore_index=True)
    set20 = load_set20()[["slide_id", "location_index"]]
    data = data.merge(set20, on=["slide_id", "location_index"], how="inner")
    metrics = {"target_ssim": "ssim", "lpips_vgg": "lpips", "target_gradient_ncc": "target_gradient_ncc",
               "source_gradient_ncc": "source_gradient_ncc"}
    data["scanner"] = data["scanner"].str.lower()
    slide = data.groupby(["slide_id", "scanner", "method"], as_index=False)[list(metrics)].mean()
    long = slide.melt(id_vars=["slide_id", "scanner", "method"], value_vars=list(metrics), var_name="metric")
    long["metric"] = long["metric"].map(metrics)
    return long.rename(columns={"slide_id": "unit"})


def _plism_benchmark_metrics() -> pd.DataFrame:
    parts = []
    for root, conventional, learned in (
        (BENCHMARK, 26, 20), (GT450_BENCHMARK, 13, 10),
    ):
        for kind, expected in (("plism_conventional", conventional), ("plism_learned", learned)):
            paths = sorted((root / kind / "shards").glob("*.csv.gz"))
            if len(paths) != expected:
                raise FileNotFoundError(f"{root / kind}: expected {expected} shards, found {len(paths)}")
            parts.extend(pd.read_csv(path) for path in paths)
    data = pd.concat(parts, ignore_index=True)
    metrics = {"target_ssim": "ssim", "lpips_vgg": "lpips", "target_gradient_ncc": "target_gradient_ncc",
               "source_gradient_ncc": "source_gradient_ncc"}
    data["scanner"] = data["scanner"].str.lower()
    data["section"] = data["section"].astype(str)
    keys = ["section", "core", "scanner", "method", "fold"]
    core = data.groupby(keys, as_index=False)[list(metrics)].mean()
    section = core.groupby(["section", "scanner", "method", "fold"], as_index=False)[list(metrics)].mean()
    section = section.groupby(["section", "scanner", "method"], as_index=False)[list(metrics)].mean()
    long = section.melt(id_vars=["section", "scanner", "method"], value_vars=list(metrics), var_name="metric")
    long["metric"] = long["metric"].map(metrics)
    return long.rename(columns={"section": "unit"})


def image_metric_arrays(table: pd.DataFrame | None, units: tuple[str, ...]) -> dict:
    """{(method, scanner, metric): unit-level values aligned to ``units`` (NaN if absent)}."""
    if table is None:
        return {}
    out = {}
    index = pd.Index(units)
    for (method, scanner, metric), group in table.groupby(["method", "scanner", "metric"]):
        values = group.set_index("unit")["value"].reindex(index).to_numpy(dtype=np.float64)
        out[(method, scanner, metric)] = values
    return out


def image_improvement(arrays: dict, method: str, scanner: str, metric: str) -> np.ndarray | None:
    """Unit-level improvement over raw, signed so that positive means closer to the target."""
    corrected = arrays.get((method, scanner, metric))
    raw = arrays.get(("raw", scanner, metric))
    if corrected is None or raw is None:
        return None
    return corrected - raw if HIGHER_IS_BETTER.get(metric, True) else raw - corrected
