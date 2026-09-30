#!/usr/bin/env python3
"""RV03 shared definitions: paths, condition names and small I/O helpers.

Imported by corrected_embeddings_images.py, corrected_embeddings_features.py,
corrected_embeddings_assemble.py and corrected_embeddings_finalize.py. It reads
only locked inputs and never writes outside ``results/corrected_embeddings``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
for _path in (PROJECT, PROJECT / "src", PROJECT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

OUTPUT = Path(__file__).resolve().parent / "results/corrected_embeddings"
PARTS = OUTPUT / "parts"
QC = OUTPUT / "qc"

STUDY = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17"
COMPLETION = STUDY / "12_manuscript_completion"
PANEL_A = COMPLETION / "02_baseline_benchmark"
LEARNED = STUDY / "06_learned_baselines"
COHORT = STUDY / "00_contract/cohort.csv"
FOLDS = PROJECT / "outputs/augmentation_ood_v1/00_contract/slide_folds.csv"
SET40 = Path(__file__).resolve().parent / "results/location_sets/set40.csv"
SET20 = Path(__file__).resolve().parent / "results/location_sets/set20.csv"
PARAMETERS = STUDY / "02_correction/parameters.json"
STAIN_REFERENCES = PANEL_A / "01_stain_v4_mu_convergence/references"
VAHADANE_SELECTION = PANEL_A / "00_contract/vahadane_selection_manifest_v4.json"
GAN_PAIRS = PROJECT / "outputs/gan_encoder_review_2026-09-25/prediction_pairs.csv"
HF_CACHE = PROJECT / "outputs/encoder_review_2026-09-25/hf_cache"
CROSS_RAW = PROJECT / "outputs/feature_crossencoder_review_2026-09-25/raw"
UNI_SHARDS = STUDY / "03_uni/shards"

PLISM_LEARNED = COMPLETION / "06_plism_learned_external"
PLISM_SELECTED = PLISM_LEARNED / "pix2pix/00_contract/selected_locations.csv"
PLISM_RENDERS = PLISM_LEARNED / "pix2pix/00_contract/read_only_renders.csv"
PLISM_TISSUE = PROJECT / "outputs/plism_factorial_external_v1/00_contract/selected_locations.csv"
PLISM_OLD_RENDER = STUDY / "08_plism_external/pix2pix_gt450_to_at2/01_rendered_inputs"
PLISM_NEW_RENDER = COMPLETION / "05_plism_external_correction/01_rendered_inputs"
PLISM_UNI_RAW = COMPLETION / "05_plism_external_correction/04_external_uni/features"
PLISM_GAN_EXTERNAL = {
    ("pix2pix", "gt450"): LEARNED / "09_bidirectional_full_training/10_plism_external_bidirectional",
    ("cyclegan", "gt450"): LEARNED / "11_cyclegan_full_training/10_plism_external_bidirectional",
    ("pix2pix", "s360"): PLISM_LEARNED / "pix2pix",
    ("pix2pix", "s60"): PLISM_LEARNED / "pix2pix",
    ("cyclegan", "s360"): PLISM_LEARNED / "cyclegan",
    ("cyclegan", "s60"): PLISM_LEARNED / "cyclegan",
}

PFMS = ("uni_v1", "uni2", "virchow2", "hoptimus1")
PFM_DIMS = {"uni_v1": 1024, "uni2": 1536, "virchow2": 2560, "hoptimus1": 1536}
SCANNERS = ("at2", "versa", "akoya", "gt450", "s360", "s60")
PAN_TARGETS = SCANNERS[1:]
PLISM_SCANNERS = ("at2", "gt450", "s360", "s60")
PLISM_TARGETS = PLISM_SCANNERS[1:]
FOLD_IDS = tuple(range(5))
CONVENTIONAL = ("reinhard", "frequency", "combined", "macenko", "vahadane")
GAN_METHODS = ("pix2pix", "cyclegan")
IMAGE_METHODS = CONVENTIONAL + GAN_METHODS
FEATURE_METHODS = ("ridge", "combat", "ols")
METHODS = IMAGE_METHODS + FEATURE_METHODS
RIDGE_ALPHAS = (0.01, 0.1, 1.0)
PLISM_LOCATIONS = 2387


def pannormal_conditions(methods: tuple[str, ...] = METHODS) -> list[str]:
    names = ["source_at2"] + [f"target_{scanner}" for scanner in PAN_TARGETS]
    names += [f"{method}_to_{scanner}" for scanner in PAN_TARGETS for method in methods]
    return names


def plism_conditions(methods: tuple[str, ...] = METHODS) -> list[str]:
    names = ["source_at2"] + [f"target_{scanner}" for scanner in PLISM_TARGETS]
    names += [f"{method}_to_{scanner}_fold{fold}"
              for scanner in PLISM_TARGETS for method in methods for fold in FOLD_IDS]
    return names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in np.asarray(values)]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n")
    temporary.replace(path)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    temporary.replace(path)


def write_h5(path: Path, datasets: dict, attrs: dict) -> None:
    """Atomically write numeric and utf-8 string datasets plus attributes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with h5py.File(temporary, "w") as store:
        for name, value in datasets.items():
            value = np.asarray(value)
            if value.dtype.kind in {"U", "O", "S"}:
                store.create_dataset(name, data=np.asarray([str(v) for v in value.ravel()],
                                                           dtype=object).reshape(value.shape),
                                     dtype=h5py.string_dtype("utf-8"))
            elif value.ndim >= 2:
                store.create_dataset(name, data=value, compression="lzf")
            else:
                store.create_dataset(name, data=value)
        for key, value in attrs.items():
            store.attrs[key] = value
    temporary.replace(path)


def load_cohort() -> pd.DataFrame:
    cohort = pd.read_csv(COHORT, dtype={"slide_id": str})
    folds = pd.read_csv(FOLDS, dtype={"slide_id": str}).set_index("slide_id")["fold"]
    if len(cohort) != 103 or not (cohort.set_index("slide_id")["fold"] == folds).all():
        raise ValueError("locked cohort/fold contract changed")
    return cohort.sort_values("slide_id", kind="stable").reset_index(drop=True)


def load_set(path: Path, per_slide: int) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"slide_id": str})
    frame = frame.sort_values(["slide_id", "location_index"], kind="stable").reset_index(drop=True)
    if frame.slide_id.nunique() != 103 or not frame.groupby("slide_id").size().eq(per_slide).all():
        raise ValueError(f"{path.name}: expected {per_slide} locations for each of 103 slides")
    return frame


def plism_locations() -> pd.DataFrame:
    """The locked 2,387 PLISM locations with tissue labels, sorted by section and location."""
    selected = pd.read_csv(PLISM_SELECTED)
    tissue = pd.read_csv(PLISM_TISSUE)
    merged = selected.merge(tissue[["stain", "location", "core", "tissue_type", "pannormal_tissue"]],
                            on=["stain", "location"], how="left", suffixes=("", "_factorial"),
                            validate="one_to_one")
    if (len(merged) != PLISM_LOCATIONS or merged.tissue_type_factorial.isna().any()
            or not (merged.core == merged.core_factorial).all()
            or not (merged.tissue_type == merged.tissue_type_factorial).all()):
        raise ValueError("PLISM benchmark and factorial location files disagree")
    merged["section"] = merged["stain"].astype(str)
    merged["pannormal_tissue"] = merged["pannormal_tissue_factorial"].fillna("")
    merged = merged.sort_values(["section", "location"], kind="stable").reset_index(drop=True)
    return merged[["section", "location", "core", "tissue_type", "pannormal_tissue", "replicate"]]


def plism_sections() -> list[str]:
    return sorted(pd.read_csv(PLISM_SELECTED).stain.astype(str).unique())


def plism_render_path(scanner: str, section: str) -> Path:
    root = PLISM_OLD_RENDER if scanner in {"at2", "gt450"} else PLISM_NEW_RENDER
    return root / scanner.upper() / f"{section}.h5"


def plism_gan_candidate(method: str, scanner: str, fold: int) -> pd.Series:
    root = PLISM_GAN_EXTERNAL[(method, scanner)]
    frame = pd.read_csv(root / "00_contract/checkpoint_candidates.csv")
    row = frame.loc[frame.direction.eq(f"at2_to_{scanner}") & frame.fold.astype(int).eq(fold)]
    if len(row) != 1:
        raise ValueError(f"expected one frozen {method} candidate for at2_to_{scanner} fold {fold}")
    row = row.iloc[0]
    if str(row.source_scanner).lower() != "at2" or str(row.target_scanner).lower() != scanner:
        raise ValueError(f"unexpected {method} candidate direction for {scanner}")
    return row


def plism_gan_feature_path(method: str, scanner: str, fold: int, section: str) -> Path:
    root = PLISM_GAN_EXTERNAL[(method, scanner)]
    return root / f"01_model_evaluations/at2_to_{scanner}/fold_{fold}/feature_shards/{section}.h5"


def pannormal_gan_uni_path(method: str, scanner: str, slide_id: str) -> Path:
    if scanner == "gt450":
        sub = "09_bidirectional_full_training" if method == "pix2pix" else "11_cyclegan_full_training"
        return LEARNED / sub / "04_uni/at2_to_gt450/feature_shards" / f"{slide_id}.h5"
    sub = "03_pix2pix" if method == "pix2pix" else "04_cyclegan"
    return PANEL_A / sub / f"04_uni/at2_to_{scanner}/feature_shards" / f"{slide_id}.h5"


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(unit(a) * unit(b), axis=-1)
