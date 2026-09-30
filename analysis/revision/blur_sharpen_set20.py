#!/usr/bin/env python3
"""RV01: blur/sharpen trajectories, scanner convergence and tissue loss on `set20`.

Aggregates `blur_sharpen_set20_features.py` shards (UNI v1, six scanners, 20 locations per
slide). Endpoint definitions follow the paper code (`analysis/paper/
compare_uni_spaces_blur_sharp.py`, `aggregate_augmentation_strong_blur.py`) and the
Supplementary Methods; each is computed three ways: from the paper's stored shards (checks
that this implementation reproduces the published values), from the re-run on the paper's
three locations (`paper_blur_probe`), and on all 20 locations (the RV01 result).

Primary endpoints: median cosine of displacement directions across 15 scanner pairs (blur
sigma = 3, sharpen alpha = 2); median interscanner distance ratio at sigma = 6; median retained
gradient energy at sigma = 6; same-tissue nearest-neighbour proportion at raw and sigma = 6.
Secondary: six-way scanner linear probe on sigma = 6 location embeddings with the locked
slide folds as cross-validation groups, raw as reference.
"""

from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from manuscript_completion.frequency_aggregate import _select_c  # noqa: E402


COHORT = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17/00_contract/cohort.csv"
FOLDS = PROJECT / "outputs/augmentation_ood_v1/00_contract/slide_folds.csv"
SET20 = PROJECT / "analysis/revision/results/location_sets/set20.csv"
PAPER_COMMON = PROJECT / "analysis/paper/results/augmentation_common_uni_space"
PAPER_STRONG = PROJECT / "analysis/paper/results/augmentation_strong_blur"
OUTPUT = Path(__file__).resolve().parent / "results/blur_sharpen_set20"
SCANNERS = ("at2", "versa", "akoya", "gt450", "s360", "s60")
BLUR_SIGMAS = (0.0, 0.5, 1.0, 2.0, 3.0, 6.0)
SHARPEN_ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)
S3, S6, A2 = BLUR_SIGMAS.index(3.0), BLUR_SIGMAS.index(6.0), SHARPEN_ALPHAS.index(2.0)
PUBLISHED = {
    "direction_cosine_blur_sigma3": 0.78,
    "direction_cosine_sharpen_alpha2": 0.69,
    "distance_ratio_sigma6": 0.543,
    "gradient_energy_retained_pct_sigma6": 0.95,
    "same_tissue_nn_pct_raw": 53.4,
    "same_tissue_nn_pct_sigma6": 36.4,
}
N_BOOT = 2000
SEED = 20260929
COLORS = {"at2": "#525252", "versa": "#7552a3", "akoya": "#ce4470",
          "gt450": "#2086b0", "s360": "#e19a23", "s60": "#23896c"}


# ----------------------------------------------------------------------------- loading
def load_rerun(cohort: pd.DataFrame, set20: pd.DataFrame) -> dict:
    parts = {key: [] for key in ("raw", "recomputed", "blur", "sharpen", "blur_detail", "sharpen_detail", "probe")}
    for slide_id in cohort.slide_id:
        path = OUTPUT / "shards" / f"{slide_id}.h5"
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as store:
            locations = np.asarray(store["location_index"], dtype=int)
            expected = set20[set20.slide_id.eq(slide_id)].sort_values("location_index")
            if not np.array_equal(locations, expected.location_index.to_numpy(int)):
                raise ValueError(f"{slide_id}: shard locations differ from set20")
            if tuple(v.decode() for v in store["scanner_names"][:]) != SCANNERS:
                raise ValueError(f"{slide_id}: scanner order changed")
            probe = np.asarray(store["paper_blur_probe"], dtype=bool)
            if not np.array_equal(probe, expected.paper_blur_probe.to_numpy(bool)) or probe.sum() != 3:
                raise ValueError(f"{slide_id}: paper probe flags inconsistent")
            parts["raw"].append(np.asarray(store["raw_features"], dtype=np.float32))
            parts["recomputed"].append(np.asarray(store["raw_recomputed"], dtype=np.float32))
            parts["blur"].append(np.asarray(store["blur_features"], dtype=np.float32))
            parts["sharpen"].append(np.asarray(store["sharpen_features"], dtype=np.float32))
            parts["blur_detail"].append(np.asarray(store["blur_detail"], dtype=np.float32))
            parts["sharpen_detail"].append(np.asarray(store["sharpen_detail"], dtype=np.float32))
            parts["probe"].append(probe)
    data = {key: np.stack(value) for key, value in parts.items()}
    shapes = {"raw": (103, 20, 6, 1024), "blur": (103, 20, 6, 6, 1024), "sharpen": (103, 20, 6, 5, 1024),
              "blur_detail": (103, 20, 6, 6, 2), "sharpen_detail": (103, 20, 6, 5, 2)}
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"{key}: shape {data[key].shape} or nonfinite values")
    if not (np.array_equal(data["blur"][:, :, :, 0], data["raw"])
            and np.array_equal(data["sharpen"][:, :, :, 0], data["raw"])):
        raise ValueError("untransformed state differs from stored raw features")
    return data


def load_paper(cohort: pd.DataFrame) -> dict:
    """Paper shards rearranged into the re-run layout (3 locations per slide)."""
    raw, blur, sharpen, detail, locations = [], [], [], [], []
    for slide_id in cohort.slide_id:
        with h5py.File(PAPER_COMMON / "shards" / f"{slide_id}.h5", "r") as store:
            common_raw = np.asarray(store["raw_features"], dtype=np.float32)
            augmented = np.asarray(store["augmented_features"], dtype=np.float32)
            common_locations = np.asarray(store["location_index"], dtype=int)
        with h5py.File(PAPER_STRONG / "shards" / f"{slide_id}.h5", "r") as store:
            strong = np.asarray(store["features"], dtype=np.float32)
            strong_detail = np.asarray(store["detail_metrics"], dtype=np.float32)
            sigmas = tuple(float(x) for x in store["sigmas"][:])
            if not np.array_equal(np.asarray(store["location_index"], dtype=int), common_locations):
                raise ValueError(f"{slide_id}: paper shards disagree on locations")
        if sigmas[:3] != (0.0, 3.0, 6.0):
            raise ValueError("paper strong-blur sigma order changed")
        raw.append(common_raw)
        blur.append(np.concatenate([augmented[:, :, 0], strong[:, :, 2:3]], axis=2))
        sharpen.append(augmented[:, :, 1])
        # Only sigma = 3 and 6 have stored gradient metrics in the paper shards.
        values = np.full((3, 6, 6, 2), np.nan, dtype=np.float32)
        values[:, :, 0] = 1.0
        values[:, :, S3] = strong_detail[:, :, 1]
        values[:, :, S6] = strong_detail[:, :, 2]
        detail.append(values)
        locations.append(common_locations)
    return {"raw": np.stack(raw), "blur": np.stack(blur), "sharpen": np.stack(sharpen),
            "blur_detail": np.stack(detail), "locations": np.stack(locations)}


# ----------------------------------------------------------------------------- endpoints
def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def pair_cosines(delta: np.ndarray) -> np.ndarray:
    """delta: (6, D) scanner-specific mean displacement -> 15 pairwise cosines."""
    d = unit(delta)
    return (d @ d.T)[np.triu_indices(6, k=1)]


def paired_distances(features: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slide-level mean cosine distance for every scanner pair: (slides, 15) raw and changed."""
    base, changed = [], []
    for i, j in combinations(range(6), 2):
        base.append((1 - np.einsum("sld,sld->sl", raw[:, :, i], raw[:, :, j])).mean(axis=1))
        changed.append((1 - np.einsum("sld,sld->sl", features[:, :, i], features[:, :, j])).mean(axis=1))
    return np.stack(base, axis=1), np.stack(changed, axis=1)


def retrieval_correct(features: np.ndarray, tissue: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """features (slides, loc, D) -> per-slide top-1 same-tissue indicator and evaluable mask."""
    vectors = unit(features.mean(axis=1))
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = similarity.argmax(axis=1)
    counts = pd.Series(tissue).value_counts()
    evaluable = np.asarray([counts[value] > 1 for value in tissue], dtype=bool)
    return (tissue[nearest] == tissue).astype(float), evaluable


def compute_endpoints(data: dict, tissue: np.ndarray, label: str) -> tuple[dict, dict]:
    raw, blur, sharpen, detail = data["raw"], data["blur"], data["sharpen"], data["blur_detail"]
    tables = {"direction": [], "gaps": [], "gradient": [], "retrieval": []}
    for operation, values, strengths in (("Gaussian blur", blur, BLUR_SIGMAS), ("Unsharp mask", sharpen, SHARPEN_ALPHAS)):
        for step in range(1, len(strengths)):
            cos = pair_cosines((values[:, :, :, step] - raw).mean(axis=(0, 1)))
            tables["direction"].append({"set": label, "operation": operation, "strength": strengths[step],
                                        "median_direction_cosine": float(np.median(cos)),
                                        "min_direction_cosine": float(cos.min()),
                                        "max_direction_cosine": float(cos.max())})
            base, changed = paired_distances(values[:, :, :, step], raw)
            ratio = changed.mean(axis=0) / base.mean(axis=0)
            for pair_index, (i, j) in enumerate(combinations(range(6), 2)):
                tables["gaps"].append({"set": label, "operation": operation, "strength": strengths[step],
                                       "scanner_a": SCANNERS[i], "scanner_b": SCANNERS[j],
                                       "raw_gap_mean": float(base[:, pair_index].mean()),
                                       "changed_gap_mean": float(changed[:, pair_index].mean()),
                                       "mean_gap_ratio": float(ratio[pair_index])})
    for step, sigma in enumerate(BLUR_SIGMAS):
        slide_scanner = detail[:, :, :, step].mean(axis=1)  # (slides, 6, 2)
        if np.isnan(slide_scanner).any():
            continue
        tables["gradient"].append({"set": label, "sigma": sigma,
                                   "median_gradient_energy_retained_pct": float(100 * np.median(slide_scanner[..., 0])),
                                   "median_source_gradient_ncc": float(np.median(slide_scanner[..., 1]))})
        for scanner_index, scanner in enumerate(SCANNERS):
            correct, evaluable = retrieval_correct(blur[:, :, scanner_index, step], tissue)
            tables["retrieval"].append({"set": label, "scanner": scanner, "sigma": sigma,
                                        "evaluable_slides": int(evaluable.sum()),
                                        "top1_same_tissue_rate": float(correct[evaluable].mean())})
    frames = {key: pd.DataFrame(value) for key, value in tables.items()}
    direction, gaps, gradient, retrieval = (frames[k] for k in ("direction", "gaps", "gradient", "retrieval"))

    def pick(frame, **query):
        mask = np.ones(len(frame), dtype=bool)
        for key, value in query.items():
            mask &= frame[key].to_numpy() == value
        return frame[mask]

    endpoints = {
        "direction_cosine_blur_sigma3": float(pick(direction, operation="Gaussian blur", strength=3.0).median_direction_cosine.iloc[0]),
        "direction_cosine_sharpen_alpha2": float(pick(direction, operation="Unsharp mask", strength=2.0).median_direction_cosine.iloc[0]),
        "distance_ratio_sigma6": float(pick(gaps, operation="Gaussian blur", strength=6.0).mean_gap_ratio.median()),
        "gradient_energy_retained_pct_sigma6": float(pick(gradient, sigma=6.0).median_gradient_energy_retained_pct.iloc[0]),
        "same_tissue_nn_pct_raw": float(100 * pick(retrieval, sigma=0.0).top1_same_tissue_rate.mean()),
        "same_tissue_nn_pct_sigma6": float(100 * pick(retrieval, sigma=6.0).top1_same_tissue_rate.mean()),
    }
    return endpoints, frames


def bootstrap_endpoints(data: dict, tissue: np.ndarray, draws: np.ndarray) -> pd.DataFrame:
    """Slide-bootstrap percentile intervals for the set20 endpoints (not reported in the paper)."""
    raw, blur, sharpen, detail = data["raw"], data["blur"], data["sharpen"], data["blur_detail"]
    samples: dict[str, np.ndarray] = {}
    for name, values in (("direction_cosine_blur_sigma3", blur[:, :, :, S3]),
                         ("direction_cosine_sharpen_alpha2", sharpen[:, :, :, A2])):
        slide_delta = (values - raw).mean(axis=1)  # (slides, 6, D); equal locations per slide
        samples[name] = np.asarray([np.median(pair_cosines(slide_delta[d].mean(axis=0))) for d in draws])
    base, changed = paired_distances(blur[:, :, :, S6], raw)
    samples["distance_ratio_sigma6"] = np.asarray(
        [np.median(changed[d].mean(axis=0) / base[d].mean(axis=0)) for d in draws])
    energy = detail[:, :, :, S6, 0].mean(axis=1)  # (slides, 6)
    samples["gradient_energy_retained_pct_sigma6"] = np.asarray([100 * np.median(energy[d]) for d in draws])
    per_slide = {}
    for step, key in ((0, "same_tissue_nn_pct_raw"), (S6, "same_tissue_nn_pct_sigma6")):
        values = []
        for scanner_index in range(6):
            correct, evaluable = retrieval_correct(blur[:, :, scanner_index, step], tissue)
            values.append(correct)
        per_slide[key] = np.mean(values, axis=0)
    # Query slides are resampled; the reference bank stays fixed.
    queries = np.flatnonzero(evaluable)
    rng = np.random.default_rng(SEED + 1)
    query_draws = queries[rng.integers(0, len(queries), (N_BOOT, len(queries)))]
    for key, values in per_slide.items():
        samples[key] = 100 * values[query_draws].mean(axis=1)
    samples["same_tissue_nn_pct_change"] = 100 * (
        per_slide["same_tissue_nn_pct_sigma6"] - per_slide["same_tissue_nn_pct_raw"])[query_draws].mean(axis=1)
    return pd.DataFrame([{"endpoint": key, "ci_low": float(np.quantile(v, 0.025)),
                          "ci_high": float(np.quantile(v, 0.975))} for key, v in samples.items()])


# ----------------------------------------------------------------------------- PCA display
def fit_pca(data: dict, cohort: pd.DataFrame) -> tuple[PCA, pd.DataFrame, pd.DataFrame]:
    raw, blur, sharpen = data["raw"], data["blur"], data["sharpen"]
    center = raw.mean(axis=2, keepdims=True)[:, :, :, None, :]
    blur_c = blur - center
    sharpen_c = sharpen - center
    # 60 mean states: 6 scanners x (blur sigma 0-3, sharpen alpha 0-2); sigma = 6 only projected.
    states = np.concatenate([blur_c[:, :, :, :5].mean(axis=(0, 1))[:, None],
                             sharpen_c.mean(axis=(0, 1))[:, None]], axis=1).reshape(60, 1024)
    pca = PCA(n_components=5, svd_solver="full").fit(states)
    components = pca.components_[:2]
    records = []
    for operation, values, strengths in (("Gaussian blur", blur_c, BLUR_SIGMAS), ("Unsharp mask", sharpen_c, SHARPEN_ALPHAS)):
        projected = (values.mean(axis=1) - pca.mean_) @ components.T  # slide mean of centred states
        for slide_index, slide in enumerate(cohort.itertuples(index=False)):
            for scanner_index, scanner in enumerate(SCANNERS):
                for step, strength in enumerate(strengths):
                    x, y = projected[slide_index, scanner_index, step]
                    records.append({"slide_id": slide.slide_id, "scanner": scanner, "operation": operation,
                                    "step": step, "strength": strength, "pc1": float(x), "pc2": float(y)})
    slides = pd.DataFrame(records)
    centroids = slides.groupby(["scanner", "operation", "step", "strength"], as_index=False)[["pc1", "pc2"]].mean()
    return pca, slides, centroids


def plot_pca(centroids: pd.DataFrame, explained: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.8), dpi=150)
    display = {"Gaussian blur": (0.0, 0.5, 1.0, 3.0, 6.0), "Unsharp mask": SHARPEN_ALPHAS}
    markers = {"Gaussian blur": "o", "Unsharp mask": "s"}
    for scanner in SCANNERS:
        for operation, strengths in display.items():
            group = centroids[centroids.scanner.eq(scanner) & centroids.operation.eq(operation)
                              & centroids.strength.isin(strengths)].sort_values("strength")
            x, y = group.pc1.to_numpy(), group.pc2.to_numpy()
            ax.plot(x, y, color=COLORS[scanner], linewidth=1.2, alpha=0.5)
            ax.scatter(x[1:], y[1:], marker=markers[operation], s=np.linspace(30, 90, len(x) - 1),
                       color=COLORS[scanner], edgecolor="white", linewidth=0.6, zorder=3)
        start = centroids[centroids.scanner.eq(scanner) & centroids.step.eq(0)].iloc[0]
        ax.scatter(start.pc1, start.pc2, marker="X", s=120, color=COLORS[scanner], edgecolor="black",
                   linewidth=0.6, zorder=4, label=scanner.upper())
    ax.axhline(0, color="0.85", linewidth=0.7, zorder=0)
    ax.axvline(0, color="0.85", linewidth=0.7, zorder=0)
    ax.set_xlabel(f"UNI PC1 ({100 * explained[0]:.1f}%)")
    ax.set_ylabel(f"UNI PC2 ({100 * explained[1]:.1f}%)")
    ax.set_title("set20: blur (circles, sigma 0.5-6) and sharpen (squares, alpha 0.25-2)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="best", title="Raw (X)", title_fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_convergence(gaps: pd.DataFrame, gradient: pd.DataFrame, retrieval: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=150)
    styles = {"set20": dict(color="#2a5d9f", marker="o"), "probe3_rerun": dict(color="#9a9a9a", marker="s", linestyle="--")}
    for label, style in styles.items():
        part = gaps[gaps.set.eq(label) & gaps.operation.eq("Gaussian blur")]
        ratio = part.groupby("strength").mean_gap_ratio.median()
        axes[0].plot([0, *ratio.index], [1, *ratio.values], label=label, **style)
        grad = gradient[gradient.set.eq(label)].sort_values("sigma")
        axes[1].plot(grad.sigma, grad.median_gradient_energy_retained_pct, label=label, **style)
        ret = retrieval[retrieval.set.eq(label)].groupby("sigma").top1_same_tissue_rate.mean() * 100
        axes[2].plot(ret.index, ret.values, label=label, **style)
    axes[0].set(xlabel="Blur sigma (px)", ylabel="Median interscanner distance ratio")
    axes[1].set(xlabel="Blur sigma (px)", ylabel="Median retained gradient energy (%)", yscale="log")
    axes[2].set(xlabel="Blur sigma (px)", ylabel="Same-tissue NN (%), six-scanner mean")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ----------------------------------------------------------------------------- scanner probe
def probe_fold(x: np.ndarray, y: np.ndarray, folds: np.ndarray, outer: int) -> tuple[int, float, np.ndarray]:
    """Mirror of `_probe_predictions` for one outer fold (C chosen by inner locked folds)."""
    c = _select_c(x, y, folds, outer)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))
    train = folds != outer
    model.fit(x[train], y[train])
    return outer, c, model.predict(x[folds == outer])


def scanner_probe(conditions: dict, meta: pd.DataFrame, n_jobs: int) -> pd.DataFrame:
    y = meta.scanner.to_numpy()
    folds = meta.fold.to_numpy(int)
    jobs = [(name, outer) for name in conditions for outer in sorted(np.unique(folds))]
    results = Parallel(n_jobs=n_jobs)(delayed(probe_fold)(conditions[name], y, folds, outer) for name, outer in jobs)
    frames = []
    for (name, _), (outer, c, predicted) in zip(jobs, results):
        part = meta[folds == outer].copy()
        part["condition"] = name
        part["selected_c"] = c
        part["predicted"] = predicted
        frames.append(part)
    frame = pd.concat(frames, ignore_index=True)
    frame["correct"] = (frame.predicted == frame.scanner).astype(int)
    return frame


def cached_probe(level: str, conditions: dict, meta: pd.DataFrame, n_jobs: int) -> pd.DataFrame:
    """Out-of-fold predictions; reused on a re-run if the same conditions were already fitted."""
    path = OUTPUT / f"scanner_probe_{level}_predictions.csv.gz"
    if path.exists():
        cached = pd.read_csv(path, dtype={"slide_id": str})
        if set(cached.condition) == set(conditions) and len(cached) == len(meta) * len(conditions):
            print(f"reusing {path.name}", flush=True)
            return cached.drop(columns="level")
    predictions = scanner_probe(conditions, meta, n_jobs)
    predictions.assign(level=level).to_csv(path, index=False)
    return predictions


def probe_summary(predictions: pd.DataFrame, level: str, draws: np.ndarray, slides: list[str]) -> pd.DataFrame:
    """Balanced accuracy (macro recall over six scanners) with joint slide-cluster bootstrap."""
    rows = []
    per_slide = {}
    for condition, part in predictions.groupby("condition"):
        table = part.groupby(["slide_id", "scanner"]).correct.mean().unstack("scanner").loc[slides, list(SCANNERS)]
        per_slide[condition] = table.to_numpy()  # (slides, 6); equal rows per slide and scanner
    for condition, values in per_slide.items():
        point = float(values.mean(axis=0).mean())
        boot = values[draws].mean(axis=1).mean(axis=1)
        row = {"level": level, "condition": condition, "balanced_accuracy": point,
               "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975)),
               "chance": 1 / 6, "slides": len(slides),
               "selected_c": ";".join(str(c) for c in sorted(predictions[predictions.condition.eq(condition)].selected_c.unique()))}
        if condition != "raw":
            delta = (values - per_slide["raw"]).mean(axis=1)
            dboot = delta[draws].mean(axis=1)
            row.update({"minus_raw": float(delta.mean()), "minus_raw_ci_low": float(np.quantile(dboot, 0.025)),
                        "minus_raw_ci_high": float(np.quantile(dboot, 0.975))})
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def fmt(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(COHORT, dtype={"slide_id": str})
    folds = pd.read_csv(FOLDS, dtype={"slide_id": str}).set_index("slide_id").fold
    if not (cohort.set_index("slide_id").fold == folds.loc[cohort.slide_id]).all():
        raise ValueError("cohort folds disagree with the locked fold file")
    set20 = pd.read_csv(SET20, dtype={"slide_id": str})
    tissue = cohort.tissue_type.astype(str).to_numpy()
    rerun = load_rerun(cohort, set20)
    paper = load_paper(cohort)
    probe = rerun["probe"]
    rerun3 = {key: np.stack([rerun[key][s][probe[s]] for s in range(103)])
              for key in ("raw", "blur", "sharpen", "blur_detail")}

    # QC: extraction parity and agreement with the paper shards at the paper locations.
    parity = np.einsum("sld,sld->sl", rerun["recomputed"].reshape(103, -1, 1024), rerun["raw"].reshape(103, -1, 1024))
    if not np.array_equal(np.stack([set20[set20.slide_id.eq(s) & set20.paper_blur_probe].sort_values("location_index").location_index
                                    for s in cohort.slide_id]), paper["locations"]):
        raise ValueError("paper probe locations differ from the paper shards")
    cos_blur = np.einsum("slckd,slckd->slck", rerun3["blur"][:, :, :, 1:], paper["blur"][:, :, :, 1:])
    cos_sharp = np.einsum("slckd,slckd->slck", rerun3["sharpen"][:, :, :, 1:], paper["sharpen"][:, :, :, 1:])
    detail_diff = np.abs(rerun3["blur_detail"][:, :, :, [S3, S6]] - paper["blur_detail"][:, :, :, [S3, S6]])
    qc = {
        "slides": 103, "locations_per_slide": 20, "scanners": list(SCANNERS),
        "blur_sigmas": list(BLUR_SIGMAS), "sharpen_alphas": list(SHARPEN_ALPHAS),
        "raw_reembedding_vs_stored_cosine": {"min": float(parity.min()), "median": float(np.median(parity)),
                                             "fraction_ge_0.999": float((parity >= 0.999).mean())},
        "paper_locations_blur_cosine_vs_paper_shards": {"min": float(cos_blur.min()), "median": float(np.median(cos_blur))},
        "paper_locations_sharpen_cosine_vs_paper_shards": {"min": float(cos_sharp.min()), "median": float(np.median(cos_sharp))},
        "paper_locations_gradient_metric_max_abs_diff": float(detail_diff.max()),
        "nonfinite_values": 0,
    }

    endpoints, tables = {}, {}
    for label, data in (("paper_shards", paper), ("probe3_rerun", rerun3), ("set20", rerun)):
        endpoints[label], tables[label] = compute_endpoints(data, tissue, label)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, 103, (N_BOOT, 103))
    intervals = bootstrap_endpoints(rerun, tissue, draws).set_index("endpoint")
    endpoint_rows = []
    for key, published in PUBLISHED.items():
        endpoint_rows.append({"endpoint": key, "published": published,
                              "paper_shards_recomputed": endpoints["paper_shards"][key],
                              "probe3_rerun": endpoints["probe3_rerun"][key],
                              "set20": endpoints["set20"][key],
                              "set20_ci_low": intervals.loc[key, "ci_low"],
                              "set20_ci_high": intervals.loc[key, "ci_high"]})
    endpoint_frame = pd.DataFrame(endpoint_rows)
    endpoint_frame.to_csv(OUTPUT / "endpoints.csv", index=False)
    change = intervals.loc["same_tissue_nn_pct_change"]
    for name in ("direction", "gaps", "gradient", "retrieval"):
        pd.concat([tables[label][name] for label in tables], ignore_index=True).to_csv(
            OUTPUT / {"direction": "direction_alignment.csv", "gaps": "paired_gaps.csv",
                      "gradient": "gradient_energy.csv", "retrieval": "tissue_retrieval.csv"}[name], index=False)

    # PCA display (60 mean states on set20) and comparison with the paper's locked basis.
    pca, slide_positions, centroids = fit_pca(rerun, cohort)
    np.savez_compressed(OUTPUT / "pca_set20.npz", components=pca.components_.astype(np.float32),
                        mean=pca.mean_.astype(np.float32),
                        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32))
    slide_positions.to_csv(OUTPUT / "pca_slide_positions.csv", index=False)
    centroids.to_csv(OUTPUT / "pca_centroids.csv", index=False)
    locked = np.load(PAPER_COMMON / "paired_centered_blur_sharp_pca.npz")
    paper_components = np.asarray(locked["components"][:2], dtype=np.float64)
    new_components = pca.components_[:2].astype(np.float64)
    singular = np.linalg.svd(new_components @ paper_components.T, compute_uv=False)
    pca_report = {
        "fit": "60 mean states (6 scanners x blur sigma 0,0.5,1,2,3 and sharpen alpha 0,0.25,0.5,1,2) after paired centring; sigma = 6 projected",
        "explained_variance_ratio_first_two": float(pca.explained_variance_ratio_[:2].sum()),
        "paper_explained_variance_ratio_first_two": float(np.asarray(locked["explained_variance_ratio"])[:2].sum()),
        "abs_cosine_pc1_vs_paper": float(abs(new_components[0] @ paper_components[0])),
        "abs_cosine_pc2_vs_paper": float(abs(new_components[1] @ paper_components[1])),
        "principal_angle_cosines_2d_subspace": [float(x) for x in singular],
    }
    base_xy = centroids[centroids.step.eq(0) & centroids.operation.eq("Gaussian blur")].set_index("scanner").loc[list(SCANNERS), ["pc1", "pc2"]].to_numpy()
    s6_xy = centroids[centroids.strength.eq(6.0) & centroids.operation.eq("Gaussian blur")].set_index("scanner").loc[list(SCANNERS), ["pc1", "pc2"]].to_numpy()
    pairs = list(combinations(range(6), 2))
    pca_report["median_2d_centroid_gap_ratio_sigma6"] = float(np.median(
        [np.linalg.norm(s6_xy[i] - s6_xy[j]) / np.linalg.norm(base_xy[i] - base_xy[j]) for i, j in pairs]))
    (OUTPUT / "pca_basis_comparison.json").write_text(json.dumps(pca_report, indent=2) + "\n")
    plot_pca(centroids, pca.explained_variance_ratio_, OUTPUT / "pca_trajectories_set20.png")
    all_tables = {name: pd.concat([tables[label][name] for label in tables], ignore_index=True)
                  for name in ("gaps", "gradient", "retrieval")}
    plot_convergence(all_tables["gaps"], all_tables["gradient"], all_tables["retrieval"],
                     OUTPUT / "convergence_and_retention.png")

    # Secondary endpoint: six-way scanner probe (location level, primary; slide means, sensitivity).
    slides = list(cohort.slide_id)
    fold_of = folds.loc[slides].to_numpy(int)
    loc_meta = pd.DataFrame({
        "slide_id": np.repeat(slides, 20 * 6),
        "fold": np.repeat(fold_of, 20 * 6),
        "location_index": np.concatenate([np.repeat(set20[set20.slide_id.eq(s)].sort_values("location_index").location_index.to_numpy(), 6) for s in slides]),
        "scanner": np.tile(SCANNERS, 103 * 20),
    })
    location_conditions = {"raw": rerun["raw"].reshape(-1, 1024),
                           "blur_sigma3": rerun["blur"][:, :, :, S3].reshape(-1, 1024),
                           "blur_sigma6": rerun["blur"][:, :, :, S6].reshape(-1, 1024)}
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    location_predictions = cached_probe("location", location_conditions, loc_meta, n_jobs)
    slide_meta = pd.DataFrame({"slide_id": np.repeat(slides, 6), "fold": np.repeat(fold_of, 6),
                               "scanner": np.tile(SCANNERS, 103)})
    slide_conditions = {"raw": unit(rerun["raw"].mean(axis=1)).reshape(-1, 1024)}
    for step, sigma in enumerate(BLUR_SIGMAS[1:], start=1):
        slide_conditions[f"blur_sigma{sigma:g}"] = unit(rerun["blur"][:, :, :, step].mean(axis=1)).reshape(-1, 1024)
    for step, alpha in enumerate(SHARPEN_ALPHAS[1:], start=1):
        slide_conditions[f"sharpen_alpha{alpha:g}"] = unit(rerun["sharpen"][:, :, :, step].mean(axis=1)).reshape(-1, 1024)
    slide_predictions = cached_probe("slide_mean", slide_conditions, slide_meta, n_jobs)
    probe_frame = pd.concat([probe_summary(location_predictions, "location", draws, slides),
                             probe_summary(slide_predictions, "slide_mean", draws, slides)], ignore_index=True)
    probe_frame.to_csv(OUTPUT / "scanner_probe_summary.csv", index=False)

    qc_path = OUTPUT / "qc.json"
    qc_path.write_text(json.dumps(qc, indent=2) + "\n")
    write_summary(endpoint_frame, change, qc, pca_report, probe_frame, tables)
    print(endpoint_frame.to_string(index=False), flush=True)
    print(probe_frame.to_string(index=False), flush=True)


def write_summary(endpoints: pd.DataFrame, change: pd.Series, qc: dict, pca_report: dict,
                  probe: pd.DataFrame, tables: dict) -> None:
    e = endpoints.set_index("endpoint")
    reproduced = all(
        abs(e.loc[k, "paper_shards_recomputed"] - e.loc[k, "published"]) <= (0.006 if "cosine" in k else 0.06 if "pct" in k else 0.0006)
        for k in e.index)
    loc_probe = probe[probe.level.eq("location")].set_index("condition")
    slide_probe = probe[probe.level.eq("slide_mean")].set_index("condition")
    lines = [
        "# RV01 blur and sharpening trajectories on set20",
        "",
        "UNI v1, six scanners, 103 PanNormal slides, 20 `set20` locations per slide (2,060 locations).",
        "Blur sigma = 0.5, 1, 2, 3, 6; unsharp alpha = 0.25, 0.5, 1, 2. Raw state = stored",
        "`augmentation_ood_v1` UNI features, as in the paper. Code: `analysis/revision/blur_sharpen_set20_features.py`",
        "(GPU) and `analysis/revision/blur_sharpen_set20.py` (this aggregation).",
        "",
        "## QC",
        "",
        f"- Raw re-embedding vs stored UNI features: min cosine {qc['raw_reembedding_vs_stored_cosine']['min']:.6f} "
        f"(fraction >= 0.999: {qc['raw_reembedding_vs_stored_cosine']['fraction_ge_0.999']:.4f}).",
        f"- Re-run vs paper shards at the paper's 3 locations: blur min cosine {qc['paper_locations_blur_cosine_vs_paper_shards']['min']:.6f}, "
        f"sharpen min cosine {qc['paper_locations_sharpen_cosine_vs_paper_shards']['min']:.6f}, "
        f"gradient metrics max abs diff {qc['paper_locations_gradient_metric_max_abs_diff']:.2e}.",
        f"- This implementation applied to the paper shards reproduces the published values within rounding: {'yes' if reproduced else 'NO - see endpoints.csv'}.",
        "- All shards complete (103 x 20 x 6), finite, locations equal `set20`.",
        "",
        "## Primary endpoints",
        "",
        "| Endpoint | Published (3 loc) | Paper shards, this code | Re-run, 3 loc | set20 | set20 95% CI (slide bootstrap) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    names = {
        "direction_cosine_blur_sigma3": "Median displacement cosine, blur sigma = 3",
        "direction_cosine_sharpen_alpha2": "Median displacement cosine, sharpen alpha = 2",
        "distance_ratio_sigma6": "Median interscanner distance ratio, sigma = 6",
        "gradient_energy_retained_pct_sigma6": "Median retained gradient energy (%), sigma = 6",
        "same_tissue_nn_pct_raw": "Same-tissue NN (%), raw",
        "same_tissue_nn_pct_sigma6": "Same-tissue NN (%), sigma = 6",
    }
    for key, row in e.iterrows():
        digits = 1 if key.startswith("same_tissue") else 3
        lines.append(f"| {names[key]} | {row.published:g} | {fmt(row.paper_shards_recomputed, digits)} | "
                     f"{fmt(row.probe3_rerun, digits)} | **{fmt(row.set20, digits)}** | "
                     f"{fmt(row.set20_ci_low, digits)} to {fmt(row.set20_ci_high, digits)} |")
    lines += [
        "",
        f"Same-tissue NN change raw -> sigma = 6 on set20: {e.loc['same_tissue_nn_pct_sigma6', 'set20'] - e.loc['same_tissue_nn_pct_raw', 'set20']:+.1f} "
        f"percentage points (95% CI {change.ci_low:+.1f} to {change.ci_high:+.1f}; query slides resampled, bank fixed).",
        "Retrieval queries: 102 slides from tissue types with at least two slides; slide mean over 20 locations;",
        "averaged over six scanners. CIs were not part of the paper's reporting and are added here.",
        "",
        "Reading against the protocol expectations:",
        f"- Shared directions (median cosine > 0.5): blur sigma = 3 {fmt(e.loc['direction_cosine_blur_sigma3', 'set20'])}, "
        f"sharpen alpha = 2 {fmt(e.loc['direction_cosine_sharpen_alpha2', 'set20'])} -> "
        f"{'holds' if min(e.loc['direction_cosine_blur_sigma3', 'set20'], e.loc['direction_cosine_sharpen_alpha2', 'set20']) > 0.5 else 'does NOT hold'}.",
        f"- Distance ratio well below 1 at sigma = 6: {fmt(e.loc['distance_ratio_sigma6', 'set20'])} -> "
        f"{'holds' if e.loc['distance_ratio_sigma6', 'set20'] < 0.8 else 'does NOT hold'}.",
        f"- Retained gradient near 1%: {fmt(e.loc['gradient_energy_retained_pct_sigma6', 'set20'], 2)}%.",
        f"- Lower tissue retrieval under strong blur: {fmt(e.loc['same_tissue_nn_pct_raw', 'set20'], 1)}% -> "
        f"{fmt(e.loc['same_tissue_nn_pct_sigma6', 'set20'], 1)}% -> "
        f"{'holds' if change.ci_high < 0 else 'direction ' + ('lower' if e.loc['same_tissue_nn_pct_sigma6', 'set20'] < e.loc['same_tissue_nn_pct_raw', 'set20'] else 'not lower') + ', CI includes 0'}.",
        "",
        "## Secondary endpoint: six-way scanner probe",
        "",
        "StandardScaler + multinomial logistic regression (class-balanced; C in {0.01, 0.1, 1} chosen by inner",
        "cross-validation over the locked slide folds, as `src/manuscript_completion/frequency_aggregate.py`), outer",
        "cross-validation over the five locked slide folds; balanced accuracy (chance 0.167); 95% CI from 2,000 slide",
        "bootstrap resamples; paired difference to raw uses the same resamples.",
        "",
        "| Level | Condition | Balanced accuracy | 95% CI | Minus raw (95% CI) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for level, frame in (("location (primary)", loc_probe), ("slide mean (sensitivity)", slide_probe)):
        for condition, row in frame.iterrows():
            delta = "" if condition == "raw" else f"{row.minus_raw:+.3f} ({row.minus_raw_ci_low:+.3f} to {row.minus_raw_ci_high:+.3f})"
            lines.append(f"| {level} | {condition} | {row.balanced_accuracy:.3f} | {row.ci_low:.3f}-{row.ci_high:.3f} | {delta} |")
    lines += [
        "",
        "## PCA display",
        "",
        f"- set20 PCA (60 mean states) first two components explain {pca_report['explained_variance_ratio_first_two']:.3f} "
        f"(paper basis: {pca_report['paper_explained_variance_ratio_first_two']:.3f}).",
        f"- |cos| with the paper basis: PC1 {pca_report['abs_cosine_pc1_vs_paper']:.3f}, PC2 {pca_report['abs_cosine_pc2_vs_paper']:.3f}; "
        f"principal-angle cosines of the 2D subspaces {', '.join(f'{x:.3f}' for x in pca_report['principal_angle_cosines_2d_subspace'])}.",
        f"- Median 2D centroid gap ratio at sigma = 6 in the set20 basis: {pca_report['median_2d_centroid_gap_ratio_sigma6']:.3f}.",
        "- Plots: `pca_trajectories_set20.png`, `convergence_and_retention.png` (diagnostic only; not paper figures).",
        "",
        "## Implementation notes (proposed deviations, for the protocol log)",
        "",
        "- Strengths are those listed in the protocol; the paper's additional sigma = 12, 24, 48 and gamma sweeps were not re-run.",
        "- The protocol does not name a classifier for the scanner probe; the existing six-way probe specification above",
        "  was used on location embeddings (primary), with slide-mean embeddings as sensitivity; sigma = 3 was added.",
        "- Slide-bootstrap CIs for the primary endpoints are additions; the retrieval CI resamples query slides with the",
        "  reference bank fixed.",
        "- Retrieval percentages are higher than published because slide means use 20 locations (anticipated by the protocol).",
        "",
        "## Files",
        "",
        "`endpoints.csv`, `direction_alignment.csv`, `paired_gaps.csv`, `gradient_energy.csv`, `tissue_retrieval.csv`",
        "(each with `set` = paper_shards / probe3_rerun / set20), `pca_set20.npz`, `pca_slide_positions.csv`,",
        "`pca_centroids.csv`, `pca_basis_comparison.json`, `scanner_probe_summary.csv`, `scanner_probe_*_predictions.csv.gz`,",
        "`qc.json`, `shards/` (per-slide embeddings).",
    ]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
