#!/usr/bin/env python3
"""RV-P0b: recompute the published PanNormal image metrics on the locked set20.

Two subcommands:

``gan-residual --task-index i``
    One slide per task. Computes the ten-measure image residual, coverage, joint
    coverage and the phenotype gradient NCCs (the 02_correction definitions) for
    the stored Pix2Pix and CycleGAN predictions at the set20 locations, which no
    earlier analysis computed. Raw, Reinhard, frequency and combined images are
    rendered again in the same task as a parity check against 02_correction.

``aggregate``
    Assembles location-level metrics for every image method from the frozen
    outputs, restricts them to set20, aggregates them as in the paper (locations
    within slide x scanner, five target scanners equally weighted within slide,
    summary over 103 slides) and writes the slide-level long table, summaries,
    the augmentation-oracle recomputation and ``comparison.csv`` against every
    published number in Table 2 and the augmentation / colour-frequency text.

Nothing outside ``analysis/revision/results/set20_reaggregation`` is written.

Submit from the repository root (the aggregate waits for the array)::

    EXPORT=ALL,JOB_CONDA_PREFIX=$CONDA_PREFIX,JOB_WORKDIR=$PWD
    GAN=$(sbatch --parsable --export=$EXPORT analysis/revision/set20_reaggregation_gan_residual.sbatch)
    sbatch --dependency=afterok:$GAN --export=$EXPORT analysis/revision/set20_reaggregation.sbatch
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

BATCH = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17"
BENCH = BATCH / "12_manuscript_completion/02_baseline_benchmark"
LEARNED = BATCH / "06_learned_baselines"
AUGMENTATION = PROJECT / "outputs/augmentation_ood_v1"
AUG_PAIRED = PROJECT / "analysis/paper/results/augmentation_paired_metrics"
GAN_REVIEW = PROJECT / "outputs/gan_encoder_review_2026-09-25"
ABLATION = PROJECT / "outputs/discussion_followup_2026-09-25/structure_ablation"
TABLE2_SHARDS = BATCH / "12_manuscript_completion/09_table2_benchmark/pannormal/shards"
MANUSCRIPT = PROJECT / "00_manuscript"
LOCATIONS = Path(__file__).resolve().parent / "results/location_sets"
OUTPUT = Path(__file__).resolve().parent / "results/set20_reaggregation"
GAN_SHARDS = OUTPUT / "gan_image_residual"

COHORT = BATCH / "00_contract/cohort.csv"
FOLDS = AUGMENTATION / "00_contract/slide_folds.csv"
SCANNERS = ("versa", "akoya", "gt450", "s360", "s60")
METHODS = ("raw", "reinhard", "macenko", "vahadane", "pix2pix", "cyclegan", "frequency", "combined")
TABLE2_METHODS = ("raw", "reinhard", "macenko", "vahadane", "pix2pix", "cyclegan", "combined")
METRICS = ("image_residual", "coverage", "joint_coverage", "ssim", "lpips_vgg",
           "target_gradient_ncc", "source_gradient_ncc", "uni_distance")
LOWER_BETTER = {"image_residual", "lpips_vgg", "uni_distance", "uni_distance_path_c",
                "invented_edge_fraction", "source_edge_deletion_fraction"}
PHENOTYPE_COLUMNS = {
    "distance": "image_residual",
    "endpoint_coverage": "coverage",
    "all_endpoints_covered": "joint_coverage",
    "target_gradient_ncc": "target_gradient_ncc",
    "fidelity_ncc": "source_gradient_ncc",
}
PARITY_ARMS = ("raw", "reinhard", "frequency", "combined")
ORACLE_ARMS = ("identity", "default_global", "default_oracle", "strong_global",
               "strong_unconstrained_oracle", "strong_oracle")
BOOTSTRAPS = 2000
SEED = 20260929


# --------------------------------------------------------------------------- io

def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    temporary.replace(path)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=float) + "\n")
    temporary.replace(path)


def load_sets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(COHORT, dtype={"slide_id": str})
    set40 = pd.read_csv(LOCATIONS / "set40.csv", dtype={"slide_id": str})
    set20 = pd.read_csv(LOCATIONS / "set20.csv", dtype={"slide_id": str})
    return cohort, set40, set20


def location_sets(frame: pd.DataFrame) -> dict[str, frozenset]:
    return {slide: frozenset(int(v) for v in group) for slide, group in
            frame.groupby("slide_id")["location_index"]}


# ------------------------------------------------------------ gan residual task

def gan_residual_task(task_index: int, workers: int) -> None:
    from augmentation_ood_study import (
        ENDPOINTS, absolute_measurements, contrasts_from_absolute,
    )
    from final_image_study import rowwise_correlation
    from scanner_batch_extensions import frequency_transform, reinhard_transform

    cohort, _, set20 = load_sets()
    row = cohort.iloc[task_index]
    slide_id, fold, tissue = str(row.slide_id), int(row.fold), str(row.tissue_type)
    locations = np.sort(set20.loc[set20.slide_id.eq(slide_id), "location_index"].to_numpy(int))
    if len(locations) != 20:
        raise ValueError(f"{slide_id}: set20 has {len(locations)} locations")
    parameters = json.loads((BATCH / "02_correction/parameters.json").read_text())["fold"][str(fold)]
    references = pd.read_csv(BATCH / "00_contract/target_references" / f"{slide_id}.csv")
    references = references.set_index(["location_index", "scanner"])
    pairs = pd.read_csv(GAN_REVIEW / "prediction_pairs.csv", dtype={"slide_id": str})
    pairs = pairs[pairs.slide_id.eq(slide_id)].set_index("scanner")
    with h5py.File(Path(row.cache_path), "r") as store:
        images = np.asarray(store["images"][locations], dtype=np.uint8)
        source_index = np.asarray(store["source_index"][locations], dtype=np.int64)

    generated: dict[tuple[str, str], np.ndarray] = {}
    pixel_rows = []
    for scanner_index, scanner in enumerate(SCANNERS, start=1):
        record = pairs.loc[scanner]
        if int(record.fold) != fold:
            raise ValueError(f"{slide_id}/{scanner}: prediction fold differs from cohort fold")
        for method in ("pix2pix", "cyclegan"):
            path = Path(record[f"{method}_path"])
            with h5py.File(path, "r") as store:
                observed = np.asarray(store["metadata/location_index"], dtype=int)
                lookup = {int(value): offset for offset, value in enumerate(observed)}
                index = np.asarray([lookup[int(value)] for value in locations])
                order = np.argsort(index)
                inverse = np.argsort(order)
                sorted_index = index[order]
                source = np.asarray(store["images/raw_source"][sorted_index])[inverse]
                target = np.asarray(store["images/real_target"][sorted_index])[inverse]
                generated[(method, scanner)] = np.asarray(store["images/generated"][sorted_index])[inverse]
                if str(store.attrs["slide_id"]) != slide_id or int(store.attrs["test_fold"]) != fold:
                    raise ValueError(f"prediction identity mismatch: {path}")
            pixel_rows.append({
                "slide_id": slide_id, "scanner": scanner, "method": method,
                "source_equal_cache": bool(np.array_equal(source, images[:, 0])),
                "target_equal_cache": bool(np.array_equal(target, images[:, scanner_index])),
            })

    rows = []
    for offset, location in enumerate(locations):
        source = images[offset, 0]
        source_scalar, source_radial, source_gradient, _ = absolute_measurements(source[None], workers)
        for scanner_index, scanner in enumerate(SCANNERS, start=1):
            target = images[offset, scanner_index]
            reinhard = reinhard_transform(source, parameters[scanner])
            conditions = {
                "raw": source,
                "reinhard": reinhard,
                "frequency": frequency_transform(source, parameters[scanner]),
                "combined": frequency_transform(reinhard, parameters[scanner]),
                "pix2pix": generated[("pix2pix", scanner)][offset],
                "cyclegan": generated[("cyclegan", scanner)][offset],
            }
            arms = tuple(conditions)
            scalar, radial, gradient, _ = absolute_measurements(np.stack([conditions[a] for a in arms]), workers)
            contrast = contrasts_from_absolute(scalar, radial, source_scalar, source_radial)
            target_scalar, target_radial, target_gradient, _ = absolute_measurements(target[None], workers)
            target_vector = contrasts_from_absolute(target_scalar, target_radial, source_scalar, source_radial)[0]
            reference = references.loc[(int(location), scanner)]
            scale = np.maximum(np.asarray([float(reference[f"scale_{e}"]) for e in ENDPOINTS]), 1e-6)
            reference_target = np.asarray([float(reference[f"target_{e}"]) for e in ENDPOINTS])
            if not np.allclose(target_vector, reference_target, atol=3e-3, rtol=3e-3):
                raise ValueError(f"target endpoint mismatch {slide_id}/{location}/{scanner}")
            fidelity = rowwise_correlation(gradient, np.repeat(source_gradient, len(arms), axis=0))
            similarity = rowwise_correlation(gradient, np.repeat(target_gradient, len(arms), axis=0))
            for arm_index, arm in enumerate(arms):
                residual = (contrast[arm_index] - target_vector) / scale
                payload = {
                    "slide_id": slide_id, "tissue_type": tissue, "fold": fold,
                    "location_index": int(location), "source_index": int(source_index[offset]),
                    "scanner": scanner, "arm": arm,
                    "distance": float(np.sqrt(np.mean(residual ** 2))),
                    "endpoint_coverage": float(np.mean(np.abs(residual) <= 1.0)),
                    "all_endpoints_covered": bool(np.all(np.abs(residual) <= 1.0)),
                    "fidelity_ncc": float(fidelity[arm_index]),
                    "target_gradient_ncc": float(similarity[arm_index]),
                    "saturation_fraction": float(scalar[arm_index, 8]),
                }
                for endpoint_index, endpoint in enumerate(ENDPOINTS):
                    payload[f"scaled_residual_{endpoint}"] = float(residual[endpoint_index])
                rows.append(payload)
    frame = pd.DataFrame(rows)
    if len(frame) != 20 * 5 * 6 or not np.isfinite(frame.select_dtypes("number")).all().all():
        raise ValueError(f"invalid GAN residual shard for {slide_id}")
    write_frame(GAN_SHARDS / "shards" / f"{slide_id}.csv.gz", frame)
    write_frame(GAN_SHARDS / "pixel_checks" / f"{slide_id}.csv", pd.DataFrame(pixel_rows))
    print(json.dumps({"slide_id": slide_id, "rows": len(frame),
                      "pixel_equal": bool(pd.DataFrame(pixel_rows)[["source_equal_cache", "target_equal_cache"]].all().all())}))


# ------------------------------------------------------------ statistics

class Bootstrap:
    """One set of slide resamples shared by every estimate (paired differences)."""

    def __init__(self, slides: list[str]):
        self.slides = list(slides)
        rng = np.random.default_rng(SEED)
        self.draws = rng.integers(0, len(slides), size=(BOOTSTRAPS, len(slides)))

    def ci(self, values: pd.Series) -> tuple[float, float]:
        array = values.reindex(self.slides).to_numpy(float)
        if np.isnan(array).any():
            raise ValueError("bootstrap input has missing slides")
        sampled = array[self.draws].mean(axis=1)
        return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def signed_pct(metric: str, value: float, raw: float) -> float:
    if raw == 0 or not np.isfinite(raw):
        return float("nan")
    if metric in LOWER_BETTER:
        return 100.0 * (raw - value) / raw
    return 100.0 * (value - raw) / raw


def slide_values(slide_scanner: pd.DataFrame, method: str, metric: str, scanner: str = "all") -> pd.Series:
    """Slide-level value; 'all' weights the five target scanners equally within slide."""
    subset = slide_scanner[slide_scanner.method.eq(method) & slide_scanner.metric.eq(metric)]
    pivot = subset.pivot(index="slide_id", columns="scanner", values="value")
    if scanner == "all":
        if list(sorted(pivot.columns)) != sorted(SCANNERS) or pivot.isna().any().any():
            raise ValueError(f"incomplete scanners for {method}/{metric}")
        return pivot[list(SCANNERS)].mean(axis=1)
    return pivot[scanner]


def summarize(slide_scanner: pd.DataFrame, boot: Bootstrap, sample: str) -> pd.DataFrame:
    rows = []
    methods = [m for m in METHODS if m in set(slide_scanner.method)]
    for metric in sorted(set(slide_scanner.metric)):
        for scanner in ("all",) + SCANNERS:
            if not ((slide_scanner.metric == metric) & (slide_scanner.method == "raw")).any():
                raw = None
            else:
                raw = slide_values(slide_scanner, "raw", metric, scanner)
            for method in methods:
                if not ((slide_scanner.metric == metric) & (slide_scanner.method == method)).any():
                    continue
                values = slide_values(slide_scanner, method, metric, scanner)
                low, high = boot.ci(values)
                row = {"sample": sample, "metric": metric, "scanner": scanner, "method": method,
                       "n_slides": int(values.notna().sum()), "mean": float(values.mean()),
                       "sd": float(values.std(ddof=1)), "ci_low": low, "ci_high": high}
                if raw is not None:
                    difference = values - raw
                    d_low, d_high = boot.ci(difference)
                    row.update({
                        "raw_mean": float(raw.mean()),
                        "pct_improvement_vs_raw": signed_pct(metric, float(values.mean()), float(raw.mean())),
                        "diff_vs_raw": float(difference.mean()),
                        "diff_vs_raw_ci_low": d_low, "diff_vs_raw_ci_high": d_high,
                    })
                rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------ assembly

def check_locations(name: str, frame: pd.DataFrame, expected: dict[str, frozenset], qc: dict) -> None:
    observed = location_sets(frame)
    ok = observed.keys() == expected.keys() and all(observed[k] == expected[k] for k in expected)
    qc["location_sets"][name] = bool(ok)
    if not ok:
        raise ValueError(f"{name}: location set differs from the locked set")


def phenotype_frames(set40: dict, set20: dict, qc: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Location-level phenotype metrics (02_correction definitions) for all eight methods."""
    columns = ["slide_id", "location_index", "scanner", "arm", *PHENOTYPE_COLUMNS]
    correction = pd.read_csv(BATCH / "02_correction/location_metrics.csv", usecols=columns,
                             dtype={"slide_id": str})
    correction = correction[correction.arm.isin(PARITY_ARMS)]
    check_locations("02_correction", correction, set40, qc)
    stain = pd.read_csv(BENCH / "01_stain_v4_mu_convergence/stain_location_metrics.csv.gz",
                        usecols=columns, dtype={"slide_id": str})
    check_locations("01_stain_v4_mu_convergence", stain, set40, qc)
    gan = pd.concat([pd.read_csv(p, dtype={"slide_id": str}) for p in
                     sorted((GAN_SHARDS / "shards").glob("*.csv.gz"))], ignore_index=True)
    if gan.slide_id.nunique() != 103:
        raise ValueError(f"GAN residual shards cover {gan.slide_id.nunique()} slides, expected 103")
    check_locations("gan_image_residual (new)", gan, set20, qc)
    pixel = pd.concat([pd.read_csv(p, dtype={"slide_id": str}) for p in
                       sorted((GAN_SHARDS / "pixel_checks").glob("*.csv"))], ignore_index=True)
    qc["gan_prediction_pixels_equal_cache"] = {
        "pairs": int(len(pixel)),
        "source_equal": int(pixel.source_equal_cache.sum()),
        "target_equal": int(pixel.target_equal_cache.sum()),
    }
    # Parity of the re-rendered conventional arms with the frozen 02_correction values.
    merged = gan[gan.arm.isin(PARITY_ARMS)].merge(
        correction, on=["slide_id", "location_index", "scanner", "arm"], suffixes=("_new", "_frozen"),
        validate="one_to_one")
    if len(merged) != 2060 * 5 * len(PARITY_ARMS):
        raise ValueError("parity merge incomplete")
    qc["gan_pipeline_parity_vs_02_correction_max_abs_diff"] = {
        column: float(np.abs(merged[f"{column}_new"].astype(float) - merged[f"{column}_frozen"].astype(float)).max())
        for column in PHENOTYPE_COLUMNS
    }
    def tidy(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.rename(columns={"arm": "method", **PHENOTYPE_COLUMNS})
        frame["joint_coverage"] = frame["joint_coverage"].astype(float)
        frame["scanner"] = frame["scanner"].str.lower()
        return frame

    in40 = tidy(pd.concat([correction, stain], ignore_index=True))
    frame = tidy(pd.concat([correction, stain, gan[gan.arm.isin(["pix2pix", "cyclegan"])][columns]],
                           ignore_index=True))
    frame = frame[[int(l) in set20[s] for s, l in zip(frame.slide_id, frame.location_index)]]
    return frame, in40


def benchmark_frame(set20: dict, qc: dict) -> pd.DataFrame:
    files = sorted(TABLE2_SHARDS.glob("*.csv.gz"))
    if len(files) != 103:
        raise ValueError("expected 103 Table 2 PanNormal shards")
    frame = pd.concat([pd.read_csv(p, dtype={"slide_id": str}) for p in files], ignore_index=True)
    check_locations("09_table2_benchmark", frame, set20, qc)
    frame["scanner"] = frame["scanner"].str.lower()
    return frame.rename(columns={
        "target_ssim": "ssim",
        "target_gradient_ncc": "target_gradient_ncc_gray",
        "source_gradient_ncc": "source_gradient_ncc_gray",
    })[["slide_id", "location_index", "scanner", "method", "ssim", "lpips_vgg",
        "target_gradient_ncc_gray", "source_gradient_ncc_gray", "invented_edge_fraction",
        "source_edge_deletion_fraction"]]


def gan_uni_path_b(method: str) -> pd.DataFrame:
    parts = []
    for scanner in SCANNERS:
        if scanner == "gt450":
            sub = "09_bidirectional_full_training" if method == "pix2pix" else "11_cyclegan_full_training"
            path = LEARNED / sub / "04_uni/at2_to_gt450/location_metrics.csv.gz"
        else:
            sub = "03_pix2pix" if method == "pix2pix" else "04_cyclegan"
            path = BENCH / sub / f"04_uni/at2_to_{scanner}/location_metrics.csv.gz"
        part = pd.read_csv(path, dtype={"slide_id": str})
        part["scanner"] = scanner
        parts.append(part[["slide_id", "location_index", "scanner", "raw_to_target_distance",
                           "method_to_target_distance"]])
    frame = pd.concat(parts, ignore_index=True)
    frame["method"] = method
    return frame


def uni_frames(set40: dict, set20: dict, qc: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    conventional = pd.read_csv(BATCH / "03_uni/location_metrics.csv", dtype={"slide_id": str})
    conventional = conventional[conventional.arm.isin(PARITY_ARMS)]
    check_locations("03_uni", conventional, set20, qc)
    stain = pd.read_csv(BENCH / "01_stain_v4_mu_convergence/uni/location_metrics.csv.gz", dtype={"slide_id": str})
    check_locations("01_stain_v4 uni", stain, set20, qc)
    parts = [conventional, stain[stain.arm.isin(["macenko", "vahadane"])]]
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.rename(columns={"arm": "method"})
    frame["uni_distance"] = 1.0 - frame["generated_target_cosine"]
    frame = frame[["slide_id", "location_index", "scanner", "method", "uni_distance"]]

    raw_a = conventional[conventional.arm.eq("raw")].set_index(["slide_id", "location_index", "scanner"])
    raw_stain = stain[stain.arm.eq("raw")].set_index(["slide_id", "location_index", "scanner"])
    qc["uni_raw_distance_parity_max_abs_diff"] = {
        "03_uni_vs_01_stain_v4_uni": float(np.abs(
            (1 - raw_a.generated_target_cosine) - (1 - raw_stain.generated_target_cosine.reindex(raw_a.index))).max()),
    }
    gan_b = pd.concat([gan_uni_path_b("pix2pix"), gan_uni_path_b("cyclegan")], ignore_index=True)
    check_locations("GAN 04_uni (path B)", gan_b[gan_b.method.eq("pix2pix")], set40, qc)
    check_locations("GAN 04_uni (path B) cyclegan", gan_b[gan_b.method.eq("cyclegan")], set40, qc)
    b_raw = gan_b[gan_b.method.eq("pix2pix")].set_index(["slide_id", "location_index", "scanner"])
    b_raw = b_raw.reindex(raw_a.index)
    qc["uni_raw_distance_parity_max_abs_diff"]["03_uni_vs_gan_04_uni_path_b_at_set20"] = float(np.abs(
        (1 - raw_a.generated_target_cosine) - b_raw.raw_to_target_distance).max())

    review = pd.concat([pd.read_csv(p, dtype={"slide_id": str}) for p in
                        sorted((GAN_REVIEW / "shards/uni_v1").glob("fold_*.csv.gz"))], ignore_index=True)
    check_locations("gan_encoder_review uni_v1 (path C, published)", review, set20, qc)
    c_raw = review.set_index(["slide_id", "location_index", "scanner"]).reindex(raw_a.index)
    delta = (1 - raw_a.generated_target_cosine) - c_raw.raw_distance
    qc["uni_raw_distance_parity_max_abs_diff"]["03_uni_vs_gan_review_path_c"] = float(delta.abs().max())
    qc["uni_raw_distance_parity_max_abs_diff"]["03_uni_minus_gan_review_path_c_mean"] = float(delta.mean())

    gan_b["uni_distance"] = gan_b["method_to_target_distance"]
    raw_b = gan_b[gan_b.method.eq("pix2pix")].assign(method="raw", uni_distance=lambda x: x.raw_to_target_distance)
    gan40 = pd.concat([gan_b, raw_b], ignore_index=True)[
        ["slide_id", "location_index", "scanner", "method", "uni_distance"]].copy()
    gan20 = gan40[gan40.method.isin(["pix2pix", "cyclegan"])]
    gan20 = gan20[[int(l) in set20[s] for s, l in zip(gan20.slide_id, gan20.location_index)]]
    frame = pd.concat([frame, gan20], ignore_index=True)
    review_long = pd.concat([
        review.assign(method=m, uni_distance_path_c=review[f"{m}_distance"])[
            ["slide_id", "location_index", "scanner", "method", "uni_distance_path_c"]]
        for m in ("pix2pix", "cyclegan")] + [
        review.drop_duplicates(["slide_id", "location_index", "scanner"]).assign(
            method="raw", uni_distance_path_c=lambda x: x.raw_distance)[
            ["slide_id", "location_index", "scanner", "method", "uni_distance_path_c"]]],
        ignore_index=True)
    return frame, gan40.merge(review_long, how="outer", on=["slide_id", "location_index", "scanner", "method"])


def slide_scanner_long(location: pd.DataFrame, metrics: tuple[str, ...], per: int) -> pd.DataFrame:
    long = location.melt(id_vars=["slide_id", "location_index", "scanner", "method"],
                         value_vars=list(metrics), var_name="metric", value_name="value")
    long = long.dropna(subset=["value"])
    grouped = long.groupby(["method", "scanner", "slide_id", "metric"])["value"]
    counts = grouped.size()
    if not (counts == per).all():
        bad = counts[counts != per]
        raise ValueError(f"expected {per} locations per slide x scanner, found {bad.head()}")
    return grouped.mean().reset_index()


# ------------------------------------------------------------ augmentation

def oracle_classifier(selected: pd.DataFrame, train_locations: dict, test_locations: dict) -> pd.DataFrame:
    from augmentation_ood_study import BOOTSTRAP_SEED, ENDPOINTS
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    generated = [f"generated_{e}" for e in ENDPOINTS]
    target = [f"target_{e}" for e in ENDPOINTS]
    in_train = np.asarray([int(l) in train_locations[s] for s, l in zip(selected.slide_id, selected.location_index)])
    in_test = np.asarray([int(l) in test_locations[s] for s, l in zip(selected.slide_id, selected.location_index)])
    rows = []
    for arm in ORACLE_ARMS:
        for scanner in SCANNERS:
            mask = (selected.arm.eq(arm) & selected.scanner.eq(scanner)).to_numpy()
            for fold in range(5):
                train = selected[mask & in_train & (selected.fold.to_numpy() != fold)]
                test = selected[mask & in_test & (selected.fold.to_numpy() == fold)]
                x_train = np.concatenate([train[generated].to_numpy(float), train[target].to_numpy(float)])
                y_train = np.concatenate([np.zeros(len(train), int), np.ones(len(train), int)])
                x_test = np.concatenate([test[generated].to_numpy(float), test[target].to_numpy(float)])
                y_test = np.concatenate([np.zeros(len(test), int), np.ones(len(test), int)])
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=0.1, max_iter=2000, class_weight="balanced", random_state=BOOTSTRAP_SEED))
                model.fit(x_train, y_train)
                prediction = model.predict(x_test)
                rows.append(pd.DataFrame({
                    "arm": arm, "scanner": scanner, "fold": fold,
                    "slide_id": np.concatenate([test.slide_id.to_numpy(), test.slide_id.to_numpy()]),
                    "location_index": np.concatenate([test.location_index.to_numpy(), test.location_index.to_numpy()]),
                    "truth": y_test, "prediction": prediction,
                    "correct": (prediction == y_test).astype(int),
                }))
    return pd.concat(rows, ignore_index=True)


def classifier_accuracy(predictions: pd.DataFrame, arm: str, scanner: str = "all") -> pd.Series:
    """Published estimator: per-slide accuracy (balanced by design), averaged over slides."""
    subset = predictions[predictions.arm.eq(arm)]
    if scanner != "all":
        subset = subset[subset.scanner.eq(scanner)]
    return subset.groupby("slide_id")["correct"].mean()


def augmentation(set40: dict, set20: dict, boot: Bootstrap, qc: dict) -> tuple[pd.DataFrame, dict]:
    selected = pd.read_csv(AUGMENTATION / "02_aggregate/selected_candidates.csv", dtype={"slide_id": str})
    check_locations("augmentation selected_candidates", selected, set40, qc)
    folds = pd.read_csv(FOLDS, dtype={"slide_id": str}).set_index("slide_id").fold
    qc["augmentation_folds_equal_locked"] = bool(
        (selected.drop_duplicates("slide_id").set_index("slide_id").fold.sort_index()
         == folds.reindex(sorted(set(selected.slide_id)))).all() and len(folds) == 103)
    frozen = pd.read_csv(AUGMENTATION / "02_aggregate/classifier_predictions.csv", dtype={"slide_id": str})
    in20 = np.asarray([int(l) in set20[s] for s, l in zip(selected.slide_id, selected.location_index)])
    rows = []
    for sample, frame in (("set40", selected), ("set20", selected[in20])):
        for arm in ORACLE_ARMS:
            for scanner in ("all",) + SCANNERS:
                sub = frame[frame.arm.eq(arm)]
                if scanner != "all":
                    sub = sub[sub.scanner.eq(scanner)]
                for column, name in (("distance", "image_residual"),
                                     ("endpoints_within_one_sd_fraction", "coverage"),
                                     ("all_endpoints_within_one_sd", "joint_coverage")):
                    values = sub.groupby("slide_id")[column].mean()
                    low, high = boot.ci(values)
                    rows.append({"sample": sample, "arm": arm, "scanner": scanner, "metric": name,
                                 "value": float(values.mean()), "pooled_location_mean": float(sub[column].astype(float).mean()),
                                 "ci_low": low, "ci_high": high})
    # Classifier: parity refit on 40/40, protocol rule (train set40 / test set20) and refit on set20.
    refit40 = oracle_classifier(selected, set40, set40)
    merged = refit40.merge(frozen, on=["arm", "scanner", "fold", "slide_id", "location_index", "truth"],
                           suffixes=("_refit", "_frozen"), validate="one_to_one")
    qc["oracle_classifier_refit_matches_frozen_predictions"] = {
        "rows": int(len(merged)), "frozen_rows": int(len(frozen)),
        "prediction_agreement": float((merged.prediction_refit == merged.prediction_frozen).mean()),
    }
    train40_test20 = frozen[[int(l) in set20[s] for s, l in zip(frozen.slide_id, frozen.location_index)]]
    refit20 = oracle_classifier(selected, set20, set20)
    for sample, predictions in (("set40_train_set40_test (published design)", frozen),
                                ("set40_train_set20_test (protocol rule 1)", train40_test20),
                                ("set20_train_set20_test (refit)", refit20)):
        for arm in ORACLE_ARMS:
            for scanner in ("all",) + SCANNERS:
                values = classifier_accuracy(predictions, arm, scanner)
                low, high = boot.ci(values)
                rows.append({"sample": sample, "arm": arm, "scanner": scanner,
                             "metric": "real_vs_oracle_balanced_accuracy", "value": float(values.mean()),
                             "pooled_location_mean": float(predictions[predictions.arm.eq(arm) & (
                                 predictions.scanner.eq(scanner) if scanner != "all"
                                 else pd.Series(True, index=predictions.index))].correct.mean()),
                             "ci_low": low, "ci_high": high})
    write_frame(OUTPUT / "augmentation_oracle_classifier_predictions_set20_refit.csv.gz", refit20)

    # Modal oracle candidate shares (supplementary text: 49.2% AKOYA, 35.4% GT450).
    modal = {}
    strong = selected[selected.arm.eq("strong_oracle")]
    for scanner in ("akoya", "gt450"):
        sub40 = strong[strong.scanner.eq(scanner)]
        counts40 = sub40.groupby(["library", "candidate_index"]).size()
        top = counts40.idxmax()
        sub20 = sub40[[int(l) in set20[s] for s, l in zip(sub40.slide_id, sub40.location_index)]]
        counts20 = sub20.groupby(["library", "candidate_index"]).size()
        modal[scanner] = {
            "set40_modal_candidate": [str(top[0]), int(top[1])],
            "set40_share": float(counts40.max() / len(sub40)),
            "set20_share_of_same_candidate": float(counts20.get(top, 0) / len(sub20)),
            "set20_modal_candidate": [str(counts20.idxmax()[0]), int(counts20.idxmax()[1])],
            "set20_locations": int(len(sub20)),
        }

    # Oracle table: SSIM/LPIPS (paired shards) and UNI distance change.
    paired = pd.concat([pd.read_csv(AUG_PAIRED / "shards" / f"{s}.csv.gz", dtype={"slide_id": str})
                        for s in sorted(set40)], ignore_index=True)
    check_locations("augmentation paired SSIM/LPIPS shards", paired, set40, qc)
    uni = pd.read_csv(AUGMENTATION / "03_pfm_uni/location_metrics.csv", dtype={"slide_id": str})
    check_locations("augmentation 03_pfm_uni", uni, set20, qc)
    for sample, frame in (("set40", paired), ("set20", paired[[int(l) in set20[s] for s, l in
                                                              zip(paired.slide_id, paired.location_index)]])):
        for scanner in ("all",) + SCANNERS:
            sub = frame if scanner == "all" else frame[frame.scanner.eq(scanner)]
            slide = sub.groupby(["slide_id", "arm"])[["ssim", "lpips_vgg"]].mean()
            for metric in ("ssim", "lpips_vgg"):
                raw = slide.xs("identity", level="arm")[metric]
                for arm in ("identity", "default_oracle", "strong_oracle", "strong_unconstrained_oracle"):
                    values = slide.xs(arm, level="arm")[metric]
                    rows.append({"sample": sample, "arm": arm, "scanner": scanner, "metric": f"{metric}_improvement_pct",
                                 "value": signed_pct(metric, float(values.mean()), float(raw.mean())),
                                 "pooled_location_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan})
    for scanner in ("all",) + SCANNERS:
        sub = uni if scanner == "all" else uni[uni.scanner.eq(scanner)]
        raw_distance = (1 - sub[sub.arm.eq("identity")].groupby("slide_id").generated_target_cosine.mean()).mean()
        for arm in ("identity", "default_oracle", "strong_oracle", "strong_unconstrained_oracle"):
            gain = sub[sub.arm.eq(arm)].groupby("slide_id").target_gain.mean().mean()
            rows.append({"sample": "set20", "arm": arm, "scanner": scanner, "metric": "uni_distance_reduction_pct",
                         "value": float(100 * gain / raw_distance), "pooled_location_mean": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan})
    return pd.DataFrame(rows), modal


# ------------------------------------------------------------ published numbers

def published_numbers() -> list[dict]:
    """Every published number within scope, with the file and printed text it comes from."""
    t2 = "tables/table_image_correction_uni.tex"
    cf = "tables/table_color_frequency_correction_full.tex"
    ora = "tables/table_augmentation_oracle.tex"
    main = "pannormal_scanner_batch_effects.tex"
    items = []

    def add(key, source, text, value, decimals, method, scanner, metric, statistic, n_published, note=""):
        items.append(dict(key=key, source_file=source, published_text=text, published_value=value,
                          decimals=decimals, method=method, scanner=scanner, metric=metric,
                          statistic=statistic, published_locations=n_published, note=note))

    table2 = {
        "raw": (("0.572", "0.074", None), ("0.232", "0.026", None), ("0.2136", "0.0365", None)),
        "reinhard": (("0.620", "0.071", "+8.4"), ("0.202", "0.025", "+12.8"), ("0.1798", "0.0334", "+15.9")),
        "macenko": (("0.552", "0.087", "$-3.5$"), ("0.244", "0.041", "$-4.9$"), ("0.3403", "0.0728", "$-59.3$")),
        "vahadane": (("0.555", "0.092", "$-3.0$"), ("0.239", "0.050", "$-3.0$"), ("0.3152", "0.0841", "$-47.5$")),
        "pix2pix": (("0.682", "0.056", "+19.2"), ("0.218", "0.027", "+5.9"), ("0.2704", "0.0617", "$-26.6$")),
        "cyclegan": (("0.658", "0.051", "+14.9"), ("0.206", "0.025", "+11.1"), ("0.2015", "0.0465", "+5.7")),
        "combined": (("0.626", "0.069", "+9.5"), ("0.192", "0.024", "+17.2"), ("0.1794", "0.0316", "+16.0")),
    }
    for method, cells in table2.items():
        for metric, (mean, sd, pct) in zip(("ssim", "lpips_vgg", "uni_distance"), cells):
            n = 20
            note = ""
            if metric == "uni_distance" and method in ("pix2pix", "cyclegan"):
                note = "published value is the 20-location re-embedding in gan_encoder_review (extraction path C)"
            add(f"table2.{method}.{metric}.mean", t2, f"{mean}\\pm{sd}", float(mean), len(mean.split(".")[1]),
                method, "all", metric, "mean", n, note)
            add(f"table2.{method}.{metric}.sd", t2, f"{mean}\\pm{sd}", float(sd), len(sd.split(".")[1]),
                method, "all", metric, "sd", n, note)
            if pct:
                add(f"table2.{method}.{metric}.pct", t2, pct, float(pct.strip("$").replace("+", "")), 1,
                    method, "all", metric, "pct_improvement_vs_raw", n, note)
    for metric, mean, sd, pct in (("ssim", "0.597", "0.071", "+4.2"), ("lpips_vgg", "0.218", "0.026", "+5.9"),
                                  ("uni_distance", "0.2049", "0.0343", "+4.1")):
        add(f"stable_cf.frequency.{metric}.mean", cf, f"{mean}\\pm{sd}", float(mean), len(mean.split(".")[1]),
            "frequency", "all", metric, "mean", 20)
        add(f"stable_cf.frequency.{metric}.sd", cf, f"{mean}\\pm{sd}", float(sd), len(sd.split(".")[1]),
            "frequency", "all", metric, "sd", 20)
        add(f"stable_cf.frequency.{metric}.pct", cf, pct, float(pct.replace("+", "")), 1,
            "frequency", "all", metric, "pct_improvement_vs_raw", 20)
    for arm, cells in (("default_oracle", ("$+8.1$", "$+7.7$", "$-1.1$")),
                       ("strong_oracle", ("$+9.1$", "$+7.0$", "$-3.9$")),
                       ("strong_unconstrained_oracle", ("$+11.3$", "$+8.6$", "$-2.2$"))):
        for metric, text in zip(("ssim_improvement_pct", "lpips_vgg_improvement_pct", "uni_distance_reduction_pct"), cells):
            add(f"oracle_table.{arm}.{metric}", ora, text, float(text.strip("$").replace("+", "")), 1,
                arm, "all", metric, "pct", 20)
    text_items = [
        ("aug.identity.image_residual", "1.311 for raw images", 1.311, 3, "identity", "all", "image_residual", "mean", 40),
        ("aug.strong_oracle.image_residual", "to 0.756", 0.756, 3, "strong_oracle", "all", "image_residual", "mean", 40),
        ("aug.strong_oracle.joint_coverage", "49.1\\%", 49.1, 1, "strong_oracle", "all", "joint_coverage", "percent", 40),
        ("aug.strong_oracle.classifier", "93.9\\%", 93.9, 1, "strong_oracle", "all",
         "real_vs_oracle_balanced_accuracy", "percent", 40),
        ("aug.ssim_gain_65_to_257", "1.0 percentage point", 1.0, 1, "strong_oracle-default_oracle", "all",
         "ssim_improvement_pct", "difference", 20),
        ("aug.modal_share.akoya", "49.2\\%", 49.2, 1, "strong_oracle", "akoya", "modal_candidate_share", "percent", 40),
        ("aug.modal_share.gt450", "35.4\\%", 35.4, 1, "strong_oracle", "gt450", "modal_candidate_share", "percent", 40),
        ("cf.reinhard.image_residual", "from 0.698", 0.698, 3, "reinhard", "all", "image_residual", "mean", 40),
        ("cf.combined.image_residual", "to 0.499", 0.499, 3, "combined", "all", "image_residual", "mean", 40),
        ("cf.paired_reduction", "paired reduction 0.199", 0.199, 3, "combined-reinhard", "all",
         "image_residual", "paired_reduction", 40),
        ("cf.paired_reduction.ci_low", "95\\% CI 0.191--0.207", 0.191, 3, "combined-reinhard", "all",
         "image_residual", "paired_reduction_ci_low", 40),
        ("cf.paired_reduction.ci_high", "95\\% CI 0.191--0.207", 0.207, 3, "combined-reinhard", "all",
         "image_residual", "paired_reduction_ci_high", 40),
        ("cf.akoya.reinhard.image_residual", "from 1.762", 1.762, 3, "reinhard", "akoya", "image_residual", "mean", 40),
        ("cf.akoya.combined.image_residual", "to 0.927", 0.927, 3, "combined", "akoya", "image_residual", "mean", 40),
        ("cf.gt450.reinhard.image_residual", "from 0.415", 0.415, 3, "reinhard", "gt450", "image_residual", "mean", 40),
        ("cf.gt450.combined.image_residual", "to 0.301", 0.301, 3, "combined", "gt450", "image_residual", "mean", 40),
        ("cf.akoya.reinhard.target_gradient_ncc", "from 0.579", 0.579, 3, "reinhard", "akoya", "target_gradient_ncc", "mean", 40),
        ("cf.akoya.combined.target_gradient_ncc", "to 0.633", 0.633, 3, "combined", "akoya", "target_gradient_ncc", "mean", 40),
        ("cf.akoya.reinhard.source_gradient_ncc", "from 0.851", 0.851, 3, "reinhard", "akoya", "source_gradient_ncc", "mean", 40),
        ("cf.akoya.combined.source_gradient_ncc", "to 0.812", 0.812, 3, "combined", "akoya", "source_gradient_ncc", "mean", 40),
        ("cf.uni_gain.all", "nearly zero at 0.0004", 0.0004, 4, "combined-reinhard", "all", "uni_target_gain", "mean", 20),
        ("cf.uni_gain.akoya", "$+0.0167$", 0.0167, 4, "combined-reinhard", "akoya", "uni_target_gain", "mean", 20),
        ("cf.uni_gain.gt450", "$-0.0153$", -0.0153, 4, "combined-reinhard", "gt450", "uni_target_gain", "mean", 20),
        ("gan.gt450.pix2pix.ssim", "SSIM from 0.693", 0.693, 3, "pix2pix", "gt450", "ssim", "mean", 40),
        ("gan.gt450.pix2pix_edge.ssim", "to 0.711", 0.711, 3, "pix2pix_edge", "gt450", "ssim", "mean", 40),
        ("gan.gt450.pix2pix.source_gradient_ncc_gray", "NCC from 0.623", 0.623, 3, "pix2pix", "gt450",
         "source_gradient_ncc_gray", "mean", 40),
        ("gan.gt450.pix2pix_edge.source_gradient_ncc_gray", "to 0.650", 0.650, 3, "pix2pix_edge", "gt450",
         "source_gradient_ncc_gray", "mean", 40),
        ("gan.gt450.pix2pix.target_gradient_ncc_gray", "NCC from 0.456", 0.456, 3, "pix2pix", "gt450",
         "target_gradient_ncc_gray", "mean", 40),
        ("gan.gt450.pix2pix_edge.target_gradient_ncc_gray", "to 0.478", 0.478, 3, "pix2pix_edge", "gt450",
         "target_gradient_ncc_gray", "mean", 40),
        ("gan.gt450.pix2pix.invented_edge_fraction", "from 0.0054", 0.0054, 4, "pix2pix", "gt450",
         "invented_edge_fraction", "mean", 40),
        ("gan.gt450.pix2pix_edge.invented_edge_fraction", "to 0.0042", 0.0042, 4, "pix2pix_edge", "gt450",
         "invented_edge_fraction", "mean", 40),
        ("gan.gt450.pix2pix.uni_gain", "from $-0.0420$", -0.0420, 4, "pix2pix", "gt450", "uni_target_gain", "mean", 40),
        ("gan.gt450.pix2pix_edge.uni_gain", "to $-0.0484$", -0.0484, 4, "pix2pix_edge", "gt450", "uni_target_gain", "mean", 40),
        ("gan.gt450.edge_minus_base.uni_gain", "paired difference $-0.0065$", -0.0065, 4, "pix2pix_edge-pix2pix",
         "gt450", "uni_target_gain", "paired_difference", 40),
        ("gan.gt450.edge_minus_base.uni_gain.ci_low", "$-0.0170$--$+0.0042$", -0.0170, 4, "pix2pix_edge-pix2pix",
         "gt450", "uni_target_gain", "paired_difference_ci_low", 40),
        ("gan.gt450.edge_minus_base.uni_gain.ci_high", "$-0.0170$--$+0.0042$", 0.0042, 4, "pix2pix_edge-pix2pix",
         "gt450", "uni_target_gain", "paired_difference_ci_high", 40),
    ]
    for key, text, value, decimals, method, scanner, metric, statistic, n in text_items:
        add(key, main, text, value, decimals, method, scanner, metric, statistic, n)
    # Guard: every printed string must occur in its source file.
    for item in items:
        source = (MANUSCRIPT / item["source_file"]).read_text()
        if item["published_text"] not in source.replace(" ", "") and item["published_text"] not in source:
            raise ValueError(f"published text not found: {item['key']}: {item['published_text']}")
    return items


# ------------------------------------------------------------ aggregate

def aggregate() -> None:
    qc: dict = {"location_sets": {}}
    cohort, set40_frame, set20_frame = load_sets()
    set40, set20 = location_sets(set40_frame), location_sets(set20_frame)
    slides = sorted(cohort.slide_id)
    qc["set20_subset_of_set40"] = all(set20[s] <= set40[s] for s in slides)
    qc["slides"] = len(slides)
    qc["set20_locations"] = int(sum(len(v) for v in set20.values()))
    if qc["set20_locations"] != 2060 or not qc["set20_subset_of_set40"]:
        raise ValueError("locked location sets are inconsistent")
    boot = Bootstrap(slides)

    phenotype20, phenotype40 = phenotype_frames(set40, set20, qc)
    bench = benchmark_frame(set20, qc)
    uni20, gan_uni = uni_frames(set40, set20, qc)

    location = phenotype20.merge(bench, on=["slide_id", "location_index", "scanner", "method"], how="outer",
                                 validate="one_to_one")
    location = location.merge(uni20, on=["slide_id", "location_index", "scanner", "method"], how="outer",
                              validate="one_to_one")
    if len(location) != 2060 * 5 * len(METHODS):
        raise ValueError(f"location table has {len(location)} rows")
    qc["location_table_nan_counts"] = {m: int(location[m].isna().sum()) for m in METRICS}
    if any(qc["location_table_nan_counts"].values()):
        raise ValueError(f"missing location metrics: {qc['location_table_nan_counts']}")
    write_frame(OUTPUT / "image_metrics_location.csv.gz", location)

    slide20 = slide_scanner_long(location, METRICS, 20)
    slide20.insert(0, "dataset", "PanNormal")
    long = slide20[["dataset", "method", "scanner", "slide_id", "metric", "value"]].sort_values(
        ["method", "scanner", "slide_id", "metric"])
    if len(long) != len(METHODS) * 5 * 103 * len(METRICS) or long.value.isna().any():
        raise ValueError("slide-level long table incomplete")
    write_frame(OUTPUT / "image_metrics_slide.csv", long)
    extra_metrics = ("target_gradient_ncc_gray", "source_gradient_ncc_gray", "invented_edge_fraction",
                     "source_edge_deletion_fraction")
    extra = slide_scanner_long(location, extra_metrics, 20)
    extra.insert(0, "dataset", "PanNormal")
    write_frame(OUTPUT / "image_metrics_slide_supplementary.csv",
                extra[["dataset", "method", "scanner", "slide_id", "metric", "value"]])

    summary20 = summarize(slide20, boot, "set20")
    # set40 counterparts where a 40-location source exists (phenotype metrics, GAN UNI path B).
    slide40_pheno = slide_scanner_long(phenotype40, tuple(PHENOTYPE_COLUMNS.values()), 40)
    gan40 = gan_uni.dropna(subset=["uni_distance"])
    slide40_gan_uni = slide_scanner_long(gan40, ("uni_distance",), 40)
    summary40 = pd.concat([summarize(slide40_pheno, boot, "set40"),
                           summarize(slide40_gan_uni, boot, "set40")], ignore_index=True)
    path_c = gan_uni.dropna(subset=["uni_distance_path_c"]).rename(columns={"uni_distance_path_c": "uni_distance_path_c"})
    slide20_c = slide_scanner_long(path_c, ("uni_distance_path_c",), 20)
    summary_c = summarize(slide20_c, boot, "set20_path_c")
    summaries = pd.concat([summary20, summary40, summary_c], ignore_index=True)
    write_frame(OUTPUT / "summary_methods.csv", summaries)
    extra_summary = summarize(extra, boot, "set20")
    write_frame(OUTPUT / "summary_methods_supplementary.csv", extra_summary)

    # Colour-frequency paired contrasts (same seed scheme as 02_correction for the 40-location parity).
    from augmentation_ood_study import stable_seed
    contrasts = {}
    for sample, frame in (("set40", phenotype40), ("set20", phenotype20)):
        pivot = frame[frame.method.isin(["reinhard", "combined"])].pivot_table(
            index="slide_id", columns="method", values="image_residual", aggfunc="mean")
        values = (pivot["combined"] - pivot["reinhard"]).to_numpy(float)
        rng = np.random.default_rng(stable_seed("combined", "reinhard", "distance"))
        draws = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
        contrasts[sample] = {"combined_minus_reinhard_image_residual": float(values.mean()),
                             "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975))}
    uni_conv = pd.read_csv(BATCH / "03_uni/location_metrics.csv", dtype={"slide_id": str})
    uni_gain = {}
    for scanner in ("all",) + SCANNERS:
        sub = uni_conv if scanner == "all" else uni_conv[uni_conv.scanner.eq(scanner)]
        pivot = sub[sub.arm.isin(["reinhard", "combined"])].pivot_table(
            index="slide_id", columns="arm", values="target_gain", aggfunc="mean")
        difference = pivot["combined"] - pivot["reinhard"]
        low, high = boot.ci(difference)
        uni_gain[scanner] = {"mean": float(difference.mean()), "ci_low": low, "ci_high": high}

    # GT450 Pix2Pix edge-constraint numbers (original and constrained models).
    edge = {}
    for sample, keep in (("set40", set40), ("set20", set20)):
        values = {}
        for label, image_path, uni_path in (
                ("pix2pix", LEARNED / "09_bidirectional_full_training/03_image_evaluation/at2_to_gt450/location_metrics.csv.gz",
                 LEARNED / "09_bidirectional_full_training/04_uni/at2_to_gt450/location_metrics.csv.gz"),
                ("pix2pix_edge", ABLATION / "03_image_evaluation/at2_to_gt450/location_metrics.csv.gz",
                 ABLATION / "04_uni/at2_to_gt450/location_metrics.csv.gz")):
            image = pd.read_csv(image_path, dtype={"slide_id": str})
            uni = pd.read_csv(uni_path, dtype={"slide_id": str})
            check_locations(f"edge ablation image {label}", image, set40, qc)
            check_locations(f"edge ablation uni {label}", uni, set40, qc)
            image = image[[int(l) in keep[s] for s, l in zip(image.slide_id, image.location_index)]]
            uni = uni[[int(l) in keep[s] for s, l in zip(uni.slide_id, uni.location_index)]]
            per_slide = image.groupby("slide_id")[["target_ssim", "source_gradient_ncc", "target_gradient_ncc",
                                                   "invented_edge_fraction"]].mean()
            per_slide["uni_gain"] = uni.groupby("slide_id").gain_to_target.mean()
            values[label] = per_slide
        difference = values["pix2pix_edge"]["uni_gain"] - values["pix2pix"]["uni_gain"]
        low, high = boot.ci(difference)
        edge[sample] = {label: frame.mean().to_dict() for label, frame in values.items()}
        edge[sample]["edge_minus_base_uni_gain"] = {"mean": float(difference.mean()), "ci_low": low, "ci_high": high}
    # The table 2 shards and the GAN image evaluation should give identical SSIM at set20.
    pix_eval = pd.read_csv(LEARNED / "09_bidirectional_full_training/03_image_evaluation/at2_to_gt450/location_metrics.csv.gz",
                           dtype={"slide_id": str})
    check = bench[bench.method.eq("pix2pix") & bench.scanner.eq("gt450")].merge(
        pix_eval, on=["slide_id", "location_index"], validate="one_to_one")
    qc["table2_shard_vs_gan_image_evaluation_ssim_max_abs_diff_gt450_pix2pix"] = float(
        np.abs(check.ssim - check.target_ssim).max())

    aug, modal = augmentation(set40, set20, boot, qc)
    write_frame(OUTPUT / "augmentation_oracle_summary.csv", aug)

    # ---------------------------------------------------------------- comparison
    def summary_value(sample, method, metric, scanner, statistic):
        row = summaries[summaries["sample"].eq(sample) & summaries.method.eq(method) &
                        summaries.metric.eq(metric) & summaries.scanner.eq(scanner)]
        if len(row) != 1:
            return np.nan
        column = {"mean": "mean", "sd": "sd", "pct_improvement_vs_raw": "pct_improvement_vs_raw"}[statistic]
        return float(row.iloc[0][column])

    def aug_value(sample, arm, metric, scanner="all"):
        row = aug[aug["sample"].eq(sample) & aug.arm.eq(arm) & aug.metric.eq(metric) & aug.scanner.eq(scanner)]
        return float(row.iloc[0].value) if len(row) == 1 else np.nan

    gan_c_mean = {}
    for method in ("pix2pix", "cyclegan"):
        for statistic in ("mean", "sd"):
            gan_c_mean[(method, statistic)] = summary_value("set20_path_c", method, "uni_distance_path_c", "all", statistic)
    raw_a = summary_value("set20", "raw", "uni_distance", "all", "mean")

    rows = []
    for item in published_numbers():
        key, method, metric, scanner, statistic = (item["key"], item["method"], item["metric"],
                                                   item["scanner"], item["statistic"])
        parity = set20_value = np.nan
        note = item["note"]
        if key.startswith(("table2.", "stable_cf.")):
            set20_value = summary_value("set20", method, metric, scanner, statistic)
            parity = set20_value
            if metric == "uni_distance" and method in ("pix2pix", "cyclegan"):
                if statistic == "pct_improvement_vs_raw":
                    parity = 100 * (raw_a - gan_c_mean[(method, "mean")]) / raw_a
                else:
                    parity = gan_c_mean[(method, statistic)]
                note += ("; set20 value uses extraction path B (feature_shards of the 40-location GAN UNI "
                         "evaluation restricted to set20), consistent with the raw/conventional UNI extraction; "
                         f"set40 path-B value {summary_value('set40', method, metric, scanner, statistic):.4f}")
        elif key.startswith("oracle_table."):
            set20_value = aug_value("set20", method, metric)
            parity = set20_value
        elif key.startswith("aug."):
            if key == "aug.ssim_gain_65_to_257":
                parity = aug_value("set20", "strong_oracle", metric) - aug_value("set20", "default_oracle", metric)
                set20_value = parity
                note = "published sample for SSIM is the 20 UNI locations (= set20)"
            elif key.startswith("aug.modal_share"):
                parity = 100 * modal[scanner]["set40_share"]
                set20_value = 100 * modal[scanner]["set20_share_of_same_candidate"]
                note = (f"share of candidate {modal[scanner]['set40_modal_candidate']}; set20 modal candidate "
                        f"{modal[scanner]['set20_modal_candidate']}")
            elif metric == "real_vs_oracle_balanced_accuracy":
                parity = 100 * aug_value("set40_train_set40_test (published design)", method, metric)
                set20_value = 100 * aug_value("set40_train_set20_test (protocol rule 1)", method, metric)
                note = ("set20 value: classifier trained on set40 of training folds, evaluated on set20 of held-out "
                        "slides (protocol rule 1); refit on set20 only = "
                        f"{100 * aug_value('set20_train_set20_test (refit)', method, metric):.2f}")
            else:
                scale = 100.0 if statistic == "percent" else 1.0
                parity = scale * aug_value("set40", method, metric)
                set20_value = scale * aug_value("set20", method, metric)
        elif key.startswith("cf."):
            if key.startswith("cf.paired_reduction"):
                field = {"paired_reduction": "combined_minus_reinhard_image_residual",
                         "paired_reduction_ci_low": "ci_high", "paired_reduction_ci_high": "ci_low"}[statistic]
                parity = -contrasts["set40"][field]
                set20_value = -contrasts["set20"][field]
            elif metric == "uni_target_gain":
                parity = set20_value = uni_gain[scanner]["mean"]
            else:
                parity = summary_value("set40", method, metric, scanner, "mean")
                set20_value = summary_value("set20", method, metric, scanner, "mean")
        elif key.startswith("gan."):
            if key.startswith("gan.gt450.edge_minus_base"):
                field = {"paired_difference": "mean", "paired_difference_ci_low": "ci_low",
                         "paired_difference_ci_high": "ci_high"}[statistic]
                parity = edge["set40"]["edge_minus_base_uni_gain"][field]
                set20_value = edge["set20"]["edge_minus_base_uni_gain"][field]
                if statistic != "paired_difference":
                    note = "bootstrap CI: published seed/replicates unknown; this uses 2,000 slide resamples"
            else:
                column = {"ssim": "target_ssim", "source_gradient_ncc_gray": "source_gradient_ncc",
                          "target_gradient_ncc_gray": "target_gradient_ncc",
                          "invented_edge_fraction": "invented_edge_fraction", "uni_target_gain": "uni_gain"}[metric]
                parity = edge["set40"][method][column]
                set20_value = edge["set20"][method][column]
        decimals = int(item["decimals"])
        published = float(item["published_value"])
        rows.append({
            **{k: item[k] for k in ("key", "source_file", "published_text", "method", "scanner", "metric",
                                    "statistic", "published_locations")},
            "published_value": published, "decimals": decimals,
            "recomputed_on_published_sample": parity,
            "parity_ok": bool(np.isfinite(parity) and round(parity, decimals) == round(published, decimals)),
            "set20_value": set20_value,
            "difference_set20_minus_published": set20_value - published,
            "set20_rounded": round(set20_value, decimals) if np.isfinite(set20_value) else np.nan,
            "difference_in_printed_units": (set20_value - published) * 10 ** decimals,
            "changes_at_printed_precision": bool(round(set20_value, decimals) != round(published, decimals)),
            "note": note,
        })
    comparison = pd.DataFrame(rows)
    write_frame(OUTPUT / "comparison.csv", comparison)

    # ---------------------------------------------------------------- sign and ranking checks
    checks = []
    for metric in ("ssim", "lpips_vgg", "uni_distance"):
        block = summaries[summaries["sample"].eq("set20") & summaries.metric.eq(metric) &
                          summaries.scanner.eq("all") & summaries.method.isin(TABLE2_METHODS)].set_index("method")
        published = comparison[comparison.key.str.startswith("table2.") & comparison.metric.eq(metric) &
                               comparison.statistic.eq("mean")].set_index("method").published_value
        ascending = metric in LOWER_BETTER
        rank20 = block["mean"].rank(ascending=ascending).astype(int)
        rank_pub = published.rank(ascending=ascending, method="min").astype(int)
        checks.append({"check": f"table2 PanNormal ranking by {metric}",
                       "published": ",".join(rank_pub.sort_values().index),
                       "set20": ",".join(rank20.sort_values().index),
                       "same": bool((rank20.reindex(rank_pub.index) == rank_pub).all())})
        for method in TABLE2_METHODS[1:]:
            pct20 = float(block.loc[method, "pct_improvement_vs_raw"])
            pub = comparison[(comparison.key == f"table2.{method}.{metric}.pct")].published_value
            checks.append({"check": f"sign of {method} {metric} change vs raw",
                           "published": f"{float(pub.iloc[0]):+.1f}", "set20": f"{pct20:+.2f}",
                           "same": bool(np.sign(pct20) == np.sign(float(pub.iloc[0])))})
    # Scanner-level "improved together" sets. Reference: the frozen scanner summary behind the
    # paper (analysis/paper/table2_scanner_summary.csv; GAN UNI there is the 40-location extraction).
    frozen = pd.read_csv(PROJECT / "analysis/paper/table2_scanner_summary.csv")
    frozen = frozen[frozen.dataset.eq("PanNormal")]
    together = {}
    for method in ("pix2pix", "cyclegan", "macenko", "vahadane", "reinhard", "combined"):
        improved = []
        for scanner in SCANNERS:
            signs = [float(summaries[summaries["sample"].eq("set20") & summaries.method.eq(method) &
                                     summaries.metric.eq(m) & summaries.scanner.eq(scanner)].pct_improvement_vs_raw.iloc[0]) > 0
                     for m in ("ssim", "lpips_vgg", "uni_distance")]
            if all(signs):
                improved.append(scanner)
        block = frozen[frozen.method.eq(method)]
        reference = sorted(str(r.scanner).lower() for r in block.itertuples(index=False)
                           if r.target_ssim_improvement_pct > 0 and r.lpips_vgg_improvement_pct > 0
                           and r.uni_distance_improvement_pct > 0)
        reference = [s for s in SCANNERS if s in reference]
        together[method] = (reference, improved)
        checks.append({"check": f"{method}: scanners where SSIM, LPIPS and UNI all improve",
                       "published": ",".join(reference) or "none", "set20": ",".join(improved) or "none",
                       "same": bool(improved == reference)})
    shared_ref = [s for s in together["pix2pix"][0] if s in together["cyclegan"][0]]
    shared_20 = [s for s in together["pix2pix"][1] if s in together["cyclegan"][1]]
    checks.append({"check": "text: 'for both models, AT2->AKOYA was the direction in which image fidelity and UNI "
                            "distance improved together' (scanners where both GANs improve all three)",
                   "published": ",".join(shared_ref) or "none", "set20": ",".join(shared_20) or "none",
                   "same": bool(shared_ref == shared_20 == ["akoya"])})
    reinhard_combined = {m: float(summaries[summaries["sample"].eq("set20") & summaries.method.eq("combined") &
                                            summaries.metric.eq(m) & summaries.scanner.eq("all")]["mean"].iloc[0]) -
                         float(summaries[summaries["sample"].eq("set20") & summaries.method.eq("reinhard") &
                                         summaries.metric.eq(m) & summaries.scanner.eq("all")]["mean"].iloc[0])
                         for m in ("ssim", "lpips_vgg")}
    checks.append({"check": "combined improves SSIM and LPIPS further than Reinhard",
                   "published": "yes", "set20": json.dumps({k: round(v, 5) for k, v in reinhard_combined.items()}),
                   "same": bool(reinhard_combined["ssim"] > 0 and reinhard_combined["lpips_vgg"] < 0)})
    write_frame(OUTPUT / "sign_ranking_checks.csv", pd.DataFrame(checks))

    qc["comparison_rows"] = int(len(comparison))
    qc["comparison_parity_failures"] = comparison.loc[~comparison.parity_ok, "key"].tolist()
    qc["comparison_changes_at_printed_precision"] = comparison.loc[
        comparison.changes_at_printed_precision, "key"].tolist()
    qc["modal_candidates"] = modal
    qc["colour_frequency_paired_contrast"] = contrasts
    qc["uni_combined_minus_reinhard_gain"] = uni_gain
    qc["edge_constraint"] = {s: {k: (v if isinstance(v, dict) else v) for k, v in d.items()} for s, d in edge.items()}
    write_json(OUTPUT / "qc.json", qc)
    write_summary(comparison, pd.DataFrame(checks), summaries, aug, qc)
    print(json.dumps({k: qc[k] for k in ("location_sets", "comparison_parity_failures",
                                         "comparison_changes_at_printed_precision")}, indent=2))


def headline(comparison: pd.DataFrame, checks: pd.DataFrame, aug: pd.DataFrame) -> list[str]:
    changed = comparison[comparison.changes_at_printed_precision]
    units = comparison.difference_in_printed_units.abs()
    lines = [
        f"- Published numbers compared: {len(comparison)} (Table 2 PanNormal, the frequency row of the "
        "colour-frequency supplementary table, the oracle table, and the augmentation, colour-frequency and "
        "learned-translation text). Reproduced from stored data on their published sample: "
        f"{int(comparison.parity_ok.sum())}/{len(comparison)} (the two misses are bootstrap CI bounds whose "
        "original seed and replicate count are not recorded).",
        f"- Changed at printed precision on set20: {len(changed)}; differing by more than one printed unit: "
        f"{int((units > 1.0 + 1e-9).sum())}; largest difference {units.max():.1f} printed units "
        f"({comparison.loc[units.idxmax(), 'key']}).",
    ]
    for row in changed.itertuples(index=False):
        lines.append(f"  - {row.key}: {row.published_value:g} -> {row.set20_value:.{row.decimals}f} "
                     f"(published on {row.published_locations} locations)")
    signs = checks[checks.check.str.startswith("sign of")]
    ranks = checks[checks.check.str.contains("ranking")]
    rest = checks[~checks.index.isin(signs.index) & ~checks.index.isin(ranks.index)]
    lines += [
        f"- Sign of every Table 2 change from raw unchanged: {bool(signs.same.all())}; method rankings by SSIM, "
        f"LPIPS and UNI distance unchanged: {bool(ranks.same.all())}.",
        f"- Scanner-level and text claims unchanged: {int(rest.same.astype(bool).sum())}/{len(rest)} "
        "(see table below).",
        "- Pix2Pix/CycleGAN UNI distance: the published values already use set20 (gan_encoder_review "
        "re-embedding, extraction path C) and are reproduced exactly; the set20 values here use the "
        "40-location GAN extraction restricted to set20 (path B, same preprocessing as the raw/conventional "
        "UNI extraction; see RV-P0c), so their differences reflect extraction, not location set.",
    ]
    clf = aug[aug.metric.eq("real_vs_oracle_balanced_accuracy") & aug.arm.eq("strong_oracle") & aug.scanner.eq("all")]
    values = {r.sample: 100 * r.value for r in clf.itertuples(index=False)}
    lines.append("- Real-vs-oracle classifier: " + "; ".join(f"{k} {v:.2f}%" for k, v in values.items()) + ".")
    return lines


def write_summary(comparison: pd.DataFrame, checks: pd.DataFrame, summaries: pd.DataFrame,
                  aug: pd.DataFrame, qc: dict) -> None:
    def fmt(value, decimals):
        return "NA" if not np.isfinite(value) else f"{value:.{decimals + 1}f}"

    lines = [
        "# RV-P0b: re-aggregation of published image metrics on set20",
        "",
        "Generated by `analysis/revision/set20_reaggregation.py aggregate`. Protocol: "
        "`analysis/revision/README.md` (RV-P0b). PanNormal only; PLISM uses its own 2,387 locations and "
        "is unaffected by set20.",
        "",
        "## What ran",
        "",
        "- `gan-residual` (SLURM array, 103 tasks): ten-measure image residual, coverage, joint coverage and "
        "phenotype gradient NCCs for the stored Pix2Pix/CycleGAN predictions at set20 (new; never computed "
        "before), with raw/Reinhard/frequency/combined re-rendered as a parity check against 02_correction.",
        "- `aggregate`: location-level metrics for eight image methods restricted to set20; locations averaged "
        "within slide x scanner, five target scanners equally weighted within slide, mean and SD over 103 slides; "
        "percentage improvement 100(M-R)/R (SSIM, higher-better) or 100(R-M)/R (lower-better); 95% CIs from "
        f"{BOOTSTRAPS} slide resamples (seed {SEED}) shared across estimates.",
        "",
        "Metric definitions in `image_metrics_slide.csv`: image residual, coverage, joint coverage, "
        "target- and source-gradient NCC use the 02_correction phenotype definitions (OD-gradient magnitude, "
        "full 256-px frame); SSIM and LPIPS-VGG16 come from the Table 2 benchmark shards (2-px valid crop); "
        "UNI distance is 1 - cosine to the real target (03_uni extraction for raw/Reinhard/frequency/combined, "
        "01_stain_v4 for Macenko/Vahadane, 40-location GAN feature extraction restricted to set20 for "
        "Pix2Pix/CycleGAN). The benchmark's grayscale gradient NCCs and edge fractions are in "
        "`image_metrics_slide_supplementary.csv`.",
        "",
        "## QC",
        "",
    ]
    for name, ok in qc["location_sets"].items():
        lines.append(f"- location set `{name}`: {'matches' if ok else 'DIFFERS'}")
    lines += [
        f"- GAN prediction images equal to the patch cache at set20: source {qc['gan_prediction_pixels_equal_cache']['source_equal']}"
        f"/{qc['gan_prediction_pixels_equal_cache']['pairs']}, target {qc['gan_prediction_pixels_equal_cache']['target_equal']}"
        f"/{qc['gan_prediction_pixels_equal_cache']['pairs']}",
        "- new residual pipeline vs frozen 02_correction (raw/Reinhard/frequency/combined, 41,200 rows), max |diff|: "
        + ", ".join(f"{k} {v:.2e}" for k, v in qc["gan_pipeline_parity_vs_02_correction_max_abs_diff"].items()),
        "- raw UNI distance, max |diff| between extractions at set20: "
        + ", ".join(f"{k} {v:.2e}" for k, v in qc["uni_raw_distance_parity_max_abs_diff"].items()),
        f"- oracle classifier refit on 40/40 vs frozen predictions: agreement "
        f"{qc['oracle_classifier_refit_matches_frozen_predictions']['prediction_agreement']:.6f} over "
        f"{qc['oracle_classifier_refit_matches_frozen_predictions']['rows']} rows",
        f"- Table 2 shard SSIM vs GAN image evaluation (GT450 Pix2Pix, set20), max |diff|: "
        f"{qc['table2_shard_vs_gan_image_evaluation_ssim_max_abs_diff_gt450_pix2pix']:.2e}",
        f"- published numbers reproduced from stored data on their published sample (parity): "
        f"{int(comparison.parity_ok.sum())}/{len(comparison)}; failures: "
        + (", ".join(qc["comparison_parity_failures"]) or "none"),
        "- no missing values in the location or slide tables (checked in code).",
        "",
        "## Headline (pre-specified endpoints)",
        "",
    ] + headline(comparison, checks, aug) + [
        "",
        "## Primary endpoint: set20 value minus published value",
        "",
        f"{int(comparison.changes_at_printed_precision.sum())} of {len(comparison)} published numbers change at the "
        "printed precision on set20.",
        "",
        "| key | published | set20 | difference | changes at printed precision | published n locations |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(f"| {row.key} | {row.published_value:g} | {fmt(row.set20_value, row.decimals)} | "
                     f"{fmt(row.difference_set20_minus_published, row.decimals)} | "
                     f"{'yes' if row.changes_at_printed_precision else 'no'} | {row.published_locations} |")
    lines += ["", "## Sign and ranking checks", "", "| check | published | set20 | same |", "| --- | --- | --- | --- |"]
    for row in checks.itertuples(index=False):
        lines.append(f"| {row.check} | {row.published} | {row.set20} | {row.same} |")
    lines += ["", "## New set20 values without a published counterpart", ""]
    new = summaries[summaries["sample"].eq("set20") & summaries.scanner.eq("all") &
                    summaries.method.isin(["pix2pix", "cyclegan"]) &
                    summaries.metric.isin(["image_residual", "coverage", "joint_coverage",
                                           "target_gradient_ncc", "source_gradient_ncc"])]
    for row in new.itertuples(index=False):
        lines.append(f"- {row.method} {row.metric}: {row.mean:.4f} (95% CI {row.ci_low:.4f}-{row.ci_high:.4f}); "
                     f"raw {row.raw_mean:.4f}")
    classifier = aug[aug.metric.eq("real_vs_oracle_balanced_accuracy") & aug.arm.eq("strong_oracle") &
                     aug.scanner.eq("all")]
    lines += ["", "## Real-vs-oracle classifier (257-candidate structure-preserving oracle, pooled)", ""]
    for row in classifier.itertuples(index=False):
        lines.append(f"- {row.sample}: {100 * row.value:.2f}% (95% CI {100 * row.ci_low:.2f}-{100 * row.ci_high:.2f})")
    lines += ["", "## Files", "",
              "- `image_metrics_slide.csv`: slide x scanner values on set20 (long; join key dataset, method, scanner, slide_id, metric)",
              "- `image_metrics_slide_supplementary.csv`: benchmark grayscale gradient NCCs and edge fractions",
              "- `image_metrics_location.csv.gz`: location-level set20 table",
              "- `summary_methods.csv`: method summaries (set20; set40 and path-C UNI where available)",
              "- `augmentation_oracle_summary.csv`: oracle residual/coverage/classifier/SSIM/LPIPS/UNI on set40 and set20",
              "- `comparison.csv`, `sign_ranking_checks.csv`, `qc.json`",
              "- `gan_image_residual/`: per-slide shards from the array job"]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    task = sub.add_parser("gan-residual")
    task.add_argument("--task-index", type=int, default=None)
    task.add_argument("--workers", type=int, default=1)
    sub.add_parser("aggregate")
    args = parser.parse_args()
    if args.command == "gan-residual":
        index = args.task_index if args.task_index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        gan_residual_task(index, args.workers)
    else:
        aggregate()


if __name__ == "__main__":
    main()
