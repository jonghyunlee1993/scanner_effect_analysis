#!/usr/bin/env python3
"""RV04: three-axis evaluation of every correction method in four PFMs.

Reads the RV03 corrected-image embeddings (``results/corrected_embeddings``) through
``corrected_embedding_reader`` and evaluates, for PanNormal (``set20``, five directions)
and PLISM (2,387 locations, three directions):

* representation alignment: cosine target distance and gain; normalized target distance
  (target distance / mean between-tissue distance of the same condition's embeddings);
  spread ratio (between-tissue distance of the condition / that of the real targets);
* target detectability: balanced accuracy of the existing UNI v1 ``03_uni`` classifier
  (StandardScaler + LogisticRegression(C=0.1, liblinear), ``uni_aggregate`` in
  ``src/scanner_batch_extensions.py``) separating real target embeddings from corrected
  embeddings, pooled over target scanners, five-fold cross-validation grouped by slide
  (locked folds) or by PLISM section (sorted sections, index modulo 5); accuracy is
  averaged within slide/section, then over slides/sections;
* tissue preservation: same-tissue top-1 retrieval with the corrected slide mean (PLISM:
  section x core mean) as query and the real target-scanner means of other slides (PLISM:
  other sections) as bank; macro recall over tissues with at least two slides (36 / 102);
* image fidelity: joined from RV-P0b (PanNormal) and the existing PLISM benchmark.

Stages (one sbatch, ``SLURM_ARRAY_TASK_ID`` selects a task; without it, ``aggregate``):
``task`` computes detectability for one dataset x PFM x method or the distance/pair/retrieval
tables for one dataset x PFM; ``aggregate`` computes 2,000-resample slide (section)
bootstrap CIs with shared resamples, parity checks and ``summary.md``.
``--source smoke`` runs the same code on the smoke adapter (UNI v1 PanNormal from
``03_uni``, UNI2-h PLISM raw + identity fixture) into ``results/three_axis_evaluation/smoke``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corrected_embedding_reader import (  # noqa: E402
    BOOTSTRAPS, CV_FOLDS, DATASETS, EMBEDDINGS, FEATURE_METHODS, HIGHER_IS_BETTER, IMAGE_METHODS,
    METHOD_LABELS, METHODS, PFM_LABELS, PFMS, RESULTS, TARGETS, bootstrap_weights, image_improvement,
    image_metric_arrays, load, load_image_metrics, manifest_status, rel, sha256, summarize, unit,
    weighted_macro, weighted_mean, write_frame, write_json, write_text,
)
from prenorm.feature_correction import macro_retrieval  # noqa: E402


OUTPUT = RESULTS / "three_axis_evaluation"
UNI_SUMMARY = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17/03_uni/summary.csv"
FEATURE40 = PROJECT / "outputs/discussion_followup_2026-09-25/feature40_forward"
TABLE2 = PROJECT / "analysis/paper/table2_main_summary.csv"
SMOKE_DETECT = {"pannormal": ("uni_v1", ("raw", "reinhard", "frequency", "combined")),
                "plism": ("uni2", ("raw", "identity"))}
IMAGE_STATS = ("ssim", "lpips", "image_residual", "coverage", "joint_coverage",
               "target_gradient_ncc", "source_gradient_ncc")


def output_root(source: str) -> Path:
    return OUTPUT / "smoke" if source == "smoke" else OUTPUT


def detectability_classifier():
    """Exact configuration of the 03_uni real-vs-corrected classifier (uni_aggregate)."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, solver="liblinear"))


def task_list(source: str) -> list[dict]:
    if source == "smoke":
        metrics = [(dataset, SMOKE_DETECT[dataset][0]) for dataset in DATASETS]
        detect = [(dataset, SMOKE_DETECT[dataset][0], method) for dataset in DATASETS
                  for method in SMOKE_DETECT[dataset][1]]
    else:
        metrics = [(dataset, pfm) for dataset in DATASETS for pfm in PFMS]
        detect = [(dataset, pfm, method) for dataset in DATASETS for pfm in PFMS for method in ("raw",) + METHODS]
    return ([{"kind": "metrics", "dataset": d, "pfm": p} for d, p in metrics] +
            [{"kind": "detect", "dataset": d, "pfm": p, "method": m} for d, p, m in detect])


# ----------------------------------------------------------------------------- tasks

def run_detect(task: dict, out: Path, source: str) -> None:
    dataset, pfm, method = task["dataset"], task["pfm"], task["method"]
    stem = f"{dataset}__{pfm}__{method}"
    status_path = out / "tasks/detect" / f"{stem}.json"
    started = time.time()
    try:
        emb = load(dataset, pfm, methods=[] if method == "raw" else [method], source=source)
    except FileNotFoundError as error:
        write_json({"status": "missing_input", "task": task, "error": str(error)}, status_path)
        return
    variants = {scanner: dict(emb.variants(method, scanner)) for scanner in emb.scanners}
    scanners = [scanner for scanner in emb.scanners if variants[scanner]]
    if not scanners:
        write_json({"status": "missing_condition", "task": task}, status_path)
        return
    labels = sorted(set.intersection(*(set(variants[s]) for s in scanners)))
    folds = emb.locations["fold"].to_numpy()
    unit_code = emb.locations["unit_code"].to_numpy()
    rows = []
    for label in labels:
        # Inputs are L2-normalized: a no-op for the stored (normalized) raw and image-corrected
        # embeddings, and it keeps unnormalized ridge/ComBat/OLS map outputs in the same cosine
        # geometry, so the classifier cannot key on vector norm.
        generated = unit(np.concatenate([variants[s][label] for s in scanners])).astype(np.float32)
        target = unit(np.concatenate([emb.target(s) for s in scanners])).astype(np.float32)
        fold = np.tile(folds, len(scanners))
        codes = np.tile(unit_code, len(scanners))
        scanner_index = np.repeat(np.arange(len(scanners)), emb.n)
        for held_out in range(CV_FOLDS):
            train = fold != held_out
            test = fold == held_out
            if not test.any():
                continue
            classifier = detectability_classifier()
            classifier.fit(np.concatenate([generated[train], target[train]]),
                           np.concatenate([np.zeros(train.sum()), np.ones(train.sum())]))
            predicted = classifier.predict(np.concatenate([generated[test], target[test]]))
            truth = np.concatenate([np.zeros(test.sum()), np.ones(test.sum())])
            correct = (predicted == truth).astype(int)
            frame = pd.DataFrame({
                "unit_code": np.concatenate([codes[test], codes[test]]),
                "scanner": np.asarray(scanners)[np.concatenate([scanner_index[test], scanner_index[test]])],
                "cls": np.where(truth == 1, "real_target", "corrected"),
                "correct": correct,
            })
            grouped = frame.groupby(["unit_code", "scanner", "cls"], as_index=False).agg(
                n=("correct", "size"), correct=("correct", "sum"))
            grouped["cv_fold"] = held_out
            grouped["variant"] = label
            rows.append(grouped)
    result = pd.concat(rows, ignore_index=True)
    result["unit"] = np.asarray(emb.units)[result["unit_code"].to_numpy()]
    result.insert(0, "method", method)
    result.insert(0, "pfm", pfm)
    result.insert(0, "dataset", dataset)
    write_frame(result.drop(columns="unit_code"), out / "tasks/detect" / f"{stem}.csv.gz")
    write_json({"status": "pass", "task": task, "source": emb.source, "scanners": scanners,
                "missing_scanners": sorted(set(emb.scanners) - set(scanners)), "variants": labels,
                "n_locations": emb.n, "n_units": len(emb.units), "seconds": time.time() - started,
                "notes": emb.notes}, status_path)


def run_metrics(task: dict, out: Path, source: str) -> None:
    dataset, pfm = task["dataset"], task["pfm"]
    stem = f"{dataset}__{pfm}"
    status_path = out / "tasks/metrics" / f"{stem}.json"
    try:
        emb = load(dataset, pfm, methods=None, source=source)
    except FileNotFoundError as error:
        write_json({"status": "missing_input", "task": task, "error": str(error)}, status_path)
        return
    locations = emb.locations
    tissue = locations["tissue_code"].to_numpy()
    different = (tissue[:, None] != tissue[None, :]).astype(np.float64)
    membership = np.zeros((len(emb.units), emb.n))
    membership[locations["unit_code"].to_numpy(), np.arange(emb.n)] = 1.0
    pair_counts = membership @ different @ membership.T
    queries = emb.queries
    same_unit = queries["unit_code"].to_numpy()[:, None] == queries["unit_code"].to_numpy()[None, :]
    query_tissue = queries["tissue_code"].to_numpy()
    methods = ["raw", "target"] + emb.methods()
    distance_rows, retrieval_rows, pairs, missing, qc = [], [], {}, [], []
    for scanner in emb.scanners:
        target = unit(emb.target(scanner))
        raw_distance = emb.unit_means(1.0 - np.sum(unit(emb.raw()) * target, axis=1))
        bank = emb.query_embedding_means(emb.target(scanner))
        for method in methods:
            variants = emb.variants(method, scanner)
            if not variants:
                missing.append({"dataset": dataset, "pfm": pfm, "method": method, "scanner": scanner})
                continue
            for label, matrix in variants:
                vectors = unit(matrix)
                distance = emb.unit_means(1.0 - np.sum(vectors * target, axis=1))
                distance_rows.append(pd.DataFrame({
                    "method": method, "scanner": scanner, "variant": label, "unit": emb.units,
                    "target_distance": distance, "raw_distance": raw_distance}))
                gram = vectors @ vectors.T
                pairs[f"{method}__{scanner}__{label}"] = membership @ ((1.0 - gram) * different) @ membership.T
                query = emb.query_embedding_means(matrix)
                scores = query @ bank.T
                scores[same_unit] = -np.inf
                predicted = query_tissue[np.argmax(scores, axis=1)]
                correct = (predicted == query_tissue).astype(int)
                retrieval_rows.append(pd.DataFrame({
                    "method": method, "scanner": scanner, "variant": label, "query": queries["query"],
                    "unit": queries["unit"], "tissue": queries["tissue"], "correct": correct}))
                if dataset == "pannormal":
                    reference = macro_retrieval(query, bank, queries["tissue"].to_numpy())
                    counts = queries.groupby("tissue")["query"].transform("size").to_numpy()
                    ours = pd.Series(correct[counts > 1]).groupby(
                        queries["tissue"].to_numpy()[counts > 1]).mean().mean()
                    qc.append({"method": method, "scanner": scanner, "variant": label,
                               "macro_retrieval_reference": reference, "macro_retrieval_ours": float(ours)})
    write_frame(pd.concat(distance_rows, ignore_index=True), out / "tasks/metrics" / f"{stem}_distances.csv.gz")
    write_frame(pd.concat(retrieval_rows, ignore_index=True), out / "tasks/metrics" / f"{stem}_retrieval.csv.gz")
    np.savez_compressed(out / "tasks/metrics" / f"{stem}_pairs.npz", units=np.asarray(emb.units),
                        pair_counts=pair_counts, **pairs)
    qc_frame = pd.DataFrame(qc)
    max_qc = float((qc_frame.macro_retrieval_reference - qc_frame.macro_retrieval_ours).abs().max()) if len(qc) else 0.0
    if max_qc > 1e-12:
        raise RuntimeError(f"retrieval disagrees with prenorm.feature_correction.macro_retrieval: {max_qc}")
    norms = np.linalg.norm(emb.raw(), axis=1)
    write_json({"status": "pass", "task": task, "source": emb.source, "n_locations": emb.n,
                "n_units": len(emb.units), "n_queries": len(queries), "n_tissues": len(emb.tissues),
                "feature_dim": int(emb.raw().shape[1]), "methods": methods, "missing": missing,
                "available_conditions": list(emb.available), "notes": emb.notes,
                "raw_unit_norm_max_error": float(np.abs(norms - 1).max()),
                "macro_retrieval_max_abs_difference_vs_prenorm": max_qc,
                "files_sha_hint": {k: v for k, v in list(emb.files.items())[:3]}}, status_path)


# ----------------------------------------------------------------------------- aggregate

def between_tissue(weights: np.ndarray, pairs: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.sum((weights @ pairs) * weights, axis=1) / np.sum((weights @ counts) * weights, axis=1)


def evaluate_pfm(dataset: str, pfm: str, out: Path, weights: np.ndarray, units: tuple,
                 image_arrays: dict) -> tuple[dict, dict, dict]:
    """Bootstrap draws [B+1] for every (method, scanner, statistic); scanner 'pooled' is scanner-equal."""
    stem = f"{dataset}__{pfm}"
    distances = pd.read_csv(out / "tasks/metrics" / f"{stem}_distances.csv.gz", dtype={"unit": str})
    retrieval = pd.read_csv(out / "tasks/metrics" / f"{stem}_retrieval.csv.gz", dtype={"unit": str, "query": str})
    stored = np.load(out / "tasks/metrics" / f"{stem}_pairs.npz")
    if tuple(stored["units"].astype(str)) != tuple(units):
        raise ValueError(f"{stem}: unit order differs between tasks")
    counts = stored["pair_counts"]
    unit_index = pd.Index(units)
    scanners = TARGETS[dataset]
    queries = retrieval.drop_duplicates("query")[["query", "unit", "tissue"]].reset_index(drop=True)
    units_per_tissue = queries.groupby("tissue")["unit"].nunique()
    eligible = queries.loc[queries.tissue.map(units_per_tissue).ge(2)].reset_index(drop=True)
    tissue_names = sorted(eligible.tissue.unique())
    tissue_code = eligible.tissue.map({name: index for index, name in enumerate(tissue_names)}).to_numpy()
    query_weights = weights[:, unit_index.get_indexer(eligible.unit)]

    draws: dict = {}
    info: dict = {}

    def unit_vector(frame: pd.DataFrame, column: str) -> np.ndarray:
        return frame.groupby("unit")[column].mean().reindex(unit_index).to_numpy(dtype=np.float64)

    def spread(method: str, scanner: str, label: str) -> np.ndarray:
        return between_tissue(weights, stored[f"{method}__{scanner}__{label}"], counts)

    def retrieval_draws(method: str, scanner: str) -> np.ndarray:
        block = retrieval.loc[retrieval.method.eq(method) & retrieval.scanner.eq(scanner)]
        per_query = block.groupby("query")["correct"].mean().reindex(eligible["query"]).to_numpy(dtype=np.float64)
        return weighted_macro(query_weights, per_query, tissue_code, len(tissue_names))

    methods = [m for m in ["raw", "target"] + list(METHODS) + sorted(set(distances.method) - set(METHODS) - {"raw", "target"})
               if m in set(distances.method)]
    for scanner in scanners:
        raw_block = distances.loc[distances.method.eq("raw") & distances.scanner.eq(scanner)]
        target_block = distances.loc[distances.method.eq("target") & distances.scanner.eq(scanner)]
        if raw_block.empty or target_block.empty:
            continue
        raw_units = unit_vector(raw_block, "target_distance")
        b_raw = spread("raw", scanner, "none")
        b_target = spread("target", scanner, "none")
        raw_normalized = weighted_mean(weights, raw_units) / b_raw
        raw_retrieval = retrieval_draws("raw", scanner)
        target_retrieval = retrieval_draws("target", scanner)
        for method in methods:
            block = distances.loc[distances.method.eq(method) & distances.scanner.eq(scanner)]
            if block.empty:
                continue
            labels = sorted(block.variant.unique())
            per_variant = {label: unit_vector(block.loc[block.variant.eq(label)], "target_distance") for label in labels}
            method_units = np.mean(np.stack(list(per_variant.values())), axis=0)
            b_method = np.mean([spread(method, scanner, label) for label in labels], axis=0)
            normalized = np.mean([weighted_mean(weights, per_variant[label]) / spread(method, scanner, label)
                                  for label in labels], axis=0)
            ret = retrieval_draws(method, scanner)
            key = (method, scanner)
            draws[key] = {
                "target_distance": weighted_mean(weights, method_units),
                "raw_distance": weighted_mean(weights, raw_units),
                "target_gain": weighted_mean(weights, raw_units - method_units),
                "between_tissue_distance": b_method,
                "spread_ratio": np.mean([spread(method, scanner, label) / b_target for label in labels], axis=0),
                "spread_ratio_vs_raw": np.mean([spread(method, scanner, label) / b_raw for label in labels], axis=0),
                "normalized_target_distance": normalized,
                "normalized_raw_distance": raw_normalized,
                "normalized_gain": raw_normalized - normalized,
                "retrieval_macro_recall": ret,
                "retrieval_minus_target": ret - target_retrieval,
                "retrieval_minus_raw": ret - raw_retrieval,
            }
            info[key] = {"n_variants": len(labels), "variants": ",".join(labels)}
            for metric in IMAGE_STATS:
                values = image_arrays.get((method, scanner, metric))
                if values is not None and np.isfinite(values).any():
                    draws[key][f"image_{metric}"] = weighted_mean(weights, values)
                improvement = image_improvement(image_arrays, method, scanner, metric)
                if improvement is not None and method != "raw" and np.isfinite(improvement).any():
                    draws[key][f"image_{metric}_improvement"] = weighted_mean(weights, improvement)

    # Detectability from the pooled classifier; per-scanner breakdown and scanner-pooled.
    detect_status = {}
    for method in methods:
        if method == "target":
            continue
        path = out / "tasks/detect" / f"{dataset}__{pfm}__{method}.csv.gz"
        status_path = path.with_suffix("").with_suffix(".json")
        detect_status[method] = json.loads(status_path.read_text())["status"] if status_path.exists() else "not_run"
        if not path.exists():
            continue
        table = pd.read_csv(path, dtype={"unit": str})
        per_variant = table.groupby(["variant", "unit"])[["n", "correct"]].sum()
        accuracy = (per_variant["correct"] / per_variant["n"]).groupby(level="unit").mean()
        pooled = weighted_mean(weights, accuracy.reindex(unit_index).to_numpy(dtype=np.float64))
        draws.setdefault((method, "pooled"), {})["detectability"] = pooled
        for scanner, group in table.groupby("scanner"):
            per = group.groupby(["variant", "unit"])[["n", "correct"]].sum()
            acc = (per["correct"] / per["n"]).groupby(level="unit").mean()
            if (method, scanner) in draws:
                draws[(method, scanner)]["detectability"] = weighted_mean(
                    weights, acc.reindex(unit_index).to_numpy(dtype=np.float64))
        info.setdefault((method, "pooled"), {})["detect_variants"] = int(table.variant.nunique())
        info[(method, "pooled")]["detect_scanners"] = ",".join(sorted(table.scanner.unique()))

    # Scanner-equal means for every per-scanner statistic (only when all scanners exist).
    for method in methods:
        present = [s for s in scanners if (method, s) in draws]
        if not present:
            continue
        pooled = draws.setdefault((method, "pooled"), {})
        stats_names = set.intersection(*(set(draws[(method, s)]) for s in present)) - {"detectability"}
        for name in stats_names:
            if len(present) == len(scanners):
                pooled[name] = np.mean([draws[(method, s)][name] for s in present], axis=0)
        info.setdefault((method, "pooled"), {})["scanners_present"] = len(present)
        info[(method, "pooled")]["n_variants"] = info[(method, present[0])]["n_variants"]
    for key, values in draws.items():
        if "target_gain" in values and "raw_distance" in values:
            values["relative_gain"] = values["target_gain"] / values["raw_distance"]
            values["normalized_relative_gain"] = values["normalized_gain"] / values["normalized_raw_distance"]
        if "detectability" in values and ("raw", key[1]) in draws and "detectability" in draws[("raw", key[1])]:
            values["detectability_minus_raw"] = values["detectability"] - draws[("raw", key[1])]["detectability"]
    return draws, info, {"detect_status": detect_status, "retrieval_tissues": len(tissue_names),
                         "retrieval_queries": len(eligible)}


def long_rows(dataset: str, pfm: str, draws: dict, info: dict) -> list[dict]:
    rows = []
    for (method, scanner), values in draws.items():
        for statistic, series in values.items():
            row = {"dataset": dataset, "pfm": pfm, "method": method, "scanner": scanner, "statistic": statistic}
            row.update(summarize(series))
            row.update({k: v for k, v in info.get((method, scanner), {}).items()
                        if k in ("n_variants", "scanners_present")})
            rows.append(row)
    return rows


def parity_checks(long: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    frame = long.loc[long.dataset.eq("pannormal") & long.pfm.eq("uni_v1")]
    if frame.empty:
        return pd.DataFrame(rows)

    def value(method, scanner, statistic):
        match = frame.loc[frame.method.eq(method) & frame.scanner.eq(scanner) & frame.statistic.eq(statistic), "estimate"]
        return float(match.iloc[0]) if len(match) else np.nan

    reference = pd.read_csv(UNI_SUMMARY).set_index("arm")
    for method in ("raw", "reinhard", "frequency", "combined"):
        rows.append({"check": "03_uni detectability", "method": method, "scanner": "pooled",
                     "reference": float(reference.loc[method, "real_vs_corrected_balanced_accuracy"]),
                     "ours": value(method, "pooled", "detectability"), "tolerance": 0.005,
                     "reference_file": rel(UNI_SUMMARY)})
        if method != "raw":
            rows.append({"check": "03_uni target gain", "method": method, "scanner": "pooled",
                         "reference": float(reference.loc[method, "target_gain"]),
                         "ours": value(method, "pooled", "target_gain"), "tolerance": 0.001,
                         "reference_file": rel(UNI_SUMMARY)})
    content = pd.read_csv(FEATURE40 / "content_summary.csv")
    mapping = {"raw": "raw", "ridge": "featmap_ridge", "combat": "combat", "ols": "featmap_ols"}
    for method, name in mapping.items():
        block = content.loc[content.method.eq(name)]
        rows.append({"check": "feature40_forward retrieval (scanner-equal)", "method": method, "scanner": "pooled",
                     "reference": float(block.macro_tissue_retrieval.mean()),
                     "ours": value(method, "pooled", "retrieval_macro_recall"), "tolerance": 0.01,
                     "reference_file": rel(FEATURE40 / "content_summary.csv")})
        for record in block.itertuples():
            rows.append({"check": "feature40_forward retrieval", "method": method, "scanner": record.scanner,
                         "reference": float(record.macro_tissue_retrieval),
                         "ours": value(method, record.scanner, "retrieval_macro_recall"), "tolerance": 0.02,
                         "reference_file": rel(FEATURE40 / "content_summary.csv")})
    rows.append({"check": "feature40_forward target self-retrieval (scanner-equal)", "method": "target",
                 "scanner": "pooled", "reference": float(content.loc[content.method.eq("raw"), "target_self_retrieval"].mean()),
                 "ours": value("target", "pooled", "retrieval_macro_recall"), "tolerance": 0.005,
                 "reference_file": rel(FEATURE40 / "content_summary.csv")})
    pooled = pd.read_csv(FEATURE40 / "pooled_summary.csv").set_index("method")
    for method, name in (("ridge", "featmap_ridge"), ("combat", "combat"), ("ols", "featmap_ols")):
        rows.append({"check": "feature40_forward pooled target gain", "method": method, "scanner": "pooled",
                     "reference": float(pooled.loc[name, "mean_target_gain"]),
                     "ours": value(method, "pooled", "target_gain"), "tolerance": 0.002,
                     "reference_file": rel(FEATURE40 / "pooled_summary.csv")})
    published = pd.read_csv(TABLE2)
    for dataset, label in (("pannormal", "PanNormal"), ("plism", "PLISM")):
        block = long.loc[long.dataset.eq(dataset) & long.scanner.eq("pooled")]
        if block.empty:
            continue
        pfm = sorted(block.pfm.unique())[0]
        for record in published.loc[published.dataset.eq(label)].itertuples():
            for statistic, column in (("image_ssim", "target_ssim_mean"), ("image_lpips", "lpips_vgg_mean")):
                match = block.loc[block.pfm.eq(pfm) & block.method.eq(record.method) & block.statistic.eq(statistic), "estimate"]
                rows.append({"check": f"published Table 2 {label} {statistic} (image join)", "method": record.method,
                             "scanner": "pooled", "reference": float(getattr(record, column)),
                             "ours": float(match.iloc[0]) if len(match) else np.nan, "tolerance": 0.005,
                             "reference_file": rel(TABLE2)})
    table = pd.DataFrame(rows)
    table["difference"] = table["ours"] - table["reference"]
    table["status"] = np.where(table["ours"].isna(), "not_available",
                               np.where(table["difference"].abs() <= table["tolerance"], "pass", "differs"))
    return table


def hypotheses(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pooled = long.loc[long.scanner.eq("pooled")]

    def get(dataset, pfm, method, statistic):
        match = pooled.loc[pooled.dataset.eq(dataset) & pooled.pfm.eq(pfm) & pooled.method.eq(method)
                           & pooled.statistic.eq(statistic)]
        return match.iloc[0] if len(match) else None

    for pfm in sorted(pooled.pfm.unique()):
        for method in IMAGE_METHODS:
            row = get("pannormal", pfm, method, "detectability")
            if row is not None:
                rows.append({"hypothesis": "H4a", "dataset": "pannormal", "pfm": pfm, "method": method,
                             "statistic": "detectability", "estimate": row.estimate, "ci_low": row.ci_low,
                             "ci_high": row.ci_high, "criterion": "estimate >= 0.95",
                             "meets_criterion": bool(row.estimate >= 0.95)})
        for dataset in DATASETS:
            for method in FEATURE_METHODS:
                row = get(dataset, pfm, method, "detectability_minus_raw")
                level = get(dataset, pfm, method, "detectability")
                if row is not None and level is not None:
                    rows.append({"hypothesis": "H4b", "dataset": dataset, "pfm": pfm, "method": method,
                                 "statistic": "detectability_minus_raw", "estimate": row.estimate,
                                 "ci_low": row.ci_low, "ci_high": row.ci_high,
                                 "criterion": f"reported; detectability = {level.estimate:.3f}",
                                 "meets_criterion": None})
            for statistic, criterion in (("spread_ratio", "CI upper < 1"),
                                         ("spread_ratio_vs_raw", "CI upper < 1")):
                row = get(dataset, pfm, "ridge", statistic)
                if row is not None:
                    rows.append({"hypothesis": "H4c", "dataset": dataset, "pfm": pfm, "method": "ridge",
                                 "statistic": statistic, "estimate": row.estimate, "ci_low": row.ci_low,
                                 "ci_high": row.ci_high, "criterion": criterion,
                                 "meets_criterion": bool(row.ci_high < 1)})
            relative = get(dataset, pfm, "ridge", "relative_gain")
            normalized = get(dataset, pfm, "ridge", "normalized_relative_gain")
            if relative is not None and normalized is not None:
                rows.append({"hypothesis": "H4c", "dataset": dataset, "pfm": pfm, "method": "ridge",
                             "statistic": "normalized_relative_gain_vs_relative_gain",
                             "estimate": normalized.estimate - relative.estimate, "ci_low": np.nan,
                             "ci_high": np.nan, "criterion": "normalized relative gain < raw relative gain",
                             "meets_criterion": bool(normalized.estimate < relative.estimate)})
            for method in IMAGE_METHODS:
                row = get(dataset, pfm, method, "retrieval_minus_raw")
                if row is not None:
                    rows.append({"hypothesis": "H4d", "dataset": dataset, "pfm": pfm, "method": method,
                                 "statistic": "retrieval_minus_raw", "estimate": row.estimate,
                                 "ci_low": row.ci_low, "ci_high": row.ci_high, "criterion": "CI upper < 0",
                                 "meets_criterion": bool(row.ci_high < 0)})
    return pd.DataFrame(rows)


def fmt(row, digits=3, percent=False) -> str:
    if row is None or not np.isfinite(row["estimate"]):
        return "n/a"
    scale = 100.0 if percent else 1.0
    text = f"{scale * row['estimate']:.{digits}f}"
    if np.isfinite(row.get("ci_low", np.nan)):
        text += f" [{scale * row['ci_low']:.{digits}f}, {scale * row['ci_high']:.{digits}f}]"
    return text


def endpoint_table(long: pd.DataFrame, dataset: str, statistic: str, digits: int, percent: bool,
                   methods: list[str]) -> str:
    pooled = long.loc[long.dataset.eq(dataset) & long.scanner.eq("pooled") & long.statistic.eq(statistic)]
    pfms = [pfm for pfm in PFMS if pfm in set(pooled.pfm)] + sorted(set(pooled.pfm) - set(PFMS))
    if not pfms:
        return "_no values_\n"
    lines = ["| Method | " + " | ".join(PFM_LABELS.get(p, p) for p in pfms) + " |",
             "| --- | " + " | ".join("---" for _ in pfms) + " |"]
    for method in methods:
        cells = []
        for pfm in pfms:
            match = pooled.loc[pooled.pfm.eq(pfm) & pooled.method.eq(method)]
            cells.append(fmt(match.iloc[0], digits, percent) if len(match) else "–")
        if any(cell != "–" for cell in cells):
            lines.append(f"| {METHOD_LABELS.get(method, method)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def aggregate(out: Path, source: str) -> None:
    tasks = task_list(source)
    datasets = sorted({task["dataset"] for task in tasks}, key=DATASETS.index)
    pfms_by_dataset = {d: [t["pfm"] for t in tasks if t["kind"] == "metrics" and t["dataset"] == d] for d in datasets}
    long, cells_info, qc, missing_rows, provenance = [], [], {}, [], {}
    for dataset in datasets:
        image_table, image_source = load_image_metrics(dataset, source)
        provenance[f"image_fidelity_{dataset}"] = image_source
        weights = None
        for pfm in pfms_by_dataset[dataset]:
            status_path = out / "tasks/metrics" / f"{dataset}__{pfm}.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "not_run"}
            qc[f"{dataset}/{pfm}"] = {k: status.get(k) for k in (
                "status", "source", "n_locations", "n_units", "n_queries", "n_tissues", "feature_dim",
                "raw_unit_norm_max_error", "macro_retrieval_max_abs_difference_vs_prenorm", "notes")}
            for item in status.get("missing", []):
                missing_rows.append({**item, "reason": "condition absent in RV03 store"})
            if status.get("status") != "pass":
                missing_rows.append({"dataset": dataset, "pfm": pfm, "method": "*", "scanner": "*",
                                     "reason": f"metrics task {status.get('status')}"})
                continue
            units = tuple(np.load(out / "tasks/metrics" / f"{dataset}__{pfm}_pairs.npz")["units"].astype(str))
            if weights is None:
                weights = bootstrap_weights(dataset, len(units))
                dataset_units = units
            elif units != dataset_units:
                raise ValueError(f"{dataset}: unit order differs between PFMs")
            image_arrays = image_metric_arrays(image_table, units)
            draws, info, extra = evaluate_pfm(dataset, pfm, out, weights, units, image_arrays)
            qc[f"{dataset}/{pfm}"].update(extra)
            for method, state in extra["detect_status"].items():
                if state != "pass":
                    missing_rows.append({"dataset": dataset, "pfm": pfm, "method": method, "scanner": "pooled",
                                         "reason": f"detectability task {state}"})
            long.extend(long_rows(dataset, pfm, draws, info))
    long = pd.DataFrame(long)
    if long.empty:
        raise RuntimeError("no RV04 results to aggregate")
    long["method_label"] = long["method"].map(lambda m: METHOD_LABELS.get(m, m))
    write_frame(long, out / "cell_statistics.csv")
    wide = long.pivot_table(index=["dataset", "pfm", "method", "scanner"], columns="statistic",
                            values="estimate", aggfunc="first").reset_index()
    write_frame(wide, out / "three_axis_cells.csv")
    primary_names = ("detectability", "retrieval_minus_target", "normalized_target_distance")
    primary = long.loc[long.scanner.eq("pooled") & long.statistic.isin(primary_names)]
    write_frame(primary, out / "primary_endpoints.csv")
    hyp = hypotheses(long)
    write_frame(hyp, out / "hypotheses.csv")
    parity = parity_checks(long, source)
    write_frame(parity, out / "parity_checks.csv")
    missing = pd.DataFrame(missing_rows, columns=["dataset", "pfm", "method", "scanner", "reason"])
    write_frame(missing, out / "missing_conditions.csv")
    identity = long.loc[long.method.eq("identity")]
    identity_check = None
    if not identity.empty:
        merged = identity.merge(long.loc[long.method.eq("raw")], on=["dataset", "pfm", "scanner", "statistic"],
                                suffixes=("", "_raw"))
        merged = merged.loc[~merged.statistic.str.contains("minus_raw|gain|relative")]
        identity_check = float((merged.estimate - merged.estimate_raw).abs().max())
    manifest = manifest_status()
    write_json({"source": source, "embedding_manifest": manifest, "qc": qc, "provenance": provenance,
                "bootstrap_replicates": BOOTSTRAPS, "identity_fixture_max_abs_difference": identity_check,
                "code_sha256": sha256(Path(__file__))}, out / "qc.json")
    write_text(render_summary(long, parity, hyp, missing, qc, provenance, source, manifest, identity_check),
               out / "summary.md")
    print((out / "summary.md").read_text())


def render_summary(long, parity, hyp, missing, qc, provenance, source, manifest, identity_check) -> str:
    lines = ["# RV04 three-axis evaluation", ""]
    if source == "smoke":
        lines += ["**SMOKE TEST** on the smoke adapter (PanNormal UNI v1 from `03_uni`: raw, Reinhard, frequency, "
                  "color + frequency; PLISM UNI2-h raw + identity fixture). Not a revision result.", ""]
    lines += ["## What ran", "",
              f"- Embeddings: `{rel(EMBEDDINGS)}` (manifest: {manifest}); source `{source}`.",
              "- Target distance: 1 − cosine(corrected AT2, real target) at the same location; gain = raw − corrected; "
              "PanNormal location → slide, PLISM location → core → section; means over slides/sections, "
              "five (three) directions equally weighted.",
              "- Between-tissue distance of a condition (per direction): mean 1 − cosine over all pairs of locations "
              "whose tissue types differ, taken among that condition's embeddings (PanNormal: all 2,060 set20 "
              "locations of the 103 slides, every one of which is a held-out prediction for cross-fitted methods, "
              "so every pair is between different slides; PLISM: all 2,387 locations, pairs from different cores "
              "in any section). Normalized target distance = target distance / between-tissue distance of the "
              "same condition; spread ratio = between-tissue distance of the condition / that of the real "
              "targets. Both denominators are recomputed in every bootstrap resample.",
              "- Detectability: `03_uni` classifier (StandardScaler + LogisticRegression C = 0.1, liblinear) on "
              "L2-normalized corrected vs real target embeddings (normalization is a no-op for raw and image "
              "conditions; ridge/ComBat/OLS outputs are unnormalized map outputs) pooled over directions; "
              "five-fold CV grouped by the locked "
              "slide folds (PLISM: section folds, sorted sections modulo 5); accuracy per slide/section, then mean. "
              "Classes are balanced within each slide, so accuracy equals balanced accuracy. Chance 0.5.",
              "- Retrieval: query = unit-normalized corrected slide mean (PLISM section × core); bank = real target "
              "slide means of all other slides (PLISM: other sections); top-1 same-tissue; macro recall over tissues "
              "with ≥ 2 slides (36 tissues / 102 slides; PLISM 46 cores); scanner-equal mean.",
              "- PLISM fold-fitted conditions: every metric is computed per fold variant and averaged with equal "
              "weight (distances within section; retrieval within query; detectability within section; "
              "between-tissue distance per variant).",
              f"- CIs: {BOOTSTRAPS} slide (PLISM: section) bootstrap resamples, shared across methods and PFMs "
              "(paired); percentile intervals.",
              f"- Image fidelity: PanNormal — {provenance.get('image_fidelity_pannormal', 'n/a')}; "
              f"PLISM — {provenance.get('image_fidelity_plism', 'n/a')}.", ""]
    lines += ["## QC", ""]
    for key, value in qc.items():
        lines.append(f"- {key}: status {value.get('status')}, locations {value.get('n_locations')}, units "
                     f"{value.get('n_units')}, tissues {value.get('n_tissues')}, dim {value.get('feature_dim')}, "
                     f"raw unit-norm error {value.get('raw_unit_norm_max_error')}, retrieval vs "
                     f"`macro_retrieval` max |Δ| {value.get('macro_retrieval_max_abs_difference_vs_prenorm')}, "
                     f"retrieval tissues/queries {value.get('retrieval_tissues')}/{value.get('retrieval_queries')}")
    if identity_check is not None:
        lines.append(f"- Identity fixture (fold-variant averaging) max |identity − raw| over statistics: {identity_check:.3g}")
    lines.append(f"- Missing conditions/tasks: {len(missing)} (see `missing_conditions.csv`).")
    lines.append("")
    if len(parity):
        lines += ["## Parity with existing UNI v1 results", "",
                  "| Check | Method | Scanner | Reference | Ours | Δ | Status |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for row in parity.itertuples():
            lines.append(f"| {row.check} | {row.method} | {row.scanner} | {row.reference:.4f} | "
                         f"{row.ours:.4f} | {row.difference:+.4f} | {row.status} |")
        lines.append("")
    order = ["raw"] + list(METHODS) + sorted(set(long.method) - set(METHODS) - {"raw", "target"}) + ["target"]
    for dataset in DATASETS:
        if dataset not in set(long.dataset):
            continue
        title = "PanNormal (set20, five directions)" if dataset == "pannormal" else "PLISM (three directions)"
        lines += [f"## Primary endpoints — {title}", "",
                  "### Detectability (balanced accuracy, pooled directions)", "",
                  endpoint_table(long, dataset, "detectability", 3, False, order),
                  "### Retrieval macro recall minus real-target value (percentage points, scanner-equal)", "",
                  endpoint_table(long, dataset, "retrieval_minus_target", 1, True, order),
                  "### Normalized target distance (scanner-equal)", "",
                  endpoint_table(long, dataset, "normalized_target_distance", 3, False, order),
                  "### Context: target gain (cosine) and spread ratio (scanner-equal)", "",
                  endpoint_table(long, dataset, "target_gain", 4, False, [m for m in order if m != "target"]),
                  endpoint_table(long, dataset, "spread_ratio", 3, False, order),
                  "### Context: retrieval macro recall (%, scanner-equal)", "",
                  endpoint_table(long, dataset, "retrieval_macro_recall", 1, True, order)]
        if "image_ssim_improvement" in set(long.loc[long.dataset.eq(dataset), "statistic"]):
            lines += ["### Image fidelity: SSIM improvement over raw (scanner-equal)", "",
                      endpoint_table(long, dataset, "image_ssim_improvement", 3, False, order)]
        if "image_image_residual_improvement" in set(long.loc[long.dataset.eq(dataset), "statistic"]):
            lines += ["### Image fidelity: image-residual reduction vs raw (scanner-equal)", "",
                      endpoint_table(long, dataset, "image_image_residual_improvement", 3, False, order)]
    if len(hyp):
        lines += ["## Hypotheses (read-outs, no threshold beyond the stated criterion)", ""]
        for name, group in hyp.groupby("hypothesis"):
            decided = group.loc[group.meets_criterion.notna()]
            met = int(decided.meets_criterion.astype(bool).sum())
            lines.append(f"- **{name}**: {met}/{len(decided)} cells meet `{group.criterion.iloc[0]}`"
                         if len(decided) else f"- **{name}**: {len(group)} cells reported (no criterion)")
            for row in group.itertuples():
                lines.append(f"  - {row.dataset} {PFM_LABELS.get(row.pfm, row.pfm)} {row.method} {row.statistic}: "
                             f"{row.estimate:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}] → {row.meets_criterion}")
        lines.append("")
    lines += ["## Files", "",
              "- `cell_statistics.csv`: every statistic × dataset × PFM × method × scanner (`pooled` = scanner-equal) "
              "with bootstrap CI.",
              "- `three_axis_cells.csv`: wide point estimates per cell (input to RV07).",
              "- `primary_endpoints.csv`, `hypotheses.csv`, `parity_checks.csv`, `missing_conditions.csv`, `qc.json`.",
              "- `tasks/`: per-task detectability counts, unit distances, retrieval outcomes and pair sums.", ""]
    return "\n".join(lines)


def reader_check(out: Path) -> None:
    """Smoke only: round-trip adapter shards through the RV03 file format and the reader."""
    import shutil

    from smoke_embedding_adapter import load_smoke, write_fixture

    root = out / "reader_fixture"
    written = write_fixture(root)
    report = {"fixture_files": {k: [rel(Path(p)) for p in v] for k, v in written.items()}}
    columns = ["unit", "tissue", "fold", "location_id", "core"]
    for dataset, pfm in (("pannormal", "uni_v1"), ("plism", "uni2")):
        adapter = load_smoke(dataset, pfm, None)
        reader = load(dataset, pfm, None, source="rv03", root=root, require_complete=False)
        mask = adapter.locations["unit"].isin(reader.units).to_numpy()
        expected = adapter.locations.loc[mask, columns].reset_index(drop=True)
        found = reader.locations[columns].reset_index(drop=True)
        report[dataset] = {
            "units": list(reader.units),
            "locations_equal": bool(expected.astype(str).equals(found.astype(str))),
            "condition_names_equal": sorted(reader.conditions) == sorted(adapter.conditions),
            "max_abs_feature_difference": float(max(np.abs(adapter.conditions[name][mask] - reader.conditions[name]).max()
                                                    for name in reader.conditions)),
            "methods": reader.methods(),
            "variants_example": reader.variant_names(reader.methods()[0], reader.scanners[0]),
        }
    shutil.rmtree(root)
    report["fixture_removed_after_check"] = True
    write_json(report, out / "reader_check.json")
    print(json.dumps(report, indent=2))
    ok = all(report[d]["locations_equal"] and report[d]["condition_names_equal"]
             and report[d]["max_abs_feature_difference"] == 0.0 for d in ("pannormal", "plism"))
    if not ok:
        raise SystemExit("reader round-trip check failed")


# ----------------------------------------------------------------------------- entry point

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("task", "aggregate", "list-tasks", "reader-check"))
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--source", choices=("rv03", "smoke"), default="rv03")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="run tasks although the RV03 manifest is not complete (manual checks only)")
    args = parser.parse_args()
    out = output_root(args.source)
    tasks = task_list(args.source)
    if args.command == "reader-check":
        reader_check(output_root("smoke"))
        return
    if args.command == "list-tasks":
        for index, task in enumerate(tasks):
            print(index, json.dumps(task))
        return
    if args.command == "task":
        if args.task_index is None or not 0 <= args.task_index < len(tasks):
            raise SystemExit(f"--task-index must be in [0, {len(tasks) - 1}]")
        task = tasks[args.task_index]
        print(json.dumps(task), flush=True)
        if args.source == "rv03" and not manifest_status()["complete"] and not args.allow_incomplete:
            raise SystemExit("RV03 corrected_embeddings/manifest.json is missing or not complete")
        (run_detect if task["kind"] == "detect" else run_metrics)(task, out, args.source)
        return
    aggregate(out, args.source)


if __name__ == "__main__":
    main()
