#!/usr/bin/env python3
"""RV05: downstream proxy — cross-scanner tissue classification.

For each PFM and target scanner a tissue classifier is built on real target-scanner
location embeddings and applied to the held-out slide under the real target (reference),
raw AT2 and every corrected AT2 condition (RV03 embeddings via
``corrected_embedding_reader``).

* Primary classifier: leave-one-slide-out nearest centroid with cosine similarity.
  Location embeddings are unit-normalized; each tissue centroid is the mean of the real
  target embeddings of all other slides (tissues with at least two slides: 36 tissues,
  102 slides). PLISM leaves one section out (46 cores, every section holds all cores).
* Secondary classifier: multinomial logistic regression (StandardScaler +
  LogisticRegression(C=0.1, lbfgs), the regularization of the RV04 detectability
  classifier) with the same leave-one-out design and the same unit-normalized inputs.
* Endpoint: tissue macro recall = location accuracy averaged within slide (PLISM:
  section x core), then within tissue, then over tissues; scanner-equal mean over
  directions. Recovery = (corrected − raw) / (real target − raw), computed only for cells
  with real target − raw ≥ 2 percentage points; if most scanner x PFM cells fall below that
  gap, recovery is not computed and the result is reported as such (protocol rule).
* Secondary: Spearman correlation across method x scanner x PFM cells between recovery
  and (a) target gain, (b) SSIM improvement and image-residual reduction (image methods).
* 95% CIs: the shared 2,000 slide (PLISM: section) bootstrap resamples; fold-fitted
  PLISM conditions are averaged with equal weight within each section x core.

Stages (one sbatch): with ``SLURM_ARRAY_TASK_ID`` a ``task`` (centroid per dataset x PFM,
logistic per dataset x PFM x scanner); without it ``aggregate``. ``--source smoke`` uses
the smoke adapter and writes to ``results/downstream_tissue_proxy/smoke``.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corrected_embedding_reader import (  # noqa: E402
    BOOTSTRAPS, DATASETS, EMBEDDINGS, IMAGE_METHODS, METHOD_LABELS, METHODS, PFM_LABELS, PFMS,
    RESULTS, TARGETS, bootstrap_weights, image_improvement, image_metric_arrays, load,
    load_image_metrics, manifest_status, rel, sha256, spearman_rows, summarize, unit,
    weighted_macro, weighted_mean, write_frame, write_json, write_text,
)


OUTPUT = RESULTS / "downstream_tissue_proxy"
SMOKE_PFM = {"pannormal": "uni_v1", "plism": "uni2"}
GAP_THRESHOLD = 0.02
CLASSIFIERS = ("centroid", "logistic")


def output_root(source: str) -> Path:
    return OUTPUT / "smoke" if source == "smoke" else OUTPUT


def task_list(source: str) -> list[dict]:
    pairs = ([(d, SMOKE_PFM[d]) for d in DATASETS] if source == "smoke"
             else [(d, p) for d in DATASETS for p in PFMS])
    tasks = [{"kind": "centroid", "dataset": d, "pfm": p} for d, p in pairs]
    tasks += [{"kind": "logistic", "dataset": d, "pfm": p, "scanner": s} for d, p in pairs for s in TARGETS[d]]
    return tasks


def eligible_classes(emb) -> tuple[np.ndarray, list[str]]:
    """Class index per location (−1 if its tissue has fewer than two slides/sections)."""
    units_per_tissue = emb.locations.groupby("tissue")["unit"].nunique()
    classes = sorted(units_per_tissue.index[units_per_tissue >= 2])
    lookup = {name: index for index, name in enumerate(classes)}
    return emb.locations["tissue"].map(lookup).fillna(-1).astype(int).to_numpy(), classes


def condition_list(emb, scanner: str) -> list[tuple[str, str, np.ndarray]]:
    items = []
    for method in ["target", "raw"] + emb.methods():
        for label, matrix in emb.variants(method, scanner):
            items.append((method, label, matrix))
    return items


def outcome_frame(emb, method, scanner, label, rows_mask, correct) -> pd.DataFrame:
    locations = emb.locations.loc[rows_mask]
    frame = pd.DataFrame({"query": locations["query"].to_numpy(), "unit": locations["unit"].to_numpy(),
                          "tissue": locations["tissue"].to_numpy(), "correct": correct.astype(int)})
    grouped = frame.groupby(["query", "unit", "tissue"], as_index=False).agg(
        n=("correct", "size"), correct=("correct", "sum"))
    grouped.insert(0, "variant", label)
    grouped.insert(0, "scanner", scanner)
    grouped.insert(0, "method", method)
    return grouped


def run_centroid(task: dict, out: Path, source: str) -> None:
    dataset, pfm = task["dataset"], task["pfm"]
    stem = f"{dataset}__{pfm}"
    status_path = out / "tasks/centroid" / f"{stem}.json"
    started = time.time()
    try:
        emb = load(dataset, pfm, None, source)
    except FileNotFoundError as error:
        write_json({"status": "missing_input", "task": task, "error": str(error)}, status_path)
        return
    labels, classes = eligible_classes(emb)
    unit_code = emb.locations["unit_code"].to_numpy()
    eligible = labels >= 0
    outcomes, gains = [], []
    for scanner in emb.scanners:
        target = unit(emb.target(scanner))
        items = [(method, label, unit(matrix)) for method, label, matrix in condition_list(emb, scanner)]
        sums = np.zeros((len(emb.units), len(classes), target.shape[1]))
        counts = np.zeros((len(emb.units), len(classes)))
        np.add.at(sums, (unit_code[eligible], labels[eligible]), target[eligible])
        np.add.at(counts, (unit_code[eligible], labels[eligible]), 1.0)
        total, total_count = sums.sum(axis=0), counts.sum(axis=0)
        correct = {(method, label): np.zeros(emb.n, dtype=int) for method, label, _ in items}
        for held_out in range(len(emb.units)):
            rows = np.flatnonzero(eligible & (unit_code == held_out))
            if len(rows) == 0:
                continue
            remaining = total_count - counts[held_out]
            centroids = unit((total - sums[held_out]) / np.maximum(remaining, 1.0)[:, None])
            for method, label, vectors in items:
                scores = vectors[rows] @ centroids.T
                scores[:, remaining == 0] = -np.inf
                correct[(method, label)][rows] = (np.argmax(scores, axis=1) == labels[rows]).astype(int)
        raw_distance = 1.0 - np.sum(unit(emb.raw()) * target, axis=1)
        for method, label, vectors in items:
            outcomes.append(outcome_frame(emb, method, scanner, label, eligible, correct[(method, label)][eligible]))
            gain = emb.unit_means(raw_distance - (1.0 - np.sum(vectors * target, axis=1)))
            gains.append(pd.DataFrame({"method": method, "scanner": scanner, "variant": label,
                                       "unit": emb.units, "target_gain": gain}))
    write_frame(pd.concat(outcomes, ignore_index=True), out / "tasks/centroid" / f"{stem}_outcomes.csv.gz")
    write_frame(pd.concat(gains, ignore_index=True), out / "tasks/centroid" / f"{stem}_gains.csv.gz")
    write_json({"status": "pass", "task": task, "source": emb.source, "n_locations": emb.n,
                "n_units": len(emb.units), "n_classes": len(classes),
                "n_evaluated_locations": int(eligible.sum()),
                "n_evaluated_units": int(len(np.unique(unit_code[eligible]))),
                "methods": ["target", "raw"] + emb.methods(), "notes": emb.notes,
                "seconds": time.time() - started}, status_path)


def logistic_classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


def run_logistic(task: dict, out: Path, source: str) -> None:
    dataset, pfm, scanner = task["dataset"], task["pfm"], task["scanner"]
    stem = f"{dataset}__{pfm}__{scanner}"
    status_path = out / "tasks/logistic" / f"{stem}.json"
    started = time.time()
    try:
        emb = load(dataset, pfm, None, source)
    except FileNotFoundError as error:
        write_json({"status": "missing_input", "task": task, "error": str(error)}, status_path)
        return
    labels, classes = eligible_classes(emb)
    unit_code = emb.locations["unit_code"].to_numpy()
    eligible = labels >= 0
    target = unit(emb.target(scanner))
    items = [(method, label, unit(matrix)) for method, label, matrix in condition_list(emb, scanner)]
    correct = {(method, label): np.zeros(emb.n, dtype=int) for method, label, _ in items}
    unconverged = 0
    for held_out in range(len(emb.units)):
        rows = np.flatnonzero(eligible & (unit_code == held_out))
        if len(rows) == 0:
            continue
        train = eligible & (unit_code != held_out)
        classifier = logistic_classifier()
        classifier.fit(target[train], labels[train])
        unconverged += int(classifier[-1].n_iter_.max() >= classifier[-1].max_iter)
        for method, label, vectors in items:
            correct[(method, label)][rows] = (classifier.predict(vectors[rows]) == labels[rows]).astype(int)
    outcomes = [outcome_frame(emb, method, scanner, label, eligible, correct[(method, label)][eligible])
                for method, label, _ in items]
    write_frame(pd.concat(outcomes, ignore_index=True), out / "tasks/logistic" / f"{stem}_outcomes.csv.gz")
    write_json({"status": "pass", "task": task, "source": emb.source, "n_classes": len(classes),
                "n_fits": int(len(np.unique(unit_code[eligible]))), "unconverged_fits": unconverged,
                "notes": emb.notes, "seconds": time.time() - started}, status_path)


# ----------------------------------------------------------------------------- aggregate

def macro_draws(outcomes: pd.DataFrame, queries: pd.DataFrame, weights_query: np.ndarray,
                tissue_code: np.ndarray, n_tissues: int) -> np.ndarray:
    """Macro recall draws: accuracy within query (variants averaged), then tissue, then mean."""
    accuracy = (outcomes["correct"] / outcomes["n"]).groupby(outcomes["query"]).mean()
    return weighted_macro(weights_query, accuracy.reindex(queries["query"]).to_numpy(dtype=np.float64),
                          tissue_code, n_tissues)


def collect(dataset: str, classifier: str, pfm: str, out: Path) -> pd.DataFrame | None:
    folder = out / "tasks" / classifier
    if classifier == "centroid":
        path = folder / f"{dataset}__{pfm}_outcomes.csv.gz"
        return pd.read_csv(path, dtype={"unit": str, "query": str}) if path.exists() else None
    frames = [pd.read_csv(folder / f"{dataset}__{pfm}__{scanner}_outcomes.csv.gz", dtype={"unit": str, "query": str})
              for scanner in TARGETS[dataset] if (folder / f"{dataset}__{pfm}__{scanner}_outcomes.csv.gz").exists()]
    return pd.concat(frames, ignore_index=True) if frames else None


def aggregate(out: Path, source: str) -> None:
    tasks = task_list(source)
    datasets = [d for d in DATASETS if any(t["dataset"] == d for t in tasks)]
    macro_rows, recovery_rows, pooled_rows, spearman_out, rule_rows, qc, provenance = [], [], [], [], [], {}, {}
    for status_path in sorted((out / "tasks").glob("*/*.json")):
        status = json.loads(status_path.read_text())
        qc[f"{status_path.parent.name}/{status_path.stem}"] = {
            k: status.get(k) for k in ("status", "n_classes", "n_evaluated_units", "n_evaluated_locations",
                                        "n_fits", "unconverged_fits", "seconds")}
    for dataset in datasets:
        pfms = [t["pfm"] for t in tasks if t["kind"] == "centroid" and t["dataset"] == dataset]
        image_table, image_source = load_image_metrics(dataset, source)
        provenance[dataset] = image_source
        scanners = TARGETS[dataset]
        for classifier in CLASSIFIERS:
            cells = {}
            for pfm in pfms:
                outcomes = collect(dataset, classifier, pfm, out)
                gains_path = out / "tasks/centroid" / f"{dataset}__{pfm}_gains.csv.gz"
                if outcomes is None or not gains_path.exists():
                    continue
                gains = pd.read_csv(gains_path, dtype={"unit": str})
                units = tuple(sorted(gains["unit"].unique()))
                weights = bootstrap_weights(dataset, len(units))
                unit_index = pd.Index(units)
                queries = outcomes.drop_duplicates("query")[["query", "unit", "tissue"]].sort_values("query")
                queries = queries.reset_index(drop=True)
                tissues = sorted(queries.tissue.unique())
                tissue_code = queries.tissue.map({t: i for i, t in enumerate(tissues)}).to_numpy()
                weights_query = weights[:, unit_index.get_indexer(queries.unit)]
                image_arrays = image_metric_arrays(image_table, units)
                for scanner in scanners:
                    block = outcomes.loc[outcomes.scanner.eq(scanner)]
                    for method in block.method.unique():
                        draws = macro_draws(block.loc[block.method.eq(method)], queries, weights_query,
                                            tissue_code, len(tissues))
                        entry = {"macro": draws}
                        g = gains.loc[gains.method.eq(method) & gains.scanner.eq(scanner)]
                        if len(g):
                            values = g.groupby("unit")["target_gain"].mean().reindex(unit_index).to_numpy(float)
                            entry["target_gain"] = weighted_mean(weights, values)
                        for metric in ("ssim", "image_residual", "lpips"):
                            improvement = image_improvement(image_arrays, method, scanner, metric)
                            if improvement is not None and np.isfinite(improvement).any():
                                entry[f"{metric}_improvement"] = weighted_mean(weights, improvement)
                        cells[(pfm, method, scanner)] = entry
                        macro_rows.append({"dataset": dataset, "classifier": classifier, "pfm": pfm,
                                           "method": method, "scanner": scanner, "n_tissues": len(tissues),
                                           "n_queries": len(queries), **summarize(draws)})
                methods = sorted({m for (p, m, s) in cells if p == pfm}, key=lambda m: (["target", "raw"] + list(METHODS)).index(m)
                                 if m in ["target", "raw"] + list(METHODS) else 99)
                for method in methods:
                    present = [s for s in scanners if (pfm, method, s) in cells]
                    if len(present) == len(scanners):
                        draws = np.mean([cells[(pfm, method, s)]["macro"] for s in present], axis=0)
                        cells[(pfm, method, "pooled")] = {"macro": draws}
                        macro_rows.append({"dataset": dataset, "classifier": classifier, "pfm": pfm,
                                           "method": method, "scanner": "pooled", "n_tissues": len(tissues),
                                           "n_queries": len(queries), **summarize(draws)})
            if not cells:
                continue
            # Gap rule over scanner x PFM cells.
            gap_cells = {}
            for (pfm, method, scanner), entry in cells.items():
                if method == "target" and scanner != "pooled" and (pfm, "raw", scanner) in cells:
                    gap_cells[(pfm, scanner)] = entry["macro"] - cells[(pfm, "raw", scanner)]["macro"]
            eligible_cells = {key for key, draws in gap_cells.items() if draws[0] >= GAP_THRESHOLD}
            below = len(gap_cells) - len(eligible_cells)
            compute_recovery = len(gap_cells) > 0 and below <= len(gap_cells) / 2
            rule_rows.append({"dataset": dataset, "classifier": classifier, "scanner_pfm_cells": len(gap_cells),
                              "cells_with_gap_ge_2pp": len(eligible_cells), "cells_below_2pp": below,
                              "recovery_computed": compute_recovery,
                              "gap_min": min((d[0] for d in gap_cells.values()), default=np.nan),
                              "gap_median": float(np.median([d[0] for d in gap_cells.values()])) if gap_cells else np.nan,
                              "gap_max": max((d[0] for d in gap_cells.values()), default=np.nan)})
            spearman_cells = []
            for (pfm, method, scanner), entry in cells.items():
                if method in ("target", "raw") or scanner == "pooled":
                    continue
                raw = cells[(pfm, "raw", scanner)]["macro"]
                gap = gap_cells.get((pfm, scanner))
                is_eligible = (pfm, scanner) in eligible_cells
                recovery = (entry["macro"] - raw) / gap if (compute_recovery and is_eligible) else None
                row = {"dataset": dataset, "classifier": classifier, "pfm": pfm, "method": method,
                       "scanner": scanner, "gap": float(gap[0]) if gap is not None else np.nan,
                       "gap_eligible": is_eligible, "recovery_computed": compute_recovery}
                row.update({f"corrected_minus_raw_{k}": v for k, v in summarize(entry["macro"] - raw).items()})
                if recovery is not None:
                    row.update({f"recovery_{k}": v for k, v in summarize(recovery).items()})
                for name in ("target_gain", "ssim_improvement", "image_residual_improvement", "lpips_improvement"):
                    if name in entry:
                        row[name] = float(entry[name][0])
                recovery_rows.append(row)
                if recovery is not None:
                    spearman_cells.append({"key": (pfm, method, scanner), "recovery": recovery,
                                           **{name: entry.get(name) for name in
                                              ("target_gain", "ssim_improvement", "image_residual_improvement")}})
            # Descriptive pooled recovery per method x PFM over gap-eligible scanners.
            if compute_recovery:
                for pfm in pfms:
                    for method in sorted({m for (p, m, s) in cells if p == pfm} - {"raw", "target"}):
                        present = [s for s in scanners if (pfm, s) in eligible_cells and (pfm, method, s) in cells]
                        if not present:
                            continue
                        numerator = np.mean([cells[(pfm, method, s)]["macro"] - cells[(pfm, "raw", s)]["macro"]
                                             for s in present], axis=0)
                        denominator = np.mean([gap_cells[(pfm, s)] for s in present], axis=0)
                        pooled_rows.append({"dataset": dataset, "classifier": classifier, "pfm": pfm,
                                            "method": method, "eligible_scanners": ",".join(present),
                                            **summarize(numerator / denominator)})
                for association, name, subset in (
                    ("a_target_gain", "target_gain", None),
                    ("b_ssim_improvement", "ssim_improvement", IMAGE_METHODS),
                    ("b_image_residual_improvement", "image_residual_improvement", IMAGE_METHODS),
                ):
                    chosen = [c for c in spearman_cells if c.get(name) is not None
                              and (subset is None or c["key"][1] in subset)]
                    if len(chosen) < 3:
                        spearman_out.append({"dataset": dataset, "classifier": classifier, "association": association,
                                             "n_cells": len(chosen), "estimate": np.nan, "ci_low": np.nan,
                                             "ci_high": np.nan, "note": "fewer than 3 cells with both values"})
                        continue
                    x = np.stack([c["recovery"] for c in chosen], axis=1)
                    y = np.stack([c[name] for c in chosen], axis=1)
                    rho = spearman_rows(x, y)
                    spearman_out.append({"dataset": dataset, "classifier": classifier, "association": association,
                                         "n_cells": len(chosen), **summarize(rho),
                                         "methods": ",".join(sorted({c["key"][1] for c in chosen})),
                                         "note": ""})
    macro = pd.DataFrame(macro_rows)
    recovery = pd.DataFrame(recovery_rows)
    pooled = pd.DataFrame(pooled_rows)
    spearman = pd.DataFrame(spearman_out)
    rules = pd.DataFrame(rule_rows)
    for frame, name in ((macro, "macro_recall.csv"), (recovery, "recovery_cells.csv"),
                        (pooled, "recovery_pooled.csv"), (spearman, "spearman.csv"), (rules, "gap_rule.csv")):
        write_frame(frame, out / name)
    manifest = manifest_status()
    write_json({"source": source, "embedding_manifest": manifest, "tasks": qc, "image_fidelity": provenance,
                "bootstrap_replicates": BOOTSTRAPS, "gap_threshold": GAP_THRESHOLD,
                "code_sha256": sha256(Path(__file__))}, out / "qc.json")
    write_text(render_summary(macro, recovery, pooled, spearman, rules, qc, provenance, source, manifest),
               out / "summary.md")
    print((out / "summary.md").read_text())


def fmt(row, digits=1, percent=True) -> str:
    if row is None or not np.isfinite(row["estimate"]):
        return "n/a"
    scale = 100.0 if percent else 1.0
    text = f"{scale * row['estimate']:.{digits}f}"
    if np.isfinite(row["ci_low"]):
        text += f" [{scale * row['ci_low']:.{digits}f}, {scale * row['ci_high']:.{digits}f}]"
    return text


def table(frame: pd.DataFrame, value_filter: dict, methods: list[str], digits: int, percent: bool) -> str:
    block = frame
    for key, value in value_filter.items():
        block = block.loc[block[key].eq(value)]
    pfms = [p for p in PFMS if p in set(block.pfm)] + sorted(set(block.pfm) - set(PFMS))
    if not pfms:
        return "_no values_\n"
    lines = ["| Method | " + " | ".join(PFM_LABELS.get(p, p) for p in pfms) + " |",
             "| --- | " + " | ".join("---" for _ in pfms) + " |"]
    for method in methods:
        cells = []
        for pfm in pfms:
            match = block.loc[block.pfm.eq(pfm) & block.method.eq(method)]
            cells.append(fmt(match.iloc[0], digits, percent) if len(match) else "–")
        if any(cell != "–" for cell in cells):
            lines.append(f"| {METHOD_LABELS.get(method, method)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_summary(macro, recovery, pooled, spearman, rules, qc, provenance, source, manifest) -> str:
    lines = ["# RV05 downstream proxy: cross-scanner tissue classification", ""]
    if source == "smoke":
        lines += ["**SMOKE TEST** on the smoke adapter (PanNormal UNI v1 from `03_uni`; PLISM UNI2-h raw + identity "
                  "fixture). Not a revision result.", ""]
    lines += ["## What ran", "",
              f"- Embeddings `{rel(EMBEDDINGS)}` (manifest {manifest}), source `{source}`.",
              "- Nearest centroid (primary): unit-normalized location embeddings; tissue centroids from real target "
              "embeddings of all other slides (PLISM: other sections); cosine; tissues with ≥ 2 slides only "
              "(PanNormal 36 tissues / 102 slides; PLISM 46 cores / 13 sections).",
              "- Multinomial logistic regression (secondary): StandardScaler + LogisticRegression(C = 0.1, lbfgs) with "
              "the same leave-one-slide/section-out design.",
              "- Macro recall: location accuracy within slide (PLISM section × core), then tissue mean, then mean over "
              "tissues; scanner-equal mean over directions. PLISM fold-fitted conditions averaged within section × core.",
              f"- Recovery = (corrected − raw)/(real target − raw) only where real target − raw ≥ {100 * GAP_THRESHOLD:.0f} pp; "
              "if most scanner × PFM cells are below that gap, recovery is not computed.",
              "- Spearman across method × scanner × PFM cells (gap-eligible): recovery vs (a) target gain (cosine, "
              "all methods), (b) SSIM improvement and image-residual reduction (image methods only; feature methods "
              "do not change the image).",
              f"- CIs: {BOOTSTRAPS} shared slide (PLISM section) bootstrap resamples; Spearman CIs recompute every "
              "cell in each resample (eligibility fixed at the point estimate).",
              f"- Image fidelity: PanNormal — {provenance.get('pannormal', 'n/a')}; PLISM — {provenance.get('plism', 'n/a')}.",
              ""]
    lines += ["## QC", ""]
    for key, value in qc.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    order = ["target", "raw"] + list(METHODS) + sorted(set(macro.method) - set(METHODS) - {"target", "raw"}) if len(macro) else []
    for dataset in DATASETS:
        if not len(macro) or dataset not in set(macro.dataset):
            continue
        for classifier in CLASSIFIERS:
            label = "nearest centroid (primary)" if classifier == "centroid" else "logistic regression (secondary)"
            rule = rules.loc[rules.dataset.eq(dataset) & rules.classifier.eq(classifier)]
            lines += [f"## {dataset} — {label}", "", "### Tissue macro recall, scanner-equal (%)", "",
                      table(macro, {"dataset": dataset, "classifier": classifier, "scanner": "pooled"}, order, 1, True)]
            if len(rule):
                r = rule.iloc[0]
                lines.append(f"- Gap rule: {r.cells_with_gap_ge_2pp}/{r.scanner_pfm_cells} scanner × PFM cells have "
                             f"real target − raw ≥ 2 pp (gap min {100 * r.gap_min:.1f}, median {100 * r.gap_median:.1f}, "
                             f"max {100 * r.gap_max:.1f} pp); recovery computed: {bool(r.recovery_computed)}.")
                if not r.recovery_computed:
                    lines.append("- Most cells are below the 2 pp gap: scanner shift has little effect on coarse tissue "
                                 "identity for this classifier; recovery not computed (protocol rule).")
                lines.append("")
            block = pooled.loc[pooled.dataset.eq(dataset) & pooled.classifier.eq(classifier)] if len(pooled) else pooled
            if len(block):
                lines += ["### Pooled recovery over gap-eligible scanners (descriptive)", "",
                          table(block, {}, order, 2, False)]
            sp = spearman.loc[spearman.dataset.eq(dataset) & spearman.classifier.eq(classifier)] if len(spearman) else spearman
            if len(sp):
                lines += ["### Spearman association with recovery", ""]
                for row in sp.itertuples():
                    lines.append(f"- {row.association}: rho {row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}] "
                                 f"over {row.n_cells} cells {row.note}")
                lines.append("")
    lines += ["## Files", "",
              "- `macro_recall.csv` (every dataset × classifier × PFM × method × scanner, `pooled` = scanner-equal), "
              "`recovery_cells.csv`, `recovery_pooled.csv`, `spearman.csv`, `gap_rule.csv`, `qc.json`; per-task "
              "outcomes in `tasks/`.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("task", "aggregate", "list-tasks"))
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--source", choices=("rv03", "smoke"), default="rv03")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="run tasks although the RV03 manifest is not complete (manual checks only)")
    args = parser.parse_args()
    out = output_root(args.source)
    tasks = task_list(args.source)
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
        (run_centroid if task["kind"] == "centroid" else run_logistic)(task, out, args.source)
        return
    aggregate(out, args.source)


if __name__ == "__main__":
    main()
