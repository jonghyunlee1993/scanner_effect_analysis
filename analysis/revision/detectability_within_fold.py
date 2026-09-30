#!/usr/bin/env python3
"""RV04b: target detectability with correction-independent training and nonlinear probes.

Reads the RV03 corrected embeddings through ``corrected_embedding_reader`` and asks whether a
classifier can still tell corrected AT2 embeddings from real target-scanner embeddings.

* PanNormal: within each locked outer fold k, only the slides held out in fold k are used
  (their corrected embeddings come from map k, and their targets were not used to fit it).
  Those slides form five inner groups (sorted slide IDs, index modulo 5); each group is
  tested once with a classifier trained on the other four.
* PLISM: corrections were fitted on PanNormal, so the RV04 section folds are kept and each
  fold-fitted variant is evaluated separately, then averaged within section.

Classifiers (fixed in the protocol): the RV04 linear probe, an MLP and a cosine k-NN.
Stages (one sbatch, ``SLURM_ARRAY_TASK_ID`` selects a task; without it, ``aggregate``):
``task`` scores one dataset x PFM x method with all three classifiers; ``aggregate`` computes
2,000-resample slide (section) bootstrap CIs with the resamples shared with RV04 and writes
``summary.md``.
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corrected_embedding_reader import (  # noqa: E402
    CV_FOLDS, DATASETS, FEATURE_METHODS, METHOD_LABELS, METHODS, PFM_LABELS, PFMS, RESULTS,
    bootstrap_weights, load, manifest_status, summarize, unit, weighted_mean, write_frame,
    write_json, write_text,
)

OUTPUT = RESULTS / "detectability_within_fold"
RV04_CELLS = RESULTS / "three_axis_evaluation/cell_statistics.csv"
CLASSIFIERS = ("linear", "mlp", "knn")
CLASSIFIER_LABELS = {"linear": "Linear", "mlp": "MLP", "knn": "k-NN"}
INNER_GROUPS = 5


def make_classifier(name: str):
    if name == "linear":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.1, max_iter=1000, solver="liblinear"))
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(256,), activation="relu", alpha=1e-3, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10, max_iter=300, random_state=0))
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=15, metric="cosine", weights="uniform")
    raise ValueError(name)


def task_list() -> list[dict]:
    return [{"dataset": dataset, "pfm": pfm, "method": method}
            for dataset in DATASETS for pfm in PFMS for method in ("raw",) + METHODS]


def splits(emb) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(split name, train mask, test mask) over locations."""
    locations = emb.locations
    fold = locations["fold"].to_numpy()
    if emb.dataset == "plism":
        return [(f"section_fold{k}", fold != k, fold == k) for k in range(CV_FOLDS) if (fold == k).any()]
    result = []
    for outer in range(CV_FOLDS):
        slides = sorted(locations.loc[fold == outer, "unit"].unique())
        group = {slide: index % INNER_GROUPS for index, slide in enumerate(slides)}
        inner = locations["unit"].map(group).to_numpy()
        for held_out in range(INNER_GROUPS):
            train = (fold == outer) & (inner != held_out)
            test = (fold == outer) & (inner == held_out)
            if test.any() and train.any():
                result.append((f"outer{outer}_inner{held_out}", train, test))
    return result


def run_task(task: dict) -> None:
    dataset, pfm, method = task["dataset"], task["pfm"], task["method"]
    stem = f"{dataset}__{pfm}__{method}"
    status_path = OUTPUT / "tasks" / f"{stem}.json"
    started = time.time()
    emb = load(dataset, pfm, methods=[] if method == "raw" else [method])
    variants = {scanner: dict(emb.variants(method, scanner)) for scanner in emb.scanners}
    scanners = [scanner for scanner in emb.scanners if variants[scanner]]
    if not scanners:
        write_json({"status": "missing_condition", "task": task}, status_path)
        return
    labels = sorted(set.intersection(*(set(variants[s]) for s in scanners)))
    unit_code = emb.locations["unit_code"].to_numpy()
    target = unit(np.concatenate([emb.target(s) for s in scanners])).astype(np.float32)
    codes = np.tile(unit_code, len(scanners))
    location_splits = splits(emb)
    rows, fits = [], 0
    for label in labels:
        generated = unit(np.concatenate([variants[s][label] for s in scanners])).astype(np.float32)
        for split_name, train_loc, test_loc in location_splits:
            train = np.tile(train_loc, len(scanners))
            test = np.tile(test_loc, len(scanners))
            x_train = np.concatenate([generated[train], target[train]])
            y_train = np.concatenate([np.zeros(train.sum()), np.ones(train.sum())])
            x_test = np.concatenate([generated[test], target[test]])
            truth = np.concatenate([np.zeros(test.sum()), np.ones(test.sum())])
            unit_test = np.concatenate([codes[test], codes[test]])
            for name in CLASSIFIERS:
                classifier = make_classifier(name)
                classifier.fit(x_train, y_train)
                fits += 1
                correct = (classifier.predict(x_test) == truth).astype(int)
                frame = pd.DataFrame({"unit_code": unit_test, "correct": correct})
                grouped = frame.groupby("unit_code", as_index=False).agg(
                    n=("correct", "size"), correct=("correct", "sum"))
                grouped["classifier"] = name
                grouped["variant"] = label
                grouped["split"] = split_name
                rows.append(grouped)
    result = pd.concat(rows, ignore_index=True)
    result["unit"] = np.asarray(emb.units)[result["unit_code"].to_numpy()]
    result.insert(0, "method", method)
    result.insert(0, "pfm", pfm)
    result.insert(0, "dataset", dataset)
    write_frame(result.drop(columns="unit_code"), OUTPUT / "tasks" / f"{stem}.csv.gz")
    write_json({"status": "pass", "task": task, "scanners": scanners, "variants": labels,
                "splits": [name for name, _, _ in location_splits], "fits": fits,
                "n_locations": emb.n, "n_units": len(emb.units),
                "seconds": time.time() - started}, status_path)


def unit_accuracy(frame: pd.DataFrame, units: list[str]) -> np.ndarray:
    """Accuracy per unit: pooled over splits within variant, then mean over variants."""
    per_variant = frame.groupby(["unit", "variant"])[["correct", "n"]].sum()
    per_variant = (per_variant["correct"] / per_variant["n"]).groupby(level="unit").mean()
    return per_variant.reindex(units).to_numpy(dtype=np.float64)


def rv04_reference() -> pd.DataFrame:
    if not RV04_CELLS.exists():
        return pd.DataFrame(columns=["dataset", "pfm", "method", "rv04_linear"])
    cells = pd.read_csv(RV04_CELLS)
    cells = cells[(cells.statistic == "detectability") & (cells.scanner == "pooled")]
    return cells.rename(columns={"estimate": "rv04_linear"})[["dataset", "pfm", "method", "rv04_linear"]]


def aggregate() -> None:
    tasks = task_list()
    statuses = [json.loads((OUTPUT / "tasks" / f"{t['dataset']}__{t['pfm']}__{t['method']}.json").read_text())
                for t in tasks]
    failed = [s for s in statuses if s.get("status") != "pass"]
    if failed:
        raise SystemExit(f"{len(failed)} tasks did not pass: {failed[:3]}")
    frames = pd.concat([pd.read_csv(OUTPUT / "tasks" / f"{t['dataset']}__{t['pfm']}__{t['method']}.csv.gz",
                                    dtype={"unit": str}) for t in tasks], ignore_index=True)
    rows = []
    for dataset in DATASETS:
        units = sorted(frames.loc[frames.dataset == dataset, "unit"].unique())
        weights = bootstrap_weights(dataset, len(units))
        for pfm in PFMS:
            for method in ("raw",) + METHODS:
                cell = frames[(frames.dataset == dataset) & (frames.pfm == pfm) & (frames.method == method)]
                accuracy = {name: unit_accuracy(cell[cell.classifier == name], units) for name in CLASSIFIERS}
                for name in CLASSIFIERS:
                    rows.append({"dataset": dataset, "pfm": pfm, "method": method, "statistic": name,
                                 **summarize(weighted_mean(weights, accuracy[name]))})
                for name in ("mlp", "knn"):
                    rows.append({"dataset": dataset, "pfm": pfm, "method": method,
                                 "statistic": f"{name}_minus_linear",
                                 **summarize(weighted_mean(weights, accuracy[name] - accuracy["linear"]))})
    table = pd.DataFrame(rows)
    write_frame(table, OUTPUT / "detectability.csv")
    write_text(render_summary(table, statuses), OUTPUT / "summary.md")
    write_json({"tasks": len(statuses), "fits": int(sum(s["fits"] for s in statuses))}, OUTPUT / "qc.json")


def fmt(row) -> str:
    return f"{row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]"


def render_summary(table: pd.DataFrame, statuses: list[dict]) -> str:
    reference = rv04_reference()
    lines = ["# RV04b target detectability: correction-independent and nonlinear probes", "",
             "## What ran", "",
             "- Embeddings: RV03 `corrected_embeddings` via `corrected_embedding_reader`; inputs L2-normalized.",
             "- PanNormal: within each locked outer fold, slides held out in that fold only; five inner slide "
             "groups (sorted IDs, index mod 5). PLISM: RV04 section folds; fold-fitted variants averaged within section.",
             "- Classes: corrected AT2 vs real target, pooled over directions; accuracy per location → unit → mean.",
             "- Classifiers: linear (StandardScaler + LogisticRegression C = 0.1, liblinear; as RV04); MLP "
             "(StandardScaler + MLPClassifier 256 ReLU, alpha 1e-3, early stopping 10%, patience 10, max 300, seed 0); "
             "k-NN (k = 15, cosine).",
             "- CIs: 2,000 slide (section) bootstrap resamples shared with RV04; differences are paired within unit.",
             f"- Tasks: {len(statuses)}; classifier fits: {sum(s['fits'] for s in statuses)}.", ""]
    for dataset in DATASETS:
        lines += [f"## {'PanNormal' if dataset == 'pannormal' else 'PLISM'}", ""]
        for statistic, title in (("linear", "Linear"), ("mlp", "MLP"), ("knn", "k-NN"),
                                 ("mlp_minus_linear", "MLP − linear"), ("knn_minus_linear", "k-NN − linear")):
            lines += [f"### {title}", "", "| Method | " + " | ".join(PFM_LABELS[p] for p in PFMS) + " |",
                      "| --- |" + " --- |" * len(PFMS)]
            for method in ("raw",) + METHODS:
                cells = []
                for pfm in PFMS:
                    row = table[(table.dataset == dataset) & (table.pfm == pfm) & (table.method == method)
                                & (table.statistic == statistic)].iloc[0]
                    cells.append(fmt(row))
                lines.append(f"| {METHOD_LABELS.get(method, method)} | " + " | ".join(cells) + " |")
            lines.append("")
        if not reference.empty:
            lines += ["### RV04 cross-fitted linear detectability (reference)", "",
                      "| Method | " + " | ".join(PFM_LABELS[p] for p in PFMS) + " |", "| --- |" + " --- |" * len(PFMS)]
            for method in ("raw",) + METHODS:
                values = []
                for pfm in PFMS:
                    match = reference[(reference.dataset == dataset) & (reference.pfm == pfm) & (reference.method == method)]
                    values.append(f"{match.rv04_linear.iloc[0]:.3f}" if len(match) else "—")
                lines.append(f"| {METHOD_LABELS.get(method, method)} | " + " | ".join(values) + " |")
            lines.append("")
    lines += ["## Hypotheses (mechanical read-outs)", ""]
    for method in ("combat", "ols"):
        for pfm in PFMS:
            row = table[(table.dataset == "pannormal") & (table.pfm == pfm) & (table.method == method)
                        & (table.statistic == "linear")].iloc[0]
            lines.append(f"- H4e PanNormal {PFM_LABELS[pfm]} {METHOD_LABELS.get(method, method)} linear {fmt(row)} → "
                         f"{'not below chance' if row.ci_high >= 0.5 else 'still below chance'}")
    for method in FEATURE_METHODS:
        for pfm in PFMS:
            row = table[(table.dataset == "pannormal") & (table.pfm == pfm) & (table.method == method)
                        & (table.statistic == "mlp_minus_linear")].iloc[0]
            lines.append(f"- H4f PanNormal {PFM_LABELS[pfm]} {METHOD_LABELS.get(method, method)} MLP − linear {fmt(row)} → "
                         f"{row.ci_low > 0}")
    raw = table[(table.dataset == "pannormal") & (table.method == "raw") & (table.statistic == "linear")]
    lines += ["", "- Design check (raw, PanNormal, linear): " +
              ", ".join(f"{PFM_LABELS[r.pfm]} {r.estimate:.3f}" for r in raw.itertuples()), "",
              "## Files", "", "- `detectability.csv`: every statistic × dataset × PFM × method with bootstrap CI.",
              "- `tasks/`: per-unit correct counts per classifier, variant and split; `qc.json`.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("task", "aggregate", "list-tasks"))
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    tasks = task_list()
    if args.command == "list-tasks":
        for index, task in enumerate(tasks):
            print(index, json.dumps(task))
        return
    if not manifest_status()["complete"]:
        raise SystemExit("RV03 corrected_embeddings/manifest.json is missing or not complete")
    if args.command == "task":
        if args.task_index is None or not 0 <= args.task_index < len(tasks):
            raise SystemExit(f"--task-index must be in [0, {len(tasks) - 1}]")
        print(json.dumps(tasks[args.task_index]), flush=True)
        run_task(tasks[args.task_index])
        return
    aggregate()


if __name__ == "__main__":
    main()
