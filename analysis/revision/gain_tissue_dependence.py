#!/usr/bin/env python3
"""RV09: tissue dependence of correction gains.

Slide x scanner target gains (mean over the 20 ``set20`` locations of raw minus corrected
cosine target distance; RV03 embeddings) for Reinhard, color + frequency, the frequency
increment (color + frequency minus Reinhard, i.e. Reinhard distance minus combined
distance), ridge affine and ComBat, in UNI v1 (primary) and UNI2-h, Virchow2 and
H-optimus-1 (secondary).

Each PFM x method is fitted with the RQ1 slide-level model, reused unchanged from
``analysis/paper/slide_level_lmm.py`` (``run``):

    gain[s, r] = beta[s] + a[t(r)] + u[s, t(r)] + c[r] + error[s, r]

fixed scanner means, random shared tissue, scanner x tissue and slide effects; REML;
restricted likelihood-ratio tests of any tissue dependence (a and u) and of
scanner-specific tissue dependence (u), each with 500 parametric bootstrap replicates;
Benjamini-Hochberg across the five methods within each PFM (``aggregate``).

Secondary: Spearman correlation across tissue types, for AKOYA and GT450, between the
tissue-mean high-frequency log2 transfer ratio (endpoint ``frequency_high`` of
``analysis/paper/results/direct_slide_lmm/direct_slide_contrasts.csv``, all matched
locations) and the tissue-mean frequency increment (``set20``), with a 2,000-resample
slide bootstrap CI (shared resamples with RV04/RV05).

Stages (one sbatch): with ``SLURM_ARRAY_TASK_ID`` one PFM x method (build the gain
contrasts, fit, bootstrap); without it ``aggregate``. ``--source smoke`` uses the smoke
adapter (UNI v1 Reinhard / combined / increment) into ``results/gain_tissue_dependence/smoke``.
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
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT / "analysis/paper"))

from corrected_embedding_reader import (  # noqa: E402
    BOOTSTRAPS, EMBEDDINGS, PFM_LABELS, PFMS, RESULTS, TARGETS, bootstrap_weights, load,
    manifest_status, rel, sha256, spearman_rows, summarize, unit, weighted_mean, write_frame,
    write_json, write_text,
)
import slide_level_lmm  # noqa: E402
from joint_interaction_lmm import benjamini_hochberg  # noqa: E402


OUTPUT = RESULTS / "gain_tissue_dependence"
DIRECT = PROJECT / "analysis/paper/results/direct_slide_lmm/direct_slide_contrasts.csv"
METHODS = ("reinhard", "combined", "frequency_increment", "ridge", "combat")
NEEDS = {"reinhard": ("reinhard",), "combined": ("combined",), "frequency_increment": ("reinhard", "combined"),
         "ridge": ("ridge",), "combat": ("combat",)}
LABELS = {"reinhard": "Reinhard target gain", "combined": "Color + frequency target gain",
          "frequency_increment": "Frequency increment (color + frequency − Reinhard)",
          "ridge": "Ridge affine target gain", "combat": "ComBat target gain"}
SMOKE_METHODS = ("reinhard", "combined", "frequency_increment")
SEED = 20260929
SPEARMAN_SCANNERS = ("akoya", "gt450")
TESTS = ("any_tissue_dependence", "scanner_specific_tissue_dependence")


def output_root(source: str) -> Path:
    return OUTPUT / "smoke" if source == "smoke" else OUTPUT


def task_list(source: str) -> list[dict]:
    if source == "smoke":
        return [{"pfm": "uni_v1", "method": method} for method in SMOKE_METHODS]
    return [{"pfm": pfm, "method": method} for pfm in PFMS for method in METHODS]


def slide_gains(emb, method: str) -> pd.DataFrame:
    """Slide x scanner gains (slide mean over set20 locations) in the RQ1 input format."""
    rows = []
    slides = emb.locations.drop_duplicates("unit").set_index("unit")["tissue"]
    counts = emb.locations.groupby("unit").size()
    for scanner in emb.scanners:
        target = unit(emb.target(scanner))

        def distance(name: str) -> np.ndarray:
            variants = emb.variants(name, scanner)
            if len(variants) != 1:
                raise KeyError(f"{name}_to_{scanner}: expected one PanNormal condition, found {len(variants)}")
            return 1.0 - np.sum(unit(variants[0][1]) * target, axis=1)

        if method == "frequency_increment":
            values = distance("reinhard") - distance("combined")
        else:
            values = distance("raw") - distance(method)
        means = emb.unit_means(values)
        for slide, value in zip(emb.units, means):
            rows.append({"slide_id": slide, "tissue_type": slides[slide], "scanner": scanner,
                         "patch_count": int(counts[slide]), "endpoint": method,
                         "endpoint_label": LABELS[method], "family": "correction gain", "value": float(value)})
    return pd.DataFrame(rows)


def run_task(index: int, out: Path, source: str, bootstrap: int, workers: int) -> None:
    task = task_list(source)[index]
    pfm, method = task["pfm"], task["method"]
    stem = f"{pfm}__{method}"
    status_path = out / "tasks" / f"{stem}.json"
    started = time.time()
    try:
        emb = load("pannormal", pfm, methods=list(NEEDS[method]), source=source)
        frame = slide_gains(emb, method)
    except (FileNotFoundError, KeyError) as error:
        write_json({"status": "missing_input", "task": task, "error": str(error)}, status_path)
        return
    input_path = out / "inputs" / f"{stem}.csv"
    write_frame(frame, input_path)
    lmm_root = out / "lmm" / stem
    arguments = argparse.Namespace(input=str(input_path), output_root=str(lmm_root), task_index=0,
                                   bootstrap=bootstrap, workers=workers, seed=SEED + index)
    try:
        slide_level_lmm.run(arguments)
        status = "pass"
        error = None
    except (RuntimeError, ValueError) as caught:
        status, error = "model_failed", str(caught)
    write_json({"status": status, "error": error, "task": task, "source": emb.source,
                "bootstrap": bootstrap, "seed": SEED + index, "n_rows": len(frame),
                "input_sha256": sha256(input_path), "seconds": time.time() - started,
                "notes": emb.notes}, status_path)


def spearman_increment(gains: pd.DataFrame, pfm: str, weights: np.ndarray, units: tuple) -> list[dict]:
    direct = pd.read_csv(DIRECT, dtype={"slide_id": str})
    direct = direct.loc[direct.endpoint.eq("frequency_high")]
    unit_index = pd.Index(units)
    tissues_of = gains.drop_duplicates("slide_id").set_index("slide_id")["tissue_type"].reindex(unit_index)
    tissue_names = sorted(tissues_of.unique())
    indicator = np.zeros((len(units), len(tissue_names)))
    indicator[np.arange(len(units)), tissues_of.map({t: i for i, t in enumerate(tissue_names)}).to_numpy()] = 1.0
    rows = []
    for scanner in SPEARMAN_SCANNERS:
        increment = gains.loc[gains.scanner.eq(scanner)].set_index("slide_id")["value"].reindex(unit_index).to_numpy(float)
        ratio = direct.loc[direct.scanner.eq(scanner)].set_index("slide_id")["value"].reindex(unit_index).to_numpy(float)
        if np.isnan(increment).any() or np.isnan(ratio).any():
            continue
        count = weights @ indicator
        with np.errstate(invalid="ignore", divide="ignore"):
            x = (weights @ (ratio[:, None] * indicator)) / count
            y = (weights @ (increment[:, None] * indicator)) / count
        rho = spearman_rows(x, y)
        point = stats.spearmanr(x[0], y[0])
        rows.append({"pfm": pfm, "scanner": scanner, "n_tissues": len(tissue_names),
                     "n_slides": len(units), **summarize(rho), "p_value": float(point.pvalue),
                     "expected_sign_akoya": "negative (larger increment where the high-frequency deficit is stronger)"
                     if scanner == "akoya" else "not pre-specified"})
    return rows


def aggregate(out: Path, source: str) -> None:
    tasks = task_list(source)
    rows, gain_rows, spearman, statuses = [], [], [], {}
    gains_by_pfm = {}
    for index, task in enumerate(tasks):
        stem = f"{task['pfm']}__{task['method']}"
        status_path = out / "tasks" / f"{stem}.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "not_run"}
        statuses[stem] = {k: status.get(k) for k in ("status", "error", "bootstrap", "seconds")}
        input_path = out / "inputs" / f"{stem}.csv"
        if input_path.exists():
            gains = pd.read_csv(input_path, dtype={"slide_id": str})
            gains_by_pfm.setdefault(task["pfm"], {})[task["method"]] = gains
        result_path = out / "lmm" / stem / "endpoints" / f"00_{task['method']}" / "result.json"
        row = {"pfm": task["pfm"], "method": task["method"], "status": status.get("status")}
        if status.get("status") == "pass" and result_path.exists():
            result = json.loads(result_path.read_text())
            row.update({"n_slides": result["n_slides"], "n_tissues": result["n_tissues"],
                        "n_observations": result["n_observations"], "bootstrap_requested": result["bootstrap_requested"]})
            for test, values in result["tests"].items():
                for key, value in values.items():
                    row[f"{test}_{key}"] = value
            for component, fraction in result["variance_fractions"].items():
                row[f"fraction_{component}"] = fraction
            for scanner, mean in result["fixed_scanner_means"].items():
                row[f"fixed_{scanner}"] = mean
        rows.append(row)
    summary = pd.DataFrame(rows)
    for test in TESTS:
        column = f"{test}_bootstrap_p"
        summary[f"{test}_bh_q"] = np.nan
        if column in summary:
            for pfm, group in summary.groupby("pfm"):
                summary.loc[group.index, f"{test}_bh_q"] = benjamini_hochberg(group[column].to_numpy(float))
    write_frame(summary, out / "lmm_summary.csv")

    for pfm, by_method in gains_by_pfm.items():
        units = tuple(sorted(next(iter(by_method.values())).slide_id.unique()))
        weights = bootstrap_weights("pannormal", len(units))
        unit_index = pd.Index(units)
        for method, gains in by_method.items():
            per_scanner = {}
            for scanner in TARGETS["pannormal"]:
                values = gains.loc[gains.scanner.eq(scanner)].set_index("slide_id")["value"].reindex(unit_index).to_numpy(float)
                per_scanner[scanner] = weighted_mean(weights, values)
                gain_rows.append({"pfm": pfm, "method": method, "scanner": scanner, **summarize(per_scanner[scanner])})
            gain_rows.append({"pfm": pfm, "method": method, "scanner": "pooled",
                              **summarize(np.mean(list(per_scanner.values()), axis=0))})
        if "frequency_increment" in by_method:
            spearman.extend(spearman_increment(by_method["frequency_increment"], pfm, weights, units))
    gains_table = pd.DataFrame(gain_rows)
    spearman = pd.DataFrame(spearman)
    write_frame(gains_table, out / "mean_gains.csv")
    write_frame(spearman, out / "spearman_frequency_increment.csv")
    manifest = manifest_status()
    write_json({"source": source, "embedding_manifest": manifest, "tasks": statuses,
                "model_code": "analysis/paper/slide_level_lmm.py (run), sha256 " + sha256(PROJECT / "analysis/paper/slide_level_lmm.py"),
                "direct_slide_contrasts_sha256": sha256(DIRECT), "code_sha256": sha256(Path(__file__))},
               out / "qc.json")
    write_text(render_summary(summary, gains_table, spearman, statuses, source, manifest), out / "summary.md")
    print((out / "summary.md").read_text())


def render_summary(summary, gains, spearman, statuses, source, manifest) -> str:
    lines = ["# RV09 tissue dependence of correction gains", ""]
    if source == "smoke":
        lines += ["**SMOKE TEST** on the smoke adapter (UNI v1 from `03_uni`; Reinhard, color + frequency and the "
                  "frequency increment only; reduced bootstrap). Not a revision result.", ""]
    lines += ["## What ran", "",
              f"- Embeddings `{rel(EMBEDDINGS)}` (manifest {manifest}), source `{source}`.",
              "- Response: slide × scanner target gain = mean over the 20 set20 locations of (raw − corrected) cosine "
              "target distance; frequency increment = Reinhard distance − color + frequency distance.",
              "- Model: `analysis/paper/slide_level_lmm.py` `run` unchanged (fixed scanner means; random shared tissue, "
              "scanner × tissue and slide effects; REML; RLRT with parametric bootstrap); BH across the five methods "
              "within each PFM, separately for each test. Primary test: any tissue dependence (a and u); secondary: "
              "scanner-specific tissue dependence (u).",
              "- Spearman across tissue types (AKOYA, GT450): tissue-mean high-frequency log2 transfer ratio "
              "(`frequency_high`, RQ1 input, all matched locations) vs tissue-mean frequency increment; "
              f"{BOOTSTRAPS} slide bootstrap resamples (shared with RV04/RV05).", "", "## QC", ""]
    for key, value in statuses.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    for pfm in [p for p in PFMS if p in set(summary.pfm)]:
        block = summary.loc[summary.pfm.eq(pfm)]
        role = "primary" if pfm == "uni_v1" else "secondary"
        lines += [f"## {PFM_LABELS[pfm]} ({role})", "",
                  "| Method | Any tissue dependence: LRT, p, BH q | Scanner-specific: LRT, p, BH q | "
                  "Variance fractions (tissue / scanner×tissue / slide / residual) | Mean gain [95% CI] |",
                  "| --- | --- | --- | --- | --- |"]
        for row in block.itertuples():
            gain = gains.loc[gains.pfm.eq(pfm) & gains.method.eq(row.method) & gains.scanner.eq("pooled")] if len(gains) else gains
            gain_text = (f"{gain.estimate.iloc[0]:.4f} [{gain.ci_low.iloc[0]:.4f}, {gain.ci_high.iloc[0]:.4f}]"
                         if len(gain) else "n/a")
            if row.status != "pass" or not hasattr(row, "any_tissue_dependence_lrt"):
                lines.append(f"| {LABELS[row.method]} | {row.status} | | | {gain_text} |")
                continue
            lines.append(
                f"| {LABELS[row.method]} | {row.any_tissue_dependence_lrt:.2f}, {row.any_tissue_dependence_bootstrap_p:.4f}, "
                f"{row.any_tissue_dependence_bh_q:.4f} | {row.scanner_specific_tissue_dependence_lrt:.2f}, "
                f"{row.scanner_specific_tissue_dependence_bootstrap_p:.4f}, {row.scanner_specific_tissue_dependence_bh_q:.4f} | "
                f"{row.fraction_tissue_shared:.2f} / {row.fraction_scanner_by_tissue:.2f} / {row.fraction_slide_shared:.2f} / "
                f"{row.fraction_residual:.2f} | {gain_text} |")
        lines.append("")
        fitted = block.loc[block.status.eq("pass")]
        for test, label in (("any_tissue_dependence", "any tissue dependence"),
                            ("scanner_specific_tissue_dependence", "scanner-specific tissue dependence")):
            column = f"{test}_bh_q"
            if column in fitted and len(fitted):
                significant = fitted.loc[fitted[column] < 0.05, "method"].tolist()
                lines.append(f"- BH q < 0.05 for {label}: {len(significant)}/{len(fitted)} fitted methods "
                             f"({', '.join(significant) if significant else 'none'}).")
        sp = spearman.loc[spearman.pfm.eq(pfm)] if len(spearman) else spearman
        for row in sp.itertuples():
            lines.append(f"- Spearman ({row.scanner.upper()}), tissue-mean high-frequency log2 ratio vs frequency "
                         f"increment: rho {row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}], p = {row.p_value:.3g}, "
                         f"{row.n_tissues} tissues")
        lines.append("")
    lines += ["## Files", "", "- `lmm_summary.csv` (tests, BH q, variance fractions, fixed scanner means), `mean_gains.csv`, "
              "`spearman_frequency_increment.csv`, `qc.json`; model inputs in `inputs/`, per-fit outputs and bootstrap "
              "draws in `lmm/`.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("task", "aggregate", "list-tasks"))
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--source", choices=("rv03", "smoke"), default="rv03")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1)
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
        print(json.dumps(tasks[args.task_index]), flush=True)
        if args.source == "rv03" and not manifest_status()["complete"] and not args.allow_incomplete:
            raise SystemExit("RV03 corrected_embeddings/manifest.json is missing or not complete")
        run_task(args.task_index, out, args.source, args.bootstrap, args.workers)
        return
    aggregate(out, args.source)


if __name__ == "__main__":
    main()
