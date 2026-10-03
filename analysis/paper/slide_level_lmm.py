#!/usr/bin/env python3
"""Direct whole-slide paired-contrast LMM for scanner and tissue effects.

Every paired location contributes to one whole-slide contrast for each target
scanner and endpoint. The resulting 515 target-minus-reference contrasts are
modeled as

    d[s,r] = beta[s] + a[t(r)] + u[s,t(r)] + c[r] + error[s,r]

The shared tissue term a and scanner-by-tissue term u are tested both jointly
(any tissue dependence) and separately (scanner-specific tissue dependence).
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from joint_interaction_lmm import (
    LOWER_LOG_VARIANCE,
    SCANNERS,
    UPPER_LOG_VARIANCE,
    reml_evaluate,
    sha256,
    simulate_response,
    write_frame,
    write_json,
)


COMPONENTS = ("tissue_shared", "scanner_by_tissue", "slide_shared", "residual")
FULL = (0, 1, 2, 3)
NO_INTERACTION = (0, 2, 3)
NO_TISSUE = (2, 3)
VERSION = "slide_level_scanner_tissue_lmm_v1"


@dataclass
class Block:
    tissue_type: str
    slide_ids: tuple[str, ...]
    y: np.ndarray
    x: np.ndarray
    kernels: tuple[np.ndarray, ...]
    designs: tuple[np.ndarray, ...]


def validate_slide_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "endpoint", "endpoint_label", "family", "slide_id", "tissue_type",
        "scanner", "patch_count", "value",
    }
    if required - set(frame):
        raise ValueError(f"missing input columns: {sorted(required - set(frame))}")
    if frame.duplicated(["endpoint", "slide_id", "scanner"]).any():
        raise ValueError("duplicate scanner--slide contrast")
    if not frame["patch_count"].gt(0).all():
        raise ValueError("nonpositive patch count")
    if not np.isfinite(frame["value"]).all():
        raise ValueError("nonfinite contrast value")
    if frame.groupby("slide_id", observed=True)["tissue_type"].nunique().ne(1).any():
        raise ValueError("slide has more than one tissue type")
    if frame.groupby("slide_id", observed=True)["patch_count"].nunique().ne(1).any():
        raise ValueError("inconsistent paired location count")
    if not frame["slide_id"].nunique() == 103 or frame["tissue_type"].nunique() != 37:
        raise ValueError("unexpected PanNormal cohort")
    if frame.groupby(["endpoint", "slide_id"], observed=True)["scanner"].nunique().ne(5).any():
        raise ValueError("incomplete scanner comparisons")
    if frame.groupby("endpoint", observed=True).size().ne(103 * len(SCANNERS)).any():
        raise ValueError("unexpected endpoint observation count")
    return frame


def build_blocks(frame: pd.DataFrame) -> tuple[list[Block], float]:
    scale = float(stats.median_abs_deviation(frame["value"], scale="normal"))
    if not np.isfinite(scale) or scale <= 1e-10:
        scale = float(frame["value"].std(ddof=1))
    if not np.isfinite(scale) or scale <= 1e-10:
        raise ValueError("no usable response scale")
    blocks = []
    for tissue, part in frame.groupby("tissue_type", sort=True):
        part = part.copy()
        part["scanner"] = pd.Categorical(part["scanner"], categories=SCANNERS, ordered=True)
        part = part.sort_values(["slide_id", "scanner"])
        scanners = part["scanner"].astype(str).to_numpy()
        slides = part["slide_id"].astype(str).to_numpy()
        slide_ids = tuple(dict.fromkeys(slides))
        x = np.column_stack([(scanners == scanner).astype(float) for scanner in SCANNERS])
        one = np.ones((len(part), 1))
        z_slide = np.column_stack([(slides == slide).astype(float) for slide in slide_ids])
        identity = np.eye(len(part))
        designs = (one, x, z_slide, identity)
        kernels = tuple(z @ z.T for z in designs[:-1]) + (identity,)
        blocks.append(Block(str(tissue), slide_ids, part["value"].to_numpy() / scale,
                            x, kernels, designs))
    return blocks, scale


def fit(blocks: list[Block], active: tuple[int, ...], y_values=None,
        start_variances=None, thorough=False) -> dict:
    if start_variances is None:
        start_variances = np.array([0.10, 0.15, 0.15, 0.60])
    start_variances = np.asarray(start_variances, dtype=float)
    starts = [np.log(np.clip(start_variances[list(active)], 1e-7, np.exp(UPPER_LOG_VARIANCE)))]
    if thorough:
        starts += [np.log(np.clip(start_variances[list(active)] * multiplier,
                                 1e-7, np.exp(UPPER_LOG_VARIANCE)))
                   for multiplier in (0.25, 4.0)]

    def unpack(log_values):
        variances = np.zeros(len(COMPONENTS))
        variances[list(active)] = np.exp(log_values)
        return variances

    def objective(log_values):
        return reml_evaluate(blocks, unpack(log_values), y_values)["negative_log_likelihood"]

    candidates = [optimize.minimize(
        objective, start, method="L-BFGS-B",
        bounds=[(LOWER_LOG_VARIANCE, UPPER_LOG_VARIANCE)] * len(active),
        options={"maxiter": 250, "ftol": 1e-10, "gtol": 1e-7, "maxls": 40},
    ) for start in starts]
    finite = [candidate for candidate in candidates if np.isfinite(candidate.fun)]
    if not finite:
        raise RuntimeError("all REML fits failed")
    best = min(finite, key=lambda candidate: candidate.fun)
    variance = unpack(best.x)
    details = reml_evaluate(blocks, variance, y_values)
    return {"variances": variance, "nll": float(best.fun),
            "converged": bool(best.success), "message": str(best.message),
            "beta": details["beta"], "fixed_covariance": details["fixed_covariance"]}


def lrt(full: dict, reduced: dict) -> float:
    return max(0.0, 2.0 * (reduced["nll"] - full["nll"]))


_BOOT_BLOCKS = None
_BOOT_NULL = None
_BOOT_ACTIVE = None


def init_bootstrap(blocks, null, active):
    global _BOOT_BLOCKS, _BOOT_NULL, _BOOT_ACTIVE
    _BOOT_BLOCKS, _BOOT_NULL, _BOOT_ACTIVE = blocks, null, active


def bootstrap_once(seed):
    y = simulate_response(_BOOT_BLOCKS, _BOOT_NULL["beta"], _BOOT_NULL["variances"], seed)
    null = fit(_BOOT_BLOCKS, _BOOT_ACTIVE, y_values=y,
               start_variances=_BOOT_NULL["variances"])
    start = null["variances"].copy()
    for component in set(FULL) - set(_BOOT_ACTIVE):
        start[component] = 0.02
    full = fit(_BOOT_BLOCKS, FULL, y_values=y, start_variances=start)
    return {"seed": seed, "lrt": lrt(full, null),
            "full_converged": full["converged"], "null_converged": null["converged"]}


def bootstrap_test(blocks, null, active, observed_lrt, seeds, workers):
    if workers == 1:
        init_bootstrap(blocks, null, active)
        rows = [bootstrap_once(int(seed)) for seed in seeds]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_bootstrap,
                                 initargs=(blocks, null, active)) as pool:
            rows = list(pool.map(bootstrap_once, [int(seed) for seed in seeds], chunksize=1))
    frame = pd.DataFrame(rows)
    valid = frame[frame["full_converged"] & frame["null_converged"]]
    if len(valid) < max(3, math.ceil(0.9 * len(seeds))):
        raise RuntimeError(f"only {len(valid)}/{len(seeds)} bootstrap fits converged")
    p = (1 + int((valid["lrt"] >= observed_lrt).sum())) / (1 + len(valid))
    return p, frame


def run(args):
    input_path = Path(args.input)
    all_means = validate_slide_contrasts(pd.read_csv(input_path, dtype={"slide_id": str}))
    endpoints = list(all_means["endpoint"].drop_duplicates())
    if not 0 <= args.task_index < len(endpoints):
        raise ValueError("invalid endpoint index")
    endpoint = endpoints[args.task_index]
    selected = all_means[all_means["endpoint"].eq(endpoint)]
    blocks, scale = build_blocks(selected)
    full = fit(blocks, FULL, thorough=True)
    no_interaction = fit(blocks, NO_INTERACTION, thorough=True)
    no_tissue = fit(blocks, NO_TISSUE, thorough=True)
    models = {"full": full, "no_interaction": no_interaction, "no_tissue": no_tissue}
    if not all(model["converged"] for model in models.values()):
        raise RuntimeError("observed model did not converge")
    results = {}
    for name, null, active in (
        ("any_tissue_dependence", no_tissue, NO_TISSUE),
        ("scanner_specific_tissue_dependence", no_interaction, NO_INTERACTION),
    ):
        statistic = lrt(full, null)
        rng = np.random.SeedSequence([args.seed, args.task_index, 0 if active == NO_TISSUE else 1])
        seeds = [int(seq.generate_state(1, dtype=np.uint32)[0]) for seq in rng.spawn(args.bootstrap)]
        p, draws = bootstrap_test(blocks, null, active, statistic, seeds, args.workers)
        results[name] = {"lrt": statistic, "bootstrap_p": p,
                         "bootstrap_successful": int((draws["full_converged"] &
                                                       draws["null_converged"]).sum())}
        outdir = Path(args.output_root) / "endpoints" / f"{args.task_index:02d}_{endpoint}"
        write_frame(draws, outdir / f"{name}_bootstrap.csv.gz")
    raw_variances = full["variances"] * scale**2
    result = {
        "version": VERSION, "endpoint": endpoint,
        "endpoint_label": str(selected["endpoint_label"].iloc[0]),
        "family": str(selected["family"].iloc[0]),
        "n_slides": int(selected["slide_id"].nunique()),
        "n_tissues": int(selected["tissue_type"].nunique()),
        "n_observations": int(len(selected)),
        "n_patches_per_slide_min": int(selected["patch_count"].min()),
        "n_patches_per_slide_max": int(selected["patch_count"].max()),
        "input_sha256": sha256(input_path),
        "response_scale": scale, "bootstrap_requested": args.bootstrap,
        "tests": results,
        "fixed_scanner_means": dict(zip(SCANNERS, map(float, full["beta"] * scale))),
        "models": {key: {"nll": value["nll"], "converged": value["converged"],
                         "variances": dict(zip(COMPONENTS, map(float, value["variances"] * scale**2)))}
                   for key, value in models.items()},
        "variance_fractions": dict(zip(COMPONENTS, map(float, raw_variances / raw_variances.sum()))),
    }
    outdir = Path(args.output_root) / "endpoints" / f"{args.task_index:02d}_{endpoint}"
    write_json(outdir / "result.json", result)
    print(json.dumps({"endpoint": endpoint, "tests": results,
                      "variance_fractions": result["variance_fractions"]}, indent=2))


def aggregate(args):
    root = Path(args.output_root)
    rows = []
    for path in sorted((root / "endpoints").glob("*/result.json")):
        result = json.loads(path.read_text())
        row = {key: result[key] for key in ("endpoint", "endpoint_label", "family", "n_slides",
                                             "n_tissues", "n_observations", "bootstrap_requested")}
        for test, data in result["tests"].items():
            for key, value in data.items():
                row[f"{test}_{key}"] = value
        for component, fraction in result["variance_fractions"].items():
            row[f"fraction_{component}"] = fraction
        for scanner, mean in result["fixed_scanner_means"].items():
            row[f"fixed_{scanner}"] = mean
        rows.append(row)
    if len(rows) != 13:
        raise RuntimeError(f"expected 13 endpoint results; found {len(rows)}")
    summary = pd.DataFrame(rows)
    for prefix in ("any_tissue_dependence", "scanner_specific_tissue_dependence"):
        p = summary[f"{prefix}_bootstrap_p"].to_numpy()
        order = np.argsort(p)
        adjusted = np.empty_like(p)
        adjusted[order] = np.minimum.accumulate((p[order] * len(p) /
                                                 np.arange(1, len(p) + 1))[::-1])[::-1]
        summary[f"{prefix}_bh_q"] = np.minimum(adjusted, 1.0)
    write_frame(summary, root / "endpoint_summary.csv")
    print(summary[["endpoint", "any_tissue_dependence_bootstrap_p",
                   "any_tissue_dependence_bh_q",
                   "scanner_specific_tissue_dependence_bootstrap_p",
                   "scanner_specific_tissue_dependence_bh_q"]].to_string(index=False))


def plot_figure(args):
    import matplotlib.pyplot as plt

    summary = pd.read_csv(args.summary)
    if len(summary) != 13:
        raise ValueError("expected 13 endpoints")
    y = np.arange(len(summary))
    fig, (left_ax, right_ax) = plt.subplots(
        1, 2, figsize=(12.5, 6.8), gridspec_kw={"width_ratios": [3.2, 1.2]}
    )
    components = (
        ("fraction_tissue_shared", "Shared tissue", "#9ecae1"),
        ("fraction_scanner_by_tissue", "Scanner × tissue", "#2171b5"),
        ("fraction_slide_shared", "Shared slide", "#fdd0a2"),
        ("fraction_residual", "Scanner-specific slide / sampling residual", "#969696"),
    )
    offsets = np.zeros(len(summary))
    for column, label, color in components:
        values = summary[column].to_numpy()
        left_ax.barh(y, values, left=offsets, color=color, edgecolor="white",
                     linewidth=0.4, label=label)
        offsets += values
    labels = summary["endpoint_label"].replace({
        "1 − gradient NCC to AT2": "1 − gradient NCC to reference"
    })
    left_ax.set_yticks(y, labels)
    left_ax.invert_yaxis()
    left_ax.set_xlim(0, 1)
    left_ax.set_xlabel("Fraction of modeled slide-level variance")
    left_ax.set_title("A  Variance decomposition", loc="left", fontweight="bold")
    left_ax.grid(axis="x", color="#dddddd", linewidth=0.6)

    scores = summary["scanner_specific_tissue_dependence_lrt"].to_numpy()
    significant = summary["scanner_specific_tissue_dependence_bh_q"].to_numpy() < 0.05
    right_ax.barh(y, scores, color=np.where(significant, "#2171b5", "#bdbdbd"))
    right_ax.invert_yaxis()
    right_ax.set_yticks([])
    right_ax.set_xlabel("Restricted likelihood-ratio statistic")
    right_ax.set_title("B  Scanner × tissue test", loc="left", fontweight="bold")
    right_ax.grid(axis="x", color="#dddddd", linewidth=0.6)

    fig.legend(*left_ax.get_legend_handles_labels(), loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.20, top=0.94, wspace=0.04)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-endpoint")
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--task-index", type=int, required=True)
    run_parser.add_argument("--bootstrap", type=int, default=500)
    run_parser.add_argument("--workers", type=int, default=1)
    run_parser.add_argument("--seed", type=int, default=20260923)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--output-root", required=True)
    plot_parser = subparsers.add_parser("plot-figure")
    plot_parser.add_argument("--summary", required=True)
    plot_parser.add_argument("--output", required=True)
    parsed = parser.parse_args()
    {"run-endpoint": run, "aggregate": aggregate, "plot-figure": plot_figure}[parsed.command](parsed)
