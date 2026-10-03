#!/usr/bin/env python3
"""Joint scanner-by-tissue REML analysis for the final PanNormal cohort.

The input contains five target-minus-AT2 paired contrasts for every endpoint,
slide, and spatial replicate.  Each endpoint is fitted jointly across targets:

    d[strj] = beta[s] + a[t] + u[s,t] + c[r(t)] + v[s,r(t)]
                + q[j(r)] + error[strj]

The shared tissue/slide/replicate terms account for correlation induced by the
common AT2 reference.  The target-specific tissue random slope u[s,t] is the
explicit scanner-by-tissue interaction tested by a restricted likelihood-ratio
test.  A parametric bootstrap supplies the primary p-value because the tested
variance lies on the boundary of the parameter space.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats


ANALYSIS_VERSION = "joint_scanner_tissue_reml_v1"
SCANNERS = ("versa", "akoya", "gt450", "s360", "s60")
COMPONENTS = (
    "tissue_shared",
    "scanner_by_tissue",
    "slide_shared",
    "scanner_by_slide",
    "matched_replicate",
    "residual",
)
FULL_ACTIVE = tuple(range(len(COMPONENTS)))
NO_INTERACTION_ACTIVE = (0, 2, 3, 4, 5)
LOWER_LOG_VARIANCE = -18.0
UPPER_LOG_VARIANCE = 6.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".gz":
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
    else:
        frame.to_csv(temporary, index=False)
    temporary.replace(path)


def benjamini_hochberg(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.full(array.shape, np.nan, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return result
    selected = array[finite]
    order = np.argsort(selected)
    ranked = selected[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[finite] = restored
    return result


@dataclass
class TissueBlock:
    tissue_type: str
    slide_ids: tuple[str, ...]
    y: np.ndarray
    x: np.ndarray
    kernels: tuple[np.ndarray, ...]
    designs: tuple[np.ndarray, ...]


def indicator_design(labels: np.ndarray, levels: Iterable[str]) -> np.ndarray:
    return np.column_stack([(labels == level).astype(float) for level in levels])


def build_blocks(frame: pd.DataFrame) -> tuple[list[TissueBlock], float]:
    expected = {"slide_id", "tissue_type", "replicate", "scanner", "value"}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    observed_scanners = tuple(frame["scanner"].drop_duplicates())
    if set(observed_scanners) != set(SCANNERS):
        raise ValueError(f"unexpected scanner set: {observed_scanners}")
    duplicate = frame.duplicated(["slide_id", "replicate", "scanner"])
    if duplicate.any():
        raise ValueError("duplicate slide/replicate/scanner rows")

    scale = float(stats.median_abs_deviation(frame["value"], scale="normal"))
    if not np.isfinite(scale) or scale <= 1e-10:
        scale = float(frame["value"].std(ddof=1))
    if not np.isfinite(scale) or scale <= 1e-10:
        raise ValueError("endpoint has no usable response scale")

    scanner_levels = np.asarray(SCANNERS, dtype=object)
    blocks: list[TissueBlock] = []
    for tissue, part in frame.groupby("tissue_type", sort=True):
        part = part.copy()
        part["scanner"] = pd.Categorical(part["scanner"], categories=SCANNERS, ordered=True)
        part = part.sort_values(["slide_id", "replicate", "scanner"])
        if part["scanner"].isna().any():
            raise ValueError(f"unknown scanner in {tissue}")
        counts = part.groupby(["slide_id", "replicate"], observed=True)["scanner"].nunique()
        if not counts.eq(len(SCANNERS)).all():
            raise ValueError(f"incomplete target set in {tissue}")

        scanners = part["scanner"].astype(str).to_numpy()
        slides = part["slide_id"].astype(str).to_numpy()
        replicates = part["replicate"].astype(int).to_numpy()
        slide_levels = tuple(part["slide_id"].astype(str).drop_duplicates())
        replicate_labels = np.asarray(
            [f"{slide}::{replicate}" for slide, replicate in zip(slides, replicates)],
            dtype=object,
        )
        replicate_levels = tuple(dict.fromkeys(replicate_labels.tolist()))

        x = indicator_design(scanners, scanner_levels)
        one = np.ones((len(part), 1), dtype=float)
        z_scanner = x.copy()
        z_slide = indicator_design(slides, slide_levels)
        z_slide_scanner = np.column_stack(
            [
                ((slides == slide) & (scanners == scanner)).astype(float)
                for slide in slide_levels
                for scanner in SCANNERS
            ]
        )
        z_replicate = indicator_design(replicate_labels, replicate_levels)
        identity = np.eye(len(part), dtype=float)
        designs = (one, z_scanner, z_slide, z_slide_scanner, z_replicate, identity)
        kernels = tuple(design @ design.T for design in designs[:-1]) + (identity,)
        blocks.append(
            TissueBlock(
                tissue_type=str(tissue),
                slide_ids=slide_levels,
                y=part["value"].to_numpy(dtype=float) / scale,
                x=x,
                kernels=kernels,
                designs=designs,
            )
        )
    return blocks, scale


def reml_evaluate(
    blocks: Sequence[TissueBlock], variances: np.ndarray, y_values: Sequence[np.ndarray] | None = None
) -> dict:
    p = len(SCANNERS)
    xt_v_x = np.zeros((p, p), dtype=float)
    xt_v_y = np.zeros(p, dtype=float)
    y_v_y = 0.0
    logdet_v = 0.0
    total_n = 0
    for index, block in enumerate(blocks):
        y = block.y if y_values is None else y_values[index]
        covariance = np.zeros_like(block.kernels[0])
        for variance, kernel in zip(variances, block.kernels):
            if variance > 0:
                covariance += variance * kernel
        try:
            factor = linalg.cho_factor(covariance, lower=True, check_finite=False)
        except linalg.LinAlgError:
            return {"negative_log_likelihood": np.inf}
        inverse_x = linalg.cho_solve(factor, block.x, check_finite=False)
        inverse_y = linalg.cho_solve(factor, y, check_finite=False)
        xt_v_x += block.x.T @ inverse_x
        xt_v_y += block.x.T @ inverse_y
        y_v_y += float(y @ inverse_y)
        logdet_v += float(2.0 * np.log(np.diag(factor[0])).sum())
        total_n += len(y)
    try:
        fixed_factor = linalg.cho_factor(xt_v_x, lower=True, check_finite=False)
        beta = linalg.cho_solve(fixed_factor, xt_v_y, check_finite=False)
        fixed_covariance = linalg.cho_solve(fixed_factor, np.eye(p), check_finite=False)
    except linalg.LinAlgError:
        return {"negative_log_likelihood": np.inf}
    logdet_fixed = float(2.0 * np.log(np.diag(fixed_factor[0])).sum())
    quadratic = max(float(y_v_y - beta @ xt_v_y), 0.0)
    n_reml = total_n - p
    negative_log_likelihood = 0.5 * (
        logdet_v + logdet_fixed + quadratic + n_reml * math.log(2.0 * math.pi)
    )
    return {
        "negative_log_likelihood": negative_log_likelihood,
        "beta": beta,
        "fixed_covariance": fixed_covariance,
        "quadratic": quadratic,
        "total_n": total_n,
    }


def fit_reml(
    blocks: Sequence[TissueBlock],
    active: Sequence[int],
    y_values: Sequence[np.ndarray] | None = None,
    starts: Sequence[np.ndarray] | None = None,
    maxiter: int = 400,
) -> dict:
    active = tuple(active)
    if starts is None:
        base = np.asarray([0.05, 0.10, 0.10, 0.25, 0.10, 0.40], dtype=float)
        starts = [
            np.log(base[list(active)]),
            np.log(np.maximum(base[list(active)] * 0.25, np.exp(LOWER_LOG_VARIANCE))),
            np.log(np.minimum(base[list(active)] * 4.0, np.exp(UPPER_LOG_VARIANCE))),
        ]

    def unpack(log_values: np.ndarray) -> np.ndarray:
        variances = np.zeros(len(COMPONENTS), dtype=float)
        variances[list(active)] = np.exp(log_values)
        return variances

    def objective(log_values: np.ndarray) -> float:
        return reml_evaluate(blocks, unpack(log_values), y_values)["negative_log_likelihood"]

    candidates = []
    for start in starts:
        fit = optimize.minimize(
            objective,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=[(LOWER_LOG_VARIANCE, UPPER_LOG_VARIANCE)] * len(active),
            options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7, "maxls": 40},
        )
        candidates.append(fit)
    finite = [candidate for candidate in candidates if np.isfinite(candidate.fun)]
    if not finite:
        raise RuntimeError("all REML optimizations failed")
    best = min(finite, key=lambda candidate: candidate.fun)
    variances = unpack(best.x)
    details = reml_evaluate(blocks, variances, y_values)
    return {
        "active": active,
        "variances": variances,
        "negative_log_likelihood": float(best.fun),
        "converged": bool(best.success),
        "message": str(best.message),
        "iterations": int(best.nit),
        "beta": details["beta"],
        "fixed_covariance": details["fixed_covariance"],
        "total_n": int(details["total_n"]),
        "log_parameters": best.x,
    }


def starts_from_fit(fit: dict, active: Sequence[int], interaction_seed: float = 1e-3) -> list[np.ndarray]:
    values = np.asarray(fit["variances"], dtype=float).copy()
    values[1] = max(values[1], interaction_seed)
    selected = np.maximum(values[list(active)], np.exp(LOWER_LOG_VARIANCE + 1.0))
    return [np.log(selected)]


def simulate_response(
    blocks: Sequence[TissueBlock], beta: np.ndarray, variances: np.ndarray, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    simulated = []
    for block in blocks:
        values = block.x @ beta
        for variance, design in zip(variances, block.designs):
            if variance <= 0:
                continue
            values = values + design @ rng.normal(0.0, math.sqrt(variance), design.shape[1])
        simulated.append(values)
    return simulated


_BOOT_BLOCKS: list[TissueBlock] | None = None
_BOOT_REDUCED: dict | None = None


def initialize_bootstrap(blocks: list[TissueBlock], reduced: dict) -> None:
    global _BOOT_BLOCKS, _BOOT_REDUCED
    _BOOT_BLOCKS = blocks
    _BOOT_REDUCED = reduced


def one_bootstrap(seed: int) -> dict:
    if _BOOT_BLOCKS is None or _BOOT_REDUCED is None:
        raise RuntimeError("bootstrap worker is not initialized")
    y_values = simulate_response(
        _BOOT_BLOCKS, _BOOT_REDUCED["beta"], _BOOT_REDUCED["variances"], int(seed)
    )
    reduced = fit_reml(
        _BOOT_BLOCKS,
        NO_INTERACTION_ACTIVE,
        y_values=y_values,
        starts=starts_from_fit(_BOOT_REDUCED, NO_INTERACTION_ACTIVE),
        maxiter=250,
    )
    full_start = np.asarray(reduced["variances"], dtype=float)
    full_start[1] = max(float(np.mean(full_start[[0, 2, 3]])) * 0.05, 1e-4)
    full = fit_reml(
        _BOOT_BLOCKS,
        FULL_ACTIVE,
        y_values=y_values,
        starts=[np.log(np.maximum(full_start, np.exp(LOWER_LOG_VARIANCE + 1.0)))],
        maxiter=250,
    )
    likelihood_ratio = max(
        0.0, 2.0 * (reduced["negative_log_likelihood"] - full["negative_log_likelihood"])
    )
    return {
        "seed": int(seed),
        "likelihood_ratio": likelihood_ratio,
        "full_converged": full["converged"],
        "reduced_converged": reduced["converged"],
        "interaction_variance": float(full["variances"][1]),
    }


def endpoint_order(frame: pd.DataFrame) -> list[str]:
    return frame["endpoint"].drop_duplicates().astype(str).tolist()


def run_endpoint(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_root = Path(args.output_root)
    frame = pd.read_csv(input_path)
    endpoints = endpoint_order(frame)
    if args.task_index < 0 or args.task_index >= len(endpoints):
        raise ValueError(f"task index {args.task_index} outside 0..{len(endpoints) - 1}")
    endpoint = endpoints[args.task_index]
    selected = frame[frame["endpoint"].eq(endpoint)].copy()
    blocks, scale = build_blocks(selected)
    full = fit_reml(blocks, FULL_ACTIVE)
    reduced = fit_reml(blocks, NO_INTERACTION_ACTIVE)
    likelihood_ratio = max(
        0.0, 2.0 * (reduced["negative_log_likelihood"] - full["negative_log_likelihood"])
    )
    mixture_p = 0.5 * float(stats.chi2.sf(likelihood_ratio, 1)) if likelihood_ratio > 0 else 1.0

    sequence = np.random.SeedSequence([args.seed, args.task_index])
    seeds = [int(item.generate_state(1, dtype=np.uint32)[0]) for item in sequence.spawn(args.bootstrap)]
    workers = max(1, int(args.workers))
    if workers == 1:
        initialize_bootstrap(blocks, reduced)
        bootstrap_rows = [one_bootstrap(seed) for seed in seeds]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=initialize_bootstrap, initargs=(blocks, reduced)
        ) as executor:
            bootstrap_rows = list(executor.map(one_bootstrap, seeds, chunksize=1))
    bootstrap = pd.DataFrame(bootstrap_rows)
    successful = bootstrap[bootstrap["full_converged"] & bootstrap["reduced_converged"]].copy()
    if len(successful) < max(50, int(0.9 * args.bootstrap)):
        raise RuntimeError(
            f"only {len(successful)}/{args.bootstrap} converged bootstrap replicates for {endpoint}"
        )
    bootstrap_p = (1.0 + float((successful["likelihood_ratio"] >= likelihood_ratio).sum())) / (
        1.0 + len(successful)
    )

    raw_variances = np.asarray(full["variances"]) * scale**2
    total_variance = float(raw_variances.sum())
    beta = np.asarray(full["beta"]) * scale
    beta_se = np.sqrt(np.diag(full["fixed_covariance"])) * scale
    result = {
        "analysis_version": ANALYSIS_VERSION,
        "completed_utc": utc_now(),
        "endpoint": endpoint,
        "endpoint_label": str(selected["endpoint_label"].iloc[0]),
        "family": str(selected["family"].iloc[0]),
        "task_index": int(args.task_index),
        "response_scale": scale,
        "n_tissues": len(blocks),
        "n_slides": int(selected["slide_id"].nunique()),
        "n_replicates": int(selected[["slide_id", "replicate"]].drop_duplicates().shape[0]),
        "n_observations": int(len(selected)),
        "full_converged": full["converged"],
        "full_message": full["message"],
        "reduced_converged": reduced["converged"],
        "reduced_message": reduced["message"],
        "full_reml_nll": full["negative_log_likelihood"],
        "reduced_reml_nll": reduced["negative_log_likelihood"],
        "interaction_lrt": likelihood_ratio,
        "interaction_p_mixture": mixture_p,
        "interaction_p_bootstrap": bootstrap_p,
        "bootstrap_requested": int(args.bootstrap),
        "bootstrap_successful": int(len(successful)),
        "bootstrap_convergence_fraction": float(len(successful) / args.bootstrap),
        "boundary_fraction": float((successful["interaction_variance"] < 1e-6).mean()),
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256(input_path),
    }
    for component, variance in zip(COMPONENTS, raw_variances):
        result[f"variance_{component}"] = float(variance)
        result[f"fraction_{component}"] = float(variance / total_variance)
    for scanner, estimate, standard_error in zip(SCANNERS, beta, beta_se):
        result[f"fixed_{scanner}"] = float(estimate)
        result[f"fixed_{scanner}_se"] = float(standard_error)

    endpoint_dir = output_root / "endpoints" / f"{args.task_index:02d}_{endpoint}"
    write_json(endpoint_dir / "result.json", result)
    write_frame(bootstrap, endpoint_dir / "bootstrap_lrt.csv.gz")
    print(json.dumps(result, indent=2, sort_keys=True))


def figure_variance(summary: pd.DataFrame, output: Path) -> None:
    labels = summary["endpoint_label"].tolist()
    components = [
        ("fraction_tissue_shared", "Tissue-shared contrast", "#9ecae1"),
        ("fraction_scanner_by_tissue", "Scanner × tissue", "#2171b5"),
        ("fraction_slide_shared", "Slide-shared contrast", "#fdd0a2"),
        ("fraction_scanner_by_slide", "Scanner × slide", "#e6550d"),
        ("fraction_matched_replicate", "Matched replicate", "#bcbddc"),
        ("fraction_residual", "Residual", "#969696"),
    ]
    y = np.arange(len(summary))
    figure, (axis_left, axis_right) = plt.subplots(
        1, 2, figsize=(12.5, 7.0), gridspec_kw={"width_ratios": [3.3, 1.2]}
    )
    left = np.zeros(len(summary), dtype=float)
    for column, name, color in components:
        values = summary[column].to_numpy(dtype=float)
        axis_left.barh(y, values, left=left, color=color, edgecolor="white", linewidth=0.4, label=name)
        left += values
    axis_left.set_yticks(y, labels)
    axis_left.invert_yaxis()
    axis_left.set_xlim(0, 1)
    axis_left.set_xlabel("Fraction of modeled variance")
    axis_left.set_title("A  Joint variance decomposition", loc="left", fontweight="bold")
    axis_left.grid(axis="x", color="#dddddd", linewidth=0.6)
    axis_left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, frameon=False)

    scores = summary["interaction_lrt"].to_numpy(dtype=float)
    colors = np.where(summary["interaction_q_bh"].to_numpy(dtype=float) < 0.05, "#2171b5", "#bdbdbd")
    axis_right.barh(y, scores, color=colors)
    axis_right.invert_yaxis()
    axis_right.set_yticks([])
    axis_right.set_xlabel("Restricted likelihood-ratio statistic")
    axis_right.set_title("B  Scanner × tissue test", loc="left", fontweight="bold")
    axis_right.grid(axis="x", color="#dddddd", linewidth=0.6)
    figure.suptitle(
        "Explicit scanner × tissue interaction in a joint paired-contrast LMM",
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def aggregate(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    old_summary_path = Path(args.old_summary)
    output_root = Path(args.output_root)
    source_path = Path(__file__).resolve()
    source_bytes = source_path.read_bytes()
    model_source = source_bytes.split(b"\ndef figure_variance", 1)[0]
    frame = pd.read_csv(input_path)
    endpoints = endpoint_order(frame)
    rows = []
    bootstrap_files = []
    for task_index, endpoint in enumerate(endpoints):
        endpoint_dir = output_root / "endpoints" / f"{task_index:02d}_{endpoint}"
        result_path = endpoint_dir / "result.json"
        bootstrap_path = endpoint_dir / "bootstrap_lrt.csv.gz"
        if not result_path.exists() or not bootstrap_path.exists():
            raise FileNotFoundError(f"missing endpoint output: {endpoint_dir}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload["endpoint"] != endpoint:
            raise ValueError(f"endpoint mismatch in {result_path}")
        rows.append(payload)
        bootstrap_files.append(bootstrap_path)
    summary = pd.DataFrame(rows)
    summary["scanner_specific_tissue_share"] = summary["variance_scanner_by_tissue"] / (
        summary["variance_scanner_by_tissue"] + summary["variance_scanner_by_slide"]
    )
    summary["interaction_q_bh"] = benjamini_hochberg(summary["interaction_p_bootstrap"])
    summary["interaction_q_mixture_bh"] = benjamini_hochberg(summary["interaction_p_mixture"])
    summary["interaction_significant"] = summary["interaction_q_bh"] < 0.05
    write_frame(summary, output_root / "endpoint_model_summary.csv")
    write_frame(
        summary[
            [
                "endpoint",
                "endpoint_label",
                "family",
                "interaction_lrt",
                "interaction_p_bootstrap",
                "interaction_q_bh",
                "interaction_p_mixture",
                "interaction_q_mixture_bh",
                "interaction_significant",
                "bootstrap_requested",
                "bootstrap_successful",
                "bootstrap_convergence_fraction",
            ]
        ],
        output_root / "interaction_lrt.csv",
    )

    old = pd.read_csv(old_summary_path)
    old["old_interaction_significant"] = old["tissue_variance_q_bh"] < 0.05
    old_counts = (
        old.groupby(["endpoint", "endpoint_label", "family"], as_index=False)
        .agg(
            old_scanner_tests=("scanner", "size"),
            old_significant_scanners=("old_interaction_significant", "sum"),
            old_mean_tissue_fraction=("tissue_fraction_between_slide_variance", "mean"),
        )
    )
    comparison = summary.merge(old_counts, on=["endpoint", "endpoint_label", "family"], how="left")
    fixed_pairs = []
    for row in summary.itertuples(index=False):
        endpoint_old = old[old["endpoint"].eq(row.endpoint)].set_index("scanner")
        old_values = np.asarray([endpoint_old.loc[scanner, "fixed_mean"] for scanner in SCANNERS])
        joint_values = np.asarray([getattr(row, f"fixed_{scanner}") for scanner in SCANNERS])
        fixed_pairs.append(
            {
                "endpoint": row.endpoint,
                "fixed_effect_correlation": float(np.corrcoef(old_values, joint_values)[0, 1]),
                "fixed_effect_max_absolute_difference": float(np.max(np.abs(old_values - joint_values))),
                "fixed_effect_sign_concordance": float(np.mean(np.sign(old_values) == np.sign(joint_values))),
            }
        )
    comparison = comparison.merge(pd.DataFrame(fixed_pairs), on="endpoint", how="left")
    write_frame(comparison, output_root / "primary_vs_joint_comparison.csv")

    figure_path = output_root / "joint_lmm_variance_decomposition.png"
    figure_variance(summary, figure_path)
    significant = int(summary["interaction_significant"].sum())
    tissue_share_low = float(summary["scanner_specific_tissue_share"].min())
    tissue_share_high = float(summary["scanner_specific_tissue_share"].max())
    max_fixed_difference = float(comparison["fixed_effect_max_absolute_difference"].max())
    min_correlation = float(comparison["fixed_effect_correlation"].min())
    decision = (
        "# Joint scanner × tissue LMM decision\n\n"
        f"- Endpoint models converged: {int(summary['full_converged'].sum())}/{len(summary)} full and "
        f"{int(summary['reduced_converged'].sum())}/{len(summary)} reduced.\n"
        f"- Explicit scanner × tissue interaction: {significant}/{len(summary)} endpoints at "
        "parametric-bootstrap BH q < 0.05.\n"
        f"- Tissue share of target-specific tissue + slide heterogeneity: "
        f"{100 * tissue_share_low:.1f}%–{100 * tissue_share_high:.1f}%.\n"
        f"- Bootstrap convergence: {summary['bootstrap_successful'].sum()}/"
        f"{summary['bootstrap_requested'].sum()} replicates.\n"
        f"- Fixed scanner-effect agreement with the previous scanner-specific models: minimum "
        f"correlation {min_correlation:.4f}; maximum absolute difference {max_fixed_difference:.4f}.\n\n"
        "Use the joint model as the primary interaction analysis when all endpoint fits converge, "
        "bootstrap convergence is at least 90% for every endpoint, and fixed-effect directions agree. "
        "Retain the previous 65 scanner-specific contrast models as a supplementary sensitivity analysis.\n"
    )
    (output_root / "decision.md").write_text(decision, encoding="utf-8")

    contract = {
        "analysis_version": ANALYSIS_VERSION,
        "created_utc": utc_now(),
        "primary_question": "Is there an explicit target-scanner-by-tissue random interaction after accounting for shared AT2 correlation?",
        "input": str(input_path.resolve()),
        "input_sha256": sha256(input_path),
        "old_model_summary": str(old_summary_path.resolve()),
        "old_model_summary_sha256": sha256(old_summary_path),
        "model_source": str(source_path),
        "model_implementation_sha256": hashlib.sha256(model_source).hexdigest(),
        "reporting_script_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "software": {
            "python": os.sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "platform": os.uname().sysname + "-" + os.uname().release,
        },
        "scanners": list(SCANNERS),
        "endpoints": endpoints,
        "model": "target fixed effects + shared tissue + scanner:tissue + shared slide + scanner:slide + matched replicate + residual",
        "primary_test": "full versus no scanner:tissue variance; parametric-bootstrap restricted LRT",
        "multiplicity": "Benjamini-Hochberg across 13 endpoint-level interaction tests",
        "bootstrap_replicates_per_endpoint": int(summary["bootstrap_requested"].iloc[0]),
        "seed": int(args.seed),
        "reference": "AT2",
    }
    write_json(output_root / "analysis_contract.json", contract)
    manifest_paths = [
        output_root / "analysis_contract.json",
        output_root / "endpoint_model_summary.csv",
        output_root / "interaction_lrt.csv",
        output_root / "primary_vs_joint_comparison.csv",
        output_root / "decision.md",
        figure_path,
        *bootstrap_files,
    ]
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "completed_utc": utc_now(),
        "status": "complete",
        "files": {
            str(path.relative_to(output_root)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in manifest_paths
        },
    }
    write_json(output_root / "manifest.json", manifest)
    print(decision)


def self_test() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for tissue in range(4):
        tissue_shared = rng.normal(0, 0.2)
        tissue_scanner = rng.normal(0, 0.5, len(SCANNERS))
        for slide_index in range(2):
            slide = f"t{tissue}_s{slide_index}"
            slide_shared = rng.normal(0, 0.2)
            slide_scanner = rng.normal(0, 0.3, len(SCANNERS))
            for replicate in range(3):
                shared = rng.normal(0, 0.1)
                for scanner_index, scanner in enumerate(SCANNERS):
                    value = (
                        scanner_index * 0.2
                        + tissue_shared
                        + tissue_scanner[scanner_index]
                        + slide_shared
                        + slide_scanner[scanner_index]
                        + shared
                        + rng.normal(0, 0.1)
                    )
                    rows.append(
                        {
                            "slide_id": slide,
                            "tissue_type": f"t{tissue}",
                            "replicate": replicate,
                            "scanner": scanner,
                            "value": value,
                        }
                    )
    blocks, scale = build_blocks(pd.DataFrame(rows))
    full = fit_reml(blocks, FULL_ACTIVE, maxiter=150)
    reduced = fit_reml(blocks, NO_INTERACTION_ACTIVE, maxiter=150)
    if not np.isfinite(full["negative_log_likelihood"]):
        raise AssertionError("full likelihood is not finite")
    if full["negative_log_likelihood"] > reduced["negative_log_likelihood"] + 1e-5:
        raise AssertionError("full model should not fit worse than reduced model")
    if scale <= 0 or len(full["beta"]) != len(SCANNERS):
        raise AssertionError("invalid scale or fixed effects")
    print("self-test passed")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    endpoint = subparsers.add_parser("run-endpoint")
    endpoint.add_argument("--input", required=True)
    endpoint.add_argument("--output-root", required=True)
    endpoint.add_argument("--task-index", type=int, required=True)
    endpoint.add_argument("--bootstrap", type=int, default=500)
    endpoint.add_argument("--workers", type=int, default=1)
    endpoint.add_argument("--seed", type=int, default=20260922)
    endpoint.set_defaults(function=run_endpoint)

    combined = subparsers.add_parser("aggregate")
    combined.add_argument("--input", required=True)
    combined.add_argument("--old-summary", required=True)
    combined.add_argument("--output-root", required=True)
    combined.add_argument("--seed", type=int, default=20260922)
    combined.set_defaults(function=aggregate)

    test = subparsers.add_parser("self-test")
    test.set_defaults(function=lambda _: self_test())
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
