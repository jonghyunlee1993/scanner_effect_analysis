"""Fit the frozen E6 LOSO correction-heterogeneity models.

The input consists of five disjoint 20-location replicate means per physical
slide.  Each PFM/method radius contrast and each PFM/method/source-scanner
content contrast is fit with the nested profiled-REML implementation already
used for E2:

    contrast = fixed mean + tissue slope + slide-within-tissue slope + error.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_exp05_tissue_random_slopes import (  # noqa: E402
    benjamini_hochberg,
    fit_nested_reml,
)
from e5_comparator_population import (  # noqa: E402
    FEATURE_CONDITIONS,
    IMAGE_CONDITIONS,
    SCANNERS,
)
from fetch_e0_pfm_checkpoints import sha256  # noqa: E402


METHODS = (*IMAGE_CONDITIONS, *FEATURE_CONDITIONS)
MODEL_LABELS = {
    "resnet50": "ResNet50",
    "uni_v1": "UNI v1",
    "conch_v1": "CONCH v1",
    "virchow2": "Virchow2",
}
METHOD_LABELS = {
    "reinhard_lab": "Reinhard",
    "paired_od_affine": "Paired OD",
    "frequency_calibration": "Frequency",
    "coral": "CORAL",
    "orthogonal_procrustes": "Procrustes",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contrasts", default="outputs/e6_loso_replicate_contrasts"
    )
    parser.add_argument(
        "--e5-frontier", default="outputs/e5_comparator_frontier"
    )
    parser.add_argument(
        "--e5-lock", default="outputs/e5_comparator_results_lock/summary.json"
    )
    parser.add_argument(
        "--contract", default="docs/e6_heterogeneity_execution_contract.md"
    )
    parser.add_argument("--output", default="outputs/e6_loso_heterogeneity")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def validate_inputs(contrast_dir: Path, e5_lock_path: Path, contract_path: Path):
    summary_path = contrast_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("contrast_gate_pass") is not True:
        raise RuntimeError("E6 contrast gate has not passed")
    if summary.get("radius_rows") != 10_900 or summary.get("content_rows") != 54_500:
        raise ValueError("E6 contrast population count mismatch")
    if summary.get("e5_result_lock_sha256") != sha256(e5_lock_path):
        raise ValueError("E5 result-lock hash changed after contrast materialization")
    if summary.get("e6_contract_sha256") != sha256(contract_path):
        raise ValueError("E6 contract hash changed after contrast materialization")
    e5_lock = json.loads(e5_lock_path.read_text())
    if e5_lock.get("result_lock_pass") is not True:
        raise RuntimeError("E5 result lock has not passed")
    return summary_path, summary


def validate_frame(frame: pd.DataFrame, value: str, *, content: bool):
    required = {
        "encoder_id",
        "method",
        "tissue_type",
        "slides_in_tissue",
        "slide_id",
        "replicate",
        "locations",
        value,
    }
    if content:
        required.add("scanner")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"contrast table is missing columns: {sorted(missing)}")
    if frame[value].isna().any() or not np.isfinite(frame[value]).all():
        raise ValueError(f"non-finite {value}")
    if set(frame["encoder_id"]) != set(MODEL_LABELS):
        raise ValueError("PFM population mismatch")
    if set(frame["method"]) != set(METHODS):
        raise ValueError("method population mismatch")
    if set(frame["replicate"]) != set(range(5)) or set(frame["locations"]) != {20}:
        raise ValueError("replicate population mismatch")
    if frame["slide_id"].nunique() != 109 or frame["tissue_type"].nunique() != 37:
        raise ValueError("slide/tissue population mismatch")
    if content and set(frame["scanner"]) != set(SCANNERS[1:]):
        raise ValueError("source-scanner population mismatch")
    observed_sizes = (
        frame[["slide_id", "tissue_type", "slides_in_tissue"]]
        .drop_duplicates()
        .groupby("tissue_type")
        .agg(slides=("slide_id", "nunique"), recorded=("slides_in_tissue", "nunique"))
    )
    if not (observed_sizes["recorded"] == 1).all():
        raise ValueError("inconsistent tissue class-size annotations")


def _fit_task(task):
    endpoint, analysis_set, keys, value_column, frame = task
    renamed = frame.rename(columns={value_column: "log2_relative_transfer"})
    result = fit_nested_reml(renamed)
    tissue_variance, slide_variance, residual_variance = result["variances"]
    n_replicates = result["n_replicates"]
    tissue_count = result["n_tissues"]
    fixed_mean = result["full"]["fixed_mean"]
    fixed_se = result["full"]["fixed_se"]
    critical = float(stats.t.ppf(0.975, df=tissue_count - 1))
    fixed_t = fixed_mean / fixed_se
    slide_mean_sampling_variance = residual_variance / n_replicates
    total_slide_mean_variance = (
        tissue_variance + slide_variance + slide_mean_sampling_variance
    )
    component = {
        "endpoint": endpoint,
        "analysis_set": analysis_set,
        **keys,
        "n_tissues": tissue_count,
        "n_slides": int(frame["slide_id"].nunique()),
        "replicate_groups": n_replicates,
        "fixed_mean": fixed_mean,
        "fixed_se": fixed_se,
        "fixed_ci95_low": fixed_mean - critical * fixed_se,
        "fixed_ci95_high": fixed_mean + critical * fixed_se,
        "fixed_t": fixed_t,
        "fixed_p": float(2.0 * stats.t.sf(abs(fixed_t), df=tissue_count - 1)),
        "tissue_variance": tissue_variance,
        "tissue_sd": float(np.sqrt(tissue_variance)),
        "slide_within_tissue_variance": slide_variance,
        "slide_within_tissue_sd": float(np.sqrt(slide_variance)),
        "replicate_sampling_variance": residual_variance,
        "replicate_sampling_sd": float(np.sqrt(residual_variance)),
        "sampling_variance_slide_mean": slide_mean_sampling_variance,
        "sampling_sd_slide_mean": float(np.sqrt(slide_mean_sampling_variance)),
        "tissue_fraction_total_slide_mean_variance": (
            tissue_variance / total_slide_mean_variance
        ),
        "slide_fraction_total_slide_mean_variance": (
            slide_variance / total_slide_mean_variance
        ),
        "sampling_fraction_total_slide_mean_variance": (
            slide_mean_sampling_variance / total_slide_mean_variance
        ),
        "tissue_fraction_between_slide_variance": (
            tissue_variance / (tissue_variance + slide_variance)
        ),
        "tissue_variance_lrt": result["likelihood_ratio"],
        "tissue_variance_p_mixture": result["tissue_variance_p"],
        "optimizer_converged": bool(result["fit"].success),
        "optimizer_message": str(result["fit"].message),
        "reduced_optimizer_converged": bool(result["reduced_fit"].success),
        "reduced_optimizer_message": str(result["reduced_fit"].message),
        "reml_negative_log_likelihood": float(result["fit"].fun),
    }
    tissue_rows = []
    for row in result["tissue_rows"]:
        raw_deviation = row["raw_tissue_slope"]
        blup_deviation = row["blup_tissue_slope"]
        tissue_rows.append({
            "endpoint": endpoint,
            "analysis_set": analysis_set,
            **keys,
            "tissue_type": row["tissue_type"],
            "slides_in_tissue": row["slides_in_tissue"],
            "fixed_mean": fixed_mean,
            "raw_tissue_deviation": raw_deviation,
            "raw_tissue_contrast": fixed_mean + raw_deviation,
            "blup_tissue_deviation": blup_deviation,
            "predicted_tissue_contrast": fixed_mean + blup_deviation,
            "blup_se_conditional": row["blup_se_conditional"],
            "signed_shrinkage_toward_fixed": raw_deviation - blup_deviation,
            "absolute_shrinkage": abs(raw_deviation) - abs(blup_deviation),
        })
    slide_rows = [
        {
            "endpoint": endpoint,
            "analysis_set": analysis_set,
            **keys,
            **row,
        }
        for row in result["slide_rows"]
    ]
    return component, tissue_rows, slide_rows


def make_tasks(radius: pd.DataFrame, content: pd.DataFrame):
    tasks = []
    for analysis_set, minimum_size in (("full_37_tissues", 1), ("min3_sensitivity", 3)):
        radius_set = radius[radius["slides_in_tissue"] >= minimum_size]
        content_set = content[content["slides_in_tissue"] >= minimum_size]
        for (encoder_id, method), frame in radius_set.groupby(
            ["encoder_id", "method"], sort=True
        ):
            tasks.append((
                "radius_benefit",
                analysis_set,
                {"encoder_id": encoder_id, "method": method},
                "delta_radius",
                frame.copy(),
            ))
        for (encoder_id, method, scanner), frame in content_set.groupby(
            ["encoder_id", "method", "scanner"], sort=True
        ):
            tasks.append((
                "content_change",
                analysis_set,
                {"encoder_id": encoder_id, "method": method, "scanner": scanner},
                "delta_content_margin",
                frame.copy(),
            ))
    return tasks


def fit_all(tasks, workers: int):
    if workers == 1:
        results = [_fit_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_fit_task, tasks, chunksize=1))
    components = pd.DataFrame([result[0] for result in results])
    tissue_rows = pd.DataFrame(
        [row for result in results for row in result[1]]
    )
    slide_rows = pd.DataFrame(
        [row for result in results for row in result[2]]
    )
    for (_, analysis_set), indices in components.groupby(
        ["endpoint", "analysis_set"], sort=False
    ).groups.items():
        components.loc[indices, "tissue_variance_q_bh"] = benjamini_hochberg(
            components.loc[indices, "tissue_variance_p_mixture"]
        )
    components["tissue_heterogeneity_claim"] = (
        components["tissue_variance_q_bh"] < 0.05
    )
    sort_keys = ["endpoint", "analysis_set", "encoder_id", "method", "scanner"]
    components = components.sort_values(
        [column for column in sort_keys if column in components], na_position="first"
    ).reset_index(drop=True)
    tissue_rows = tissue_rows.sort_values(
        [column for column in sort_keys + ["tissue_type"] if column in tissue_rows],
        na_position="first",
    ).reset_index(drop=True)
    slide_rows = slide_rows.sort_values(
        [column for column in sort_keys + ["tissue_type", "slide_id"] if column in slide_rows],
        na_position="first",
    ).reset_index(drop=True)
    return components, tissue_rows, slide_rows


def build_winner_diagnostic(tissue_rows: pd.DataFrame, endpoints: pd.DataFrame):
    endpoints = endpoints[
        endpoints["condition"].isin(METHODS)
    ].copy()
    safe = endpoints[endpoints["safe_for_pfm"].astype(bool)]
    rows = []
    summary_rows = []
    radius = tissue_rows[tissue_rows["endpoint"] == "radius_benefit"]
    for analysis_set in ("full_37_tissues", "min3_sensitivity"):
        analysis = radius[radius["analysis_set"] == analysis_set]
        for encoder_id in MODEL_LABELS:
            safe_methods = set(
                safe.loc[safe["encoder_id"] == encoder_id, "condition"]
            )
            global_table = safe[
                (safe["encoder_id"] == encoder_id)
                & safe["safe_and_invariance_improved"].astype(bool)
            ].sort_values(
                ["safe_improved_rr_rank", "condition"], na_position="last"
            )
            if global_table.empty:
                raise ValueError(f"{encoder_id}: no frozen safe-and-improved winner")
            global_winner = str(global_table.iloc[0]["condition"])
            if global_winner != "orthogonal_procrustes":
                raise ValueError(f"{encoder_id}: unexpected frozen global winner {global_winner}")
            pfm = analysis[
                (analysis["encoder_id"] == encoder_id)
                & analysis["method"].isin(safe_methods)
            ]
            tissue_winners = []
            for tissue_type, frame in pfm.groupby("tissue_type", sort=True):
                ranked = frame.sort_values(
                    ["predicted_tissue_contrast", "method"],
                    ascending=[False, True],
                ).copy()
                predicted_winner = str(ranked.iloc[0]["method"])
                retained = predicted_winner == global_winner
                tissue_winners.append(retained)
                for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                    rows.append({
                        "analysis_set": analysis_set,
                        "encoder_id": encoder_id,
                        "tissue_type": tissue_type,
                        "slides_in_tissue": int(row["slides_in_tissue"]),
                        "method": row["method"],
                        "globally_safe_method": True,
                        "predicted_radius_benefit": row["predicted_tissue_contrast"],
                        "predicted_rank": rank,
                        "predicted_tissue_winner": predicted_winner,
                        "frozen_global_winner": global_winner,
                        "global_winner_remains_first": retained,
                    })
            summary_rows.append({
                "analysis_set": analysis_set,
                "encoder_id": encoder_id,
                "frozen_global_winner": global_winner,
                "globally_safe_methods": len(safe_methods),
                "tissues": len(tissue_winners),
                "global_winner_first_tissues": int(sum(tissue_winners)),
                "global_winner_first_fraction": float(np.mean(tissue_winners)),
            })
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def render_figure(components: pd.DataFrame, winner_summary: pd.DataFrame, output: Path):
    full = components[components["analysis_set"] == "full_37_tissues"]
    radius = full[full["endpoint"] == "radius_benefit"]
    content = full[full["endpoint"] == "content_change"]
    figure, axes = plt.subplots(1, 3, figsize=(18.5, 5.3))

    matrix = radius.pivot(
        index="encoder_id", columns="method", values="tissue_sd"
    ).reindex(index=MODEL_LABELS, columns=METHODS)
    q_matrix = radius.pivot(
        index="encoder_id", columns="method", values="tissue_variance_q_bh"
    ).reindex(index=MODEL_LABELS, columns=METHODS)
    image = axes[0].imshow(matrix, cmap="magma", aspect="auto")
    color_scale = plt.Normalize(
        vmin=float(np.nanmin(matrix.to_numpy())),
        vmax=float(np.nanmax(matrix.to_numpy())),
    )
    color_map = plt.get_cmap("magma")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            marker = "*" if q_matrix.iloc[row, column] < 0.05 else ""
            red, green, blue, _ = color_map(color_scale(matrix.iloc[row, column]))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            axes[0].text(
                column,
                row,
                f"{matrix.iloc[row, column]:.3f}{marker}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if luminance < 0.52 else "black",
            )
    axes[0].set_xticks(range(len(METHODS)), [METHOD_LABELS[m] for m in METHODS], rotation=35, ha="right")
    axes[0].set_yticks(range(len(MODEL_LABELS)), [MODEL_LABELS[m] for m in MODEL_LABELS])
    axes[0].set_title("A  Radius-benefit tissue SD")
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    content_fraction = (
        content.groupby(["encoder_id", "method"])["tissue_heterogeneity_claim"]
        .mean()
        .unstack("method")
        .reindex(index=MODEL_LABELS, columns=METHODS)
    )
    image = axes[1].imshow(content_fraction, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for row in range(content_fraction.shape[0]):
        for column in range(content_fraction.shape[1]):
            value = content_fraction.iloc[row, column]
            axes[1].text(
                column,
                row,
                f"{int(round(5 * value))}/5",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value >= 0.6 else "black",
            )
    axes[1].set_xticks(range(len(METHODS)), [METHOD_LABELS[m] for m in METHODS], rotation=35, ha="right")
    axes[1].set_yticks(range(len(MODEL_LABELS)), [MODEL_LABELS[m] for m in MODEL_LABELS])
    axes[1].set_title("B  Source scanners with content heterogeneity")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="Fraction (BH q<0.05)")

    full_winner = winner_summary[
        winner_summary["analysis_set"] == "full_37_tissues"
    ].set_index("encoder_id").reindex(MODEL_LABELS)
    sensitivity = winner_summary[
        winner_summary["analysis_set"] == "min3_sensitivity"
    ].set_index("encoder_id").reindex(MODEL_LABELS)
    x = np.arange(len(MODEL_LABELS))
    width = 0.36
    axes[2].bar(
        x - width / 2,
        full_winner["global_winner_first_fraction"],
        width,
        color="#2563EB",
        label="all 37 tissues",
    )
    axes[2].bar(
        x + width / 2,
        sensitivity["global_winner_first_fraction"],
        width,
        color="#93C5FD",
        label="≥3-slide tissues",
    )
    axes[2].set_xticks(x, [MODEL_LABELS[m] for m in MODEL_LABELS], rotation=25, ha="right")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Fraction of tissues")
    axes[2].set_title("C  Global Procrustes winner remains first")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].axhline(1.0, color="#6B7280", linewidth=0.8, linestyle="--")
    for offset, table in ((-width / 2, full_winner), (width / 2, sensitivity)):
        for index, row in enumerate(table.itertuples()):
            axes[2].text(
                index + offset,
                row.global_winner_first_fraction - 0.035,
                f"{row.global_winner_first_tissues}/{row.tissues}",
                ha="center",
                va="top",
                fontsize=8,
                color="white",
            )

    figure.suptitle(
        "Cross-fitted correction response varies across tissue and slide content",
        fontsize=14,
        y=1.02,
    )
    figure.text(
        0.01,
        0.005,
        "* tissue-variance mixture LRT, BH q<0.05 within the 20-cell radius family",
        fontsize=8,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    paths = {}
    for suffix in ("png", "pdf"):
        path = output / f"figure_e6_correction_heterogeneity.{suffix}"
        figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths[suffix] = str(path)
    plt.close(figure)
    return paths


def main():
    args = parse_args()
    contrast_dir = Path(args.contrasts)
    e5_frontier = Path(args.e5_frontier)
    e5_lock_path = Path(args.e5_lock)
    contract_path = Path(args.contract)
    contrast_summary_path, contrast_summary = validate_inputs(
        contrast_dir, e5_lock_path, contract_path
    )
    radius_path = contrast_dir / "radius_replicate_contrasts.csv"
    content_path = contrast_dir / "content_replicate_contrasts.csv"
    if sha256(radius_path) != contrast_summary["radius_sha256"]:
        raise ValueError("radius contrast hash mismatch")
    if sha256(content_path) != contrast_summary["content_sha256"]:
        raise ValueError("content contrast hash mismatch")
    radius = pd.read_csv(radius_path, dtype={"slide_id": str})
    content = pd.read_csv(content_path, dtype={"slide_id": str})
    validate_frame(radius, "delta_radius", content=False)
    validate_frame(content, "delta_content_margin", content=True)
    tissue_classes = (
        radius[["slide_id", "tissue_type", "slides_in_tissue"]]
        .drop_duplicates()
        .groupby("tissue_type", as_index=False)
        .agg(slides=("slide_id", "nunique"), recorded_size=("slides_in_tissue", "first"))
        .sort_values(["slides", "tissue_type"])
    )
    tissue_classes["included_min3_sensitivity"] = tissue_classes["slides"] >= 3
    if (
        len(tissue_classes) != 37
        or (tissue_classes["slides"] == 1).sum() != 1
        or (tissue_classes["slides"] == 2).sum() != 5
        or tissue_classes.loc[tissue_classes["slides"] >= 3, "slides"].sum() != 98
    ):
        raise ValueError("frozen tissue class-size sensitivity population mismatch")

    tasks = make_tasks(radius, content)
    if len(tasks) != 240:
        raise ValueError(f"expected 240 full+sensitivity fits, got {len(tasks)}")
    components, tissue_rows, slide_rows = fit_all(tasks, args.workers)
    endpoints_path = e5_frontier / "endpoint_summary.csv"
    endpoints = pd.read_csv(endpoints_path)
    winner_rows, winner_summary = build_winner_diagnostic(tissue_rows, endpoints)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    table_paths = {
        "variance_components": output / "variance_components.csv",
        "tissue_blups": output / "tissue_blups.csv",
        "slide_within_tissue_blups": output / "slide_within_tissue_blups.csv",
        "tissue_class_sizes": output / "tissue_class_sizes.csv",
        "tissue_winner_ranks": output / "tissue_winner_ranks.csv",
        "winner_summary": output / "winner_summary.csv",
    }
    components.to_csv(table_paths["variance_components"], index=False)
    tissue_rows.to_csv(table_paths["tissue_blups"], index=False)
    slide_rows.to_csv(table_paths["slide_within_tissue_blups"], index=False)
    tissue_classes.to_csv(table_paths["tissue_class_sizes"], index=False)
    winner_rows.to_csv(table_paths["tissue_winner_ranks"], index=False)
    winner_summary.to_csv(table_paths["winner_summary"], index=False)
    figures = render_figure(components, winner_summary, output)

    observed_counts = {
        "component_rows": len(components),
        "radius_component_rows": int((components["endpoint"] == "radius_benefit").sum()),
        "content_component_rows": int((components["endpoint"] == "content_change").sum()),
        "tissue_blup_rows": len(tissue_rows),
        "slide_blup_rows": len(slide_rows),
        "winner_rank_rows": len(winner_rows),
    }
    expected_counts = {
        "component_rows": 240,
        "radius_component_rows": 40,
        "content_component_rows": 200,
        "tissue_blup_rows": 8_160,
        "slide_blup_rows": 24_840,
        "winner_rank_rows": 1_224,
    }
    convergence_pass = bool(
        components["optimizer_converged"].all()
        and components["reduced_optimizer_converged"].all()
    )
    gate = bool(
        observed_counts == expected_counts
        and convergence_pass
        and np.isfinite(components.select_dtypes(include=[np.number])).all().all()
        and set(winner_summary["frozen_global_winner"]) == {"orthogonal_procrustes"}
        and len(winner_summary) == 8
    )
    artifacts = [*table_paths.values(), *(Path(path) for path in figures.values())]
    summary = {
        "analysis": "e6_loso_correction_heterogeneity",
        "estimation": "profiled REML, nested tissue and slide-within-tissue random slopes",
        "primary_population": "E5 exact LOSO cross-fitted held-out predictions",
        "models": 4,
        "methods": list(METHODS),
        "source_scanners": list(SCANNERS[1:]),
        "full_tissues": 37,
        "full_slides": 109,
        "min3_tissues": 31,
        "min3_slides": 98,
        "replicate_groups_per_slide": 5,
        "locations_per_replicate": 20,
        "bh_families": {"radius": 20, "content": 100},
        "observed_counts": observed_counts,
        "expected_counts": expected_counts,
        "convergence_pass": convergence_pass,
        "radius_heterogeneity_claims_full": int(
            ((components["endpoint"] == "radius_benefit")
             & (components["analysis_set"] == "full_37_tissues")
             & components["tissue_heterogeneity_claim"]).sum()
        ),
        "content_heterogeneity_claims_full": int(
            ((components["endpoint"] == "content_change")
             & (components["analysis_set"] == "full_37_tissues")
             & components["tissue_heterogeneity_claim"]).sum()
        ),
        "winner_summary": winner_summary.to_dict("records"),
        "contrast_summary_sha256": sha256(contrast_summary_path),
        "radius_contrasts_sha256": sha256(radius_path),
        "content_contrasts_sha256": sha256(content_path),
        "e5_result_lock_sha256": sha256(e5_lock_path),
        "e5_endpoint_summary_sha256": sha256(endpoints_path),
        "e6_contract_sha256": sha256(contract_path),
        "artifacts": {
            path.name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in artifacts
        },
        "analysis_gate_pass": gate,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
