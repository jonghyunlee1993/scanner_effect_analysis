"""Summarize the frozen E7 grouped tissue probe and build main Figure 6."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from e5_comparator_population import FEATURE_CONDITIONS, IMAGE_CONDITIONS, SCANNERS
from e6_loto_population import load_tissue_annotation
from fetch_e0_pfm_checkpoints import sha256


CONDITIONS = ("raw", *IMAGE_CONDITIONS, *FEATURE_CONDITIONS)
METHODS = (*IMAGE_CONDITIONS, *FEATURE_CONDITIONS)
METRICS = (
    "top1_accuracy",
    "top5_accuracy",
    "correct_tissue_margin",
    "centroid_profile_agreement",
)
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 20_260_804
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
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--rows", default="outputs/e7_tissue_probe_rows")
    parser.add_argument("--geometry", default="outputs/e0_native_geometry_final/native_geometry_manifest.csv")
    parser.add_argument("--e7-contract", default="docs/e7_tissue_probe_execution_contract.md")
    parser.add_argument("--e5-frontier", default="outputs/e5_comparator_frontier")
    parser.add_argument("--output", default="outputs/e7_tissue_probe")
    return parser.parse_args()


def bootstrap_weights(slide_frame: pd.DataFrame, minimum_size: int, seed: int):
    selected = slide_frame[slide_frame["slides_in_tissue"] >= minimum_size].copy()
    selected = selected.sort_values("slide_id").reset_index(drop=True)
    tissues = sorted(selected["tissue_type"].unique())
    slide_ids = selected["slide_id"].tolist()
    index_by_tissue = {
        tissue: selected.index[selected["tissue_type"] == tissue].to_numpy()
        for tissue in tissues
    }
    rng = np.random.default_rng(seed)
    weights = np.zeros((BOOTSTRAP_REPLICATES, len(selected)), dtype=np.float32)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_tissues = rng.integers(0, len(tissues), size=len(tissues))
        for tissue_index in sampled_tissues:
            indices = index_by_tissue[tissues[tissue_index]]
            sampled_slides = rng.choice(indices, size=len(indices), replace=True)
            np.add.at(
                weights[replicate],
                sampled_slides,
                1.0 / (len(tissues) * len(indices)),
            )
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("invalid hierarchical bootstrap weights")
    return selected, tissues, slide_ids, weights


def tissue_balanced_estimate(values: np.ndarray, tissue_labels: np.ndarray):
    return float(
        np.mean(
            [values[tissue_labels == tissue].mean() for tissue in sorted(set(tissue_labels))]
        )
    )


def summarize_vector(frame: pd.DataFrame, metric: str, plan):
    selected, _, slide_ids, weights = plan
    ordered = selected[["slide_id", "tissue_type"]].merge(
        frame[["slide_id", metric]],
        on="slide_id",
        how="left",
        validate="one_to_one",
    )
    if ordered[metric].isna().any() or ordered["slide_id"].tolist() != slide_ids:
        raise ValueError(f"{metric}: incomplete slide vector for bootstrap")
    values = ordered[metric].to_numpy(dtype=float)
    point = tissue_balanced_estimate(
        values, ordered["tissue_type"].to_numpy()
    )
    bootstraps = weights @ values
    lower, upper = np.quantile(bootstraps, [0.025, 0.975])
    return point, float(lower), float(upper), values


def summarize_source(source: pd.DataFrame, plans: dict):
    endpoint_rows = []
    scanner_rows = []
    transfer_rows = []
    for analysis_set, plan in plans.items():
        for population in ("loso", "loto"):
            population_frame = source[source["correction_population"] == population]
            for model_id in MODEL_LABELS:
                model_frame = population_frame[population_frame["encoder_id"] == model_id]
                pooled = (
                    model_frame.groupby(
                        [
                            "condition",
                            "slide_id",
                            "tissue_type",
                            "slides_in_tissue",
                        ],
                        as_index=False,
                    )[list(METRICS)]
                    .mean()
                )
                raw_pooled = pooled[pooled["condition"] == "raw"]
                for condition in CONDITIONS:
                    condition_frame = pooled[pooled["condition"] == condition]
                    for metric in METRICS:
                        point, lower, upper, _ = summarize_vector(
                            condition_frame, metric, plan
                        )
                        raw_point, _, _, _ = summarize_vector(
                            raw_pooled, metric, plan
                        )
                        difference_frame = condition_frame[
                            ["slide_id", "tissue_type", metric]
                        ].merge(
                            raw_pooled[["slide_id", metric]],
                            on="slide_id",
                            suffixes=("_condition", "_raw"),
                            validate="one_to_one",
                        )
                        difference_frame[metric] = (
                            difference_frame[f"{metric}_condition"]
                            - difference_frame[f"{metric}_raw"]
                        )
                        delta, delta_lower, delta_upper, _ = summarize_vector(
                            difference_frame, metric, plan
                        )
                        endpoint_rows.append({
                            "analysis_set": analysis_set,
                            "correction_population": population,
                            "encoder_id": model_id,
                            "condition": condition,
                            "metric": metric,
                            "tissues": len(plan[1]),
                            "slides": len(plan[2]),
                            "estimate": point,
                            "ci95_lower": lower,
                            "ci95_upper": upper,
                            "raw_estimate": raw_point,
                            "delta_from_raw": delta,
                            "delta_ci95_lower": delta_lower,
                            "delta_ci95_upper": delta_upper,
                        })
                for scanner in SCANNERS[1:]:
                    scanner_frame = model_frame[model_frame["scanner"] == scanner]
                    raw_scanner = scanner_frame[scanner_frame["condition"] == "raw"]
                    for condition in CONDITIONS:
                        condition_frame = scanner_frame[
                            scanner_frame["condition"] == condition
                        ]
                        for metric in METRICS:
                            point, lower, upper, _ = summarize_vector(
                                condition_frame, metric, plan
                            )
                            raw_point, _, _, _ = summarize_vector(
                                raw_scanner, metric, plan
                            )
                            difference_frame = condition_frame[
                                ["slide_id", "tissue_type", metric]
                            ].merge(
                                raw_scanner[["slide_id", metric]],
                                on="slide_id",
                                suffixes=("_condition", "_raw"),
                                validate="one_to_one",
                            )
                            difference_frame[metric] = (
                                difference_frame[f"{metric}_condition"]
                                - difference_frame[f"{metric}_raw"]
                            )
                            delta, delta_lower, delta_upper, _ = summarize_vector(
                                difference_frame, metric, plan
                            )
                            scanner_rows.append({
                                "analysis_set": analysis_set,
                                "correction_population": population,
                                "encoder_id": model_id,
                                "condition": condition,
                                "scanner": scanner,
                                "metric": metric,
                                "tissues": len(plan[1]),
                                "slides": len(plan[2]),
                                "estimate": point,
                                "ci95_lower": lower,
                                "ci95_upper": upper,
                                "raw_estimate": raw_point,
                                "delta_from_raw": delta,
                                "delta_ci95_lower": delta_lower,
                                "delta_ci95_upper": delta_upper,
                            })
        endpoint = pd.DataFrame(endpoint_rows)
        local = endpoint[
            (endpoint["analysis_set"] == analysis_set)
            & endpoint["condition"].isin(METHODS)
        ]
        for (model_id, condition, metric), frame in local.groupby(
            ["encoder_id", "condition", "metric"]
        ):
            loso = frame[frame["correction_population"] == "loso"].iloc[0]
            loto = frame[frame["correction_population"] == "loto"].iloc[0]
            # Exact paired LOTO-minus-LOSO uncertainty is reconstructed below from slide rows.
            model_source = source[source["encoder_id"] == model_id]
            difference = []
            for population in ("loso", "loto"):
                local_source = model_source[
                    (model_source["correction_population"] == population)
                    & (model_source["condition"] == condition)
                ]
                local_source = (
                    local_source.groupby(["slide_id", "tissue_type"], as_index=False)[metric]
                    .mean()
                    .rename(columns={metric: population})
                )
                difference.append(local_source)
            paired = difference[0].merge(
                difference[1], on=["slide_id", "tissue_type"], validate="one_to_one"
            )
            paired[metric] = paired["loto"] - paired["loso"]
            delta, lower, upper, _ = summarize_vector(paired, metric, plan)
            transfer_rows.append({
                "analysis_set": analysis_set,
                "encoder_id": model_id,
                "condition": condition,
                "metric": metric,
                "loso_estimate": loso["estimate"],
                "loto_estimate": loto["estimate"],
                "loto_minus_loso": delta,
                "difference_ci95_lower": lower,
                "difference_ci95_upper": upper,
            })
    return (
        pd.DataFrame(endpoint_rows),
        pd.DataFrame(scanner_rows),
        pd.DataFrame(transfer_rows),
    )


def summarize_at2(at2: pd.DataFrame, plans: dict):
    rows = []
    for analysis_set, plan in plans.items():
        for model_id in MODEL_LABELS:
            frame = at2[at2["encoder_id"] == model_id]
            for metric in METRICS:
                point, lower, upper, _ = summarize_vector(frame, metric, plan)
                rows.append({
                    "analysis_set": analysis_set,
                    "encoder_id": model_id,
                    "metric": metric,
                    "tissues": len(plan[1]),
                    "slides": len(plan[2]),
                    "estimate": point,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                })
    return pd.DataFrame(rows)


def render_figure(
    endpoints: pd.DataFrame,
    tissue_sizes: pd.DataFrame,
    e5_frontier: Path,
    output: Path,
):
    e5 = pd.read_csv(e5_frontier / "endpoint_summary.csv")
    collapse = pd.read_csv(e5_frontier / "collapse_summary.csv")
    actual = e5[e5["condition"].isin(METHODS)]
    content_lower = actual.pivot(
        index="encoder_id", columns="condition", values="delta_content_ci_lower"
    ).reindex(index=MODEL_LABELS, columns=METHODS)
    worst_collapse = (
        collapse[collapse["condition"].isin(METHODS)]
        .groupby(["encoder_id", "condition"])["ratio_ci_lower"]
        .min()
        .unstack("condition")
        .reindex(index=MODEL_LABELS, columns=METHODS)
    )
    probe = endpoints[
        (endpoints["analysis_set"] == "full_evaluable")
        & (endpoints["metric"] == "top1_accuracy")
        & endpoints["condition"].isin(METHODS)
    ]
    loso_probe = probe[probe["correction_population"] == "loso"].pivot(
        index="encoder_id", columns="condition", values="delta_from_raw"
    ).reindex(index=MODEL_LABELS, columns=METHODS)
    loto_probe = probe[probe["correction_population"] == "loto"].pivot(
        index="encoder_id", columns="condition", values="delta_from_raw"
    ).reindex(index=MODEL_LABELS, columns=METHODS)

    figure, axes = plt.subplots(1, 4, figsize=(23, 5.3))
    image = axes[0].imshow(content_lower, cmap="coolwarm", vmin=-0.04, vmax=0.10, aspect="auto")
    for row in range(content_lower.shape[0]):
        for column in range(content_lower.shape[1]):
            value = content_lower.iloc[row, column]
            axes[0].text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=8)
    axes[0].set_title("A  Content ΔM lower CI\n(pass ≥ −0.020)")
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    image = axes[1].imshow(worst_collapse, cmap="viridis", vmin=0.82, vmax=1.03, aspect="auto")
    for row in range(worst_collapse.shape[0]):
        for column in range(worst_collapse.shape[1]):
            value = worst_collapse.iloc[row, column]
            axes[1].text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value < 0.91 else "black",
            )
    axes[1].set_title("B  Worst collapse lower CI\n(pass ≥ 0.850 + point gate)")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

    limit = max(
        0.01,
        float(np.nanmax(np.abs(np.concatenate((loso_probe.to_numpy(), loto_probe.to_numpy()))))),
    )
    image = axes[2].imshow(loso_probe, cmap="PiYG", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(loso_probe.shape[0]):
        for column in range(loso_probe.shape[1]):
            axes[2].text(
                column,
                row,
                f"L {100 * loso_probe.iloc[row, column]:+.1f}\nT {100 * loto_probe.iloc[row, column]:+.1f}",
                ha="center",
                va="center",
                fontsize=7.5,
            )
    axes[2].set_title("C  Tissue top-1 Δ vs raw (pp)\nL=LOSO, T=LOTO")
    figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04, label="LOSO Δ")

    counts = tissue_sizes.groupby("slides").size().reindex(range(1, 7), fill_value=0)
    colors = ["#F59E0B" if size < 3 else "#2563EB" for size in counts.index]
    axes[3].bar(counts.index, counts.values, color=colors)
    for size, count in counts.items():
        axes[3].text(size, count + 0.15, str(count), ha="center", fontsize=9)
    axes[3].set_xticks(range(1, 7))
    axes[3].set_xlabel("Physical slides per tissue type")
    axes[3].set_ylabel("Number of tissue types")
    axes[3].set_title("D  Tissue class size\norange excluded by ≥3 sensitivity")

    for axis in axes[:3]:
        axis.set_xticks(
            range(len(METHODS)),
            [METHOD_LABELS[method] for method in METHODS],
            rotation=35,
            ha="right",
        )
        axis.set_yticks(
            range(len(MODEL_LABELS)),
            [MODEL_LABELS[model] for model in MODEL_LABELS],
        )
    figure.suptitle(
        "Content fidelity gates and coarse tissue-type evidence",
        fontsize=15,
        y=1.03,
    )
    figure.tight_layout()
    paths = {}
    for suffix in ("png", "pdf"):
        path = output / f"figure6_content_tissue_evidence.{suffix}"
        figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths[suffix] = str(path)
    plt.close(figure)
    return paths


def main():
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text())
    e7_contract_path = Path(args.e7_contract)
    if "**Status:** FROZEN" not in e7_contract_path.read_text():
        raise RuntimeError("E7 tissue-probe execution contract is not frozen")
    source_frames = []
    at2_frames = []
    input_manifest = []
    for model in contract["models"]:
        root = Path(args.rows) / model["encoder_id"]
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text())
        source_path = root / "slide_source_probe.csv"
        at2_path = root / "slide_at2_probe.csv"
        if not (
            summary.get("row_gate_pass") is True
            and summary.get("source_rows") == 6_480
            and summary.get("at2_rows") == 108
            and summary.get("e7_contract_sha256") == sha256(e7_contract_path)
            and summary.get("source_sha256") == sha256(source_path)
            and summary.get("at2_sha256") == sha256(at2_path)
        ):
            raise RuntimeError(f"{model['encoder_id']}: E7 row gate has not passed")
        source_frames.append(pd.read_csv(source_path, dtype={"slide_id": str}))
        at2_frames.append(pd.read_csv(at2_path, dtype={"slide_id": str}))
        input_manifest.append({
            "encoder_id": model["encoder_id"],
            "summary_sha256": sha256(summary_path),
            "source_sha256": sha256(source_path),
            "at2_sha256": sha256(at2_path),
        })
    source = pd.concat(source_frames, ignore_index=True)
    at2 = pd.concat(at2_frames, ignore_index=True)
    if len(source) != 25_920 or len(at2) != 432:
        raise ValueError("E7 combined slide-row count mismatch")
    tissue_by_slide, tissues = load_tissue_annotation(Path(args.geometry))
    tissue_sizes = pd.DataFrame([
        {
            "tissue_type": tissue,
            "slides": sum(value == tissue for value in tissue_by_slide.values()),
        }
        for tissue in tissues
    ]).sort_values(["slides", "tissue_type"])
    tissue_sizes["evaluable_probe"] = tissue_sizes["slides"] >= 2
    tissue_sizes["included_min3_sensitivity"] = tissue_sizes["slides"] >= 3
    slide_frame = source[["slide_id", "tissue_type", "slides_in_tissue"]].drop_duplicates()
    plans = {
        "full_evaluable": bootstrap_weights(slide_frame, 2, BOOTSTRAP_SEED),
        "min3_sensitivity": bootstrap_weights(slide_frame, 3, BOOTSTRAP_SEED + 1),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / f".bootstrap_weights.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        temporary,
        full_slide_ids=np.asarray(plans["full_evaluable"][2]),
        full_weights=plans["full_evaluable"][3],
        min3_slide_ids=np.asarray(plans["min3_sensitivity"][2]),
        min3_weights=plans["min3_sensitivity"][3],
    )
    bootstrap_path = output / "hierarchical_bootstrap_weights.npz"
    os.replace(temporary, bootstrap_path)
    endpoints, scanner_endpoints, transfer = summarize_source(source, plans)
    at2_summary = summarize_at2(at2, plans)
    tables = {
        "slide_source_probe.csv": source,
        "slide_at2_probe.csv": at2,
        "endpoint_summary.csv": endpoints,
        "scanner_endpoint_summary.csv": scanner_endpoints,
        "loto_loso_transfer_summary.csv": transfer,
        "at2_endpoint_summary.csv": at2_summary,
        "tissue_class_sizes.csv": tissue_sizes,
    }
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False)
    figures = render_figure(
        endpoints, tissue_sizes, Path(args.e5_frontier), output
    )
    expected_rows = {
        "slide_source_probe.csv": 25_920,
        "slide_at2_probe.csv": 432,
        "endpoint_summary.csv": 384,
        "scanner_endpoint_summary.csv": 1_920,
        "loto_loso_transfer_summary.csv": 160,
        "at2_endpoint_summary.csv": 32,
        "tissue_class_sizes.csv": 37,
    }
    observed_rows = {filename: len(frame) for filename, frame in tables.items()}
    gate = bool(
        observed_rows == expected_rows
        and np.isfinite(endpoints.select_dtypes(include=[np.number])).all().all()
        and np.isfinite(scanner_endpoints.select_dtypes(include=[np.number])).all().all()
        and np.isfinite(transfer.select_dtypes(include=[np.number])).all().all()
        and len(plans["full_evaluable"][1]) == 36
        and len(plans["full_evaluable"][2]) == 108
        and len(plans["min3_sensitivity"][1]) == 31
        and len(plans["min3_sensitivity"][2]) == 98
    )
    summary = {
        "analysis": "e7_grouped_tissue_probe",
        "interpretation": "coarse secondary tissue-type representation diagnostic only",
        "pfms": 4,
        "conditions_including_raw": len(CONDITIONS),
        "correction_populations": ["loso", "loto"],
        "full_evaluable": {"tissues": 36, "slides": 108},
        "min3_sensitivity": {"tissues": 31, "slides": 98},
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed_full": BOOTSTRAP_SEED,
        "bootstrap_seed_min3": BOOTSTRAP_SEED + 1,
        "bootstrap_unit": "tissue then physical slide within tissue",
        "expected_rows": expected_rows,
        "observed_rows": observed_rows,
        "input_manifest": input_manifest,
        "e7_contract_sha256": sha256(e7_contract_path),
        "bootstrap_weights_sha256": sha256(bootstrap_path),
        "tables": {
            filename: {"rows": len(frame), "sha256": sha256(output / filename)}
            for filename, frame in tables.items()
        },
        "figures": {
            Path(path).name: {"path": path, "sha256": sha256(Path(path))}
            for path in figures.values()
        },
        "analysis_gate_pass": gate,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
