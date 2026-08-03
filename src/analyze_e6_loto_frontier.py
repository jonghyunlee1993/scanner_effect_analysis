"""Score exact 37-fold LOTO transfer and compare it with the E5 LOSO frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_e5_comparator_frontier import (
    slide_rows,
    summarize,
    summarize_clipping,
)
from e5_comparator_population import FEATURE_CONDITIONS, IMAGE_CONDITIONS
from e6_loto_population import E6_LOTO_VERSION, load_tissue_annotation
from fetch_e0_pfm_checkpoints import sha256


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
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--image", default="outputs/e6_loto_image_features")
    parser.add_argument("--feature", default="outputs/e6_loto_feature_harmonization")
    parser.add_argument("--audit", default="outputs/e6_loto_features/audit/summary.json")
    parser.add_argument("--e6-contract", default="docs/e6_heterogeneity_execution_contract.md")
    parser.add_argument("--geometry", default="outputs/e0_native_geometry_final/native_geometry_manifest.csv")
    parser.add_argument("--loso", default="outputs/e5_comparator_frontier/endpoint_summary.csv")
    parser.add_argument("--output", default="outputs/e6_loto_frontier")
    return parser.parse_args()


def compare_loso_loto(loso: pd.DataFrame, loto: pd.DataFrame):
    columns = [
        "encoder_id",
        "condition",
        "relative_radius_reduction",
        "difference_ci_lower",
        "delta_content_margin",
        "delta_content_ci_lower",
        "content_noninferiority_pass",
        "collapse_every_scanner_pass",
        "safe_for_pfm",
        "safe_and_invariance_improved",
        "invariance_improved_but_unsafe",
        "safe_improved_rr_rank",
    ]
    left = loso[loso["condition"].isin(METHODS)][columns]
    right = loto[loto["condition"].isin(METHODS)][columns]
    merged = left.merge(
        right,
        on=["encoder_id", "condition"],
        suffixes=("_loso", "_loto"),
        validate="one_to_one",
    )
    merged["relative_radius_reduction_delta_loto_minus_loso"] = (
        merged["relative_radius_reduction_loto"]
        - merged["relative_radius_reduction_loso"]
    )
    merged["content_delta_change_loto_minus_loso"] = (
        merged["delta_content_margin_loto"] - merged["delta_content_margin_loso"]
    )
    merged["safe_decision_changed"] = (
        merged["safe_for_pfm_loto"].astype(bool)
        != merged["safe_for_pfm_loso"].astype(bool)
    )
    merged["safe_improved_decision_changed"] = (
        merged["safe_and_invariance_improved_loto"].astype(bool)
        != merged["safe_and_invariance_improved_loso"].astype(bool)
    )
    return merged.sort_values(["encoder_id", "condition"]).reset_index(drop=True)


def render_figure(endpoints: pd.DataFrame, comparison: pd.DataFrame, output: Path):
    actual = endpoints[endpoints["condition"].isin(METHODS)]
    rr = (
        actual.pivot(index="encoder_id", columns="condition", values="relative_radius_reduction")
        .reindex(index=MODEL_LABELS, columns=METHODS)
        * 100
    )
    delta = (
        comparison.pivot(
            index="encoder_id",
            columns="condition",
            values="relative_radius_reduction_delta_loto_minus_loso",
        ).reindex(index=MODEL_LABELS, columns=METHODS)
        * 100
    )
    safe = actual.pivot(
        index="encoder_id", columns="condition", values="safe_and_invariance_improved"
    ).reindex(index=MODEL_LABELS, columns=METHODS)
    changed = comparison.pivot(
        index="encoder_id", columns="condition", values="safe_improved_decision_changed"
    ).reindex(index=MODEL_LABELS, columns=METHODS)

    figure, axes = plt.subplots(1, 3, figsize=(18.5, 5.2))
    image = axes[0].imshow(rr, cmap="RdYlBu", vmin=-20, vmax=45, aspect="auto")
    for row in range(rr.shape[0]):
        for column in range(rr.shape[1]):
            axes[0].text(column, row, f"{rr.iloc[row, column]:+.1f}%", ha="center", va="center", fontsize=8)
    axes[0].set_title("A  Exact LOTO radius reduction")
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04, label="RR (%)")

    limit = max(1.0, float(np.nanmax(np.abs(delta.to_numpy()))))
    image = axes[1].imshow(delta, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(delta.shape[0]):
        for column in range(delta.shape[1]):
            axes[1].text(column, row, f"{delta.iloc[row, column]:+.2f}", ha="center", va="center", fontsize=8)
    axes[1].set_title("B  LOTO − LOSO radius reduction")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="Percentage points")

    status = safe.astype(float)
    image = axes[2].imshow(status, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    for row in range(status.shape[0]):
        for column in range(status.shape[1]):
            marker = "changed" if changed.iloc[row, column] else "stable"
            decision = "pass" if bool(safe.iloc[row, column]) else "fail"
            axes[2].text(
                column,
                row,
                f"{decision}\n{marker}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if bool(safe.iloc[row, column]) else "black",
            )
    axes[2].set_title("C  Safe + improved transfer decision")
    figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04, ticks=[0, 1])

    for axis in axes:
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
        "Leave-one-tissue-out correction transfer sensitivity",
        fontsize=14,
        y=1.02,
    )
    figure.tight_layout()
    paths = {}
    for suffix in ("png", "pdf"):
        path = output / f"figure_s_loto_transfer_sensitivity.{suffix}"
        figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths[suffix] = str(path)
    plt.close(figure)
    return paths


def main():
    args = parse_args()
    audit_path = Path(args.audit)
    audit = json.loads(audit_path.read_text())
    if not (
        audit.get("audit_pass") is True
        and audit.get("e6_loto_version") == E6_LOTO_VERSION
        and audit.get("total_features_observed") == 1_308_000
    ):
        raise RuntimeError("E6 exact LOTO comparator population audit has not passed")
    e6_contract_path = Path(args.e6_contract)
    if audit.get("e6_contract_sha256") != sha256(e6_contract_path):
        raise RuntimeError("E6 contract hash drift after LOTO population audit")
    tissue_by_slide, tissues = load_tissue_annotation(Path(args.geometry))
    contract = json.loads(Path(args.contract).read_text())
    invariance_rows = []
    content_rows = []
    collapse_rows = []
    clipping_rows = []
    for model in contract["models"]:
        local = slide_rows(
            model, Path(args.raw), Path(args.image), Path(args.feature)
        )
        invariance_rows.extend(local[0])
        content_rows.extend(local[1])
        collapse_rows.extend(local[2])
        clipping_rows.extend(local[3])
    invariance = pd.DataFrame(invariance_rows)
    content = pd.DataFrame(content_rows)
    collapse = pd.DataFrame(collapse_rows)
    clipping = pd.DataFrame(clipping_rows)
    for frame in (invariance, content, collapse, clipping):
        frame["tissue_type"] = frame["slide_id"].map(tissue_by_slide)
        if frame["tissue_type"].isna().any():
            raise ValueError("LOTO slide result lacks tissue identity")
    endpoints, content_scanner, collapse_summary, common, decisions = summarize(
        invariance, content, collapse
    )
    clipping_summary = summarize_clipping(clipping)
    loso_path = Path(args.loso)
    loso = pd.read_csv(loso_path)
    comparison = compare_loso_loto(loso, endpoints)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "slide_invariance.csv": invariance,
        "slide_content_margin.csv": content,
        "slide_collapse.csv": collapse,
        "slide_image_clipping.csv": clipping,
        "endpoint_summary.csv": endpoints,
        "content_scanner_summary.csv": content_scanner,
        "collapse_summary.csv": collapse_summary,
        "common_pfm_summary.csv": common,
        "decision_comparison.csv": decisions,
        "image_clipping_summary.csv": clipping_summary,
        "loso_loto_endpoint_comparison.csv": comparison,
    }
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False)
    figures = render_figure(endpoints, comparison, output)
    summary = {
        "analysis": "e6_exact_loto_transfer_frontier",
        "e6_loto_version": E6_LOTO_VERSION,
        "folds": len(tissues),
        "slides": 109,
        "pfms": 4,
        "methods": list(METHODS),
        "bootstrap_unit": "physical_slide",
        "bootstrap_replicates": 5000,
        "bootstrap_seed": 20260803,
        "endpoint_rows": len(endpoints),
        "common_safe_methods": common[
            common["common_safe_across_four_pfms"]
        ]["condition"].tolist(),
        "common_safe_and_improved_methods": common[
            common["safe_and_improved_all_four_pfms"]
        ]["condition"].tolist(),
        "decision_differs_pfms": int(decisions["decision_differs"].sum()),
        "safe_decision_changes_from_loso": int(comparison["safe_decision_changed"].sum()),
        "safe_improved_decision_changes_from_loso": int(
            comparison["safe_improved_decision_changed"].sum()
        ),
        "max_absolute_rr_change_from_loso": float(
            comparison["relative_radius_reduction_delta_loto_minus_loso"].abs().max()
        ),
        "population_audit_sha256": sha256(audit_path),
        "loso_endpoint_summary_sha256": sha256(loso_path),
        "e6_contract_sha256": sha256(e6_contract_path),
        "tables": {
            filename: {"rows": len(frame), "sha256": sha256(output / filename)}
            for filename, frame in tables.items()
        },
        "figures": {
            Path(path).name: {"path": path, "sha256": sha256(Path(path))}
            for path in figures.values()
        },
        "analysis_gate_pass": bool(
            len(endpoints) == 24
            and len(comparison) == 20
            and len(invariance) == 2_616
            and len(content) == 13_080
            and len(collapse) == 39_240
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["analysis_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
