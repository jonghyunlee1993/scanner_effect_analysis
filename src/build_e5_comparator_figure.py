"""Render Main Figure 5 for the frozen E5 five-method comparator frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from e5_comparator_population import E5_VERSION, FEATURE_CONDITIONS, IMAGE_CONDITIONS
from fetch_e0_pfm_checkpoints import sha256


CONDITIONS = (*IMAGE_CONDITIONS, *FEATURE_CONDITIONS)
LABELS = {
    "reinhard_lab": "Reinhard\nLab",
    "paired_od_affine": "Paired OD\naffine",
    "frequency_calibration": "Frequency\ncalibration",
    "coral": "CORAL",
    "orthogonal_procrustes": "Orthogonal\nProcrustes",
}
MODEL_LABELS = {
    "resnet50": "ResNet50",
    "uni_v1": "UNI v1",
    "conch_v1": "CONCH v1",
    "virchow2": "Virchow2",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--frontier", default="outputs/e5_comparator_frontier")
    parser.add_argument("--output", default="outputs/e5_comparator_figure")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def matrix(frame: pd.DataFrame, models: list[str], column: str):
    indexed = frame.set_index(["encoder_id", "condition"])
    return np.asarray(
        [[indexed.loc[(model, condition), column] for condition in CONDITIONS] for model in models],
        dtype=float,
    )


def heatmap(axis, values, models, title, cmap, norm, formatter, pass_mask=None):
    image = axis.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    axis.set_xticks(range(len(CONDITIONS)), [LABELS[value] for value in CONDITIONS], fontsize=8)
    axis.set_yticks(range(len(models)), [MODEL_LABELS[value] for value in models], fontsize=9)
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "white" if abs(norm(value) - 0.5) > 0.32 else "black"
            suffix = " ✓" if pass_mask is not None and pass_mask[row, column] else ""
            axis.text(column, row, formatter(value) + suffix, ha="center", va="center", fontsize=7.5, color=color)
    axis.tick_params(length=0)
    return image


def main():
    args = parse_args()
    frontier = Path(args.frontier)
    summary_path = frontier / "summary.json"
    summary = json.loads(summary_path.read_text())
    if not (
        summary.get("analysis") == "e5_comparator_frontier"
        and summary.get("e5_version") == E5_VERSION
        and summary.get("analysis_gate_pass") is True
    ):
        raise RuntimeError("E5 comparator frontier has not passed")
    contract = json.loads(Path(args.contract).read_text())
    models = [model["encoder_id"] for model in contract["models"]]
    endpoints = pd.read_csv(frontier / "endpoint_summary.csv")
    endpoints = endpoints[endpoints["condition"].isin(CONDITIONS)].copy()
    collapse = pd.read_csv(frontier / "collapse_summary.csv")
    collapse = collapse[collapse["condition"].isin(CONDITIONS)]
    decisions = pd.read_csv(frontier / "decision_comparison.csv")
    if len(endpoints) != 20 or set(endpoints["encoder_id"]) != set(models):
        raise ValueError("E5 endpoint grid is incomplete")

    rr = 100.0 * matrix(endpoints, models, "relative_radius_reduction")
    content_lower = matrix(endpoints, models, "delta_content_ci_lower")
    content_pass = matrix(endpoints, models, "content_noninferiority_pass").astype(bool)
    collapse_min = (
        collapse.groupby(["encoder_id", "condition"], sort=False)["ratio_ci_lower"]
        .min()
        .reset_index()
    )
    collapse_lower = matrix(collapse_min, models, "ratio_ci_lower")
    collapse_pass = matrix(
        endpoints, models, "collapse_every_scanner_pass"
    ).astype(bool)

    figure = plt.figure(figsize=(13.2, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 0.72])
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, 0])
    axis_d = figure.add_subplot(grid[1, 1])

    rr_limit = max(10.0, float(np.nanmax(np.abs(rr))))
    image_a = heatmap(
        axis_a,
        rr,
        models,
        "A  Scanner-radius reduction (unconstrained)",
        "RdBu_r",
        TwoSlopeNorm(vmin=-rr_limit, vcenter=0.0, vmax=rr_limit),
        lambda value: f"{value:+.1f}%",
        matrix(endpoints, models, "invariance_improved").astype(bool),
    )
    figure.colorbar(image_a, ax=axis_a, fraction=0.045, pad=0.03, label="Relative radius reduction (%)")

    content_extent = max(0.04, float(np.nanmax(np.abs(content_lower))))
    image_b = heatmap(
        axis_b,
        content_lower,
        models,
        "B  Content Δmargin lower 95% CI",
        "PiYG",
        TwoSlopeNorm(vmin=-content_extent, vcenter=-0.02, vmax=max(0.02, content_extent)),
        lambda value: f"{value:+.3f}",
        content_pass,
    )
    figure.colorbar(image_b, ax=axis_b, fraction=0.045, pad=0.03, label="Lower CI (NI boundary −0.02)")

    collapse_extent_low = min(0.5, float(np.nanmin(collapse_lower)))
    collapse_extent_high = max(1.05, float(np.nanmax(collapse_lower)))
    image_c = heatmap(
        axis_c,
        collapse_lower,
        models,
        "C  Worst scanner/metric collapse lower 95% CI",
        "RdYlGn",
        TwoSlopeNorm(vmin=collapse_extent_low, vcenter=0.85, vmax=collapse_extent_high),
        lambda value: f"{value:.3f}",
        collapse_pass,
    )
    figure.colorbar(image_c, ax=axis_c, fraction=0.045, pad=0.03, label="Minimum lower-CI ratio (gate 0.85)")

    axis_d.set_xlim(0, 1)
    axis_d.set_ylim(-0.8, len(models) + 0.5)
    axis_d.axis("off")
    axis_d.set_title("D  Separability-only versus fidelity-constrained choice", loc="left", fontsize=11, fontweight="bold")
    axis_d.text(0.30, len(models) + 0.05, "Unconstrained best", ha="center", fontsize=9, fontweight="bold")
    axis_d.text(0.78, len(models) + 0.05, "Safe + improved best", ha="center", fontsize=9, fontweight="bold")
    indexed = decisions.set_index("encoder_id")
    for row, model in enumerate(models):
        y = len(models) - 1 - row
        record = indexed.loc[model]
        left = LABELS[str(record["unconstrained_best"])].replace("\n", " ")
        right_value = record["fidelity_constrained_best"]
        right = "None" if pd.isna(right_value) else LABELS[str(right_value)].replace("\n", " ")
        left_color = "#2e7d32" if bool(record["unconstrained_best_safe"]) else "#b71c1c"
        axis_d.text(0.0, y, MODEL_LABELS[model], va="center", fontsize=9, fontweight="bold")
        axis_d.text(0.30, y, left, ha="center", va="center", fontsize=8.5, color=left_color)
        axis_d.annotate("", xy=(0.64, y), xytext=(0.46, y), arrowprops={"arrowstyle": "->", "color": "#777777", "lw": 1.2})
        axis_d.text(0.78, y, right, ha="center", va="center", fontsize=8.5, color="#263238")
    axis_d.text(
        0.0,
        -0.62,
        "✓ denotes the prespecified panel-specific gate in A–C. Green/red text in D indicates\n"
        "whether the unconstrained winner itself passed fidelity. AT2 is raw in every method.",
        fontsize=7.8,
        color="#444444",
        va="bottom",
    )

    figure.suptitle(
        "Actual correction on the control-bounded invariance–fidelity frontier",
        fontsize=14,
        fontweight="bold",
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    png = output / "figure5_actual_correction_frontier.png"
    pdf = output / "figure5_actual_correction_frontier.pdf"
    figure.savefig(png, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    result = {
        "analysis": "e5_comparator_figure",
        "e5_version": E5_VERSION,
        "frontier_summary": str(summary_path.resolve()),
        "frontier_summary_sha256": sha256(summary_path),
        "endpoint_summary_sha256": sha256(frontier / "endpoint_summary.csv"),
        "png": str(png.resolve()),
        "png_sha256": sha256(png),
        "pdf": str(pdf.resolve()),
        "pdf_sha256": sha256(pdf),
        "figure_gate_pass": bool(png.stat().st_size > 100_000 and pdf.stat().st_size > 10_000),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["figure_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
