"""Build Main Figure 4 from the audited frozen E4 endpoint table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fetch_e0_pfm_checkpoints import sha256


MODEL_TITLES = {
    "resnet50": "ResNet50",
    "uni_v1": "UNI v1",
    "conch_v1": "CONCH v1",
    "virchow2": "Virchow2",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/e4_control_frontier")
    parser.add_argument("--output", default="outputs/e4_control_figure")
    return parser.parse_args()


def condition_style(condition: str):
    if condition == "raw":
        return "Raw", "#4d4d4d"
    if condition.startswith("hf_retention"):
        retention = condition.rsplit("_", 1)[-1].replace("p", ".")
        colors = {
            "0.75": "#9ecae1",
            "0.50": "#6baed6",
            "0.25": "#3182bd",
            "0.00": "#08519c",
        }
        return f"HF×{retention}", colors[retention]
    if condition.startswith("hf_boost"):
        gain = condition.rsplit("_", 1)[-1].replace("p", ".")
        colors = {"1.25": "#fdae6b", "1.50": "#e6550d", "2.00": "#a63603"}
        return f"HF×{gain}", colors[gain]
    if condition.startswith("registered_loo"):
        return "Paired HF 25%", "#31a354"
    if condition.startswith("global_train"):
        return "Global HF mean", "#756bb1"
    raise ValueError(f"unknown E4 condition: {condition}")


def plot_panel(axis, frame: pd.DataFrame, model_id: str):
    selected = frame[frame["encoder_id"] == model_id].copy()
    if len(selected) != 10:
        raise ValueError(f"{model_id}: expected 10 endpoint rows, got {len(selected)}")
    raw_radius = float(selected.loc[selected["condition"] == "raw", "scanner_centroid_rms"].iloc[0])
    for row in selected.itertuples(index=False):
        label, color = condition_style(row.condition)
        x = float(row.delta_content_margin)
        y = float(row.relative_radius_reduction)
        if not np.isfinite(y):
            y = float(row.raw_minus_condition) / raw_radius
        xerr = np.asarray(
            [
                [max(0.0, x - float(row.delta_content_ci_lower))],
                [max(0.0, float(row.delta_content_ci_upper) - x)],
            ]
        )
        y_lower = float(row.difference_ci_lower) / raw_radius
        y_upper = float(row.difference_ci_upper) / raw_radius
        yerr = np.asarray([[max(0.0, y - y_lower)], [max(0.0, y_upper - y)]])
        safe = bool(row.safe_for_pfm)
        marker = "o" if safe else "X"
        face = color if safe else "white"
        axis.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            markersize=7,
            markerfacecolor=face,
            markeredgecolor=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=2,
            alpha=0.95,
            label=label,
        )
    axis.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    axis.axvline(-0.02, color="#b2182b", linewidth=0.9, linestyle=":")
    axis.set_title(MODEL_TITLES[model_id], fontsize=11, fontweight="bold")
    axis.set_xlabel("Δ matched-content margin\n(non-inferiority boundary −0.02)")
    axis.set_ylabel("Relative scanner-radius reduction")
    axis.grid(alpha=0.18, linewidth=0.6)


def main():
    args = parse_args()
    input_root = Path(args.input)
    summary_path = input_root / "summary.json"
    endpoint_path = input_root / "endpoint_summary.csv"
    summary = json.loads(summary_path.read_text())
    if summary.get("analysis_complete") is not True:
        raise RuntimeError("frozen E4 endpoint analysis is incomplete")
    endpoints = pd.read_csv(endpoint_path)
    if len(endpoints) != 40:
        raise ValueError(f"expected 40 E4 endpoint rows, got {len(endpoints)}")

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.2), constrained_layout=False)
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.90,
        bottom=0.17,
        hspace=0.34,
        wspace=0.22,
    )
    for axis, model_id in zip(axes.flat, MODEL_TITLES):
        plot_panel(axis, endpoints, model_id)
    figure.suptitle(
        "E4 — Control-bounded scanner invariance and representation fidelity",
        fontsize=14,
        fontweight="bold",
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    figure.text(
        0.5,
        0.018,
        "Filled circles pass pooled content non-inferiority and every-scanner collapse gates; "
        "X markers fail at least one fidelity gate. Error bars are 95% physical-slide bootstrap CIs.",
        ha="center",
        fontsize=8,
    )

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    png = output_root / "figure4_control_bounded_frontier.png"
    pdf = output_root / "figure4_control_bounded_frontier.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    figure_summary = {
        "analysis": "e4_control_figure",
        "endpoint_summary": str(endpoint_path.resolve()),
        "endpoint_summary_sha256": sha256(endpoint_path),
        "analysis_summary_sha256": sha256(summary_path),
        "png": str(png.resolve()),
        "png_sha256": sha256(png),
        "pdf": str(pdf.resolve()),
        "pdf_sha256": sha256(pdf),
        "figure_gate_pass": png.stat().st_size > 100_000 and pdf.stat().st_size > 10_000,
    }
    (output_root / "summary.json").write_text(json.dumps(figure_summary, indent=2) + "\n")
    print(json.dumps(figure_summary, indent=2))
    if not figure_summary["figure_gate_pass"]:
        raise RuntimeError("E4 Figure 4 output gate failed")


if __name__ == "__main__":
    main()
