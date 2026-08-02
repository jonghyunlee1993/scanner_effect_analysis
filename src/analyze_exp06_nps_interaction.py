"""Descriptive scanner/slide decomposition for the Exp-06 NPS pilot."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_exp05_spectral_pilot import save_figure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="outputs/exp06_raw_nps_pilot_5slides_final"
    )
    parser.add_argument(
        "--annotation",
        default=(
            "/mnt/isilon/oldridge_lab/batch_effects/metadata/"
            "pan_normal_annotation.csv"
        ),
    )
    return parser.parse_args()


def annotate_heatmap(axis, values, labels, text_color="black"):
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                labels[row, column],
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )


def main():
    args = parse_args()
    output_dir = Path(args.input)
    with (output_dir / "summary.json").open() as handle:
        summary = json.load(handle)
    scanners = summary["scanners"]
    slides = summary["slides"]

    annotation = pd.read_csv(args.annotation).rename(
        columns={"AnonSlideID": "slide_id", "Tissue Type": "tissue_type"}
    )
    annotation["slide_id"] = annotation["slide_id"].astype(str)
    tissue_map = annotation.set_index("slide_id")["tissue_type"].to_dict()
    slide_labels = [
        f"{slide_id}\n{textwrap.fill(tissue_map.get(slide_id, 'unknown'), 15)}"
        for slide_id in slides
    ]

    bands_path = output_dir / "operational_nps_band_summary.csv"
    bands = pd.read_csv(bands_path).drop(columns="tissue_type", errors="ignore")
    bands = bands.merge(
        annotation[["slide_id", "tissue_type"]], on="slide_id", how="left"
    )
    bands.to_csv(bands_path, index=False)
    high = (
        bands.pivot(index="scanner", columns="slide_id", values="high_band_nps")
        .reindex(index=scanners, columns=slides)
        .to_numpy()
    )
    log_high = np.log10(high)
    grand = log_high.mean()
    scanner_mean = log_high.mean(axis=1, keepdims=True)
    slide_mean = log_high.mean(axis=0, keepdims=True)
    interaction = log_high - scanner_mean - slide_mean + grand

    replicate = pd.read_csv(output_dir / "raw_glass_nps_replicates.csv")
    replicate = replicate[
        (replicate["detrend"] == "plane")
        & (replicate["frequency_cyc_per_um"] >= 0.60)
        & (replicate["frequency_cyc_per_um"] < 0.90)
    ]
    replicate = (
        replicate.groupby(["scanner", "slide_id", "replicate"])[
            "nps_od2_um2"
        ]
        .apply(
            lambda values: float(
                np.log10(
                    np.exp(
                        np.mean(np.log(np.maximum(values.to_numpy(), 1e-30)))
                    )
                )
            )
        )
        .rename("log10_high_band_nps")
        .reset_index()
    )
    replicate_ids = sorted(replicate["replicate"].unique())
    tensor = (
        replicate.pivot_table(
            index="scanner",
            columns=["slide_id", "replicate"],
            values="log10_high_band_nps",
        )
        .reindex(
            index=scanners,
            columns=pd.MultiIndex.from_product([slides, replicate_ids]),
        )
        .to_numpy()
        .reshape(len(scanners), len(slides), len(replicate_ids))
    )
    tensor_grand = tensor.mean()
    tensor_scanner = tensor.mean(axis=(1, 2), keepdims=True)
    tensor_slide = tensor.mean(axis=(0, 2), keepdims=True)
    tensor_cell = tensor.mean(axis=2, keepdims=True)
    sums_of_squares = {
        "scanner": len(slides)
        * len(replicate_ids)
        * float(np.square(tensor_scanner - tensor_grand).sum()),
        "slide": len(scanners)
        * len(replicate_ids)
        * float(np.square(tensor_slide - tensor_grand).sum()),
        "scanner_x_slide": len(replicate_ids)
        * float(
            np.square(
                tensor_cell
                - tensor_scanner
                - tensor_slide
                + tensor_grand
            ).sum()
        ),
        "within_cell_sampling": float(np.square(tensor - tensor_cell).sum()),
    }
    total = sum(sums_of_squares.values())
    variance = pd.DataFrame(
        {
            "component": list(sums_of_squares),
            "sum_of_squares": list(sums_of_squares.values()),
            "fraction": [value / total for value in sums_of_squares.values()],
        }
    )
    variance.to_csv(
        output_dir / "operational_nps_variance_components.csv", index=False
    )
    interaction_rows = []
    for row, scanner in enumerate(scanners):
        for column, slide_id in enumerate(slides):
            interaction_rows.append(
                {
                    "scanner": scanner,
                    "slide_id": slide_id,
                    "tissue_type": tissue_map.get(slide_id),
                    "high_band_nps": high[row, column],
                    "interaction_log10": interaction[row, column],
                    "interaction_fold": 10 ** interaction[row, column],
                }
            )
    pd.DataFrame(interaction_rows).to_csv(
        output_dir / "operational_nps_interaction_residuals.csv", index=False
    )

    rejection = pd.read_csv(output_dir / "glass_rejection_summary.csv")
    rejection["acceptance_rate"] = rejection["accepted"] / rejection["attempted"]
    acceptance = (
        rejection.pivot(index="scanner", columns="slide_id", values="acceptance_rate")
        .reindex(index=scanners, columns=slides)
        .to_numpy()
    )

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    axis = axes[0, 0]
    image = axis.imshow(log_high, cmap="viridis", aspect="auto")
    labels = np.vectorize(lambda value: f"{value:.1e}")(high)
    annotate_heatmap(axis, log_high, labels, text_color="white")
    axis.set_title("A  High-band operational NPS")
    axis.set_yticks(range(len(scanners)), [value.upper() for value in scanners])
    axis.set_xticks(range(len(slides)), slide_labels, rotation=35, ha="right")
    figure.colorbar(image, ax=axis, label="log₁₀ NPS (OD²·µm²)")

    axis = axes[0, 1]
    limit = float(np.abs(interaction).max())
    image = axis.imshow(
        interaction,
        cmap="coolwarm",
        aspect="auto",
        vmin=-limit,
        vmax=limit,
    )
    labels = np.vectorize(lambda value: f"{10 ** value:.2f}×")(interaction)
    annotate_heatmap(axis, interaction, labels)
    axis.set_title("B  Scanner × slide residual after additive effects")
    axis.set_yticks(range(len(scanners)), [value.upper() for value in scanners])
    axis.set_xticks(range(len(slides)), slide_labels, rotation=35, ha="right")
    figure.colorbar(image, ax=axis, label="Interaction residual (log₁₀)")

    axis = axes[1, 0]
    labels = ["Scanner", "Slide", "Scanner × slide", "20-patch sampling"]
    colours = ["#3B82F6", "#94A3B8", "#F59E0B", "#CBD5E1"]
    percentages = 100 * variance["fraction"].to_numpy()
    bars = axis.bar(labels, percentages, color=colours)
    axis.bar_label(bars, labels=[f"{value:.1f}%" for value in percentages])
    axis.set_ylim(0, 100)
    axis.set_ylabel("Fraction of log-NPS sum of squares")
    axis.set_title("C  Descriptive variance decomposition")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.2)

    axis = axes[1, 1]
    image = axis.imshow(acceptance, cmap="YlOrRd_r", aspect="auto", vmin=0, vmax=1)
    labels = np.vectorize(lambda value: f"{100 * value:.1f}%")(acceptance)
    annotate_heatmap(axis, acceptance, labels)
    axis.set_title("D  Strict glass-QC acceptance rate")
    axis.set_yticks(range(len(scanners)), [value.upper() for value in scanners])
    axis.set_xticks(range(len(slides)), slide_labels, rotation=35, ha="right")
    figure.colorbar(image, ax=axis, label="Accepted / attempted")

    figure.suptitle(
        "Operational background NPS: scanner main effect with localized slide interactions",
        fontsize=15,
    )
    figure.text(
        0.5,
        0.005,
        "Pilot descriptive decomposition only; five slides have five distinct tissue labels, so tissue and slide are confounded.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.97))
    figure_paths = save_figure(
        figure, output_dir, "figure5_nps_scanner_slide_decomposition"
    )

    summary["figures"] = list(dict.fromkeys(summary["figures"] + figure_paths))
    summary["tables"]["variance_components"] = str(
        output_dir / "operational_nps_variance_components.csv"
    )
    summary["tables"]["interaction_residuals"] = str(
        output_dir / "operational_nps_interaction_residuals.csv"
    )
    summary["descriptive_log_high_band_variance_fractions"] = dict(
        zip(variance["component"], variance["fraction"])
    )
    summary["tissue_slide_confounded_in_five_slide_pilot"] = True
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(variance.to_string(index=False), flush=True)
    print(f"[exp06-interaction] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
