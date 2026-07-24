"""Render Exp-02 pairwise-transform and matched-blur figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from prenorm.exp01.frequency import FixedLaplacianPyramid
from prenorm.exp02.pairwise import apply_affine


SCANNER_COLORS = {
    "at2": "#3B82F6",
    "gt450": "#F59E0B",
    "versa": "#10B981",
    "akoya": "#8B5CF6",
    "s60": "#EF4444",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/exp02_visualization")
    parser.add_argument(
        "--output", default="outputs/exp02_visualization/figures"
    )
    return parser.parse_args(argv)


def rgb(image):
    return np.clip((np.moveaxis(image, 0, -1) + 1.0) * 0.5, 0, 1)


def lattice_label(lattice):
    return "internal" if lattice == "internal_v3" else "S60 lattice"


def short_direction(direction):
    lattice, pair = direction.split("__", maxsplit=1)
    source, target = pair.split("_to_")
    return f"{source.upper()}→{target.upper()}\n{lattice_label(lattice)}"


def save_figure(figure, output_dir: Path, stem: str):
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [str(png), str(pdf)]


def one_direction_per_reference(directions):
    """Select one qualitative source for every target scanner and lattice."""
    selected = []
    seen = set()
    for direction in directions:
        lattice, pair = direction.split("__", maxsplit=1)
        _, target = pair.split("_to_")
        key = (lattice, target)
        if key not in seen:
            selected.append(direction)
            seen.add(key)
    return selected


def render_pairwise_images(
    pair_frame,
    summary,
    exemplars,
    output_dir,
    *,
    directions,
    stem,
    title,
):
    pyramid = FixedLaplacianPyramid(
        int(summary.get("pyramid_levels", 4))
    )
    rendered = {}
    for direction in directions:
        prefix = summary["exemplars"][direction]["array_prefix"]
        source = torch.from_numpy(exemplars[f"{prefix}__source"])[None]
        target = torch.from_numpy(exemplars[f"{prefix}__target"])[None]
        clipped = torch.from_numpy(exemplars[f"{prefix}__clipped"])[None]
        lattice_id, short_pair = direction.split("__", maxsplit=1)
        affine = torch.tensor(
            summary["pairwise_affines"][lattice_id][short_pair]
        )
        source_low = pyramid.coarsest(source)
        target_low = pyramid.coarsest(target)
        affine_low = apply_affine(source_low, affine)
        clipped_low = pyramid.coarsest(clipped)

        def upsample(low):
            return F.interpolate(
                low,
                size=source.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        rendered[direction] = {
            "panels": [
                rgb(source[0].numpy()),
                rgb(upsample(source_low)[0].numpy()),
                rgb(upsample(affine_low)[0].numpy()),
                rgb(upsample(target_low)[0].numpy()),
                rgb(clipped[0].numpy()),
                rgb(target[0].numpy()),
            ],
            "low_mae": (
                float((source_low - target_low).abs().mean()),
                float((affine_low - target_low).abs().mean()),
                float((clipped_low - target_low).abs().mean()),
            ),
        }

    figure, axes = plt.subplots(
        len(directions),
        6,
        figsize=(15.5, 2.15 * len(directions)),
        squeeze=False,
    )
    titles = (
        "Source image",
        "Source LF",
        "Affine-transformed LF",
        "Paired target LF",
        "Clipped composite",
        "Paired target",
    )
    for column, column_title in enumerate(titles):
        axes[0, column].set_title(
            column_title, fontsize=10, weight="bold"
        )

    for row, direction in enumerate(directions):
        result = rendered[direction]
        for column, panel in enumerate(result["panels"]):
            axis = axes[row, column]
            axis.imshow(panel)
            axis.set_xticks([])
            axis.set_yticks([])
        raw_mae, affine_mae, clipped_mae = result["low_mae"]
        axes[row, 0].set_ylabel(
            f"{short_direction(direction)}\n"
            f"LF MAE raw/requested/clipped\n"
            f"{raw_mae:.3f}/{affine_mae:.3f}/{clipped_mae:.3f}",
            fontsize=7.7,
            rotation=0,
            ha="right",
        )

    figure.suptitle(title, fontsize=13, weight="bold", y=1.002)
    figure.tight_layout()
    return save_figure(figure, output_dir, stem)


def render_uni_response(pair_frame, output_dir):
    directions = pair_frame["direction"].drop_duplicates().tolist()
    labels = [short_direction(direction) for direction in directions]
    x = np.arange(len(directions))
    grouped = pair_frame.groupby("direction", sort=False).mean(
        numeric_only=True
    )

    figure, axes = plt.subplots(1, 3, figsize=(20, 5.2))
    width = 0.25
    axes[0].bar(
        x - width,
        grouped.loc[directions, "low_mae_raw"],
        width,
        label="Raw source",
        color="#94A3B8",
    )
    axes[0].bar(
        x,
        grouped.loc[directions, "low_mae_affine_requested"],
        width,
        label="Requested affine LF",
        color="#10B981",
    )
    axes[0].bar(
        x + width,
        grouped.loc[directions, "low_mae_clipped"],
        width,
        label="Operational clipped image",
        color="#2563EB",
    )
    axes[0].set_title("A. Low-frequency correction")
    axes[0].set_ylabel("Paired LF MAE ↓")
    axes[0].legend(frameon=False, fontsize=8)

    source_clipped = (
        1.0
        - grouped.loc[
            directions, "uni_source_to_clipped_cos"
        ].to_numpy()
    )
    source_target = (
        1.0
        - grouped.loc[directions, "uni_source_to_target_cos"].to_numpy()
    )
    axes[1].bar(
        x - width / 2,
        np.clip(source_clipped, 1e-6, None),
        width,
        label="Source ↔ LF-corrected",
        color="#10B981",
    )
    axes[1].bar(
        x + width / 2,
        np.clip(source_target, 1e-6, None),
        width,
        label="Source ↔ paired target",
        color="#EF4444",
    )
    axes[1].set_yscale("log")
    axes[1].set_title("B. UNI embedding displacement")
    axes[1].set_ylabel("Cosine distance ↓ (log scale)")
    axes[1].legend(frameon=False, fontsize=8)

    colors = [
        SCANNER_COLORS[
            direction.split("__", 1)[1].split("_to_", 1)[1]
        ]
        for direction in directions
    ]
    axes[2].bar(
        x,
        grouped.loc[directions, "uni_target_cos_delta"],
        color=colors,
    )
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("C. Change in UNI similarity to target")
    axes[2].set_ylabel("Δ cosine (LF-corrected − raw)")

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=50, ha="right", fontsize=7.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)

    figure.suptitle(
        "Pairwise low-frequency correction in image and UNI space",
        fontsize=14,
        weight="bold",
    )
    figure.tight_layout()
    return save_figure(
        figure, output_dir, "figure2_uni_low_frequency_response"
    )


def paired_blur_panel(transformed, target):
    top = rgb(transformed)
    bottom = rgb(target)
    divider = np.ones((4, top.shape[1], 3), dtype=top.dtype)
    return np.concatenate((top, divider, bottom), axis=0)


def render_blur_frontier(
    blur_frame,
    summary,
    exemplars,
    output_dir,
):
    all_directions = blur_frame["direction"].drop_duplicates().tolist()
    examples = one_direction_per_reference(all_directions)
    retention = sorted(
        blur_frame["detail_retention"].unique(), reverse=True
    )

    figure = plt.figure(figsize=(15.5, 2.0 * len(examples) + 5.3))
    grid = GridSpec(
        len(examples) + 2,
        len(retention) + 1,
        figure=figure,
        height_ratios=[1] * len(examples) + [0.12, 2.25],
    )

    for row, direction in enumerate(examples):
        prefix = summary["exemplars"][direction]["array_prefix"]
        for column, value in enumerate(retention):
            suffix = f"retention_{value:.2f}".replace(".", "p")
            axis = figure.add_subplot(grid[row, column])
            axis.imshow(paired_blur_panel(
                exemplars[f"{prefix}__transformed_{suffix}"],
                exemplars[f"{prefix}__target_{suffix}"],
            ))
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(
                    f"Detail {value:.2f}\nblur {1-value:.2f}",
                    fontsize=9,
                )
            if column == 0:
                axis.set_ylabel(
                    f"{short_direction(direction)}\n"
                    "LF affine source / matched target",
                    fontsize=7.4,
                    rotation=0,
                    ha="right",
                )
        sharp_axis = figure.add_subplot(grid[row, len(retention)])
        sharp_axis.imshow(paired_blur_panel(
            exemplars[f"{prefix}__source"],
            exemplars[f"{prefix}__target"],
        ))
        sharp_axis.set_xticks([])
        sharp_axis.set_yticks([])
        if row == 0:
            sharp_axis.set_title("Sharp source /\ntarget", fontsize=9)

    matched_axis = figure.add_subplot(
        grid[len(examples) + 1, :3]
    )
    delta_axis = figure.add_subplot(
        grid[len(examples) + 1, 3:]
    )

    group_columns = ["lattice_id", "target_scanner"]
    for (lattice, target), part in blur_frame.groupby(
        group_columns, sort=False
    ):
        curve = part.groupby("blur_severity", sort=True).mean(
            numeric_only=True
        )
        color = SCANNER_COLORS[target]
        label = f"{target.upper()} · {lattice_label(lattice)}"
        matched_axis.plot(
            curve.index,
            curve["uni_transformed_to_matched_target_cos"],
            color=color,
            marker="o",
            linewidth=1.7,
            label=label,
        )
        matched_axis.plot(
            curve.index,
            curve["uni_raw_to_matched_target_cos"],
            color=color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
        )
        delta_axis.plot(
            curve.index,
            curve["uni_matched_alignment_delta"],
            color=color,
            marker="o",
            linewidth=1.7,
            label=label,
        )

    overall = blur_frame.groupby("blur_severity", sort=True).mean(
        numeric_only=True
    )
    matched_axis.plot(
        overall.index,
        overall["uni_transformed_to_matched_target_cos"],
        color="black",
        linewidth=3.0,
        label="All references · LF affine",
    )
    matched_axis.plot(
        overall.index,
        overall["uni_raw_to_matched_target_cos"],
        color="black",
        linestyle=":",
        linewidth=2.4,
        label="All references · raw blur",
    )
    delta_axis.plot(
        overall.index,
        overall["uni_matched_alignment_delta"],
        color="black",
        linewidth=3.0,
        label="All references",
    )
    delta_axis.axhline(0, color="black", linewidth=0.8)

    matched_axis.set_title(
        "UNI similarity to target blurred at the same detail retention"
    )
    delta_axis.set_title(
        "Matched-blur alignment gain from low-frequency affine"
    )
    matched_axis.set_ylabel("Cosine similarity")
    delta_axis.set_ylabel("Δ cosine (LF affine − raw blur)")
    for axis in (matched_axis, delta_axis):
        axis.set_xlabel("Controlled blur severity (1 − detail retention)")
        axis.set_xlim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    reference_legend = matched_axis.legend(
        frameon=False, fontsize=7, ncol=2, loc="lower right"
    )
    delta_axis.legend(frameon=False, fontsize=7, ncol=2)

    style_handles = [
        Line2D([0], [0], color="#475569", linewidth=2, label="LF affine"),
        Line2D(
            [0], [0], color="#475569", linewidth=2,
            linestyle=":", label="Raw source blur",
        ),
    ]
    matched_axis.legend(
        handles=style_handles,
        loc="lower left",
        frameon=False,
        fontsize=7,
    )
    matched_axis.add_artist(reference_legend)

    figure.suptitle(
        "Rotating scanner references with matched high-frequency damage",
        fontsize=14,
        weight="bold",
    )
    figure.tight_layout()
    return save_figure(
        figure, output_dir, "figure3_blur_embedding_frontier"
    )


def main(argv=None):
    args = parse_args(argv)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_frame = pd.read_csv(input_dir / "pairwise_metrics.csv")
    blur_frame = pd.read_csv(input_dir / "blur_frontier_metrics.csv")
    summary = json.loads((input_dir / "summary.json").read_text())
    exemplars = np.load(input_dir / "exemplars.npz")
    all_directions = pair_frame["direction"].drop_duplicates().tolist()
    reference_examples = one_direction_per_reference(all_directions)

    figures = {
        "low_frequency_rotating_reference": render_pairwise_images(
            pair_frame,
            summary,
            exemplars,
            output_dir,
            directions=reference_examples,
            stem="figure1_pairwise_image_transform",
            title="Pairwise low-frequency correction with rotating references",
        ),
        "all_pairwise_feasibility": render_pairwise_images(
            pair_frame,
            summary,
            exemplars,
            output_dir,
            directions=all_directions,
            stem="figure_s1_all_pairwise_feasibility",
            title="All directed scanner pairs with operational clipping",
        ),
        "uni_low_frequency_response": render_uni_response(
            pair_frame, output_dir
        ),
        "matched_blur_embedding_frontier": render_blur_frontier(
            blur_frame, summary, exemplars, output_dir
        ),
    }
    manifest = {
        "experiment": "Exp-02",
        "analysis_contract_version": 3,
        "source_summary": str(input_dir / "summary.json"),
        "figures": figures,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"[exp02-render] wrote {len(figures)} figure sets -> {output_dir}")


if __name__ == "__main__":
    main()
