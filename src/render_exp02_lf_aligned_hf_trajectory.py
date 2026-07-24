"""Compare blur/sharpen UNI motion before and after LF affine alignment."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from prenorm.exp02.trajectory import (
    chord_direction,
    pairwise_direction_cosine,
    spherical_mean,
)
from render_exp02_hf_trajectory import (
    FAMILY_CMAPS,
    FAMILY_LABELS,
    SCANNER_COLORS,
    SCANNER_MARKERS,
    draw_gradient_path,
    padded_limits,
    save_figure,
    save_png,
)


REFERENCE_ORDER = ("unaligned", "at2", "gt450", "versa", "akoya")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aligned-input",
        default="outputs/exp02_lf_aligned_hf_trajectory",
    )
    parser.add_argument(
        "--unaligned-input",
        default="outputs/exp02_hf_trajectory",
    )
    parser.add_argument(
        "--output",
        default="outputs/exp02_lf_aligned_hf_trajectory/figures",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args(argv)


def load_projection_data(aligned_dir, unaligned_dir):
    aligned_frame = pd.read_csv(
        aligned_dir / "trajectory_metadata.csv"
    )
    aligned_arrays = np.load(aligned_dir / "trajectory_arrays.npz")
    aligned_embeddings = aligned_arrays["embeddings"]

    unaligned_frame = pd.read_csv(
        unaligned_dir / "trajectory_metadata.csv"
    )
    unaligned_arrays = np.load(unaligned_dir / "trajectory_arrays.npz")
    keep = unaligned_frame["family"].isin(("raw", "blur", "sharpen"))
    unaligned_frame = unaligned_frame.loc[keep].copy()
    unaligned_embeddings = unaligned_arrays["embeddings"][
        unaligned_frame["row_id"].to_numpy(dtype=int)
    ]
    unaligned_frame["lf_reference_scanner"] = "unaligned"
    unaligned_frame["source_scanner"] = unaligned_frame["scanner"]
    unaligned_frame["source_row_id"] = unaligned_frame["row_id"]

    aligned_frame = aligned_frame.copy()
    aligned_frame["source_row_id"] = aligned_frame["row_id"]
    projection_frame = pd.concat(
        (unaligned_frame, aligned_frame),
        ignore_index=True,
        sort=False,
    )
    embeddings = np.concatenate(
        (unaligned_embeddings, aligned_embeddings), axis=0
    )
    projection_frame["projection_id"] = np.arange(
        len(projection_frame)
    )
    return (
        projection_frame,
        embeddings,
        aligned_frame,
        aligned_arrays["images"],
    )


def fit_projection(frame, embeddings, seed):
    n_components = min(30, len(embeddings) - 1, embeddings.shape[1])
    reduced = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=seed,
    ).fit_transform(embeddings)
    import umap

    coordinates = umap.UMAP(
        n_neighbors=min(15, len(embeddings) - 1),
        min_dist=0.18,
        metric="euclidean",
        random_state=seed,
        n_jobs=1,
    ).fit_transform(reduced)
    result = frame.copy()
    result["umap_x"] = coordinates[:, 0]
    result["umap_y"] = coordinates[:, 1]
    return result


def baseline_family(reference):
    return "raw" if reference == "unaligned" else "aligned_raw"


def trajectory_path(frame, location, reference, scanner, family):
    baseline = frame[
        (frame["location_index"] == location)
        & (frame["lf_reference_scanner"] == reference)
        & (frame["source_scanner"] == scanner)
        & (frame["family"] == baseline_family(reference))
    ]
    selected = frame[
        (frame["location_index"] == location)
        & (frame["lf_reference_scanner"] == reference)
        & (frame["source_scanner"] == scanner)
        & (frame["family"] == family)
    ].sort_values("parameter")
    if len(baseline) != 1 or selected.empty:
        raise ValueError(
            f"incomplete path for {location}/{reference}/{scanner}/{family}"
        )
    return pd.concat((baseline, selected), ignore_index=True)


def render_aligned_family_umap(
    frame,
    summary,
    output_dir,
    family,
    stem,
):
    locations = sorted(frame["location_index"].unique())
    scanners = summary["scanners"]
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap(FAMILY_CMAPS[family])
    figure, axes = plt.subplots(
        len(locations),
        len(REFERENCE_ORDER),
        figsize=(18.5, 4.0 * len(locations)),
        squeeze=False,
    )
    for row, location in enumerate(locations):
        for column, reference in enumerate(REFERENCE_ORDER):
            axis = axes[row, column]
            local = frame[
                (frame["location_index"] == location)
                & (frame["lf_reference_scanner"] == reference)
                & frame["family"].isin((
                    baseline_family(reference), family
                ))
            ]
            xlim, ylim = padded_limits(local, pad=0.14)
            for scanner in scanners:
                draw_gradient_path(
                    axis,
                    trajectory_path(
                        frame,
                        location,
                        reference,
                        scanner,
                        family,
                    ),
                    cmap,
                    SCANNER_MARKERS[scanner],
                    norm,
                )
            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(alpha=0.13)
            axis.spines[["top", "right"]].set_visible(False)
            if row == 0:
                axis.set_title(
                    "No LF affine"
                    if reference == "unaligned"
                    else f"LF aligned → {reference.upper()}",
                    fontsize=10,
                    weight="bold",
                )
            if column == 0:
                selected = summary["locations"][location]
                axis.set_ylabel(
                    f"Pair {location + 1}\n"
                    f"{selected['slide_id']} / tuple "
                    f"{selected['tuple_id']}\n"
                    f"min q={selected['min_q_reg']:.3f}",
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=8.5,
                )

    handles = [
        Line2D(
            [0], [0],
            color="none",
            marker=SCANNER_MARKERS[scanner],
            markerfacecolor="white",
            markeredgecolor="#111827",
            label=scanner.upper(),
        )
        for scanner in scanners
    ]
    axes[0, 0].legend(
        handles=handles,
        frameon=False,
        fontsize=7.5,
        ncol=2,
        loc="best",
        title="Source scanner",
        title_fontsize=7.5,
    )
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        fraction=0.012,
        pad=0.012,
    )
    colorbar.set_label(f"{FAMILY_LABELS[family]} strength")
    figure.suptitle(
        f"{FAMILY_LABELS[family]} after rotating pairwise LF affine alignment",
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "One fixed UMAP for all panels · open marker = condition baseline · "
        "darker = stronger HF intervention · arrows show direction",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.94,
        bottom=0.055,
        top=0.93,
        wspace=0.14,
        hspace=0.18,
    )
    return save_figure(figure, output_dir, stem)


def render_pair_family_umap(
    frame,
    summary,
    output_dir,
    family,
    location,
):
    scanners = summary["scanners"]
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap(FAMILY_CMAPS[family])
    figure, axes = plt.subplots(
        1, len(REFERENCE_ORDER), figsize=(18.0, 4.1), squeeze=False
    )
    axes = axes[0]
    for axis, reference in zip(axes, REFERENCE_ORDER):
        local = frame[
            (frame["location_index"] == location)
            & (frame["lf_reference_scanner"] == reference)
            & frame["family"].isin((
                baseline_family(reference), family
            ))
        ]
        xlim, ylim = padded_limits(local, pad=0.14)
        for scanner in scanners:
            draw_gradient_path(
                axis,
                trajectory_path(
                    frame,
                    location,
                    reference,
                    scanner,
                    family,
                ),
                cmap,
                SCANNER_MARKERS[scanner],
                norm,
            )
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(alpha=0.13)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(
            "No LF affine"
            if reference == "unaligned"
            else f"LF aligned → {reference.upper()}",
            fontsize=10,
            weight="bold",
        )
    handles = [
        Line2D(
            [0], [0],
            color="none",
            marker=SCANNER_MARKERS[scanner],
            markerfacecolor="white",
            markeredgecolor="#111827",
            label=scanner.upper(),
        )
        for scanner in scanners
    ]
    axes[0].legend(
        handles=handles,
        frameon=False,
        fontsize=7.5,
        ncol=2,
        loc="best",
    )
    figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        orientation="horizontal",
        fraction=0.055,
        pad=0.13,
        shrink=0.32,
        aspect=35,
        label=f"{FAMILY_LABELS[family]} strength",
    )
    selected = summary["locations"][location]
    figure.suptitle(
        f"{FAMILY_LABELS[family]} · Pair {location + 1} · "
        f"{selected['slide_id']} / tuple {selected['tuple_id']}",
        fontsize=13,
        weight="bold",
    )
    figure.subplots_adjust(
        left=0.025, right=0.99, bottom=0.24, top=0.82, wspace=0.14
    )
    return save_png(
        figure,
        output_dir / "umap_pairs" / family,
        f"pair_{location + 1:02d}_{family}_lf_reference_umap",
    )


def direction_cosines(frame, embeddings, scanners):
    lookup = {
        int(row.projection_id): embeddings[int(row.projection_id)]
        for row in frame.itertuples()
    }
    rows = []
    for location in sorted(frame["location_index"].unique()):
        for reference in REFERENCE_ORDER:
            for family in ("blur", "sharpen"):
                parameters = sorted(frame.loc[
                    (frame["lf_reference_scanner"] == reference)
                    & (frame["family"] == family),
                    "parameter",
                ].unique())
                for parameter in parameters:
                    directions = []
                    for scanner in scanners:
                        path = trajectory_path(
                            frame, location, reference, scanner, family
                        )
                        baseline = lookup[
                            int(path.iloc[0]["projection_id"])
                        ]
                        endpoint_row = path[
                            np.isclose(path["parameter"], parameter)
                        ].iloc[-1]
                        endpoint = lookup[
                            int(endpoint_row["projection_id"])
                        ]
                        directions.append(chord_direction(
                            endpoint[None], baseline[None]
                        )[0])
                    cosine = pairwise_direction_cosine(
                        np.stack(directions)
                    )
                    for left, right in combinations(
                        range(len(scanners)), 2
                    ):
                        rows.append({
                            "location_index": int(location),
                            "lf_reference_scanner": reference,
                            "family": family,
                            "parameter": float(parameter),
                            "scanner_a": scanners[left],
                            "scanner_b": scanners[right],
                            "direction_cosine": float(
                                cosine[left, right]
                            ),
                        })
    return pd.DataFrame(rows)


def direction_matrix(
    direction_frame,
    scanners,
    reference,
    family,
    parameter,
):
    selected = direction_frame[
        (direction_frame["lf_reference_scanner"] == reference)
        & (direction_frame["family"] == family)
        & np.isclose(direction_frame["parameter"], parameter)
    ].groupby(
        ["scanner_a", "scanner_b"], sort=False
    )["direction_cosine"].mean()
    matrix = np.eye(len(scanners))
    for left, scanner_a in enumerate(scanners):
        for right, scanner_b in enumerate(scanners):
            if left == right:
                continue
            key = (
                (scanner_a, scanner_b)
                if left < right
                else (scanner_b, scanner_a)
            )
            matrix[left, right] = selected.loc[key]
    return matrix


def akoya_agreement(matrix, scanners):
    index = scanners.index("akoya")
    return float(np.mean(np.delete(matrix[index], index)))


def render_sharpen_direction_cosines(
    direction_frame,
    scanners,
    output_dir,
):
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), squeeze=False)
    flat_axes = axes.flat
    image = None
    used_axes = []
    for axis, reference in zip(flat_axes, REFERENCE_ORDER):
        used_axes.append(axis)
        matrix = direction_matrix(
            direction_frame, scanners, reference, "sharpen", 1.0
        )
        image = axis.imshow(
            matrix,
            vmin=-1,
            vmax=1,
            cmap="RdBu_r",
            interpolation="nearest",
        )
        for row in range(len(scanners)):
            for column in range(len(scanners)):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > 0.55 else "#111827",
                )
        title = (
            "No LF affine"
            if reference == "unaligned"
            else f"LF aligned → {reference.upper()}"
        )
        scanner_agreement = {
            scanner: float(np.mean(np.delete(matrix[index], index)))
            for index, scanner in enumerate(scanners)
        }
        lowest = min(scanner_agreement, key=scanner_agreement.get)
        axis.set_title(
            f"{title}\nlowest: {lowest.upper()} "
            f"{scanner_agreement[lowest]:+.2f} · "
            f"Akoya {scanner_agreement['akoya']:+.2f}",
            fontsize=9.5,
            weight="bold",
        )
        axis.set_xticks(range(len(scanners)))
        axis.set_yticks(range(len(scanners)))
        axis.set_xticklabels(
            [scanner.upper() for scanner in scanners],
            rotation=40,
            ha="right",
            fontsize=8,
        )
        axis.set_yticklabels(
            [scanner.upper() for scanner in scanners],
            fontsize=8,
        )
        for tick, scanner in zip(axis.get_xticklabels(), scanners):
            if scanner == "akoya":
                tick.set_color(SCANNER_COLORS["akoya"])
                tick.set_weight("bold")
        for tick, scanner in zip(axis.get_yticklabels(), scanners):
            if scanner == "akoya":
                tick.set_color(SCANNER_COLORS["akoya"])
                tick.set_weight("bold")
    axes[1, 2].axis("off")
    figure.colorbar(
        image,
        ax=used_axes,
        fraction=0.025,
        pad=0.025,
        label="Mean cosine between sharpening displacement directions",
    )
    figure.suptitle(
        "Scanner agreement of Sharpen 1.0 directions after LF alignment",
        fontsize=14,
        weight="bold",
    )
    figure.subplots_adjust(
        left=0.055, right=0.90, bottom=0.08, top=0.88,
        wspace=0.30, hspace=0.38,
    )
    return save_figure(
        figure,
        output_dir,
        "figure6c_sharpen_direction_cosine",
    )


def scanner_agreements(matrix, scanners):
    return {
        scanner: float(np.mean(np.delete(matrix[index], index)))
        for index, scanner in enumerate(scanners)
    }


def render_sharpen_agreement_by_strength(
    direction_frame,
    scanners,
    output_dir,
):
    figure, axes = plt.subplots(
        1, len(REFERENCE_ORDER), figsize=(18.0, 4.2), squeeze=False
    )
    axes = axes[0]
    parameters = sorted(direction_frame.loc[
        direction_frame["family"] == "sharpen", "parameter"
    ].unique())
    for axis, reference in zip(axes, REFERENCE_ORDER):
        values = {scanner: [] for scanner in scanners}
        for parameter in parameters:
            matrix = direction_matrix(
                direction_frame,
                scanners,
                reference,
                "sharpen",
                parameter,
            )
            agreement = scanner_agreements(matrix, scanners)
            for scanner in scanners:
                values[scanner].append(agreement[scanner])
        for scanner in scanners:
            axis.plot(
                parameters,
                values[scanner],
                color=SCANNER_COLORS[scanner],
                marker=SCANNER_MARKERS[scanner],
                linewidth=1.8,
                markersize=5,
                label=scanner.upper(),
            )
        axis.axhline(0, color="#111827", linewidth=0.7)
        axis.set_ylim(-0.05, 0.85)
        axis.set_xticks(parameters)
        axis.set_xlabel("Sharpen strength")
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(
            "No LF affine"
            if reference == "unaligned"
            else f"LF aligned → {reference.upper()}",
            fontsize=10,
            weight="bold",
        )
    axes[0].set_ylabel(
        "Mean direction cosine to the other scanners ↑"
    )
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    figure.suptitle(
        "Scanner-specific sharpening directions persist across strengths",
        fontsize=14,
        weight="bold",
    )
    figure.subplots_adjust(
        left=0.055, right=0.99, bottom=0.15, top=0.82, wspace=0.22
    )
    return save_figure(
        figure,
        output_dir,
        "figure6d_sharpen_direction_by_strength",
    )


def scanner_radius_metrics(frame, embeddings):
    rows = []
    for location in sorted(frame["location_index"].unique()):
        for reference in REFERENCE_ORDER:
            baseline = frame[
                (frame["location_index"] == location)
                & (frame["lf_reference_scanner"] == reference)
                & (frame["family"] == baseline_family(reference))
            ]
            for family in ("blur", "sharpen"):
                states = [(0.0, baseline)]
                states.extend(
                    (
                        float(parameter),
                        frame[
                            (frame["location_index"] == location)
                            & (
                                frame["lf_reference_scanner"]
                                == reference
                            )
                            & (frame["family"] == family)
                            & np.isclose(
                                frame["parameter"], parameter
                            )
                        ],
                    )
                    for parameter in sorted(frame.loc[
                        (
                            frame["lf_reference_scanner"]
                            == reference
                        )
                        & (frame["family"] == family),
                        "parameter",
                    ].unique())
                )
                for parameter, selected in states:
                    vectors = embeddings[
                        selected["projection_id"].to_numpy(dtype=int)
                    ]
                    centroid = spherical_mean(vectors)
                    rows.append({
                        "location_index": int(location),
                        "lf_reference_scanner": reference,
                        "family": family,
                        "parameter": parameter,
                        "scanner_radius": float(np.mean(
                            1.0 - vectors @ centroid
                        )),
                    })
    return pd.DataFrame(rows)


def render_scanner_radius(radius_frame, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    reference_colors = {
        "unaligned": "#475569",
        **SCANNER_COLORS,
    }
    reference_labels = {
        "unaligned": "No LF affine",
        **{
            scanner: f"LF → {scanner.upper()}"
            for scanner in SCANNER_COLORS
        },
    }
    aggregate = radius_frame.groupby(
        ["lf_reference_scanner", "family", "parameter"],
        sort=False,
    )["scanner_radius"].mean().reset_index()
    for axis, family in zip(axes, ("blur", "sharpen")):
        for reference in REFERENCE_ORDER:
            selected = aggregate[
                (aggregate["lf_reference_scanner"] == reference)
                & (aggregate["family"] == family)
            ].sort_values("parameter")
            axis.plot(
                selected["parameter"],
                selected["scanner_radius"],
                color=reference_colors[reference],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=reference_labels[reference],
            )
        axis.set_title(
            f"{FAMILY_LABELS[family]}",
            fontsize=11,
            weight="bold",
        )
        axis.set_xlabel("Intervention strength")
        axis.set_ylabel("Within-patch scanner radius ↓")
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    figure.suptitle(
        "LF affine does not make the UNI response reference-independent",
        fontsize=14,
        weight="bold",
    )
    figure.tight_layout()
    paths = save_figure(
        figure,
        output_dir,
        "figure6e_scanner_radius_after_lf_alignment",
    )
    return paths, aggregate


def selected_aligned_image(
    frame,
    images,
    location,
    reference,
    scanner,
    family,
    parameter,
):
    selected = frame[
        (frame["location_index"] == location)
        & (frame["lf_reference_scanner"] == reference)
        & (frame["source_scanner"] == scanner)
        & (frame["family"] == family)
        & np.isclose(frame["parameter"], parameter)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one aligned image: "
            f"{location}/{reference}/{scanner}/{family}/{parameter}"
        )
    return images[int(selected.iloc[0]["source_row_id"])]


def render_pair_endpoint_images(
    aligned_frame,
    aligned_images,
    summary,
    output_dir,
    location,
):
    scanners = summary["scanners"]
    references = summary["lf_references"]
    states = (
        ("aligned_raw", 0.0, "Aligned"),
        ("blur", 1.0, "Blur 1"),
        ("sharpen", 1.0, "Sharp 1"),
    )
    figure, axes = plt.subplots(
        len(references),
        len(scanners) * len(states),
        figsize=(18.0, 6.5),
        squeeze=False,
    )
    for row, reference in enumerate(references):
        for scanner_index, scanner in enumerate(scanners):
            for state_index, (family, parameter, label) in enumerate(states):
                column = scanner_index * len(states) + state_index
                axis = axes[row, column]
                axis.imshow(selected_aligned_image(
                    aligned_frame,
                    aligned_images,
                    location,
                    reference,
                    scanner,
                    family,
                    parameter,
                ))
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(
                        f"{scanner.upper()}\n{label}",
                        fontsize=7.5,
                        color=SCANNER_COLORS[scanner],
                        weight="bold",
                    )
                if column == 0:
                    axis.set_ylabel(
                        f"LF → {reference.upper()}",
                        rotation=0,
                        ha="right",
                        va="center",
                        fontsize=8.5,
                        weight="bold",
                    )
                if state_index == len(states) - 1:
                    axis.spines["right"].set_linewidth(1.5)
                    axis.spines["right"].set_color("#64748B")
    selected = summary["locations"][location]
    figure.suptitle(
        f"Pair {location + 1} · LF affine, complete blur, and sharpening · "
        f"{selected['slide_id']} / tuple {selected['tuple_id']}",
        fontsize=12,
        weight="bold",
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.995,
        bottom=0.02,
        top=0.86,
        wspace=0.025,
        hspace=0.03,
    )
    return save_png(
        figure,
        output_dir / "pairs",
        f"pair_{location + 1:02d}_lf_aligned_hf_endpoints",
    )


def main(argv=None):
    args = parse_args(argv)
    aligned_dir = Path(args.aligned_input)
    unaligned_dir = Path(args.unaligned_input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, embeddings, aligned_frame, aligned_images = (
        load_projection_data(aligned_dir, unaligned_dir)
    )
    summary = json.loads((aligned_dir / "summary.json").read_text())
    projected = fit_projection(frame, embeddings, args.seed)
    projected.to_csv(
        aligned_dir / "combined_projection_coordinates.csv", index=False
    )
    direction_frame = direction_cosines(
        projected, embeddings, summary["scanners"]
    )
    direction_frame.to_csv(
        aligned_dir / "direction_cosines.csv", index=False
    )
    radius_frame = scanner_radius_metrics(projected, embeddings)
    radius_frame.to_csv(
        aligned_dir / "scanner_radius_metrics.csv", index=False
    )
    alignment = (
        aligned_frame[
            aligned_frame["family"] == "aligned_raw"
        ]
        .groupby(
            ["lf_reference_scanner", "source_scanner"], sort=False
        )[[
            "low_mae_raw_to_reference",
            "low_mae_affine_to_reference",
            "low_mae_operational_to_reference",
            "preclip_range_fraction",
            "clip_pixel_mae",
        ]]
        .mean()
        .reset_index()
    )
    alignment.to_csv(
        aligned_dir / "low_frequency_alignment_summary.csv", index=False
    )

    blur_paths = render_aligned_family_umap(
        projected,
        summary,
        output_dir,
        "blur",
        "figure6a_lf_aligned_blur_umap",
    )
    sharpen_paths = render_aligned_family_umap(
        projected,
        summary,
        output_dir,
        "sharpen",
        "figure6b_lf_aligned_sharpen_umap",
    )
    direction_paths = render_sharpen_direction_cosines(
        direction_frame,
        summary["scanners"],
        output_dir,
    )
    strength_paths = render_sharpen_agreement_by_strength(
        direction_frame,
        summary["scanners"],
        output_dir,
    )
    radius_paths, radius_aggregate = render_scanner_radius(
        radius_frame, output_dir
    )
    umap_pair_paths = {
        family: [
            render_pair_family_umap(
                projected,
                summary,
                output_dir,
                family,
                location,
            )
            for location in sorted(
                aligned_frame["location_index"].unique()
            )
        ]
        for family in ("blur", "sharpen")
    }
    pair_paths = [
        render_pair_endpoint_images(
            aligned_frame,
            aligned_images,
            summary,
            output_dir,
            location,
        )
        for location in sorted(
            aligned_frame["location_index"].unique()
        )
    ]

    direction_summary = []
    for reference in REFERENCE_ORDER:
        for parameter in sorted(direction_frame.loc[
            direction_frame["family"] == "sharpen", "parameter"
        ].unique()):
            matrix = direction_matrix(
                direction_frame,
                summary["scanners"],
                reference,
                "sharpen",
                parameter,
            )
            agreements = scanner_agreements(
                matrix, summary["scanners"]
            )
            direction_summary.append({
                "lf_reference_scanner": reference,
                "parameter": float(parameter),
                "scanner_to_others_mean_direction_cosine": agreements,
                "akoya_to_other_scanners_mean_direction_cosine":
                    agreements["akoya"],
                "lowest_agreement_scanner": min(
                    agreements, key=agreements.get
                ),
                "mean_all_off_diagonal_direction_cosine": float(
                    matrix[np.triu_indices(len(matrix), k=1)].mean()
                ),
            })
    report = {
        "experiment": "Exp-02",
        "analysis": "lf_affine_then_high_frequency_trajectory",
        "direction_metric": (
            "Cosine between endpoint-minus-baseline chord directions in "
            "the original normalized UNI embedding space; averaged over "
            "the three registered locations."
        ),
        "sharpen_direction_summary": direction_summary,
        "scanner_radius_summary": json.loads(
            radius_aggregate.to_json(orient="records")
        ),
        "figures": {
            "blur_umap": blur_paths,
            "sharpen_umap": sharpen_paths,
            "sharpen_direction_cosine": direction_paths,
            "sharpen_direction_by_strength": strength_paths,
            "scanner_radius_after_lf_alignment": radius_paths,
            "pair_specific_umap": umap_pair_paths,
            "pair_endpoint_images": pair_paths,
        },
    }
    (aligned_dir / "analysis_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[exp02-lf-aligned-hf-render] {len(projected)} projected rows, "
        f"{len(direction_frame)} direction rows -> {output_dir}"
    )


if __name__ == "__main__":
    main()
