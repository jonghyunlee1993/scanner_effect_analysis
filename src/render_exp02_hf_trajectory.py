"""Measure and render high-frequency UNI embedding trajectories."""

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
    embedding_trajectory_metrics,
    spherical_mean,
)


SCANNER_COLORS = {
    "at2": "#3B82F6",
    "gt450": "#F59E0B",
    "versa": "#10B981",
    "akoya": "#8B5CF6",
}
SCANNER_MARKERS = {
    "at2": "o",
    "gt450": "s",
    "versa": "^",
    "akoya": "D",
}
FAMILY_COLORS = {
    "blur": "#2563EB",
    "sharpen": "#DC2626",
    "registered_mean": "#059669",
    "global_mean": "#7C3AED",
}
FAMILY_LABELS = {
    "blur": "Blur",
    "sharpen": "Sharpen",
    "registered_mean": "Registered HF mean",
    "global_mean": "Global HF mean",
}
FAMILY_CMAPS = {
    "blur": "Blues",
    "sharpen": "Reds",
    "registered_mean": "Greens",
    "global_mean": "Purples",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="outputs/exp02_hf_trajectory"
    )
    parser.add_argument(
        "--output", default="outputs/exp02_hf_trajectory/figures"
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args(argv)


def save_figure(figure, output_dir, stem):
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [str(png), str(pdf)]


def save_png(figure, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return str(path)


def raw_lookup(frame, embeddings):
    raw = frame[frame["family"] == "raw"]
    return {
        (int(row.location_index), row.scanner):
            embeddings[int(row.row_id)]
        for row in raw.itertuples()
    }


def add_embedding_metrics(frame, embeddings):
    raw_vectors = raw_lookup(frame, embeddings)
    values = {
        name: np.empty(len(frame), dtype=np.float64)
        for name in (
            "embedding_displacement",
            "direction_cosine_to_consensus",
            "self_cosine",
            "consensus_cosine",
            "consensus_cosine_gain",
            "signed_consensus_progress",
            "orthogonal_drift",
            "content_margin",
            "cross_scanner_retrieval_hit",
        )
    }
    scanners = frame["scanner"].drop_duplicates().tolist()
    locations = sorted(frame["location_index"].unique())
    for row in frame.itertuples():
        index = int(row.row_id)
        location = int(row.location_index)
        scanner = row.scanner
        raw = raw_vectors[(location, scanner)]
        consensus = spherical_mean(np.stack([
            raw_vectors[(location, other)]
            for other in scanners
            if other != scanner
        ]))
        metric = embedding_trajectory_metrics(
            embeddings[index:index + 1],
            raw[None],
            consensus[None],
        )
        for name, result in metric.items():
            values[name][index] = result[0]

        negative = np.stack([
            raw_vectors[(other_location, other_scanner)]
            for other_location in locations
            for other_scanner in scanners
            if other_location != location
            and other_scanner != scanner
        ])
        negative_max = float(
            np.max(negative @ embeddings[index])
        )
        values["content_margin"][index] = (
            metric["consensus_cosine"][0] - negative_max
        )
        gallery = [
            (other_location, other_scanner)
            for other_location in locations
            for other_scanner in scanners
            if other_scanner != scanner
        ]
        gallery_vectors = np.stack([
            raw_vectors[key] for key in gallery
        ])
        nearest = gallery[int(np.argmax(
            gallery_vectors @ embeddings[index]
        ))]
        values["cross_scanner_retrieval_hit"][index] = float(
            nearest[0] == location
        )

    result = frame.copy()
    for name, value in values.items():
        result[name] = value
    return result


def centroid_metrics(frame, embeddings):
    rows = []
    for keys, group in frame.groupby(
        ["family", "parameter", "location_index"], sort=False
    ):
        family, parameter, location = keys
        vectors = embeddings[group["row_id"].to_numpy(dtype=int)]
        centroid = spherical_mean(vectors)
        cosine = vectors @ centroid
        pairwise = [
            float(vectors[left] @ vectors[right])
            for left, right in combinations(range(len(vectors)), 2)
        ]
        rows.append({
            "family": family,
            "parameter": parameter,
            "location_index": location,
            "patch_centroid_radius": float(np.mean(1.0 - cosine)),
            "patch_pairwise_cosine": float(np.mean(pairwise)),
        })
    patch = pd.DataFrame(rows)

    scanner_rows = []
    for (family, parameter), group in frame.groupby(
        ["family", "parameter"], sort=False
    ):
        centroids = []
        for scanner, selected in group.groupby("scanner", sort=False):
            vectors = embeddings[
                selected["row_id"].to_numpy(dtype=int)
            ]
            centroids.append((scanner, spherical_mean(vectors)))
        grand = spherical_mean(np.stack([value for _, value in centroids]))
        radius = np.mean([
            1.0 - float(value @ grand) for _, value in centroids
        ])
        scanner_rows.append({
            "family": family,
            "parameter": parameter,
            "scanner_centroid_radius": float(radius),
        })
    scanner = pd.DataFrame(scanner_rows)
    return patch.merge(scanner, on=["family", "parameter"], how="left")


def projection_coordinates(frame, embeddings, seed):
    n_components = min(30, len(embeddings) - 1, embeddings.shape[1])
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=seed,
    )
    reduced = pca.fit_transform(embeddings)
    pca_xy = reduced[:, :2]
    import umap

    mapper = umap.UMAP(
        n_neighbors=min(15, len(embeddings) - 1),
        min_dist=0.18,
        metric="euclidean",
        random_state=seed,
        n_jobs=1,
    )
    umap_xy = mapper.fit_transform(reduced)
    result = frame[["row_id", "location_index", "scanner", "family",
                    "parameter"]].copy()
    result["pca_x"] = pca_xy[:, 0]
    result["pca_y"] = pca_xy[:, 1]
    result["umap_x"] = umap_xy[:, 0]
    result["umap_y"] = umap_xy[:, 1]
    return result, pca, mapper


def centroid_trajectories(frame, embeddings, pca, mapper):
    """Project spherical scanner centroids and retain exact UNI-space shifts."""
    rows = []
    vectors = []
    for location in sorted(frame["location_index"].unique()):
        raw_group = frame[
            (frame["location_index"] == location)
            & (frame["family"] == "raw")
        ]
        raw_vectors = embeddings[
            raw_group["row_id"].to_numpy(dtype=int)
        ]
        raw_centroid = spherical_mean(raw_vectors)
        families = [("raw", 0.0)]
        families.extend(
            (family, float(parameter))
            for family in ("registered_mean", "global_mean")
            for parameter in sorted(
                frame.loc[
                    frame["family"] == family, "parameter"
                ].unique()
            )
        )
        for family, parameter in families:
            if family == "raw":
                selected_vectors = raw_vectors
            else:
                selected = frame[
                    (frame["location_index"] == location)
                    & (frame["family"] == family)
                    & np.isclose(frame["parameter"], parameter)
                ]
                selected_vectors = embeddings[
                    selected["row_id"].to_numpy(dtype=int)
                ]
            centroid = spherical_mean(selected_vectors)
            rows.append({
                "location_index": int(location),
                "family": family,
                "parameter": float(parameter),
                "centroid_self_cosine": float(
                    centroid @ raw_centroid
                ),
                "centroid_shift": float(
                    1.0 - centroid @ raw_centroid
                ),
                "patch_centroid_radius": float(np.mean(
                    1.0 - selected_vectors @ centroid
                )),
            })
            vectors.append(centroid)
    centroid_vectors = np.stack(vectors)
    reduced = pca.transform(centroid_vectors)
    umap_xy = mapper.transform(reduced)
    result = pd.DataFrame(rows)
    result["umap_x"] = umap_xy[:, 0]
    result["umap_y"] = umap_xy[:, 1]
    return result


def family_path(frame, location, scanner, family):
    raw = frame[
        (frame["location_index"] == location)
        & (frame["scanner"] == scanner)
        & (frame["family"] == "raw")
    ]
    selected = frame[
        (frame["location_index"] == location)
        & (frame["scanner"] == scanner)
        & (frame["family"] == family)
    ].sort_values("parameter")
    return pd.concat((raw, selected), ignore_index=True)


def padded_limits(frame, pad=0.12):
    x_span = max(
        float(frame["umap_x"].max() - frame["umap_x"].min()), 0.5
    )
    y_span = max(
        float(frame["umap_y"].max() - frame["umap_y"].min()), 0.5
    )
    return (
        (
            float(frame["umap_x"].min() - pad * x_span),
            float(frame["umap_x"].max() + pad * x_span),
        ),
        (
            float(frame["umap_y"].min() - pad * y_span),
            float(frame["umap_y"].max() + pad * y_span),
        ),
    )


def draw_gradient_path(axis, path, cmap, marker, norm):
    points = path[["umap_x", "umap_y"]].to_numpy()
    parameters = path["parameter"].to_numpy(dtype=float)
    for index in range(len(path) - 1):
        color = cmap(norm(parameters[index + 1]))
        axis.annotate(
            "",
            xy=points[index + 1],
            xytext=points[index],
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": 1.8,
                "mutation_scale": 9,
                "shrinkA": 1,
                "shrinkB": 1,
            },
            zorder=2,
        )
    axis.scatter(
        points[1:, 0],
        points[1:, 1],
        c=parameters[1:],
        cmap=cmap,
        norm=norm,
        marker=marker,
        s=42,
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
    )
    axis.scatter(
        points[0, 0],
        points[0, 1],
        marker=marker,
        s=54,
        facecolor="white",
        edgecolor="#111827",
        linewidth=1.2,
        zorder=4,
    )


def render_family_umap(
    metrics,
    coordinates,
    summary,
    output_dir,
    family,
    stem,
):
    frame = metrics.merge(coordinates, on=[
        "row_id", "location_index", "scanner", "family", "parameter"
    ])
    locations = sorted(frame["location_index"].unique())
    scanners = summary["scanners"]
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap(FAMILY_CMAPS[family])
    figure, axes = plt.subplots(
        1, len(locations), figsize=(15.5, 4.8), squeeze=False
    )
    axes = axes[0]

    for axis, location in zip(axes, locations):
        family_rows = frame[
            (frame["location_index"] == location)
            & (frame["family"].isin(("raw", family)))
        ]
        xlim, ylim = padded_limits(family_rows)
        for scanner in scanners:
            draw_gradient_path(
                axis,
                family_path(frame, location, scanner, family),
                cmap,
                SCANNER_MARKERS[scanner],
                norm,
            )
        selected = summary["locations"][location]
        axis.set_title(
            f"Patch {location + 1} · {selected['slide_id']} / "
            f"tuple {selected['tuple_id']}\n"
            f"min registration q = {selected['min_q_reg']:.3f}",
            fontsize=10,
        )
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(alpha=0.13)
        axis.spines[["top", "right"]].set_visible(False)

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
        ncol=2,
        fontsize=8,
        loc="best",
        title="Scanner marker",
        title_fontsize=8,
    )
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        fraction=0.02,
        pad=0.025,
    )
    colorbar.set_label(f"{FAMILY_LABELS[family]} strength")
    figure.suptitle(
        f"{FAMILY_LABELS[family]}: scanner-specific UNI motion in one fixed UMAP",
        fontsize=14,
        weight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Open marker = raw · darker color = stronger intervention · "
        "arrows show direction (UMAP is descriptive)",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.04, right=0.91, bottom=0.10, top=0.82, wspace=0.16)
    return save_figure(figure, output_dir, stem)


def centroid_family_path(frame, location, family):
    raw = frame[
        (frame["location_index"] == location)
        & (frame["family"] == "raw")
    ]
    selected = frame[
        (frame["location_index"] == location)
        & (frame["family"] == family)
    ].sort_values("parameter")
    return pd.concat((raw, selected), ignore_index=True)


def render_reference_centroids(
    coordinates,
    centroids,
    summary,
    output_dir,
):
    locations = sorted(centroids["location_index"].unique())
    figure, axes = plt.subplots(
        len(locations),
        2,
        figsize=(12.5, 4.1 * len(locations)),
        squeeze=False,
    )
    norm = Normalize(vmin=0.0, vmax=1.0)
    for row_index, location in enumerate(locations):
        umap_axis, tradeoff_axis = axes[row_index]
        raw_points = coordinates[
            (coordinates["location_index"] == location)
            & (coordinates["family"] == "raw")
        ]
        local_centroids = centroids[
            centroids["location_index"] == location
        ]
        limits = pd.concat((
            raw_points[["umap_x", "umap_y"]],
            local_centroids[["umap_x", "umap_y"]],
        ))
        xlim, ylim = padded_limits(limits, pad=0.16)
        for scanner in summary["scanners"]:
            selected = raw_points[raw_points["scanner"] == scanner]
            umap_axis.scatter(
                selected["umap_x"],
                selected["umap_y"],
                color=SCANNER_COLORS[scanner],
                marker=SCANNER_MARKERS[scanner],
                s=46,
                alpha=0.65,
                label=scanner.upper() if row_index == 0 else None,
                zorder=2,
            )
        raw_centroid = local_centroids[
            local_centroids["family"] == "raw"
        ].iloc[0]
        umap_axis.scatter(
            raw_centroid["umap_x"],
            raw_centroid["umap_y"],
            marker="*",
            s=145,
            color="#111827",
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        tradeoff_axis.scatter(
            0.0,
            raw_centroid["patch_centroid_radius"],
            marker="*",
            s=130,
            color="#111827",
            zorder=5,
        )
        for family in ("registered_mean", "global_mean"):
            path = centroid_family_path(
                local_centroids, location, family
            )
            cmap = plt.get_cmap(FAMILY_CMAPS[family])
            points = path[["umap_x", "umap_y"]].to_numpy()
            parameter = path["parameter"].to_numpy(dtype=float)
            for index in range(len(path) - 1):
                umap_axis.annotate(
                    "",
                    xy=points[index + 1],
                    xytext=points[index],
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": cmap(norm(parameter[index + 1])),
                        "linewidth": 2.0,
                        "mutation_scale": 9,
                    },
                    zorder=3,
                )
            umap_axis.scatter(
                points[1:, 0],
                points[1:, 1],
                c=parameter[1:],
                cmap=cmap,
                norm=norm,
                s=46,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
            tradeoff_axis.plot(
                path["centroid_shift"],
                path["patch_centroid_radius"],
                color=FAMILY_COLORS[family],
                linewidth=1.8,
                label=FAMILY_LABELS[family] if row_index == 0 else None,
            )
            tradeoff_axis.scatter(
                path["centroid_shift"].iloc[1:],
                path["patch_centroid_radius"].iloc[1:],
                c=parameter[1:],
                cmap=cmap,
                norm=norm,
                s=45,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            for point in path.iloc[1:].itertuples():
                tradeoff_axis.annotate(
                    f"{point.parameter:.2g}",
                    (point.centroid_shift, point.patch_centroid_radius),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=FAMILY_COLORS[family],
                )

        selected = summary["locations"][location]
        umap_axis.set_ylabel(
            f"Patch {location + 1}\n{selected['slide_id']} / "
            f"tuple {selected['tuple_id']}\n"
            f"min q={selected['min_q_reg']:.3f}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=9,
        )
        umap_axis.set_xlim(*xlim)
        umap_axis.set_ylim(*ylim)
        umap_axis.set_xticks([])
        umap_axis.set_yticks([])
        umap_axis.grid(alpha=0.13)
        tradeoff_axis.set_xlabel(
            "Centroid shift from raw\n1 − cosine(raw centroid, mixed centroid) ↑"
        )
        tradeoff_axis.set_ylabel(
            "Scanner radius around centroid\nmean 1 − cosine(scanner, centroid) ↓"
        )
        tradeoff_axis.grid(alpha=0.22)
        for axis in (umap_axis, tradeoff_axis):
            axis.spines[["top", "right"]].set_visible(False)
        if row_index == 0:
            umap_axis.set_title(
                "A. Raw scanner embeddings and centroid paths\n"
                "(fixed UMAP; descriptive)"
            )
            tradeoff_axis.set_title(
                "B. Exact centroid movement versus scanner gathering\n"
                "(original normalized UNI space)"
            )
            umap_axis.legend(
                frameon=False,
                fontsize=8,
                ncol=2,
                loc="best",
                title="Raw scanner embeddings",
                title_fontsize=8,
            )
            tradeoff_axis.legend(frameon=False, fontsize=8, loc="best")
    figure.suptitle(
        "Image-space HF references: where does the resulting UNI centroid move?",
        fontsize=14,
        weight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "★ = spherical mean of raw scanner embeddings. "
        "References are means of image HF bands, not means of UNI features.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.12, right=0.97, bottom=0.07, top=0.91, hspace=0.34, wspace=0.30
    )
    return save_figure(
        figure, output_dir, "figure4c_reference_centroid_umap"
    )


def aggregate_with_raw(frame, value_columns):
    pieces = []
    raw = frame[frame["family"] == "raw"]
    for family in FAMILY_COLORS:
        start = raw.copy()
        start["family"] = family
        start["parameter"] = 0.0
        pieces.append(start)
        pieces.append(frame[frame["family"] == family])
    combined = pd.concat(pieces, ignore_index=True)
    return combined.groupby(
        ["family", "parameter"], sort=False
    )[value_columns].mean().reset_index()


def render_quantitative_metrics(
    metrics, centroid, summary, output_dir
):
    value_columns = [
        "self_cosine",
        "consensus_cosine_gain",
        "direction_cosine_to_consensus",
        "content_margin",
        "cross_scanner_retrieval_hit",
    ]
    aggregate = aggregate_with_raw(metrics, value_columns)
    centroid_aggregate = aggregate_with_raw(
        centroid,
        ["patch_centroid_radius", "scanner_centroid_radius"],
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9.2))

    for family, color in FAMILY_COLORS.items():
        selected = aggregate[aggregate["family"] == family].sort_values(
            "parameter"
        )
        axes[0, 0].plot(
            selected["self_cosine"],
            selected["consensus_cosine_gain"],
            marker="o", color=color, label=FAMILY_LABELS[family],
        )
        axes[0, 1].plot(
            selected["parameter"],
            selected["direction_cosine_to_consensus"],
            marker="o", color=color, label=FAMILY_LABELS[family],
        )
        selected_centroid = centroid_aggregate[
            centroid_aggregate["family"] == family
        ].sort_values("parameter")
        axes[1, 0].plot(
            selected_centroid["parameter"],
            selected_centroid["patch_centroid_radius"],
            marker="o", color=color,
            label=f"{FAMILY_LABELS[family]} · patch",
        )
        axes[1, 0].plot(
            selected_centroid["parameter"],
            selected_centroid["scanner_centroid_radius"],
            color=color, linestyle=":", alpha=0.8,
        )
        axes[1, 1].plot(
            selected["parameter"],
            selected["content_margin"],
            marker="o", color=color, label=FAMILY_LABELS[family],
        )

    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_title("A. Consensus progress versus self preservation")
    axes[0, 0].set_xlabel("UNI cosine to own raw image ↑")
    axes[0, 0].set_ylabel("Δ UNI cosine to LOO patch consensus ↑")

    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("B. Direction of embedding motion")
    axes[0, 1].set_xlabel("Intervention strength")
    axes[0, 1].set_ylabel("Cosine angle toward raw patch consensus ↑")

    axes[1, 0].set_title("C. Do scanner centroids gather?")
    axes[1, 0].set_xlabel("Intervention strength")
    axes[1, 0].set_ylabel("Cosine radius ↓")
    axes[1, 0].text(
        0.98, 0.96, "solid: within-patch scanner radius\n"
        "dotted: scanner-centroid radius",
        transform=axes[1, 0].transAxes, ha="right", va="top", fontsize=8,
    )

    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("D. Content margin against other patches")
    axes[1, 1].set_xlabel("Intervention strength")
    axes[1, 1].set_ylabel(
        "LOO consensus cosine − max other-location cosine ↑"
    )

    for axis in axes.flat:
        axis.grid(alpha=0.23)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "High-frequency perturbation: alignment gains versus content loss",
        fontsize=14, weight="bold",
    )
    figure.tight_layout()
    paths = save_figure(
        figure, output_dir, "figure5_hf_trajectory_metrics"
    )
    return paths, aggregate, centroid_aggregate


def condition_variants(condition):
    if condition == "blur":
        return [
            ("raw", 0.0, "Raw"),
            *[
                ("blur", parameter, f"Blur {parameter:.2f}")
                for parameter in (0.25, 0.50, 0.75, 1.00)
            ],
        ]
    if condition == "sharpen":
        return [
            ("raw", 0.0, "Raw"),
            *[
                ("sharpen", parameter, f"Sharpen {parameter:.2f}")
                for parameter in (0.25, 0.50, 1.00)
            ],
        ]
    if condition == "reference_mixing":
        return [
            ("raw", 0.0, "Raw"),
            *[
                (
                    "registered_mean",
                    parameter,
                    f"Registered\n{parameter:.2f}",
                )
                for parameter in (0.25, 0.50, 0.75, 1.00)
            ],
            *[
                (
                    "global_mean",
                    parameter,
                    f"Global train\n{parameter:.2f}",
                )
                for parameter in (0.25, 0.50, 0.75, 1.00)
            ],
        ]
    raise ValueError(f"unknown image condition: {condition}")


def selected_image(frame, images, location, scanner, family, parameter):
    selected = frame[
        (frame["location_index"] == location)
        & (frame["scanner"] == scanner)
        & (frame["family"] == family)
        & np.isclose(frame["parameter"], parameter)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one image for {location}/{scanner}/{family}/{parameter}"
        )
    return images[int(selected.iloc[0]["row_id"])]


def render_pair_images(
    frame,
    images,
    summary,
    output_dir,
    location,
    condition,
):
    variants = condition_variants(condition)
    scanners = summary["scanners"]
    figure, axes = plt.subplots(
        len(scanners),
        len(variants),
        figsize=(1.75 * len(variants), 1.75 * len(scanners)),
        squeeze=False,
    )
    for row, scanner in enumerate(scanners):
        for column, (family, parameter, label) in enumerate(variants):
            axis = axes[row, column]
            axis.imshow(selected_image(
                frame,
                images,
                location,
                scanner,
                family,
                parameter,
            ))
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(label, fontsize=8.5, weight="bold")
            if column == 0:
                axis.set_ylabel(
                    scanner.upper(),
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=9,
                    weight="bold",
                    color=SCANNER_COLORS[scanner],
                )
            if condition == "reference_mixing" and column == 4:
                axis.spines["right"].set_color("#64748B")
                axis.spines["right"].set_linewidth(2.0)
    selected = summary["locations"][location]
    condition_label = {
        "blur": "Blur",
        "sharpen": "Sharpen",
        "reference_mixing": "HF-band reference mixing",
    }[condition]
    figure.suptitle(
        f"{condition_label} · Pair {location + 1} · "
        f"{selected['slide_id']} / tuple {selected['tuple_id']} · "
        f"min registration q={selected['min_q_reg']:.3f}",
        fontsize=12,
        weight="bold",
    )
    figure.subplots_adjust(
        left=0.07, right=0.995, bottom=0.02, top=0.88,
        wspace=0.035, hspace=0.035,
    )
    condition_dir = output_dir / "pairs" / condition
    return save_png(
        figure,
        condition_dir,
        f"pair_{location + 1:02d}_{condition}",
    )


def render_all_pair_images(frame, images, summary, output_dir):
    paths = {}
    locations = sorted(frame["location_index"].unique())
    for condition in ("blur", "sharpen", "reference_mixing"):
        paths[condition] = [
            render_pair_images(
                frame,
                images,
                summary,
                output_dir,
                location,
                condition,
            )
            for location in locations
        ]
    return paths


def main(argv=None):
    args = parse_args(argv)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_dir / "trajectory_metadata.csv")
    arrays = np.load(input_dir / "trajectory_arrays.npz")
    embeddings = arrays["embeddings"]
    images = arrays["images"]
    summary = json.loads((input_dir / "summary.json").read_text())
    if not np.array_equal(
        frame["row_id"].to_numpy(), np.arange(len(frame))
    ):
        raise ValueError("trajectory rows must be contiguous and ordered")

    metrics = add_embedding_metrics(frame, embeddings)
    centroid = centroid_metrics(frame, embeddings)
    coordinates, pca, mapper = projection_coordinates(
        frame, embeddings, args.seed
    )
    reference_centroids = centroid_trajectories(
        frame, embeddings, pca, mapper
    )
    metrics.to_csv(input_dir / "trajectory_metrics.csv", index=False)
    centroid.to_csv(input_dir / "centroid_metrics.csv", index=False)
    coordinates.to_csv(
        input_dir / "projection_coordinates.csv", index=False
    )
    reference_centroids.to_csv(
        input_dir / "reference_centroid_trajectories.csv", index=False
    )

    blur_paths = render_family_umap(
        metrics,
        coordinates,
        summary,
        output_dir,
        "blur",
        "figure4a_blur_umap",
    )
    sharpen_paths = render_family_umap(
        metrics,
        coordinates,
        summary,
        output_dir,
        "sharpen",
        "figure4b_sharpen_umap",
    )
    reference_paths = render_reference_centroids(
        coordinates,
        reference_centroids,
        summary,
        output_dir,
    )
    metric_paths, aggregate, centroid_aggregate = (
        render_quantitative_metrics(
            metrics, centroid, summary, output_dir
        )
    )
    pair_image_paths = render_all_pair_images(
        frame, images, summary, output_dir
    )

    strongest = aggregate.sort_values(
        "consensus_cosine_gain", ascending=False
    ).iloc[0]
    aggregate_records = json.loads(
        aggregate.to_json(orient="records")
    )
    centroid_records = json.loads(
        centroid_aggregate.to_json(orient="records")
    )
    reference_centroid_records = json.loads(
        reference_centroids.to_json(orient="records")
    )
    report = {
        "experiment": "Exp-02",
        "analysis": "high_frequency_embedding_trajectory",
        "geometry_note": (
            "All distances, angles, centroids, and margins are computed in "
            "the original normalized UNI embedding space. UMAP is descriptive."
        ),
        "global_high_frequency_reference":
            summary["global_high_frequency_reference"],
        "reference_definition": {
            "registered_mean": (
                "For each source scanner and registered location, the "
                "Laplacian high-frequency bands are mixed toward the mean "
                "bands of the other three scanners at that same location."
            ),
            "global_mean": (
                "Each source image's Laplacian high-frequency bands are "
                "mixed toward coefficient-wise mean bands estimated from "
                "1,024 training images."
            ),
            "resulting_uni_centroid": (
                "At each mix amount, the displayed virtual centroid is the "
                "spherical mean of the four resulting UNI embeddings; the "
                "HF references themselves are not means of UNI features."
            ),
        },
        "aggregate_metrics": aggregate_records,
        "aggregate_centroid_metrics": centroid_records,
        "reference_centroid_trajectories": reference_centroid_records,
        "strongest_mean_consensus_gain": {
            "family": strongest["family"],
            "parameter": float(strongest["parameter"]),
            "value": float(strongest["consensus_cosine_gain"]),
            "self_cosine": float(strongest["self_cosine"]),
        },
        "figures": {
            "blur_umap_trajectories": blur_paths,
            "sharpen_umap_trajectories": sharpen_paths,
            "reference_centroid_trajectories": reference_paths,
            "embedding_metrics": metric_paths,
            "pair_image_perturbations": pair_image_paths,
        },
    }
    (input_dir / "analysis_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[exp02-hf-render] {len(metrics)} embedding rows, "
        f"{len(centroid)} centroid rows -> {output_dir}"
    )


if __name__ == "__main__":
    main()
