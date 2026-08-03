"""Render joint pre/post UMAPs for image corrections across the four frozen PFMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from e5_comparator_population import IMAGE_CONDITIONS, SCANNERS
from e5_reinhard_residual_frequency import RF1_CONDITION
from fetch_e0_pfm_checkpoints import sha256


SEED = 20_260_803
PCA_COMPONENTS = 50
UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.15
CONDITIONS = (
    "raw",
    *IMAGE_CONDITIONS,
    "macenko_supplement",
    RF1_CONDITION,
)
CONDITION_LABELS = {
    "raw": "Pre: raw",
    "reinhard_lab": "Post: Reinhard",
    "paired_od_affine": "Post: Paired OD",
    "frequency_calibration": "Post: Frequency",
    "macenko_supplement": "Post: Macenko",
    RF1_CONDITION: "Post: Ours",
}
MODEL_LABELS = {
    "resnet50": "ResNet50",
    "uni_v1": "UNI v1",
    "conch_v1": "CONCH v1",
    "virchow2": "Virchow2",
}
SCANNER_COLORS = {
    "at2": "#111827",
    "gt450": "#2563EB",
    "versa": "#059669",
    "akoya": "#DC2626",
    "s60": "#D97706",
    "s360": "#7C3AED",
}
METHOD_MARKERS = {
    "raw": "o",
    "reinhard_lab": "s",
    "paired_od_affine": "D",
    "frequency_calibration": "^",
    "macenko_supplement": "P",
    RF1_CONDITION: "X",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--image", default="outputs/e5_image_features")
    parser.add_argument("--macenko", default="outputs/e5_macenko_features")
    parser.add_argument("--rf1", default="outputs/e5_rf1_features")
    parser.add_argument("--e5-audit", default="outputs/e5_comparator_features/audit/summary.json")
    parser.add_argument("--macenko-audit", default="outputs/e5_macenko_features/audit/summary.json")
    parser.add_argument("--rf1-audit", default="outputs/e5_rf1_features/audit/summary.json")
    parser.add_argument("--e5-endpoints", default="outputs/e5_comparator_frontier/endpoint_summary.csv")
    parser.add_argument("--macenko-endpoints", default="outputs/e5_macenko_supplement/endpoint_summary.csv")
    parser.add_argument("--rf1-endpoints", default="outputs/e5_rf1_frontier/endpoint_summary.csv")
    parser.add_argument("--output", default="outputs/e5_rf1_visual_comparison")
    parser.add_argument("--dpi", type=int, default=250)
    return parser.parse_args()


def strings(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def patch_normalized_centroids(features: np.ndarray):
    value = np.asarray(features, dtype=np.float32)
    if value.ndim != 3 or value.shape[0:2] != (6, 100):
        raise ValueError(f"invalid feature population for centroids: {value.shape}")
    norms = np.linalg.norm(value.astype(np.float64), axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("non-finite or zero patch feature norm")
    unit = value / norms.astype(np.float32)
    centroids = unit.mean(axis=1)
    centroid_norms = np.linalg.norm(centroids.astype(np.float64), axis=-1, keepdims=True)
    if not np.isfinite(centroid_norms).all() or np.any(centroid_norms <= 0):
        raise ValueError("non-finite or zero slide-scanner centroid norm")
    return centroids / centroid_norms.astype(np.float32)


def read_condition_features(raw_path: Path, image_path: Path, macenko_path: Path, rf1_path: Path):
    with h5py.File(raw_path, "r") as source:
        raw = np.asarray(source["features"][:], dtype=np.float32)
        scanners = strings(source["scanner"][:])
        locations = source["location_id"][:]
    with h5py.File(image_path, "r") as source:
        image = np.asarray(source["features"][:], dtype=np.float32)
        image_conditions = strings(source["condition"][:])
        image_locations = source["location_id"][:]
    with h5py.File(macenko_path, "r") as source:
        macenko = np.asarray(source["features"][:], dtype=np.float32)
        macenko_conditions = strings(source["condition"][:])
        macenko_locations = source["location_id"][:]
    with h5py.File(rf1_path, "r") as source:
        rf1 = np.asarray(source["features"][:], dtype=np.float32)
        rf1_conditions = strings(source["condition"][:])
        rf1_locations = source["location_id"][:]
    dimension = raw.shape[-1]
    if (
        raw.shape != (6, 100, dimension)
        or image.shape != (3, 6, 100, dimension)
        or macenko.shape != (1, 6, 100, dimension)
        or rf1.shape != (1, 6, 100, dimension)
        or scanners != list(SCANNERS)
        or image_conditions != list(IMAGE_CONDITIONS)
        or macenko_conditions != ["macenko_supplement"]
        or rf1_conditions != [RF1_CONDITION]
        or not np.array_equal(locations, image_locations)
        or not np.array_equal(locations, macenko_locations)
        or not np.array_equal(locations, rf1_locations)
        or not all(np.array_equal(raw[0], image[index, 0]) for index in range(3))
        or not np.array_equal(raw[0], macenko[0, 0])
        or not np.array_equal(raw[0], rf1[0, 0])
    ):
        raise ValueError(f"{raw_path}: condition identity/schema mismatch")
    values = {"raw": raw}
    values.update({condition: image[index] for index, condition in enumerate(IMAGE_CONDITIONS)})
    values["macenko_supplement"] = macenko[0]
    values[RF1_CONDITION] = rf1[0]
    return values


def load_model_centroids(model: dict, args):
    model_id = model["encoder_id"]
    raw_paths = sorted((Path(args.raw) / model_id / "shards").glob("*.h5"))
    if len(raw_paths) != 109:
        raise ValueError(f"{model_id}: expected 109 raw feature shards")
    rows = []
    matrices = []
    for slide_index, raw_path in enumerate(raw_paths):
        slide_id = raw_path.stem
        values = read_condition_features(
            raw_path,
            Path(args.image) / model_id / "shards" / raw_path.name,
            Path(args.macenko) / model_id / "shards" / raw_path.name,
            Path(args.rf1) / model_id / "shards" / raw_path.name,
        )
        for condition in CONDITIONS:
            centroids = patch_normalized_centroids(values[condition])
            for scanner_index, scanner in enumerate(SCANNERS):
                rows.append(
                    {
                        "encoder_id": model_id,
                        "slide_id": slide_id,
                        "condition": condition,
                        "scanner": scanner,
                        "fit_member": bool(condition == "raw" or scanner != "at2"),
                    }
                )
                matrices.append(centroids[scanner_index])
        if (slide_index + 1) % 20 == 0 or slide_index + 1 == len(raw_paths):
            print(f"[{slide_index + 1}/109] {model_id} centroids", flush=True)
    frame = pd.DataFrame(rows)
    matrix = np.asarray(matrices, dtype=np.float32)
    expected_rows = 109 * len(CONDITIONS) * len(SCANNERS)
    if len(frame) != expected_rows or matrix.shape != (expected_rows, int(model["feature_dim"])):
        raise RuntimeError(f"{model_id}: incomplete centroid population")
    return frame, matrix


def joint_umap(frame: pd.DataFrame, matrix: np.ndarray):
    import umap

    fit_indices = np.flatnonzero(frame["fit_member"].to_numpy(dtype=bool))
    fit_matrix = matrix[fit_indices]
    components = min(PCA_COMPONENTS, fit_matrix.shape[0] - 1, fit_matrix.shape[1])
    reducer = PCA(n_components=components, svd_solver="randomized", random_state=SEED)
    reduced = reducer.fit_transform(fit_matrix)
    mapper = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=SEED,
        transform_seed=SEED,
        n_jobs=1,
    )
    fitted_coordinates = mapper.fit_transform(reduced)
    coordinates = np.full((len(frame), 2), np.nan, dtype=np.float32)
    coordinates[fit_indices] = fitted_coordinates
    raw_at2 = {
        row.slide_id: coordinates[index]
        for index, row in frame.iterrows()
        if row.condition == "raw" and row.scanner == "at2"
    }
    duplicate_target = (~frame["fit_member"]) & frame["scanner"].eq("at2")
    for index in frame.index[duplicate_target]:
        coordinates[index] = raw_at2[frame.at[index, "slide_id"]]
    if not np.isfinite(coordinates).all():
        raise RuntimeError("joint UMAP contains unassigned/non-finite coordinates")
    result = frame.copy()
    result["umap_x"] = coordinates[:, 0]
    result["umap_y"] = coordinates[:, 1]
    return result, float(reducer.explained_variance_ratio_.sum())


def endpoint_map(args):
    values = {}
    e5 = pd.read_csv(args.e5_endpoints)
    for row in e5[e5["condition"].isin(("raw", *IMAGE_CONDITIONS))].itertuples():
        values[(row.encoder_id, row.condition)] = (
            float(row.relative_radius_reduction),
            bool(row.safe_for_pfm),
        )
    macenko = pd.read_csv(args.macenko_endpoints)
    for row in macenko.itertuples():
        values[(row.encoder_id, row.condition)] = (
            float(row.relative_radius_reduction),
            bool(row.safe_for_pfm),
        )
    rf1 = pd.read_csv(args.rf1_endpoints)
    for row in rf1.itertuples():
        values[(row.encoder_id, row.condition)] = (
            float(row.relative_radius_reduction),
            bool(row.safe_for_pfm),
        )
    return values


def model_limits(frame: pd.DataFrame):
    x_span = max(float(frame["umap_x"].max() - frame["umap_x"].min()), 1.0)
    y_span = max(float(frame["umap_y"].max() - frame["umap_y"].min()), 1.0)
    return (
        (float(frame["umap_x"].min() - 0.04 * x_span), float(frame["umap_x"].max() + 0.04 * x_span)),
        (float(frame["umap_y"].min() - 0.04 * y_span), float(frame["umap_y"].max() + 0.04 * y_span)),
    )


def scatter_condition(axis, frame: pd.DataFrame, condition: str, model_id: str, endpoints: dict):
    selected = frame[frame["condition"] == condition]
    for scanner in SCANNERS:
        local = selected[selected["scanner"] == scanner]
        axis.scatter(
            local["umap_x"],
            local["umap_y"],
            s=15 if scanner == "at2" else 9,
            marker="*" if scanner == "at2" else "o",
            c=SCANNER_COLORS[scanner],
            alpha=0.58 if scanner == "at2" else 0.38,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            local["umap_x"].mean(),
            local["umap_y"].mean(),
            s=82,
            marker="*" if scanner == "at2" else "X",
            c=SCANNER_COLORS[scanner],
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
    rr, safe = endpoints[(model_id, condition)]
    suffix = "safe" if safe else "unsafe"
    axis.set_title(
        f"{CONDITION_LABELS[condition]}\nRR {100 * rr:+.1f}% · {suffix}",
        fontsize=9,
        color="#111827" if safe else "#B91C1C",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(alpha=0.12)


def save_figure(figure, output: Path, stem: str, dpi: int):
    manifest = {}
    for suffix in ("png", "pdf"):
        path = output / f"{stem}.{suffix}"
        figure.savefig(path, dpi=dpi if suffix == "png" else None, bbox_inches="tight")
        manifest[path.name] = {"bytes": int(path.stat().st_size), "sha256": sha256(path)}
    plt.close(figure)
    return manifest


def scanner_legend():
    return [
        Line2D(
            [0],
            [0],
            marker="*" if scanner == "at2" else "o",
            linestyle="none",
            color=SCANNER_COLORS[scanner],
            markersize=8,
            label=scanner.upper(),
        )
        for scanner in SCANNERS
    ]


def render_model(frame: pd.DataFrame, model_id: str, endpoints: dict, output: Path, dpi: int):
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 9.3))
    xlim, ylim = model_limits(frame)
    for axis, condition in zip(axes.flat, CONDITIONS):
        scatter_condition(axis, frame, condition, model_id, endpoints)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
    figure.legend(
        handles=scanner_legend(),
        loc="lower center",
        ncol=len(SCANNERS),
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    figure.suptitle(
        f"{MODEL_LABELS[model_id]} — one joint UMAP for pre/post slide-scanner centroids",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.02,
        "Small points: physical slides · large markers: scanner grand centroids · UMAP is descriptive",
        ha="center",
        fontsize=8.5,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    return save_figure(figure, output, f"joint_umap_{model_id}", dpi)


def render_combined(frames: dict, endpoints: dict, output: Path, dpi: int):
    figure, axes = plt.subplots(4, len(CONDITIONS), figsize=(26, 17.5))
    for row_index, (model_id, frame) in enumerate(frames.items()):
        xlim, ylim = model_limits(frame)
        for column_index, condition in enumerate(CONDITIONS):
            axis = axes[row_index, column_index]
            scatter_condition(axis, frame, condition, model_id, endpoints)
            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)
            if column_index == 0:
                axis.set_ylabel(MODEL_LABELS[model_id], fontsize=11, fontweight="bold")
    figure.legend(
        handles=scanner_legend(),
        loc="lower center",
        ncol=len(SCANNERS),
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.suptitle(
        "Pre/post image-correction representations in a fixed joint UMAP per PFM",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.028,
        "Axes are shared across conditions within each row, but not comparable across PFMs · "
        "small points are slides; large markers are scanner centroids",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0.01, 0.045, 1, 0.98))
    return save_figure(figure, output, "joint_umap_all_pfms", dpi)


def render_trajectories(frames: dict, output: Path, dpi: int):
    figure, axes = plt.subplots(2, 2, figsize=(15, 13))
    method_conditions = CONDITIONS[1:]
    for axis, (model_id, frame) in zip(axes.flat, frames.items()):
        means = frame.groupby(["condition", "scanner"])[["umap_x", "umap_y"]].mean()
        target = means.loc[("raw", "at2")].to_numpy()
        axis.scatter(
            target[0], target[1], marker="*", s=260, c=SCANNER_COLORS["at2"],
            edgecolors="white", linewidths=1.0, zorder=8,
        )
        axis.text(target[0], target[1], " AT2", va="center", fontsize=8, fontweight="bold")
        for scanner in SCANNERS[1:]:
            raw = means.loc[("raw", scanner)].to_numpy()
            axis.scatter(
                raw[0], raw[1], marker="o", s=90, facecolors="none",
                edgecolors=SCANNER_COLORS[scanner], linewidths=1.8, zorder=7,
            )
            axis.text(raw[0], raw[1], f" {scanner.upper()}", fontsize=7.5, va="center")
            for condition in method_conditions:
                point = means.loc[(condition, scanner)].to_numpy()
                axis.plot(
                    [raw[0], point[0]], [raw[1], point[1]],
                    color=SCANNER_COLORS[scanner], alpha=0.25, linewidth=0.8,
                )
                axis.scatter(
                    point[0], point[1], marker=METHOD_MARKERS[condition], s=70,
                    c=SCANNER_COLORS[scanner], edgecolors="white", linewidths=0.55, zorder=6,
                )
        axis.set_title(MODEL_LABELS[model_id], fontsize=12, fontweight="bold")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(alpha=0.13)
    method_handles = [
        Line2D(
            [0], [0], marker=METHOD_MARKERS[condition], linestyle="none", color="#374151",
            markersize=8, label=CONDITION_LABELS[condition].replace("Post: ", ""),
            markerfacecolor="none" if condition == "raw" else "#374151",
        )
        for condition in CONDITIONS
    ]
    scanner_handles = scanner_legend()[1:]
    figure.legend(
        handles=method_handles,
        loc="lower center",
        ncol=len(method_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
        title="Method marker",
    )
    figure.legend(
        handles=scanner_handles,
        loc="lower center",
        ncol=len(scanner_handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        title="Source-scanner color",
    )
    figure.suptitle(
        "Scanner-centroid motion from raw toward the paired AT2 reference",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.072,
        "Open circle = raw source; lines connect raw to each post-correction centroid · UMAP is descriptive",
        ha="center",
        fontsize=8.5,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.96))
    return save_figure(figure, output, "umap_scanner_centroid_trajectories", dpi)


def main():
    args = parse_args()
    audit_paths = {
        "e5": Path(args.e5_audit),
        "macenko": Path(args.macenko_audit),
        "rf1": Path(args.rf1_audit),
    }
    audits = {name: json.loads(path.read_text()) for name, path in audit_paths.items()}
    if not (
        audits["e5"].get("audit_pass") is True
        and audits["e5"].get("total_features_observed") == 1_308_000
        and audits["macenko"].get("audit_pass") is True
        and audits["macenko"].get("features_observed") == 261_600
        and audits["rf1"].get("audit_pass") is True
        and audits["rf1"].get("features_observed") == 261_600
    ):
        raise RuntimeError("one or more feature populations have not passed audit")
    contract = json.loads(Path(args.contract).read_text())
    endpoints = endpoint_map(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frames = {}
    pca_variance = {}
    figures = {}
    for model in contract["models"]:
        model_id = model["encoder_id"]
        frame, matrix = load_model_centroids(model, args)
        coordinates, variance = joint_umap(frame, matrix)
        coordinates_path = output / f"joint_umap_coordinates_{model_id}.csv"
        coordinates.to_csv(coordinates_path, index=False)
        frames[model_id] = coordinates
        pca_variance[model_id] = variance
        figures.update(render_model(coordinates, model_id, endpoints, output, args.dpi))
        print(f"{model_id}: UMAP complete", flush=True)
    combined = pd.concat(frames.values(), ignore_index=True)
    coordinates_path = output / "joint_umap_coordinates.csv"
    combined.to_csv(coordinates_path, index=False)
    figures.update(render_combined(frames, endpoints, output, args.dpi))
    figures.update(render_trajectories(frames, output, args.dpi))
    expected_rows = len(contract["models"]) * 109 * len(CONDITIONS) * len(SCANNERS)
    summary = {
        "analysis": "e5_rf1_joint_pre_post_umap",
        "interpretation": "descriptive visualization; inferential conclusions use the locked original feature space",
        "models": [model["encoder_id"] for model in contract["models"]],
        "conditions": list(CONDITIONS),
        "scanners": list(SCANNERS),
        "slides": 109,
        "centroid_definition": "mean of patch-L2-normalized embeddings, then centroid L2 normalization",
        "joint_fit_definition": "raw all scanners plus corrected source scanners; corrected AT2 reuses the exact raw-AT2 coordinate",
        "pca_components": PCA_COMPONENTS,
        "pca_explained_variance": pca_variance,
        "umap_neighbors": UMAP_NEIGHBORS,
        "umap_min_dist": UMAP_MIN_DIST,
        "umap_metric": "cosine",
        "seed": SEED,
        "coordinate_rows": len(combined),
        "coordinates_sha256": sha256(coordinates_path),
        "feature_audit_sha256": {name: sha256(path) for name, path in audit_paths.items()},
        "endpoint_sha256": {
            "e5": sha256(Path(args.e5_endpoints)),
            "macenko": sha256(Path(args.macenko_endpoints)),
            "rf1": sha256(Path(args.rf1_endpoints)),
        },
        "figures": figures,
        "umap_gate_pass": bool(
            len(combined) == expected_rows
            and np.isfinite(combined[["umap_x", "umap_y"]]).all().all()
            and len(figures) == 12
            and all(value["bytes"] > 10_000 for value in figures.values())
        ),
    }
    (output / "joint_umap_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["umap_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
