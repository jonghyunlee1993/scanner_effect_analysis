"""Render a structured, reusable HTML report from completed evaluation artifacts."""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from utils.config import load_config


TEMPLATE = Path(__file__).resolve().parent / "templates" / "evaluation_report.html"
REPORT_SCHEMA_VERSION = 2
MIXED_RETRIEVAL_SCHEMA = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.ckpt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--external-dir", type=Path)
    parser.add_argument("--external-store", type=Path)
    parser.add_argument(
        "--validation-samples-dir", type=Path,
        help="Optional checkpoint-specific validation panels for report section 05.",
    )
    return parser.parse_args()


def recover_metrics(eval_dir):
    metrics_path = eval_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    report = eval_dir / "report.html"
    metrics = {}
    if report.exists():
        for key, value in re.findall(r"<tr><th>([^<]+)</th><td>([^<]+)</td></tr>", report.read_text()):
            try:
                metrics[key] = float(value)
            except ValueError:
                pass
    if not metrics:
        raise FileNotFoundError("No metrics.json or recoverable report.html was found")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def table(rows, headers=("Item", "Value")):
    frame = pd.DataFrame(rows, columns=headers)
    return Markup(frame.to_html(index=False, border=0, escape=True))


def config_mapping(value):
    """Convert resolved dict-like or SimpleNamespace config nodes to a mapping."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return dict(value)


def image_data_uri(path):
    """Embed an image so report.html remains portable as a single file."""
    path = Path(path)
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def refresh_grouped_scanner_probe(eval_dir, metrics, workers):
    """Replace legacy patch-wise accuracy with grouped balanced accuracy."""
    method = "5-fold StratifiedGroupKFold(location), balanced accuracy"
    if metrics.get("scanner_probe_method") == method:
        return metrics
    embeddings = np.load(eval_dir / "embeddings.npz")
    scanner_id = embeddings["scanner_id"]
    location_id = embeddings["location_id"]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1234)
    for representation in ("raw", "canonical"):
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
        scores = cross_val_score(
            classifier, embeddings[representation], scanner_id,
            groups=location_id, cv=splitter, scoring="balanced_accuracy",
            n_jobs=min(int(workers), 5),
        )
        metrics[f"{representation}_scanner_probe"] = float(scores.mean())
        metrics[f"{representation}_scanner_probe_std"] = float(scores.std(ddof=1))
    metrics["scanner_probe_method"] = method
    (eval_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    return metrics


def loss_curve_svg(history, width=900, height=320):
    if history.empty:
        return Markup("<p>Training history unavailable.</p>")
    columns = [column for column in ("train_loss", "val_loss") if column in history]
    values = history[columns].to_numpy(float)
    finite = values[np.isfinite(values)]
    left, right, top, bottom = 65, 20, 24, 42
    y_min, y_max = float(finite.min()), float(finite.max())
    pad = max((y_max - y_min) * .06, 1e-6)
    y_min, y_max = y_min - pad, y_max + pad
    epochs = history.epoch.to_numpy(float)
    x_min, x_max = float(epochs.min()), max(float(epochs.max()), float(epochs.min()) + 1)
    def point(epoch, value):
        x = left + (epoch-x_min)/(x_max-x_min)*(width-left-right)
        y = top + (y_max-value)/(y_max-y_min)*(height-top-bottom)
        return f"{x:.1f},{y:.1f}"
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Loss by epoch'>"]
    for fraction in np.linspace(0, 1, 5):
        y = top + fraction*(height-top-bottom); value = y_max-fraction*(y_max-y_min)
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' stroke='#dce3ec'/><text x='4' y='{y+4:.1f}' fill='#687386'>{value:.3g}</text>")
    for index, (column, color, label) in enumerate((("train_loss", "#2156d7", "train"), ("val_loss", "#c2414b", "validation"))):
        if column not in history: continue
        selected = history[["epoch", column]].dropna().to_numpy()
        points = " ".join(point(epoch, value) for epoch, value in selected)
        parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.5' points='{points}'/><circle cx='{92+index*120}' cy='14' r='5' fill='{color}'/><text x='{103+index*120}' y='18' fill='{color}'>{label}</text>")
    parts.append(f"<text x='{width/2}' y='{height-7}' text-anchor='middle' fill='#687386'>epoch</text></svg>")
    return Markup("".join(parts))


def umap_plot(eval_dir, scanners, workers):
    cache = eval_dir / "umap_coordinates.npz"
    embeddings = np.load(eval_dir / "embeddings.npz")
    scanner_id = embeddings["scanner_id"]
    if cache.exists():
        coordinates = np.load(cache)
        raw_xy, canonical_xy = coordinates["raw"], coordinates["canonical"]
    else:
        import umap
        joined = np.concatenate([embeddings["raw"], embeddings["canonical"]], axis=0)
        reduced = PCA(
            n_components=min(50, joined.shape[0], joined.shape[1]),
            svd_solver="randomized",
            random_state=1234,
        ).fit_transform(joined)
        xy = umap.UMAP(n_neighbors=30, min_dist=.15, metric="cosine", random_state=1234, n_jobs=workers).fit_transform(reduced)
        raw_xy, canonical_xy = np.split(xy, 2)
        np.savez_compressed(cache, raw=raw_xy, canonical=canonical_xy, scanner_id=scanner_id)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#d97706"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, sharex=True, sharey=True)
    for ax, xy, title in zip(axes, (raw_xy, canonical_xy), ("Raw UNI", "Canonical UNI")):
        for index, scanner in enumerate(scanners):
            selected = scanner_id == index
            ax.scatter(xy[selected, 0], xy[selected, 1], s=3, alpha=.34, color=colors[index], label=scanner, rasterized=True)
        ax.set_title(title, loc="left", weight="bold"); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
    axes[1].legend(frameon=False, markerscale=3, ncol=2, loc="best")
    fig.patch.set_facecolor("white"); fig.tight_layout()
    buffer = io.BytesIO(); fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def nearest_neighbor_plot(eval_dir, cfg, checkpoint, frame, scanners, k=5):
    """Render scanner-stratified raw/canonical UNI retrievals with image thumbnails."""
    cached_plot = eval_dir / "nearest_neighbors.png"
    if cached_plot.exists():
        return image_data_uri(cached_plot)

    import torch
    from prenorm.checkpoint import load_model_for_inference
    from prenorm.data.dataset import PairedTupleDataset

    if not torch.cuda.is_available():
        raise RuntimeError(
            "nearest-neighbor thumbnail cache is absent; render once in a GPU allocation"
        )
    embedding_data = np.load(eval_dir / "embeddings.npz")
    raw_embedding = embedding_data["raw"]
    canonical_embedding = embedding_data["canonical"]
    scanner_id = embedding_data["scanner_id"]
    location_id = embedding_data["location_id"]
    random = np.random.default_rng(1234)
    queries = np.asarray([
        random.choice(np.flatnonzero(scanner_id == index))
        for index in range(len(scanners))
    ])

    def neighbors(embedding):
        similarity = embedding[queries] @ embedding.T
        similarity[np.arange(len(queries)), queries] = -np.inf
        candidates = np.argpartition(similarity, -k, axis=1)[:, -k:]
        order = np.argsort(
            np.take_along_axis(similarity, candidates, axis=1), axis=1
        )[:, ::-1]
        indices = np.take_along_axis(candidates, order, axis=1)
        scores = np.take_along_axis(similarity, indices, axis=1)
        return indices, scores

    raw_neighbors, raw_scores = neighbors(raw_embedding)
    canonical_neighbors, canonical_scores = neighbors(canonical_embedding)
    all_indices = np.unique(np.concatenate([
        queries, raw_neighbors.ravel(), canonical_neighbors.ravel()
    ]))
    dataset = PairedTupleDataset(
        cfg.paths.output_index, cfg.paths.output_store, scanners, cfg, "test", train=False
    )
    scanner_lookup = {scanner: index for index, scanner in enumerate(scanners)}
    raw_tensors = {}
    for index in all_indices:
        row = frame.iloc[int(index)]
        raw_tensors[int(index)] = dataset[int(row.location)]["rgb"][
            scanner_lookup[row.scanner]
        ]
    device = torch.device("cuda")
    model = load_model_for_inference(checkpoint, cfg, map_location="cpu").to(device)
    canonical_tensors = {}
    ordered = list(map(int, all_indices))
    with torch.no_grad():
        for start in range(0, len(ordered), 16):
            selected = ordered[start:start + 16]
            batch = torch.stack([raw_tensors[index] for index in selected]).to(device)
            output = model.canonicalize(batch).cpu()
            canonical_tensors.update(zip(selected, output))

    def as_image(tensor):
        return ((tensor.clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)

    fig, axes = plt.subplots(len(queries) * 2, k + 1, figsize=(13, len(queries) * 4.1), dpi=135)
    for query_row, query in enumerate(queries):
        query = int(query)
        for representation, neighbor_array, score_array, tensors, row_offset in (
            ("Raw UNI", raw_neighbors, raw_scores, raw_tensors, 0),
            ("Canonical UNI", canonical_neighbors, canonical_scores, canonical_tensors, 1),
        ):
            row = query_row * 2 + row_offset
            query_meta = frame.iloc[query]
            ax = axes[row, 0]
            ax.imshow(as_image(tensors[query]))
            ax.set_title(f"{representation} query\n{query_meta.scanner} · loc {int(query_meta.location)}", fontsize=8, weight="bold")
            for spine in ax.spines.values(): spine.set_color("#2156d7"); spine.set_linewidth(3)
            for rank, (neighbor, score) in enumerate(zip(neighbor_array[query_row], score_array[query_row]), start=1):
                neighbor = int(neighbor); neighbor_meta = frame.iloc[neighbor]
                hit = int(neighbor_meta.location) == int(query_meta.location)
                ax = axes[row, rank]
                ax.imshow(as_image(tensors[neighbor]))
                ax.set_title(f"NN{rank} · {neighbor_meta.scanner}\nloc {int(neighbor_meta.location)} · cos {score:.3f}", fontsize=7)
                for spine in ax.spines.values():
                    spine.set_color("#14805e" if hit else "#c2414b"); spine.set_linewidth(3)
        for ax in axes[query_row * 2:query_row * 2 + 2].ravel():
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("UNI nearest-neighbor retrieval · blue=query, green=same location, red=different location", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, .985))
    fig.savefig(cached_plot, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    details = {
        "queries": queries.tolist(),
        "raw_neighbors": raw_neighbors.tolist(),
        "raw_scores": raw_scores.tolist(),
        "canonical_neighbors": canonical_neighbors.tolist(),
        "canonical_scores": canonical_scores.tolist(),
    }
    (eval_dir / "nearest_neighbors.json").write_text(json.dumps(details, indent=2) + "\n")
    return image_data_uri(cached_plot)


def _directional_metrics(query_embedding, gallery_embedding, query_location,
                         gallery_location, k=5, chunk_size=512):
    """Evaluate a fixed source→target gallery without allocating the full matrix."""
    target_lookup = {int(location): index for index, location in enumerate(gallery_location)}
    paired = np.asarray([target_lookup[int(location)] for location in query_location])
    rank1, rank5 = [], []
    for start in range(0, len(query_embedding), chunk_size):
        stop = min(start + chunk_size, len(query_embedding))
        similarity = query_embedding[start:stop] @ gallery_embedding.T
        neighbors = np.argpartition(similarity, -min(k, len(gallery_embedding)), axis=1)[:, -min(k, len(gallery_embedding)):]
        top_scores = np.take_along_axis(similarity, neighbors, axis=1)
        order = np.argsort(top_scores, axis=1)[:, ::-1]
        neighbors = np.take_along_axis(neighbors, order, axis=1)
        expected = paired[start:stop, None]
        rank1.append((neighbors[:, :1] == expected).any(axis=1))
        rank5.append((neighbors == expected).any(axis=1))
    paired_cosine = np.sum(query_embedding * gallery_embedding[paired], axis=1)
    return {
        "n_queries": int(len(query_embedding)),
        "r1": float(np.concatenate(rank1).mean()),
        "r5": float(np.concatenate(rank5).mean()),
        "paired_cosine": float(paired_cosine.mean()),
    }


def directional_retrieval(eval_dir, cfg, checkpoint, frame, scanners,
                          external_dir, external_store, k=5):
    """Compare AT2→target retrieval under matched raw and canonical conditions."""
    cached_plot = eval_dir / "directional_retrieval.png"
    cached_json = eval_dir / "directional_retrieval.json"
    if cached_plot.exists() and cached_json.exists():
        details = json.loads(cached_json.read_text())
        return image_data_uri(cached_plot), details["metrics"]

    import h5py
    import torch
    from prenorm.checkpoint import load_model_for_inference
    from prenorm.data.dataset import PairedTupleDataset
    from utils.store import decode

    if not torch.cuda.is_available():
        raise RuntimeError("directional retrieval thumbnails require one GPU render pass")
    internal = np.load(eval_dir / "embeddings.npz")
    external = np.load(Path(external_dir) / "uni_embeddings.npz")
    external_frame = pd.read_csv(Path(external_dir) / "per_tile.csv")
    scanner_id = internal["scanner_id"]
    location_id = internal["location_id"]
    scanner_lookup = {scanner: index for index, scanner in enumerate(scanners)}
    reference = cfg.reference_scanner
    reference_id = scanner_lookup[reference]
    target_scanners = [scanner for scanner in scanners if scanner != reference]
    metric_rows = []
    plot_specs = []

    # ID queries are restricted to locations that have both AT2 and the target.
    common_locations = None
    for target in target_scanners:
        target_id = scanner_lookup[target]
        query_indices = np.flatnonzero(scanner_id == reference_id)
        gallery_indices = np.flatnonzero(scanner_id == target_id)
        gallery_locations = set(map(int, location_id[gallery_indices]))
        eligible_query = np.asarray([
            index for index in query_indices if int(location_id[index]) in gallery_locations
        ], dtype=int)
        eligible_locations = set(map(int, location_id[eligible_query]))
        common_locations = eligible_locations if common_locations is None else common_locations & eligible_locations
        target_gallery_locations = location_id[gallery_indices]
        for representation in ("raw", "canonical"):
            result = _directional_metrics(
                internal[representation][eligible_query], internal[representation][gallery_indices],
                location_id[eligible_query], target_gallery_locations, k=k,
            )
            result.update({"target": target.upper(), "representation": representation, "cohort": "ID"})
            metric_rows.append(result)
        plot_specs.append({
            "target": target.upper(), "query_indices": eligible_query,
            "gallery_indices": gallery_indices, "cohort": "ID",
        })
    if not common_locations:
        raise RuntimeError("no ID AT2 location is shared by all target galleries")
    display_location = sorted(common_locations)[len(common_locations) // 2]

    # The external pair is a separate cohort, but follows the identical AT2→target protocol.
    external_location = np.arange(len(external_frame), dtype=int)
    for representation, query_key, gallery_key in (
        ("raw", "paired_at2", "raw_s60"),
        ("canonical", "standard_at2", "standard_s60"),
    ):
        result = _directional_metrics(
            external[query_key], external[gallery_key], external_location, external_location, k=k,
        )
        result.update({"target": "S60", "representation": representation, "cohort": "OOD"})
        metric_rows.append(result)
    external_query = len(external_frame) // 2
    plot_specs.append({
        "target": "S60", "query_indices": np.asarray([external_query]),
        "gallery_indices": np.arange(len(external_frame)), "cohort": "OOD",
    })

    # Select deterministic thumbnails, then canonicalize exactly those tensors.
    internal_dataset = PairedTupleDataset(
        cfg.paths.output_index, cfg.paths.output_store, scanners, cfg, "test", train=False
    )
    raw_images = {}
    row_records = []
    for spec in plot_specs:
        for representation in ("raw", "canonical"):
            if spec["cohort"] == "ID":
                query_candidates = spec["query_indices"]
                query = int(query_candidates[np.flatnonzero(
                    location_id[query_candidates] == display_location
                )[0]])
                gallery = spec["gallery_indices"]
                embeddings = internal[representation]
                similarity = embeddings[query] @ embeddings[gallery].T
                top = gallery[np.argsort(similarity)[-k:][::-1]]
                row_records.append((spec, representation, query, top, similarity[np.argsort(similarity)[-k:][::-1]]))
                for index in np.concatenate(([query], top)):
                    meta = frame.iloc[int(index)]
                    key = ("ID", int(index))
                    if key not in raw_images:
                        raw_images[key] = internal_dataset[int(meta.location)]["rgb"][scanner_lookup[meta.scanner]]
            else:
                query = external_query
                query_key = "paired_at2" if representation == "raw" else "standard_at2"
                gallery_key = "raw_s60" if representation == "raw" else "standard_s60"
                similarity = external[query_key][query] @ external[gallery_key].T
                top = np.argsort(similarity)[-k:][::-1]
                row_records.append((spec, representation, query, top, similarity[top]))
                needed = np.unique(np.concatenate(([query], top)))
                for index in needed:
                    row = external_frame.iloc[int(index)]
                    with h5py.File(Path(external_store) / f"{row.slide_id}.h5", "r") as handle:
                        if ("OOD-query", int(index)) not in raw_images:
                            raw_images[("OOD-query", int(index))] = torch.from_numpy(
                                decode(handle["at2"]["rgb"][int(row.tuple_id)]).copy()
                            ).permute(2, 0, 1).float() / 127.5 - 1
                        if ("OOD-target", int(index)) not in raw_images:
                            raw_images[("OOD-target", int(index))] = torch.from_numpy(
                                decode(handle["s60"]["rgb"][int(row.tuple_id)]).copy()
                            ).permute(2, 0, 1).float() / 127.5 - 1
    device = torch.device("cuda")
    model = load_model_for_inference(checkpoint, cfg, map_location="cpu").to(device).eval()
    canonical_images = {}
    keys = list(raw_images)
    with torch.no_grad():
        for start in range(0, len(keys), 16):
            selected = keys[start:start + 16]
            images = torch.stack([raw_images[key] for key in selected]).to(device)
            canonical = model.canonicalize(images).cpu()
            canonical_images.update(zip(selected, canonical))

    def image_array(tensor):
        return ((tensor.clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).round().astype(np.uint8)

    fig, axes = plt.subplots(len(row_records), k + 1, figsize=(14, 2.55 * len(row_records)), dpi=145)
    figure_rows = []
    for row_number, (spec, representation, query, neighbors, scores) in enumerate(row_records):
        tensors = raw_images if representation == "raw" else canonical_images
        if spec["cohort"] == "ID":
            query_key = ("ID", int(query))
            query_location = int(location_id[query])
            query_note = f"ID loc {query_location}"
        else:
            query_key = ("OOD-query", int(query))
            query_location = int(query)
            query_note = f"OOD pair {query}"
        axes[row_number, 0].imshow(image_array(tensors[query_key]))
        axes[row_number, 0].set_title(
            f"{representation.title()} AT2 query\n{query_note}", fontsize=8, weight="bold"
        )
        for spine in axes[row_number, 0].spines.values():
            spine.set_color("#2156d7"); spine.set_linewidth(2.7)
        neighbor_ids = []
        for rank, (neighbor, score) in enumerate(zip(neighbors, scores), start=1):
            neighbor = int(neighbor)
            if spec["cohort"] == "ID":
                target_key = ("ID", neighbor)
                hit = int(location_id[neighbor]) == query_location
                location_note = f"loc {int(location_id[neighbor])}"
            else:
                target_key = ("OOD-target", neighbor)
                hit = neighbor == query
                location_note = f"pair {neighbor}"
            axes[row_number, rank].imshow(image_array(tensors[target_key]))
            axes[row_number, rank].set_title(
                f"{spec['target']} NN{rank}\n{location_note} · {score:.3f}", fontsize=7.5
            )
            for spine in axes[row_number, rank].spines.values():
                spine.set_color("#14805e" if hit else "#c2414b"); spine.set_linewidth(2.7)
            neighbor_ids.append(neighbor)
        for ax in axes[row_number]:
            ax.set_xticks([]); ax.set_yticks([])
        figure_rows.append({
            "target": spec["target"], "cohort": spec["cohort"],
            "representation": representation, "query": int(query),
            "neighbors": neighbor_ids, "scores": [float(value) for value in scores],
        })
    fig.suptitle(
        "Directional UNI retrieval · AT2 query → target-specific gallery\n"
        "blue=query · green=exact registered pair · red=other location",
        fontsize=13, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(cached_plot, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    details = {
        "protocol": "AT2 query to one target-scanner gallery; raw-to-raw and canonical-to-canonical",
        "id_display_location": int(display_location),
        "ood_display_pair_index": int(external_query),
        "metrics": metric_rows,
        "figure_rows": figure_rows,
    }
    cached_json.write_text(json.dumps(details, indent=2) + "\n")
    return image_data_uri(cached_plot), metric_rows


def _mixed_gallery_metrics(embedding, location_id, scanner_id, k=5, chunk_size=512):
    """Measure semantic recovery and scanner bias in one self-excluded gallery."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectors = torch.as_tensor(embedding, device=device)
    locations = torch.as_tensor(location_id, device=device, dtype=torch.long)
    scanners = torch.as_tensor(scanner_id, device=device, dtype=torch.long)
    location_counts = torch.bincount(locations)
    accumulators = {name: [] for name in (
        "same_location_top1", "same_location_hit_at5", "counterpart_recall_at5",
        "all_counterparts_at5", "same_scanner_top1", "same_scanner_purity_at5",
        "all5_same_scanner",
    )}
    for start in range(0, len(vectors), chunk_size):
        stop = min(start + chunk_size, len(vectors))
        similarity = vectors[start:stop] @ vectors.T
        local_rows = torch.arange(stop - start, device=device)
        similarity[local_rows, torch.arange(start, stop, device=device)] = -torch.inf
        neighbors = similarity.topk(min(k, len(vectors) - 1), dim=1).indices
        same_location = locations[neighbors] == locations[start:stop, None]
        same_scanner = scanners[neighbors] == scanners[start:stop, None]
        counterpart_count = location_counts[locations[start:stop]] - 1
        recovered = same_location.sum(1)
        eligible = counterpart_count > 0
        accumulators["same_location_top1"].append(same_location[:, 0].float().cpu())
        accumulators["same_location_hit_at5"].append(same_location.any(1).float().cpu())
        accumulators["counterpart_recall_at5"].append(
            (recovered[eligible].float() / counterpart_count[eligible].float()).cpu()
        )
        accumulators["all_counterparts_at5"].append(
            (recovered[eligible] == counterpart_count[eligible]).float().cpu()
        )
        accumulators["same_scanner_top1"].append(same_scanner[:, 0].float().cpu())
        accumulators["same_scanner_purity_at5"].append(same_scanner.float().mean(1).cpu())
        accumulators["all5_same_scanner"].append(same_scanner.all(1).float().cpu())
    return {
        "n_queries": int(len(vectors)),
        **{name: float(torch.cat(values).mean()) for name, values in accumulators.items()},
    }


def _mixed_neighbors(embedding, queries, k=5):
    similarity = embedding[queries] @ embedding.T
    similarity[np.arange(len(queries)), queries] = -np.inf
    neighbors = np.argpartition(similarity, -k, axis=1)[:, -k:]
    scores = np.take_along_axis(similarity, neighbors, axis=1)
    order = np.argsort(scores, axis=1)[:, ::-1]
    return (
        np.take_along_axis(neighbors, order, axis=1),
        np.take_along_axis(scores, order, axis=1),
    )


def mixed_scanner_retrieval(eval_dir, cfg, checkpoint, frame, scanners,
                            external_dir=None, external_store=None, k=5):
    """Audit whether mixed galleries retrieve registered semantics or scanner style."""
    cache_json = eval_dir / "mixed_scanner_neighbors.json"
    id_plot = eval_dir / "mixed_scanner_neighbors_id.png"
    ood_plot = eval_dir / "mixed_scanner_neighbors_s60.png"
    with_external = external_dir is not None
    if with_external and external_store is None:
        raise ValueError("external_store is required when external_dir is provided")
    cache_complete = cache_json.exists() and id_plot.exists() and (
        not with_external or ood_plot.exists()
    )
    if cache_complete:
        details = json.loads(cache_json.read_text())
        valid_cache = (
            details.get("schema_version") == MIXED_RETRIEVAL_SCHEMA
            and details.get("id_scanners") == list(scanners)
            and (not with_external or "ood_caption" in details)
        )
        if valid_cache:
            panels = [
                {"src": image_data_uri(id_plot),
                 "label": f"ID · {len(scanners)}-scanner mixed gallery",
                 "caption": details["id_caption"]},
            ]
            metrics = [row for row in details["metrics"] if row["cohort"].startswith("ID")]
            if with_external:
                panels.append({
                    "src": image_data_uri(ood_plot),
                    "label": "OOD · paired AT2/S60 mixed gallery",
                    "caption": details["ood_caption"],
                })
                metrics = details["metrics"]
            return panels, metrics

    import h5py
    import torch
    from prenorm.checkpoint import load_model_for_inference
    from prenorm.data.dataset import PairedTupleDataset
    from utils.store import decode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    internal = np.load(eval_dir / "embeddings.npz")
    internal_location = internal["location_id"].astype(int)
    internal_scanner = internal["scanner_id"].astype(int)
    id_cohort = f"ID {len(scanners)}-scanner"
    if with_external:
        external = np.load(Path(external_dir) / "uni_embeddings.npz")
        external_frame = pd.read_csv(Path(external_dir) / "per_tile.csv")
        external_count = len(external_frame)
        external_location = np.tile(np.arange(external_count, dtype=int), 2)
        external_scanner = np.repeat(np.arange(2, dtype=int), external_count)
        standard_at2 = (
            external["standard_at2"]
            if "standard_at2" in external.files else external["paired_at2"]
        )
        external_embeddings = {
            "raw": np.concatenate((external["paired_at2"], external["raw_s60"])),
            "canonical": np.concatenate((standard_at2, external["standard_s60"])),
        }
    metrics = []
    for representation in ("raw", "canonical"):
        result = _mixed_gallery_metrics(
            internal[representation], internal_location, internal_scanner, k=k
        )
        result.update({"cohort": id_cohort, "representation": representation})
        metrics.append(result)
        if with_external:
            result = _mixed_gallery_metrics(
                external_embeddings[representation], external_location, external_scanner, k=k
            )
            result.update({"cohort": "OOD AT2–S60", "representation": representation})
            metrics.append(result)

    # Two complete ID locations × every scanner present in the experiment.
    location_counts = np.bincount(internal_location)
    complete_locations = np.flatnonzero(location_counts == len(scanners))
    if len(complete_locations) < 2:
        raise RuntimeError("two complete ID locations are required for the mixed-gallery panel")
    selected_locations = complete_locations[
        np.rint(np.linspace(.33, .67, 2) * (len(complete_locations) - 1)).astype(int)
    ]
    id_queries = np.asarray([
        np.flatnonzero((internal_location == location) & (internal_scanner == scanner_id))[0]
        for location in selected_locations for scanner_id in range(len(scanners))
    ], dtype=int)
    neighbor_data = {
        "ID": {
            representation: _mixed_neighbors(internal[representation], id_queries, k=k)
            for representation in ("raw", "canonical")
        },
    }
    if with_external:
        # Three external pairs × both query scanners.
        selected_pairs = np.rint(
            np.linspace(.25, .75, 3) * (external_count - 1)
        ).astype(int)
        ood_queries = np.asarray([
            pair + scanner_id * external_count
            for pair in selected_pairs for scanner_id in range(2)
        ], dtype=int)
        neighbor_data["OOD"] = {
            representation: _mixed_neighbors(external_embeddings[representation], ood_queries, k=k)
            for representation in ("raw", "canonical")
        }

    internal_dataset = PairedTupleDataset(
        cfg.paths.output_index, cfg.paths.output_store, scanners, cfg, "test", train=False
    )
    scanner_lookup = {scanner: index for index, scanner in enumerate(scanners)}
    raw_images = {}
    internal_needed = np.unique(np.concatenate([
        id_queries,
        *(neighbor_data["ID"][representation][0].ravel()
          for representation in ("raw", "canonical")),
    ]))
    for index in internal_needed:
        meta = frame.iloc[int(index)]
        raw_images[("ID", int(index))] = internal_dataset[int(meta.location)]["rgb"][
            scanner_lookup[meta.scanner]
        ]
    if with_external:
        ood_needed = np.unique(np.concatenate([
            ood_queries,
            *(neighbor_data["OOD"][representation][0].ravel()
              for representation in ("raw", "canonical")),
        ]))
        for index in ood_needed:
            index = int(index)
            scanner_id = index // external_count
            pair_index = index % external_count
            meta = external_frame.iloc[pair_index]
            scanner = "at2" if scanner_id == 0 else "s60"
            with h5py.File(Path(external_store) / f"{meta.slide_id}.h5", "r") as handle:
                image = decode(handle[scanner]["rgb"][int(meta.tuple_id)])
            raw_images[("OOD", index)] = (
                torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 127.5 - 1
            )

    model = load_model_for_inference(checkpoint, cfg, map_location="cpu").to(device).eval()
    canonical_images = {}
    image_keys = list(raw_images)
    with torch.no_grad():
        for start in range(0, len(image_keys), 16):
            selected = image_keys[start:start + 16]
            images = torch.stack([raw_images[key] for key in selected]).to(device)
            outputs = model.canonicalize(images).cpu()
            canonical_images.update(zip(selected, outputs))

    def as_image(tensor):
        return ((tensor.clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).round().astype(np.uint8)

    def render_panel(cohort, queries, embeddings, locations, scanner_ids,
                     scanner_names, path, title):
        fig, axes = plt.subplots(len(queries), 2 * (k + 1), figsize=(25, 2.6 * len(queries)), dpi=125)
        records = []
        for row, query in enumerate(queries):
            record = {"query": int(query), "raw": {}, "canonical": {}}
            for block, representation in enumerate(("raw", "canonical")):
                neighbors, scores = neighbor_data[cohort][representation]
                tensors = raw_images if representation == "raw" else canonical_images
                offset = block * (k + 1)
                axes[row, offset].imshow(as_image(tensors[(cohort, int(query))]))
                axes[row, offset].set_title(
                    f"{representation.title()} query · {scanner_names[scanner_ids[query]]}\n"
                    f"location {int(locations[query])}", fontsize=8, weight="bold"
                )
                for spine in axes[row, offset].spines.values():
                    spine.set_color("#2156d7"); spine.set_linewidth(2.7)
                record[representation] = {"neighbors": [], "scores": []}
                for rank, (neighbor, score) in enumerate(zip(neighbors[row], scores[row]), start=1):
                    neighbor = int(neighbor)
                    same_location = int(locations[neighbor]) == int(locations[query])
                    same_scanner = int(scanner_ids[neighbor]) == int(scanner_ids[query])
                    color = "#14805e" if same_location else ("#d97706" if same_scanner else "#c2414b")
                    ax = axes[row, offset + rank]
                    ax.imshow(as_image(tensors[(cohort, neighbor)]))
                    ax.set_title(
                        f"NN{rank} · {scanner_names[scanner_ids[neighbor]]}\n"
                        f"loc {int(locations[neighbor])} · {score:.3f}", fontsize=7.5
                    )
                    for spine in ax.spines.values():
                        spine.set_color(color); spine.set_linewidth(2.7)
                    record[representation]["neighbors"].append(neighbor)
                    record[representation]["scores"].append(float(score))
            for ax in axes[row]:
                ax.set_xticks([]); ax.set_yticks([])
            records.append(record)
        fig.suptitle(
            title + "\nblue=query · green=same registered location · orange=same scanner · red=neither",
            fontsize=13, weight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(path, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return records

    id_records = render_panel(
        "ID", id_queries, internal, internal_location, internal_scanner,
        [scanner.upper() for scanner in scanners], id_plot,
        "ID mixed-scanner UNI retrieval · self excluded",
    )
    details = {
        "schema_version": MIXED_RETRIEVAL_SCHEMA,
        "id_scanners": list(scanners),
        "protocol": "single mixed-scanner gallery per cohort; only the query itself is excluded",
        "metrics": metrics,
        "id_caption": (
            f"Two deterministic complete registered locations, queried once from each of "
            f"{', '.join(scanner.upper() for scanner in scanners)}; raw and canonical top-5 "
            "are shown side by side."
        ),
        "id_queries": id_records,
    }
    if with_external:
        ood_records = render_panel(
            "OOD", ood_queries, external_embeddings, external_location, external_scanner,
            ["paired AT2", "S60"], ood_plot,
            "OOD paired AT2/S60 mixed-gallery UNI retrieval · self excluded",
        )
        details.update({
            "ood_caption": (
                "Three deterministic external registered pairs, queried once from paired AT2 and "
                "once from S60; raw and canonical top-5 are shown side by side."
            ),
            "ood_queries": ood_records,
        })
    cache_json.write_text(json.dumps(details, indent=2) + "\n")
    panels = [{
        "src": image_data_uri(id_plot),
        "label": f"ID · {len(scanners)}-scanner mixed gallery",
        "caption": details["id_caption"],
    }]
    if with_external:
        panels.append({
            "src": image_data_uri(ood_plot),
            "label": "OOD · paired AT2/S60 mixed gallery",
            "caption": details["ood_caption"],
        })
    return panels, metrics


def _save_input_channel_plot(tensors, row_labels, builder, path, title):
    """Render the RGB composite, individual RGB planes, and all derived inputs."""
    import torch

    with torch.no_grad():
        channels = builder(tensors, mode="full")
    channel_index = {name: offset for offset, name in enumerate(builder.channel_names)}
    columns = (
        ("RGB", None), ("R", "r"), ("G", "g"), ("B", "b"),
        ("Edge", "edge_lp"), ("Gray", "gray"), ("H", "h"),
        ("E", "e"), ("Background", "background"),
    )
    fig, axes = plt.subplots(
        len(row_labels), len(columns),
        figsize=(2.05 * len(columns), 2.12 * len(row_labels)), dpi=145,
        squeeze=False,
    )
    for row, label in enumerate(row_labels):
        rgb = ((tensors[row].clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * .5)
        for column, (column_title, channel) in enumerate(columns):
            ax = axes[row, column]
            if channel is None:
                ax.imshow(rgb)
            else:
                image = channels[row, channel_index[channel]].cpu().numpy()
                if channel in {"r", "g", "b"}:
                    image = (image + 1) * .5
                ax.imshow(image, cmap="magma" if channel == "edge_lp" else "gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(column_title, fontsize=9, weight="bold")
            for spine in ax.spines.values():
                spine.set_color("#9eb0c8")
        axes[row, 0].set_ylabel(
            label, rotation=0, ha="right", va="center", labelpad=38,
            fontsize=9, weight="bold",
        )
    fig.suptitle(title, fontsize=12, weight="bold")
    fig.tight_layout(rect=(.035, 0, 1, .96))
    fig.savefig(path, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def input_channel_panels(eval_dir, cfg, scanners, external_dir=None, external_store=None):
    """Return explicit ID and OOD input panels without implying a five-way tuple."""
    import h5py
    import torch
    from prenorm.data.dataset import PairedTupleDataset
    from prenorm.models.input_builder import DifferentiableInputBuilder
    from utils.store import decode

    builder = DifferentiableInputBuilder(
        cfg.paths.stain_reference, mode="full", edge_sigma=cfg.input.edge_sigma,
        group_dropout_p=cfg.input.group_dropout_p,
    )
    dataset = PairedTupleDataset(
        cfg.paths.output_index, cfg.paths.output_store, scanners, cfg, "test", train=False
    )
    candidates = []
    for index, row in enumerate(dataset.rows):
        if all(bool(row[f"present_{scanner}"]) for scanner in scanners):
            minimum_q = min(float(row[f"q_reg_{scanner}"]) for scanner in scanners)
            if minimum_q >= max(float(cfg.registration.q_min), .8):
                candidates.append((float(row["tissue_density"]), minimum_q, index))
    if not candidates:
        raise RuntimeError("test split has no location present in every ID scanner")
    candidates.sort()
    index = candidates[round(.75 * (len(candidates) - 1))][2]
    item = dataset[index]
    id_plot = eval_dir / "input_channels_id.png"
    if not id_plot.exists():
        _save_input_channel_plot(
            item["rgb"], [scanner.upper() for scanner in scanners], builder, id_plot,
            "ID registered tuple · model input channels",
        )
    panels = [{
        "src": image_data_uri(id_plot),
        "label": f"In-domain registered {len(scanners)}-scanner tuple",
        "caption": (
            f"test · slide {item['slide_id']} · tile {item['tile_id']} · "
            f"x={item['x']}, y={item['y']}; tissue-density 75th-percentile "
            "selection among complete tuples with q_reg ≥ 0.8"
        ),
    }]
    if external_dir is None:
        return panels
    if external_store is None:
        raise ValueError("external_store is required for the S60 input panel")
    external_frame = pd.read_csv(Path(external_dir) / "per_tile.csv")
    median_focus = external_frame["raw_s60_focus"].median()
    external_index = int((external_frame["raw_s60_focus"] - median_focus).abs().idxmin())
    row = external_frame.iloc[external_index]
    with h5py.File(Path(external_store) / f"{row.slide_id}.h5", "r") as handle:
        s60 = decode(handle["s60"]["rgb"][int(row.tuple_id)])
        at2 = decode(handle["at2"]["rgb"][int(row.tuple_id)])
    to_tensor = lambda image: torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 127.5 - 1
    ood_tensors = torch.stack((to_tensor(at2), to_tensor(s60)))
    ood_plot = eval_dir / "input_channels_s60.png"
    if not ood_plot.exists():
        _save_input_channel_plot(
            ood_tensors, ["paired AT2", "S60"], builder, ood_plot,
            "OOD registered pair · model input channels",
        )
    panels.append({
        "src": image_data_uri(ood_plot),
        "label": "Out-of-domain registered AT2–S60 pair",
        "caption": (
            f"external · slide {row.slide_id} · tuple {int(row.tuple_id)}; "
            "deterministic median-S60-focus example"
        ),
    })
    return panels


def schedule_rows(cfg):
    epochs = int(cfg.trainer.max_epochs)
    marks = [0, cfg.schedule.bootstrap_end*epochs, cfg.schedule.identification_end*epochs,
             cfg.schedule.hardening_end*epochs, epochs]
    return [
        (f"{marks[0]:g}–{marks[1]:g}", "Bootstrap", f"pair={cfg.loss.pair_start:g}; cross/nuisance/gradient=0"),
        (f"{marks[1]:g}–{marks[2]:g}", "Identification", f"pair {cfg.loss.pair_start:g}→{cfg.loss.pair_end:g}; cross 0→1"),
        (f"{marks[2]:g}–{marks[3]:g}", "Hardening", f"pair={cfg.loss.pair_end:g}; nuisance 0→1; gradient 0→{cfg.loss.gradient_end:g}"),
        (f"{marks[3]:g}–{marks[4]:g}", "Final", f"pair={cfg.loss.pair_end:g}; cross/nuisance=1; gradient={cfg.loss.gradient_end:g}"),
    ]


def objective_formula(cfg):
    """Render only the loss terms active in the resolved experiment config."""
    loss = cfg.loss
    terms = []
    if float(loss.standard):
        terms.append(f"{float(loss.standard):g} L<sub>std</sub>")
    if float(loss.translate):
        terms.append(
            f"{float(loss.translate):g} [L<sub>self</sub> + c(t)L<sub>cross</sub>]"
        )
    if float(loss.pair_start) or float(loss.pair_end):
        terms.append("p(t)L<sub>pair</sub>")
    optional_terms = (
        ("detail", "L<sub>detail</sub>", ""),
        ("at2_identity_detail", "L<sub>AT2-id</sub>", ""),
        ("paired_target_detail", "L<sub>paired-detail</sub>", "r<sub>g</sub>(t)"),
        ("nuclei_rgb_detail", "L<sub>nuc-RGB</sub>", "c(t)"),
        ("neighborhood_consistency", "L<sub>neighbor</sub>", "n(t)"),
        ("nuclei_reconstruction", "L<sub>nuc-recon</sub>", "c(t)"),
        ("variance", "L<sub>var</sub>", ""),
        ("style", "L<sub>style</sub>", ""),
    )
    for key, label, ramp in optional_terms:
        weight = float(getattr(loss, key, 0.0))
        if weight:
            terms.append(f"{weight:g}{ramp}{label}")
    pair_weights = config_mapping(getattr(loss, "pair_scanner_weights", {}))
    pair_note = ""
    if pair_weights:
        factors = ", ".join(
            f"{scanner.upper()}={float(weight):g}"
            for scanner, weight in sorted(pair_weights.items())
        )
        pair_note = f" Pair endpoint factors: {factors}."
    return Markup(
        "<div class='formula'><span class='formula-main'>L(t) = "
        + (" + ".join(terms) if terms else "0")
        + "</span><div class='formula-note'>"
        + f"p(t): {float(loss.pair_start):g}→{float(loss.pair_end):g}; "
        + "c(t): 0→1. n(t): 0→1 scales scanner-independent content-input nuisance augmentation. "
        + "The translation distance internally ramps its gradient term "
        + f"g(t): 0→{float(loss.gradient_end):g}; r_g(t) is its 0→1 normalized ramp."
        + pair_note
        + "</div></div>"
    )


def objective_roles(cfg):
    loss = cfg.loss
    rows = []
    if float(loss.standard):
        rows.append(("L_std", "Canonical output → registered AT2", "Main color/canvas anchor; registration-weighted"))
    if float(loss.translate):
        rows += [
            ("L_self", "Rendered source → observed source", "Keeps content sufficient for reconstruction"),
            ("L_cross", "Rendered target → registered target", "Learns scanner-conditioned translation; c(t) ramped"),
        ]
    if float(loss.pair_start) or float(loss.pair_end):
        rows.append(("L_pair", "Content maps at the same registered coordinate", "Suppresses scanner-specific content; p(t) ramped"))
    if float(getattr(loss, "detail", 0.0)):
        rows.append(("L_detail", "Canonical foreground detail → source/AT2 detail", "Preserves foreground structure"))
    if float(getattr(loss, "at2_identity_detail", 0.0)):
        rows.append(("L_AT2-id", "Canonical AT2 detail → observed AT2 detail", "Protects exact reference-scanner identity"))
    if float(getattr(loss, "paired_target_detail", 0.0)):
        rows.append(("L_paired-detail", "Non-AT2 canonical detail → registered AT2 detail", "High-q target detail; gradient-ramped"))
    if float(getattr(loss, "nuclei_rgb_detail", 0.0)):
        rows.append((
            "L_nuc-RGB", "Canonical vs source RGB detail inside dilated StarDist nuclei",
            "Preserves nuclear Sobel/LoG detail without forcing whole-image source color",
        ))
    if float(getattr(loss, "neighborhood_consistency", 0.0)):
        rows.append((
            "L_neighbor", "Non-AT2 content neighborhood → detached AT2 neighborhood",
            "Aligns same-slide relational geometry; hardening-ramped",
        ))
    if float(getattr(loss, "nuclei_reconstruction", 0.0)):
        rows.append((
            "L_nuc-recon", "Auxiliary nuclei occupancy/boundary → StarDist labels",
            "Makes content explicitly retain nuclear morphology; c(t) ramped",
        ))
    if float(loss.variance):
        rows.append(("L_var", "Coordinate-wise content variance floor", "Discourages collapsed content features"))
    if float(loss.style):
        rows.append(("L_style", "E_set style code → scanner prototype", "Makes the renderer's style control identifiable"))
    return rows


def schedule_diagram_svg(cfg, width=940, height=330):
    """Show the exact piecewise schedule as four aligned lanes."""
    epochs = float(cfg.trainer.max_epochs)
    bootstrap = float(cfg.schedule.bootstrap_end) * epochs
    identification = float(cfg.schedule.identification_end) * epochs
    hardening = float(cfg.schedule.hardening_end) * epochs
    marks = (0.0, bootstrap, identification, hardening, epochs)
    left, right, top, bottom = 118, 26, 38, 42
    lane_height, lane_gap = 43, 17
    plot_width = width - left - right
    x = lambda epoch: left + epoch / epochs * plot_width
    phase_colors = ("#e8eef8", "#dbeafe", "#dcfce7", "#fef3c7")
    phase_names = ("bootstrap", "identification", "hardening", "final")
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Training schedule diagram'>"]
    for index, name in enumerate(phase_names):
        x0, x1 = x(marks[index]), x(marks[index + 1])
        parts.append(
            f"<rect x='{x0:.1f}' y='{top-22}' width='{x1-x0:.1f}' height='{height-top-bottom+24}' "
            f"fill='{phase_colors[index]}' opacity='.65'/><text x='{(x0+x1)/2:.1f}' y='19' "
            f"text-anchor='middle' fill='#46546a' font-size='12'>{name}</text>"
        )
    lane_specs = (
        ("pair p(t)", "#2156d7", (0, 0, 1, 1, 1), f"{cfg.loss.pair_start:g} → {cfg.loss.pair_end:g}"),
        ("cross c(t)", "#14805e", (0, 0, 1, 1, 1), "0 → 1"),
        ("nuisance n(t)", "#d97706", (0, 0, 0, 1, 1), "0 → 1"),
        ("gradient g(t)", "#c2414b", (0, 0, 0, 1, 1), f"0 → {cfg.loss.gradient_end:g}"),
    )
    for lane, (label, color, values, value_label) in enumerate(lane_specs):
        y0 = top + lane * (lane_height + lane_gap)
        y_low, y_high = y0 + lane_height, y0 + 5
        points = " ".join(
            f"{x(mark):.1f},{(y_low - value * (y_low-y_high)):.1f}"
            for mark, value in zip(marks, values)
        )
        parts.append(
            f"<text x='{left-10}' y='{y0+25}' text-anchor='end' fill='#18212f' font-size='12'>{label}</text>"
            f"<line x1='{left}' y1='{y_low}' x2='{width-right}' y2='{y_low}' stroke='#b8c3d1'/>"
            f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='3'/>"
            f"<text x='{width-right-3}' y='{y0+14}' text-anchor='end' fill='{color}' font-size='11'>{value_label}</text>"
        )
    for mark in marks:
        parts.append(
            f"<line x1='{x(mark):.1f}' y1='{top-22}' x2='{x(mark):.1f}' y2='{height-bottom+2}' "
            "stroke='#94a3b8' stroke-dasharray='3 4'/><text "
            f"x='{x(mark):.1f}' y='{height-10}' text-anchor='middle' fill='#46546a'>{mark:g}</text>"
        )
    parts.append(
        f"<text x='{width/2:.1f}' y='{height-1}' text-anchor='middle' fill='#687386' font-size='11'>epoch</text></svg>"
    )
    return Markup("".join(parts))


def external_report_context(eval_dir, external_dir, external_store, workers):
    """Build OOD S60 blocks that are merged into the common audit sections."""
    from render_external_s60_report import joint_umap

    external_dir = Path(external_dir)
    metrics = json.loads((external_dir / "metrics.json").read_text())
    frame = pd.read_csv(external_dir / "per_tile.csv")
    joint_umap(eval_dir, external_dir, eval_dir, workers)
    recon_prefix = "recon_s60" if "recon_s60_focus" in metrics else "recon_at2"
    recon_label = "S60 context recon" if recon_prefix == "recon_s60" else "Legacy AT2 render"
    image_rows = [
        ("Distance to paired AT2 ↓", metrics["raw_s60_distance_to_at2"],
         metrics["standard_distance_to_at2"], metrics[f"{recon_prefix}_distance_to_at2"]),
        ("SSIM to paired AT2 ↑", metrics["raw_s60_ssim_to_at2"],
         metrics["standard_ssim_to_at2"], metrics[f"{recon_prefix}_ssim_to_at2"]),
        ("Focus score", metrics["raw_s60_focus"],
         metrics["standard_focus"], metrics[f"{recon_prefix}_focus"]),
    ]
    if recon_prefix == "recon_s60":
        image_rows += [
            ("Recon distance to raw S60 ↓", np.nan, np.nan,
             metrics["recon_s60_distance_to_raw_s60"]),
            ("Recon SSIM to raw S60 ↑", np.nan, np.nan,
             metrics["recon_s60_ssim_to_raw_s60"]),
        ]
    def display(value):
        return "—" if not np.isfinite(value) else f"{value:.5f}"
    sample_paths = sorted((external_dir / "samples").glob("*.png"))[:12]
    standard_uni = metrics.get(
        "uni_cosine_standard_s60_to_standard_at2",
        metrics["uni_cosine_standard_s60_to_at2"],
    )
    return {
        "n_tiles": int(metrics["n_tiles"]),
        "n_slides": int(metrics.get("n_slides", frame.slide_id.nunique())),
        "raw_uni": metrics["uni_cosine_raw_s60_to_at2"],
        "standard_uni": standard_uni,
        "focus_ratio": metrics["standard_focus"] / max(metrics["raw_s60_focus"], 1e-12),
        "uni_table": table([
            ("Matched-condition paired cosine ↑",
             f"{metrics['uni_cosine_raw_s60_to_at2']:.5f}",
             f"{standard_uni:.5f}",
             f"{standard_uni - metrics['uni_cosine_raw_s60_to_at2']:+.5f}"),
            ("S60 output ↔ raw paired AT2 ↑",
             f"{metrics['uni_cosine_raw_s60_to_at2']:.5f}",
             f"{metrics['uni_cosine_standard_s60_to_at2']:.5f}",
             f"{metrics['uni_cosine_standard_s60_to_at2'] - metrics['uni_cosine_raw_s60_to_at2']:+.5f}"),
            ("AT2 canonical identity ↔ raw AT2 ↑", "—",
             f"{metrics.get('uni_cosine_standard_at2_to_at2', float('nan')):.5f}", "—"),
        ], ("OOD UNI metric", "Raw condition", "Canonical condition", "Δ")),
        "image_table": table(
            [(label, display(raw), display(standard), display(recon))
             for label, raw, standard, recon in image_rows],
            ("OOD image metric", "Raw S60", "Canonical S60", recon_label),
        ),
        "joint_umap_image": image_data_uri(eval_dir / "joint_umap.png"),
        "metrics": metrics,
        "recon_prefix": recon_prefix,
        "recon_label": recon_label,
        "style_note": (
            f"{metrics['evaluation_mode']}. Context policy: {metrics.get('context_policy', 'cached per slide and query-disjoint')}. "
            "Paired AT2 is sealed and is used only for evaluation."
            if "M1 metadata-free" in metrics.get("evaluation_mode", "") else
            f"Reconstruction style is pooled by frozen E_set from "
            f"{int(metrics.get('style_context_size', 0))} random non-target patches on the same S60 slide "
            f"(seed {int(metrics.get('style_seed', 0))})."
            if recon_prefix == "recon_s60" else
            "This artifact predates slide-context S60 reconstruction and uses the legacy AT2 render."
        ),
        "samples": [
            {"src": image_data_uri(path),
             "label": f"S60 raw · canonical · {recon_label} · paired AT2"}
            for path in sample_paths
        ],
    }


def combined_image_results(metrics, frame, external):
    """One cohort/representation table; scanner-stratified rows are intentionally absent."""
    def display(value):
        if value is None or not np.isfinite(value):
            return "—"
        return f"{float(value):.5f}"

    rows = [
        ("ID pooled · raw", metrics.get("raw_distance_to_at2"), metrics.get("raw_ssim_to_at2"),
         None, None, frame["raw_focus"].mean()),
        ("ID pooled · canonical", metrics["standard_distance"], metrics["standard_ssim"],
         metrics.get("standard_distance_to_source"), metrics.get("standard_ssim_to_source"),
         frame["canonical_focus"].mean()),
        ("ID pooled · observed-scanner recon", None, None,
         metrics["renderer_distance"], metrics["renderer_ssim"], None),
    ]
    if external is not None:
        ext = external["metrics"]
        recon_prefix = external.get("recon_prefix", "recon_s60")
        recon_label = external.get("recon_label", "S60 context recon")
        rows += [
            ("OOD S60 · raw", ext["raw_s60_distance_to_at2"], ext["raw_s60_ssim_to_at2"],
             None, None, ext["raw_s60_focus"]),
            ("OOD S60 · canonical", ext["standard_distance_to_at2"], ext["standard_ssim_to_at2"],
             ext.get("standard_distance_to_raw_s60"), ext.get("standard_ssim_to_raw_s60"),
             ext["standard_focus"]),
            (f"OOD S60 · {recon_label}", ext.get(f"{recon_prefix}_distance_to_at2"),
             ext.get(f"{recon_prefix}_ssim_to_at2"),
             ext.get("recon_s60_distance_to_raw_s60"), ext.get("recon_s60_ssim_to_raw_s60"),
             ext.get(f"{recon_prefix}_focus")),
        ]
    return table(
        [(label, display(at2_distance), display(at2_ssim), display(source_distance),
          display(source_ssim), display(focus))
         for label, at2_distance, at2_ssim, source_distance, source_ssim, focus in rows],
        ("Cohort · output", "Distance to paired AT2 ↓", "SSIM to paired AT2 ↑",
         "Distance to source ↓", "SSIM to source ↑", "Focus"),
    )


def directional_table(metric_rows):
    rows = []
    for target in ("GT450", "VERSA", "AKOYA", "S60"):
        selected = {row["representation"]: row for row in metric_rows if row["target"] == target}
        if not selected:
            continue
        for representation in ("raw", "canonical"):
            row = selected[representation]
            rows.append((
                target, representation.title(), f"{row['n_queries']:,}",
                f"{row['r1']:.4f}", f"{row['r5']:.4f}", f"{row['paired_cosine']:.4f}",
            ))
    return table(
        rows,
        ("Target gallery", "Representation", "Queries", "Exact R@1 ↑", "Exact R@5 ↑", "Paired cosine ↑"),
    )


def mixed_retrieval_table(metric_rows):
    rows = []
    cohorts = list(dict.fromkeys(row["cohort"] for row in metric_rows))
    for cohort in cohorts:
        for representation in ("raw", "canonical"):
            selected = next((
                row for row in metric_rows
                if row["cohort"] == cohort and row["representation"] == representation
            ), None)
            if selected is None:
                continue
            rows.append((
                cohort, representation.title(), f"{selected['n_queries']:,}",
                f"{selected['same_location_top1']:.4f}",
                f"{selected['same_location_hit_at5']:.4f}",
                f"{selected['counterpart_recall_at5']:.4f}",
                f"{selected['all_counterparts_at5']:.4f}",
                f"{selected['same_scanner_top1']:.4f}",
                f"{selected['same_scanner_purity_at5']:.4f}",
                f"{selected['all5_same_scanner']:.4f}",
            ))
    return table(
        rows,
        ("Mixed gallery", "Representation", "Queries", "Same loc @1 ↑",
         "Same loc hit @5 ↑", "Counterpart recall @5 ↑", "All counterparts @5 ↑",
         "Same scanner @1 ↓", "Scanner purity @5 ↓", "All-5 same scanner ↓"),
    )


def critical_readout(metrics, external, retrieval_rows):
    """Produce a short, deliberately conservative interpretation of the snapshot."""
    points = [
        (
            "ID alignment is substantially stronger after canonicalization: same-location UNI "
            f"R@5 {metrics['raw_same_location_at5']:.3f} → {metrics['canonical_same_location_at5']:.3f}. "
            f"However, the grouped scanner probe remains {metrics['canonical_scanner_probe']:.3f} "
            f"(raw {metrics['raw_scanner_probe']:.3f}), so scanner information is reduced but far from removed."
        )
    ]
    if external is None or not retrieval_rows:
        return points
    ext = external["metrics"]
    s60 = {
        row["representation"]: row for row in retrieval_rows
        if row["cohort"] == "OOD AT2–S60"
    }
    focus_retained = ext["standard_focus"] / max(ext["raw_s60_focus"], 1e-12)
    matched_uni = ext.get(
        "uni_cosine_standard_s60_to_standard_at2",
        ext["uni_cosine_standard_s60_to_at2"],
    )
    points += [
        (
            "S60 zero-shot correspondence is supported in the self-excluded AT2/S60 mixed gallery: "
            f"same-location @1 {s60['raw']['same_location_top1']:.3f} → "
            f"{s60['canonical']['same_location_top1']:.3f}, while same-scanner @1 "
            f"{s60['raw']['same_scanner_top1']:.3f} → {s60['canonical']['same_scanner_top1']:.3f}. "
            "This directly contrasts semantic pairing with scanner-style retrieval."
        ),
        (
            "The standard image is closer to the raw paired AT2 in image space "
            f"(distance {ext['raw_s60_distance_to_at2']:.3f} → {ext['standard_distance_to_at2']:.3f}; "
            f"SSIM {ext['raw_s60_ssim_to_at2']:.3f} → {ext['standard_ssim_to_at2']:.3f}), but its UNI cosine "
            f"to raw AT2 changes {ext['uni_cosine_raw_s60_to_at2']:.3f} → "
            f"{ext['uni_cosine_standard_s60_to_at2']:.3f}. The matched canonical↔canonical cosine "
            f"({matched_uni:.3f}) must therefore not be read as absolute raw-AT2 identity."
        ),
        (
            f"Fine-detail preservation remains the limiting failure: S60 focus retains only "
            f"{focus_retained:.1%} ({ext['raw_s60_focus']:.4f} → {ext['standard_focus']:.4f}). "
            "This checkpoint supports alignment, but not yet a claim of morphology-neutral normalization."
        ),
    ]
    return points


def render_report(config_path, eval_dir, checkpoint="best.ckpt", split="test", workers=8,
                  external_dir=None, external_store=None, validation_samples_dir=None):
    cfg = load_config(str(config_path)); eval_dir = Path(eval_dir)
    metrics = recover_metrics(eval_dir)
    metrics = refresh_grouped_scanner_probe(eval_dir, metrics, workers)
    frame = pd.read_csv(eval_dir / "per_image.csv")
    history = pd.read_csv(eval_dir / "training_history.csv")
    uni_specs = [
        ("Scanner probe balanced accuracy ↓ · location-grouped 5-fold", "raw_scanner_probe", "canonical_scanner_probe"),
        ("Same-location cosine ↑", "raw_same_location_cosine", "canonical_same_location_cosine"),
        ("Same-location cosine distance ↓", "raw_same_location_cosine", "canonical_same_location_cosine"),
        ("Same-location retrieval @5 ↑", "raw_same_location_at5", "canonical_same_location_at5"),
        ("Scanner purity @5 ↓", "raw_scanner_purity_at5", "canonical_scanner_purity_at5"),
    ]
    uni_rows = []
    for label, raw_key, canonical_key in uni_specs:
        raw, canonical = metrics[raw_key], metrics[canonical_key]
        if "distance" in label: raw, canonical = 1-raw, 1-canonical
        uni_rows.append((label, f"{raw:.4f}", f"{canonical:.4f}", f"{canonical-raw:+.4f}"))
    protocol = [("Run", cfg.run.name), ("Report schema", f"shared-v{REPORT_SCHEMA_VERSION}"),
                ("Design version", cfg.design_version), ("Reference scanner", cfg.reference_scanner),
                ("Images", f"{int(metrics['n_images']):,}"), ("Registered locations", f"{int(metrics['n_locations']):,}"),
                ("Patch", f"{cfg.patch.size} px @ {cfg.patch.target_mag}×"), ("UNI embedding", "MahmoodLab/UNI; frozen"),
                ("Scanner probe", metrics["scanner_probe_method"])]
    if "checkpoint_epoch" in metrics:
        protocol.insert(3, ("Sealed checkpoint epoch", int(metrics["checkpoint_epoch"])))
    split_rows = [(name, ", ".join(getattr(cfg.split.slides, name))) for name in ("train", "val", "test")]
    training_rows = [("max epochs", cfg.trainer.max_epochs), ("batches / epoch", cfg.loader.batches_per_epoch),
                     ("base learning rate", cfg.optim.lr), ("precision", cfg.optim.precision),
                     ("gradient clip", cfg.optim.grad_clip), ("accumulation", cfg.trainer.accumulate_grad_batches),
                     ("standard weight", cfg.loss.standard), ("translation weight", cfg.loss.translate),
                     ("pair weight", f"{cfg.loss.pair_start:g} → {cfg.loss.pair_end:g}"),
                     ("nuclei RGB detail weight", getattr(cfg.loss, "nuclei_rgb_detail", 0.0)),
                     ("neighborhood weight", getattr(cfg.loss, "neighborhood_consistency", 0.0)),
                     ("variance weight", cfg.loss.variance), ("style weight", cfg.loss.style)]
    samples_root = eval_dir.parent / "samples"
    epochs = sorted((int(path.name[5:]), path) for path in samples_root.glob("epoch*") if path.name[5:].isdigit())
    if "checkpoint_epoch" in metrics:
        epochs = [entry for entry in epochs if entry[0] <= int(metrics["checkpoint_epoch"])]
    def samples(path, label):
        return [{"src": image_data_uri(image), "label": f"{label} · {image.stem.replace('_',' ')}"}
                for image in sorted(path.glob("grid_loc*.png"))]
    environment = Environment(loader=FileSystemLoader(TEMPLATE.parent), autoescape=select_autoescape(["html"]))
    template = environment.get_template(TEMPLATE.name)
    external = None
    if external_dir is not None:
        if external_store is None:
            raise ValueError("external_store is required when external_dir is provided")
        external = external_report_context(eval_dir, external_dir, external_store, workers)
    input_panels = input_channel_panels(
        eval_dir, cfg, list(cfg.scanners), external_dir, external_store
    )
    nearest_neighbor_panels, retrieval_rows = mixed_scanner_retrieval(
        eval_dir, cfg, checkpoint, frame, list(cfg.scanners),
        external_dir, external_store,
    )
    retrieval_results = mixed_retrieval_table(retrieval_rows)
    kpis = [
        {"label":"ID · Test images", "value":f"{int(metrics['n_images']):,}", "note":f"{int(metrics['n_locations']):,} registered locations"},
        {"label":"ID · UNI same-location cosine", "value":f"{metrics['canonical_same_location_cosine']:.3f}", "note":f"raw {metrics['raw_same_location_cosine']:.3f}"},
        {"label":"ID · UNI scanner probe", "value":f"{metrics['canonical_scanner_probe']:.3f}", "note":f"raw {metrics['raw_scanner_probe']:.3f}; lower is better"},
        {"label":"ID · Retrieval @5", "value":f"{metrics['canonical_same_location_at5']:.3f}", "note":f"raw {metrics['raw_same_location_at5']:.3f}"},
    ]
    if external:
        kpis.append({
            "label": "OOD · S60 UNI cosine",
            "value": f"{external['standard_uni']:.3f}",
            "note": f"raw {external['raw_uni']:.3f}; paired AT2",
        })
    if validation_samples_dir is not None:
        validation_samples_dir = Path(validation_samples_dir)
        final_samples = samples(validation_samples_dir, "best checkpoint validation")
        final_samples_title = "Best checkpoint validation panels"
    else:
        final_samples = samples(epochs[-1][1], f"epoch {epochs[-1][0]}") if epochs else []
        final_samples_title = "Final epoch"
    html_text = template.render(
        run_name=cfg.run.name, split=split, checkpoint=Path(checkpoint).name,
        generated_at=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        kpis=kpis, external=external,
        critical_points=critical_readout(metrics, external, retrieval_rows),
        protocol_table=table(protocol), split_table=table(split_rows, ("Split", "Slides")),
        input_panels=input_panels,
        uni_table=table(uni_rows, ("UNI metric", "Raw", "Canonical", "Δ canonical−raw")),
        umap_image=(external["joint_umap_image"] if external else
                    umap_plot(eval_dir, list(cfg.scanners), workers)),
        nearest_neighbor_panels=nearest_neighbor_panels,
        retrieval_table=retrieval_results,
        image_aggregate_table=combined_image_results(metrics, frame, external),
        loss_svg=loss_curve_svg(history), schedule_table=table(schedule_rows(cfg), ("Epoch interval", "Phase", "Dynamic weights")),
        objective_formula=objective_formula(cfg),
        objective_roles_table=table(objective_roles(cfg), ("Term", "Constraint", "Role")),
        schedule_diagram=schedule_diagram_svg(cfg),
        training_table=table(training_rows),
        final_samples=final_samples,
        final_samples_title=final_samples_title,
        initial_samples=samples(epochs[0][1], f"epoch {epochs[0][0]}") if epochs else [],
        template_path=str(TEMPLATE.relative_to(Path(cfg.paths.repo))),
    )
    (eval_dir / "report.html").write_text(html_text)


def main():
    args = parse_args()
    render_report(
        args.config, args.eval_dir, args.checkpoint, args.split, args.workers,
        args.external_dir, args.external_store, args.validation_samples_dir,
    )
    print(f"[report] rendered {args.eval_dir / 'report.html'}")


if __name__ == "__main__":
    main()
